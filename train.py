import argparse
import os
from datetime import datetime
import torch
import yaml
from torch import nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, MultiplicativeLR, SequentialLR
from torchvision.transforms.v2 import MixUp, CutMix, RandomChoice
from tqdm.auto import tqdm

from checkpoint_manager import CheckpointManager
from data.dataloader import create_dataloaders, DEVICE
from logger import Logger
from plot import plot_training
from models.model import Dino_vits16_100

DEBUG = False
USE_AMP = DEVICE == "cuda"
NUM_CLASSES = 100

mixup = MixUp(num_classes=NUM_CLASSES, alpha=0.8)
cutmix = CutMix(num_classes=NUM_CLASSES, alpha=1.0)
apply_mixup_cutmix = RandomChoice([cutmix, mixup])

def train_one_epoch(epoch, model, train_loader, criterion, optimizer, scaler, classifier_only=False):
    if classifier_only:
        model.backbone.eval()
        model.classifier.train()
    else:
        model.train()
    running_loss = 0.0

    progress_bar = tqdm(train_loader, f'Train Epoch {epoch}', leave=False)

    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        if DEBUG and batch_idx > 1:
            break

        inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
        inputs, targets_mix = apply_mixup_cutmix(inputs, targets)

        with torch.amp.autocast(device_type=DEVICE, enabled=USE_AMP):
            outputs = model(inputs)
            loss = criterion(outputs, targets_mix)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    lr = optimizer.param_groups[0]["lr"]
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Lr: {lr:e}')
    return train_loss, lr


# Validation loop
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0

    correct, total = 0, 0

    progress_bar = tqdm(val_loader, 'Validation', leave=False)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            if DEBUG and batch_idx > 1:
                break

            inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)

            with torch.amp.autocast(device_type=DEVICE, enabled=USE_AMP):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            val_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            total += batch_size
            correct += predicted.eq(targets).sum().item()

    val_loss /= total
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_loss, val_accuracy


def train(num_epochs, run, model, train_loader, val_loader, criterion, optimizer,
          scaler, scheduler, logger, manager, patience, start_epoch=1, classifier_only=False):

    for epoch in range(start_epoch, num_epochs + 1):
        train_loss, lr = train_one_epoch(epoch, model, train_loader, criterion, optimizer, scaler, classifier_only)

        # At the end of each training iteration, perform a validation step
        val_loss, val_acc = validate(model, val_loader, criterion)

        # Update learning rate
        scheduler.step()

        # Log
        logger.log(epoch, train_loss, val_loss, val_acc, lr)

        # Checkpoint
        manager.save(epoch, model, optimizer, scaler, scheduler, logger, val_acc)

        # Early stopping
        if patience > 0 and run['total_epochs'] - run['best_epoch'] >= patience:
            break


    print(f'Best validation accuracy: {logger.get_best_accuracy():.2f}%')

def apply_llrd(model, learning_rate, decay_rate, backbone_only=False):
    # Assign a lr to each layer

    param_groups = []

    if not backbone_only:
        # classifier
        param_groups.append({"params": model.classifier.parameters(), "lr": learning_rate})

    # norm
    learning_rate *= decay_rate
    param_groups.append({"params": model.backbone.norm.parameters(), "lr": learning_rate})

    # blocks
    for block in reversed(model.backbone.blocks):
        learning_rate *= decay_rate
        param_groups.append({"params": block.parameters(), "lr": learning_rate})

    # pos_drop

    # patch_embed
    learning_rate *= decay_rate
    param_groups.append({"params": model.backbone.patch_embed.parameters(), "lr": learning_rate})

    param_groups.append({"params": model.backbone.pos_embed, "lr": learning_rate})
    param_groups.append({"params": model.backbone.cls_token, "lr": learning_rate})

    #print(list(a for (a,_) in model.backbone.named_parameters()))
    return param_groups


def freeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = False


def build_training_objects(run, classifier_only=False):
    # Import data
    train_loader, val_loader = create_dataloaders(run['batch_size'])

    # Define model
    model = Dino_vits16_100().to(DEVICE)

    if classifier_only:
        # Freeze backbone
        freeze_backbone(model)

        optimizer = torch.optim.SGD(model.classifier.parameters(), lr=run['max_learning_rate'],
                                    momentum=0.9, weight_decay=run['weight_decay'])
    else:
        # Learning rate decaying
        parameters = apply_llrd(model, run['max_learning_rate'], run['decay_rate'])

        optimizer = torch.optim.SGD(parameters, momentum=0.9, weight_decay=run['weight_decay'])

    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    # Scheduler
    warmup_epochs = run['warmup_epochs']
    cosine_epochs = run['cosine_epochs']
    warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs + 1)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=cosine_epochs)
    constant_sched = MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 1.0)

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched, constant_sched],
        milestones=[warmup_epochs, warmup_epochs + cosine_epochs]
    )

    return train_loader, val_loader, model, criterion, optimizer, scaler, scheduler


def resume(run_name, total_epochs, patience, checkpoints_dir, logs_dir, plots_dir, separate=False, classifier_only=False):

    # Cleanup, restoring files and checkpoint loading
    manager = CheckpointManager(checkpoints_dir, run_name, classifier_only)
    logger_state_dict, epoch = manager.resume()

    logger = Logger(logs_dir, run_name)
    logger.resume(logger_state_dict)

    if separate:
        run_name = ('debug_' if DEBUG else '') + datetime.now().strftime('%Y%m%d_%H%M%S')
        manager.set_run_name(run_name)
        logger.new_run_name(run_name)

    run = logger.get_run()

    if DEBUG:
        if not run['debug']:
            raise RuntimeError('Attempted to resume in debug mode a non-debug training run')
        print('Debug mode')

    print('Using device:', DEVICE)

    if USE_AMP:
        print('Using automatic mixed precision')

    train_loader, val_loader, model, criterion, optimizer, scaler, scheduler = build_training_objects(run, classifier_only)

    # Restore state
    manager.restore_state(model, optimizer, scaler, scheduler)

    # Run the training process for {num_epochs} epochs
    print(f'Run name: {run['name']}')
    print('Resume training')
    train(total_epochs, run, model, train_loader, val_loader, criterion, optimizer, scaler, scheduler, logger, manager, patience, epoch + 1, classifier_only)
    plot_training(run_name, logs_dir, plots_dir)

    if classifier_only:
        # Save a file with only the head
        torch.save(model.classifier.state_dict(), f'{checkpoints_dir}/{run_name}.head.pth')

    return logger.get_run()

def start(num_epochs, batch_size, max_lr, decay_rate, weight_decay, warmup_epochs,
          cosine_epochs, patience, checkpoints_dir, logs_dir, plots_dir, run_name=None, classifier_only=False):

    if DEBUG:
        batch_size = 1

    # Generate a run name if one was not provided
    if run_name is None:
        run_name = ('debug_' if DEBUG else '') + datetime.now().strftime('%Y%m%d_%H%M%S')

    if os.path.exists(f'{checkpoints_dir}/{run_name}.pth'):
        print(
            f'Error: checkpoint "{run_name}.pth" already exists. '
            'Training cannot start with this run name. '
            'Use "train.py resume" to continue from the existing checkpoint, '
            'or remove/rename the checkpoint to start a new training run with this name.'
        )
        return {'best_accuracy': -1}

    # Init checkpoint manager and logger
    run = {
        'name': run_name,
        'model': 'dino_vits16_100_centralized',
        'batch_size': batch_size,
        'max_learning_rate': max_lr,
        'decay_rate': decay_rate,
        'weight_decay': weight_decay,
        'optimizer': 'SGD(momentum=0.9)',
        'scheduler': 'CosineAnnealingLR with warm-up',
        'warmup_epochs': warmup_epochs,
        'cosine_epochs': cosine_epochs,
        'augmentation': 'MixUp/CutMix',
        'total_epochs': 0,
        'best_epoch': -1,
        'best_accuracy': -1.0,
        'debug': DEBUG
    }
    manager = CheckpointManager(checkpoints_dir, run['name'], classifier_only)
    logger = Logger(logs_dir, run['name'])
    logger.start(run)

    if DEBUG:
        print('Debug mode')

    print('Using device:', DEVICE)

    if USE_AMP:
        print('Using automatic mixed precision')

    train_loader, val_loader, model, criterion, optimizer, scaler, scheduler = build_training_objects(run, classifier_only)

    # Run the training process for {num_epochs} epochs
    print(f'Run name: {run['name']}')
    print('Start training')
    train(num_epochs, run, model, train_loader, val_loader, criterion, optimizer, scaler, scheduler, logger, manager, patience, classifier_only=classifier_only)
    plot_training(run['name'], logs_dir, plots_dir)

    if classifier_only:
        # Save a file with only the head
        torch.save(model.classifier.state_dict(), f'{checkpoints_dir}/{run_name}.head.pth')

    return logger.get_run()


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config["train"]

if __name__ == '__main__':
    config = load_config()
    parser = argparse.ArgumentParser(description="Start or resume a centralized model training.")
    subparsers = parser.add_subparsers(dest="action", required=True)

    # Start
    start_parser = subparsers.add_parser(
        "start",
        help="Start a new training run"
    )

    start_parser.add_argument(
        "-n", "--run-name",
        type=str,
        default=None,
        help="Name of the new run. If omitted, a timestamp-based name is generated."
    )

    start_parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=config["num_epochs"],
        help="Total number of training epochs. (default: config.yaml)"
    )

    start_parser.add_argument(
        "--batch-size",
        type=int,
        default=config["batch_size"],
        help="Batch size. (default: config.yaml)"
    )

    start_parser.add_argument(
        "--max-lr",
        type=float,
        default=config["max_lr"],
        help="Maximum learning rate. (default: config.yaml)"
    )

    start_parser.add_argument(
        "--decay-rate",
        type=float,
        default=config["decay_rate"],
        help="LLRD decay rate. (default: config.yaml)"
    )

    start_parser.add_argument(
        "--weight-decay",
        type=float,
        default=config["weight_decay"],
        help="Weight decay. (default: config.yaml)"
    )

    start_parser.add_argument(
        "--classifier-only",
        action="store_true",
        help="Train only the classifier. (default: false)"
    )

    # Resume
    resume_parser = subparsers.add_parser(
        "resume",
        help="Resume an existing training run"
    )

    resume_parser.add_argument(
        "-n", "--run-name",
        type=str,
        required=True,
        help="Name of the run to resume"
    )

    resume_parser.add_argument(
        "-e", "--total-epochs",
        type=int,
        default=config["num_epochs"],
        help="Total number of training epochs. (default: config.yaml)"
    )

    resume_parser.add_argument(
        "--separate",
        action="store_true",
        help="Resume training as a new run, preserving the original run's checkpoints and logs. (default: false)"
    )

    resume_parser.add_argument(
        "--classifier-only",
        action="store_true",
        help="Train only the classifier. (default: false)"
    )

    args = parser.parse_args()

    if args.action == "start":
        start(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            max_lr=args.max_lr,
            decay_rate=args.decay_rate,
            weight_decay=args.weight_decay,
            warmup_epochs=config["warmup_epochs"],
            cosine_epochs=config["cosine_epochs"],
            patience=config["patience"],
            checkpoints_dir=config["checkpoints_dir"],
            logs_dir=config["logs_dir"],
            plots_dir=config["plots_dir"],
            run_name=args.run_name,
            classifier_only = args.classifier_only
        )

    elif args.action == "resume":
        resume(
            run_name=args.run_name,
            total_epochs=args.total_epochs,
            patience=config["patience"],
            checkpoints_dir=config["checkpoints_dir"],
            logs_dir=config["logs_dir"],
            plots_dir=config["plots_dir"],
            separate=args.separate,
            classifier_only=args.classifier_only
        )
