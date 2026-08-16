import argparse
import sys
from datetime import datetime
import torch
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

def train_one_epoch(epoch, model, train_loader, criterion, optimizer, scaler):
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
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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


def train(num_epochs, model, train_loader, val_loader, criterion, optimizer, scaler, scheduler, logger, manager, start_epoch=1):
    for epoch in range(start_epoch, num_epochs + 1):
        train_loss, lr = train_one_epoch(epoch, model, train_loader, criterion, optimizer, scaler)

        # At the end of each training iteration, perform a validation step
        val_loss, val_acc = validate(model, val_loader, criterion)

        # Update learning rate
        scheduler.step()

        # Log
        logger.log(epoch, train_loss, val_loss, val_acc, lr)

        # Checkpoint
        manager.save(epoch, model, optimizer, scaler, scheduler, logger, val_acc)


    print(f'Best validation accuracy: {logger.get_best_accuracy():.2f}%')

def apply_llrd(model, learning_rate, decay_rate):
    # Assign a lr to each layer

    # classifier
    param_groups = [{"params": model.classifier.parameters(), "lr": learning_rate}]

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


def build_training_objects(run):
    # Import data
    train_loader, val_loader = create_dataloaders(run['batch_size'])

    # Define model
    model = Dino_vits16_100().to(DEVICE)

    # Learning rate decaying
    parameters = apply_llrd(model, run['max_learning_rate'], run['decay_rate'])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(parameters, momentum=0.9, weight_decay=run['weight_decay'])
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


def resume(run_name, total_epochs, separate=False):
    checkpoints_dir = 'centralized_model/checkpoints/'
    logs_dir = 'centralized_model/logs/'
    plots_dir = 'centralized_model/plots/'

    # Cleanup, restoring files and checkpoint loading
    manager = CheckpointManager(checkpoints_dir, run_name)
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
            sys.exit('Error: Attempted to resume in debug mode a non debug training')
        print('Debug mode')

    print('Using device:', DEVICE)

    if USE_AMP:
        print('Using automatic mixed precision')

    train_loader, val_loader, model, criterion, optimizer, scaler, scheduler = build_training_objects(run)

    # Restore state
    manager.restore_state(model, optimizer, scaler, scheduler)

    # Run the training process for {num_epochs} epochs
    print(f'Run name: {run['name']}')
    print('Resume training')
    train(total_epochs, model, train_loader, val_loader, criterion, optimizer, scaler, scheduler, logger, manager, epoch + 1)
    plot_training(run_name, logs_dir, plots_dir)
    return logger.get_run()

def start(num_epochs, batch_size, max_lr, decay_rate, weight_decay):
    warmup_epochs = 3
    cosine_epochs = 17

    if DEBUG:
        batch_size = 1

    logs_dir = 'centralized_model/logs/'
    checkpoints_dir = 'centralized_model/checkpoints/'
    plots_dir = 'centralized_model/plots/'

    # Init checkpoint manager and logger
    run = {
        'name': ('debug_' if DEBUG else '') + datetime.now().strftime('%Y%m%d_%H%M%S'),
        'model': 'dino_vits16_100_centralized',
        'batch_size': batch_size,
        'max_learning_rate': max_lr,
        'decay_rate': decay_rate,
        'weight_decay': weight_decay,
        'optimizer': 'SGD(momentum=0.9)',
        'scheduler': 'CosineAnnealingLR with warm-up',
        'warmup_epochs' : warmup_epochs,
        'cosine_epochs': cosine_epochs,
        'total_epochs': 0,
        'best_epoch': -1,
        'best_accuracy': -1.0,
        'debug': DEBUG
    }
    manager = CheckpointManager(checkpoints_dir, run['name'])
    logger = Logger(logs_dir, run['name'])
    logger.start(run)

    if DEBUG:
        print('Debug mode')

    print('Using device:', DEVICE)

    if USE_AMP:
        print('Using automatic mixed precision')

    train_loader, val_loader, model, criterion, optimizer, scaler, scheduler = build_training_objects(run)

    # Run the training process for {num_epochs} epochs
    print(f'Run name: {run['name']}')
    print('Start training')
    train(num_epochs, model, train_loader, val_loader, criterion, optimizer, scaler, scheduler, logger, manager)
    plot_training(run['name'], logs_dir, plots_dir)
    return logger.get_run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Start or resume a centralized model training.")
    subparsers = parser.add_subparsers(dest="action", required=True)

    # Start
    start_parser = subparsers.add_parser(
        "start",
        help="Start a new training run"
    )

    start_parser.add_argument(
        "-e", "--epochs",
        type=int,
        required=True,
        help="Total number of training epochs"
    )

    start_parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size"
    )

    start_parser.add_argument(
        "--max-lr",
        type=float,
        default=0.01,
        help="Maximum learning rate"
    )

    start_parser.add_argument(
        "--decay-rate",
        type=float,
        default=0.75,
        help="LLRD decay rate")

    start_parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay"
    )

    # Resume
    resume_parser = subparsers.add_parser(
        "resume",
        help="Resume an existing training run"
    )

    resume_parser.add_argument(
        "-r", "--run-name",
        type=str,
        required=True,
        help="Name of the run to resume"
    )

    resume_parser.add_argument(
        "-e", "--total-epochs",
        type=int,
        required=True,
        help="Total number of training epochs"
    )

    resume_parser.add_argument(
        "--separate",
        action="store_true",
        help="Resume training as a new run, preserving the original run's checkpoints and logs"
    )


    args = parser.parse_args()

    if args.action == "start":
        start(args.epochs, args.batch_size, args.max_lr, args.decay_rate, args.weight_decay)

    elif args.action == "resume":
        resume(args.run_name, args.total_epochs, args.separate)
