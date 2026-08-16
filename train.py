import argparse
import sys
from datetime import datetime
import torch
from torch import nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, MultiplicativeLR, SequentialLR
from tqdm.auto import tqdm

from checkpoint_manager import CheckpointManager
from data.dataloader import create_dataloaders, DEVICE
from logger import Logger
from plot import plot_training
from models.model import Dino_vits16_100

DEBUG = False
USE_AMP = DEVICE == "cuda"
NUM_CLASSES = 100

def set_backbone(model, frozen):
    for param in model.backbone.parameters():
        param.requires_grad = not frozen

    state = 'frozen_backbone' if frozen else 'unfrozen'
    print(f'Backbone {state}')

def train_one_epoch(epoch, model, train_loader, criterion, optimizer, scaler, frozen_backbone):
    model.train()

    # Keep frozen_backbone backbone in eval mode
    if frozen_backbone:
        model.backbone.eval()

    running_loss = 0.0

    progress_bar = tqdm(train_loader, f'Train Epoch {epoch}', leave=False)

    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        if DEBUG and batch_idx > 1:
            break

        inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)

        with torch.amp.autocast(device_type=DEVICE, enabled=USE_AMP):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

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


def train(num_epochs, model, train_loader, val_loader, criterion, optimizer, scaler, scheduler, logger, manager, freeze_backbone_epochs, start_epoch=1):
    freeze_backbone = start_epoch <= freeze_backbone_epochs
    set_backbone(model, freeze_backbone)
    for epoch in range(start_epoch, num_epochs + 1):

        if epoch == freeze_backbone_epochs + 1:
            freeze_backbone = False
            set_backbone(model, freeze_backbone)

        train_loss, lr = train_one_epoch(epoch, model, train_loader, criterion, optimizer, scaler, freeze_backbone)

        # At the end of each training iteration, perform a validation step
        val_loss, val_acc = validate(model, val_loader, criterion)

        # Update learning rate
        scheduler.step()

        # Log
        logger.log(epoch, train_loss, val_loss, val_acc, lr, freeze_backbone)

        # Checkpoint
        manager.save(epoch, model, optimizer, scaler, scheduler, logger, val_acc)


    print(f'Best validation accuracy: {logger.get_best_accuracy():.2f}%')


def build_training_objects(run):
    # Import data
    train_loader, val_loader = create_dataloaders(run['batch_size'])

    # Define model
    model = Dino_vits16_100().to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer
    lr = run['max_learning_rate']
    parameters = [
        {"params": model.classifier.parameters(), "lr": lr},
        {"params": model.backbone.parameters(), "lr": lr/10}
    ]
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
    train(total_epochs, model, train_loader, val_loader, criterion, optimizer, scaler, scheduler, logger, manager, run['freeze_backbone_epochs'], epoch + 1)
    plot_training(run_name, logs_dir, plots_dir)
    return logger.get_run()

def start(num_epochs, batch_size, max_lr, weight_decay):
    warmup_epochs = 2
    cosine_epochs = 18
    freeze_backbone_epochs = 4

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
        'weight_decay': weight_decay,
        'optimizer': 'SGD(momentum=0.9)',
        'scheduler': 'CosineAnnealingLR with warm-up',
        'warmup_epochs' : warmup_epochs,
        'cosine_epochs': cosine_epochs,
        'freeze_backbone_epochs': freeze_backbone_epochs,
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
    train(num_epochs, model, train_loader, val_loader, criterion, optimizer, scaler, scheduler, logger, manager, freeze_backbone_epochs)
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
        default=64,
        help="Batch size"
    )

    start_parser.add_argument(
        "--max-lr",
        type=float,
        default=0.01,
        help="Maximum learning rate"
    )

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
        start(args.epochs, args.batch_size, args.max_lr, args.weight_decay)

    elif args.action == "resume":
        resume(args.run_name, args.total_epochs, args.separate)
