import os
from datetime import datetime

import torch
from torch import nn
from torchvision.transforms.v2 import MixUp, CutMix
from data.dataloader import create_dataloaders
from logger import Logger
from plot import plot_training
from models.model import Dino_vits16_100

DEBUG = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 100

mixup = MixUp(num_classes=NUM_CLASSES, alpha=0.8)
cutmix = CutMix(num_classes=NUM_CLASSES, alpha=1.0)

def apply_mixup_cutmix(inputs, targets):
    if torch.rand(1).item() < 0.5:
        return mixup(inputs, targets)
    return cutmix(inputs, targets)

def save(checkpoint, filename):
    torch.save(checkpoint, filename+'.tmp')
    os.replace(filename+'.tmp', filename)

def train_one_epoch(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if DEBUG:
            if batch_idx > 1:
                break
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        inputs, targets_mix = apply_mixup_cutmix(inputs, targets)

        outputs = model(inputs)
        # targets_mix are soft labels
        loss = criterion(outputs, targets_mix)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # fixme: training accuracy uses original hard labels
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}% Lr: {optimizer.param_groups[0]["lr"]:e}')
    return train_loss, train_accuracy


# Validation loop
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0

    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            if DEBUG:
                if batch_idx > 1:
                    break
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_loss, val_accuracy


def train(num_epochs, run_name, model, train_loader, val_loader, criterion, optimizer, scheduler, logger, checkpoints_dir, best_acc, start_epoch=1):
    for epoch in range(start_epoch, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(epoch, model, train_loader, criterion, optimizer)

        # At the end of each training iteration, perform a validation step
        val_loss, val_acc = validate(model, val_loader, criterion)

        #Update learning rate
        if epoch < scheduler.T_max:
            scheduler.step()

        # Log
        logger.log(epoch, train_loss, train_acc, val_loss, val_acc)

        # Save last checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'accuracy': val_acc
        }
        save(checkpoint, f'{checkpoints_dir}/{run_name}_last.pth')

        # If it is the best model
        if val_acc > best_acc or epoch == 1:
            # Update best accuracy
            best_acc = val_acc
            logger.update_best_acc(best_acc)
            # Save best checkpoint
            save(checkpoint, f'{checkpoints_dir}/{run_name}_best.pth')

        if hasattr(os, 'sync'):
            os.sync()

    print(f'Best validation accuracy: {best_acc:.2f}%')

def resume(run_name, num_epochs):
    checkpoints_dir = 'centralized_model/checkpoints/'
    logs_dir = 'centralized_model/logs/'
    plots_dir = 'centralized_model/plots/'

    # delete temp files
    try:
        os.remove(f'{checkpoints_dir}/{run_name}_last.pth.tmp')
    except FileNotFoundError:
        pass

    try:
        os.remove(f'{checkpoints_dir}/{run_name}_best.pth.tmp')
    except FileNotFoundError:
        pass


    last = torch.load(f'{checkpoints_dir}/{run_name}_last.pth')
    best = torch.load(f'{checkpoints_dir}/{run_name}_best.pth')

    # update best checkpoint
    if last['accuracy'] > best['accuracy']:
        save(last, f'{checkpoints_dir}/{run_name}_best.pth')

    best_acc = max(last['accuracy'], best['accuracy'])
    logger = Logger(log_dir=logs_dir, run_name=run_name)
    logger.resume(last['epoch'], best_acc)
    run = logger.get_run()

    if hasattr(os, 'sync'):
        os.sync()

    if DEBUG:
        if not run['debug']:
            exit('Error: Attempted to resume in debug mode a non debug training')
        print('Debug mode')
    print('Using device:', DEVICE)

    # Import data
    train_loader, val_loader = create_dataloaders(run['batch_size'])

    # Define model
    model = Dino_vits16_100().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=run['learning_rate'], momentum=0.9, weight_decay=run['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=run['scheduler_T_max'])

    # Restore state
    model.load_state_dict(last['model_state_dict'])
    optimizer.load_state_dict(last['optimizer_state_dict'])
    scheduler.load_state_dict(last['scheduler_state_dict'])

    # Run the training process for {num_epochs} epochs
    print(f'Run name: {run['name']}')
    print('Resume training')
    train(num_epochs, run['name'], model, train_loader, val_loader, criterion, optimizer, scheduler, logger, checkpoints_dir, best_acc, last['epoch']+1)
    plot_training(run['name'], logs_dir, plots_dir)

def start(num_epochs):
    batch_size = 32
    learning_rate = 0.001
    weight_decay = 1e-4
    scheduler_T_max = 10

    logs_dir = 'centralized_model/logs'
    checkpoints_dir = 'centralized_model/checkpoints/'
    os.makedirs(checkpoints_dir, exist_ok=True)
    plots_dir = 'centralized_model/plots/'

    # Init logger
    run = {
        'name': ('debug_' if DEBUG else '') + datetime.now().strftime('%Y%m%d_%H%M%S'),
        'model': 'dino_vits16_100_centralized',
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'optimizer': 'SGD(momentum=0.9)',
        'scheduler': 'CosineAnnealingLR',
        'scheduler_T_max': scheduler_T_max,
        'best_accuracy': 0,
        'debug': DEBUG
    }
    logger = Logger(log_dir=logs_dir, run_name=run['name'])
    logger.start(run)

    if DEBUG:
        print('Debug mode')
    print('Using device:', DEVICE)

    # Import data
    train_loader, val_loader = create_dataloaders(batch_size)

    # Define model
    model = Dino_vits16_100().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_T_max)

    # Run the training process for {num_epochs} epochs
    print(f'Run name: {run['name']}')
    print('Start training')
    train(num_epochs, run['name'], model, train_loader, val_loader, criterion, optimizer, scheduler, logger, checkpoints_dir, 0)
    plot_training(run['name'], logs_dir, plots_dir)

if __name__ == '__main__':
    start(10 if not DEBUG else 1)
