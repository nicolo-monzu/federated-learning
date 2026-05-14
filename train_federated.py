import os
from datetime import datetime

import torch
from torch import nn
from logger import Logger

DEBUG = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_client(model, train_dataloader, epochs):
    return

"""
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

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        masks = [torch.ones_like(param) for param in model.parameters()]
        optimizer.step(masks)

        running_loss += loss.item()
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
        scheduler.step()

        # At the end of each training iteration, perform a validation step
        val_loss, val_acc = validate(model, val_loader, criterion)

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
        # removed save
        # If it is the best model
        if val_acc > best_acc or epoch == 1:
            # Update best accuracy
            best_acc = val_acc
            logger.update_best_acc(best_acc)
            # Save best checkpoint
            # removed save

        if hasattr(os, 'sync'):
            os.sync()

    print(f'Best validation accuracy: {best_acc:.2f}%')

def start(num_epochs, optimizer):
    batch_size = 32
    learning_rate = 0.001
    weight_decay = 1e-4
    scheduler_period = 10

    logs_dir='federated_model/logs'
    checkpoints_dir = 'federated_model/checkpoints/'
    os.makedirs(checkpoints_dir, exist_ok=True)
    plots_dir = 'federated_model/plots/'

    # Init logger
    run = {
        'name': ('debug_' if DEBUG else '') + datetime.now().strftime('%Y%m%d_%H%M%S'),
        'model': 'dino_vits16_federated',
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'optimizer': optimizer.__name__,
        'scheduler': 'CosineAnnealingLR',
        'scheduler_period': scheduler_period,
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
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = SparseSGDM(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_period)

    # Run the training process for {num_epochs} epochs
    print(f'Run name: {run['name']}')
    print('Start training')
    train(num_epochs, run['name'], model, train_loader, val_loader, criterion, optimizer, scheduler, logger, checkpoints_dir, 0)
    plot_training(run['name'], logs_dir, plots_dir)

if __name__ == '__main__':
    start(10 if not DEBUG else 1)
"""
