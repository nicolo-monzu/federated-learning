import os
from datetime import datetime

import torch
from torch import nn
from data.dataloader import create_dataloaders
from logger import Logger

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def save(checkpoint, filename):
    torch.save(checkpoint, filename+'.tmp')
    os.replace(filename+'.tmp', filename)

def train_one_epoch(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Debugging
        # if batch_idx > 1:
        #     break
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')
    return train_loss, train_accuracy


# Validation loop
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0

    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            # Debugging
            # if batch_idx > 1:
            #     break
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


def train(num_epochs, run_name, model, train_loader, val_loader, criterion, optimizer, logger, checkpoints_dir, best_acc):
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(epoch, model, train_loader, criterion, optimizer)

        # At the end of each training iteration, perform a validation step
        val_loss, val_acc = validate(model, val_loader, criterion)

        # Log
        logger.log(epoch, train_loss, train_acc, val_loss, val_acc)

        # Save last checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc
        }
        save(checkpoint, f'{checkpoints_dir}/{run_name}_last.pth')

        # If it is the best model
        if val_acc > best_acc:
            # Update best accuracy
            best_acc = val_acc
            logger.update_best_acc(best_acc)
            # Save best checkpoint
            save(checkpoint, f'{checkpoints_dir}/{run_name}_best.pth')

        if hasattr(os, 'sync'):
            os.sync()

    print(f'Best validation accuracy: {best_acc:.2f}%')

def resume():
    #todo
    pass

def start():
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10

    print('Using device:', DEVICE)

    # Import data
    train_loader, val_loader = create_dataloaders(batch_size)

    # Define model
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Init logger
    run = {
        'name': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'model': 'dino_vits16_centralized',
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'optimizer': 'SGD(momentum=0.9)',
        'best_accuracy': 0
    }
    logger = Logger(log_dir='centralized_model/logs', run_name=run['name'])
    logger.start(run)

    checkpoints_dir = 'centralized_model/checkpoints/'
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Run the training process for {num_epochs} epochs
    print('Start training')
    train(num_epochs, run['name'], model, train_loader, val_loader, criterion, optimizer, logger, checkpoints_dir, 0)

if __name__ == '__main__':
    start()