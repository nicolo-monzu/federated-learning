from datetime import datetime

import torch
from torch import nn
from data.dataloader import create_dataloaders
from logger import Logger

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_one_epoch(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
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

def main():
    batch_size = 32
    learning_rate = 0.001

    print('Using device:', DEVICE)

    # Import data
    train_loader, val_loader = create_dataloaders(batch_size)

    # Define model
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    best_acc = 0

    # Init logger
    exp = {
        'name': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'model': 'dino_vits16_centralized',
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'optimizer': 'SGD(momentum=0.9)',
        'best_accuracy': 0
    }
    exp_dir = f'checkpoints/{exp['name']}'
    logger = Logger(log_dir=exp_dir)

    logger.start(exp)


    # Run the training process for {num_epochs} epochs
    num_epochs = 10
    print('Start training')
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(epoch, model, train_loader, criterion, optimizer)

        # At the end of each training iteration, perform a validation step
        val_loss, val_acc = validate(model, val_loader, criterion)

        # Best validation accuracy
        best_acc = max(best_acc, val_acc)
        logger.update_best_acc(best_acc)

        # Log
        logger.log(epoch, train_loss, train_acc, val_loss, val_acc)

    print(f'Best validation accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    main()