import numpy as np
import torch
import matplotlib.pyplot as plt
from data.dataloader import create_dataloaders
from train import apply_mixup_cutmix

def denormalize(image):
    image = image.to('cpu').numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image

def visualize():
    train_loader, val_loader = create_dataloaders(16)

    # Determine the number of classes and samples
    num_classes = len(train_loader.dataset.dataset.classes)
    num_samples = len(train_loader.dataset.dataset)
    num_train_samples = len(train_loader.dataset)
    num_val_samples = len(val_loader.dataset)

    print(f'Number of classes: {num_classes}')
    print(f'Number of training samples: {num_train_samples}')
    print(f'Number of validation samples: {num_val_samples}')
    print(f'Number of total samples: {num_samples}')

    _, axes = plt.subplots(3, 3, figsize=(6, 6))

    # Get class names
    class_names = train_loader.dataset.dataset.classes

    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = apply_mixup_cutmix(inputs, targets)
        # Select first image from the batch
        img, scores = inputs[0], targets[0]
        # Extract labels
        labels = torch.nonzero(scores).squeeze(dim=-1)
        # Sort labels
        labels = labels[torch.argsort(scores[labels], descending=True)]

        axes.flat[i].imshow(denormalize(img))
        # Title
        title_parts = [f'{class_names[idx]} {(scores[idx] * 100).round()}%' for idx in labels]
        axes.flat[i].set_title('\n'.join(title_parts))
        # axes.flat[i].axis('off')
        if i >= 8:
            break
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    visualize()
