import matplotlib.pyplot as plt
from data.dataloader import create_dataloaders

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

    # Visualize one example for each class for 9 classes (only from the training set)
    _, axes = plt.subplots(3, 3, figsize=(6, 6))
    classes_sampled = []
    found_classes = 0

    # Get class names
    class_names = train_loader.dataset.dataset.classes

    for i, data in enumerate(train_loader):
        for img, label in zip(*data):
            if label.item() not in classes_sampled:
                classes_sampled.append(label.item())
                axes.flat[found_classes].imshow(img.numpy().transpose((1, 2, 0)))
                axes.flat[found_classes].set_title(class_names[label.item()])
                # axes.flat[found_classes].axis('off')
                found_classes += 1
            if found_classes == 9:
                break
        if found_classes == 9:
            break
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    visualize()
