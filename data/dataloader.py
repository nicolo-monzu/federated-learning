import torchvision
import torchvision.transforms as T
from torch import Generator
from torch.utils.data import DataLoader, random_split


def create_dataloaders(batch_size):
    transform_train = T.Compose([
        #T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
        T.ToTensor(),
        #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_val = T.Compose([
        #T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
        T.ToTensor(),
        #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    imagenet_data = torchvision.datasets.CIFAR100('dataset/', train=True, download=True, transform=transform_train)
    data, _ = random_split(imagenet_data, [0.8, 0.2], generator=Generator().manual_seed(1234))
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)

    imagenet_data = torchvision.datasets.CIFAR100('dataset/', train=True, transform=transform_val)
    _, data = random_split(imagenet_data, [0.8, 0.2], generator=Generator().manual_seed(1234))
    val_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader