import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import data.sharding as shard
import os

k=3
nc=9

def create_dataloaders(batch_size):
    dataset_dir = os.path.dirname(os.path.abspath(__file__))+"/../dataset"

    transform_train = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet mean and std
    ])

    transform_val = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet mean and std
    ])

    dataset = torchvision.datasets.CIFAR100(dataset_dir, train=True, download=True, transform=transform_train)
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.1, random_state=1234, stratify=dataset.targets)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=2)

    # clients_non_iid = shard.non_iid_sharding(subset, k, nc)
    # clients_iid = shard.iid_sharding(subset, k)
    # clients_advanced_non_iid = shard.advanced_non_iid_sharding(subset, k, nc)

    dataset = torchvision.datasets.CIFAR100(dataset_dir, train=True, transform=transform_val)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=True)

    return train_loader, val_loader