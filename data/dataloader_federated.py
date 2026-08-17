import random
import sys
import warnings
import torchvision
from numpy.exceptions import VisibleDeprecationWarning
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import os

from .dataloader import DEVICE, transform_train, transform_val
from .iid_sharding import split_subset_iid
from .non_iid_sharding import split_subset_non_iid

def create_dataloader_federated(batch_size, num_clients, num_classes_per_client):
    rng = random.Random(1234)

    K = num_clients
    Nc = num_classes_per_client

    dataset_dir = os.path.dirname(os.path.abspath(__file__)) + "/../dataset"

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', VisibleDeprecationWarning)
        dataset = torchvision.datasets.CIFAR100(dataset_dir, train=True, download=True, transform=transform_train)

    C = len(dataset.classes)
    if (K * Nc) % C != 0:
        sys.exit(f"num_clients * num_classes_per_client must be divisible by num_classes ({C})")

    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.1, random_state=1234, stratify=dataset.targets)


    # Note: The iid algorithm implemented need K = C = Nc. If an iid run has K != C, the algorith executed is non-iid
    # with Nc = C, that may produce slightly differences in the number of samples of each dataloader
    if K == Nc and Nc == C:
        print("iid sharding")
        subsets = split_subset_iid(Subset(dataset, train_idx), C, rng)
    else:
        print(f"non iid sharding (Nc = {Nc})")
        subsets = split_subset_non_iid(Subset(dataset, train_idx), K, Nc, C, rng)


    train_loaders = [
        DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=DEVICE == "cuda")
        for subset in subsets
    ]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', VisibleDeprecationWarning)
        dataset = torchvision.datasets.CIFAR100(dataset_dir, train=True, transform=transform_val)

    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, pin_memory=DEVICE == "cuda")

    return train_loaders, val_loader