from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from torch.utils.data import Subset, Dataset, DataLoader
from torch import load
from torchvision.datasets import CIFAR100
import os
import torchvision.transforms as t
import matplotlib.pyplot as plt
import torch
from train import DEBUG

B = 5   # minibatch size (batch size of each client)
C = 0.1 # fraction of clients
K = 100  # clients
LOSS_CRITERION = CrossEntropyLoss()
ROUNDS = ([1] if DEBUG else [20, 20, 10, 5])
J = ([1] if DEBUG else [4, 4, 8, 16])   # local steps of each client
IID = [True, False, False, False]
NC = [1, 5, 10, 50]  # number of classes in each client (for non iid)

IMGS_SHOWN = 0

def load_dataset(t_transforms, e_transforms) -> tuple[Subset, Subset]:
    dataset_dir = os.path.dirname(os.path.abspath(__file__)) + "/../dataset"
    dataset = CIFAR100(dataset_dir, train=True, download=True, transform=t_transforms)
    train_idx, eval_idx = train_test_split(list(range(len(dataset))), test_size=0.1, random_state=1234,
                                           stratify=dataset.targets)
    t_dataset = Subset(dataset, train_idx)
    dataset = CIFAR100(dataset_dir, train=True, download=True, transform=e_transforms)
    e_dataset = Subset(dataset, eval_idx)
    print("[SETUP] - train and eval datasets loaded")
    return t_dataset, e_dataset

"""
def define_eval_sets(clients_datasets: list[SubsetToDataset]) -> list[tuple[SubsetToDataset, SubsetToDataset]]:
    defined_sets = []
    for c in clients_datasets:
        train_idx, eval_idx = train_test_split(list(range(len(c))), test_size=0.1, random_state=1234, stratify=c.targets)
        defined_sets.append(
            (
                SubsetToDataset(Subset(c, train_idx)), SubsetToDataset(Subset(c, eval_idx))
            )
        )
    return defined_sets
"""

def define_transforms():
    transform_train = t.Compose([
        t.Resize(256, interpolation=t.InterpolationMode.BICUBIC),
        t.RandomCrop(224),
        t.RandomHorizontalFlip(),
        t.ToTensor(),
        t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet mean and std
    ])
    transform_val = t.Compose([
        t.Resize(224, interpolation=t.InterpolationMode.BICUBIC),
        t.ToTensor(),
        t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet mean and std
    ])
    print(f"[SETUP] - transforms defined")
    return transform_train, transform_val

def enable_transforms(datasets: tuple[list[SubsetToDataset], SubsetToDataset], train_transform: t.Compose, eval_transform: t.Compose):
    i = 0
    tr_datasets, ev_dataset = datasets
    for tr in tr_datasets:
        tr.add_transforms(train_transform)
        print(f"SETUP: transforms applied to client #{i}")
        i += 1
    ev_dataset.add_transforms(eval_transform)
    print("SETUP: transforms applied to eval dataset")

def create_client_dataloaders(t_datasets: list[Subset], e_dataset: Subset) -> tuple[list[DataLoader], DataLoader, list[int]]:
    t_dataloaders = []
    i = 0
    for td in t_datasets:
        train_loader = DataLoader(td, batch_size=B, shuffle=True, num_workers=2)
        t_dataloaders.append(train_loader)
        print(f"[DATALOADER]: client #{i}")
        i += 1
    e_dataloader = DataLoader(e_dataset, batch_size=B, shuffle=True)
    print(f"[DATALOADER]: eval")
    return t_dataloaders, e_dataloader, [len(d) for d in t_datasets]

def show_images():
    dataset_dir = os.path.dirname(os.path.abspath(__file__)) + "/../dataset"
    dataset = CIFAR100(dataset_dir, train=True, download=True)
    for i in range(IMGS_SHOWN):
        plt.imshow(dataset[i][0], cmap="Greys_r")
        plt.title(dataset.classes[dataset[i][1]])
        plt.show()

def calculate_client_contributions(w_local: list[dict], samples_selected: list[int], total_selected: int) -> dict:
    # update weight of each client based on a contribution factor
    for i in range(len(w_local)):
        for layer in w_local[i]:
            w_local[i][layer] = w_local[i][layer] * samples_selected[i] / total_selected
    # sum weights of same layer in all clients
    return {layer_name: layer_wise_addition(w_local, layer_name, len(w_local)) for layer_name in w_local[0]}

def layer_wise_addition(scaled_w_local, layer_name, clients_amount):
    s = scaled_w_local[0][layer_name]
    for i in range(1, clients_amount):
        s += scaled_w_local[i][layer_name]
    return s

def set_label_threshold_for_debug(main_dataset: Subset, threshold: int) -> Subset:
    if threshold < 1:
        return main_dataset
    indices_to_keep = [i for i, (_, l) in enumerate(main_dataset) if l < threshold]
    print(f"[SETUP] - filtering only samples of the following classes: {[l for l in range(0, threshold)]}")
    return Subset(main_dataset, indices_to_keep)
    # return SubsetToDataset(Subset(main_dataset, indices_to_keep))


def eval_phase(model, dataloader, w_path):
    model.load_state_dict(load(w_path))
    size = len(dataloader.dataset)
    eval_loss, correct = 0, 0
    with torch.no_grad():
        for b, (img, trg) in enumerate(dataloader):
            pred = model(img)
            eval_loss += LOSS_CRITERION(pred, trg).item()
            correct += (pred.argmax(1) == trg).type(torch.float).sum().item()
            print(f"[EVAL BATCH {b}/{len(dataloader) - 1}]")
    eval_loss /= size
    correct /= size
    print(f"Accuracy: {100 * correct}, avg loss: {eval_loss}")

# this is done to avoid getting data in this format all the time: subset.dataset.data
class SubsetToDataset(Dataset):
    def __init__(self, subset: Subset):
        self.data = subset.dataset.data[subset.indices]
        self.targets = [subset.dataset.targets[x] for x in subset.indices]
        self.transform = None

    def add_transforms(self, trn: t.Compose):
        self.transform = trn

    def __getitem__(self, index):
        if self.transform is not None:
            return self.transform(self.data[index]), self.targets[index]
        else:
            return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.shape[0]
