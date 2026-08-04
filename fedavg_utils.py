from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss, Module
from torch.utils.data import Subset, DataLoader
from torch import load
from torchvision.datasets import CIFAR100
import os
import torchvision.transforms as t
import matplotlib.pyplot as plt
import torch
from random import uniform
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, MultiplicativeLR, SequentialLR
from torch.optim import SGD
from fedavg_logger import FEDAVGLogger
from train import DEBUG

B = 128   # minibatch size (batch size of each client)
C = 0.1 # fraction of clients
K = 100  # clients
LOSS_CRITERION = CrossEntropyLoss()
WEIGHT_DECAY = weight_decay = 10 ** uniform(-6, -2)

ROUNDS = ([1] if DEBUG else [20, 20, 10, 5])
J = ([1] if DEBUG else [4, 4, 8, 16])  # local steps of each client
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


def eval_phase(model, dataloader, w_path, logger):
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
    logger.add_eval_results(100 * correct, eval_loss)


def generate_hyperparameters_combinations(model: Module):
    loggers = []
    for i, n_rounds in enumerate(ROUNDS):
        if not IID[i]:
            for n_classes in NC:
                loggers.append(FEDAVGLogger(IID[i], n_rounds, J[i], generate_scheduler(model, J[i]), n_classes))
        else:
            loggers.append(FEDAVGLogger(IID[i], n_rounds, J[i], generate_scheduler(model, J[i])))
    return loggers


def generate_scheduler(model, j):
    optimizer = SGD(model.parameters(), momentum=0.9, lr=18)
    #warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=int(j / 4))
    cosine_sched = CosineAnnealingLR(optimizer, T_max=j)
    #constant_sched = MultiplicativeLR(optimizer, lr_lambda=lambda epoch: j / 4)
    # scheduler = SequentialLR(
    #    optimizer,
    #    schedulers=[warmup_sched, cosine_sched, constant_sched],
    #    milestones=[int(j / 4), int(3 * j / 4)]
    # )
    return cosine_sched