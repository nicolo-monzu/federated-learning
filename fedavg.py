from torchvision.datasets import CIFAR100

DEBUG = True

from train_federated import train_client
from data.client_dataloader import create_client_dataloaders
from data.sharding import iid_sharding, advanced_non_iid_sharding
from utils.fedavg_utils import calculate_client_contributions, K, C
from models.model import Dino_vits16_100
from torch.utils.data import DataLoader
from torch.nn import Module
from torch import save, load
from torch.cuda import is_available
import random

ROUNDS = 10
NC = 3  # number of classes in each client (for non iid)
LEARNING_RATE = 0
E = 3   # local epoch of each client
IID = True
DEVICE = "cuda" if is_available() else "cpu"
FEDAVG_WEIGHTS_PATH = "fedavg_weights.pth"


def FedAvg(dataset: torchvision.datasets.CIFAR100, model: Module) -> str:
    # sharding: turn (train + eval) dataset into [train_0 + eval_0, ..., train_k-1 + eval_k-1]
    if IID:
        clients = iid_sharding(main_dataset=dataset, k=K)
    else:
        clients = advanced_non_iid_sharding(main_dataset=dataset, k=K, nc=NC)

    # get initial weights and biases
    if DEBUG:
        print("SETUP: w0 acquired")
    w_global = model.state_dict()

    # dataloader creation: for all selected (s_train_i + s_eval_i) generate (s_train_i, s_eval_i)
    dataloaders, samples_all = create_client_dataloaders(clients)
    if DEBUG:
        print(f"SETUP: dataloaders created, samples in train dataloaders: {samples_all}")

    for round_ in range(ROUNDS):

        # selection: among [train_0 + eval_0,  ..., train_k-1 + eval_k-1] choose a portion of it,
        # generating [s_train_0 + s_eval_0,  ..., train_m-1 + eval_m-1]
        # with m <= k
        selected_indices = random.sample(range(len(dataloaders)), int(max(C * K, 1)))   # subset list, [n_0, n_m-1], mt
        selected_dataloaders = [dataloaders[i] for i in selected_indices] # [n0, n1, ..., nm-1]
        samples_selected = [samples_all[i] for i in selected_indices]
        total_selected = sum(samples_selected)
        if DEBUG:
            print(f"CLIENT SELECTION: round #{round_ + 1}, selected clients {selected_indices}, "
                  f"samples dataloaders: {samples_selected}, total {total_selected}")


        # local training in each client
        i = 1
        w_local = []
        for client_dataloader in selected_dataloaders:
            print(f"CLIENT TRAINING: round {round_}/{ROUNDS} - client {i}/{len(selected_dataloaders)}")
            i += 1
            w_local.append(ClientUpdate(client_dataloader, w_global, model_))

        # scale weights on each client based on contribution and sum them all layer-wise
        w_global = calculate_client_contributions(w_local, samples_selected, total_selected)

    save(w_global, FEDAVG_WEIGHTS_PATH)
    return FEDAVG_WEIGHTS_PATH


def ClientUpdate(client_dataloader: DataLoader, w_global, model__: Dino_vits16_100) -> dict:
    train_dl, eval_dl = client_dataloader
    if DEBUG:
        return w_global
    for epoch in range(E):
        train_client(model__, client_dataloader)
    return w_global

if DEBUG:
    import os
    import torchvision.transforms as t
    import torchvision
    dataset_dir = os.path.dirname(os.path.abspath(__file__))+"/../dataset"
    transform_train = t.Compose([
        t.Resize(256, interpolation=t.InterpolationMode.BICUBIC),
        t.RandomCrop(224),
        t.RandomHorizontalFlip(),
        t.ToTensor(),
        t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet mean and std
    ])
    transform_val = t.Compose([
        t.Resize(224, interpolation=t.InterpolationMode.BICUBIC),
        t.ToTensor(),
        t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet mean and std
    ])
    dataset_ = torchvision.datasets.CIFAR100(dataset_dir, train=True, download=True, transform=transform_train)
    if DEBUG:
        print("SETUP: dataset loaded")
    # model_ = Dino_vits16_100().to_device(DEVICE)
    model_ = torchvision.models.resnet50()
    if DEBUG:
        print("SETUP: model loaded")
    path = FedAvg(dataset_, model_)
    model_.load_state_dict(load(path))
