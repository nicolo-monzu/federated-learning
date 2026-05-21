DEBUG = True

from train_federated import train_client
from data.client_dataloader import create_client_dataloaders
from data.sharding import iid_sharding, advanced_non_iid_sharding
from utils.fedavg_utils import selection_wrapper, get_initial_weights, K, Dino_vits16_100
from torch.utils.data import Subset

ROUNDS = 10
NC = 3  # number of classes in each client (for non iid)
LEARNING_RATE = 0
IID = True


def FedAvg(dataset: torchvision.datasets.CIFAR100):
    # sharding: turn (train + eval) dataset into [train_0 + eval_0, ..., train_k-1 + eval_k-1]
    if IID:
        clients = iid_sharding(main_dataset=dataset, k=K)
    else:
        clients = advanced_non_iid_sharding(main_dataset=dataset, k=K, nc=NC)

    # get initial weights and biases
    w_global = get_initial_weights()

    for round_ in range(ROUNDS):

        # selection: among [train_0 + eval_0,  ..., train_k-1 + eval_k-1] choose a portion of it,
        # generating [s_train_0 + s_eval_0,  ..., train_m-1 + eval_m-1]
        # with m <= k
        selected_clients, samples_per_client, total_samples_selected = selection_wrapper(clients)   # subset list, [n_0, n_m-1], mt

        # dataloader creation: for all selected (s_train_i + s_eval_i) generate (s_train_i, s_eval_i)
        selected_dataloaders = create_client_dataloaders(selected_clients)

        w_local_new = []
        for client_dataloader in selected_dataloaders:
            w_local_new.append(ClientUpdate(client_dataloader, w_global))
        w_new = [w_local_new[i] * samples_per_client[i]/total_samples_selected for i in range(len(samples_per_client))]
        w_global = sum(w_new)

def ClientUpdate(client_dataloader: Subset, w_global):
    train_client(Dino_vits16_100, client_dataloader)

    return w_global

if DEBUG:
    import os
    import torchvision.transforms as T
    import torchvision
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
    dataset_ = torchvision.datasets.CIFAR100(dataset_dir, train=True, download=True, transform=transform_train)
    FedAvg(dataset_)
