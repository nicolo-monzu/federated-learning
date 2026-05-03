import random

from data.sharding import advanced_non_iid_sharding, iid_sharding
from torch.utils.data import Subset
from data.dataloader import create_dataloaders

ROUNDS = 10
C = 0.1 # fraction of clients
K = 10  # clients
NC = 3  # number of classes in each client (for non iid)
B = 5   # minibatch size (batch size of each client)
E = 3   # local epoch of each client
LEARNING_RATE = 0
IID = True


def FedAvg(main_dataset: Subset):
    clients = iid_sharding(main_dataset=main_dataset, k=K) if IID is True else (
        advanced_non_iid_sharding(main_dataset=main_dataset, k=K, nc=NC))
    w_global = []
    for round_ in range(ROUNDS):
        amount_selected_clients = int(max(C * K, 1))
        selected_clients = random.sample(clients, amount_selected_clients)
        w_local_new = []
        for client in selected_clients:
            w_local_new.append(ClientUpdate(client, w_global))
        samples_per_client = [len(client) for client in selected_clients]
        total_samples_selected = sum(samples_per_client)
        w_new = [w_local_new[i] * samples_per_client[i]/total_samples_selected for i in range(amount_selected_clients)]
        w_global = sum(w_new)

def ClientUpdate(client, w_global):
    # add dataloader
    create_dataloaders(B)
    return w_global