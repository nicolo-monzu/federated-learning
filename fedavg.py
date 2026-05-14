import random

from data.sharding import advanced_non_iid_sharding, iid_sharding
from torch.utils.data import Subset
from data.client_dataloader import create_client_dataloaders
from train_federated import train_client
from models import model

ROUNDS = 10
C = 0.1 # fraction of clients
K = 10  # clients
NC = 3  # number of classes in each client (for non iid)
B = 5   # minibatch size (batch size of each client)
E = 3   # local epoch of each client
LEARNING_RATE = 0
IID = True


def FedAvg(train_dataset: Subset, eval_dataset: Subset):
    clients_train, clients_eval = sharding_wrapper(train_dataset, eval_dataset)
    w_global = model_.state_dict()
    for round_ in range(ROUNDS):
        selected_clients, samples_per_client, total_samples_selected = (selection_wrapper(clients_train, clients_eval))
        w_local_new = []
        for client in selected_clients:
            w_local_new.append(ClientUpdate(client, w_global))
        w_new = [w_local_new[i] * samples_per_client[i]/total_samples_selected for i in range(amount_selected_clients)]
        w_global = sum(w_new)

def ClientUpdate(client: Subset, w_global):
    train_dataloader, eval_dataloader = create_client_dataloaders(client, B)
    train_client(model_, train_dataloader, epochs=B)

    return w_global

def sharding_wrapper(train_dataset: Subset, eval_dataset: Subset):
    clients_train, clients_eval = None, None
    if IID is True:
        clients_train = iid_sharding(main_dataset=train_dataset, k=K)
        clients_eval = iid_sharding(main_dataset=eval_dataset, k=K)
    else:
        clients_train = advanced_non_iid_sharding(main_dataset=train_dataset, k=K, nc=NC)
        clients_eval = advanced_non_iid_sharding(main_dataset=eval_dataset, k=K, nc=NC)
    return clients_train, clients_eval

def selection_wrapper(clients_train, clients_eval):
    amount_selected_clients = int(max(C * K, 1))
    selected_indices = random.sample(range(len(clients_train)), amount_selected_clients)
    selected_clients_train = [clients_train[i] for i in selected_indices]
    selected_clients_eval = [clients_eval[i] for i in selected_indices]
    samples_per_client = [len(client_train) for client_train in selected_clients_train]
    total_samples_selected = sum(samples_per_client)
    return list(zip(selected_clients_train, selected_clients_eval)), samples_per_client, total_samples_selected