from torch.utils.data import Subset
from models.model import Dino_vits16_100
import random
import models

C = 0.1 # fraction of clients
K = 10  # clients

def selection_wrapper(clients: list[Subset]) -> (list[Subset], int, int):
    selected_indices = random.sample(range(len(clients)), int(max(C * K, 1)))
    selected_clients = [clients[i] for i in selected_indices]
    samples_per_client = [len(client_train) for client_train in selected_clients]
    return selected_clients, samples_per_client, sum(samples_per_client) # m, [n0, n1, ..., nm-1], mt

def get_initial_weights():
    for layer in Dino_vits16_100:
        layer

