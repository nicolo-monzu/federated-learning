from torch.utils.data import Subset, Dataset
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
    # for layer in Dino_vits16_100:
    return None

class SubsetToDataset(Dataset):
    def __init__(self, subset: Subset):
        self.data = subset.dataset.data[subset.indices]
        self.targets = [subset.dataset.targets[x] for x in subset.indices]
    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    def __len__(self):
        return self.data.shape[0]