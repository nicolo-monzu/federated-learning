from torch.utils.data import Subset, Dataset
import random
from torch.nn import Module

C = 0.5 # fraction of clients
K = 10  # clients


def calculate_client_contributions(w_local: list[dict], samples_selected: list[int], total_selected: int) -> dict:
    # update weight of each client based on a contribution factor
    for i in range(len(w_local)):
        for layer in w_local[i]:
            w_local[i][layer] = w_local[i][layer] * samples_selected[i] / total_selected

    # sum weights of same layer in all clients
    return {layer: layer_wise_addition(w_local, layer, len(w_local)) for layer in w_local[0]}

def layer_wise_addition(w_local, layer, clients_amount):
    s = w_local[0][layer]
    for i in range(1, clients_amount):
        s += w_local[i][layer]
    return s

def set_weights(model: Module, new_params: dict[str, list[float]]) -> Module:
    for key in new_params.keys():
        model.parameters()
    return model

class SubsetToDataset(Dataset):
    def __init__(self, subset: Subset):
        self.data = subset.dataset.data[subset.indices]
        self.targets = [subset.dataset.targets[x] for x in subset.indices]
    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    def __len__(self):
        return self.data.shape[0]
