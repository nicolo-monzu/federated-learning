import numpy as np
from torch.utils.data import Subset

# K: number of clients
# Nc: number of classes per client
# C: total number of classes in the dataset

def assign_class_to_client(K, Nc, C):
    A = int(Nc * K / C)

    np_rng = np.random.default_rng(1234)

    tensor = np.ones(A, dtype=int) * np.arange(C).reshape(C,1)
    tensor = tensor.transpose()
    np_rng.permuted(tensor, axis=1, out=tensor) # Shuffle each row independently (in-place)
    tensor = tensor.reshape(K, Nc)
    np_rng.shuffle(tensor)

    return tensor


def divide_data(subset, K, Nc, C, rng):
    A = int(Nc * K / C)

    indices = np.asarray(subset.indices)
    targets = np.asarray(subset.dataset.targets)
    labels = targets[indices]

    chunks = [
        [chunk.tolist() for chunk in np.array_split(indices[labels == c], A)]
        for c in range(C)
    ]

    for row in chunks:
        rng.shuffle(row)

    return chunks


def split_subset_non_iid(subset, K, Nc, C, rng):
    chunks = divide_data(subset, K, Nc, C, rng)
    assigned = assign_class_to_client(K, Nc, C)

    indices = [[] for _ in range(K)]

    for i in range(assigned.shape[0]):
        for j in range(assigned.shape[1]):
            indices[i] += chunks[assigned[i, j]].pop()
        rng.shuffle(indices[i])


    subsets = [Subset(subset.dataset, i) for i in indices]
    return subsets