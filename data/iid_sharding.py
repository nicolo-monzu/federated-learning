import numpy as np
from torch.utils.data import Subset


def assign_chunks_to_client(N):
    np_rng = np.random.default_rng(1234)

    tensor = (np.arange(N).reshape((N, 1)) + np.arange(N)) % N
    np_rng.shuffle(tensor, axis=0) # Shuffle rows
    np_rng.shuffle(tensor, axis=1) # Shuffle columns
    return tensor


def divide_data(subset, N):
    indices = np.asarray(subset.indices)
    targets = np.asarray(subset.dataset.targets)
    labels = targets[indices]

    chunks = [
        [chunk.tolist() for chunk in np.array_split(indices[labels == c], N)]
        for c in range(N)
    ]

    return chunks


def split_subset_iid(subset, N, rng):
    chunks = divide_data(subset, N)
    assigned = assign_chunks_to_client(N)

    indices = [[] for _ in range(N)]

    for i in range(assigned.shape[0]):
        for j in range(assigned.shape[1]):
            indices[i] += chunks[j][assigned[i, j]]
        rng.shuffle(indices[i])

    subsets = [Subset(subset.dataset, i) for i in indices]
    return subsets