from torch.utils.data import Subset
import math
import matplotlib.pyplot as plt
from .sharding_classes import Class, Client, find_available_client

TESTING = True
THRESHOLD = 20

def testing_stuff(main_dataset, threshold) -> Subset:
    indices_to_keep = [i for i, (_, l) in enumerate(main_dataset) if l < threshold]
    return Subset(main_dataset, indices_to_keep)

def printing_stuff(sub_datasets: list[Subset], classes_total: int, mode: str, clients: int, nc: int=0) -> None:
    # print(f"\n\n{mode} MODE")
    counter_matrix = [[] for _ in range(clients)]
    for i, c in enumerate(sub_datasets):
        # print(f"\nclient {i}")
        counter_matrix[i] = [0 for _ in range(classes_total)]
        for s, l in c:
            counter_matrix[i][l] += 1
        # for j in range(classes_total):
            # print("samples class " + str(j) + ": " + str(counter_matrix[i][j]))

    fig, ax = plt.subplots()
    im = ax.imshow(counter_matrix, cmap="viridis_r")
    ax.set_title(f"{mode}, clients: {clients}" + (", nc: " + str(nc) if mode.__contains__("non-iid") else ""))
    ax.set_xticks(range(classes_total), labels=[f"class {i}" for i in range(classes_total)],
                  rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(clients), labels=[f"client {i}" for i in range(clients)])
    for i in range(clients):
        for j in range(classes_total):
            text = ax.text(j, i, counter_matrix[i][j],
                           ha="center", va="center", color="w")
    fig.tight_layout()
    plt.show()

def prevent_code_breaking(main_dataset: Subset, classes_total, k) -> Subset:
    labels = [l for (_, l) in main_dataset]
    indices = [i for i in range(len(labels))]
    new_indices = []
    for c in range(k):  # iterate thru clients
        for l in range(classes_total):  # iterate thru classes
            target_index = labels.index(l)
            new_indices.append(target_index)
            labels[target_index] = -1
            indices[target_index] = -1
    return Subset(main_dataset, new_indices + [i for i in indices if i != -1])


def iid_sharding(main_dataset: Subset, k: int):
    """
    each training subset has the same distribution among classes

    main_dataset: training dataset
    k: amount of federated learning clients
    """
    if TESTING:
        main_dataset = testing_stuff(main_dataset, THRESHOLD)

    classes_total = 1 + max([l for s, l in main_dataset])
    sub_datasets = []
    counters = [0 for _ in range(classes_total)]    # each element represent the amount of samples of classes i encountered
    indices = [[] for _ in range(k)] # each list is a set of indices included in a client
    for i, (s, l) in enumerate(main_dataset):
        current_index = counters[l] % k
        indices[current_index].append(i)
        counters[l] += 1
    for clients in range(k):
        sub_datasets.append(Subset(main_dataset, indices[clients]))

    if TESTING:
        printing_stuff(sub_datasets, classes_total, "iid", k)

    return sub_datasets

"""
def non_iid_sharding(main_dataset: Subset, k: int, nc: int) -> list[Subset]:
    if TESTING:
        main_dataset = testing_stuff(main_dataset, THRESHOLD)

    classes_total = 1 + max([l for s, l in main_dataset])
    main_dataset = prevent_code_breaking(main_dataset, classes_total, k)
    if k > 1 + classes_total//nc:
        k = math.ceil(classes_total/nc)
        print("error: these parameters will result in empty clients. self adjusting k to ", k)
    elif k < classes_total/nc:
        k = math.ceil(classes_total / nc)
        print("error: these parameters will result in not enough available classes. self adjusting k to ", k)
    sub_datasets = []
    indices = [[] for _ in range(k)]
    for i in range(k):
        indices[i] = [i2 for i2, (_, label) in enumerate(main_dataset) if label//nc == i]
        sub_datasets.append(Subset(main_dataset, indices[i]))

    if TESTING:
        printing_stuff(sub_datasets, classes_total, "non-iid", k, nc)

    return sub_datasets
"""

def advanced_non_iid_sharding(main_dataset: Subset, k: int, nc: int) -> list[Subset]:
    if TESTING:
        main_dataset = testing_stuff(main_dataset, THRESHOLD)

    classes_total = 1 + max([l for s, l in main_dataset])
    main_dataset = prevent_code_breaking(main_dataset, classes_total, k)
    if TESTING:
        limit = classes_total * k
        print([l for (_, l) in main_dataset][:limit])
        print([l for (_, l) in main_dataset][limit:2*limit])
    if nc * k < classes_total:
        print(f"combination of clients and classes per client cannot accomodate all classes! (Nc * k) < {classes_total}")
        exit(0)
    if nc > classes_total:
        print(f"required classes in each client ({nc}) are more than total classes present in the dataset ({classes_total})")
        exit(0)
    clients_per_class = (nc * k)//classes_total
    remainder = (nc * k) % classes_total
    classes = [Class(i, clients_per_class + int(i < remainder)) for i in range(classes_total)]
    clients = [Client(nc) for _ in range(k)]
    for i, (s, l) in enumerate(main_dataset):
        target_client, new_client_flag = find_available_client(clients, classes[l])
        if new_client_flag:
            clients[target_client].add_classes(l)
            classes[l].add_client(target_client)
        clients[target_client].add_index(i)
        classes[l].update_counter()
    sub_datasets = [Subset(main_dataset, indices) for indices in [c.get_indices() for c in clients]]

    if TESTING:
        printing_stuff(sub_datasets, classes_total, " advanced non-iid", k, nc)

    return sub_datasets