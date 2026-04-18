from torch.utils.data import Subset
import math

TESTING = True

def testing_stuff(main_dataset, threshold) -> Subset:
    indices_to_keep = [i for i, (_, l) in enumerate(main_dataset) if l <= 5]
    return Subset(main_dataset, indices_to_keep)

def printing_stuff(sub_datasets: list[Subset], classes_total: int, mode: str) -> None:
    print(f"\n\n{mode} MODE")
    for i, c in enumerate(sub_datasets):
        print(f"\nclient {i}")
        new_counters = [0 for i in range(classes_total)]
        for s, l in c:
            new_counters[l] += 1
        for i in range(classes_total):
            print("samples class " + str(i) + ": " + str(new_counters[i]))

def iid_sharding(main_dataset: Subset, k: int):
    """
    each training subset has the same distribution among classes

    main_dataset: training dataset
    k: amount of federated learning clients
    """
    if TESTING:
        main_dataset = testing_stuff(main_dataset, 5)

    classes_total = 1 + max([l for s, l in main_dataset])
    sub_datasets = []
    counters = [0 for i in range(classes_total)]    # each element represent the amount of samples of classes i encountered
    indices = [[] for i in range(k)] # each list is a set of indices included in a client
    for i, (s, l) in enumerate(main_dataset):
        current_index = counters[l] % k
        indices[current_index].append(i)
        counters[l] += 1
    for clients in range(k):
        sub_datasets.append(Subset(main_dataset, indices[clients]))

    if TESTING:
        printing_stuff(sub_datasets, classes_total, "iid")

    return


def non_iid_sharding(main_dataset: Subset, k: int, nc: int) -> list[Subset]:
    """
    each training subset has  nc different classes

    main_dataset: training dataset
    K: amount of federated learning clients
    nc: amount of different classes in each client
    """
    if TESTING:
        main_dataset = testing_stuff(main_dataset, 5)

    classes_total = 1 + max([l for s, l in main_dataset])
    if k > 1 + classes_total//nc:
        k = math.ceil(classes_total/nc)
        print("error: these parameters will result in empty clients. self adjusting k to ", k)
    elif k < classes_total/nc:
        k = math.ceil(classes_total / nc)
        print("error: these parameters will result in not enough available classes. self adjusting k to ", k)
    sub_datasets = []
    indices = [[] for i in range(k)]
    for i in range(k):
        indices[i] = [i2 for i2, (_, label) in enumerate(main_dataset) if label//nc == i]
        sub_datasets.append(Subset(main_dataset, indices[i]))

    if TESTING:
        printing_stuff(sub_datasets, classes_total, "non iid")

    return sub_datasets