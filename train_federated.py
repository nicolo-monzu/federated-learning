from train import DEBUG, DEVICE, Dino_vits16_100
from fedavg import FedAvg
from fedavg_utils import (load_dataset, set_label_threshold_for_debug, create_client_dataloaders, define_transforms,
                          show_images, eval_phase, K, IID, NC, J, ROUNDS)
from data.sharding import iid_sharding, advanced_non_iid_sharding
from copy import deepcopy

if __name__ == "__main__":

    model = Dino_vits16_100().to(DEVICE)
    print("[SETUP] - model loaded")

    if DEBUG:
        show_images()

    t_transforms, e_transforms = define_transforms()
    t_dataset, e_dataset = load_dataset(t_transforms, e_transforms)

    t_dataset = set_label_threshold_for_debug(t_dataset, 5 if DEBUG else 0)
    e_dataset = set_label_threshold_for_debug(e_dataset, 5 if DEBUG else 0)

    for i in range(4):
        # sharding: train => [train_0, ..., train_k-1]
        t_datasets = iid_sharding(main_dataset=deepcopy(t_dataset), k=K) if IID[i] else (advanced_non_iid_sharding(main_dataset=deepcopy(t_dataset), k=K, nc=NC[i]))

        # dataloader creation
        train_dataloaders, eval_dataloader, samples_client_dataloaders = create_client_dataloaders(t_datasets, e_dataset)

        # calculate averaged weights
        w_path = FedAvg(model, train_dataloaders, samples_client_dataloaders, J[i], ROUNDS[i])

        # evaluation phase
        eval_phase(model, eval_dataloader, w_path)
