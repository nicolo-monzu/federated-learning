from train import DEVICE
from fedavg_utils import calculate_client_contributions, C, K, LOSS_CRITERION
from torch.optim import SGD
from models.model import Dino_vits16_100
from torch.utils.data import DataLoader
from torch import save
import random
from copy import deepcopy
from torch.optim.lr_scheduler import CosineAnnealingLR

FEDAVG_WEIGHTS_PATH = "fedavg_weights.pth"


def FedAvg(model: Dino_vits16_100, t_dataloaders: list[DataLoader], samples_dataloaders: list[int], local_steps: int, rounds: int):
    print("[FEDAVG] - w0 acquired")
    w_global = model.state_dict()   # get initial weights and biases
    for round_ in range(rounds):
        # selection: among [train_0 + eval_0,  ..., train_k-1 + eval_k-1] choose a portion of it,
        # generating [s_train_0 + s_eval_0,  ..., train_m-1 + eval_m-1]
        # with m <= k
        selected_indices = random.sample(range(len(t_dataloaders)), int(max(C * K, 1)))   # subset list, [n_0, n_mt-1]
        selected_dataloaders = [t_dataloaders[i] for i in selected_indices] # [n0, n1, ..., nm-1]
        samples_selected = [samples_dataloaders[i] for i in selected_indices]
        total_selected = sum(samples_selected)
        print(f"\n[ROUND {round_}/{rounds-1}] - CLIENT SELECTION: {selected_indices}, samples: {samples_selected}, total {total_selected}")
        # local training in each client
        i = 0
        w_local = []
        for client_dataloader in selected_dataloaders:
            print(f"\t[CLIENT {i}/{len(selected_dataloaders) - 1}]")
            i += 1
            w_local.append(ClientUpdate(client_dataloader, deepcopy(model), local_steps))

        # scale weights on each client based on contribution and sum them all layer-wise
        print(f"[ROUND {round_}/{rounds-1}] - SERVER UPDATE: calculating client contributions")
        w_global = calculate_client_contributions(w_local, samples_selected, total_selected)
        print(f"[ROUND {round_}/{rounds-1}] - SERVER UPDATE: global parameters updated")
        model.load_state_dict(w_global)

    print("\n[FEDAVG] - terminated")
    save(w_global, FEDAVG_WEIGHTS_PATH)
    return FEDAVG_WEIGHTS_PATH


def ClientUpdate(client_dataloader: DataLoader, model: Dino_vits16_100, local_steps: int) -> dict:
    optimizer = SGD(model.parameters())  # , momentum=0.9, weight_decay=WEIGHT_DECAY)
    CosineAnnealingLR(optimizer)
    for epoch in range(local_steps):
        print(f"\t\t[LOCAL EPOCH {epoch}/{local_steps-1}]")
        for batch, (img, trg) in enumerate(client_dataloader):
            print(f"\t\t\t[MINIBATCH {batch}/{len(client_dataloader)-1}] - TRAINING")
            img, trg = img.to(DEVICE), trg.to(DEVICE)
            pred = model(img)
            loss = LOSS_CRITERION(pred, trg)
            loss.backward()
            optimizer.step()
    return model.state_dict()
