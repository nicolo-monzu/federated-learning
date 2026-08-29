import argparse
import os
import warnings

import torch
import torchvision
from numpy.exceptions import VisibleDeprecationWarning
from torch import nn
from torch.utils.data import DataLoader

from data.dataloader import transform_val, DEVICE
from models.model import Dino_vits16_100
from train import validate


def test(checkpoint_path):
    dataset_dir = os.path.dirname(os.path.abspath(__file__)) + "/dataset"

    model = Dino_vits16_100().to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    model.load_state_dict(checkpoint['best_model_state_dict'])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', VisibleDeprecationWarning)
        dataset = torchvision.datasets.CIFAR100(dataset_dir, train=False, download=True, transform=transform_val)

    val_loader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=DEVICE == "cuda")

    criterion = nn.CrossEntropyLoss()

    test_loss, test_accuracy = validate(model, val_loader, criterion, print_res=False)

    run_name = os.path.splitext(os.path.basename(checkpoint_path))[0]

    print(f'Run: {run_name} Test Loss: {test_loss:.6f} Acc: {test_accuracy:.2f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a checkpoint on the test set of CIFAR-100.'
    )
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to the checkpoint'
    )

    args = parser.parse_args()
    test(args.checkpoint)
