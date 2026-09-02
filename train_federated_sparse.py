"""
Combination of "FederatedAveraging" algorithm [1] with Task-Localized Sparse Fine-tuning [2]

[1] McMahan et al., Communication-Efficient Learning of Deep Networks fromDecentralized Data, AISTATS 2017
[2] Iurada et al., Efficient Model Editing with Task-Localized Sparse Fine-tuning. ICLR 2025.
"""


import argparse
import os
from datetime import datetime
import torch
import yaml
from torch import nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, MultiplicativeLR, SequentialLR

from SparseSGDM import SparseSGDM
from checkpoint_manager_federated import CheckpointManagerFederated
from data.dataloader import DEVICE
from data.dataloader_federated import create_dataloader_federated
from logger import Logger
from masking import MaskRule, calibrate_federated_mask
from plot import plot_training
from models.model import Dino_vits16_100
from train import USE_AMP, apply_llrd

from train_federated import Client, train_federated


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config["train_federated_sparse"]

config = load_config()

class ClientSparse(Client):
    def __init__(self, model, criterion, train_loader):
        super().__init__(model, criterion, train_loader)
        self.mask_by_param = None

    def set_mask(self, mask_by_name):
        self.mask_by_param = {
            param: mask_by_name[name]
            for name, param in self.model.named_parameters()
            if name in mask_by_name
        }

    def update(self, num_steps, model_dict, learning_rate, decay_rate, weight_decay, grad_scale):
        # load server weights
        self.model.load_state_dict(model_dict)
        # create optimizer
        parameters = apply_llrd(self.model, learning_rate, decay_rate, backbone_only=True)
        optimizer = SparseSGDM(parameters, momentum=0.9, weight_decay=weight_decay, masks=self.mask_by_param)
        # create scaler
        scaler = torch.amp.GradScaler("cuda", init_scale=grad_scale, enabled=USE_AMP)
        # training one epoch
        train_loss, new_grad_scale = self._train_for_num_steps(num_steps, optimizer, scaler)
        # return new weights
        return self.model.state_dict(), train_loss, new_grad_scale


def set_parameters_requires_grad(model):
    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze desired modules in the backbone
    for name, module in model.backbone.named_modules():

        # Keep patch embedding frozen
        if name.startswith("patch_embed"):
            continue

        if isinstance(module, (nn.Linear, nn.MultiheadAttention, nn.LayerNorm, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            for param in module.parameters():
                param.requires_grad = True


def build_training_objects(run):
    # Import data
    train_loaders, val_loader = create_dataloader_federated(run['batch_size'], run['num_clients'], run['num_classes_per_client'])

    # Define model
    model = Dino_vits16_100().to(DEVICE)
    set_parameters_requires_grad(model)

    criterion = nn.CrossEntropyLoss()

    # Create clients
    clients = [ClientSparse(model, criterion, dataloader) for dataloader in train_loaders]

    # Scheduler
    warmup_scheduler_steps = run['warmup_scheduler_steps']
    cosine_scheduler_steps = run['cosine_scheduler_steps']
    dummy_optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=run['max_learning_rate'])
    warmup_sched = LinearLR(dummy_optimizer, start_factor=0.1, total_iters=warmup_scheduler_steps + 1)
    cosine_sched = CosineAnnealingLR(dummy_optimizer, T_max=cosine_scheduler_steps)
    constant_sched = MultiplicativeLR(dummy_optimizer, lr_lambda=lambda epoch: 1.0)

    scheduler = SequentialLR(
        dummy_optimizer,
        schedulers=[warmup_sched, cosine_sched, constant_sched],
        milestones=[warmup_scheduler_steps, warmup_scheduler_steps + cosine_scheduler_steps]
    )

    return clients, val_loader, model, criterion, scheduler


def resume(run_name, total_rounds, patience, checkpoints_dir, logs_dir, plots_dir, separate=False):

    # Cleanup, restoring files and checkpoint loading
    manager = CheckpointManagerFederated(checkpoints_dir, run_name, sparse=True)
    logger_state_dict, round = manager.resume()

    logger = Logger(logs_dir, run_name, federated=True)
    logger.resume(logger_state_dict)

    if separate:
        run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        manager.set_run_name(run_name)
        logger.new_run_name(run_name)

    run = logger.get_run()

    print('Using device:', DEVICE)

    if USE_AMP:
        print('Using automatic mixed precision')

    clients, val_loader, model, criterion, scheduler = build_training_objects(run)

    # Restore state
    scale, mask = manager.restore_state(model, clients, scheduler)
    mask = {
        k: v.to(DEVICE) for k, v in mask.items()
    }

    # Run the training process for {num_epochs} epochs
    print(f'Run name: {run['name']}')
    print('Resume training')

    train_federated(total_rounds, run, model, clients, val_loader, criterion, scheduler,
                    logger, manager, run['validation_interval'], patience, round + 1, scale, mask)

    plot_training(run_name, logs_dir, plots_dir, federated=True)
    return logger.get_run()


def start(num_rounds,
          num_steps_per_client,
          num_classes_per_client,
          rounds_per_scheduler_step,
          scale_grow_interval,
          validation_interval,
          num_clients=config["num_clients"],
          client_ratio=config["client_ratio"],
          warmup_scheduler_steps=config["warmup_scheduler_steps"],
          cosine_scheduler_steps=config["cosine_scheduler_steps"],
          sparsity = config["sparsity"],
          num_calibration_round = config["num_calibration_round"],
          mask_rule = config["mask_rule"],
          batch_size=config["batch_size"],
          max_lr=config["max_lr"],
          decay_rate=config["decay_rate"],
          weight_decay=config["weight_decay"],
          patience=config["patience"],
          checkpoints_dir=config["checkpoints_dir"],
          logs_dir=config["logs_dir"],
          plots_dir=config["plots_dir"],
          head_path = config["head_path"],
          run_name=None
          ):

    mask_rule = MaskRule(mask_rule)

    # Generate a run name if one was not provided
    if run_name is None:
        run_name = datetime.now().strftime('%Y%m%d_%H%M%S')

    if os.path.exists(f'{checkpoints_dir}/{run_name}.pth'):
        print(
            f'Error: checkpoint "{run_name}.pth" already exists. '
            'Training cannot start with this run name. '
            'Use "train_federated_sparse.py resume" to continue from the existing checkpoint, '
            'or remove/rename the checkpoint to start a new training run with this name.'
        )
        return {'best_accuracy': -1}

    # Init checkpoint manager and logger
    run = {
        'name': run_name,
        'model': 'dino_vits16_100_federated',
        'num_clients': num_clients,
        'num_steps_per_client': num_steps_per_client,
        'client_ratio': client_ratio,
        'num_classes_per_client': num_classes_per_client,
        'batch_size': batch_size,
        'max_learning_rate': max_lr,
        'decay_rate': decay_rate,
        'weight_decay': weight_decay,
        'optimizer': 'SGD(momentum=0.9)',
        'scheduler': 'CosineAnnealingLR with warm-up',
        'rounds_per_scheduler_step': rounds_per_scheduler_step,
        'warmup_scheduler_steps': warmup_scheduler_steps,
        'cosine_scheduler_steps': cosine_scheduler_steps,
        'scale_grow_interval': scale_grow_interval,
        'validation_interval': validation_interval,
        'sparsity': sparsity,
        'num_calibration_round': num_calibration_round,
        'mask_rule': mask_rule.name.lower(),
        'augmentation': 'MixUp/CutMix',
        'total_rounds': 0,
        'best_round': -1,
        'best_accuracy': -1.0,
    }
    manager = CheckpointManagerFederated(checkpoints_dir, run['name'], sparse=True)
    logger = Logger(logs_dir, run['name'], federated=True)
    logger.start(run)

    print('Using device:', DEVICE)

    clients, val_loader, model, criterion, scheduler = build_training_objects(run)

    # Load pre-trained head
    classifier_dict = torch.load(head_path, map_location=DEVICE)
    model.classifier.load_state_dict(classifier_dict)

    print(f'Run name: {run['name']}')

    mask = calibrate_federated_mask(model, clients, client_ratio, sparsity, num_calibration_round, mask_rule)

    for client in clients:
        client.set_mask(mask)

    manager.save(0, model, 2**16, scheduler, logger, -1.0, mask)

    # Run the training process for {num_rounds} rounds
    print('Start training')
    train_federated(num_rounds, run, model, clients, val_loader, criterion, scheduler, logger, manager, validation_interval, patience, mask=mask)
    plot_training(run['name'], logs_dir, plots_dir, federated=True)
    return logger.get_run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Start or resume a federated model training.")
    subparsers = parser.add_subparsers(dest="action", required=True)

    # Start
    start_parser = subparsers.add_parser(
        "start",
        help="Start a new federated training run"
    )

    start_parser.add_argument(
        "-n", "--run-name",
        type=str,
        default=None,
        help="Name of the new run. If omitted, a timestamp-based name is generated."
    )

    start_parser.add_argument(
        "-r", "--rounds",
        type=int,
        default=config["num_rounds"],
        help="Total number of federated training rounds. (default: config.yaml)"
    )

    # Resume
    resume_parser = subparsers.add_parser(
        "resume",
        help="Resume an existing federated training run"
    )

    resume_parser.add_argument(
        "-n", "--run-name",
        type=str,
        required=True,
        help="Name of the run to resume"
    )

    resume_parser.add_argument(
        "-r", "--total-rounds",
        type=int,
        default=config["num_rounds"],
        help="Total number of federated training rounds. (default: config.yaml)"
    )

    resume_parser.add_argument(
        "--separate",
        action="store_true",
        help=(
            "Resume training as a new run, preserving the original run's "
            "checkpoints and logs. (default: false)"
        )
    )

    args = parser.parse_args()

    if args.action == "start":
        start(
            num_rounds=args.rounds,
            num_steps_per_client=config["num_steps_per_client"],
            num_classes_per_client=config["num_classes_per_client"],
            rounds_per_scheduler_step=config["rounds_per_scheduler_step"],
            scale_grow_interval=config["scale_grow_interval"],
            validation_interval=config["validation_interval"],
            run_name=args.run_name
        )

    elif args.action == "resume":
        resume(
            run_name=args.run_name,
            total_rounds=args.total_rounds,
            patience=config["patience"],
            checkpoints_dir=config["checkpoints_dir"],
            logs_dir=config["logs_dir"],
            plots_dir=config["plots_dir"],
            separate=args.separate
        )