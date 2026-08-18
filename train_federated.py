import argparse
import os
import random
import warnings
from datetime import datetime
import torch
import yaml
from torch import nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, MultiplicativeLR, SequentialLR
from tqdm import tqdm

from checkpoint_manager_federated import CheckpointManagerFederated
from data.dataloader import DEVICE
from data.dataloader_federated import create_dataloader_federated
from logger import Logger
from plot import plot_training
from models.model import Dino_vits16_100
from train import USE_AMP, apply_llrd, validate, apply_mixup_cutmix
from copy import deepcopy

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config["train_federated"]

config = load_config()

class Client:
    def __init__(self, model, criterion, train_loader):
        self.model = model # Shared reference (to save memory)
        self.criterion = criterion
        self.train_loader = train_loader

    def update(self, num_steps, model_dict, learning_rate, decay_rate, weight_decay, grad_scale):
        # load server weights
        self.model.load_state_dict(model_dict)
        # create optimizer
        parameters = apply_llrd(self.model, learning_rate, decay_rate)
        optimizer = torch.optim.SGD(parameters, momentum=0.9, weight_decay=weight_decay)
        # create scaler
        scaler = torch.amp.GradScaler("cuda", init_scale=grad_scale, enabled=USE_AMP)
        # training one epoch
        train_loss, new_grad_scale = self.__train_for_num_steps(num_steps, optimizer, scaler)
        # return new weights
        return self.model.state_dict(), train_loss, new_grad_scale

    def __train_for_num_steps(self, num_steps, optimizer, scaler):
        self.model.train()
        current_steps = 0
        running_loss = 0.0

        train_iter = iter(self.train_loader)
        while current_steps < num_steps:
            try:
                inputs, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                inputs, targets = next(train_iter)

            inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
            inputs, targets_mix = apply_mixup_cutmix(inputs, targets)

            with torch.amp.autocast(device_type=DEVICE, enabled=USE_AMP):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets_mix)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            # if the scaler executes optimizer.step()
            if scaler.step(optimizer, lambda: True):
                running_loss += loss.item()
                current_steps += 1
            scaler.update()

        train_loss = running_loss / num_steps
        new_grad_scale = scaler.get_scale()
        return train_loss, new_grad_scale

def running_sum(current, next_state):
    with torch.no_grad():
        if current is None:
            current = deepcopy(next_state)
        else:
            for k in current:
                current[k].add_(next_state[k])
    return current

def train_federated(num_rounds, run, model, clients, val_loader, criterion, scheduler, logger,
                    manager, validation_interval, patience, start_round = 1, grad_scale = 2**16):
    lr = scheduler.optimizer.param_groups[0]["lr"]

    # Dict to pass to clients
    with torch.no_grad():
        model_dict = deepcopy(model.state_dict())

    for round in range(start_round, num_rounds + 1):
        weights_sum = None
        running_loss = 0.0
        min_scale = grad_scale

        num_selected_clients = max(int(run['num_clients'] * run['client_ratio']), 1)
        progress_bar = tqdm(random.sample(clients, num_selected_clients), f'Train round {round}', leave=False)

        for client in progress_bar:
            client_weights, client_loss, client_scale = client.update(run['num_steps_per_client'], model_dict, lr, run['decay_rate'], run['weight_decay'], grad_scale)

            weights_sum = running_sum(weights_sum, client_weights)
            running_loss += client_loss
            min_scale = min(min_scale, client_scale)

        train_loss = running_loss / num_selected_clients

        print(f'Train Round: {round} Loss: {train_loss:.6f} Lr: {lr:e}')

        # Server
        with torch.no_grad():
            for k in weights_sum:
                weights_sum[k].div_(num_selected_clients)
                model_dict[k].copy_(weights_sum[k])

        val_loss, val_acc = None, None

        if round % validation_interval == 0  or round == num_rounds:
            model.load_state_dict(model_dict)
            val_loss, val_acc = validate(model, val_loader, criterion)

        logger.log(round, train_loss, val_loss, val_acc, lr)

        # update learning rate
        if round % run['rounds_per_scheduler_step'] == 0:
            with warnings.catch_warnings():
                # Suppression of "UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`."
                warnings.simplefilter('ignore', UserWarning)
                scheduler.step()

            lr = scheduler.optimizer.param_groups[0]["lr"]

        # update scale
        grad_scale = min_scale
        if round % run['scale_grow_interval'] == 0:
            grad_scale *= 2

        if round % validation_interval == 0 or round == num_rounds:
            manager.save(round, model, grad_scale, scheduler, logger, val_acc)

        # Early stopping
        if (run['total_rounds'] - run['best_round']) // validation_interval >= patience:
            break


def build_training_objects(run):
    # Import data
    train_loaders, val_loader = create_dataloader_federated(run['batch_size'], run['num_clients'], run['num_classes_per_client'])

    # Define model
    model = Dino_vits16_100().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # Create clients
    clients = [Client(model, criterion, dataloader) for dataloader in train_loaders]

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
    manager = CheckpointManagerFederated(checkpoints_dir, run_name)
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
    scale = manager.restore_state(model, clients, scheduler)

    # Run the training process for {num_epochs} epochs
    print(f'Run name: {run['name']}')
    print('Resume training')

    train_federated(total_rounds, run, model, clients, val_loader, criterion, scheduler,
                    logger, manager, run['validation_interval'], patience, round + 1, scale)

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
          batch_size=config["batch_size"],
          max_lr=config["max_lr"],
          decay_rate=config["decay_rate"],
          weight_decay=config["weight_decay"],
          patience=config["patience"],
          checkpoints_dir=config["checkpoints_dir"],
          logs_dir=config["logs_dir"],
          plots_dir=config["plots_dir"],
          run_name=None
          ):

    # Generate a run name if one was not provided
    if run_name is None:
        run_name = datetime.now().strftime('%Y%m%d_%H%M%S')

    if os.path.exists(f'{checkpoints_dir}/{run_name}.pth'):
        print(
            f'Error: checkpoint "{run_name}.pth" already exists. '
            'Training cannot start with this run name. '
            'Use "train.py resume" to continue from the existing checkpoint, '
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
        'augmentation': 'MixUp/CutMix',
        'total_rounds': 0,
        'best_round': -1,
        'best_accuracy': -1.0,
    }
    manager = CheckpointManagerFederated(checkpoints_dir, run['name'])
    logger = Logger(logs_dir, run['name'], federated=True)
    logger.start(run)

    print('Using device:', DEVICE)

    clients, val_loader, model, criterion, scheduler = build_training_objects(run)

    # Run the training process for {num_rounds} rounds
    print(f'Run name: {run['name']}')
    print('Start training')
    train_federated(num_rounds, run, model, clients, val_loader, criterion, scheduler, logger, manager, validation_interval, patience)
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
            num_clients=config["num_clients"],
            client_ratio=config["client_ratio"],
            num_steps_per_client=config["num_steps_per_client"],
            num_classes_per_client=config["num_classes_per_client"],
            rounds_per_scheduler_step=config["rounds_per_scheduler_step"],
            warmup_scheduler_steps=config["warmup_scheduler_steps"],
            cosine_scheduler_steps=config["cosine_scheduler_steps"],
            scale_grow_interval=config["scale_grow_interval"],
            validation_interval=config["validation_interval"],
            batch_size=config["batch_size"],
            max_lr=config["max_lr"],
            decay_rate=config["decay_rate"],
            weight_decay=config["weight_decay"],
            patience=config["patience"],
            checkpoints_dir=config["checkpoints_dir"],
            logs_dir=config["logs_dir"],
            plots_dir=config["plots_dir"],
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