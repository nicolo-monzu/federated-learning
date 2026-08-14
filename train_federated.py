import random
from datetime import datetime
import torch
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
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            # if the scaler executes optimizer.step()
            if scaler.step(optimizer, lambda: True):
                running_loss += loss.item()
                current_steps += 1
            scaler.update()

        train_loss = running_loss / num_steps
        new_grad_scale = scaler.get_scale()
        return train_loss, new_grad_scale

def running_model_avg(current, next_state, mul_factor):
    with torch.no_grad():
        if current is None:
            current = deepcopy(next_state)
            for k in current:
                current[k] *= mul_factor
        else:
            for k in current:
                current[k] += next_state[k] * mul_factor
    return current

def train_federated(num_rounds, run, model, clients, val_loader, criterion, scheduler, logger, manager, validation_interval, start_round = 1):
    lr = run['max_learning_rate']

    grad_scale = 2**16

    # Weights to pass to clients
    model_dict = deepcopy(model.state_dict())

    for round in range(start_round, num_rounds + 1):
        avg_weights = None
        running_loss = 0.0
        min_scale = grad_scale

        num_selected_clients = max(int(run['num_clients'] * run['client_ratio']), 1)
        for client in tqdm(random.sample(clients, num_selected_clients)):
            client_weights, client_loss, client_scale = client.update(run['num_steps_per_client'], model_dict, lr, run['decay_rate'], run['weight_decay'], grad_scale)

            avg_weights = running_model_avg(avg_weights, client_weights, 1/num_selected_clients)
            running_loss += client_loss
            min_scale = min(min_scale, client_scale)

        train_loss = running_loss / num_selected_clients

        print(f'Train Round: {round} Loss: {train_loss:.6f} Lr: {lr:e}')

        # Server
        model_dict = avg_weights

        if round % validation_interval == 0  or round == num_rounds:
            val_loss, val_acc = validate(model, val_loader, criterion)
            logger.log(round, train_loss, val_loss, val_acc, lr)

        # update learning rate
        if round % run['rounds_per_scheduler_step'] == 0:
            scheduler.step()
            lr = scheduler.optimizer.param_groups[0]["lr"]

        # update scale
        grad_scale = min_scale
        if round % run['scale_grow_interval'] == 0:
            grad_scale *= 2

        if round % validation_interval == 0 or round == num_rounds:
            manager.save(round, model, grad_scale, scheduler, logger, val_acc)


def build_training_objects(run):
    # Import data
    train_loaders, val_loader = create_dataloader_federated(run['batch_size'], run['num_clients'], run['num_classes_per_client'])

    # Define model
    model = Dino_vits16_100().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # Create clients
    clients = [Client(model, criterion, dataloader) for dataloader in train_loaders]

    # Scheduler
    warmup_steps = run['warmup_steps']
    cosine_steps = run['cosine_steps']
    dummy_optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=run['max_learning_rate'])
    warmup_sched = LinearLR(dummy_optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(dummy_optimizer, T_max=cosine_steps)
    constant_sched = MultiplicativeLR(dummy_optimizer, lr_lambda=lambda epoch: 1.0)

    scheduler = SequentialLR(
        dummy_optimizer,
        schedulers=[warmup_sched, cosine_sched, constant_sched],
        milestones=[warmup_steps, warmup_steps + cosine_steps]
    )

    return clients, val_loader, model, criterion, scheduler


def resume(run_name, total_rounds, separate=False):
    checkpoints_dir = 'federated_model/checkpoints/'
    logs_dir = 'federated_model/logs/'
    plots_dir = 'federated_model/plots/'

    # Cleanup, restoring files and checkpoint loading
    manager = CheckpointManagerFederated(checkpoints_dir, run_name)
    logger_state_dict, round = manager.resume()

    logger = Logger(logs_dir, run_name)
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
    manager.restore_state(model, clients, scheduler)

    # Run the training process for {num_epochs} epochs
    print(f'Run name: {run['name']}')
    print('Resume training')
    train_federated(total_rounds, run, model, clients, val_loader, criterion, scheduler, logger, manager, run['validation_interval'], round + 1)
    plot_training(run_name, logs_dir, plots_dir)
    return logger.get_run()

def start(num_rounds, num_steps_per_client, num_classes_per_client, rounds_per_scheduler_step, warmup_steps, cosine_steps, scale_grow_interval, validation_interval, batch_size, max_lr, decay_rate, weight_decay):
    num_clients = 100
    client_ratio = 0.1

    logs_dir = 'federated_model/logs'
    checkpoints_dir = 'federated_model/checkpoints/'
    plots_dir = 'federated_model/plots/'

    # Init checkpoint manager and logger
    run = {
        'name': datetime.now().strftime('%Y%m%d_%H%M%S'),
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
        'warmup_steps': warmup_steps,
        'cosine_steps': cosine_steps,
        'scale_grow_interval': scale_grow_interval,
        'validation_interval': validation_interval,
        'total_rounds': 0,
        'best_round': -1,
        'best_accuracy': -1.0,
    }
    manager = CheckpointManagerFederated(checkpoints_dir, run['name'])
    logger = Logger(logs_dir, run['name'], step_name="round", total_key="total_rounds", best_key="best_round")
    logger.start(run)

    print('Using device:', DEVICE)

    clients, val_loader, model, criterion, scheduler = build_training_objects(run)

    # Run the training process for {num_rounds} rounds
    print(f'Run name: {run['name']}')
    print('Start training')
    train_federated(num_rounds, run, model, clients, val_loader, criterion, scheduler, logger, manager, validation_interval)
    plot_training(run['name'], logs_dir, plots_dir)
    return logger.get_run()

if __name__ == '__main__':
    start(num_rounds = 10,
          num_steps_per_client = 4,
          num_classes_per_client = 10,
          rounds_per_scheduler_step = 1,
          warmup_steps = 3,
          cosine_steps = 17,
          scale_grow_interval= 5,
          validation_interval= 5,
          batch_size = 64,
          max_lr = 0.01,
          decay_rate = 0.75,
          weight_decay = 1e-4)
