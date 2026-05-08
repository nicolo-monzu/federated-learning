import torch
from torch.optim import Optimizer


class SparseSGDM(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.0, dampening=0.0,
                 weight_decay=0.0, nesterov=False):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
        super().__init__(params, defaults)

    def step(self, masks=None):

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            dampening = group['dampening']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']

            params = group['params']

            if masks is None:
                masks_iter = [None] * len(params)
            else:
                if len(masks) != len(params):
                    raise ValueError("Numero di mask diverso dai parametri")
                masks_iter = masks

            for p, mask in zip(params, masks_iter):

                if p.grad is None:
                    continue

                grad = p.grad.data

                if mask is not None:
                    grad = grad * mask

                if weight_decay != 0:
                    grad = grad + weight_decay * p.data

                if momentum != 0:
                    state = self.state[p]

                    if 'momentum_buffer' not in state:
                        buf = state['momentum_buffer'] = grad.clone().detach()
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(grad, alpha=1 - dampening)

                    if nesterov:
                        grad = grad + momentum * buf
                    else:
                        grad = buf

                p.data.add_(grad, alpha=-lr)

        return None