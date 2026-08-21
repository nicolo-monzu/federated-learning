import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT


class SparseSGDM(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        momentum: float = 0,
        weight_decay: float = 0,
        masks: dict[str, Tensor] | None = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay
        }

        super().__init__(params, defaults)

        self.masks = masks if masks is not None else {}

    def _init_group(self, group, params, grads, masks, momentum_buffer_list):
        for p in group["params"]:
            if p.grad is not None:
                params.append(p)
                grads.append(p.grad)

                mask = self.masks.get(p)
                masks.append(mask)

                if group["momentum"] != 0:
                    state = self.state[p]
                    momentum_buffer_list.append(state.get("momentum_buffer"))


    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []
            masks: list[Tensor | None] = []
            momentum_buffer_list: list[Tensor | None] = []

            self._init_group(group, params, grads, masks, momentum_buffer_list)

            _sparse_sgdm(
                params,
                grads,
                masks,
                momentum_buffer_list,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
            )

            if group["momentum"] != 0:
                # update momentum_buffers in state
                for p, momentum_buffer in zip(
                    params, momentum_buffer_list, strict=True
                ):
                    state = self.state[p]
                    state["momentum_buffer"] = momentum_buffer

        return loss


def _sparse_sgdm(
    params: list[Tensor],
    grads: list[Tensor],
    masks: list[Tensor | None],
    momentum_buffer_list: list[Tensor | None],
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
) -> None:
    for i, param in enumerate(params):
        grad = grads[i]
        mask = masks[i]

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = grad.detach().clone()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(grad)

            # Mask application
            if mask is not None:
                buf.mul_(mask)
            grad = buf

        # Mask application
        elif mask is not None:
            grad = grad * mask

        # Weight update
        param.add_(grad, alpha=-lr)