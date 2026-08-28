import random
from enum import Enum
import torch
from torch.func import functional_call, grad, vmap
from tqdm import tqdm

from data.dataloader import DEVICE

class MaskRule(Enum):
    LEAST_SENSITIVE = 0
    MOST_SENSITIVE = 1
    LOWEST_MAGNITUDE = 2
    HIGHEST_MAGNITUDE = 3
    RANDOM = 4


class ModelMasker:
    def __init__(self, model):
        self.original_parameters = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def mask(self, model, mask, min_value=0.1):
        for name, param in model.named_parameters():
            if name in mask:
                soft_mask = mask[name].to(dtype=param.dtype).clamp_min(min_value)
                param.copy_(self.original_parameters[name] * soft_mask)

    @torch.no_grad()
    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self.original_parameters:
                param.copy_(self.original_parameters[name])


@torch.no_grad()
def update_masks(masks, scores, density, mask_rule):
    # Note: mask_rule only affects printing behavior

    global_scores = torch.cat([score.flatten() for score in scores.values()])
    global_masks = torch.cat([masks[name].flatten() for name in scores])

    active_scores = global_scores[global_masks == 1.0]
    t, _ = torch.kthvalue(
        active_scores,
        max(
            1,
            min(
                int(density * global_scores.numel()),
                active_scores.numel() - 1
            )
        )
    )

    print('[+] Threshold:', t.item())
    tot, not_masked = 0, 0

    for name, mask in masks.items():
        if name in scores:
            score = scores[name]
            score = score.masked_fill(mask != 1.0, torch.inf)

            mask.copy_((score <= t).to(dtype=mask.dtype))

            tot += mask.numel()
            not_masked += int(mask.sum().item())

    if mask_rule == MaskRule.LEAST_SENSITIVE:
        masked = tot - not_masked
    else: # Most sensitive
        masked = not_masked
    print(f'[+] Masked weights: {masked / tot} ({masked} / {tot})')


def running_sum(current, next_scores):
    if current is None:
        current = next_scores
    else:
        for name in current:
            current[name].add_(next_scores[name])
    return current


def keep_least_sensitive(model, clients, client_ratio, sparsity, num_calibration_round, mask_rule):
    # Note: mask_rule only affects printing behavior

    density = 1.0 - sparsity

    masker = ModelMasker(model)

    masks = {
        name: torch.ones_like(param)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    if density < 1.0:
        for round in range(1, num_calibration_round + 1):
            print(f'Calibration round: {round}')
            round_density = density ** (round / num_calibration_round)
            if mask_rule == MaskRule.LEAST_SENSITIVE:
                print('[+] Target sparsity:', 1 - round_density)
            else: # Most sensitive
                print('[+] Target sparsity:', round_density)

            total_scores = None
            num_clients = len(clients)
            num_selected_clients = max(int(num_clients * client_ratio), 1)
            progress_bar = tqdm(random.sample(clients, num_selected_clients), f'Calibration round {round}', leave=False)
            for client in progress_bar:
                scores = compute_local_sensitivity(model, client.train_loader)
                total_scores = running_sum(total_scores, scores)

            update_masks(masks, total_scores, round_density, mask_rule)
            masker.mask(model, masks)

    masker.restore(model)

    del masker
    torch.cuda.empty_cache()
    return masks


def compute_local_sensitivity(model, loader, num_batches=2, microbatch_size=16):
    model.to(DEVICE)
    model.eval()

    # Parameters and buffers used by functional_call
    params = {
        name: param
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    buffers = dict(model.named_buffers())

    sensitivity = {
        name: torch.zeros_like(param)
        for name, param in params.items()
    }

    # Compute the gradient of one selected logit for one sample
    def single_sample_logit(params, buffers, input, cls):
        logits = functional_call(
            model,
            (params, buffers),
            (input.unsqueeze(0))
        )
        return logits.gather(1, cls.reshape(1, 1)).squeeze()

    grad_single = grad(single_sample_logit)

    # Vectorize over samples
    grad_batch = vmap(grad_single, in_dims=(None, None, 0, 0))

    for batch_idx, (inputs, _) in enumerate(loader):
        if num_batches > 0 and batch_idx >= num_batches:
            break

        inputs = inputs.to(DEVICE, non_blocking=True)

        with torch.no_grad():
            logits = model(inputs)
            classes = torch.distributions.Categorical(logits=logits).sample()

        batch_size = inputs.shape[0]

        for start in range(0, batch_size, microbatch_size):
            end = min(start + microbatch_size, batch_size)

            mb_inputs = inputs[start:end]
            mb_classes = classes[start:end]

            grads = grad_batch(params, buffers, mb_inputs, mb_classes)

            for name in sensitivity:
                sensitivity[name].add_(grads[name].detach().square().sum(dim=0))

    return sensitivity


def calibrate_federated_mask(model, clients, client_ratio, sparsity, num_calibration_round, mask_rule):
    match mask_rule:
        case MaskRule.LEAST_SENSITIVE:
            mask = keep_least_sensitive(model, clients, client_ratio, sparsity, num_calibration_round, mask_rule)
        case MaskRule.MOST_SENSITIVE:
            mask = keep_least_sensitive(model, clients, client_ratio, 1 - sparsity, num_calibration_round, mask_rule)
            # Invert mask
            mask = {
                k: torch.ones_like(v) - v
                for k, v in mask.items()
            }
        case MaskRule.LOWEST_MAGNITUDE:
            mask = None
        case MaskRule.HIGHEST_MAGNITUDE:
            mask = None
        case MaskRule.RANDOM:
            mask = None
        case _:
            raise ValueError(f"Invalid mask rule: {mask_rule}")

    # Disable grad on model parameters whose corresponding mask is entirely zero
    for name, param in model.named_parameters():
        if name in mask and torch.count_nonzero(mask[name]) == 0:
            param.requires_grad_(False)

    return mask