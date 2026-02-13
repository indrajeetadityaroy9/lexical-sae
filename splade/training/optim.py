import math

import torch
from schedulefree import AdamWScheduleFree
from transformers import AutoConfig

from splade.utils.cuda import unwrap_compiled


def build_optimizer(
    model: torch.nn.Module,
    base_lr: float = 3e-4,
) -> AdamWScheduleFree:
    """Build Schedule-Free AdamW optimizer (Defazio & Mishchenko, arXiv:2405.15682).

    Subsumes LR scheduling, warmup, and model EMA into a single optimizer via
    Primal Averaging. Call optimizer.train() during training steps and
    optimizer.eval() before validation/inference to switch to averaged params.

    Args:
        model: The model to optimize.
        base_lr: Base learning rate (default 3e-4, the standard AdamW default).
    """
    param_groups = _build_param_groups(model, base_lr)
    return AdamWScheduleFree(param_groups, warmup_steps=0)


def _infer_batch_size(model_name: str, max_length: int) -> int:
    config = AutoConfig.from_pretrained(model_name)
    mem_ratio = (768 * 128) / (config.hidden_size * max_length)
    # Base batch size for ~24GB reference GPU (22GB usable)
    base_power = min(6, max(3, int(math.log2(max(1, 32 * mem_ratio)))))
    base_bs = 2 ** base_power
    # Scale by available GPU memory relative to 22GB usable reference
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    usable_mem = max(1.0, total_mem - 2.0)
    scale_factor = usable_mem / 22.0
    scaled_bs = int(base_bs * scale_factor)
    # Round down to power of 2, cap at 512
    power = min(9, int(math.log2(max(1, scaled_bs))))
    return 2 ** power


def _build_param_groups(model: torch.nn.Module, base_lr: float) -> list[dict]:
    """Build param groups with derived weight decay.

    Weight decay is derived as 0.1 * base_lr (Loshchilov-Hutter scaling),
    eliminating the need for a separate WEIGHT_DECAY constant.
    """
    weight_decay = 0.1 * base_lr
    _orig = unwrap_compiled(model)
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for name, param in _orig.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "LayerNorm" in name or "layer_norm" in name or "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "lr": base_lr, "weight_decay": weight_decay},
        {"params": no_decay_params, "lr": base_lr, "weight_decay": 0.0},
    ]


def _gradient_centralization(
    model: torch.nn.Module,
    skip_params: set | None = None,
) -> None:
    """Parameter-free gradient conditioning via centralization.

    Subtracts the mean from each gradient tensor (for weight matrices only),
    constraining gradients to the hyperplane of zero-mean vectors. This
    improves the Lipschitz smoothness of the loss landscape without any
    tunable hyperparameters.

    Reference: Yong et al., "Gradient Centralization" (arXiv:2004.01461).
    """
    skip_ids = frozenset(id(p) for p in skip_params) if skip_params else frozenset()
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if id(parameter) in skip_ids:
            continue
        if parameter.data.ndim >= 2:
            # Centralize: subtract mean across all dims except the output dim
            parameter.grad.data -= parameter.grad.data.mean(
                dim=tuple(range(1, parameter.grad.data.ndim)),
                keepdim=True,
            )
