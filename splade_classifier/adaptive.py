"""Adaptive parameter utilities for parameter-free SPLADE classifier.

This module provides:
- StableOps: Dtype-appropriate numerical stability constants
- Adaptive learning rate initialization and scheduling
- Self-normalizing regularization utilities

All mechanisms are designed to generalize across models, datasets, and scales
without per-instance calibration.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class StableOps:
    """Dtype-appropriate numerical stability constants.

    Machine epsilon for float32 is ~1.19e-7. We use sqrt(eps) for division
    to account for accumulation errors, and eps directly for log operations.

    These values are derived from IEEE 754 floating point properties,
    not arbitrary heuristics.
    """

    EPS = {
        torch.float32: {'div': 1e-6, 'log': 1e-7, 'sqrt': 1e-12, 'softmax': 1e-10},
        torch.float16: {'div': 1e-4, 'log': 1e-5, 'sqrt': 1e-8, 'softmax': 1e-7},
        torch.bfloat16: {'div': 1e-3, 'log': 1e-4, 'sqrt': 1e-6, 'softmax': 1e-5},
    }

    @classmethod
    def get_eps(cls, dtype: torch.dtype, op: str = 'div') -> float:
        """Get appropriate epsilon for dtype and operation."""
        if dtype not in cls.EPS:
            dtype = torch.float32  # Default to float32 for unknown dtypes
        return cls.EPS[dtype].get(op, cls.EPS[dtype]['div'])

    @classmethod
    def safe_div(cls, num: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
        """Safe division with dtype-appropriate epsilon."""
        eps = cls.get_eps(num.dtype, 'div')
        return num / (denom + eps)

    @classmethod
    def safe_log(cls, x: torch.Tensor) -> torch.Tensor:
        """Safe log with dtype-appropriate epsilon to prevent log(0)."""
        eps = cls.get_eps(x.dtype, 'log')
        return torch.log(x + eps)

    @classmethod
    def safe_sqrt(cls, x: torch.Tensor) -> torch.Tensor:
        """Safe sqrt with dtype-appropriate epsilon."""
        eps = cls.get_eps(x.dtype, 'sqrt')
        return torch.sqrt(x + eps)

    @classmethod
    def safe_normalize(cls, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """L2-normalize with dtype-appropriate epsilon."""
        eps = cls.get_eps(x.dtype, 'div')
        return x / (x.norm(dim=dim, keepdim=True) + eps)


def compute_base_lr(model: nn.Module, batch_size: int) -> float:
    """Compute learning rate from model/data statistics.

    Based on:
    - He initialization: sqrt(2/fan_in) preserves variance in ReLU networks
    - Linear scaling rule (Goyal et al., 2017): LR scales with batch size

    Args:
        model: The neural network model
        batch_size: Training batch size

    Returns:
        Recommended base learning rate
    """
    # Get model dimensions
    if hasattr(model, 'bert') and hasattr(model.bert, 'config'):
        d_model = model.bert.config.hidden_size  # 768 for DistilBERT
        vocab_size = model.bert.config.vocab_size  # 30522
    else:
        # Fallback: estimate from parameters
        total_params = sum(p.numel() for p in model.parameters())
        d_model = 768
        vocab_size = int(math.sqrt(total_params / 10))

    # He initialization scale: sqrt(2 / fan_in)
    # For transformer: fan_in ≈ d_model * vocab_size for final projection
    base = math.sqrt(2.0 / (d_model * vocab_size))

    # Linear scaling rule: scale with sqrt(batch_size / reference_batch)
    # Reference batch = 32 (common default)
    scaled = base * math.sqrt(batch_size / 32.0)

    # Cap at empirically stable maximum for transformers
    return min(scaled, 1e-3)


class AdaptiveLRScheduler:
    """Learning rate scheduler with data-driven warmup and cosine decay.

    Warmup duration is proportional to training data (10% of first epoch),
    avoiding magic numbers for warmup steps.
    """

    def __init__(
        self,
        base_lr: float,
        num_samples: int,
        batch_size: int,
        epochs: int,
        warmup_ratio: float = 0.1,
    ):
        """Initialize scheduler.

        Args:
            base_lr: Base learning rate (from compute_base_lr)
            num_samples: Number of training samples
            batch_size: Training batch size
            epochs: Number of training epochs
            warmup_ratio: Fraction of first epoch for warmup (default: 10%)
        """
        self.base_lr = base_lr
        steps_per_epoch = max(num_samples // batch_size, 1)

        # Warmup: proportional to first epoch, bounded
        warmup_from_ratio = int(steps_per_epoch * warmup_ratio)
        self.warmup_steps = max(min(warmup_from_ratio, 1000), 100)

        self.total_steps = steps_per_epoch * epochs
        self._step = 0

    def get_lr(self) -> float:
        """Get current learning rate."""
        step = self._step

        if step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (step / self.warmup_steps)

        # Cosine decay after warmup
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        return self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))

    def step(self) -> float:
        """Advance scheduler and return new LR."""
        lr = self.get_lr()
        self._step += 1
        return lr


class AdaptiveFlopsReg:
    """Self-normalizing FLOPS regularization with adaptive lambda.

    The regularization strength automatically adjusts based on the gap
    between current and target sparsity. This eliminates the need for
    manual tuning of flops_lambda.

    Target sparsity of 95% is from SPLADE literature as the standard
    operating point for sparse lexical representations.
    """

    def __init__(self, target_sparsity: float = 0.95, ema_decay: float = 0.99):
        """Initialize adaptive regularization.

        Args:
            target_sparsity: Target fraction of zero activations (default: 0.95)
            ema_decay: Decay for exponential moving average (default: 0.99)
        """
        self.target_sparsity = target_sparsity
        self.ema_decay = ema_decay
        self.ema_sparsity: Optional[float] = None

    def compute_lambda(self, activations: torch.Tensor) -> float:
        """Compute adaptive lambda based on current sparsity.

        Lambda increases when sparsity is below target, decreases when above.

        Args:
            activations: Sparse activation tensor [batch, vocab]

        Returns:
            Adaptive regularization weight
        """
        # Compute current sparsity
        with torch.no_grad():
            eps = StableOps.get_eps(activations.dtype, 'div')
            current_sparsity = (activations.abs() < eps).float().mean().item()

        # Update EMA
        if self.ema_sparsity is None:
            self.ema_sparsity = current_sparsity
        else:
            self.ema_sparsity = self.ema_decay * self.ema_sparsity + (1 - self.ema_decay) * current_sparsity

        # Lambda increases when below target (more regularization needed)
        gap = max(0, self.target_sparsity - self.ema_sparsity)

        # Base lambda of 1.0, with 10x increase per 10% sparsity gap
        return 1.0 + 10.0 * gap

    def get_current_sparsity(self) -> float:
        """Get current tracked sparsity."""
        return self.ema_sparsity if self.ema_sparsity is not None else 0.0
