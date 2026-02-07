"""Activation functions for SPLADE sparse representations."""

import torch
import torch.nn as nn

class DReLU(nn.Module):
    """Shifted ReLU with learnable threshold for hard sparsity.

    f(x) = max(0, x - theta) where theta is a learnable per-dimension parameter.
    Promotes exact zeros in sparse representations by raising the activation floor.
    Note: This is distinct from Turbo Sparse's DReLU (arXiv:2406.05955) which uses
    a gated dual-ReLU for decoder LLMs.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assuming x is [Batch, Seq, Dim] or [Batch, Dim]
        # Broadcasting theta across batch/seq
        return torch.relu(x - self.theta)