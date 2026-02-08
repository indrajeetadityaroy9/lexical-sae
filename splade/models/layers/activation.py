import torch
import torch.nn as nn


class DReLU(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x - self.theta)
