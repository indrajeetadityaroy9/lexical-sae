"""Standard AL dual variable updater: λ = clamp(λ + ρ·v_slow, 0).

O(1/k) convergence guarantee (Rockafellar 1976).
"""

import torch
from torch import Tensor

device = torch.device("cuda")


class DualUpdater:
    """λ = clamp(λ + ρ·v_slow, 0). Called at slow_update_interval only."""

    def __init__(self, n_constraints: int = 3) -> None:
        self.n_constraints = n_constraints
        self._lambdas = torch.zeros(n_constraints, device=device)

    def step(self, v_slow: Tensor, rhos: Tensor) -> None:
        """Standard AL dual update using slow EMA violations and CAPU penalties."""
        self._lambdas = torch.clamp(self._lambdas + rhos * v_slow, min=0.0)

    @property
    def lambdas(self) -> Tensor:
        """Current dual variables [n_constraints]."""
        return self._lambdas

    def state_dict(self) -> dict:
        return {"lambdas": self._lambdas}

    def load_state_dict(self, sd: dict) -> None:
        self._lambdas = sd["lambdas"].to(device)
