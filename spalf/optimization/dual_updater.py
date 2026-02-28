"""nuPI dual controller: λ = clamp(λ + ρ·e + 0.5·ρ·(e - e_prev), 0).

Proportional term damps oscillations around constraint boundaries.
κ_p = 0.5 is the critically-damped gain for a PI controller sampling
once per EMA time constant (structural, not tunable).
O(1/k) convergence preserved (P-term vanishes asymptotically).
"""

import torch
from torch import Tensor

device = torch.device("cuda")

# Critically-damped proportional gain. Derived: slow_update_interval and
# EMA time constant both equal 1/(1-β_slow), so sampling-to-timescale
# ratio is 1:1. Tustin bilinear with ζ=1 gives κ_p = 1/(2·1) = 0.5.
_KAPPA_P: float = 0.5


class NuPIDualUpdater:
    """PI dual variable controller. Called at slow_update_interval only."""

    def __init__(self, n_constraints: int = 3) -> None:
        self.n_constraints = n_constraints
        self._lambdas = torch.zeros(n_constraints, device=device)
        self._e_prev = torch.zeros(n_constraints, device=device)

    def step(self, v_slow: Tensor, rhos: Tensor) -> None:
        """PI dual update using slow EMA violations and CAPU penalties."""
        e = v_slow.detach()
        i_term = rhos * e
        p_term = _KAPPA_P * rhos * (e - self._e_prev)
        self._lambdas = torch.clamp(self._lambdas + i_term + p_term, min=0.0)
        self._e_prev = e.clone()

    @property
    def lambdas(self) -> Tensor:
        """Current dual variables [n_constraints]."""
        return self._lambdas

    def state_dict(self) -> dict:
        return {"lambdas": self._lambdas, "e_prev": self._e_prev}

    def load_state_dict(self, sd: dict) -> None:
        self._lambdas = sd["lambdas"].to(device)
        self._e_prev = sd["e_prev"].to(device)
