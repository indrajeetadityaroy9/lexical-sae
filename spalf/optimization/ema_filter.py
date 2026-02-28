"""Dual-rate EMA filtering for constraint violation signals."""

import torch
from torch import Tensor

device = torch.device("cuda")


class DualRateEMA:
    """Dual-rate EMA filter with Adam-style bias correction for constraint violations."""

    def __init__(
        self,
        n_constraints: int = 3,
        beta_fast: float = 0.9,
        beta_slow: float = 0.99,
    ) -> None:
        self.n_constraints = n_constraints
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        self._v_fast_raw = torch.zeros(n_constraints, device=device)
        self._v_slow_raw = torch.zeros(n_constraints, device=device)
        self._n_updates = 0

    def update(self, violations: Tensor) -> None:
        """Update both EMA channels with raw constraint violations.

        Args:
            violations: [n_constraints] raw per-batch violations (g_i(θ)).
        """
        v = violations.detach()
        self._n_updates += 1
        self._v_fast_raw = self.beta_fast * self._v_fast_raw + (1 - self.beta_fast) * v
        self._v_slow_raw = self.beta_slow * self._v_slow_raw + (1 - self.beta_slow) * v

    @property
    def v_fast(self) -> Tensor:
        """Bias-corrected fast EMA of violations [n_constraints]."""
        if self._n_updates == 0:
            return self._v_fast_raw
        return self._v_fast_raw / (1 - self.beta_fast ** self._n_updates)

    @property
    def v_slow(self) -> Tensor:
        """Bias-corrected slow EMA of violations [n_constraints]."""
        if self._n_updates == 0:
            return self._v_slow_raw
        return self._v_slow_raw / (1 - self.beta_slow ** self._n_updates)

    def state_dict(self) -> dict:
        return {
            "v_fast_raw": self._v_fast_raw,
            "v_slow_raw": self._v_slow_raw,
            "n_updates": torch.tensor(self._n_updates, device=device),
        }

    def load_state_dict(self, sd: dict) -> None:
        self._v_fast_raw = sd["v_fast_raw"].to(device)
        self._v_slow_raw = sd["v_slow_raw"].to(device)
        self._n_updates = int(sd["n_updates"].item())
