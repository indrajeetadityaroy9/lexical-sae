"""Dual-rate EMA filtering for constraint violation signals (§4.4)."""

from __future__ import annotations

import torch
from torch import Tensor

from src.runtime import DEVICE


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

        self._v_fast_raw = torch.zeros(n_constraints, device=DEVICE)
        self._v_slow_raw = torch.zeros(n_constraints, device=DEVICE)
        self._v_fast_prev = torch.zeros(n_constraints, device=DEVICE)
        self._n_updates = 0

    def update(self, violations: Tensor) -> None:
        """Update both EMA channels with raw constraint violations.

        Args:
            violations: [n_constraints] raw per-batch violations (g_i(θ)).
        """
        v = violations.detach()

        # Store bias-corrected current as prev BEFORE updating
        if self._n_updates > 0:
            self._v_fast_prev = self.v_fast.clone()

        self._n_updates += 1
        self._v_fast_raw = self.beta_fast * self._v_fast_raw + (1 - self.beta_fast) * v
        self._v_slow_raw = self.beta_slow * self._v_slow_raw + (1 - self.beta_slow) * v

        # On first update, set prev to current (no delta)
        if self._n_updates == 1:
            self._v_fast_prev = self.v_fast.clone()

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

    @property
    def v_fast_prev(self) -> Tensor:
        """Previous step's bias-corrected fast EMA [n_constraints]."""
        return self._v_fast_prev

    def state_dict(self) -> dict:
        return {
            "v_fast_raw": self._v_fast_raw,
            "v_slow_raw": self._v_slow_raw,
            "v_fast_prev": self._v_fast_prev,
            "n_updates": self._n_updates,
        }
