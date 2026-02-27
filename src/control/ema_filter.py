"""Dual-rate EMA filtering for constraint violation signals (§4.4)."""

from __future__ import annotations

import torch
from torch import Tensor


class DualRateEMA:
    """Dual-rate EMA filter for constraint violations.

    Two timescales:
    - Fast (β=0.9, ~10-step smoothing): feeds ESO and proportional term.
    - Slow (β=0.99, ~100-step smoothing): feeds integral accumulator.

    Timescale separation ratio r = (1-β_fast)/(1-β_slow) = 10 satisfies
    the minimum r ≥ 10 for two-timescale stochastic approximation convergence.
    """

    def __init__(
        self,
        n_constraints: int = 3,
        beta_fast: float = 0.9,
        beta_slow: float = 0.99,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        self.n_constraints = n_constraints
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        self._v_fast = torch.zeros(n_constraints, device=device)
        self._v_slow = torch.zeros(n_constraints, device=device)
        self._v_fast_prev = torch.zeros(n_constraints, device=device)
        self._initialized = False

    def update(self, violations: Tensor) -> None:
        """Update both EMA channels with raw constraint violations.

        Args:
            violations: [n_constraints] raw per-batch violations (g_i(θ)).
        """
        v = violations.detach()

        if not self._initialized:
            self._v_fast = v.clone()
            self._v_slow = v.clone()
            self._v_fast_prev = v.clone()
            self._initialized = True
            return

        self._v_fast_prev = self._v_fast.clone()
        self._v_fast = self.beta_fast * self._v_fast + (1 - self.beta_fast) * v
        self._v_slow = self.beta_slow * self._v_slow + (1 - self.beta_slow) * v

    @property
    def v_fast(self) -> Tensor:
        """Fast EMA of violations [n_constraints]."""
        return self._v_fast

    @property
    def v_slow(self) -> Tensor:
        """Slow EMA of violations [n_constraints]."""
        return self._v_slow

    @property
    def v_fast_prev(self) -> Tensor:
        """Previous step's fast EMA [n_constraints]."""
        return self._v_fast_prev

    def state_dict(self) -> dict:
        return {
            "v_fast": self._v_fast,
            "v_slow": self._v_slow,
            "v_fast_prev": self._v_fast_prev,
            "initialized": self._initialized,
        }

