"""Modified CAPU: non-monotone per-constraint adaptive penalty (§5.2)."""

from __future__ import annotations

import torch
from torch import Tensor


class ModifiedCAPU:
    """Non-monotone per-constraint adaptive penalty controller."""

    def __init__(
        self,
        initial_violations: Tensor,
        c_eta: float = 1.0,
        rho_0: float = 1.0,
        beta_slow: float = 0.99,
        eps_num: float = 1e-8,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        self.n_constraints = initial_violations.shape[0]
        self.beta_slow = beta_slow
        self.eps_num = eps_num
        self.rho_min = 0.1 * rho_0
        self._frozen = False

        self._etas = (
            c_eta / (initial_violations.abs().to(device) + eps_num).sqrt()
        )

        self._v_bar = torch.ones(self.n_constraints, device=device)

        self._rhos = torch.full((self.n_constraints,), rho_0, device=device)

    def step(self, v_fast: Tensor) -> None:
        """Update penalty coefficients from fast EMA violations."""
        if self._frozen:
            return

        self._v_bar = (
            self.beta_slow * self._v_bar
            + (1 - self.beta_slow) * v_fast.detach() ** 2
        )

        target = self._etas / (self._v_bar + self.eps_num).sqrt()
        self._rhos = torch.clamp(target, min=self.rho_min)

    def freeze(self) -> None:
        """Freeze penalty updates."""
        self._frozen = True

    @property
    def rhos(self) -> Tensor:
        """Current penalty coefficients [n_constraints]."""
        return self._rhos

    def state_dict(self) -> dict:
        return {
            "etas": self._etas,
            "v_bar": self._v_bar,
            "rhos": self._rhos,
            "frozen": self._frozen,
            "rho_min": self.rho_min,
        }
