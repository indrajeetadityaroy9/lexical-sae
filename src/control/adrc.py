"""ADRC multiplier update with PI + ESO cancellation (§4.2–§4.3).

Per-constraint observer gains (replaces scalar omega_o).
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.runtime import DEVICE


class ExtendedStateObserver:
    """Extended state observer with per-constraint gains for disturbance estimation."""

    def __init__(
        self,
        n_constraints: int = 3,
        omega_o: float = 0.3,
    ) -> None:
        self.n_constraints = n_constraints
        self._omega_o = torch.full((n_constraints,), omega_o, device=DEVICE)

        self._xi = torch.zeros(n_constraints, device=DEVICE)
        self._f_hat = torch.zeros(n_constraints, device=DEVICE)

    def step(self, v_fast: Tensor, lambdas: Tensor) -> Tensor:
        """Compute disturbance estimate and advance observer state."""
        omega = self._omega_o

        self._f_hat = self._xi + omega * v_fast

        self._xi = (
            (1 - omega) * self._xi
            - omega.pow(2) * v_fast
            - omega * lambdas
        )

        return self._f_hat

    def set_omega(self, omega_o: Tensor) -> None:
        """Update per-constraint observer gains."""
        self._omega_o = omega_o

    @property
    def omega_o(self) -> Tensor:
        return self._omega_o

    def state_dict(self) -> dict:
        return {
            "xi": self._xi,
            "f_hat": self._f_hat,
            "omega_o": self._omega_o,
        }


class ADRCController:
    """ADRC update for non-negative dual variables with per-constraint gains."""

    def __init__(
        self,
        n_constraints: int = 3,
        omega_o_init: float = 0.3,
        beta_fast: float = 0.9,
    ) -> None:
        self.n_constraints = n_constraints
        self._omega_o = torch.full((n_constraints,), omega_o_init, device=DEVICE)
        self._k_ap = 2.0 * self._omega_o
        self._k_i = self._omega_o.pow(2)

        self._lambdas = torch.zeros(n_constraints, device=DEVICE)
        self.eso = ExtendedStateObserver(n_constraints, omega_o_init)

        self._L_hat_ema = torch.zeros(n_constraints, device=DEVICE)
        self._L_hat_ema_beta = beta_fast

    def step(self, v_fast: Tensor, v_fast_prev: Tensor, v_slow: Tensor) -> None:
        """Run one ADRC dual update."""
        f_hat = self.eso.step(v_fast.detach(), self._lambdas)

        proportional = self._k_ap * (v_fast.detach() - v_fast_prev.detach())
        integral = self._k_i * v_slow.detach()
        u = proportional + integral - f_hat

        self._lambdas = torch.clamp(self._lambdas + u, min=0.0)

    def update_omega(self, v_fast: Tensor, v_fast_prev: Tensor) -> None:
        """Update per-constraint observer gains from online Lipschitz proxy."""
        L_instant = (v_fast.detach() - v_fast_prev.detach()).abs()

        self._L_hat_ema = (
            self._L_hat_ema_beta * self._L_hat_ema
            + (1 - self._L_hat_ema_beta) * L_instant
        )

        # Per-constraint gain update (elementwise clamp)
        new_omega = self._L_hat_ema.clamp(0.3, 1.0)
        self._omega_o = new_omega
        self._k_ap = 2.0 * new_omega
        self._k_i = new_omega.pow(2)

        self.eso.set_omega(new_omega)

    @property
    def lambdas(self) -> Tensor:
        """Current dual variables [n_constraints]."""
        return self._lambdas

    @property
    def omega_o(self) -> Tensor:
        """Per-constraint observer gains [n_constraints]."""
        return self._omega_o

    def state_dict(self) -> dict:
        return {
            "lambdas": self._lambdas,
            "omega_o": self._omega_o,
            "L_hat_ema": self._L_hat_ema,
            "eso": self.eso.state_dict(),
        }
