"""ADRC multiplier update with PI + ESO cancellation (§4.2–§4.3)."""

from __future__ import annotations

import torch
from torch import Tensor


class ExtendedStateObserver:
    """Extended state observer for disturbance estimation in dual dynamics."""

    def __init__(
        self,
        n_constraints: int = 3,
        omega_o: float = 0.3,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        self.n_constraints = n_constraints
        self._omega_o = omega_o

        self._xi = torch.zeros(n_constraints, device=device)
        self._f_hat = torch.zeros(n_constraints, device=device)

    def step(self, v_fast: Tensor, lambdas: Tensor) -> Tensor:
        """Compute disturbance estimate and advance observer state."""
        omega = self._omega_o

        self._f_hat = self._xi + omega * v_fast

        self._xi = (
            (1 - omega) * self._xi
            - omega**2 * v_fast
            - omega * lambdas
        )

        return self._f_hat.clone()

    def set_omega(self, omega_o: float) -> None:
        """Update observer gain. Must satisfy ω_o ∈ (0, 1) for discrete stability."""
        self._omega_o = max(0.01, min(omega_o, 0.99))

    def state_dict(self) -> dict:
        return {
            "xi": self._xi,
            "f_hat": self._f_hat,
            "omega_o": self._omega_o,
        }



class ADRCController:
    """ADRC update for non-negative dual variables."""

    def __init__(
        self,
        n_constraints: int = 3,
        omega_o_init: float = 0.3,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        self.n_constraints = n_constraints
        self._omega_o = omega_o_init
        self._k_ap = 2.0 * omega_o_init
        self._k_i = omega_o_init**2

        self._lambdas = torch.zeros(n_constraints, device=device)
        self.eso = ExtendedStateObserver(n_constraints, omega_o_init, device)

        self._L_hat_ema = torch.zeros(n_constraints, device=device)
        self._L_hat_ema_beta = 0.9

    def step(self, v_fast: Tensor, v_fast_prev: Tensor, v_slow: Tensor) -> None:
        """Run one ADRC dual update."""
        f_hat = self.eso.step(v_fast.detach(), self._lambdas)

        proportional = self._k_ap * (v_fast.detach() - v_fast_prev.detach())
        integral = self._k_i * v_slow.detach()
        u = proportional + integral - f_hat

        self._lambdas = torch.clamp(self._lambdas + u, min=0.0)

    def update_omega(self, v_fast: Tensor, v_fast_prev: Tensor) -> None:
        """Update observer gain from an online Lipschitz proxy."""
        L_instant = (v_fast.detach() - v_fast_prev.detach()).abs()

        self._L_hat_ema = (
            self._L_hat_ema_beta * self._L_hat_ema
            + (1 - self._L_hat_ema_beta) * L_instant
        )

        L_hat = self._L_hat_ema.max().item()

        new_omega = max(0.3, min(L_hat, 1.0))
        self._omega_o = new_omega
        self._k_ap = 2.0 * new_omega
        self._k_i = new_omega**2

        self.eso.set_omega(new_omega)

    @property
    def lambdas(self) -> Tensor:
        """Current dual variables [n_constraints]."""
        return self._lambdas

    @property
    def omega_o(self) -> float:
        return self._omega_o

    def state_dict(self) -> dict:
        return {
            "lambdas": self._lambdas,
            "omega_o": self._omega_o,
            "k_ap": self._k_ap,
            "k_i": self._k_i,
            "L_hat_ema": self._L_hat_ema,
            "eso": self.eso.state_dict(),
        }
