"""ADRC multiplier update with PI + ESO cancellation (§4.2–§4.3)."""

from __future__ import annotations

import torch
from torch import Tensor


class ExtendedStateObserver:
    """ESO for estimating lumped disturbance in constraint dynamics.

    Models constraint dynamics as first-order: λ̇_i = u_i + f_i,
    where f_i is the lumped disturbance (stochastic noise, constraint coupling,
    nonstationarity).

    ESO update (discrete-time, Δt=1):
        f̂_t = ξ_t + ω_o · ṽ_fast_t              [estimate BEFORE ξ update]
        ξ_{t+1} = (1-ω_o)ξ_t - ω_o²·ṽ_fast_t - ω_o·λ_t   [update ξ]

    Update order matters: compute f̂ from CURRENT ξ, THEN update ξ for next step.
    """

    def __init__(
        self,
        n_constraints: int = 3,
        omega_o: float = 0.3,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        self.n_constraints = n_constraints
        self._omega_o = omega_o

        # Auxiliary observer state
        self._xi = torch.zeros(n_constraints, device=device)
        # Estimated disturbance
        self._f_hat = torch.zeros(n_constraints, device=device)

    def step(self, v_fast: Tensor, lambdas: Tensor) -> Tensor:
        """Compute disturbance estimate, then update internal state.

        Args:
            v_fast: [n_constraints] fast EMA of constraint violations.
            lambdas: [n_constraints] current dual variables.

        Returns:
            f_hat: [n_constraints] estimated disturbance.
        """
        omega = self._omega_o

        # Step 1: Compute f̂ from CURRENT ξ (before update)
        self._f_hat = self._xi + omega * v_fast

        # Step 2: Update ξ for next step
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
    """ADRC-controlled dual variable update.

    Owns the ESO (composition). Computes:
        u_i = k_ap·(ṽ_fast - ṽ_fast_prev) + k_i·ṽ_slow - f̂_i
        λ_i = max(0, λ_i + u_i)

    where k_ap = 2ω_o, k_i = ω_o² (critically damped gains).
    The characteristic polynomial (s + ω_o)² = s² + 2ω_o·s + ω_o²
    gives critically damped response (no oscillation, fastest non-oscillatory convergence).
    """

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

        # EMA for Lipschitz estimate (for self-calibrating ω_o)
        self._L_hat_ema = torch.zeros(n_constraints, device=device)
        self._L_hat_ema_beta = 0.9  # Fast EMA for Lipschitz estimate

    def step(self, v_fast: Tensor, v_fast_prev: Tensor, v_slow: Tensor) -> None:
        """Full ADRC update: ESO step → compute u_i → update λ_i.

        Called every training step.

        Args:
            v_fast: [n_constraints] current fast EMA violations.
            v_fast_prev: [n_constraints] previous step's fast EMA violations.
            v_slow: [n_constraints] slow EMA violations.
        """
        # ESO: estimate disturbance
        f_hat = self.eso.step(v_fast.detach(), self._lambdas)

        # PI + ESO cancellation
        proportional = self._k_ap * (v_fast.detach() - v_fast_prev.detach())
        integral = self._k_i * v_slow.detach()
        u = proportional + integral - f_hat

        # Update dual variables with non-negativity projection
        self._lambdas = torch.clamp(self._lambdas + u, min=0.0)

    def update_omega(self, v_fast: Tensor, v_fast_prev: Tensor) -> None:
        """Self-calibrating observer gain (§4.3). Called every slow_update_interval steps.

        ω_o = clip(L̂, 0.3, 1.0) where L̂ is the online Lipschitz estimate
        of constraint dynamics.
        """
        # Per-constraint Lipschitz estimate
        L_instant = (v_fast.detach() - v_fast_prev.detach()).abs()

        # EMA of Lipschitz estimate
        self._L_hat_ema = (
            self._L_hat_ema_beta * self._L_hat_ema
            + (1 - self._L_hat_ema_beta) * L_instant
        )

        # Take max across constraints for a single ω_o
        L_hat = self._L_hat_ema.max().item()

        # Clip to valid range
        new_omega = max(0.3, min(L_hat, 1.0))
        self._omega_o = new_omega
        self._k_ap = 2.0 * new_omega
        self._k_i = new_omega**2

        # Propagate to ESO
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

