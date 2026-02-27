"""Modified CAPU: non-monotone per-constraint adaptive penalty (§5.2)."""

from __future__ import annotations

import torch
from torch import Tensor


class ModifiedCAPU:
    """Non-monotone per-constraint adaptive penalty.

    Original CAPU uses a monotone ratchet: ρ_i ← max(ρ_i, η_i/√(v̄_i+ε)).
    SPALF modifies this to allow penalty relaxation:
        ρ_i = max(ρ_min, η_i / √(v̄_i + ε_num))

    The max is against ρ_min (fixed floor), NOT ρ_{i,t-1} (no ratchet).
    When violations grow, ρ decreases (RMSprop-style normalization).
    When violations shrink, ρ increases, tightening enforcement.

    η_i = c_η / √(|v_{i,0}| + ε_num) — self-calibrated from first-batch violations.
    v̄_i = EMA of ṽ_fast² (second moment of fast signal).
    """

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

        # Self-calibrate η_i from initial violations (§8.3)
        self._etas = (
            c_eta / (initial_violations.abs().to(device) + eps_num).sqrt()
        )

        # Running second moment of fast EMA signal
        self._v_bar = torch.ones(self.n_constraints, device=device)

        # Current penalty coefficients
        self._rhos = torch.full((self.n_constraints,), rho_0, device=device)

    def step(self, v_fast: Tensor) -> None:
        """Update penalty coefficients from fast EMA violations.

        Called every slow_update_interval steps (typically 100).

        Args:
            v_fast: [n_constraints] current fast EMA of violations.
        """
        if self._frozen:
            return

        # Update second moment: v̄ ← β·v̄ + (1-β)·ṽ_fast²
        self._v_bar = (
            self.beta_slow * self._v_bar
            + (1 - self.beta_slow) * v_fast.detach() ** 2
        )

        # Non-monotone penalty: max against floor, not previous value
        target = self._etas / (self._v_bar + self.eps_num).sqrt()
        self._rhos = torch.clamp(target, min=self.rho_min)

    def freeze(self) -> None:
        """Freeze ρ updates (called at Phase 2 transition).

        Prevents transient penalty spikes from the new KL signal.
        """
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

