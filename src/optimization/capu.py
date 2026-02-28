"""Monotone CAPU: per-constraint adaptive penalty."""

import torch
from torch import Tensor

device = torch.device("cuda")


class MonotoneCAPU:
    """Monotone per-constraint adaptive penalty controller.

    Rho can only increase, preserving monotone convergence guarantees.
    """

    def __init__(
        self,
        initial_violations: Tensor,
        rho_0: float = 1.0,
        beta_slow: float = 0.99,
        eps_num: float = 1e-8,
    ) -> None:
        self.n_constraints = initial_violations.shape[0]
        self.beta_slow = beta_slow
        self.eps_num = eps_num

        self._etas = 1.0 / (initial_violations.abs().to(device) + eps_num).sqrt()

        self._v_bar = torch.ones(self.n_constraints, device=device)

        self._rhos = torch.full((self.n_constraints,), rho_0, device=device)

    def step(self, v_fast: Tensor) -> None:
        """Update penalty coefficients from fast EMA violations (monotone: can only increase)."""
        self._v_bar = (
            self.beta_slow * self._v_bar
            + (1 - self.beta_slow) * v_fast.detach() ** 2
        )

        target = self._etas / (self._v_bar + self.eps_num).sqrt()
        self._rhos = torch.max(self._rhos, target)

    @property
    def rhos(self) -> Tensor:
        """Current penalty coefficients [n_constraints]."""
        return self._rhos

    def state_dict(self) -> dict:
        return {
            "etas": self._etas,
            "v_bar": self._v_bar,
            "rhos": self._rhos,
        }

    def load_state_dict(self, sd: dict) -> None:
        self._etas = sd["etas"].to(device)
        self._v_bar = sd["v_bar"].to(device)
        self._rhos = sd["rhos"].to(device)
