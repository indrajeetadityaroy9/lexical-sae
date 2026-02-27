"""Welford online mean + covariance estimation with convergence detection."""

from __future__ import annotations

import math

import torch


class OnlineCovariance:
    """Welford online covariance estimation with snapshot-based convergence."""

    def __init__(self, d: int, delta_cov: float = 1e-3) -> None:
        self.d = d
        self.delta_cov = delta_cov
        self.check_interval = min(math.ceil(d**2), 1_000_000)

        self._n = 0
        self._mean = torch.zeros(d, dtype=torch.float64)
        self._M2 = torch.zeros(d, d, dtype=torch.float64)

        self._snapshot_cov: torch.Tensor | None = None
        self._n_at_snapshot = 0
        self._converged = False

    def update(self, x: torch.Tensor) -> None:
        """Update statistics with a batch of activations.

        Args:
            x: [batch, d] activation tensor.
        """
        x = x.to(dtype=torch.float64, device="cpu")
        batch_size = x.shape[0]

        batch_mean = x.mean(dim=0)
        batch_n = batch_size

        delta = batch_mean - self._mean
        new_n = self._n + batch_n
        new_mean = self._mean + delta * (batch_n / new_n)

        batch_centered = x - batch_mean  # [batch, d]
        batch_M2 = batch_centered.T @ batch_centered  # [d, d]
        self._M2 += batch_M2 + torch.outer(delta, delta) * (
            self._n * batch_n / new_n
        )

        self._mean = new_mean
        self._n = new_n

        if (
            not self._converged
            and self._n - self._n_at_snapshot >= self.check_interval
        ):
            self._check_convergence()

    def _check_convergence(self) -> None:
        """Check if covariance has converged relative to snapshot."""
        current_cov = self.get_covariance()

        if self._snapshot_cov is not None:
            diff_norm = torch.norm(current_cov - self._snapshot_cov, p="fro")
            current_norm = torch.norm(current_cov, p="fro")
            relative_change = diff_norm / current_norm

            if relative_change < self.delta_cov:
                self._converged = True

        self._snapshot_cov = current_cov.clone()
        self._n_at_snapshot = self._n

    def has_converged(self) -> bool:
        return self._converged

    def get_mean(self) -> torch.Tensor:
        """Return the estimated mean [d] in float64."""
        return self._mean.clone()

    def get_covariance(self) -> torch.Tensor:
        """Return the estimated covariance [d, d] in float64."""
        if self._n < 2:
            return torch.eye(self.d, dtype=torch.float64)
        return self._M2 / (self._n - 1)

    @property
    def n_samples(self) -> int:
        return self._n
