"""Soft-ZCA whitening transform (§1.2): full and low-rank variants."""

from __future__ import annotations

import torch
from torch import Tensor

from src.whitening.covariance import OnlineCovariance


class SoftZCAWhitener:
    """Frozen Soft-ZCA whitening transform."""

    def __init__(
        self,
        mean: Tensor,
        eigenvalues: Tensor,
        eigenvectors: Tensor,
        kappa_target: float = 100.0,
        delta_rank: float = 0.01,
    ) -> None:
        self.d = mean.shape[0]
        self.kappa_target = kappa_target

        self._mean = mean.float()
        self._eigenvalues = eigenvalues.float()
        self._eigenvectors = eigenvectors.float()

        full_eigenvalues = eigenvalues.float()
        cumvar = full_eigenvalues.cumsum(0) / full_eigenvalues.sum()
        self._effective_rank = int((cumvar >= 1.0 - delta_rank).nonzero(as_tuple=True)[0][0]) + 1
        self._k = self._effective_rank

        self._alpha = float(full_eigenvalues[self._k - 1] / kappa_target)

        self._low_rank = self.d > 4096 and self._k < self.d

        if self._low_rank:
            self._U_k = self._eigenvectors[:, : self._k].contiguous()
            self._Lambda_k = self._eigenvalues[: self._k]

            if self._k < self.d:
                self._lambda_bar = float(
                    full_eigenvalues[self._k :].mean()
                )
            else:
                self._lambda_bar = float(self._alpha)

            self._scale_k = (self._Lambda_k + self._alpha).rsqrt()
            self._scale_tail = 1.0 / (self._lambda_bar + self._alpha) ** 0.5

            self._inv_scale_k = (self._Lambda_k + self._alpha).sqrt()
            self._inv_scale_tail = (self._lambda_bar + self._alpha) ** 0.5

            print(f"Low-rank whitener: d={self.d}, k={self._k}, α={self._alpha:.4f}")
        else:
            scales = (self._eigenvalues + self._alpha).rsqrt()
            U = self._eigenvectors
            self._W_white = U @ torch.diag(scales) @ U.T

            inv_scales = (self._eigenvalues + self._alpha).sqrt()
            self._W_white_inv = U @ torch.diag(inv_scales) @ U.T

            self._precision = self._W_white.T @ self._W_white

            print(f"Full whitener: d={self.d}, k={self._k}, α={self._alpha:.4f}")

    @classmethod
    def from_covariance(
        cls,
        cov: OnlineCovariance,
        kappa_target: float = 100.0,
        delta_rank: float = 0.01,
    ) -> "SoftZCAWhitener":
        """Build whitener from a converged covariance estimate."""
        Sigma = cov.get_covariance()
        mean = cov.get_mean()

        eigenvalues, eigenvectors = torch.linalg.eigh(Sigma)

        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)

        eigenvalues = eigenvalues.clamp(min=1e-12)

        print(f"Eigenspectrum: λ_max={eigenvalues[0]:.4f}, λ_min={eigenvalues[-1]:.6f}, κ={eigenvalues[0] / eigenvalues[-1]:.0f}, n_samples={cov.n_samples}")

        return cls(
            mean=mean,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            kappa_target=kappa_target,
            delta_rank=delta_rank,
        )

    def to(self, device: torch.device) -> "SoftZCAWhitener":
        """Move all tensors to a device."""
        self._mean = self._mean.to(device)
        self._eigenvalues = self._eigenvalues.to(device)
        self._eigenvectors = self._eigenvectors.to(device)

        if self._low_rank:
            self._U_k = self._U_k.to(device)
            self._Lambda_k = self._Lambda_k.to(device)
            self._scale_k = self._scale_k.to(device)
            self._inv_scale_k = self._inv_scale_k.to(device)
        else:
            self._W_white = self._W_white.to(device)
            self._W_white_inv = self._W_white_inv.to(device)
            self._precision = self._precision.to(device)

        return self

    def forward(self, x: Tensor) -> Tensor:
        """Whiten activations."""
        centered = x - self._mean

        if self._low_rank:
            proj = centered @ self._U_k
            whitened_proj = proj * self._scale_k
            result_top = whitened_proj @ self._U_k.T

            complement = centered - (proj @ self._U_k.T)
            result_tail = complement * self._scale_tail

            return result_top + result_tail
        else:
            return centered @ self._W_white.T

    def inverse(self, x_tilde: Tensor) -> Tensor:
        """Map whitened activations back to original space."""
        if self._low_rank:
            proj = x_tilde @ self._U_k
            unwhitened_proj = proj * self._inv_scale_k
            result_top = unwhitened_proj @ self._U_k.T

            complement = x_tilde - (proj @ self._U_k.T)
            result_tail = complement * self._inv_scale_tail

            return result_top + result_tail + self._mean
        else:
            return x_tilde @ self._W_white_inv.T + self._mean

    def compute_mahalanobis_sq(self, diff: Tensor) -> Tensor:
        """Compute ||diff||^2 in the regularized precision metric."""
        if self._low_rank:
            proj = diff @ self._U_k
            scaled_proj = proj * self._scale_k
            top_term = (scaled_proj**2).sum(dim=1)

            complement = diff - (proj @ self._U_k.T)
            tail_term = (complement**2).sum(dim=1) * (self._scale_tail**2)

            return top_term + tail_term
        else:
            return (diff @ self._precision * diff).sum(dim=1)

    @property
    def W_white(self) -> Tensor:
        """Full whitening matrix [d, d]. Only available for non-low-rank."""
        return self._W_white

    @property
    def W_white_inv(self) -> Tensor:
        """Full inverse whitening matrix [d, d]. Only available for non-low-rank."""
        return self._W_white_inv

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def effective_rank(self) -> int:
        return self._effective_rank

    @property
    def is_low_rank(self) -> bool:
        return self._low_rank

    @property
    def mean(self) -> Tensor:
        return self._mean

    def state_dict(self) -> dict:
        """Serialize whitener state for checkpointing."""
        return {
            "mean": self._mean,
            "eigenvalues": self._eigenvalues,
            "eigenvectors": self._eigenvectors,
            "kappa_target": self.kappa_target,
            "alpha": self._alpha,
            "effective_rank": self._effective_rank,
            "d": self.d,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> SoftZCAWhitener:
        """Reconstruct whitener from checkpoint state dict."""
        return cls(
            mean=d["mean"],
            eigenvalues=d["eigenvalues"],
            eigenvectors=d["eigenvectors"],
            kappa_target=d["kappa_target"],
        )
