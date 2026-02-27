"""Soft-ZCA whitening transform (§1.2): full and low-rank variants."""

from __future__ import annotations

import torch
from torch import Tensor

from src.whitening.covariance import OnlineCovariance


class SoftZCAWhitener:
    """Frozen Soft-ZCA whitening transform.

    W_white = U(Λ + αI)^{-1/2} U^T

    Properties:
    - At α=0: full ZCA whitening (Σ^{-1/2}).
    - At α→∞: identity (no whitening).
    - ZCA preserves maximal alignment with original coordinates.
    - α = λ_k / κ_target bounds effective condition number to ~κ_target.

    For d > 4096: uses low-rank variant storing only top-k eigenvectors.
    """

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

        # Store in float32 for inference (eigendecomp was float64)
        self._mean = mean.float()
        self._eigenvalues = eigenvalues.float()  # [d] or [k]
        self._eigenvectors = eigenvectors.float()  # [d, d] or [d, k]

        # Rank selection: k = min{k : Σ_{i≤k} λ_i / tr(Λ) ≥ 1 - δ_rank}
        full_eigenvalues = eigenvalues.float()
        cumvar = full_eigenvalues.cumsum(0) / full_eigenvalues.sum()
        self._effective_rank = int((cumvar >= 1.0 - delta_rank).nonzero(as_tuple=True)[0][0]) + 1
        self._k = self._effective_rank

        # α = λ_k / κ_target
        self._alpha = float(full_eigenvalues[self._k - 1] / kappa_target)

        # Determine if using low-rank variant
        self._low_rank = self.d > 4096 and self._k < self.d

        if self._low_rank:
            # Store only top-k eigenvectors
            self._U_k = self._eigenvectors[:, : self._k].contiguous()  # [d, k]
            self._Lambda_k = self._eigenvalues[: self._k]  # [k]

            # Mean eigenvalue of tail
            if self._k < self.d:
                self._lambda_bar = float(
                    full_eigenvalues[self._k :].mean()
                )
            else:
                self._lambda_bar = float(self._alpha)

            # Precompute scaling factors
            self._scale_k = (self._Lambda_k + self._alpha).rsqrt()  # [k]
            self._scale_tail = 1.0 / (self._lambda_bar + self._alpha) ** 0.5

            # Inverse scaling factors
            self._inv_scale_k = (self._Lambda_k + self._alpha).sqrt()  # [k]
            self._inv_scale_tail = (self._lambda_bar + self._alpha) ** 0.5

            print(f"Low-rank whitener: d={self.d}, k={self._k}, α={self._alpha:.4f}")
        else:
            # Full whitening matrices [d, d]
            scales = (self._eigenvalues + self._alpha).rsqrt()  # [d]
            U = self._eigenvectors  # [d, d]
            self._W_white = U @ torch.diag(scales) @ U.T  # [d, d]

            inv_scales = (self._eigenvalues + self._alpha).sqrt()
            self._W_white_inv = U @ torch.diag(inv_scales) @ U.T

            # Precompute precision matrix for faithfulness: Σ_α^{-1} = W^T W
            self._precision = self._W_white.T @ self._W_white  # [d, d]

            print(f"Full whitener: d={self.d}, k={self._k}, α={self._alpha:.4f}")

    @classmethod
    def from_covariance(
        cls,
        cov: OnlineCovariance,
        kappa_target: float = 100.0,
        delta_rank: float = 0.01,
    ) -> "SoftZCAWhitener":
        """Build whitener from converged covariance estimator.

        Eigendecomposition is performed in float64 for numerical stability,
        results are stored in float32.
        """
        Sigma = cov.get_covariance()  # [d, d] float64
        mean = cov.get_mean()  # [d] float64

        # Eigendecomposition in float64
        eigenvalues, eigenvectors = torch.linalg.eigh(Sigma)

        # eigh returns ascending order; reverse to descending
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)

        # Clamp small/negative eigenvalues (numerical noise)
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
        """Whiten: x → x̃ = W_white(x - μ).

        Args:
            x: [B, d] activations in original space.

        Returns:
            [B, d] whitened activations.
        """
        centered = x - self._mean

        if self._low_rank:
            # φ_k(x) = U_k Λ_k^{-1/2} U_k^T (x-μ) + (1/√(λ̄+α)) P_⊥(x-μ)
            proj = centered @ self._U_k  # [B, k]
            whitened_proj = proj * self._scale_k  # [B, k]
            result_top = whitened_proj @ self._U_k.T  # [B, d]

            # Complement: P_⊥ = I - U_k U_k^T
            complement = centered - (proj @ self._U_k.T)  # [B, d]
            result_tail = complement * self._scale_tail

            return result_top + result_tail
        else:
            return centered @ self._W_white.T

    def inverse(self, x_tilde: Tensor) -> Tensor:
        """Unwhiten: x̃ → x = W_white^{-1} x̃ + μ.

        Args:
            x_tilde: [B, d] whitened activations.

        Returns:
            [B, d] activations in original space.
        """
        if self._low_rank:
            proj = x_tilde @ self._U_k  # [B, k]
            unwhitened_proj = proj * self._inv_scale_k
            result_top = unwhitened_proj @ self._U_k.T

            complement = x_tilde - (proj @ self._U_k.T)
            result_tail = complement * self._inv_scale_tail

            return result_top + result_tail + self._mean
        else:
            return x_tilde @ self._W_white_inv.T + self._mean

    def compute_mahalanobis_sq(self, diff: Tensor) -> Tensor:
        """Compute ‖diff‖²_{Σ_α^{-1}} = diff^T Σ_α^{-1} diff.

        This equals ‖W_white · diff‖² but avoids materializing the
        whitened vector. Used for faithfulness violation.

        Args:
            diff: [B, d] difference vector (x - x̂) in original space.

        Returns:
            [B] squared Mahalanobis distances.
        """
        if self._low_rank:
            # Σ_α^{-1} = U_k diag(1/(λ_i+α)) U_k^T + (1/(λ̄+α)) P_⊥
            proj = diff @ self._U_k  # [B, k]
            scaled_proj = proj * self._scale_k  # [B, k] — scale_k = (λ+α)^{-1/2}
            top_term = (scaled_proj**2).sum(dim=1)  # [B]

            complement = diff - (proj @ self._U_k.T)
            tail_term = (complement**2).sum(dim=1) * (self._scale_tail**2)

            return top_term + tail_term
        else:
            # diff @ Σ_α^{-1} @ diff.T, take diagonal
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
