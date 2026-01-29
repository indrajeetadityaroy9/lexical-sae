"""Sparse Autoencoder architectures for transformer interpretability.

Implements TopK and BatchTopK SAE variants based on recent literature:
- TopK SAE: Per-sample top-k sparsity (Gao et al., 2024)
- BatchTopK SAE: Batch-level top-k for better feature utilization (Bussmann et al., 2024)

References:
    - https://arxiv.org/abs/2406.04093 (Scaling and Evaluating SAEs)
    - https://arxiv.org/abs/2412.06410 (BatchTopK SAEs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple


class SAEOutput(NamedTuple):
    """Output from SAE forward pass."""
    x_hat: torch.Tensor      # Reconstructed activations [batch, d_model]
    latents: torch.Tensor    # Sparse feature activations [batch, d_sae]
    loss: torch.Tensor       # Reconstruction loss (MSE)
    aux_loss: torch.Tensor   # Auxiliary loss for dead latent prevention


class TopKSAE(nn.Module):
    """Sparse Autoencoder with per-sample TopK activation.

    For each input, keeps only the k largest activations and zeros the rest.
    This ensures exactly k features are active per sample.

    Args:
        d_model: Input dimension (e.g., 768 for DistilBERT)
        expansion_factor: Multiplier for latent dimension (d_sae = d_model * expansion_factor)
        k: Number of active features per sample
        normalize_decoder: Whether to normalize decoder columns to unit norm
        aux_loss_coef: Coefficient for auxiliary loss (dead latent prevention)
    """

    def __init__(
        self,
        d_model: int = 768,
        expansion_factor: int = 8,
        k: int = 32,
        normalize_decoder: bool = True,
        aux_loss_coef: float = 1/32,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_model * expansion_factor
        self.k = k
        self.normalize_decoder = normalize_decoder
        self.aux_loss_coef = aux_loss_coef

        # Encoder: d_model -> d_sae
        self.W_enc = nn.Parameter(torch.empty(d_model, self.d_sae))
        self.b_enc = nn.Parameter(torch.zeros(self.d_sae))

        # Decoder: d_sae -> d_model
        self.W_dec = nn.Parameter(torch.empty(self.d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        self._init_weights()

        # Track feature activation frequency for dead latent detection
        self.register_buffer("feature_acts", torch.zeros(self.d_sae))
        self.register_buffer("num_batches", torch.tensor(0))

    def _init_weights(self):
        """Initialize weights using Kaiming uniform for encoder, transpose for decoder."""
        nn.init.kaiming_uniform_(self.W_enc, nonlinearity="relu")
        # Initialize decoder as transpose of encoder (helps avoid dead latents early)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)

    @property
    def W_dec_normalized(self) -> torch.Tensor:
        """Return decoder weights, optionally normalized to unit norm."""
        if self.normalize_decoder:
            return F.normalize(self.W_dec, dim=1)
        return self.W_dec

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations to sparse latent representation.

        Args:
            x: Input activations [batch, d_model]

        Returns:
            Sparse latent activations [batch, d_sae] with exactly k non-zeros per row
        """
        # Linear projection
        pre_acts = x @ self.W_enc + self.b_enc  # [batch, d_sae]

        # TopK activation: keep only k largest values per sample
        topk_values, topk_indices = torch.topk(pre_acts, self.k, dim=-1)

        # Create sparse output
        latents = torch.zeros_like(pre_acts)
        latents.scatter_(-1, topk_indices, F.relu(topk_values))

        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode sparse latents back to activation space.

        Args:
            latents: Sparse latent activations [batch, d_sae]

        Returns:
            Reconstructed activations [batch, d_model]
        """
        return latents @ self.W_dec_normalized + self.b_dec

    def forward(self, x: torch.Tensor) -> SAEOutput:
        """Forward pass with reconstruction and auxiliary losses.

        Args:
            x: Input activations [batch, d_model]

        Returns:
            SAEOutput with x_hat, latents, loss, and aux_loss
        """
        latents = self.encode(x)
        x_hat = self.decode(latents)

        # Reconstruction loss (MSE)
        loss = F.mse_loss(x_hat, x)

        # Auxiliary loss: encourage activation of underused features
        # Based on approach from Anthropic's scaling paper
        aux_loss = self._compute_aux_loss(x, latents)

        # Update activation tracking (for monitoring, not training)
        if self.training:
            with torch.no_grad():
                self.feature_acts += (latents > 0).float().sum(dim=0)
                self.num_batches += 1

        return SAEOutput(x_hat=x_hat, latents=latents, loss=loss, aux_loss=aux_loss)

    def _compute_aux_loss(self, x: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """Compute auxiliary loss to prevent dead latents.

        Uses the approach from Anthropic: reconstruct using features that would
        have been selected if we used a larger k, weighted by how close they
        were to being selected.
        """
        if self.aux_loss_coef == 0:
            return torch.tensor(0.0, device=x.device)

        # Get pre-activations for features not in top-k
        pre_acts = x @ self.W_enc + self.b_enc

        # Find features just below the threshold (k+1 to 2k)
        k_aux = min(self.k, self.d_sae - self.k)
        if k_aux <= 0:
            return torch.tensor(0.0, device=x.device)

        # Mask out already-selected features
        mask = (latents > 0).float()
        masked_pre_acts = pre_acts * (1 - mask) - 1e9 * mask

        # Get next-k features
        topk_aux_values, topk_aux_indices = torch.topk(masked_pre_acts, k_aux, dim=-1)

        # Create auxiliary latents
        aux_latents = torch.zeros_like(pre_acts)
        aux_latents.scatter_(-1, topk_aux_indices, F.relu(topk_aux_values))

        # Reconstruct residual using auxiliary features
        residual = x - x_hat if 'x_hat' in dir() else x - self.decode(latents)
        aux_reconstruction = aux_latents @ self.W_dec_normalized

        # Auxiliary loss: how well can auxiliary features explain the residual?
        aux_loss = self.aux_loss_coef * F.mse_loss(aux_reconstruction, residual)

        return aux_loss

    def get_dead_latent_fraction(self) -> float:
        """Return fraction of latents that never activated during training."""
        if self.num_batches == 0:
            return 0.0
        return (self.feature_acts == 0).float().mean().item()

    def reset_stats(self):
        """Reset activation tracking statistics."""
        self.feature_acts.zero_()
        self.num_batches.zero_()


class BatchTopKSAE(TopKSAE):
    """Sparse Autoencoder with batch-level TopK activation.

    Instead of selecting top-k per sample, selects top-(k * batch_size) across
    the entire batch. This improves feature utilization by allowing the same
    total number of activations but distributing them more flexibly.

    Reference: https://arxiv.org/abs/2412.06410

    Args:
        d_model: Input dimension (e.g., 768 for DistilBERT)
        expansion_factor: Multiplier for latent dimension
        k: Average number of active features per sample
        normalize_decoder: Whether to normalize decoder columns to unit norm
        aux_loss_coef: Coefficient for auxiliary loss
    """

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode with batch-level TopK selection.

        Args:
            x: Input activations [batch, d_model]

        Returns:
            Sparse latent activations [batch, d_sae]
        """
        batch_size = x.shape[0]

        # Linear projection
        pre_acts = x @ self.W_enc + self.b_enc  # [batch, d_sae]

        # Flatten for batch-level selection
        flat_pre_acts = pre_acts.flatten()  # [batch * d_sae]

        # Select top-(k * batch_size) across entire batch
        total_k = self.k * batch_size
        total_k = min(total_k, flat_pre_acts.numel())  # Handle small batches

        topk_values, topk_indices = torch.topk(flat_pre_acts, total_k)

        # Create sparse output
        flat_latents = torch.zeros_like(flat_pre_acts)
        flat_latents[topk_indices] = F.relu(topk_values)

        # Reshape back to [batch, d_sae]
        latents = flat_latents.view(batch_size, self.d_sae)

        return latents


def create_sae(
    variant: str = "topk",
    d_model: int = 768,
    expansion_factor: int = 8,
    k: int = 32,
    **kwargs,
) -> TopKSAE:
    """Factory function to create SAE variants.

    Args:
        variant: "topk" or "batchtopk"
        d_model: Input dimension
        expansion_factor: Latent expansion factor
        k: Sparsity level (features per sample)
        **kwargs: Additional arguments passed to SAE constructor

    Returns:
        Configured SAE instance
    """
    variants = {
        "topk": TopKSAE,
        "batchtopk": BatchTopKSAE,
    }

    if variant not in variants:
        raise ValueError(f"Unknown SAE variant: {variant}. Choose from {list(variants.keys())}")

    return variants[variant](
        d_model=d_model,
        expansion_factor=expansion_factor,
        k=k,
        **kwargs,
    )
