"""
Sparse Autoencoder (SAE) for interpreting SPLADE document vectors.

SAEs decompose the polysemantic SPLADE representations into
monosemantic features that are more interpretable. Each learned
feature corresponds to a specific semantic concept.

Key insight: SPLADE vectors are high-dimensional (30k+ dims) but
semantically "superposed" - many concepts share the same dimensions.
SAEs disentangle these into sparse, interpretable features.

Optimizations:
- Uses Triton-accelerated TopK activation when available
- Automatic fallback to PyTorch for unsupported devices

Reference:
    Sparse Autoencoders Find Highly Interpretable Features in Language Models
    https://arxiv.org/abs/2309.08600
"""

from typing import Tuple, Optional, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import optimized ops (no fallback - src.ops must be available)
from src.ops import topk_activation, TRITON_AVAILABLE
USE_TRITON_OPS = TRITON_AVAILABLE


@dataclass
class SAEOutput:
    """Output container for SAE forward pass."""
    reconstruction: torch.Tensor  # Reconstructed input
    hidden: torch.Tensor  # Sparse hidden activations
    loss: torch.Tensor  # Total loss
    reconstruction_loss: torch.Tensor  # MSE reconstruction loss
    sparsity_loss: torch.Tensor  # L1 sparsity penalty
    active_features: int  # Number of non-zero features


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for interpreting SPLADE document vectors.

    Architecture:
        Encoder: Linear(input_dim, hidden_dim) + activation
        Decoder: Linear(hidden_dim, input_dim)

    The hidden layer is encouraged to be sparse via:
    1. TopK activation (only k largest values kept)
    2. L1 penalty on hidden activations

    Attributes:
        input_dim: Dimension of input vectors (vocab_size, e.g., 30522)
        hidden_dim: Number of SAE features (typically 4-16x input_dim)
        k: Number of active features for TopK activation
        sparsity_coefficient: Weight for L1 sparsity penalty
    """

    def __init__(
        self,
        input_dim: int = 30522,
        hidden_dim: int = 16384,
        k: int = 32,
        sparsity_coefficient: float = 1e-3,
        activation: str = "topk",
        tied_weights: bool = False,
        normalize_decoder: bool = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.sparsity_coefficient = sparsity_coefficient
        self.activation_type = activation
        self.tied_weights = tied_weights
        self.normalize_decoder = normalize_decoder

        # Encoder: project to high-dimensional sparse space
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)

        # Decoder: project back to input space
        if tied_weights:
            # Decoder weights are transpose of encoder
            self.decoder = None
            self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        else:
            self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/He initialization."""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)

        if self.decoder is not None:
            nn.init.xavier_uniform_(self.decoder.weight)
            nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to sparse hidden representation.

        Args:
            x: Input tensor [batch, input_dim]

        Returns:
            hidden: Sparse hidden activations [batch, hidden_dim]
        """
        # Linear projection
        hidden = self.encoder(x)

        # Apply activation
        if self.activation_type == "topk":
            hidden = self._topk_activation(hidden)
        elif self.activation_type == "relu":
            hidden = F.relu(hidden)
        elif self.activation_type == "gelu":
            hidden = F.gelu(hidden)
        else:
            raise ValueError(f"Unknown activation: {self.activation_type}")

        return hidden

    def _topk_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        TopK activation: keep only top-k values, zero out the rest.

        This ensures exactly k features are active per sample,
        providing strong sparsity guarantees.

        Uses Triton-accelerated scatter only during inference (no autograd).
        """
        # Use Triton only during inference (no autograd support)
        use_triton = (USE_TRITON_OPS and
                     x.is_cuda and
                     x.is_contiguous() and
                     not torch.is_grad_enabled())

        if use_triton:
            return topk_activation(x, self.k)

        # PyTorch implementation (supports autograd for training)
        topk_values, topk_indices = torch.topk(x, self.k, dim=-1)
        sparse_output = torch.zeros_like(x)
        sparse_output.scatter_(-1, topk_indices, topk_values)
        return sparse_output

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse hidden representation back to input space.

        Args:
            hidden: Sparse hidden [batch, hidden_dim]

        Returns:
            reconstruction: Reconstructed input [batch, input_dim]
        """
        if self.tied_weights:
            # Use transposed encoder weights
            # encoder.weight is [hidden_dim, input_dim]
            # For F.linear(input, weight), weight needs to be [out_features, in_features]
            # We want [batch, hidden_dim] -> [batch, input_dim]
            # So weight should be [input_dim, hidden_dim]
            weights = self.encoder.weight.t()  # [input_dim, hidden_dim]
            if self.normalize_decoder:
                weights = F.normalize(weights, dim=1)
            reconstruction = F.linear(hidden, weights, self.decoder_bias)
        else:
            if self.normalize_decoder:
                # Normalize decoder columns for stability
                with torch.no_grad():
                    self.decoder.weight.data = F.normalize(
                        self.decoder.weight.data, dim=0
                    )
            reconstruction = self.decoder(hidden)

        return reconstruction

    def forward(
        self,
        x: torch.Tensor,
        return_loss: bool = True
    ) -> SAEOutput:
        """
        Forward pass with optional loss computation.

        Args:
            x: Input tensor [batch, input_dim]
            return_loss: Whether to compute losses

        Returns:
            SAEOutput containing reconstruction, hidden, and losses
        """
        # Encode to sparse hidden
        hidden = self.encode(x)

        # Decode back to input space
        reconstruction = self.decode(hidden)

        # Compute losses if requested
        if return_loss:
            # Reconstruction loss (MSE)
            reconstruction_loss = F.mse_loss(reconstruction, x)

            # Sparsity loss (L1 on hidden activations)
            sparsity_loss = self.sparsity_coefficient * hidden.abs().mean()

            # Total loss
            loss = reconstruction_loss + sparsity_loss
        else:
            loss = torch.tensor(0.0)
            reconstruction_loss = torch.tensor(0.0)
            sparsity_loss = torch.tensor(0.0)

        # Count active features
        active_features = (hidden.abs() > 1e-6).sum(dim=-1).float().mean().item()

        return SAEOutput(
            reconstruction=reconstruction,
            hidden=hidden,
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            sparsity_loss=sparsity_loss,
            active_features=int(active_features)
        )

    def get_feature_weights(self, feature_idx: int) -> torch.Tensor:
        """
        Get decoder weights for a specific feature.

        These weights indicate which vocabulary terms are
        associated with this feature.

        Args:
            feature_idx: Index of the SAE feature

        Returns:
            weights: Vocabulary weights [input_dim]
        """
        if self.tied_weights:
            return self.encoder.weight[feature_idx]
        else:
            return self.decoder.weight[:, feature_idx]

    def get_top_tokens_for_feature(
        self,
        feature_idx: int,
        tokenizer,
        k: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Get top vocabulary tokens associated with a feature.

        Args:
            feature_idx: Index of the SAE feature
            tokenizer: HuggingFace tokenizer
            k: Number of top tokens to return

        Returns:
            List of (token, weight) tuples
        """
        weights = self.get_feature_weights(feature_idx)
        values, indices = torch.topk(weights.abs(), k)

        results = []
        for idx, val in zip(indices.tolist(), values.tolist()):
            token = tokenizer.decode([idx]).strip()
            actual_weight = weights[idx].item()
            results.append((token, actual_weight))

        return results


if __name__ == "__main__":
    # Quick test
    print("Testing Sparse Autoencoder...")

    # Create dummy SPLADE vectors
    batch_size = 64
    vocab_size = 30522
    hidden_dim = 4096
    k = 32

    # Simulate sparse SPLADE vectors (95% zeros)
    vectors = torch.zeros(batch_size, vocab_size)
    for i in range(batch_size):
        active_indices = torch.randint(0, vocab_size, (int(vocab_size * 0.05),))
        vectors[i, active_indices] = torch.rand(len(active_indices))

    # Create and test SAE
    sae = SparseAutoencoder(
        input_dim=vocab_size,
        hidden_dim=hidden_dim,
        k=k,
        sparsity_coefficient=1e-3
    )

    output = sae(vectors)

    print(f"Input shape: {vectors.shape}")
    print(f"Hidden shape: {output.hidden.shape}")
    print(f"Reconstruction shape: {output.reconstruction.shape}")
    print(f"Loss: {output.loss.item():.4f}")
    print(f"Reconstruction loss: {output.reconstruction_loss.item():.4f}")
    print(f"Sparsity loss: {output.sparsity_loss.item():.4f}")
    print(f"Active features: {output.active_features}")
    print("SAE test passed!")
