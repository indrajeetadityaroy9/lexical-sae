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

from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

# Import optimized ops with fallback
try:
    from src.ops import topk_activation, TRITON_AVAILABLE
    USE_TRITON_OPS = TRITON_AVAILABLE
except ImportError:
    USE_TRITON_OPS = False
    topk_activation = None


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


class SAETrainer:
    """
    Trainer for Sparse Autoencoder.

    Handles the training loop, logging, and checkpointing
    for SAE training on pre-computed SPLADE vectors.
    """

    def __init__(
        self,
        sae: SparseAutoencoder,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        device: Optional[torch.device] = None
    ):
        self.sae = sae
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.sae.to(self.device)

        self.optimizer = torch.optim.AdamW(
            sae.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.history: List[Dict[str, float]] = []

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int = 0
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.sae.train()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_sparsity_loss = 0.0
        total_active = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            if isinstance(batch, (list, tuple)):
                vectors = batch[0]
            else:
                vectors = batch

            vectors = vectors.to(self.device)

            self.optimizer.zero_grad()

            output = self.sae(vectors, return_loss=True)

            output.loss.backward()
            self.optimizer.step()

            total_loss += output.loss.item()
            total_recon_loss += output.reconstruction_loss.item()
            total_sparsity_loss += output.sparsity_loss.item()
            total_active += output.active_features
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{output.loss.item():.4f}",
                "recon": f"{output.reconstruction_loss.item():.4f}",
                "active": f"{output.active_features}"
            })

        metrics = {
            "loss": total_loss / num_batches,
            "reconstruction_loss": total_recon_loss / num_batches,
            "sparsity_loss": total_sparsity_loss / num_batches,
            "active_features": total_active / num_batches
        }

        self.history.append(metrics)
        return metrics

    def train(
        self,
        vectors: torch.Tensor,
        epochs: int = 10,
        batch_size: int = 256,
        shuffle: bool = True
    ) -> List[Dict[str, float]]:
        """
        Train SAE on pre-computed SPLADE vectors.

        Args:
            vectors: SPLADE vectors [num_docs, vocab_size]
            epochs: Number of training epochs
            batch_size: Training batch size
            shuffle: Whether to shuffle data

        Returns:
            Training history
        """
        dataset = TensorDataset(vectors)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

        for epoch in range(epochs):
            metrics = self.train_epoch(dataloader, epoch)
            print(f"Epoch {epoch}: loss={metrics['loss']:.4f}, "
                  f"recon={metrics['reconstruction_loss']:.4f}, "
                  f"active={metrics['active_features']:.1f}")

        return self.history

    def save_checkpoint(self, path: str):
        """Save SAE checkpoint."""
        torch.save({
            "sae_state_dict": self.sae.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "config": {
                "input_dim": self.sae.input_dim,
                "hidden_dim": self.sae.hidden_dim,
                "k": self.sae.k,
                "sparsity_coefficient": self.sae.sparsity_coefficient,
                "activation": self.sae.activation_type
            }
        }, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: Optional[torch.device] = None):
        """Load SAE from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]

        sae = SparseAutoencoder(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            k=config["k"],
            sparsity_coefficient=config["sparsity_coefficient"],
            activation=config["activation"]
        )
        sae.load_state_dict(checkpoint["sae_state_dict"])

        trainer = cls(sae, device=device)
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.history = checkpoint["history"]

        return trainer


def extract_splade_vectors(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> torch.Tensor:
    """
    Extract SPLADE vectors from a trained model.

    Args:
        model: Trained SPLADE classifier
        dataloader: DataLoader for the dataset
        device: Device to use

    Returns:
        vectors: Stacked SPLADE vectors [num_docs, vocab_size]
    """
    model.eval()
    all_vectors = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting vectors"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Get sparse vectors (not logits)
            _, sparse_vec = model(input_ids, attention_mask)
            all_vectors.append(sparse_vec.cpu())

    return torch.cat(all_vectors, dim=0)


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
