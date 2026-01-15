"""
Interpretability module for understanding SPLADE representations.

This module provides Sparse Autoencoders (SAE) for feature decomposition,
enabling analysis of what SPLADE models learn.

Key classes:
- SparseAutoencoder: Decomposes sparse vectors into interpretable features
- SAEOutput: Dataclass containing SAE forward pass outputs
"""

from .sparse_autoencoder import (
    SparseAutoencoder,
    SAEOutput,
)

__all__ = [
    "SparseAutoencoder",
    "SAEOutput",
]
