"""
Interpretability module for understanding SPLADE representations.

This module provides tools for analyzing what SPLADE models learn:
1. Sparse Autoencoders (SAE) for feature decomposition
2. Feature analysis for identifying monosemantic concepts
3. Visualization tools for term importance and clustering

Key classes:
- SparseAutoencoder: Decomposes sparse vectors into interpretable features
- FeatureAnalyzer: Analyzes SAE features to find meaningful concepts
- InterpretabilityVisualizer: Plots and visualizations
"""

from .sparse_autoencoder import (
    SparseAutoencoder,
    SAEOutput,
    SAETrainer,
    extract_splade_vectors,
)
from .feature_analysis import (
    FeatureAnalyzer,
    FeatureInfo,
)
from .visualization import (
    InterpretabilityVisualizer,
)

__all__ = [
    # SAE
    "SparseAutoencoder",
    "SAEOutput",
    "SAETrainer",
    "extract_splade_vectors",
    # Analysis
    "FeatureAnalyzer",
    "FeatureInfo",
    # Visualization
    "InterpretabilityVisualizer",
]
