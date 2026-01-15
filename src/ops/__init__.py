"""
Optimized GPU operations for SPLADE and SAE.

Provides Triton-accelerated implementations with automatic PyTorch fallback.

Usage:
    from src.ops import splade_aggregate, topk_activation

    # Automatic backend selection (Triton on CUDA, PyTorch otherwise)
    doc_vectors = splade_aggregate(logits, attention_mask)

    # Force specific backend
    doc_vectors = splade_aggregate(logits, attention_mask, use_triton=False)

Supported Operations:
    - splade_aggregate: Fused log1p + relu + mask + max_pool
    - flops_reg: FLOPS regularization loss
    - topk_activation: TopK activation with scatter

Hardware Requirements:
    - CUDA compute capability >= 7.0 (Volta or newer)
    - Triton >= 2.0
"""

import torch

# Check Triton availability
try:
    import triton
    TRITON_AVAILABLE = True
    TRITON_VERSION = triton.__version__
except ImportError:
    TRITON_AVAILABLE = False
    TRITON_VERSION = None

# Import public API
from .splade_kernels import splade_aggregate, flops_reg
from .sae_kernels import topk_activation


__all__ = [
    "splade_aggregate",
    "flops_reg",
    "topk_activation",
    "TRITON_AVAILABLE",
    "TRITON_VERSION",
]
