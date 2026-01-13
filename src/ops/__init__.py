"""
Optimized GPU operations for SPLADE and SAE.

This module provides Triton-accelerated implementations of performance-critical
operations, with automatic fallback to PyTorch for unsupported devices/shapes.

Usage:
    from src.ops import splade_aggregate, topk_activation, normalized_linear

    # Automatic backend selection (Triton on CUDA, PyTorch otherwise)
    doc_vectors = splade_aggregate(logits, attention_mask)

    # Force specific backend
    doc_vectors = splade_aggregate(logits, attention_mask, use_triton=True)
    doc_vectors = splade_aggregate(logits, attention_mask, use_triton=False)

All functions follow the same pattern:
- If use_triton=None (default): auto-select based on device, shape, contiguity
- If use_triton=True: use Triton (raises error if not possible)
- If use_triton=False: use PyTorch reference implementation

Supported Operations:
    SPLADE:
        - log_saturation: log(1 + ReLU(x)) activation
        - splade_aggregate: Fused log1p + relu + mask + max_pool
        - flops_reg: FLOPS regularization loss

    SAE:
        - topk_activation: TopK with scatter/threshold
        - normalized_linear: Fused normalize + matmul for tied-weight decode

Hardware Requirements:
    - CUDA compute capability >= 7.0 (Volta or newer)
    - Triton >= 2.0

Numerical Guarantees:
    - All Triton kernels are tested for numerical parity with PyTorch
    - Maximum relative error < 1e-5 for float32
    - Results are deterministic given same inputs
"""

import torch
from typing import Optional

# Check Triton availability
try:
    import triton
    TRITON_AVAILABLE = True
    TRITON_VERSION = triton.__version__
except ImportError:
    TRITON_AVAILABLE = False
    TRITON_VERSION = None

# Import operations
from .splade_kernels import (
    log_saturation,
    log_saturation_triton,
    log_saturation_pytorch,
    splade_aggregate,
    splade_aggregate_triton,
    splade_aggregate_pytorch,
    flops_reg,
    flops_regularization_triton,
    flops_regularization_pytorch,
)

from .sae_kernels import (
    topk_activation,
    topk_activation_triton,
    topk_activation_pytorch,
    normalized_linear,
    normalized_linear_triton,
    normalized_linear_pytorch,
)


__all__ = [
    # SPLADE operations
    "log_saturation",
    "splade_aggregate",
    "flops_reg",
    # SAE operations
    "topk_activation",
    "normalized_linear",
    # Low-level (explicit backend)
    "log_saturation_triton",
    "log_saturation_pytorch",
    "splade_aggregate_triton",
    "splade_aggregate_pytorch",
    "flops_regularization_triton",
    "flops_regularization_pytorch",
    "topk_activation_triton",
    "topk_activation_pytorch",
    "normalized_linear_triton",
    "normalized_linear_pytorch",
    # Utilities
    "TRITON_AVAILABLE",
    "TRITON_VERSION",
    "get_backend_info",
    "benchmark_op",
]


def get_backend_info() -> dict:
    """
    Get information about available backends.

    Returns:
        Dictionary with backend availability and versions
    """
    info = {
        "triton_available": TRITON_AVAILABLE,
        "triton_version": TRITON_VERSION,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name()
        info["cuda_capability"] = torch.cuda.get_device_capability()

    return info


def benchmark_op(
    op_fn,
    *args,
    warmup_iters: int = 10,
    benchmark_iters: int = 100,
    **kwargs
) -> dict:
    """
    Benchmark an operation.

    Args:
        op_fn: Function to benchmark
        *args: Positional arguments for op_fn
        warmup_iters: Number of warmup iterations
        benchmark_iters: Number of benchmark iterations
        **kwargs: Keyword arguments for op_fn

    Returns:
        Dictionary with timing results
    """
    import time

    # Warmup
    for _ in range(warmup_iters):
        op_fn(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(benchmark_iters):
        op_fn(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()

    total_time = end - start
    avg_time = total_time / benchmark_iters

    return {
        "total_time_s": total_time,
        "avg_time_ms": avg_time * 1000,
        "iterations": benchmark_iters,
    }


def _check_numerical_parity(
    triton_fn,
    pytorch_fn,
    *args,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    **kwargs
) -> bool:
    """
    Check numerical parity between Triton and PyTorch implementations.

    Args:
        triton_fn: Triton implementation
        pytorch_fn: PyTorch reference implementation
        *args: Input arguments
        rtol: Relative tolerance
        atol: Absolute tolerance
        **kwargs: Keyword arguments

    Returns:
        True if outputs match within tolerance
    """
    with torch.no_grad():
        triton_out = triton_fn(*args, **kwargs)
        pytorch_out = pytorch_fn(*args, **kwargs)

    return torch.allclose(triton_out, pytorch_out, rtol=rtol, atol=atol)
