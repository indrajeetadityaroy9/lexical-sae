"""
Triton kernels for Sparse Autoencoder operations.

Provides optimized implementations of:
- TopK activation with scatter

Hardware Assumptions:
- CUDA compute capability >= 7.0 (Volta+)
- Contiguous tensor layouts
- Float32 or Float16 inputs
"""

import torch
from typing import Optional

# Triton availability check
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False
    triton = None
    tl = None


# =============================================================================
# TopK Activation Kernel
# =============================================================================

if _TRITON_AVAILABLE:
    @triton.jit
    def _topk_scatter_kernel(
        input_ptr,
        output_ptr,
        indices_ptr,         # [batch_size, k] pre-computed topk indices
        values_ptr,          # [batch_size, k] pre-computed topk values
        batch_size,
        hidden_dim,
        k,
        output_batch_stride,
        indices_batch_stride,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Scatter topk values to output tensor.

        This kernel handles the scatter operation after PyTorch topk computes
        the indices and values. The scatter is parallelized across batch and k.
        """
        batch_idx = tl.program_id(0)
        k_block_idx = tl.program_id(1)

        # Offsets within k
        k_start = k_block_idx * BLOCK_SIZE
        k_offsets = k_start + tl.arange(0, BLOCK_SIZE)
        k_mask = k_offsets < k

        # Load indices and values
        idx_offset = batch_idx * indices_batch_stride + k_offsets
        indices = tl.load(indices_ptr + idx_offset, mask=k_mask, other=0)
        values = tl.load(values_ptr + idx_offset, mask=k_mask, other=0.0)

        # Scatter to output
        output_offset = batch_idx * output_batch_stride + indices
        tl.store(output_ptr + output_offset, values, mask=k_mask)


def topk_activation_triton(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Triton implementation of TopK activation.

    Keeps top-k values per row, zeros out the rest.
    Uses PyTorch topk for index computation, Triton for scatter.

    Args:
        x: Input tensor [batch_size, hidden_dim]
        k: Number of values to keep per row

    Returns:
        Sparse tensor with only top-k values non-zero

    Shape Assumptions:
        - x must be 2D and contiguous
        - k < hidden_dim
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available. Install with: pip install triton")
    assert x.is_cuda, "Input must be on CUDA"
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    assert x.is_contiguous(), "Input must be contiguous"

    batch_size, hidden_dim = x.shape
    assert k < hidden_dim, f"k ({k}) must be less than hidden_dim ({hidden_dim})"

    # Compute topk indices and values using PyTorch (highly optimized)
    topk_values, topk_indices = torch.topk(x, k, dim=-1)

    # Allocate output (zeros)
    output = torch.zeros_like(x)

    # Use Triton for parallel scatter
    BLOCK_SIZE = min(256, k)
    grid = (batch_size, triton.cdiv(k, BLOCK_SIZE))

    _topk_scatter_kernel[grid](
        x,
        output,
        topk_indices,
        topk_values,
        batch_size,
        hidden_dim,
        k,
        output.stride(0),
        topk_indices.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


# =============================================================================
# Reference PyTorch Implementations
# =============================================================================

def topk_activation_pytorch(x: torch.Tensor, k: int) -> torch.Tensor:
    """Reference PyTorch implementation of TopK activation."""
    topk_values, topk_indices = torch.topk(x, k, dim=-1)
    sparse_output = torch.zeros_like(x)
    sparse_output.scatter_(-1, topk_indices, topk_values)
    return sparse_output


# =============================================================================
# Unified Interface
# =============================================================================


def topk_activation(
    x: torch.Tensor,
    k: int,
    use_triton: Optional[bool] = None
) -> torch.Tensor:
    """
    TopK activation with automatic backend selection.

    Args:
        x: Input tensor [batch, hidden]
        k: Number of values to keep
        use_triton: Force backend (None=auto)

    Returns:
        Sparse output with top-k values
    """
    if use_triton is None:
        # Use Triton for large hidden_dim where threshold method is faster
        use_triton = (_TRITON_AVAILABLE and
                     x.is_cuda and
                     x.is_contiguous() and
                     x.dim() == 2 and
                     x.shape[1] > 4096)

    if use_triton:
        return topk_activation_triton(x, k)
    else:
        return topk_activation_pytorch(x, k)
