"""
Triton kernels for SPLADE operations.

Provides fused implementations of:
- Log-saturation activation: log(1 + ReLU(x))
- Masked SPLADE aggregation: log1p(relu(x)) * mask + max_pool

These kernels minimize memory bandwidth by fusing multiple operations
that would otherwise require separate kernel launches and intermediate
tensor allocations.

Hardware Assumptions:
- CUDA compute capability >= 7.0 (Volta+)
- Contiguous tensor layouts (no strided views)
- Float32 or Float16 inputs
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


# =============================================================================
# Fused Log-Saturation Kernel: log(1 + ReLU(x))
# =============================================================================

@triton.jit
def _log_saturation_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused log(1 + ReLU(x)) activation.

    Replaces two kernel launches (relu + log1p) with one.
    Memory reads: 1x, Memory writes: 1x (vs 2x reads, 2x writes unfused)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Fused computation: log(1 + max(0, x))
    # ReLU
    x_relu = tl.maximum(x, 0.0)
    # log1p = log(1 + x)
    result = tl.log(1.0 + x_relu)

    # Store output
    tl.store(output_ptr + offsets, result, mask=mask)


def log_saturation_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of log(1 + ReLU(x)).

    Args:
        x: Input tensor of any shape

    Returns:
        Output tensor with same shape

    Shape Assumptions:
        - Input must be contiguous
        - Total elements must fit in GPU memory
    """
    assert x.is_cuda, "Input must be on CUDA device"
    assert x.is_contiguous(), "Input must be contiguous"

    output = torch.empty_like(x)
    n_elements = x.numel()

    # Tune block size based on tensor size
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _log_saturation_kernel[grid](
        x, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


# =============================================================================
# Fused Masked SPLADE Aggregation: log1p(relu(x)) * mask -> max_pool
# =============================================================================

@triton.jit
def _splade_aggregate_kernel(
    logits_ptr,           # [batch, seq_len, vocab_size]
    mask_ptr,             # [batch, seq_len]
    output_ptr,           # [batch, vocab_size]
    batch_size,
    seq_len,
    vocab_size,
    logits_batch_stride,
    logits_seq_stride,
    mask_batch_stride,
    output_batch_stride,
    BLOCK_SIZE_V: tl.constexpr,  # Block over vocab dimension
):
    """
    Fused SPLADE aggregation kernel.

    For each (batch, vocab) position:
    1. Load all seq_len logits
    2. Apply log1p(relu(x)) to each
    3. Apply attention mask
    4. Compute max over sequence
    5. Store result

    This fuses 4 operations into 1 kernel, reducing memory traffic by ~4x.
    """
    # 2D grid: (batch, vocab_blocks)
    batch_idx = tl.program_id(0)
    vocab_block_idx = tl.program_id(1)

    # Vocab offsets for this block
    vocab_offsets = vocab_block_idx * BLOCK_SIZE_V + tl.arange(0, BLOCK_SIZE_V)
    vocab_mask = vocab_offsets < vocab_size

    # Initialize max values to -inf
    max_vals = tl.zeros([BLOCK_SIZE_V], dtype=tl.float32) - float('inf')

    # Iterate over sequence positions
    for seq_idx in range(seq_len):
        # Load attention mask for this position
        mask_offset = batch_idx * mask_batch_stride + seq_idx
        attn_mask = tl.load(mask_ptr + mask_offset)

        # Skip if masked out (attention_mask == 0)
        if attn_mask > 0:
            # Load logits for this (batch, seq, vocab_block)
            logits_offset = (batch_idx * logits_batch_stride +
                           seq_idx * logits_seq_stride +
                           vocab_offsets)
            logits = tl.load(logits_ptr + logits_offset, mask=vocab_mask, other=0.0)

            # Fused log1p(relu(logits))
            logits_relu = tl.maximum(logits, 0.0)
            weights = tl.log(1.0 + logits_relu)

            # Update max (masked positions contribute 0, which won't affect max)
            max_vals = tl.maximum(max_vals, weights)

    # Handle case where all positions were masked (set to 0 instead of -inf)
    max_vals = tl.where(max_vals == -float('inf'), 0.0, max_vals)

    # Store output
    output_offset = batch_idx * output_batch_stride + vocab_offsets
    tl.store(output_ptr + output_offset, max_vals, mask=vocab_mask)


def splade_aggregate_triton(
    logits: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Triton implementation of fused SPLADE aggregation.

    Equivalent to:
        weights = torch.log1p(F.relu(logits))
        mask = attention_mask.unsqueeze(-1).expand_as(weights)
        weights = weights * mask
        doc_vector, _ = torch.max(weights, dim=1)

    Args:
        logits: MLM logits [batch_size, seq_len, vocab_size]
        attention_mask: Attention mask [batch_size, seq_len]

    Returns:
        doc_vector: Sparse document vectors [batch_size, vocab_size]

    Shape Assumptions:
        - logits: 3D tensor, contiguous, float32
        - attention_mask: 2D tensor, contiguous
        - batch_size * vocab_size fits in GPU memory
    """
    assert logits.is_cuda and attention_mask.is_cuda, "Inputs must be on CUDA"
    assert logits.dim() == 3, f"Expected 3D logits, got {logits.dim()}D"
    assert attention_mask.dim() == 2, f"Expected 2D mask, got {attention_mask.dim()}D"
    assert logits.is_contiguous(), "Logits must be contiguous"

    batch_size, seq_len, vocab_size = logits.shape
    assert attention_mask.shape == (batch_size, seq_len), "Mask shape mismatch"

    # Allocate output
    output = torch.empty(batch_size, vocab_size, device=logits.device, dtype=logits.dtype)

    # Block size for vocab dimension (tuned for common vocab sizes)
    BLOCK_SIZE_V = 256

    # Grid: (batch_size, ceil(vocab_size / BLOCK_SIZE_V))
    grid = (batch_size, triton.cdiv(vocab_size, BLOCK_SIZE_V))

    _splade_aggregate_kernel[grid](
        logits,
        attention_mask,
        output,
        batch_size,
        seq_len,
        vocab_size,
        logits.stride(0),  # batch stride
        logits.stride(1),  # seq stride
        attention_mask.stride(0),  # mask batch stride
        output.stride(0),  # output batch stride
        BLOCK_SIZE_V=BLOCK_SIZE_V,
    )

    return output


# =============================================================================
# Optimized FLOPS Regularization
# =============================================================================

@triton.jit
def _flops_reg_kernel(
    input_ptr,           # [batch_size, vocab_size]
    output_ptr,          # [1] scalar output
    partial_sums_ptr,    # [num_vocab_blocks] partial results
    batch_size,
    vocab_size,
    input_batch_stride,
    BLOCK_SIZE_V: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
):
    """
    FLOPS regularization: sum_j (mean_i(|w_ij|))^2

    Computes partial sums per vocab block, to be reduced in a second pass.
    """
    vocab_block_idx = tl.program_id(0)
    vocab_start = vocab_block_idx * BLOCK_SIZE_V
    vocab_offsets = vocab_start + tl.arange(0, BLOCK_SIZE_V)
    vocab_mask = vocab_offsets < vocab_size

    # Accumulate sum of abs values for this vocab block
    sums = tl.zeros([BLOCK_SIZE_V], dtype=tl.float32)

    # Process batch in blocks
    for batch_start in range(0, batch_size, BLOCK_SIZE_B):
        batch_offsets = batch_start + tl.arange(0, BLOCK_SIZE_B)
        batch_mask = batch_offsets < batch_size

        # Load block [BLOCK_SIZE_B, BLOCK_SIZE_V]
        for i in range(BLOCK_SIZE_B):
            if batch_start + i < batch_size:
                offset = (batch_start + i) * input_batch_stride + vocab_offsets
                vals = tl.load(input_ptr + offset, mask=vocab_mask, other=0.0)
                sums += tl.abs(vals)

    # Compute mean and square
    means = sums / batch_size
    squared = means * means

    # Sum squared means for this block
    block_sum = tl.sum(squared)

    # Store partial sum
    tl.store(partial_sums_ptr + vocab_block_idx, block_sum)


def flops_regularization_triton(activations: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of FLOPS regularization.

    L_FLOPS = sum_j (mean_i(|w_ij|))^2

    Args:
        activations: Sparse activation tensor [batch_size, vocab_size]

    Returns:
        Scalar loss tensor
    """
    assert activations.is_cuda, "Input must be on CUDA"
    assert activations.dim() == 2, f"Expected 2D tensor, got {activations.dim()}D"
    assert activations.is_contiguous(), "Input must be contiguous"

    batch_size, vocab_size = activations.shape

    BLOCK_SIZE_V = 256
    BLOCK_SIZE_B = 32

    num_vocab_blocks = triton.cdiv(vocab_size, BLOCK_SIZE_V)
    partial_sums = torch.empty(num_vocab_blocks, device=activations.device, dtype=torch.float32)
    output = torch.empty(1, device=activations.device, dtype=torch.float32)

    grid = (num_vocab_blocks,)

    _flops_reg_kernel[grid](
        activations,
        output,
        partial_sums,
        batch_size,
        vocab_size,
        activations.stride(0),
        BLOCK_SIZE_V=BLOCK_SIZE_V,
        BLOCK_SIZE_B=BLOCK_SIZE_B,
    )

    # Final reduction on CPU (small tensor)
    return partial_sums.sum()


# =============================================================================
# Reference PyTorch Implementations (for fallback and testing)
# =============================================================================

def log_saturation_pytorch(x: torch.Tensor) -> torch.Tensor:
    """Reference PyTorch implementation of log(1 + ReLU(x))."""
    return torch.log1p(torch.relu(x))


def splade_aggregate_pytorch(
    logits: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """Reference PyTorch implementation of SPLADE aggregation."""
    import torch.nn.functional as F
    weights = torch.log1p(F.relu(logits))
    mask = attention_mask.unsqueeze(-1).expand_as(weights)
    weights = weights * mask
    doc_vector, _ = torch.max(weights, dim=1)
    return doc_vector


def flops_regularization_pytorch(activations: torch.Tensor) -> torch.Tensor:
    """Reference PyTorch implementation of FLOPS regularization."""
    mean_activations = torch.mean(torch.abs(activations), dim=0)
    return torch.sum(mean_activations ** 2)


# =============================================================================
# Unified Interface with Automatic Fallback
# =============================================================================

_TRITON_AVAILABLE = True


def log_saturation(x: torch.Tensor, use_triton: Optional[bool] = None) -> torch.Tensor:
    """
    Log-saturation activation with automatic backend selection.

    Args:
        x: Input tensor
        use_triton: Force Triton (True), PyTorch (False), or auto (None)

    Returns:
        log(1 + ReLU(x))
    """
    if use_triton is None:
        use_triton = _TRITON_AVAILABLE and x.is_cuda and x.is_contiguous()

    if use_triton:
        return log_saturation_triton(x)
    else:
        return log_saturation_pytorch(x)


def splade_aggregate(
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
    use_triton: Optional[bool] = None
) -> torch.Tensor:
    """
    Fused SPLADE aggregation with automatic backend selection.

    Args:
        logits: MLM logits [batch, seq, vocab]
        attention_mask: Attention mask [batch, seq]
        use_triton: Force Triton (True), PyTorch (False), or auto (None)

    Returns:
        Document vectors [batch, vocab]
    """
    if use_triton is None:
        use_triton = (_TRITON_AVAILABLE and
                     logits.is_cuda and
                     logits.is_contiguous() and
                     logits.dim() == 3)

    if use_triton:
        return splade_aggregate_triton(logits, attention_mask)
    else:
        return splade_aggregate_pytorch(logits, attention_mask)


def flops_reg(
    activations: torch.Tensor,
    use_triton: Optional[bool] = None
) -> torch.Tensor:
    """
    FLOPS regularization with automatic backend selection.

    Args:
        activations: Activation tensor [batch, vocab]
        use_triton: Force Triton (True), PyTorch (False), or auto (None)

    Returns:
        Scalar loss
    """
    if use_triton is None:
        use_triton = (_TRITON_AVAILABLE and
                     activations.is_cuda and
                     activations.is_contiguous() and
                     activations.dim() == 2)

    if use_triton:
        return flops_regularization_triton(activations)
    else:
        return flops_regularization_pytorch(activations)
