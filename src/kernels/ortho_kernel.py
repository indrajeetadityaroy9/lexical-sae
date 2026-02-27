"""Fused co-activation orthogonality kernel.

Computes C3 constraint violation: mean per-sample pairwise cos² of co-active
decoder columns, without materializing a [B, MAX_ACTIVE, d] intermediate.

Triton kernel: each program handles one sample, computes tiled dot products
on-the-fly from the [d, F] decoder. Eliminates the B-serial Python loop.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

# Maximum active features per sample. Samples exceeding this are subsampled.
MAX_ACTIVE = 256


def compute_ortho_triton(
    z: Tensor,
    W_dec_A: Tensor,
    W_dec_B: Tensor,
    tau_ortho: float,
) -> Tensor:
    """Compute C3 orthogonality violation via Triton kernel.

    Pre-computes active indices and column norms in PyTorch, then launches
    one Triton program per sample for the pairwise cos² computation.

    Args:
        z: [B, F] sparse codes.
        W_dec_A: [d, V] anchored decoder.
        W_dec_B: [d, F-V] free decoder.
        tau_ortho: Orthogonality threshold.

    Returns:
        Scalar violation (positive = constraint violated).
    """
    B, F = z.shape
    W_dec = torch.cat([W_dec_A, W_dec_B], dim=1)  # [d, F]
    d = W_dec.shape[0]

    # Pre-normalize decoder columns (shared across all samples)
    col_norms = W_dec.norm(dim=0, keepdim=True)
    W_dec_norm = W_dec / col_norms  # [d, F]
    # Ensure contiguous row-major layout so stride_wd = F
    W_dec_norm = W_dec_norm.contiguous()

    # Build padded index tensor [B, MAX_ACTIVE] and count tensor [B]
    active_mask = z > 0  # [B, F]
    active_counts = active_mask.sum(dim=1)  # [B]

    # Clamp counts to MAX_ACTIVE
    clamped_counts = active_counts.clamp(max=MAX_ACTIVE).to(torch.int32)

    # Build index tensor: for each sample, collect active feature indices
    indices = torch.zeros(B, MAX_ACTIVE, device=z.device, dtype=torch.int32)
    for b in range(B):
        n = int(active_counts[b].item())
        if n == 0:
            continue
        active_idx = active_mask[b].nonzero(as_tuple=True)[0]
        if n > MAX_ACTIVE:
            # Subsample
            perm = torch.randperm(n, device=z.device)[:MAX_ACTIVE]
            active_idx = active_idx[perm]
            n = MAX_ACTIVE
        indices[b, :n] = active_idx[:n].to(torch.int32)

    # Output: per-sample scores
    output = torch.zeros(B, device=z.device, dtype=torch.float32)

    # Determine BLOCK_D for tiled dot products
    BLOCK_D = min(triton.next_power_of_2(d), 512)

    _ortho_kernel[(B,)](
        W_dec_norm,
        indices,
        clamped_counts,
        output,
        W_dec_norm.stride(0),  # stride for row dim = F (contiguous)
        indices.stride(0),     # stride for batch dim = MAX_ACTIVE
        d,
        F,
        MAX_ACTIVE=MAX_ACTIVE,
        BLOCK_D=BLOCK_D,
    )

    # Average over valid samples (those with >= 2 active features)
    valid_mask = clamped_counts >= 2
    if not valid_mask.any():
        return torch.tensor(-tau_ortho, device=z.device, requires_grad=False)

    mean_score = output[valid_mask].mean()
    return mean_score - tau_ortho


@triton.jit
def _ortho_kernel(
    W_dec_norm_ptr,
    indices_ptr,
    counts_ptr,
    output_ptr,
    stride_wd: tl.constexpr,   # W_dec_norm row stride (= F for contiguous)
    stride_idx: tl.constexpr,   # indices batch stride (= MAX_ACTIVE)
    d: tl.constexpr,
    F: tl.constexpr,
    MAX_ACTIVE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Per-sample co-activation orthogonality via tiled pairwise dot products.

    Each program handles one sample:
    1. Load active indices from pre-computed index tensor.
    2. For each pair (i < j), compute dot(w_i, w_j) via tiled loads.
    3. Accumulate cos²(w_i, w_j) = dot² (columns are pre-normalized).
    4. Normalize by |A|(|A|-1) and write scalar output.

    Grid: (B,)
    """
    bid = tl.program_id(0)
    n_active = tl.load(counts_ptr + bid)

    if n_active < 2:
        tl.store(output_ptr + bid, 0.0)
        return

    idx_base = bid * stride_idx
    total_cos_sq = 0.0

    # Iterate over unique pairs (i < j) among active features
    for i in tl.range(0, MAX_ACTIVE):
        if i >= n_active:
            break
        col_i = tl.load(indices_ptr + idx_base + i).to(tl.int64)

        for j in tl.range(0, MAX_ACTIVE):
            if j <= i or j >= n_active:
                continue

            col_j = tl.load(indices_ptr + idx_base + j).to(tl.int64)

            # Tiled dot product over d dimension
            dot = 0.0
            for k_start in tl.range(0, d, BLOCK_D):
                offs = k_start + tl.arange(0, BLOCK_D)
                mask = offs < d
                # W_dec_norm is [d, F] row-major: element [row, col] at row*F + col
                wi = tl.load(
                    W_dec_norm_ptr + offs * stride_wd + col_i,
                    mask=mask, other=0.0,
                )
                wj = tl.load(
                    W_dec_norm_ptr + offs * stride_wd + col_j,
                    mask=mask, other=0.0,
                )
                dot += tl.sum(wi * wj)

            total_cos_sq += dot * dot

    # Normalize: upper triangle sum * 2 / (n_active * (n_active - 1))
    n_pairs = n_active * (n_active - 1)
    score = 2.0 * total_cos_sq / n_pairs.to(tl.float32)
    tl.store(output_ptr + bid, score)
