"""SPLADE GPU kernels: CUDA C++ forward, Triton backward."""

import torch
import triton
import triton.language as tl
from splade_classifier._cuda import splade_cuda_kernels as _cuda
from splade_classifier.adaptive import StableOps


@triton.jit
def _backward_kernel(
    logits_ptr, mask_ptr, grad_out_ptr, grad_logits_ptr,
    batch_size, seq_len, vocab_size,
    logits_batch_stride, logits_seq_stride, mask_batch_stride,
    grad_out_batch_stride, grad_logits_batch_stride, grad_logits_seq_stride,
    BLOCK_SIZE_V: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    vocab_block_idx = tl.program_id(1)
    vocab_offsets = vocab_block_idx * BLOCK_SIZE_V + tl.arange(0, BLOCK_SIZE_V)
    vocab_mask = vocab_offsets < vocab_size

    max_vals = tl.zeros([BLOCK_SIZE_V], dtype=tl.float32) - float('inf')
    max_seq_idx = tl.zeros([BLOCK_SIZE_V], dtype=tl.int32)

    for seq_idx in range(seq_len):
        mask_offset = batch_idx * mask_batch_stride + seq_idx
        attn_mask = tl.load(mask_ptr + mask_offset)
        if attn_mask > 0:
            logits_offset = batch_idx * logits_batch_stride + seq_idx * logits_seq_stride + vocab_offsets
            logits = tl.load(logits_ptr + logits_offset, mask=vocab_mask, other=0.0)
            weights = tl.log(1.0 + tl.maximum(logits, 0.0))
            update_mask = weights > max_vals
            max_vals = tl.where(update_mask, weights, max_vals)
            max_seq_idx = tl.where(update_mask, seq_idx, max_seq_idx)

    grad_out_offset = batch_idx * grad_out_batch_stride + vocab_offsets
    grad_out = tl.load(grad_out_ptr + grad_out_offset, mask=vocab_mask, other=0.0)

    for seq_idx in range(seq_len):
        grad_logits_offset = batch_idx * grad_logits_batch_stride + seq_idx * grad_logits_seq_stride + vocab_offsets
        is_argmax = (max_seq_idx == seq_idx)
        logits_offset = batch_idx * logits_batch_stride + seq_idx * logits_seq_stride + vocab_offsets
        logits = tl.load(logits_ptr + logits_offset, mask=vocab_mask, other=0.0)
        mask_offset = batch_idx * mask_batch_stride + seq_idx
        attn_mask = tl.load(mask_ptr + mask_offset)
        relu_logits = tl.maximum(logits, 0.0)
        log1p_grad = tl.where(logits > 0, 1.0 / (1.0 + relu_logits), 0.0)
        grad = tl.where(is_argmax, grad_out * log1p_grad * attn_mask, 0.0)
        tl.store(grad_logits_ptr + grad_logits_offset, grad, mask=vocab_mask)


def _backward_triton(mlm_logits, attention_mask, grad_output):
    B, S, V = mlm_logits.shape
    grad = torch.zeros_like(mlm_logits)
    _backward_kernel[(B, triton.cdiv(V, 256))](
        mlm_logits.contiguous(), attention_mask.contiguous(), grad_output.contiguous(), grad,
        B, S, V, mlm_logits.stride(0), mlm_logits.stride(1), attention_mask.stride(0),
        grad_output.stride(0), grad.stride(0), grad.stride(1), BLOCK_SIZE_V=256)
    return grad


def splade_aggregate(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """logits [B,S,V], mask [B,S] -> sparse_vec [B,V]."""
    return _cuda.splade_aggregate(logits.float().contiguous(), mask.contiguous())


def compute_flops_regularization(activations: torch.Tensor) -> torch.Tensor:
    """activations [B,V] -> scalar."""
    return _cuda.flops_reg(activations.contiguous())


class SpladeAggregateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mlm_logits, attention_mask):
        ctx.save_for_backward(mlm_logits, attention_mask)
        return splade_aggregate(mlm_logits, attention_mask)

    @staticmethod
    def backward(ctx, grad_output):
        return _backward_triton(*ctx.saved_tensors, grad_output), None


class FlopsRegFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activations):
        ctx.save_for_backward(activations)
        return compute_flops_regularization(activations)

    @staticmethod
    def backward(ctx, grad_output):
        act, = ctx.saved_tensors
        B = act.shape[0]
        mean_abs = torch.abs(act).mean(dim=0)
        sign = torch.where(act != 0, torch.sign(act), torch.zeros_like(act))
        return grad_output * 2.0 * mean_abs.unsqueeze(0) * sign / B


def compute_si_flops_regularization(activations: torch.Tensor) -> torch.Tensor:
    """Self-normalizing FLOPS regularization.

    Scale-invariant formulation:
        L_SI = (Σᵥ (meanᵦ |a[b,v]| / ||a||_F)² / V)

    Where:
    - ||a||_F = Frobenius norm (provides scale invariance)
    - /V = vocabulary normalization (provides size invariance)

    This eliminates the need for manual tuning of flops_lambda.

    Args:
        activations: [batch_size, vocab_size] sparse activations

    Returns:
        Scalar regularization loss
    """
    B, V = activations.shape
    eps = StableOps.get_eps(activations.dtype, 'div')
    frob_norm = torch.norm(activations, p='fro') + eps
    normalized = activations / frob_norm
    mean_abs = normalized.abs().mean(dim=0)
    return (mean_abs ** 2).sum() / V


class SelfNormalizingFlopsRegFunction(torch.autograd.Function):
    """Self-normalizing FLOPS with autograd support."""

    @staticmethod
    def forward(ctx, activations):
        ctx.save_for_backward(activations)
        return compute_si_flops_regularization(activations)

    @staticmethod
    def backward(ctx, grad_output):
        act, = ctx.saved_tensors
        B, V = act.shape
        eps = StableOps.get_eps(act.dtype, 'div')
        frob_norm = torch.norm(act, p='fro') + eps
        act_normalized = act / frob_norm
        mean_abs = act_normalized.abs().mean(dim=0)
        sign = torch.where(act_normalized != 0, torch.sign(act_normalized), torch.zeros_like(act_normalized))
        scale = 1.0 / (V * frob_norm)
        return grad_output * 2.0 * scale * mean_abs.unsqueeze(0) * sign / B
