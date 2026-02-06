"""SPLADE GPU kernels: PyTorch forward, Triton backward."""

import torch
import triton
import triton.language as tl

# Epsilon constants for numerical stability.
_EPS = {
    torch.float32: {'div': 1e-6, 'log': 1e-7},
    torch.bfloat16: {'div': 1e-3, 'log': 1e-4},
}


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


def splade_aggregate(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """logits [B,S,V], mask [B,S] -> sparse_vec [B,V]."""
    activated = torch.log1p(torch.relu(logits.float()))
    activated = activated.masked_fill(~mask.unsqueeze(-1).bool(), 0.0)
    return activated.max(dim=1).values


class SpladeAggregateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mlm_logits, attention_mask):
        mlm_logits_f32 = mlm_logits.float()
        ctx.save_for_backward(mlm_logits_f32, attention_mask)
        activated = torch.log1p(torch.relu(mlm_logits_f32))
        activated = activated.masked_fill(~attention_mask.unsqueeze(-1).bool(), 0.0)
        return activated.max(dim=1).values

    @staticmethod
    def backward(ctx, grad_output):
        mlm_logits, attention_mask = ctx.saved_tensors
        B, S, V = mlm_logits.shape
        grad = torch.zeros_like(mlm_logits)
        _backward_kernel[(B, triton.cdiv(V, 256))](
            mlm_logits.contiguous(), attention_mask.contiguous(), grad_output.contiguous(), grad,
            B, S, V, mlm_logits.stride(0), mlm_logits.stride(1), attention_mask.stride(0),
            grad_output.stride(0), grad.stride(0), grad.stride(1), BLOCK_SIZE_V=256)
        return grad, None


class DocumentFrequencyTracker:
    """Track document frequency for DF-FLOPS regularization (arXiv:2505.15070).

    Penalizes high-DF terms to reduce posting list lengths and latency.
    """

    def __init__(self, vocab_size: int, device: str | torch.device = 'cuda'):
        self.vocab_size = vocab_size
        self.device = device
        self.df_counts = torch.zeros(vocab_size, device=device)
        self.doc_count = 0

    def update(self, sparse_vectors: torch.Tensor) -> None:
        term_presence = (sparse_vectors.detach() > 0).float()
        self.df_counts += term_presence.sum(dim=0)
        self.doc_count += sparse_vectors.shape[0]

    def get_weights(self, alpha: float = 0.1, beta: float = 5.0) -> torch.Tensor:
        """Compute DF-based penalty weights: w_t = 1 / (1 + (x^(log_alpha(2)) - 1)^beta)."""
        df_ratio = self.df_counts / self.doc_count
        eps = _EPS[df_ratio.dtype]['log']
        log_alpha = torch.log(torch.tensor(alpha, device=self.device))
        log_alpha = torch.where(log_alpha.abs() < eps, torch.tensor(eps, device=self.device), log_alpha)
        log_alpha_2 = torch.log(torch.tensor(2.0, device=self.device)) / log_alpha

        x_clamped = df_ratio.clamp(min=1e-8)
        x_pow = x_clamped.pow(log_alpha_2)
        inner = (x_pow - 1.0).clamp(min=0.0)
        return 1.0 / (1.0 + inner.pow(beta))

    def get_stats(self) -> dict:
        df_ratio = self.df_counts / self.doc_count
        return {
            "doc_count": self.doc_count,
            "top1_df_pct": df_ratio.max().item() * 100,
            "mean_df_pct": df_ratio.mean().item() * 100,
        }


class DFFlopsRegFunction(torch.autograd.Function):
    """DF-FLOPS regularization with autograd support."""

    @staticmethod
    def forward(ctx, activations: torch.Tensor, df_weights: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(activations, df_weights)
        weighted_mean = (df_weights.unsqueeze(0) * activations.abs()).mean(dim=0)
        return (weighted_mean ** 2).sum()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        act, df_weights = ctx.saved_tensors
        B = act.shape[0]
        weighted_mean = (df_weights.unsqueeze(0) * act.abs()).mean(dim=0)
        sign = torch.where(act != 0, torch.sign(act), torch.zeros_like(act))
        grad_act = grad_output * 2.0 * df_weights.unsqueeze(0) * weighted_mean.unsqueeze(0) * sign / B
        return grad_act, None
