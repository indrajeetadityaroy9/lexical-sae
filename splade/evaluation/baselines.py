"""Baseline explainer methods for comparison with DLA.

All baselines produce [B, V] attributions at the sparse_vector level
(bottleneck-level) for fair comparison via ERASER metrics. This ensures
all methods face the same evaluation conditions (FMM-correct erasure).

Methods:
  - DLA: Exact algebraic decomposition (0 extra passes)
  - Gradient x Input: d(logit_c)/d(sparse_j) * sparse_j (1 backward)
  - Integrated Gradients: Path integral from zero baseline (steps backwards)
  - Attention: CLS attention weights projected to vocabulary space
"""

import torch

from splade.mechanistic.attribution import compute_attribution_tensor
from splade.utils.cuda import COMPUTE_DTYPE, unwrap_compiled


def dla_attribution(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_classes: torch.Tensor,
) -> torch.Tensor:
    """Exact DLA attribution (baseline reference).

    Returns [B, V] attribution tensor. Zero extra compute — W_eff is
    already available from the forward pass.
    """
    _model = unwrap_compiled(model)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
        sparse_seq = _model(input_ids, attention_mask)
        _, sparse_vector, W_eff, _ = _model.classify(sparse_seq, attention_mask)
    return compute_attribution_tensor(sparse_vector, W_eff, target_classes).float()


def gradient_attribution(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_classes: torch.Tensor,
) -> torch.Tensor:
    """Gradient x Input in sparse_vector space.

    Computes d(logit_c)/d(sparse_j) * sparse_j via one backward pass
    through the classifier head only (BERT frozen during attribution).

    Returns [B, V].
    """
    _model = unwrap_compiled(model)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
        sparse_seq = _model(input_ids, attention_mask)
        sparse_vector = _model.to_pooled(sparse_seq, attention_mask)

    sparse_vector = sparse_vector.detach().float().requires_grad_(True)
    logits = _model.classifier_logits_only(sparse_vector)
    batch_indices = torch.arange(len(target_classes), device=sparse_vector.device)
    target_logits = logits[batch_indices, target_classes]
    target_logits.sum().backward()

    grad = sparse_vector.grad  # [B, V]
    return (sparse_vector.detach() * grad).detach()


def integrated_gradients_attribution(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_classes: torch.Tensor,
    steps: int = 50,
) -> torch.Tensor:
    """Integrated Gradients from zero baseline to sparse_vector.

    Path integral computed via Riemann sum with `steps` interpolation
    points. Each step requires one backward pass through the classifier
    (microseconds each — classifier is a 2-layer MLP).

    Returns [B, V].
    """
    _model = unwrap_compiled(model)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
        sparse_seq = _model(input_ids, attention_mask)
        sparse_vector = _model.to_pooled(sparse_seq, attention_mask)

    sparse_vector = sparse_vector.detach().float()
    batch_indices = torch.arange(len(target_classes), device=sparse_vector.device)

    accumulated_grad = torch.zeros_like(sparse_vector)

    for step in range(1, steps + 1):
        alpha = step / steps
        interpolated = (alpha * sparse_vector).requires_grad_(True)
        logits = _model.classifier_logits_only(interpolated)
        target_logits = logits[batch_indices, target_classes]
        target_logits.sum().backward()
        accumulated_grad += interpolated.grad.detach()

    ig = sparse_vector * (accumulated_grad / steps)
    return ig


def _get_last_layer_cls_attention(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute CLS attention weights from the last transformer layer.

    Computes Q @ K^T manually rather than relying on output_attentions=True,
    which SDPA attention does not support.

    Returns:
        (cls_attn [B, L], hidden_output [B, L, H])
    """
    encoder = model.encoder

    # Get last transformer layer
    if hasattr(encoder, "transformer"):
        layers = list(encoder.transformer.layer)
    elif hasattr(encoder, "encoder"):
        layers = list(encoder.encoder.layer)
    elif hasattr(encoder, "layers"):
        # ModernBERT: encoder.layers
        layers = list(encoder.layers)
    else:
        raise ValueError("Unknown BERT architecture")
    last_layer = layers[-1]

    # Capture input to the last layer via pre-hook
    captured = {}

    def _capture(module, args):
        captured["hidden"] = args[0].detach()

    handle = last_layer.register_forward_pre_hook(_capture)
    try:
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            encoder_output = encoder(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()

    hidden_in = captured["hidden"]  # Input to last layer [B, L, H]
    hidden_out = encoder_output.last_hidden_state  # Output [B, L, H]

    # Compute Q, K from the last layer's attention module
    if hasattr(last_layer, "attn") and hasattr(last_layer.attn, "Wqkv"):
        # ModernBERT: fused QKV projection
        attn_mod = last_layer.attn
        with torch.inference_mode():
            qkv = attn_mod.Wqkv(hidden_in)  # [B, L, 3 * D]
        D = qkv.shape[-1] // 3
        Q, K, _ = qkv.split(D, dim=-1)
        n_heads = encoder.config.num_attention_heads
    else:
        attn_mod = last_layer.attention
        if hasattr(attn_mod, "q_lin"):  # DistilBERT
            with torch.inference_mode():
                Q = attn_mod.q_lin(hidden_in)
                K = attn_mod.k_lin(hidden_in)
            n_heads = attn_mod.n_heads
        elif hasattr(attn_mod, "self"):  # BERT/RoBERTa
            with torch.inference_mode():
                Q = attn_mod.self.query(hidden_in)
                K = attn_mod.self.key(hidden_in)
            n_heads = attn_mod.self.num_attention_heads
        else:
            raise ValueError("Unknown attention module structure")

    B, L, D = Q.shape
    d_k = D // n_heads

    Q = Q.view(B, L, n_heads, d_k).transpose(1, 2)  # [B, H, L, d_k]
    K = K.view(B, L, n_heads, d_k).transpose(1, 2)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # [B, H, L, L]

    # Apply attention mask
    mask_expanded = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * -1e9
    scores = scores + mask_expanded.to(scores.device)
    attn_weights = torch.softmax(scores.float(), dim=-1)  # [B, H, L, L]

    # CLS row, mean over heads
    cls_attn = attn_weights[:, :, 0, :].mean(dim=1)  # [B, L]
    cls_attn = cls_attn * attention_mask.float()
    cls_attn = cls_attn / cls_attn.sum(dim=1, keepdim=True).clamp(min=1e-8)

    return cls_attn, hidden_out


def attention_attribution(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_classes: torch.Tensor,
) -> torch.Tensor:
    """Attention-weighted attribution projected to vocabulary space.

    Uses CLS token attention from the last transformer layer to weight
    per-position sparse contributions, producing a [B, V] attribution.
    Computes attention weights manually (Q @ K^T) to avoid SDPA limitations.
    """
    _model = unwrap_compiled(model)

    cls_attn, _ = _get_last_layer_cls_attention(
        _model, input_ids, attention_mask,
    )

    # Get per-position sparse representations via backbone (second encoder pass)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
        sparse_sequence = _model._compute_sparse_sequence(
            attention_mask, input_ids=input_ids,
        )  # [B, L, V]

    # Attention-weighted sum (instead of max-pooling)
    attn_sparse = (cls_attn.unsqueeze(-1) * sparse_sequence.float()).sum(dim=1)  # [B, V]
    return attn_sparse
