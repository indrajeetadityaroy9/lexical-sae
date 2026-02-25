"""Baseline explainer methods and ERASER-based comparison with DLA.

All baselines produce [B, V] attributions at the sparse_vector level
(bottleneck-level) for fair comparison via ERASER metrics.

Methods:
  - DLA: Exact algebraic decomposition (0 extra passes)
  - Gradient x Input: d(logit_c)/d(sparse_j) * sparse_j (1 backward)
  - Integrated Gradients: Path integral from zero baseline (steps backwards)
  - Attention: CLS attention weights projected to vocabulary space
"""

import time

import torch

from cajt.evaluation.collect import collect_sparse_vectors
from cajt.runtime import autocast, DEVICE
from cajt.evaluation.eraser import (
    compute_aopc,
    compute_comprehensiveness,
    compute_sufficiency,
)
from cajt.core.attribution import compute_attribution_tensor
from cajt.core.constants import EVAL_BATCH_SIZE
from cajt.evaluation.layerwise import get_transformer_layers


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

    with torch.inference_mode(), autocast():
        sparse_seq, *_ = model(input_ids, attention_mask)
        _, sparse_vector, W_eff, _ = model.classify(sparse_seq, attention_mask)
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


    with torch.no_grad(), autocast():
        sparse_seq, *_ = model(input_ids, attention_mask)
        sparse_vector = model.to_pooled(sparse_seq, attention_mask)

    sparse_vector = sparse_vector.detach().float().requires_grad_(True)
    logits = model.classifier_logits_only(sparse_vector)
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


    with torch.no_grad(), autocast():
        sparse_seq, *_ = model(input_ids, attention_mask)
        sparse_vector = model.to_pooled(sparse_seq, attention_mask)

    sparse_vector = sparse_vector.detach().float()
    batch_indices = torch.arange(len(target_classes), device=sparse_vector.device)

    accumulated_grad = torch.zeros_like(sparse_vector)

    for step in range(1, steps + 1):
        alpha = step / steps
        interpolated = (alpha * sparse_vector).requires_grad_(True)
        logits = model.classifier_logits_only(interpolated)
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
    layers = get_transformer_layers(model)
    last_layer = layers[-1]

    # Capture input to the last layer via pre-hook
    captured = {}

    def _capture(module, args):
        captured["hidden"] = args[0].detach()

    handle = last_layer.register_forward_pre_hook(_capture)
    try:
        with torch.inference_mode(), autocast():
            encoder_output = encoder(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()

    hidden_in = captured["hidden"]  # Input to last layer [B, L, H]
    hidden_out = encoder_output.last_hidden_state  # Output [B, L, H]

    # Compute Q, K from ModernBERT's fused QKV projection
    attn_mod = last_layer.attn
    with torch.inference_mode():
        qkv = attn_mod.Wqkv(hidden_in)  # [B, L, 3 * D]
    D = qkv.shape[-1] // 3
    Q, K, _ = qkv.split(D, dim=-1)
    n_heads = encoder.config.num_attention_heads

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


    cls_attn, _ = _get_last_layer_cls_attention(
        model, input_ids, attention_mask,
    )

    # Get per-position sparse representations via backbone (second encoder pass)
    with torch.inference_mode(), autocast():
        sparse_sequence, *_ = model.compute_sparse_sequence(
            input_ids, attention_mask,
        )  # [B, L, V]

    # Attention-weighted sum (instead of max-pooling)
    attn_sparse = (cls_attn.unsqueeze(-1) * sparse_sequence.float()).sum(dim=1)  # [B, V]
    return attn_sparse



# Explainer comparison (merged from compare_explainers.py)


_EXPLAINERS = {
    "dla": dla_attribution,
    "gradient": gradient_attribution,
    "integrated_gradients": integrated_gradients_attribution,
    "attention": attention_attribution,
}


def run_explainer_comparison(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels: list[int],
) -> dict[str, dict]:
    """Run all explainers and compute ERASER metrics for each.

    Returns:
        {name: {"comprehensiveness": {k: v}, "sufficiency": {k: v},
                "aopc_comp": float, "aopc_suff": float, "time_seconds": float}}
    """

    eval_batch = EVAL_BATCH_SIZE
    labels_t = torch.tensor(labels, device=DEVICE)

    # Pre-compute sparse_vectors once (shared across explainers for ERASER eval)
    sparse_vectors = collect_sparse_vectors(model, input_ids_list, attention_mask_list)

    results = {}
    for name, explainer_fn in _EXPLAINERS.items():
        t0 = time.perf_counter()

        all_attr = []
        for start in range(0, len(input_ids_list), eval_batch):
            end = min(start + eval_batch, len(input_ids_list))
            batch_ids = torch.cat(input_ids_list[start:end], dim=0)
            batch_mask = torch.cat(attention_mask_list[start:end], dim=0)
            batch_labels = labels_t[start:end]
            attr = explainer_fn(model, batch_ids, batch_mask, batch_labels)
            all_attr.append(attr.float())
        attributions = torch.cat(all_attr, dim=0)

        elapsed = time.perf_counter() - t0

        comp = compute_comprehensiveness(model, sparse_vectors, attributions, labels_t)
        suff = compute_sufficiency(model, sparse_vectors, attributions, labels_t)

        results[name] = {
            "comprehensiveness": comp,
            "sufficiency": suff,
            "aopc_comp": compute_aopc(comp),
            "aopc_suff": compute_aopc(suff),
            "time_seconds": elapsed,
        }

    return results
