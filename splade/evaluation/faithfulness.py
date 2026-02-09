"""Removability metric for faithfulness evaluation.

Measures whether removing top-k DLA-attributed sparse dimensions flips
the model's prediction. Since Lexical-SAE has exact DLA (zero approximation
error), removing top attributed tokens should reliably flip predictions —
unlike post-hoc explainers operating on standard models.

This is a thin wrapper around the existing ERASER infrastructure, reframed
as "removability" for the SOTA narrative.
"""

import time

import torch

from splade.evaluation.baselines import (
    attention_attribution,
    dla_attribution,
    gradient_attribution,
    integrated_gradients_attribution,
)
from splade.inference import _run_inference_loop
from splade.mechanistic.attribution import compute_attribution_tensor
from splade.utils.cuda import COMPUTE_DTYPE, DEVICE, unwrap_compiled


def compute_removability(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    labels: list[int],
    max_length: int,
    top_k: int = 5,
    batch_size: int = 64,
) -> dict:
    """Remove top-k DLA-attributed tokens from sparse vector, measure prediction flip rate.

    For each sample, identifies the top-k attributed sparse dimensions via
    exact DLA, zeros them, and checks if argmax(logits) changes.

    Returns:
        {"flip_rate": float, "mean_prob_drop": float}
    """
    _model = unwrap_compiled(model)
    encoding = tokenizer(
        texts, max_length=max_length, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    logits, sparse, weff = _run_inference_loop(
        model, input_ids, attention_mask,
        extract_sparse=True, extract_weff=True, batch_size=batch_size,
    )

    labels_t = torch.tensor(labels, device=DEVICE)
    original_preds = logits.argmax(dim=-1)
    original_probs = torch.softmax(logits, dim=-1)
    batch_idx = torch.arange(len(labels), device=DEVICE)
    original_target_probs = original_probs[batch_idx, labels_t]

    # Compute DLA attributions
    attr = compute_attribution_tensor(sparse, weff, labels_t)

    # Zero top-k attributed dims per sample
    patched = sparse.clone()
    for i in range(len(labels)):
        abs_attr = attr[i].abs()
        _, top_idx = abs_attr.topk(min(top_k, (abs_attr > 0).sum().item()))
        patched[i, top_idx] = 0.0

    # Re-classify with patched sparse vectors
    with torch.inference_mode():
        patched_logits = _model.classifier_logits_only(patched)

    patched_preds = patched_logits.argmax(dim=-1)
    patched_probs = torch.softmax(patched_logits, dim=-1)
    patched_target_probs = patched_probs[batch_idx, labels_t]

    flip_rate = (original_preds != patched_preds).float().mean().item()
    mean_prob_drop = (original_target_probs - patched_target_probs).mean().item()

    return {"flip_rate": flip_rate, "mean_prob_drop": mean_prob_drop}


def _removability_for_explainer(
    model: torch.nn.Module,
    explainer_fn,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels_t: torch.Tensor,
    sparse_vectors: torch.Tensor,
    top_k: int,
    batch_size: int,
) -> dict:
    """Compute removability using a specific explainer's attributions."""
    _model = unwrap_compiled(model)
    eval_batch = batch_size

    t0 = time.perf_counter()
    all_attr = []
    for start in range(0, len(labels_t), eval_batch):
        end = min(start + eval_batch, len(labels_t))
        batch_ids = input_ids[start:end].to(DEVICE)
        batch_mask = attention_mask[start:end].to(DEVICE)
        batch_labels = labels_t[start:end]
        attr = explainer_fn(_model, batch_ids, batch_mask, batch_labels)
        all_attr.append(attr.float())
    attributions = torch.cat(all_attr, dim=0)
    elapsed = time.perf_counter() - t0

    # Zero top-k per sample
    patched = sparse_vectors.clone()
    for i in range(len(labels_t)):
        abs_attr = attributions[i].abs()
        n_active = (abs_attr > 0).sum().item()
        if n_active == 0:
            continue
        _, top_idx = abs_attr.topk(min(top_k, n_active))
        patched[i, top_idx] = 0.0

    original_preds = _model.classifier_logits_only(sparse_vectors).argmax(dim=-1)
    original_probs = torch.softmax(
        _model.classifier_logits_only(sparse_vectors), dim=-1
    )
    batch_idx = torch.arange(len(labels_t), device=DEVICE)
    original_target_probs = original_probs[batch_idx, labels_t]

    with torch.inference_mode():
        patched_logits = _model.classifier_logits_only(patched)

    patched_preds = patched_logits.argmax(dim=-1)
    patched_probs = torch.softmax(patched_logits, dim=-1)
    patched_target_probs = patched_probs[batch_idx, labels_t]

    return {
        "flip_rate": (original_preds != patched_preds).float().mean().item(),
        "mean_prob_drop": (original_target_probs - patched_target_probs).mean().item(),
        "time_seconds": elapsed,
    }


_EXPLAINERS = {
    "dla": dla_attribution,
    "gradient": gradient_attribution,
    "integrated_gradients": integrated_gradients_attribution,
    "attention": attention_attribution,
}


def compare_with_baseline_explainer(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    labels: list[int],
    max_length: int,
    top_k: int = 5,
    batch_size: int = 64,
) -> dict[str, dict]:
    """Run removability for DLA + all baseline explainers.

    Returns:
        {explainer_name: {"flip_rate": float, "mean_prob_drop": float, "time_seconds": float}}
    """
    encoding = tokenizer(
        texts, max_length=max_length, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    # Pre-compute sparse vectors once
    _, sparse, _ = _run_inference_loop(
        model, input_ids, attention_mask,
        extract_sparse=True, batch_size=batch_size,
    )

    labels_t = torch.tensor(labels, device=DEVICE)

    results = {}
    for name, explainer_fn in _EXPLAINERS.items():
        results[name] = _removability_for_explainer(
            model, explainer_fn, input_ids, attention_mask,
            labels_t, sparse, top_k, batch_size,
        )

    return results
