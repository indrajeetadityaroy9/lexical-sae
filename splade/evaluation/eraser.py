"""ERASER faithfulness metrics at the sparse bottleneck level.

Comprehensiveness, sufficiency, and AOPC metrics operating on sparse_vector
dimensions rather than input tokens. This exploits CIS's FMM (Faithfulness
Measurable Model) property: zeroing sparse_vector entries causes no
distribution shift, unlike input-space erasure (Madsen 2024).

Reference: DeYoung et al. 2020 (ACL), "ERASER: A Benchmark to Evaluate
Rationalized NLP Models."
"""

import torch

from splade.mechanistic.attribution import compute_attribution_tensor
from splade.utils.cuda import COMPUTE_DTYPE, DEVICE, unwrap_compiled


_K_PERCENTAGES = (0.01, 0.05, 0.10, 0.20, 0.50)


def _get_topk_mask(attributions: torch.Tensor, k_frac: float) -> torch.Tensor:
    """Return a boolean mask selecting the top-k% of ACTIVE dimensions.

    k is computed as a fraction of non-zero dimensions per sample (not total
    vocab size), reflecting the sparse bottleneck structure where only ~100
    of ~30K vocab dims are active. This ensures k-percentages produce
    meaningful variation in the erasure curves.

    Args:
        attributions: [B, V] attribution magnitudes.
        k_frac: fraction of active dimensions to select (e.g. 0.05 = top 5%).

    Returns:
        [B, V] boolean mask (True = selected).
    """
    B, V = attributions.shape
    mask = torch.zeros(B, V, dtype=torch.bool, device=attributions.device)
    abs_attr = attributions.abs()

    for i in range(B):
        n_active = (abs_attr[i] > 0).sum().item()
        k = max(1, int(k_frac * n_active)) if n_active > 0 else 1
        _, top_idx = abs_attr[i].topk(min(k, V))
        mask[i, top_idx] = True

    return mask


def compute_comprehensiveness(
    model: torch.nn.Module,
    sparse_vectors: torch.Tensor,
    attributions: torch.Tensor,
    labels: torch.Tensor,
    k_percentages: tuple[float, ...] = _K_PERCENTAGES,
) -> dict[float, float]:
    """Remove top-k% attributed dimensions, measure prediction probability drop.

    Higher comprehensiveness = more faithful (removing important features hurts).

    Args:
        model: LexicalSAE (or compiled wrapper).
        sparse_vectors: [N, V] pre-computed sparse activations.
        attributions: [N, V] pre-computed DLA attributions.
        labels: [N] ground-truth class indices.
        k_percentages: fractions to evaluate.

    Returns:
        {k: mean_comprehensiveness} for each k.
    """
    _model = unwrap_compiled(model)
    device = sparse_vectors.device

    with torch.inference_mode():
        original_logits = _model.classifier_logits_only(sparse_vectors)
        original_probs = torch.softmax(original_logits, dim=-1)
        batch_indices = torch.arange(len(labels), device=device)
        original_target_probs = original_probs[batch_indices, labels]

    results = {}
    for k in k_percentages:
        top_mask = _get_topk_mask(attributions, k)
        patched = sparse_vectors.clone()
        patched[top_mask] = 0.0

        with torch.inference_mode():
            patched_logits = _model.classifier_logits_only(patched)
            patched_probs = torch.softmax(patched_logits, dim=-1)
            patched_target_probs = patched_probs[batch_indices, labels]

        comp = (original_target_probs - patched_target_probs).mean().item()
        results[k] = comp

    return results


def compute_sufficiency(
    model: torch.nn.Module,
    sparse_vectors: torch.Tensor,
    attributions: torch.Tensor,
    labels: torch.Tensor,
    k_percentages: tuple[float, ...] = _K_PERCENTAGES,
) -> dict[float, float]:
    """Keep only top-k% attributed dimensions, measure prediction preservation.

    Lower sufficiency = more faithful (keeping important features is enough).

    Args:
        model: LexicalSAE (or compiled wrapper).
        sparse_vectors: [N, V] pre-computed sparse activations.
        attributions: [N, V] pre-computed DLA attributions.
        labels: [N] ground-truth class indices.
        k_percentages: fractions to evaluate.

    Returns:
        {k: mean_sufficiency} for each k.
    """
    _model = unwrap_compiled(model)
    device = sparse_vectors.device

    with torch.inference_mode():
        original_logits = _model.classifier_logits_only(sparse_vectors)
        original_probs = torch.softmax(original_logits, dim=-1)
        batch_indices = torch.arange(len(labels), device=device)
        original_target_probs = original_probs[batch_indices, labels]

    results = {}
    for k in k_percentages:
        top_mask = _get_topk_mask(attributions, k)
        patched = sparse_vectors.clone()
        patched[~top_mask] = 0.0  # zero everything EXCEPT top-k

        with torch.inference_mode():
            patched_logits = _model.classifier_logits_only(patched)
            patched_probs = torch.softmax(patched_logits, dim=-1)
            patched_target_probs = patched_probs[batch_indices, labels]

        suff = (original_target_probs - patched_target_probs).mean().item()
        results[k] = suff

    return results


def compute_aopc(scores: dict[float, float]) -> float:
    """Area Over Perturbation Curve: mean score across k values."""
    if not scores:
        return 0.0
    return sum(scores.values()) / len(scores)


def run_eraser_evaluation(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels: list[int],
) -> dict:
    """Run full ERASER evaluation using DLA attributions.

    Batches data, computes DLA attributions, then evaluates comprehensiveness,
    sufficiency, and AOPC at the sparse bottleneck level.

    Returns:
        {"comprehensiveness": {k: score}, "sufficiency": {k: score},
         "aopc_comprehensiveness": float, "aopc_sufficiency": float}
    """
    _model = unwrap_compiled(model)
    eval_batch = 32

    all_sparse = []
    all_attr = []
    labels_t = torch.tensor(labels, device=DEVICE)

    for start in range(0, len(input_ids_list), eval_batch):
        end = min(start + eval_batch, len(input_ids_list))
        batch_ids = torch.cat(input_ids_list[start:end], dim=0)
        batch_mask = torch.cat(attention_mask_list[start:end], dim=0)
        batch_labels = labels_t[start:end]

        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            sparse_seq = _model(batch_ids, batch_mask)
            _, sparse_vector, W_eff, _ = _model.classify(sparse_seq, batch_mask)

        attr = compute_attribution_tensor(sparse_vector, W_eff, batch_labels)
        all_sparse.append(sparse_vector.float())
        all_attr.append(attr.float())

    sparse_vectors = torch.cat(all_sparse, dim=0)
    attributions = torch.cat(all_attr, dim=0)

    comp = compute_comprehensiveness(_model, sparse_vectors, attributions, labels_t)
    suff = compute_sufficiency(_model, sparse_vectors, attributions, labels_t)

    return {
        "comprehensiveness": comp,
        "sufficiency": suff,
        "aopc_comprehensiveness": compute_aopc(comp),
        "aopc_sufficiency": compute_aopc(suff),
    }
