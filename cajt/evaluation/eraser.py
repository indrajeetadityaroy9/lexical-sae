"""ERASER faithfulness metrics at the sparse bottleneck level.

Comprehensiveness, sufficiency, and AOPC metrics operating on sparse_vector
dimensions rather than input tokens. This exploits CIS's FMM (Faithfulness
Measurable Model) property: zeroing sparse_vector entries causes no
distribution shift, unlike input-space erasure (Madsen 2024).

Reference: DeYoung et al. 2020 (ACL), "ERASER: A Benchmark to Evaluate
Rationalized NLP Models."
"""

import torch

from cajt.evaluation.collect import collect_sparse_and_attributions


# 50-point grid from 1% to 50% in 1% steps — cheap (no model forward pass per point,
# only sparse vector masking) and gives proper trapezoidal AOPC.
_K_PERCENTAGES = tuple(round(i / 100, 2) for i in range(1, 51))


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

    device = sparse_vectors.device

    with torch.inference_mode():
        original_logits = model.classifier_logits_only(sparse_vectors)
        original_probs = torch.softmax(original_logits, dim=-1)
        batch_indices = torch.arange(len(labels), device=device)
        original_target_probs = original_probs[batch_indices, labels]

    results = {}
    for k in k_percentages:
        top_mask = _get_topk_mask(attributions, k)
        patched = sparse_vectors.clone()
        patched[top_mask] = 0.0

        with torch.inference_mode():
            patched_logits = model.classifier_logits_only(patched)
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

    device = sparse_vectors.device

    with torch.inference_mode():
        original_logits = model.classifier_logits_only(sparse_vectors)
        original_probs = torch.softmax(original_logits, dim=-1)
        batch_indices = torch.arange(len(labels), device=device)
        original_target_probs = original_probs[batch_indices, labels]

    results = {}
    for k in k_percentages:
        top_mask = _get_topk_mask(attributions, k)
        patched = sparse_vectors.clone()
        patched[~top_mask] = 0.0  # zero everything EXCEPT top-k

        with torch.inference_mode():
            patched_logits = model.classifier_logits_only(patched)
            patched_probs = torch.softmax(patched_logits, dim=-1)
            patched_target_probs = patched_probs[batch_indices, labels]

        suff = (original_target_probs - patched_target_probs).mean().item()
        results[k] = suff

    return results


def compute_aopc(scores: dict[float, float]) -> float:
    """Area Over Perturbation Curve via trapezoidal integration.

    Gives a proper AUC over the k-percentage axis, weighted by spacing
    between evaluation points. Falls back to simple mean if < 2 points.
    """
    if len(scores) < 2:
        return sum(scores.values()) / max(1, len(scores))
    sorted_items = sorted(scores.items())
    ks = [k for k, _ in sorted_items]
    vals = [v for _, v in sorted_items]
    area = sum(
        (ks[i + 1] - ks[i]) * (vals[i] + vals[i + 1]) / 2
        for i in range(len(ks) - 1)
    )
    return area / (ks[-1] - ks[0])  # normalize by k-range


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

    sparse_vectors, attributions, _, labels_t = collect_sparse_and_attributions(
        model, input_ids_list, attention_mask_list, labels,
    )

    comp = compute_comprehensiveness(model, sparse_vectors, attributions, labels_t)
    suff = compute_sufficiency(model, sparse_vectors, attributions, labels_t)

    return {
        "comprehensiveness": comp,
        "sufficiency": suff,
        "aopc_comprehensiveness": compute_aopc(comp),
        "aopc_sufficiency": compute_aopc(suff),
    }
