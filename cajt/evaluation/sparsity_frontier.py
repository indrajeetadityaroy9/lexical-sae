"""Sparsity-fidelity frontier and NAOPC normalization.

Sweeps JumpReLU threshold post-hoc to trace the L0 vs downstream loss
Pareto curve. Also computes Normalized AOPC for fair cross-model comparison.
"""

import math

import torch

from cajt.evaluation.collect import collect_sparse_and_attributions
from cajt.evaluation.downstream_loss import compute_downstream_loss
from cajt.evaluation.eraser import (
    compute_aopc,
    compute_comprehensiveness,
    compute_sufficiency,
)
from cajt.runtime import autocast
from cajt.core.constants import EVAL_BATCH_SIZE


def sweep_sparsity_frontier(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels: list[int],
    n_points: int = 20,
) -> list[dict]:
    """Sweep JumpReLU threshold to trace the sparsity-fidelity frontier.

    Uses a log-uniform grid of n_points multipliers from 0.1 to 5.0,
    giving denser coverage of the interesting low-multiplier region.

    Temporarily scales log_threshold by adding log(multiplier), measures
    L0 + downstream loss at each point, then restores the original threshold.

    Returns list of {threshold_multiplier, mean_l0, delta_ce, kl_divergence,
    accuracy} sorted by mean_l0 ascending.
    """
    log_min, log_max = math.log(0.1), math.log(5.0)
    threshold_multipliers = tuple(
        round(math.exp(log_min + i * (log_max - log_min) / (n_points - 1)), 4)
        for i in range(n_points)
    )

    original_log_threshold = model.activation.log_threshold.data.clone()

    results = []
    for mult in threshold_multipliers:
        # Shift in log-space = multiply threshold
        model.activation.log_threshold.data = original_log_threshold + math.log(mult)

        # Measure L0 (mean active dims)
        total_active = 0.0
        n = 0
        for start in range(0, len(input_ids_list), EVAL_BATCH_SIZE):
            end = min(start + EVAL_BATCH_SIZE, len(input_ids_list))
            batch_ids = torch.cat(input_ids_list[start:end], dim=0)
            batch_mask = torch.cat(attention_mask_list[start:end], dim=0)
            with torch.inference_mode(), autocast():
                sparse_seq, *_ = model(batch_ids, batch_mask)
                sparse_vector = model.to_pooled(sparse_seq, batch_mask)
            total_active += (sparse_vector > 0).sum(dim=-1).float().sum().item()
            n += end - start
        mean_l0 = total_active / n if n > 0 else 0.0

        dl = compute_downstream_loss(model, input_ids_list, attention_mask_list, labels)

        results.append({
            "threshold_multiplier": mult,
            "mean_l0": mean_l0,
            "delta_ce": dl["delta_ce"],
            "kl_divergence": dl["kl_divergence"],
            "accuracy": dl["sparse_accuracy"],
        })

    # Restore original threshold
    model.activation.log_threshold.data = original_log_threshold

    results.sort(key=lambda x: x["mean_l0"])
    return results


def compute_naopc(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels: list[int],
    n_random_trials: int = 5,
) -> dict:
    """Normalized AOPC: AOPC(DLA) / AOPC(random attribution).

    Normalizes AOPC by the expected AOPC of random attributions,
    enabling fair comparison across models with different sparsity levels.
    """


    sparse_vectors, attributions, _, labels_t = collect_sparse_and_attributions(
        model, input_ids_list, attention_mask_list, labels,
    )

    # DLA AOPC
    dla_comp = compute_comprehensiveness(model, sparse_vectors, attributions, labels_t)
    dla_suff = compute_sufficiency(model, sparse_vectors, attributions, labels_t)
    dla_aopc_comp = compute_aopc(dla_comp)
    dla_aopc_suff = compute_aopc(dla_suff)

    # Random baseline AOPC (average over trials)
    random_aopc_comp_sum = 0.0
    random_aopc_suff_sum = 0.0
    for _ in range(n_random_trials):
        # Randomly permute attribution values per sample
        perm_attr = attributions.clone()
        for i in range(perm_attr.shape[0]):
            perm_attr[i] = perm_attr[i, torch.randperm(perm_attr.shape[1], device=perm_attr.device)]
        rand_comp = compute_comprehensiveness(model, sparse_vectors, perm_attr, labels_t)
        rand_suff = compute_sufficiency(model, sparse_vectors, perm_attr, labels_t)
        random_aopc_comp_sum += compute_aopc(rand_comp)
        random_aopc_suff_sum += compute_aopc(rand_suff)

    random_aopc_comp = random_aopc_comp_sum / n_random_trials
    random_aopc_suff = random_aopc_suff_sum / n_random_trials

    return {
        "naopc_comprehensiveness": dla_aopc_comp / max(random_aopc_comp, 1e-8),
        "naopc_sufficiency": dla_aopc_suff / max(random_aopc_suff, 1e-8),
        "dla_aopc_comp": dla_aopc_comp,
        "dla_aopc_suff": dla_aopc_suff,
        "random_aopc_comp": random_aopc_comp,
        "random_aopc_suff": random_aopc_suff,
    }
