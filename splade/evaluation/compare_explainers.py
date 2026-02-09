"""Compare DLA against baseline explainers using ERASER metrics.

Runs all explainers on the same data, evaluates each with
comprehensiveness/sufficiency/AOPC, and records wall-clock time.
"""

import time

import torch

from splade.evaluation.baselines import (
    attention_attribution,
    dla_attribution,
    gradient_attribution,
    integrated_gradients_attribution,
)
from splade.evaluation.eraser import (
    compute_aopc,
    compute_comprehensiveness,
    compute_sufficiency,
)
from splade.utils.cuda import COMPUTE_DTYPE, DEVICE, unwrap_compiled


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
    _model = unwrap_compiled(model)
    eval_batch = 32
    labels_t = torch.tensor(labels, device=DEVICE)

    # Pre-compute sparse_vectors once (shared across explainers for ERASER eval)
    all_sparse = []
    for start in range(0, len(input_ids_list), eval_batch):
        end = min(start + eval_batch, len(input_ids_list))
        batch_ids = torch.cat(input_ids_list[start:end], dim=0)
        batch_mask = torch.cat(attention_mask_list[start:end], dim=0)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            sparse_seq = _model(batch_ids, batch_mask)
            sparse_vector = _model.to_pooled(sparse_seq, batch_mask)
        all_sparse.append(sparse_vector.float())
    sparse_vectors = torch.cat(all_sparse, dim=0)

    results = {}
    for name, explainer_fn in _EXPLAINERS.items():
        t0 = time.perf_counter()

        # Compute attributions
        all_attr = []
        for start in range(0, len(input_ids_list), eval_batch):
            end = min(start + eval_batch, len(input_ids_list))
            batch_ids = torch.cat(input_ids_list[start:end], dim=0)
            batch_mask = torch.cat(attention_mask_list[start:end], dim=0)
            batch_labels = labels_t[start:end]
            attr = explainer_fn(_model, batch_ids, batch_mask, batch_labels)
            all_attr.append(attr.float())
        attributions = torch.cat(all_attr, dim=0)

        elapsed = time.perf_counter() - t0

        # ERASER evaluation using shared sparse_vectors
        comp = compute_comprehensiveness(_model, sparse_vectors, attributions, labels_t)
        suff = compute_sufficiency(_model, sparse_vectors, attributions, labels_t)

        results[name] = {
            "comprehensiveness": comp,
            "sufficiency": suff,
            "aopc_comp": compute_aopc(comp),
            "aopc_suff": compute_aopc(suff),
            "time_seconds": elapsed,
        }

    return results
