"""Feature absorption detection for JumpReLU bottleneck.

Detects "B except A" features: dimensions that activate for concept B
but are suppressed when concept A co-occurs, indicating the JumpReLU
threshold has absorbed feature A into a composite feature.

Reference: Rajamanoharan et al. 2024 (arXiv:2407.14435).
"""

import torch

from splade.mechanistic.attribution import compute_attribution_tensor
from splade.utils.cuda import COMPUTE_DTYPE, DEVICE, unwrap_compiled


def detect_feature_absorption(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels: list[int],
    tokenizer,
    top_k_features: int = 50,
    min_co_occurrence: int = 10,
) -> dict:
    """Detect potential feature absorption in JumpReLU sparse representations.

    For each pair of top-attributed features (A, B) for a given class,
    compares activation of feature A when B is present vs absent in
    the sparse representation. If ratio < 0.5, flags as absorption.

    Returns:
        {
            "absorption_pairs": list of dicts with feature_a, feature_b,
                activation_ratio, co_occurrence_count,
            "absorption_score": fraction of tested pairs showing absorption,
            "num_pairs_tested": int,
        }
    """
    _model = unwrap_compiled(model)
    labels_t = torch.tensor(labels, device=DEVICE)

    # Collect sparse vectors
    all_sparse = []
    for start in range(0, len(input_ids_list), 32):
        end = min(start + 32, len(input_ids_list))
        batch_ids = torch.cat(input_ids_list[start:end], dim=0)
        batch_mask = torch.cat(attention_mask_list[start:end], dim=0)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            sparse_seq, *_ = _model(batch_ids, batch_mask)
            sparse_vector = _model.to_pooled(sparse_seq, batch_mask)
        all_sparse.append(sparse_vector.float())
    sparse_vectors = torch.cat(all_sparse, dim=0)  # [N, V]

    # Find top-k most frequently active features across all samples
    active_counts = (sparse_vectors > 0).float().sum(dim=0)  # [V]
    _, top_indices = active_counts.topk(min(top_k_features, (active_counts > 0).sum().item()))
    top_indices = top_indices.cpu().tolist()

    absorption_pairs = []
    num_tested = 0

    for i, feat_a in enumerate(top_indices):
        for feat_b in top_indices[i + 1:]:
            a_active = sparse_vectors[:, feat_a] > 0  # [N]
            b_active = sparse_vectors[:, feat_b] > 0  # [N]

            both_active = a_active & b_active
            a_only = a_active & ~b_active

            if both_active.sum().item() < min_co_occurrence or a_only.sum().item() < min_co_occurrence:
                continue

            num_tested += 1

            # Mean activation of A when B is present vs absent
            mean_a_with_b = sparse_vectors[both_active, feat_a].mean().item()
            mean_a_without_b = sparse_vectors[a_only, feat_a].mean().item()

            ratio = mean_a_with_b / max(mean_a_without_b, 1e-8)

            if ratio < 0.5:
                token_a = tokenizer.decode([feat_a]).strip() if feat_a < tokenizer.vocab_size else f"vpe_{feat_a}"
                token_b = tokenizer.decode([feat_b]).strip() if feat_b < tokenizer.vocab_size else f"vpe_{feat_b}"
                absorption_pairs.append({
                    "feature_a": token_a,
                    "feature_b": token_b,
                    "feature_a_id": feat_a,
                    "feature_b_id": feat_b,
                    "activation_ratio": ratio,
                    "co_occurrence_count": int(both_active.sum().item()),
                })

    return {
        "absorption_pairs": absorption_pairs,
        "absorption_score": len(absorption_pairs) / max(num_tested, 1),
        "num_pairs_tested": num_tested,
    }
