"""Feature absorption detection for JumpReLU bottleneck.

Detects "B except A" features: dimensions that activate for concept B
but are suppressed when concept A co-occurs, indicating the JumpReLU
threshold has absorbed feature A into a composite feature.

Reference: Rajamanoharan et al. 2024 (arXiv:2407.14435).
"""

import torch

from cajt.evaluation.collect import collect_sparse_vectors


def detect_feature_absorption(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels: list[int],
    tokenizer,
    min_frequency: float = 0.01,
    min_co_occurrence: int = 10,
) -> dict:
    """Detect potential feature absorption in JumpReLU sparse representations.

    For each pair of frequently-active features (A, B), compares activation
    of feature A when B is present vs absent. Uses a self-calibrating z-test
    (2-sigma below population mean ratio) instead of a fixed threshold.

    Features are selected by frequency percentile (>= min_frequency of samples)
    rather than fixed top-k.

    Returns:
        {
            "absorption_pairs": list of dicts with feature_a, feature_b,
                activation_ratio, co_occurrence_count,
            "absorption_score": fraction of tested pairs showing absorption,
            "num_pairs_tested": int,
        }
    """
    sparse_vectors = collect_sparse_vectors(
        model, input_ids_list, attention_mask_list,
    )  # [N, V]
    n_samples = sparse_vectors.shape[0]

    # Frequency-percentile feature selection
    freqs = (sparse_vectors > 0).float().mean(dim=0)  # [V]
    active_features = (freqs >= min_frequency).nonzero(as_tuple=True)[0].cpu().tolist()

    # Collect all pairwise ratios first for z-test calibration
    all_ratios = []
    pair_data = []

    for i, feat_a in enumerate(active_features):
        for feat_b in active_features[i + 1:]:
            a_active = sparse_vectors[:, feat_a] > 0
            b_active = sparse_vectors[:, feat_b] > 0

            both_active = a_active & b_active
            a_only = a_active & ~b_active

            if both_active.sum().item() < min_co_occurrence or a_only.sum().item() < min_co_occurrence:
                continue

            mean_a_with_b = sparse_vectors[both_active, feat_a].mean().item()
            mean_a_without_b = sparse_vectors[a_only, feat_a].mean().item()
            ratio = mean_a_with_b / max(mean_a_without_b, 1e-8)

            all_ratios.append(ratio)
            pair_data.append((feat_a, feat_b, ratio, int(both_active.sum().item())))

    num_tested = len(pair_data)
    if num_tested == 0:
        return {"absorption_pairs": [], "absorption_score": 0.0, "num_pairs_tested": 0}

    # Self-calibrating z-test: absorption if ratio is 2-sigma below mean
    mean_ratio = sum(all_ratios) / len(all_ratios)
    std_ratio = (sum((r - mean_ratio) ** 2 for r in all_ratios) / len(all_ratios)) ** 0.5
    absorption_threshold = mean_ratio - 2 * std_ratio

    absorption_pairs = []
    for feat_a, feat_b, ratio, co_count in pair_data:
        if ratio < absorption_threshold:
            token_a = tokenizer.decode([feat_a]).strip() if feat_a < tokenizer.vocab_size else f"vpe_{feat_a}"
            token_b = tokenizer.decode([feat_b]).strip() if feat_b < tokenizer.vocab_size else f"vpe_{feat_b}"
            absorption_pairs.append({
                "feature_a": token_a,
                "feature_b": token_b,
                "feature_a_id": feat_a,
                "feature_b_id": feat_b,
                "activation_ratio": ratio,
                "co_occurrence_count": co_count,
            })

    return {
        "absorption_pairs": absorption_pairs,
        "absorption_score": len(absorption_pairs) / max(num_tested, 1),
        "num_pairs_tested": num_tested,
    }
