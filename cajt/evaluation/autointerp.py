"""AutoInterp: offline feature interpretation scoring.

For each top-active feature (which IS a vocabulary token in Lexical-SAE),
checks whether the token name literally appears in the top-activating
examples. High overlap = the feature is interpretable as that token.
"""

import torch

from cajt.evaluation.collect import collect_sparse_vectors


def run_autointerp(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    min_frequency: float = 0.01,
    examples_per_feature: int = 10,
) -> dict:
    """Offline interpretability scoring for frequently-active features.

    For each feature activated in >= min_frequency of samples, finds the
    top-activating examples and checks if the feature's token name
    appears in those examples.

    Returns:
        {
            "mean_score": float,
            "per_feature": list of {feature_id, token_name, score, frequency},
        }
    """
    sparse_matrix = collect_sparse_vectors(
        model, input_ids_list, attention_mask_list,
    ).cpu()  # [N, V]

    n_samples = sparse_matrix.shape[0]

    # Frequency-percentile feature selection
    freqs = (sparse_matrix > 0).float().mean(dim=0)
    active_mask = freqs >= min_frequency
    if not active_mask.any():
        return {"mean_score": 0.0, "per_feature": []}

    top_feat_ids = active_mask.nonzero(as_tuple=True)[0]

    per_feature = []
    for feat_id in top_feat_ids.tolist():
        token_name = tokenizer.decode([feat_id]).strip().lower() if feat_id < tokenizer.vocab_size else f"vpe_{feat_id}"
        frequency = freqs[feat_id].item()

        # Get top-activating examples for this feature
        activations = sparse_matrix[:, feat_id]
        n_top = min(examples_per_feature, (activations > 0).sum().item())
        if n_top == 0:
            per_feature.append({"feature_id": feat_id, "token_name": token_name, "score": 0.0, "frequency": frequency})
            continue

        _, top_example_ids = activations.topk(n_top)

        # Check if token name appears in the top-activating examples
        matches = 0
        for idx in top_example_ids.tolist():
            if token_name in texts[idx].lower():
                matches += 1

        score = matches / n_top
        per_feature.append({
            "feature_id": feat_id,
            "token_name": token_name,
            "score": score,
            "frequency": frequency,
        })

    mean_score = sum(f["score"] for f in per_feature) / len(per_feature) if per_feature else 0.0

    return {
        "mean_score": mean_score,
        "per_feature": per_feature,
    }
