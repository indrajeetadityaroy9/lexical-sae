"""AutoInterp: offline feature interpretation scoring.

For each top-active feature (which IS a vocabulary token in Lexical-SAE),
checks whether the token name literally appears in the top-activating
examples. High overlap = the feature is interpretable as that token.
"""

import torch

from splade.utils.cuda import COMPUTE_DTYPE, DEVICE, unwrap_compiled


def run_autointerp(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    labels: list[int],
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    top_k_features: int = 20,
    examples_per_feature: int = 10,
) -> dict:
    """Offline interpretability scoring for top features.

    For each of the top-k most frequently active features, finds the
    top-activating examples and checks if the feature's token name
    appears in those examples.

    Returns:
        {
            "mean_score": float,
            "per_feature": list of {feature_id, token_name, score, frequency},
        }
    """
    _model = unwrap_compiled(model)

    # Collect sparse vectors
    all_sparse = []
    for start in range(0, len(input_ids_list), 32):
        end = min(start + 32, len(input_ids_list))
        batch_ids = torch.cat(input_ids_list[start:end], dim=0)
        batch_mask = torch.cat(attention_mask_list[start:end], dim=0)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            sparse_seq, *_ = _model(batch_ids, batch_mask)
            sparse_vector = _model.to_pooled(sparse_seq, batch_mask)
        all_sparse.append(sparse_vector.float().cpu())
    sparse_matrix = torch.cat(all_sparse, dim=0)  # [N, V]

    n_samples = sparse_matrix.shape[0]

    # Find top-k most frequently active features
    active_counts = (sparse_matrix > 0).float().sum(dim=0)
    k = min(top_k_features, (active_counts > 0).sum().item())
    if k == 0:
        return {"mean_score": 0.0, "per_feature": []}

    _, top_feat_ids = active_counts.topk(k)

    per_feature = []
    for feat_id in top_feat_ids.tolist():
        token_name = tokenizer.decode([feat_id]).strip().lower() if feat_id < tokenizer.vocab_size else f"vpe_{feat_id}"
        frequency = active_counts[feat_id].item() / n_samples

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
