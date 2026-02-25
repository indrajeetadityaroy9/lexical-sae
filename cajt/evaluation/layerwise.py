"""Layerwise attribution decomposition through BERT.

Decomposes DLA to identify which BERT layers produce the vocabulary
dimensions that drive classification. Inspired by AtP* (Kramar 2024).

Approach: Zero-ablation. For each layer l, replace its output with
its input (removing the layer's residual contribution), then measure
the change in sparse_vector. This gives each layer's contribution
to the final sparse representation without linearity assumptions.

Composes with W_eff for end-to-end attribution:
    end_to_end[b, l, j] = contribution_l[b, j] * W_eff[b, c_b, j]
    layer_importance[b, l] = sum_j |end_to_end[b, l, j]|
"""

import torch

from cajt.runtime import autocast, DEVICE


_LAYERWISE_SAMPLES = 100


def get_transformer_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Get the list of transformer layers from the ModernBERT encoder."""
    return list(model.encoder.layers)


def decompose_sparse_vector_by_layer(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Decompose sparse_vector contributions by BERT layer.

    For each layer l, ablates the layer's residual contribution and
    measures the change in sparse_vector.

    Args:
        model: LexicalSAE.
        input_ids: [B, L] token IDs.
        attention_mask: [B, L] attention mask.

    Returns:
        [B, num_layers, V] per-layer contributions to sparse_vector.
    """

    layers = get_transformer_layers(model)
    num_layers = len(layers)

    # Clean forward pass
    with torch.inference_mode(), autocast():
        sparse_seq_clean, *_ = model(input_ids, attention_mask)
        sparse_clean = model.to_pooled(sparse_seq_clean, attention_mask)
    sparse_clean = sparse_clean.float()

    V = sparse_clean.shape[1]
    B = sparse_clean.shape[0]
    contributions = torch.zeros(B, num_layers, V, device=sparse_clean.device)

    # For each layer, ablate its contribution
    for layer_idx in range(num_layers):
        def _zero_layer_hook(module, input, output):
            return (input[0],) + output[1:]

        hook_handle = layers[layer_idx].register_forward_hook(_zero_layer_hook)
        try:
            with torch.inference_mode(), autocast():
                sparse_seq_ablated, *_ = model(input_ids, attention_mask)
                sparse_ablated = model.to_pooled(sparse_seq_ablated, attention_mask)
            sparse_ablated = sparse_ablated.float()
            contributions[:, layer_idx, :] = sparse_clean - sparse_ablated
        finally:
            hook_handle.remove()

    return contributions


def compute_layerwise_attribution(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_classes: torch.Tensor,
) -> torch.Tensor:
    """End-to-end layerwise attribution: which layers drive classification.

    Composes layer decomposition with W_eff:
        end_to_end[b, l, j] = contribution_l[b, j] * W_eff[b, c_b, j]
        layer_importance[b, l] = sum_j |end_to_end[b, l, j]|

    Args:
        model: LexicalSAE.
        input_ids: [B, L] token IDs.
        attention_mask: [B, L] attention mask.
        target_classes: [B] class indices.

    Returns:
        [B, num_layers] layer importance scores.
    """


    # Get layer contributions [B, num_layers, V]
    contributions = decompose_sparse_vector_by_layer(model, input_ids, attention_mask)

    # Get W_eff for the target class
    with torch.inference_mode(), autocast():
        sparse_seq, *_ = model(input_ids, attention_mask)
        W_eff = model.classify(sparse_seq, attention_mask).W_eff

    device = W_eff.device
    batch_indices = torch.arange(len(target_classes), device=device)
    target_weights = W_eff[batch_indices, target_classes].float()  # [B, V]

    # Compose: [B, num_layers, V] * [B, 1, V] -> sum over V
    end_to_end = contributions * target_weights.unsqueeze(1)  # [B, num_layers, V]
    layer_importance = end_to_end.abs().sum(dim=-1)  # [B, num_layers]

    return layer_importance


def run_layerwise_evaluation(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels: list[int],
) -> dict:
    """Run layerwise attribution on a subset of test examples.

    Returns:
        {"layer_importance": list of per-layer mean importance scores,
         "decomposition_error": mean relative error of layer sum vs full sparse_vector}
    """

    n = min(_LAYERWISE_SAMPLES, len(input_ids_list))

    all_importance = []
    decomposition_errors = []

    eval_batch = 8  # smaller batch for layerwise (multiple BERT passes per sample)
    for start in range(0, n, eval_batch):
        end = min(start + eval_batch, n)
        batch_ids = torch.cat(input_ids_list[start:end], dim=0)
        batch_mask = torch.cat(attention_mask_list[start:end], dim=0)
        batch_labels = torch.tensor(labels[start:end], device=DEVICE)

        # Layer importance
        importance = compute_layerwise_attribution(
            model, batch_ids, batch_mask, batch_labels,
        )
        all_importance.append(importance.cpu())

        # Decomposition error: sum of layer contributions vs actual sparse_vector
        contributions = decompose_sparse_vector_by_layer(
            model, batch_ids, batch_mask,
        )
        reconstructed = contributions.sum(dim=1)  # [B, V]

        with torch.inference_mode(), autocast():
            sparse_seq_actual, *_ = model(batch_ids, batch_mask)
            sparse_actual = model.to_pooled(sparse_seq_actual, batch_mask)
        sparse_actual = sparse_actual.float()

        error = (sparse_actual - reconstructed).abs().sum(dim=-1)
        norm = sparse_actual.abs().sum(dim=-1).clamp(min=1e-8)
        decomposition_errors.append((error / norm).cpu())

    all_importance = torch.cat(all_importance, dim=0)  # [n, num_layers]
    mean_importance = all_importance.mean(dim=0).tolist()

    all_errors = torch.cat(decomposition_errors, dim=0)
    mean_error = float(all_errors.mean().item())

    return {
        "layer_importance": mean_importance,
        "decomposition_error": mean_error,
        "num_samples": n,
    }
