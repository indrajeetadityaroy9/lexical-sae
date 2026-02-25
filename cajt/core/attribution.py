import torch


def compute_attribution_tensor(
    sparse_vector: torch.Tensor,
    W_eff: torch.Tensor,
    class_indices: torch.Tensor,
) -> torch.Tensor:
    """Core DLA: attr[b,j] = s[b,j] * W_eff[b, c_b, j].

    Args:
        sparse_vector: [B, V] sparse activations.
        W_eff: [B, C, V] effective weight matrix.
        class_indices: [B] per-sample class index.
    Returns:
        [B, V] attribution tensor.
    """
    batch_indices = torch.arange(sparse_vector.shape[0], device=sparse_vector.device)
    target_weights = W_eff[batch_indices, class_indices]
    return sparse_vector * target_weights
