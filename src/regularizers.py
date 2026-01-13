"""
Regularization losses for SPLADE sparse models.

Includes:
- FLOPS regularization (from SPLADE paper)
- Block L1 loss (Group Lasso for structured sparsity)
"""

import torch


def block_l1_loss(tensor, block_size=4):
    """
    Computes Group Lasso (Block L1) loss to encourage block sparsity.

    Args:
        tensor (torch.Tensor): The activation tensor [batch_size, feature_dim].
        block_size (int): The size of the contiguous blocks.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    batch_size, feature_dim = tensor.shape

    # Pad if feature_dim is not divisible by block_size
    if feature_dim % block_size != 0:
        pad_size = block_size - (feature_dim % block_size)
        tensor = torch.nn.functional.pad(tensor, (0, pad_size))
        feature_dim += pad_size

    # Reshape to [batch_size, num_blocks, block_size]
    num_blocks = feature_dim // block_size
    reshaped = tensor.view(batch_size, num_blocks, block_size)

    # Compute L2 norm of each block: sqrt(sum(x^2))
    # We add a small epsilon for numerical stability
    block_norms = torch.sqrt(torch.sum(reshaped ** 2, dim=2) + 1e-8)

    # Compute L1 norm of the block norms: sum(block_norms)
    loss = torch.sum(block_norms)

    return loss / batch_size


def flops_regularization(activations):
    """
    Computes the FLOPS regularization loss from the SPLADE paper.
    Encourages sparsity by minimizing the sum of squared mean activations per feature.

    L_FLOPS = sum_j (mean_i(w_ij))^2

    Note: Always uses PyTorch implementation since this is a loss function
    that requires gradient computation during training.

    Args:
        activations (torch.Tensor): The sparse activation tensor [batch_size, vocab_size].

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Always use PyTorch for loss computation (requires gradients)
    # 1. Compute mean activation for each term across the batch
    mean_activations = torch.mean(torch.abs(activations), dim=0)  # [vocab_size]

    # 2. Sum of squares
    loss = torch.sum(mean_activations ** 2)

    return loss
