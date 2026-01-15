"""
Evaluation metrics for SPLADE models.

Provides functions for computing:
- Classification accuracy
- F1 score (binary and multi-class)
- Sparsity percentage
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from typing import Tuple


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 1e-2,
    num_labels: int = 1,
) -> Tuple[float, float, float]:
    """
    Evaluates the model on the given loader.

    Supports both binary classification (num_labels=1) and multi-class
    classification (num_labels > 1).

    Args:
        model: The model to evaluate.
        loader: The data loader.
        device: CPU or CUDA device.
        threshold: Threshold for considering a value as zero (for sparsity).
        num_labels: Number of output labels (1 for binary, >1 for multi-class).

    Returns:
        tuple: (accuracy, f1, sparsity_percentage)

    Example:
        # Binary classification
        acc, f1, sparsity = evaluate(model, loader, device, num_labels=1)

        # Multi-class classification (4 classes)
        acc, f1, sparsity = evaluate(model, loader, device, num_labels=4)
    """
    model.eval()
    all_preds = []
    all_labels = []
    non_zeros = 0
    total_elements = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            y = batch['label']

            logits, sparse_vec = model(input_ids, attention_mask)

            if num_labels == 1:
                # Binary classification: sigmoid + threshold
                preds = (torch.sigmoid(logits) > 0.5).float().squeeze(-1)
            else:
                # Multi-class classification: argmax
                preds = torch.argmax(logits, dim=-1).float()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(y.numpy().flatten())

            # Sparsity check
            non_zeros += (torch.abs(sparse_vec) > threshold).sum().item()
            total_elements += sparse_vec.numel()

    # Compute accuracy
    if all_labels:
        accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
    else:
        accuracy = 0.0

    # Compute F1 score
    if all_labels:
        if num_labels == 1:
            # Binary classification: standard F1
            f1 = f1_score(all_labels, all_preds, zero_division=0)
        else:
            # Multi-class classification: macro-averaged F1
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    else:
        f1 = 0.0

    # Compute sparsity
    sparsity = 100 * (1 - non_zeros / total_elements) if total_elements > 0 else 0.0

    return accuracy, f1, sparsity


__all__ = ["evaluate"]
