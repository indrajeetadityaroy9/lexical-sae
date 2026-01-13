"""
Evaluation metrics for SPLADE models.

Provides functions for computing:
- Classification accuracy
- F1 score
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
    threshold: float = 1e-2
) -> Tuple[float, float, float]:
    """
    Evaluates the model on the given loader.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): The data loader.
        device (torch.device): CPU or CUDA device.
        threshold (float): Threshold for considering a value as zero (for sparsity).

    Returns:
        tuple: (accuracy, f1, sparsity_percentage)
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
            preds = (torch.sigmoid(logits) > 0.5).float()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(y.numpy().flatten())

            # Sparsity check
            non_zeros += (torch.abs(sparse_vec) > threshold).sum().item()
            total_elements += sparse_vec.numel()

    accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item() if all_labels else 0.0
    f1 = f1_score(all_labels, all_preds) if all_labels else 0.0
    sparsity = 100 * (1 - non_zeros / total_elements) if total_elements > 0 else 0.0

    return accuracy, f1, sparsity


__all__ = ["evaluate"]
