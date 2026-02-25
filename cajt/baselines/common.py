"""Shared utilities for baseline model training.

Consolidates the duplicated training loop and probe construction
from sae_baseline.py and transcoder_baseline.py.
"""

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegressionCV
from torch.utils.data import DataLoader, TensorDataset

# Baseline comparison constants
BASELINE_OVERCOMPLETENESS = 4
BASELINE_LR = 1e-3
BASELINE_L1_COEFF = 1e-3
BASELINE_BATCH_SIZE = 64
_MAX_BASELINE_EPOCHS = 200
_BASELINE_PATIENCE = 10


def train_sparse_baseline(
    model: nn.Module,
    train_dataset: TensorDataset,
    lr: float = BASELINE_LR,
    l1_coeff: float = BASELINE_L1_COEFF,
    batch_size: int = BASELINE_BATCH_SIZE,
    max_epochs: int = _MAX_BASELINE_EPOCHS,
    patience: int = _BASELINE_PATIENCE,
) -> None:
    """Shared training loop for SAE and Transcoder baselines.

    Trains `model` in-place with MSE + L1 loss, Adam optimizer, and early stopping.
    The model must return (reconstruction, hidden_activations) from forward().

    Args:
        model: nn.Module on DEVICE with forward() -> (recon, hidden)
        train_dataset: TensorDataset with 1 tensor (SAE) or 2 tensors (Transcoder)
        lr: Learning rate
        l1_coeff: L1 regularization coefficient on hidden activations
        batch_size: Training batch size
        max_epochs: Maximum number of training epochs
        patience: Early stopping patience
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    model.train()
    for _ in range(max_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            # SAE: batch = (input,), Transcoder: batch = (input, target)
            inputs = batch[0]
            targets = batch[-1]  # same as inputs for SAE, different for Transcoder

            optimizer.zero_grad()
            reconstruction, hidden = model(inputs)
            recon_loss = nn.functional.mse_loss(reconstruction, targets)
            l1_loss = hidden.abs().mean()
            loss = recon_loss + l1_coeff * l1_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()


def build_logistic_probe(random_state: int = 42) -> LogisticRegressionCV:
    """Shared LogisticRegressionCV factory for sparse probing and disentanglement."""
    return LogisticRegressionCV(
        penalty="l1", solver="saga", Cs=10, cv=5,
        max_iter=2000, random_state=random_state,
    )
