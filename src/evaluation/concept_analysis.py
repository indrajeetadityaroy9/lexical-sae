"""Concept Vocabulary Bottleneck analysis metrics (arXiv:2412.07992).

Evaluates the quality of SPLADE's sparse concept vocabulary as an
interpretable intermediate representation using intervention-based
metrics from the Concept Bottleneck literature.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from src.models.classifier import SPLADEClassifier


def concept_intervention(
    clf: SPLADEClassifier,
    texts: list[str],
    labels: list[int],
    concept_indices: list[int],
    num_trials: int,
    seed: int,
) -> dict[str, float]:
    """Intervene on specific concepts and measure prediction change.

    Sets concept activations to zero and checks how predictions change.
    Higher change = concepts are more causally relevant.
    """
    rng = np.random.default_rng(seed)
    sparse = clf.transform(texts)  # [n, vocab]
    labels_arr = np.array(labels)

    # Baseline accuracy
    preds_orig = np.array(clf.predict(texts))
    acc_orig = (preds_orig == labels_arr).mean()

    # Intervene: zero out concept dimensions
    intervention_accs = []
    for _ in range(num_trials):
        # Subsample concepts to intervene on
        n_concepts = max(1, rng.integers(1, len(concept_indices) + 1))
        chosen = rng.choice(concept_indices, size=n_concepts, replace=False)

        sparse_intervened = sparse.copy()
        sparse_intervened[:, chosen] = 0.0

        # Predict using modified sparse vectors directly through classifier
        with torch.inference_mode():
            logits = clf.model.classifier(
                torch.tensor(sparse_intervened, dtype=torch.float32).to(
                    next(clf.model.classifier.parameters()).device
                )
            )
        preds = logits.argmax(dim=-1).cpu().numpy()
        intervention_accs.append((preds == labels_arr).mean())

    acc_intervened = float(np.mean(intervention_accs))
    return {
        "accuracy_original": float(acc_orig),
        "accuracy_intervened": float(acc_intervened),
        "accuracy_drop": float(acc_orig - acc_intervened),
    }


def concept_sufficiency(
    clf: SPLADEClassifier,
    texts: list[str],
    labels: list[int],
    top_k_values: list[int],
) -> dict[int, float]:
    """Can top-k concepts alone recover full-model accuracy?

    Masks all but top-k concept dimensions and evaluates accuracy.
    """
    sparse = clf.transform(texts)  # [n, vocab]
    labels_arr = np.array(labels)
    importance = np.abs(sparse).mean(axis=0)

    results = {}
    for k in top_k_values:
        top_idx = np.argsort(importance)[-k:]
        mask = np.zeros(sparse.shape[1])
        mask[top_idx] = 1.0
        sparse_masked = sparse * mask

        with torch.inference_mode():
            logits = clf.model.classifier(
                torch.tensor(sparse_masked, dtype=torch.float32).to(
                    next(clf.model.classifier.parameters()).device
                )
            )
        preds = logits.argmax(dim=-1).cpu().numpy()
        results[k] = float((preds == labels_arr).mean())

    return results


def concept_necessity(
    clf: SPLADEClassifier,
    texts: list[str],
    labels: list[int],
    top_k_values: list[int],
) -> dict[int, float]:
    """Does removing top-k concepts destroy accuracy?

    Masks top-k concept dimensions and evaluates accuracy drop.
    """
    sparse = clf.transform(texts)  # [n, vocab]
    labels_arr = np.array(labels)
    importance = np.abs(sparse).mean(axis=0)

    # Baseline accuracy with all concepts
    preds_orig = np.array(clf.predict(texts))
    acc_orig = (preds_orig == labels_arr).mean()

    results = {}
    for k in top_k_values:
        top_idx = np.argsort(importance)[-k:]
        sparse_ablated = sparse.copy()
        sparse_ablated[:, top_idx] = 0.0

        with torch.inference_mode():
            logits = clf.model.classifier(
                torch.tensor(sparse_ablated, dtype=torch.float32).to(
                    next(clf.model.classifier.parameters()).device
                )
            )
        preds = logits.argmax(dim=-1).cpu().numpy()
        acc_ablated = (preds == labels_arr).mean()
        results[k] = float(acc_orig - acc_ablated)

    return results
