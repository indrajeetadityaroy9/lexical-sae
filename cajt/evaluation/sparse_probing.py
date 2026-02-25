"""Sparse probing: linear probe accuracy on sparse bottleneck features.

Trains L1-regularized logistic regression on the sparse bottleneck
features to measure concept detection accuracy.
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from cajt.baselines.common import build_logistic_probe
from cajt.evaluation.collect import collect_sparse_and_attributions


def run_sparse_probing(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels: list[int],
    num_classes: int,
    train_fraction: float = 0.8,
) -> dict:
    """Train sparse linear probe on bottleneck features, report accuracy.

    Returns:
        {
            "probe_accuracy": float,
            "probe_f1_macro": float,
            "n_nonzero_weights": int,
            "n_features_used": int,
            "classifier_accuracy": float,
        }
    """
    sparse_vectors, _, classifier_acc, _ = collect_sparse_and_attributions(
        model, input_ids_list, attention_mask_list, labels,
    )
    sparse_np = sparse_vectors.cpu().numpy()

    # Train/test split
    n_train = int(len(labels) * train_fraction)
    X_train, X_test = sparse_np[:n_train], sparse_np[n_train:]
    y_train, y_test = labels[:n_train], labels[n_train:]

    if len(set(y_train)) < 2 or len(X_test) == 0:
        return {
            "probe_accuracy": 0.0,
            "probe_f1_macro": 0.0,
            "n_nonzero_weights": 0,
            "n_features_used": 0,
            "classifier_accuracy": classifier_acc,
        }

    clf = build_logistic_probe()
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probe_acc = accuracy_score(y_test, preds) if len(y_test) > 0 else 0.0
    probe_f1 = f1_score(y_test, preds, average="macro", zero_division=0.0)

    coef = clf.coef_
    n_nonzero = int(np.count_nonzero(coef))
    n_features_used = int((np.abs(coef).sum(axis=0) > 0).sum())

    return {
        "probe_accuracy": float(probe_acc),
        "probe_f1_macro": float(probe_f1),
        "n_nonzero_weights": n_nonzero,
        "n_features_used": n_features_used,
        "classifier_accuracy": classifier_acc,
    }
