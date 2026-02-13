"""Sparse probing: linear probe accuracy on sparse bottleneck features.

Trains L1-regularized logistic regression on the sparse bottleneck
features to measure concept detection accuracy.
"""

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from splade.utils.cuda import COMPUTE_DTYPE, DEVICE, unwrap_compiled


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
    _model = unwrap_compiled(model)

    # Collect sparse vectors
    all_sparse = []
    correct = 0
    for start in range(0, len(input_ids_list), 32):
        end = min(start + 32, len(input_ids_list))
        batch_ids = torch.cat(input_ids_list[start:end], dim=0)
        batch_mask = torch.cat(attention_mask_list[start:end], dim=0)
        batch_labels = torch.tensor(labels[start:end], device=DEVICE)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            sparse_seq, *_ = _model(batch_ids, batch_mask)
            cs = _model.classify(sparse_seq, batch_mask)
            correct += (cs.logits.argmax(dim=-1) == batch_labels).sum().item()
        all_sparse.append(cs.sparse_vector.float().cpu())

    sparse_np = torch.cat(all_sparse, dim=0).numpy()
    labels_np = labels

    classifier_acc = correct / len(labels) if labels else 0.0

    # Train/test split
    n_train = int(len(labels) * train_fraction)
    X_train, X_test = sparse_np[:n_train], sparse_np[n_train:]
    y_train, y_test = labels_np[:n_train], labels_np[n_train:]

    if len(set(y_train)) < 2 or len(X_test) == 0:
        return {
            "probe_accuracy": 0.0,
            "probe_f1_macro": 0.0,
            "n_nonzero_weights": 0,
            "n_features_used": 0,
            "classifier_accuracy": classifier_acc,
        }

    clf = LogisticRegression(
        penalty="l1", solver="saga", C=1000.0,
        max_iter=1000, random_state=42,
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probe_acc = accuracy_score(y_test, preds) if len(y_test) > 0 else 0.0
    probe_f1 = f1_score(y_test, preds, average="macro", zero_division=0.0)

    import numpy as np
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
