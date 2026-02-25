"""SCR and TPP disentanglement metrics from SAEBench.

SCR (Spurious Correlation Removal): measures whether zeroing a spurious
feature removes the spurious correlation from the representation.

TPP (Targeted Probe Perturbation): measures whether perturbing a feature
affects probes for the target concept more than probes for other concepts.

Reference: SAEBench (2025, arXiv:2503.09532).
"""

import numpy as np
import torch

from cajt.baselines.common import build_logistic_probe
from cajt.core.constants import CIRCUIT_MASS_FRACTION
from cajt.evaluation.collect import collect_sparse_vectors


def compute_scr(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels: list[int],
    spurious_token_ids: list[int],
) -> dict:
    """Spurious Correlation Removal.

    Measures correlation between spurious features and labels before/after
    zeroing those features. Lower correlation_after = better disentanglement.

    Args:
        spurious_token_ids: vocabulary indices of known spurious features.

    Returns:
        {
            "scr_score": float,  correlation_before - correlation_after
            "correlation_before": float,
            "correlation_after": float,
        }
    """
    sparse_np = collect_sparse_vectors(
        model, input_ids_list, attention_mask_list,
    ).cpu().numpy()
    labels_np = np.array(labels)

    if len(set(labels)) < 2:
        return {"scr_score": 0.0, "correlation_before": 0.0, "correlation_after": 0.0}

    # Probe on spurious features only
    spurious_features = sparse_np[:, spurious_token_ids]

    def _probe_accuracy(features):
        if features.shape[1] == 0:
            return 0.5
        n_train = int(0.8 * len(labels))
        clf = build_logistic_probe()
        clf.fit(features[:n_train], labels_np[:n_train])
        return float(clf.score(features[n_train:], labels_np[n_train:]))

    corr_before = _probe_accuracy(spurious_features)

    # Zero spurious features and re-probe on full modified representation
    modified = sparse_np.copy()
    modified[:, spurious_token_ids] = 0.0
    corr_after = _probe_accuracy(modified)

    return {
        "scr_score": corr_before - corr_after,
        "correlation_before": corr_before,
        "correlation_after": corr_after,
    }


def compute_tpp(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels: list[int],
    num_classes: int,
    mass_fraction: float = CIRCUIT_MASS_FRACTION,
) -> dict:
    """Targeted Probe Perturbation.

    For each of the top-n most class-discriminative features:
    1. Train a probe to predict labels from all features
    2. Zero the target feature
    3. Measure accuracy drop

    Good disentanglement = zeroing one feature doesn't collapse
    probes for unrelated concepts.

    Returns:
        {
            "tpp_score": float,  mean accuracy drop from single-feature perturbation
            "per_feature": list of {feature_id, accuracy_full, accuracy_perturbed, drop},
        }
    """
    sparse_np = collect_sparse_vectors(
        model, input_ids_list, attention_mask_list,
    ).cpu().numpy()
    labels_np = np.array(labels)

    if len(set(labels)) < 2:
        return {"tpp_score": 0.0, "per_feature": []}

    n_train = int(0.8 * len(labels))

    # Train full probe with cross-validated C
    clf_full = build_logistic_probe()
    clf_full.fit(sparse_np[:n_train], labels_np[:n_train])
    full_acc = float(clf_full.score(sparse_np[n_train:], labels_np[n_train:]))

    # Coefficient-mass selection: features capturing mass_fraction of total coef mass
    coef_abs = np.abs(clf_full.coef_).sum(axis=0)
    sorted_idx = np.argsort(coef_abs)[::-1]
    cumsum = np.cumsum(coef_abs[sorted_idx])
    total = coef_abs.sum()
    if total < 1e-12:
        return {"tpp_score": 0.0, "per_feature": []}
    n_target = int(np.searchsorted(cumsum, mass_fraction * total)) + 1
    top_features = sorted_idx[:n_target]

    per_feature = []
    for feat_id in top_features:
        # Zero this feature and re-evaluate
        modified = sparse_np.copy()
        modified[:, feat_id] = 0.0
        perturbed_acc = float(clf_full.score(modified[n_train:], labels_np[n_train:]))
        drop = full_acc - perturbed_acc

        per_feature.append({
            "feature_id": int(feat_id),
            "accuracy_full": full_acc,
            "accuracy_perturbed": perturbed_acc,
            "drop": drop,
        })

    tpp_score = np.mean([f["drop"] for f in per_feature]) if per_feature else 0.0

    return {
        "tpp_score": float(tpp_score),
        "per_feature": per_feature,
    }
