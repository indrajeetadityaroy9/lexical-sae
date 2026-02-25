"""MIB benchmark circuit metrics: CPR and CMD.

CPR (Circuit Performance Recovery): fraction of model performance
retained by the circuit alone.

CMD (Circuit Minimality Distance): how much smaller the circuit is
compared to the full model while maintaining performance.

Reference: MIB Benchmark (ICML 2025).
"""

import torch

from cajt.core.types import circuit_mask_by_mass
from cajt.evaluation.collect import collect_sparse_and_attributions
from cajt.core.constants import CIRCUIT_MASS_FRACTION


def compute_cpr(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels: list[int],
    mass_fraction: float = CIRCUIT_MASS_FRACTION,
) -> dict:
    """Circuit Performance Recovery = accuracy(circuit_only) / accuracy(full_model).

    Uses cumulative attribution mass to select circuit features.

    Returns:
        {
            "cpr": float,
            "full_accuracy": float,
            "circuit_accuracy": float,
            "mass_fraction": float,
        }
    """

    sparse_vectors, attributions, full_accuracy, labels_t = collect_sparse_and_attributions(
        model, input_ids_list, attention_mask_list, labels,
    )

    # Mass-based circuit selection
    mass_mask = circuit_mask_by_mass(attributions, mass_fraction)
    circuit_sparse = sparse_vectors * mass_mask

    with torch.inference_mode():
        circuit_logits = model.classifier_logits_only(circuit_sparse)
    circuit_correct = (circuit_logits.argmax(dim=-1) == labels_t).sum().item()
    circuit_accuracy = circuit_correct / len(labels) if labels else 0.0

    return {
        "cpr": circuit_accuracy / max(full_accuracy, 1e-8),
        "full_accuracy": full_accuracy,
        "circuit_accuracy": circuit_accuracy,
        "mass_fraction": mass_fraction,
    }


def compute_cmd(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels: list[int],
    performance_threshold: float = 0.95,
    precision: float = 0.01,
) -> dict:
    """Circuit Minimality Distance via binary search over mass fractions.

    CMD = 1 - min_mass_fraction (higher = more minimal circuit = better).

    Binary search finds the smallest mass_fraction where
    circuit accuracy >= threshold * full accuracy. ~7 iterations for 1% precision.

    Returns:
        {
            "cmd": float,
            "min_mass_fraction": float,
            "performance_threshold": float,
        }
    """

    sparse_vectors, attributions, full_accuracy, labels_t = collect_sparse_and_attributions(
        model, input_ids_list, attention_mask_list, labels,
    )

    def _cpr_at_mass(mf: float) -> float:
        mask = circuit_mask_by_mass(attributions, mf)
        circuit_sparse = sparse_vectors * mask
        with torch.inference_mode():
            logits = model.classifier_logits_only(circuit_sparse)
        acc = (logits.argmax(dim=-1) == labels_t).float().mean().item()
        return acc / max(full_accuracy, 1e-8)

    lo, hi = 0.0, 1.0
    while hi - lo > precision:
        mid = (lo + hi) / 2
        if _cpr_at_mass(mid) >= performance_threshold:
            hi = mid
        else:
            lo = mid

    return {
        "cmd": 1.0 - hi,
        "min_mass_fraction": hi,
        "performance_threshold": performance_threshold,
    }
