"""MIB benchmark circuit metrics: CPR and CMD.

CPR (Circuit Performance Recovery): fraction of model performance
retained by the circuit alone.

CMD (Circuit Minimality Distance): how much smaller the circuit is
compared to the full model while maintaining performance.

Reference: MIB Benchmark (ICML 2025).
"""

import torch

from splade.evaluation.eraser import _get_topk_mask
from splade.mechanistic.attribution import compute_attribution_tensor
from splade.utils.cuda import COMPUTE_DTYPE, DEVICE, unwrap_compiled


def compute_cpr(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels: list[int],
    circuit_fraction: float = 0.1,
) -> dict:
    """Circuit Performance Recovery = accuracy(circuit_only) / accuracy(full_model).

    Keeps only the top circuit_fraction of attributed features, zeros the rest,
    and measures classification accuracy of the circuit-only model.

    Returns:
        {
            "cpr": float,
            "full_accuracy": float,
            "circuit_accuracy": float,
            "circuit_fraction": float,
        }
    """
    _model = unwrap_compiled(model)
    labels_t = torch.tensor(labels, device=DEVICE)

    all_sparse = []
    all_attr = []
    full_correct = 0

    for start in range(0, len(input_ids_list), 32):
        end = min(start + 32, len(input_ids_list))
        batch_ids = torch.cat(input_ids_list[start:end], dim=0)
        batch_mask = torch.cat(attention_mask_list[start:end], dim=0)
        batch_labels = labels_t[start:end]

        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            sparse_seq, *_ = _model(batch_ids, batch_mask)
            logits, sparse_vector, W_eff, _ = _model.classify(sparse_seq, batch_mask)

        full_correct += (logits.argmax(dim=-1) == batch_labels).sum().item()
        attr = compute_attribution_tensor(sparse_vector, W_eff, batch_labels)
        all_sparse.append(sparse_vector.float())
        all_attr.append(attr.float())

    sparse_vectors = torch.cat(all_sparse, dim=0)
    attributions = torch.cat(all_attr, dim=0)
    full_accuracy = full_correct / len(labels) if labels else 0.0

    # Keep only top circuit_fraction of attributed features
    top_mask = _get_topk_mask(attributions, circuit_fraction)
    circuit_sparse = sparse_vectors.clone()
    circuit_sparse[~top_mask] = 0.0

    with torch.inference_mode():
        circuit_logits = _model.classifier_logits_only(circuit_sparse)
    circuit_correct = (circuit_logits.argmax(dim=-1) == labels_t).sum().item()
    circuit_accuracy = circuit_correct / len(labels) if labels else 0.0

    return {
        "cpr": circuit_accuracy / max(full_accuracy, 1e-8),
        "full_accuracy": full_accuracy,
        "circuit_accuracy": circuit_accuracy,
        "circuit_fraction": circuit_fraction,
    }


def compute_cmd(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels: list[int],
    performance_threshold: float = 0.95,
) -> dict:
    """Circuit Minimality Distance: find smallest circuit meeting performance threshold.

    CMD = 1 - min_fraction (higher = more minimal circuit = better).

    Sweeps circuit_fraction from small to large, finds the smallest
    fraction where circuit accuracy >= threshold * full accuracy.

    Returns:
        {
            "cmd": float,
            "min_fraction": float,
            "performance_threshold": float,
            "sweep": list of {fraction, accuracy, cpr},
        }
    """
    fractions = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]

    sweep = []
    min_fraction = 1.0

    for frac in fractions:
        result = compute_cpr(model, input_ids_list, attention_mask_list, labels, circuit_fraction=frac)
        sweep.append({
            "fraction": frac,
            "accuracy": result["circuit_accuracy"],
            "cpr": result["cpr"],
        })
        if result["cpr"] >= performance_threshold and frac < min_fraction:
            min_fraction = frac

    return {
        "cmd": 1.0 - min_fraction,
        "min_fraction": min_fraction,
        "performance_threshold": performance_threshold,
        "sweep": sweep,
    }
