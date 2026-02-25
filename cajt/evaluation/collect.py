"""Shared batched inference utilities for evaluation modules.

Consolidates the ~10-line batched forward-pass pattern that was previously
duplicated across 11+ evaluation files into reusable functions.
"""

import torch

from cajt.core.attribution import compute_attribution_tensor
from cajt.core.constants import EVAL_BATCH_SIZE
from cajt.runtime import autocast, DEVICE


def prepare_mechanistic_inputs(
    tokenizer,
    texts: list[str],
    max_length: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Tokenize texts into per-sample tensor lists for mechanistic evaluation."""
    encoding = tokenizer(
        texts, max_length=max_length, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    input_ids_list = [
        encoding["input_ids"][i:i+1].to(DEVICE) for i in range(len(texts))
    ]
    attention_mask_list = [
        encoding["attention_mask"][i:i+1].to(DEVICE) for i in range(len(texts))
    ]
    return input_ids_list, attention_mask_list


def collect_sparse_vectors(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    batch_size: int = EVAL_BATCH_SIZE,
) -> torch.Tensor:
    """Batched inference returning [N, V] pooled sparse vectors.

    Used by sparse_probing, autointerp, feature_absorption, disentanglement, etc.
    """

    all_sparse = []

    for start in range(0, len(input_ids_list), batch_size):
        end = min(start + batch_size, len(input_ids_list))
        batch_ids = torch.cat(input_ids_list[start:end], dim=0)
        batch_mask = torch.cat(attention_mask_list[start:end], dim=0)

        with torch.inference_mode(), autocast():
            sparse_seq, *_ = model(batch_ids, batch_mask)
            cs = model.classify(sparse_seq, batch_mask)

        all_sparse.append(cs.sparse_vector.float())

    return torch.cat(all_sparse, dim=0)


def collect_sparse_and_attributions(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels: list[int],
    batch_size: int = EVAL_BATCH_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """Batched inference returning sparse vectors, attributions, accuracy, and labels tensor.

    Used by orchestrator (DLA verification), eraser, baselines, mib_metrics,
    sparsity_frontier, circuit_metrics, etc.

    Returns:
        sparse_vectors: [N, V] float tensor
        attributions: [N, V] float tensor (DLA attributions for true class)
        accuracy: float (model accuracy on these samples)
        labels_t: [N] long tensor on DEVICE
    """

    labels_t = torch.tensor(labels, device=DEVICE)
    all_sparse, all_attr = [], []
    correct = 0

    for start in range(0, len(input_ids_list), batch_size):
        end = min(start + batch_size, len(input_ids_list))
        batch_ids = torch.cat(input_ids_list[start:end], dim=0)
        batch_mask = torch.cat(attention_mask_list[start:end], dim=0)
        batch_labels = labels_t[start:end]

        with torch.inference_mode(), autocast():
            sparse_seq, *_ = model(batch_ids, batch_mask)
            logits, sparse_vector, W_eff, _ = model.classify(sparse_seq, batch_mask)

        correct += (logits.argmax(dim=-1) == batch_labels).sum().item()
        attr = compute_attribution_tensor(sparse_vector, W_eff, batch_labels)
        all_sparse.append(sparse_vector.float())
        all_attr.append(attr.float())

    sparse_vectors = torch.cat(all_sparse, dim=0)
    attributions = torch.cat(all_attr, dim=0)
    accuracy = correct / len(labels) if labels else 0.0
    return sparse_vectors, attributions, accuracy, labels_t
