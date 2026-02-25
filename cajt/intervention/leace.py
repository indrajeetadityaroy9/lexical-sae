"""LEACE concept erasure baseline for surgery comparison.

Uses the concept-erasure library (Belrose et al. 2023) for mathematically
correct Least-Squares Concept Erasure via covariance-based projection.

Reference: Belrose et al. "LEACE: Perfect linear concept erasure in closed form"
(NeurIPS 2023, arXiv:2306.03819).
"""

import torch
import torch.nn as nn
from concept_erasure import LeaceFitter
from torch.utils.data import DataLoader, TensorDataset

from cajt.core.types import CircuitState
from cajt.runtime import autocast, DEVICE


def fit_leace_eraser(
    model: nn.Module,
    tokenizer,
    texts: list[str],
    labels: list[int],
    max_length: int,
    batch_size: int = 32,
):
    """Fit a LEACE eraser on pooled sparse representations.

    Two-pass procedure:
      1. Collect [N, V] sparse vectors via inference.
      2. Fit LeaceFitter on (sparse_vectors, labels).

    Args:
        model: Trained LexicalSAE.
        tokenizer: HuggingFace tokenizer.
        texts: Training texts for fitting.
        labels: Corresponding class labels.
        max_length: Tokenizer max length.
        batch_size: Inference batch size.

    Returns:
        LeaceEraser object from the concept-erasure library.
    """


    encoding = tokenizer(
        texts, max_length=max_length, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    loader = DataLoader(
        TensorDataset(encoding["input_ids"], encoding["attention_mask"]),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
    )

    # Pass 1: collect sparse vectors
    all_sparse = []
    with torch.inference_mode(), autocast():
        for batch_ids, batch_mask in loader:
            batch_ids = batch_ids.to(DEVICE, non_blocking=True)
            batch_mask = batch_mask.to(DEVICE, non_blocking=True)
            sparse_seq, *_ = model(batch_ids, batch_mask)
            sparse_vector = model.to_pooled(sparse_seq, batch_mask)
            all_sparse.append(sparse_vector.float())

    sparse_all = torch.cat(all_sparse, dim=0)  # [N, V]
    labels_t = torch.tensor(labels, dtype=torch.long, device=DEVICE)

    # Pass 2: fit LEACE
    num_classes = len(set(labels))
    fitter = LeaceFitter(sparse_all.shape[1], num_classes, device=DEVICE, dtype=torch.float32)
    fitter.update(sparse_all, labels_t)
    return fitter.eraser


class LEACEWrappedModel(nn.Module):
    """Wraps LexicalSAE, applying LEACE erasure to pooled sparse vectors.

    forward() delegates to the underlying model (returns tuple).
    classify() applies LEACE erasure before the classifier head.
    """

    def __init__(self, model: nn.Module, eraser):
        super().__init__()
        self.model = model
        self.eraser = eraser

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def classify(self, sparse_sequence, attention_mask) -> CircuitState:
        sparse_vector = self.model.to_pooled(sparse_sequence, attention_mask)
        erased = self.eraser(sparse_vector.float()).to(sparse_vector.dtype)
        logits, W_eff, b_eff = self.model.classifier_forward(erased)
        return CircuitState(logits, erased, W_eff, b_eff)
