"""Downstream loss: CE increase and KL divergence from sparse bottleneck.

The SAE community's primary evaluation metric. Measures how much information
the JumpReLU sparse bottleneck destroys by comparing classifier logits
with vs without the bottleneck active.

Dense path: MLM logits -> ReLU (non-negativity only, no sparsity) -> pool -> classify
Sparse path: MLM logits -> JumpReLU (sparse gating) -> pool -> classify (normal)
"""

import torch
import torch.nn.functional as F

from cajt.core.constants import EVAL_BATCH_SIZE
from cajt.runtime import autocast, DEVICE


def compute_downstream_loss(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels: list[int],
    batch_size: int = EVAL_BATCH_SIZE,
) -> dict:
    """Compare classifier output with vs without JumpReLU bottleneck.

    Returns:
        {
            "delta_ce": float,          CE(sparse) - CE(dense)
            "kl_divergence": float,     KL(dense || sparse)
            "dense_ce": float,
            "sparse_ce": float,
            "dense_accuracy": float,
            "sparse_accuracy": float,
        }
    """

    labels_t = torch.tensor(labels, device=DEVICE)

    all_dense_logits = []
    all_sparse_logits = []

    for start in range(0, len(input_ids_list), batch_size):
        end = min(start + batch_size, len(input_ids_list))
        batch_ids = torch.cat(input_ids_list[start:end], dim=0)
        batch_mask = torch.cat(attention_mask_list[start:end], dim=0)

        with torch.inference_mode(), autocast():
            # Dense path: bypass JumpReLU, use ReLU for non-negativity only
            mlm_logits = model.backbone_forward(
                batch_ids, batch_mask,
            ).logits

            if model.virtual_expander is not None and model._captured_hidden is not None:
                mlm_logits = model.virtual_expander(model._captured_hidden, mlm_logits)

            dense_seq = torch.relu(mlm_logits)
            dense_seq = dense_seq.masked_fill(~batch_mask.unsqueeze(-1).bool(), 0.0)
            if model.attention_pool is not None:
                dense_pooled = model.attention_pool(dense_seq, batch_mask)
            else:
                dense_pooled = dense_seq.max(dim=1).values
            dense_logits = model.classifier_logits_only(dense_pooled)

            # Sparse path: normal forward through JumpReLU
            sparse_seq, *_ = model(batch_ids, batch_mask)
            cs = model.classify(sparse_seq, batch_mask)
            sparse_logits = cs.logits

        all_dense_logits.append(dense_logits.float())
        all_sparse_logits.append(sparse_logits.float())

    dense_logits = torch.cat(all_dense_logits, dim=0)
    sparse_logits = torch.cat(all_sparse_logits, dim=0)

    dense_ce = F.cross_entropy(dense_logits, labels_t).item()
    sparse_ce = F.cross_entropy(sparse_logits, labels_t).item()

    dense_probs = F.softmax(dense_logits, dim=-1)
    sparse_log_probs = F.log_softmax(sparse_logits, dim=-1)
    kl = F.kl_div(sparse_log_probs, dense_probs, reduction="batchmean").item()

    dense_acc = (dense_logits.argmax(dim=-1) == labels_t).float().mean().item()
    sparse_acc = (sparse_logits.argmax(dim=-1) == labels_t).float().mean().item()

    return {
        "delta_ce": sparse_ce - dense_ce,
        "kl_divergence": kl,
        "dense_ce": dense_ce,
        "sparse_ce": sparse_ce,
        "dense_accuracy": dense_acc,
        "sparse_accuracy": sparse_acc,
    }
