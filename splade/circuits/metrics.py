"""Circuit evaluation metrics.

Circuit extraction uses fraction of ACTIVE (non-zero) vocabulary dimensions,
reflecting the sparse bottleneck where only ~100 of ~30K dims are active.
Cosine separation is the primary metric (matching training). Jaccard is
retained as a supplementary diagnostic.
"""

from dataclasses import dataclass, field

import torch

from splade.mechanistic.attribution import compute_attribution_tensor
from splade.utils.cuda import COMPUTE_DTYPE, DEVICE, unwrap_compiled


def _id_to_name(tid: int, tokenizer, model: torch.nn.Module) -> str:
    """Map token/virtual-slot ID to a human-readable name."""
    _orig = unwrap_compiled(model)
    V = _orig.vocab_size
    if tid < V:
        return tokenizer.convert_ids_to_tokens([tid])[0]
    if _orig.virtual_expander is not None:
        vpe = _orig.virtual_expander
        offset = tid - V
        M = vpe.num_senses - 1
        parent_idx = offset // M
        sense = offset % M + 1
        parent_tid = vpe.token_ids[parent_idx]
        parent_name = tokenizer.convert_ids_to_tokens([parent_tid])[0]
        return f"{parent_name}@sense{sense}"
    return f"[virtual_{tid}]"


@dataclass
class VocabularyCircuit:
    class_idx: int
    token_ids: list[int] = field(default_factory=list)
    token_names: list[str] = field(default_factory=list)
    attribution_scores: list[float] = field(default_factory=list)
    completeness_score: float = 0.0
    total_attribution: float = 0.0


def extract_vocabulary_circuit(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    tokenizer,
    class_idx: int,
    circuit_fraction: float = 0.1,
    precomputed_attributions: torch.Tensor | None = None,
) -> VocabularyCircuit:
    """Extract vocabulary circuit as the top fraction of ACTIVE dimensions.

    Selects the top circuit_fraction of non-zero attribution dimensions,
    not total vocab. With ~100 active dims and circuit_fraction=0.1,
    this yields ~10 tokens — a meaningful interpretable subset.
    """
    _model = unwrap_compiled(model)

    if precomputed_attributions is not None:
        mean_attributions = precomputed_attributions
    else:
        vocab_size = _model.vocab_size_expanded
        token_attribution_sums = torch.zeros(vocab_size, device=DEVICE)
        n_examples = 0
        eval_batch = 32

        for start in range(0, len(input_ids_list), eval_batch):
            end = min(start + eval_batch, len(input_ids_list))
            batch_ids = torch.cat(input_ids_list[start:end], dim=0)
            batch_mask = torch.cat(attention_mask_list[start:end], dim=0)
            bs = end - start

            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
                sparse_seq, *_ = _model(batch_ids, batch_mask)
                _, sparse_vector, W_eff, _ = _model.classify(sparse_seq, batch_mask)

            class_labels_t = torch.full((bs,), class_idx, device=DEVICE)
            attr = compute_attribution_tensor(sparse_vector, W_eff, class_labels_t)
            token_attribution_sums += attr.abs().sum(dim=0)
            n_examples += bs

        if n_examples == 0:
            return VocabularyCircuit(class_idx=class_idx)

        mean_attributions = token_attribution_sums / n_examples

    total_mass = mean_attributions.sum()
    if total_mass < 1e-12:
        return VocabularyCircuit(class_idx=class_idx)

    # Fraction of ACTIVE dims (not total vocab) — only ~100-200 of ~30K
    # dims are non-zero due to sparse bottleneck, so fraction of total vocab
    # would trivially select all active dims.
    active_mask = mean_attributions > 0
    n_active = int(active_mask.sum().item())
    if n_active == 0:
        return VocabularyCircuit(class_idx=class_idx)
    k = max(1, int(circuit_fraction * n_active))
    _, top_indices = mean_attributions.topk(min(k, n_active))
    circuit_token_ids = top_indices

    sorted_indices = torch.argsort(mean_attributions[circuit_token_ids], descending=True)
    circuit_token_ids = circuit_token_ids[sorted_indices].tolist()

    token_names = [_id_to_name(tid, tokenizer, model) for tid in circuit_token_ids]
    scores = [float(mean_attributions[tid].item()) for tid in circuit_token_ids]

    return VocabularyCircuit(
        class_idx=class_idx,
        token_ids=circuit_token_ids,
        token_names=token_names,
        attribution_scores=scores,
        total_attribution=float(total_mass.item()),
    )


def measure_circuit_completeness(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels: list[int],
    circuit: VocabularyCircuit,
) -> float:
    """Per-class accuracy retention when keeping only circuit tokens.

    Evaluates only on samples of the circuit's target class, measuring
    whether the circuit captures the features needed to classify that class.
    """
    _model = unwrap_compiled(model)
    vocab_size = _model.vocab_size
    circuit_set = set(circuit.token_ids)
    non_circuit_ids = [i for i in range(vocab_size) if i not in circuit_set]

    # Filter to same-class samples only
    target_class = circuit.class_idx
    class_indices = [i for i, lbl in enumerate(labels) if lbl == target_class]
    if not class_indices:
        return 0.0
    class_ids = [input_ids_list[i] for i in class_indices]
    class_masks = [attention_mask_list[i] for i in class_indices]
    class_labels = [labels[i] for i in class_indices]

    correct = 0
    total = 0
    eval_batch = 32
    non_circuit_ids_t = torch.tensor(non_circuit_ids, device=DEVICE, dtype=torch.long)

    for start in range(0, len(class_ids), eval_batch):
        end = min(start + eval_batch, len(class_ids))
        batch_ids = torch.cat(class_ids[start:end], dim=0)
        batch_mask = torch.cat(class_masks[start:end], dim=0)
        batch_labels_t = torch.tensor(class_labels[start:end], device=DEVICE)

        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            sparse_seq, *_ = _model(batch_ids, batch_mask)
            original_sparse = _model.to_pooled(sparse_seq, batch_mask)
            patched_sparse = original_sparse.clone()
            patched_sparse[:, non_circuit_ids_t] = 0.0
            patched_logits = _model.classifier_logits_only(patched_sparse)
        preds = patched_logits.argmax(dim=-1)
        correct += (preds == batch_labels_t).sum().item()
        total += end - start

    return correct / total if total > 0 else 0.0


def measure_separation_cosine(centroid_tracker) -> float:
    """Cosine separation: 1 - mean pairwise cosine (matches training loss)."""
    centroids = centroid_tracker.get_normalized_centroids()
    n = centroids.shape[0]
    if n < 2:
        return 1.0
    sim_matrix = centroids @ centroids.T
    mask = torch.triu(torch.ones(n, n, device=DEVICE, dtype=torch.bool), diagonal=1)
    return 1.0 - float(sim_matrix[mask].mean().item())


def measure_separation_jaccard(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels: list[int],
    tokenizer,
    precomputed_attributions: dict[int, torch.Tensor] | None = None,
) -> dict:
    """Supplementary Jaccard-based separation diagnostic.

    Reports within-class consistency, cross-class overlap, and class separation.
    """
    top_k = 10
    _model = unwrap_compiled(model)
    num_classes = _model.classifier_fc2.weight.shape[0]

    sample_top_tokens: list[set[int]] = []
    sample_labels: list[int] = []
    class_token_counts: dict[int, dict[int, int]] = {c: {} for c in range(num_classes)}

    eval_batch = 32
    for start in range(0, len(input_ids_list), eval_batch):
        end = min(start + eval_batch, len(input_ids_list))
        batch_ids = torch.cat(input_ids_list[start:end], dim=0)
        batch_mask = torch.cat(attention_mask_list[start:end], dim=0)
        batch_labels_slice = labels[start:end]
        batch_labels_t = torch.tensor(batch_labels_slice, device=DEVICE)

        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            sparse_seq, *_ = _model(batch_ids, batch_mask)
            _, sparse_vector, W_eff, _ = _model.classify(sparse_seq, batch_mask)

        attr = compute_attribution_tensor(sparse_vector, W_eff, batch_labels_t)

        for i in range(end - start):
            label = batch_labels_slice[i]
            top_k_ids = attr[i].abs().topk(top_k).indices.tolist()
            sample_top_tokens.append(set(top_k_ids))
            sample_labels.append(label)
            for tid in top_k_ids:
                class_token_counts[label][tid] = class_token_counts[label].get(tid, 0) + 1

    class_top_tokens: dict[int, set[int]] = {}
    if precomputed_attributions:
        for c, attr_vec in precomputed_attributions.items():
            class_top_tokens[c] = set(attr_vec.abs().topk(top_k).indices.tolist())
    else:
        for c in range(num_classes):
            counts = class_token_counts[c]
            sorted_tokens = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
            class_top_tokens[c] = set(tid for tid, _ in sorted_tokens)

    within_class_overlaps = []
    for c in range(num_classes):
        class_indices = [i for i, lbl in enumerate(sample_labels) if lbl == c]
        for i_idx in range(len(class_indices)):
            for j_idx in range(i_idx + 1, len(class_indices)):
                set_i = sample_top_tokens[class_indices[i_idx]]
                set_j = sample_top_tokens[class_indices[j_idx]]
                if set_i and set_j:
                    intersection = len(set_i & set_j)
                    union = len(set_i | set_j)
                    within_class_overlaps.append(intersection / union if union > 0 else 0.0)
    within_class = sum(within_class_overlaps) / len(within_class_overlaps) if within_class_overlaps else 0.0

    cross_class_vals = []
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            set_i = class_top_tokens.get(i, set())
            set_j = class_top_tokens.get(j, set())
            if set_i and set_j:
                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                cross_class_vals.append(intersection / union if union > 0 else 0.0)
            else:
                cross_class_vals.append(0.0)
    cross_class = sum(cross_class_vals) / len(cross_class_vals) if cross_class_vals else 0.0

    separation = within_class - cross_class

    class_top_names = {}
    for c in range(num_classes):
        token_ids = list(class_top_tokens.get(c, set()))
        class_top_names[c] = tokenizer.convert_ids_to_tokens(token_ids) if token_ids else []

    return {
        "within_class_consistency": within_class,
        "cross_class_overlap": cross_class,
        "class_separation": separation,
        "class_top_tokens": class_top_names,
    }


