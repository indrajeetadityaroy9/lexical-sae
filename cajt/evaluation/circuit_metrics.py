"""Circuit evaluation metrics.

Circuit extraction uses fraction of ACTIVE (non-zero) vocabulary dimensions,
reflecting the sparse bottleneck where only ~100 of ~30K dims are active.
Cosine separation is the primary metric (matching training). Jaccard is
retained as a supplementary diagnostic.
"""

from dataclasses import dataclass, field

import torch

from cajt.core.types import circuit_mask_by_mass
from cajt.core.attribution import compute_attribution_tensor
from cajt.core.constants import CIRCUIT_MASS_FRACTION, EVAL_BATCH_SIZE
from cajt.runtime import autocast, DEVICE


def _id_to_name(tid: int, tokenizer, model: torch.nn.Module) -> str:
    """Map token/virtual-slot ID to a human-readable name."""
    V = model.vocab_size
    if tid < V:
        return tokenizer.convert_ids_to_tokens([tid])[0]
    if model.virtual_expander is not None:
        vpe = model.virtual_expander
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
    mass_fraction: float = CIRCUIT_MASS_FRACTION,
    precomputed_attributions: torch.Tensor | None = None,
) -> VocabularyCircuit:
    """Extract vocabulary circuit via cumulative attribution mass.

    Selects the minimum set of features capturing mass_fraction of total
    attribution mass. Data-adaptive: sparse models get small circuits,
    dense models get larger ones.
    """


    if precomputed_attributions is not None:
        mean_attributions = precomputed_attributions
    else:
        vocab_size = model.vocab_size_expanded
        token_attribution_sums = torch.zeros(vocab_size, device=DEVICE)
        n_examples = 0
        eval_batch = EVAL_BATCH_SIZE

        for start in range(0, len(input_ids_list), eval_batch):
            end = min(start + eval_batch, len(input_ids_list))
            batch_ids = torch.cat(input_ids_list[start:end], dim=0)
            batch_mask = torch.cat(attention_mask_list[start:end], dim=0)
            bs = end - start

            with torch.inference_mode(), autocast():
                sparse_seq, *_ = model(batch_ids, batch_mask)
                _, sparse_vector, W_eff, _ = model.classify(sparse_seq, batch_mask)

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

    # Select features capturing mass_fraction of total attribution mass.
    # Data-adaptive: sparse models yield small circuits, dense models larger.
    mass_mask = circuit_mask_by_mass(
        mean_attributions.unsqueeze(0), mass_fraction,
    ).squeeze(0)
    circuit_token_ids = mass_mask.nonzero(as_tuple=True)[0]
    if len(circuit_token_ids) == 0:
        return VocabularyCircuit(class_idx=class_idx)

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

    vocab_size = model.vocab_size
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
    eval_batch = EVAL_BATCH_SIZE
    non_circuit_ids_t = torch.tensor(non_circuit_ids, device=DEVICE, dtype=torch.long)

    for start in range(0, len(class_ids), eval_batch):
        end = min(start + eval_batch, len(class_ids))
        batch_ids = torch.cat(class_ids[start:end], dim=0)
        batch_mask = torch.cat(class_masks[start:end], dim=0)
        batch_labels_t = torch.tensor(class_labels[start:end], device=DEVICE)

        with torch.inference_mode(), autocast():
            sparse_seq, *_ = model(batch_ids, batch_mask)
            original_sparse = model.to_pooled(sparse_seq, batch_mask)
            patched_sparse = original_sparse.clone()
            patched_sparse[:, non_circuit_ids_t] = 0.0
            patched_logits = model.classifier_logits_only(patched_sparse)
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

    num_classes = model.classifier_fc2.weight.shape[0]

    sample_top_tokens: list[set[int]] = []
    sample_labels: list[int] = []
    class_token_counts: dict[int, dict[int, int]] = {c: {} for c in range(num_classes)}

    eval_batch = EVAL_BATCH_SIZE
    for start in range(0, len(input_ids_list), eval_batch):
        end = min(start + eval_batch, len(input_ids_list))
        batch_ids = torch.cat(input_ids_list[start:end], dim=0)
        batch_mask = torch.cat(attention_mask_list[start:end], dim=0)
        batch_labels_slice = labels[start:end]
        batch_labels_t = torch.tensor(batch_labels_slice, device=DEVICE)

        with torch.inference_mode(), autocast():
            sparse_seq, *_ = model(batch_ids, batch_mask)
            _, sparse_vector, W_eff, _ = model.classify(sparse_seq, batch_mask)

        attr = compute_attribution_tensor(sparse_vector, W_eff, batch_labels_t)

        # Mass-based circuit selection per sample
        batch_mass_mask = circuit_mask_by_mass(attr.abs(), CIRCUIT_MASS_FRACTION)
        for i in range(end - start):
            label = batch_labels_slice[i]
            circuit_ids = batch_mass_mask[i].nonzero(as_tuple=True)[0].tolist()
            sample_top_tokens.append(set(circuit_ids))
            sample_labels.append(label)
            for tid in circuit_ids:
                class_token_counts[label][tid] = class_token_counts[label].get(tid, 0) + 1

    class_top_tokens: dict[int, set[int]] = {}
    if precomputed_attributions:
        for c, attr_vec in precomputed_attributions.items():
            mass_mask = circuit_mask_by_mass(attr_vec.abs().unsqueeze(0), CIRCUIT_MASS_FRACTION).squeeze(0)
            class_top_tokens[c] = set(mass_mask.nonzero(as_tuple=True)[0].tolist())
    else:
        for c in range(num_classes):
            counts = class_token_counts[c]
            if not counts:
                class_top_tokens[c] = set()
                continue
            # Aggregate class-level: use mass-based selection on count-weighted attributions
            count_tensor = torch.zeros(model.vocab_size_expanded, device=DEVICE)
            for tid, cnt in counts.items():
                count_tensor[tid] = cnt
            mass_mask = circuit_mask_by_mass(count_tensor.unsqueeze(0), CIRCUIT_MASS_FRACTION).squeeze(0)
            class_top_tokens[c] = set(mass_mask.nonzero(as_tuple=True)[0].tolist())

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


