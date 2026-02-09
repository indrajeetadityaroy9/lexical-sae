from dataclasses import dataclass, field

import torch

from splade.evaluation.compare_explainers import run_explainer_comparison
from splade.evaluation.eraser import run_eraser_evaluation
from splade.mechanistic.attribution import compute_attribution_tensor
from splade.mechanistic.layerwise import run_layerwise_evaluation
from splade.circuits.metrics import (VocabularyCircuit,
                                     extract_vocabulary_circuit,
                                     measure_circuit_completeness,
                                     measure_separation_cosine,
                                     measure_separation_jaccard,
                                     visualize_circuit)
from splade.utils.cuda import COMPUTE_DTYPE, DEVICE, unwrap_compiled


@dataclass
class MechanisticResults:
    accuracy: float = 0.0
    circuits: dict[int, VocabularyCircuit] = field(default_factory=dict)
    circuit_completeness: dict[int, float] = field(default_factory=dict)
    semantic_fidelity: dict = field(default_factory=dict)
    dla_verification_error: float = 0.0
    mean_active_dims: float = 0.0
    eraser_metrics: dict = field(default_factory=dict)
    explainer_comparison: dict = field(default_factory=dict)
    layerwise_attribution: dict = field(default_factory=dict)
    sae_comparison: dict = field(default_factory=dict)


def _run_sae_comparison(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels: list[int],
    tokenizer,
) -> dict:
    """Train SAE and compare with DLA in terms of active feature counts."""
    from splade.mechanistic.sae import compute_sae_attribution, train_sae_on_splade

    _model = unwrap_compiled(model)

    print("Training SAE on hidden states...")
    sae = train_sae_on_splade(model, input_ids_list, attention_mask_list)

    with torch.inference_mode():
        classifier_weight = _model.classifier_fc2.weight
        vocab_projector_weight = _model.backbone.get_output_embeddings().weight

    dla_active_counts = []
    sae_active_counts = []

    for input_ids, attention_mask, label in zip(input_ids_list, attention_mask_list, labels):
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            sparse_seq = _model(input_ids, attention_mask)
            sparse_vector = _model.to_pooled(sparse_seq, attention_mask)
            transformed = _model._get_mlm_head_input(input_ids, attention_mask)
            cls_hidden = transformed[:, 0, :]

        # DLA active count
        dla_active = int((sparse_vector[0] > 0).sum().item())
        dla_active_counts.append(dla_active)

        # SAE active count
        sae_attrib = compute_sae_attribution(
            sae, cls_hidden, classifier_weight, label, vocab_projector_weight,
        )
        sae_active = int((sae_attrib.abs() > 1e-6).sum().item())
        sae_active_counts.append(sae_active)

    # SAE reconstruction error
    all_hidden = []
    for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            transformed = _model._get_mlm_head_input(input_ids, attention_mask)
        all_hidden.append(transformed[:, 0, :].float())

    hidden_tensor = torch.cat(all_hidden, dim=0)
    with torch.inference_mode():
        reconstruction, _ = sae(hidden_tensor)
        recon_error = float(torch.nn.functional.mse_loss(
            reconstruction, hidden_tensor,
        ).item())

    return {
        "dla_active_tokens": sum(dla_active_counts) / len(dla_active_counts),
        "sae_active_features": sum(sae_active_counts) / len(sae_active_counts),
        "reconstruction_error": recon_error,
    }


def run_mechanistic_evaluation(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels: list[int],
    tokenizer,
    num_classes: int,
    circuit_fraction: float = 0.1,
    run_sae_comparison: bool = False,
    centroid_tracker=None,
) -> MechanisticResults:
    """Run the full mechanistic interpretability evaluation suite."""
    _model = unwrap_compiled(model)
    results = MechanisticResults()

    # 1. Verify DLA invariant: sum(s_j * W_eff[c,j]) + b_eff_c == logit_c
    total_error = 0.0
    total_active_dims = 0.0
    correct = 0
    n = 0
    eval_batch = 32
    for start in range(0, len(input_ids_list), eval_batch):
        end = min(start + eval_batch, len(input_ids_list))
        batch_ids = torch.cat(input_ids_list[start:end], dim=0)
        batch_mask = torch.cat(attention_mask_list[start:end], dim=0)
        batch_labels_t = torch.tensor(labels[start:end], device=DEVICE)

        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            sparse_seq = _model(batch_ids, batch_mask)
            logits, sparse_vector, W_eff, b_eff = _model.classify(sparse_seq, batch_mask)

        preds = logits.argmax(dim=-1)
        correct += (preds == batch_labels_t).sum().item()
        total_active_dims += (sparse_vector > 0).sum(dim=-1).float().sum().item()

        attr = compute_attribution_tensor(sparse_vector, W_eff, batch_labels_t)
        actual = logits.gather(1, batch_labels_t.unsqueeze(1)).squeeze(1)
        reconstructed = attr.sum(dim=-1) + b_eff.gather(1, batch_labels_t.unsqueeze(1)).squeeze(1)
        total_error += (actual - reconstructed).abs().sum().item()
        n += end - start

    results.accuracy = correct / n if n > 0 else 0.0
    results.dla_verification_error = total_error / n if n > 0 else 0.0
    results.mean_active_dims = total_active_dims / n if n > 0 else 0.0

    # 2. Extract circuits per class
    for class_idx in range(num_classes):
        class_inputs = [
            (ids, mask) for ids, mask, label in zip(input_ids_list, attention_mask_list, labels)
            if label == class_idx
        ]
        if not class_inputs:
            continue

        class_ids, class_masks = zip(*class_inputs)
        precomputed = None
        if centroid_tracker is not None and class_idx < centroid_tracker.num_classes:
            if centroid_tracker._initialized[class_idx]:
                precomputed = centroid_tracker.centroids[class_idx]

        circuit = extract_vocabulary_circuit(
            model, list(class_ids), list(class_masks),
            tokenizer, class_idx, circuit_fraction=circuit_fraction,
            precomputed_attributions=precomputed,
        )
        results.circuits[class_idx] = circuit

    # 3. Measure circuit completeness
    for class_idx, circuit in results.circuits.items():
        if not circuit.token_ids:
            continue
        completeness = measure_circuit_completeness(
            model, input_ids_list, attention_mask_list, labels, circuit,
        )
        circuit.completeness_score = completeness
        results.circuit_completeness[class_idx] = completeness

    # 4. Separation metrics (cosine primary, Jaccard supplementary)
    precomputed_dict = None
    if centroid_tracker is not None:
        precomputed_dict = {
            c: centroid_tracker.centroids[c]
            for c in range(num_classes)
            if centroid_tracker._initialized[c]
        }

    # Only report cosine separation when centroids were actually trained
    # (uninitialized zero centroids trivially give 1.0, which is meaningless)
    has_centroids = (
        centroid_tracker is not None
        and centroid_tracker._initialized.any()
    )
    cosine_sep = measure_separation_cosine(centroid_tracker) if has_centroids else None
    jaccard_result = measure_separation_jaccard(
        model, input_ids_list, attention_mask_list, labels, tokenizer,
        precomputed_attributions=precomputed_dict,
    )
    results.semantic_fidelity = {
        "cosine_separation": cosine_sep,
        **jaccard_result,
    }

    # 5. ERASER faithfulness metrics (FMM bottleneck-level erasure)
    results.eraser_metrics = run_eraser_evaluation(
        model, input_ids_list, attention_mask_list, labels,
    )

    # 6. Baseline explainer comparison (DLA vs gradient vs IG vs attention)
    results.explainer_comparison = run_explainer_comparison(
        model, input_ids_list, attention_mask_list, labels,
    )

    # 7. Layerwise attribution decomposition
    results.layerwise_attribution = run_layerwise_evaluation(
        model, input_ids_list, attention_mask_list, labels,
    )

    # 8. SAE baseline comparison (optional)
    if run_sae_comparison:
        results.sae_comparison = _run_sae_comparison(
            model, input_ids_list, attention_mask_list, labels, tokenizer,
        )

    return results


def print_mechanistic_results(results: MechanisticResults) -> None:
    """Print tiered evaluation report.

    Tier 1 (Performance): Always shown.
    Tier 2 (Faithfulness): Shown if DLA verification passes (error < 0.01).
    Tier 3 (Interpretability): Shown if Tier 2 passes.
    """
    print("\n" + "=" * 80)
    print("CIS EVALUATION REPORT")
    print("=" * 80)

    # --- Tier 1: Performance ---
    print("\n[Tier 1: Performance]")
    print(f"  Accuracy:              {results.accuracy:.4f}")
    print(f"  DLA Verification Error: {results.dla_verification_error:.6f}")

    tier1_pass = results.dla_verification_error < 0.01
    if not tier1_pass:
        print("\n  !! DLA verification failed (error >= 0.01). Skipping Tiers 2-3.")
        print("=" * 80)
        return

    # --- Tier 2: Faithfulness ---
    print("\n[Tier 2: Faithfulness]")
    print(f"  Mean active dims:      {results.mean_active_dims:.1f}")

    if results.circuit_completeness:
        for class_idx, comp in sorted(results.circuit_completeness.items()):
            n_tokens = len(results.circuits[class_idx].token_ids) if class_idx in results.circuits else 0
            print(f"  Class {class_idx} completeness: {comp:.4f} ({n_tokens} circuit tokens)")

    # --- Tier 3: Interpretability ---
    print("\n[Tier 3: Interpretability]")

    if results.semantic_fidelity:
        sf = results.semantic_fidelity
        cos_sep = sf.get('cosine_separation')
        cos_str = f"{cos_sep:.4f}" if cos_sep is not None else "N/A (no trained centroids)"
        print(f"  Cosine separation:     {cos_str}")

    if results.circuits:
        print("\n  Example circuits:")
        for class_idx, circuit in sorted(results.circuits.items()):
            tokens = circuit.token_names[:5]
            scores = circuit.attribution_scores[:5]
            token_strs = [f"{t}({s:.3f})" for t, s in zip(tokens, scores)]
            print(f"    Class {class_idx}: {', '.join(token_strs)}")

    print("\n" + "=" * 80)
