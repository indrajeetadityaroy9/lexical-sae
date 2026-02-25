import torch

from cajt.training.losses import AttributionCentroidTracker
from cajt.core.constants import CIRCUIT_MASS_FRACTION, EVAL_BATCH_SIZE
from cajt.evaluation.circuit_metrics import (extract_vocabulary_circuit,
                                     measure_circuit_completeness,
                                     measure_separation_cosine,
                                     measure_separation_jaccard)
from cajt.evaluation.autointerp import run_autointerp
from cajt.evaluation.baselines import run_explainer_comparison
from cajt.evaluation.downstream_loss import compute_downstream_loss
from cajt.evaluation.eraser import run_eraser_evaluation
from cajt.evaluation.feature_absorption import detect_feature_absorption
from cajt.evaluation.mib_metrics import compute_cmd, compute_cpr
from cajt.evaluation.polysemy import compute_contextual_consistency_score
from cajt.evaluation.results import MechanisticResults
from cajt.evaluation.sparse_probing import run_sparse_probing
from cajt.evaluation.sparsity_frontier import compute_naopc, sweep_sparsity_frontier
from cajt.core.attribution import compute_attribution_tensor
from cajt.evaluation.layerwise import run_layerwise_evaluation
from cajt.runtime import autocast, DEVICE


def run_mechanistic_evaluation(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels: list[int],
    tokenizer,
    num_classes: int,
    mass_fraction: float = CIRCUIT_MASS_FRACTION,
    run_sae_comparison: bool = True,
    centroid_tracker: AttributionCentroidTracker = None,
    texts: list[str] = None,
    max_length: int = 128,
    run_sparsity_frontier: bool = False,
    run_transcoder_comparison: bool = False,
    run_disentanglement: bool = False,
    spurious_token_ids: list[int] = [],
) -> MechanisticResults:
    """Run the full mechanistic interpretability evaluation suite."""
    results = MechanisticResults()

    # 1. Verify DLA invariant: sum(s_j * W_eff[c,j]) + b_eff_c == logit_c
    total_error = 0.0
    total_active_dims = 0.0
    correct = 0
    n = 0
    eval_batch = EVAL_BATCH_SIZE
    for start in range(0, len(input_ids_list), eval_batch):
        end = min(start + eval_batch, len(input_ids_list))
        batch_ids = torch.cat(input_ids_list[start:end], dim=0)
        batch_mask = torch.cat(attention_mask_list[start:end], dim=0)
        batch_labels_t = torch.tensor(labels[start:end], device=DEVICE)

        with torch.inference_mode(), autocast():
            sparse_seq, *_ = model(batch_ids, batch_mask)
            logits, sparse_vector, W_eff, b_eff = model.classify(sparse_seq, batch_mask)

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
        if class_idx < centroid_tracker.num_classes and centroid_tracker._initialized[class_idx]:
            precomputed = centroid_tracker.centroids[class_idx]

        circuit = extract_vocabulary_circuit(
            model, list(class_ids), list(class_masks),
            tokenizer, class_idx, mass_fraction=mass_fraction,
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
    precomputed_dict = {
        c: centroid_tracker.centroids[c]
        for c in range(num_classes)
        if centroid_tracker._initialized[c]
    }
    cosine_sep = measure_separation_cosine(centroid_tracker)
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

    # 8. Polysemy defense: Contextual Consistency Score
    results.polysemy_scores = compute_contextual_consistency_score(
        model, tokenizer, texts, labels, max_length,
    )

    # 9. SAE baseline comparison (optional)
    if run_sae_comparison:
        from cajt.baselines.sae import compare_sae_with_dla
        results.sae_comparison = compare_sae_with_dla(
            model, input_ids_list, attention_mask_list, labels, tokenizer,
        )

    # 10. Downstream loss: CE/KL from sparse bottleneck
    results.downstream_loss = compute_downstream_loss(
        model, input_ids_list, attention_mask_list, labels,
    )

    # 11. NAOPC normalization
    results.naopc = compute_naopc(model, input_ids_list, attention_mask_list, labels)

    # 12. Sparsity-fidelity frontier (expensive, opt-in)
    if run_sparsity_frontier:
        results.sparsity_frontier = sweep_sparsity_frontier(
            model, input_ids_list, attention_mask_list, labels,
        )

    # 13. Feature absorption detection
    results.feature_absorption = detect_feature_absorption(
        model, input_ids_list, attention_mask_list, labels, tokenizer,
    )

    # 14. Sparse probing
    results.sparse_probing = run_sparse_probing(
        model, input_ids_list, attention_mask_list, labels, num_classes,
    )

    # 15. AutoInterp (offline)
    results.autointerp = run_autointerp(
        model, tokenizer, texts, input_ids_list, attention_mask_list,
    )

    # 16. Transcoder baseline (expensive, opt-in)
    if run_transcoder_comparison:
        from cajt.baselines.transcoder import run_transcoder_comparison as _run_tc
        results.transcoder_comparison = _run_tc(
            model, input_ids_list, attention_mask_list, labels,
        )

    # 17. Disentanglement metrics (opt-in)
    if run_disentanglement:
        from cajt.evaluation.disentanglement import compute_scr, compute_tpp
        results.disentanglement = {
            "scr": compute_scr(
                model, input_ids_list, attention_mask_list, labels, spurious_token_ids,
            ),
            "tpp": compute_tpp(
                model, input_ids_list, attention_mask_list, labels, num_classes,
            ),
        }

    # 18. MIB circuit metrics (CPR + CMD)
    results.mib_metrics = {
        "cpr": compute_cpr(model, input_ids_list, attention_mask_list, labels, mass_fraction),
        "cmd": compute_cmd(model, input_ids_list, attention_mask_list, labels),
    }

    return results
