from dataclasses import dataclass, field

import torch

from splade.circuits.losses import AttributionCentroidTracker
from splade.circuits.metrics import (VocabularyCircuit,
                                     extract_vocabulary_circuit,
                                     measure_circuit_completeness,
                                     measure_separation_cosine,
                                     measure_separation_jaccard)
from splade.evaluation.autointerp import run_autointerp
from splade.evaluation.compare_explainers import run_explainer_comparison
from splade.evaluation.downstream_loss import compute_downstream_loss
from splade.evaluation.eraser import run_eraser_evaluation
from splade.evaluation.feature_absorption import detect_feature_absorption
from splade.evaluation.mib_metrics import compute_cmd, compute_cpr
from splade.evaluation.polysemy import compute_contextual_consistency_score
from splade.evaluation.sparse_probing import run_sparse_probing
from splade.evaluation.sparsity_frontier import compute_naopc, sweep_sparsity_frontier
from splade.mechanistic.attribution import compute_attribution_tensor
from splade.mechanistic.layerwise import run_layerwise_evaluation
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
    polysemy_scores: dict = field(default_factory=dict)
    downstream_loss: dict = field(default_factory=dict)
    sparsity_frontier: list = field(default_factory=list)
    naopc: dict = field(default_factory=dict)
    feature_absorption: dict = field(default_factory=dict)
    sparse_probing: dict = field(default_factory=dict)
    autointerp: dict = field(default_factory=dict)
    transcoder_comparison: dict = field(default_factory=dict)
    disentanglement: dict = field(default_factory=dict)
    mib_metrics: dict = field(default_factory=dict)


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
            sparse_seq, *_ = _model(input_ids, attention_mask)
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

        # Dead feature analysis
        all_features = sae.encode(hidden_tensor)
        ever_active = (all_features > 0).any(dim=0)
        dead_frac = 1.0 - ever_active.float().mean().item()

    return {
        "dla_active_tokens": sum(dla_active_counts) / len(dla_active_counts),
        "sae_active_features": sum(sae_active_counts) / len(sae_active_counts),
        "reconstruction_error": recon_error,
        "sae_dead_feature_fraction": dead_frac,
        "sae_hidden_dim": int(all_features.shape[1]),
    }


def run_mechanistic_evaluation(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels: list[int],
    tokenizer,
    num_classes: int,
    circuit_fraction: float = 0.1,
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
            sparse_seq, *_ = _model(batch_ids, batch_mask)
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
        if class_idx < centroid_tracker.num_classes and centroid_tracker._initialized[class_idx]:
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
        results.sae_comparison = _run_sae_comparison(
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
        model, tokenizer, texts, labels, input_ids_list, attention_mask_list,
    )

    # 16. Transcoder baseline (expensive, opt-in)
    if run_transcoder_comparison:
        from splade.evaluation.transcoder_baseline import run_transcoder_comparison as _run_tc
        results.transcoder_comparison = _run_tc(
            model, input_ids_list, attention_mask_list, labels,
        )

    # 17. Disentanglement metrics (opt-in)
    if run_disentanglement:
        from splade.evaluation.disentanglement import compute_scr, compute_tpp
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
        "cpr": compute_cpr(model, input_ids_list, attention_mask_list, labels, circuit_fraction),
        "cmd": compute_cmd(model, input_ids_list, attention_mask_list, labels),
    }

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
    dl = results.downstream_loss
    print(f"  Delta-CE (bottleneck):  {dl['delta_ce']:.4f}")
    print(f"  KL divergence:         {dl['kl_divergence']:.4f}")

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

    sf = results.semantic_fidelity
    print(f"  Cosine separation:     {sf['cosine_separation']:.4f}")

    print("\n  Example circuits:")
    for class_idx, circuit in sorted(results.circuits.items()):
        tokens = circuit.token_names[:5]
        scores = circuit.attribution_scores[:5]
        token_strs = [f"{t}({s:.3f})" for t, s in zip(tokens, scores)]
        print(f"    Class {class_idx}: {', '.join(token_strs)}")

    ps = results.polysemy_scores
    print(f"\n[Polysemy Defense: Contextual Consistency Score]")
    print(f"  Mean cross-context Jaccard: {ps['mean_jaccard']:.4f}")
    print(f"  Words evaluated: {ps['num_words_evaluated']}")
    for word, info in sorted(ps.get("per_word", {}).items(), key=lambda x: x[1]["jaccard"]):
        print(f"    {word:<12} Jaccard={info['jaccard']:.4f}  pairs={info['n_pairs']}  n={info['n_occurrences']}")

    # --- Tier 4: SAEBench Metrics ---
    print("\n[Tier 4: SAEBench Metrics]")

    print(f"  NAOPC Comp:            {results.naopc['naopc_comprehensiveness']:.4f}")
    print(f"  NAOPC Suff:            {results.naopc['naopc_sufficiency']:.4f}")

    fa = results.feature_absorption
    print(f"  Feature absorption:    {fa['absorption_score']:.4f} ({fa['num_pairs_tested']} pairs tested)")

    sp = results.sparse_probing
    print(f"  Sparse probe accuracy: {sp['probe_accuracy']:.4f} (F1={sp['probe_f1_macro']:.4f}, {sp['n_features_used']} features)")

    print(f"  AutoInterp score:      {results.autointerp['mean_score']:.4f}")

    mib = results.mib_metrics
    print(f"  CPR:                   {mib['cpr']['cpr']:.4f}")
    print(f"  CMD:                   {mib['cmd']['cmd']:.4f} (min_frac={mib['cmd']['min_fraction']:.2f})")

    if results.transcoder_comparison:
        tc = results.transcoder_comparison
        print(f"  Transcoder MSE:        {tc['reconstruction_mse']:.4f} (active={tc['mean_active_features']:.0f}, dead={tc['dead_feature_fraction']:.1%})")

    if results.disentanglement:
        print(f"  SCR score:             {results.disentanglement['scr']['scr_score']:.4f}")
        print(f"  TPP score:             {results.disentanglement['tpp']['tpp_score']:.4f}")

    print("\n" + "=" * 80)
