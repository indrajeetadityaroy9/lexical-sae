"""MechanisticResults dataclass and display logic."""

from dataclasses import dataclass, field, asdict

from cajt.evaluation.circuit_metrics import VocabularyCircuit


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


    def to_dict(self) -> dict:
        """Canonical serialization for JSON output."""
        d = {}
        for f in self.__dataclass_fields__:
            val = getattr(self, f)
            if f == "circuits":
                d[f] = {
                    str(k): {
                        "token_ids": c.token_ids,
                        "token_names": c.token_names,
                        "attribution_scores": c.attribution_scores,
                        "completeness_score": c.completeness_score,
                    }
                    for k, c in val.items()
                }
            elif f == "circuit_completeness":
                d[f] = {str(k): v for k, v in val.items()}
            else:
                d[f] = val
        return d


def print_comparison_table(
    rows: list[dict[str, str]],
    columns: list[tuple[str, int]],
    title: str = "COMPARISON TABLE",
) -> None:
    """Print a formatted comparison table.

    Args:
        rows: list of dicts mapping column name to display value.
        columns: list of (column_name, width) tuples.
        title: table title.
    """
    width = sum(w for _, w in columns) + len(columns) - 1
    print(f"\n{'=' * width}")
    print(title)
    print(f"{'=' * width}")
    header = "".join(f"{name:>{w}}" if i > 0 else f"{name:<{w}}" for i, (name, w) in enumerate(columns))
    print(header)
    print("-" * width)
    for row in rows:
        line = ""
        for i, (col, w) in enumerate(columns):
            val = row.get(col, "N/A")
            line += f"{val:<{w}}" if i == 0 else f"{val:>{w}}"
        print(line)
    print(f"{'=' * width}")


def print_bias_results(bias: dict) -> None:
    """Print formatted bias evaluation results (used by surgery experiments)."""
    print(f"Overall accuracy: {bias['overall_accuracy']:.4f}")
    print(f"Overall FPR: {bias['overall_fpr']:.4f}")
    if "collateral_gap" in bias:
        print(f"Collateral gap: {bias['collateral_gap']:+.4f} "
              f"(identity={bias['nontoxic_identity_accuracy']:.4f}, "
              f"no-identity={bias['nontoxic_noidentity_accuracy']:.4f})")
    for name, metrics in sorted(bias["per_identity"].items(), key=lambda x: -abs(x[1]["fpr_gap"])):
        print(f"  {name:<35} FPR={metrics['fpr']:.4f}  gap={metrics['fpr_gap']:+.4f}  n={metrics['count']}")


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
    print(f"  CMD:                   {mib['cmd']['cmd']:.4f} (min_mass_frac={mib['cmd']['min_mass_fraction']:.2f})")

    if results.transcoder_comparison:
        tc = results.transcoder_comparison
        print(f"  Transcoder MSE:        {tc['reconstruction_mse']:.4f} (active={tc['mean_active_features']:.0f}, dead={tc['dead_feature_fraction']:.1%})")

    if results.disentanglement:
        print(f"  SCR score:             {results.disentanglement['scr']['scr_score']:.4f}")
        print(f"  TPP score:             {results.disentanglement['tpp']['tpp_score']:.4f}")

    print("\n" + "=" * 80)
