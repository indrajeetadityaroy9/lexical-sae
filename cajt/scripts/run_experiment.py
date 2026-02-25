"""CIS experiment pipeline: Train -> Mechanistic Evaluation.

Two-phase pipeline:
  Phase 1: CIS Training (DF-FLOPS + circuit losses + GECO)
  Phase 2: Mechanistic Evaluation (DLA verification, circuit extraction, completeness, separation)

Optional dense baseline comparison via config.evaluation.train_dense_baseline.
"""

from cajt.baselines.dense import score_dense_baseline, train_dense_baseline
from cajt.config import Config
from cajt.evaluation.orchestrator import run_mechanistic_evaluation
from cajt.evaluation.results import print_comparison_table, print_mechanistic_results
from cajt.evaluation.collect import prepare_mechanistic_inputs
from cajt.training.pipeline import setup_and_train


def run(config: Config) -> dict:
    """Run the full CIS experiment pipeline."""
    train_str = "full" if config.data.train_samples <= 0 else str(config.data.train_samples)
    test_str = "full" if config.data.test_samples <= 0 else str(config.data.test_samples)
    print("\n--- Lexical-SAE (Circuit-Anchored JumpReLU Transcoder) ---")
    print(f"  Model:       {config.model.name}")
    print(f"  Dataset:     {config.data.dataset_name} (train={train_str}, test={test_str})")
    print(f"  Seed:        {config.seed}")
    print()

    # Phase 1: CIS Training
    print("\n--- PHASE 1: CIS TRAINING ---")
    exp = setup_and_train(config, config.seed)
    print(f"Accuracy: {exp.accuracy:.4f}")

    # Optional: Dense baseline comparison
    dense_test_acc = None
    if config.evaluation.train_dense_baseline:
        print("\n--- Training Dense Baseline ---")
        dense_model, dense_tokenizer, dense_val_acc = train_dense_baseline(
            model_name=config.model.name,
            train_texts=exp.train_texts,
            train_labels=exp.train_labels,
            val_texts=exp.val_texts,
            val_labels=exp.val_labels,
            num_labels=exp.num_labels,
            max_length=exp.max_length,
            seed=config.seed,
        )
        dense_test_acc = score_dense_baseline(
            dense_model, dense_tokenizer, exp.test_texts, exp.test_labels, exp.max_length,
        )
        print(f"Dense baseline test accuracy: {dense_test_acc:.4f}")

    # Phase 2: Mechanistic Evaluation
    print("\n--- PHASE 2: MECHANISTIC EVALUATION ---")

    input_ids_list, attention_mask_list = prepare_mechanistic_inputs(
        exp.tokenizer, exp.test_texts, exp.max_length,
    )

    eval_cfg = config.evaluation
    mechanistic_results = run_mechanistic_evaluation(
        exp.model, input_ids_list, attention_mask_list,
        exp.test_labels, exp.tokenizer, num_classes=exp.num_labels,
        run_sae_comparison=True,
        centroid_tracker=exp.centroid_tracker,
        texts=exp.test_texts,
        max_length=exp.max_length,
        run_sparsity_frontier=eval_cfg.run_sparsity_frontier,
        run_transcoder_comparison=eval_cfg.run_transcoder_comparison,
        run_disentanglement=eval_cfg.run_disentanglement,
        spurious_token_ids=eval_cfg.spurious_token_ids,
    )

    print_mechanistic_results(mechanistic_results)

    # Dense baseline comparison table
    if dense_test_acc is not None:
        dl = mechanistic_results.downstream_loss
        cols = [("Method", 25), ("Accuracy", 10), ("Active Dims", 12), ("Delta-CE", 10), ("KL Div", 8)]
        rows = [
            {"Method": f"Dense {config.model.name.split('/')[-1]}",
             "Accuracy": f"{dense_test_acc:.4f}", "Active Dims": "N/A", "Delta-CE": "N/A", "KL Div": "N/A"},
            {"Method": "Lexical-SAE",
             "Accuracy": f"{exp.accuracy:.4f}", "Active Dims": f"{mechanistic_results.mean_active_dims:.0f}",
             "Delta-CE": f"{dl.get('delta_ce', 0):.4f}", "KL Div": f"{dl.get('kl_divergence', 0):.4f}"},
        ]
        if mechanistic_results.sae_comparison:
            sae = mechanistic_results.sae_comparison
            rows.append({"Method": "Post-hoc SAE",
                         "Accuracy": "N/A", "Active Dims": f"{sae['sae_active_features']:.0f}",
                         "Delta-CE": "N/A", "KL Div": "N/A"})
        print_comparison_table(rows, cols)
        if mechanistic_results.sae_comparison:
            sae = mechanistic_results.sae_comparison
            dead_str = f"{sae.get('sae_dead_feature_fraction', 0):.1%} dead" if 'sae_dead_feature_fraction' in sae else "N/A"
            print(f"  SAE details: hidden_dim={sae.get('sae_hidden_dim', 'N/A')}, recon_mse={sae['reconstruction_error']:.4f}, {dead_str}")

    result = mechanistic_results.to_dict()
    result["seed"] = config.seed
    result["accuracy"] = exp.accuracy
    if dense_test_acc is not None:
        result["dense_baseline_accuracy"] = dense_test_acc

    return result
