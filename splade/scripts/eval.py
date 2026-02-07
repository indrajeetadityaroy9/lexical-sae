"""CLI entry point for the canonical interpretability benchmark via YAML."""

import argparse
import os
import yaml
from dataclasses import asdict
from transformers import AutoTokenizer

from splade.data.loader import load_dataset_by_name
from splade.evaluation.adversarial import CharacterAttack, TextFoolerAttack, WordNetAttack
from splade.evaluation.benchmark import (
    aggregate_results,
    benchmark_explainer,
    print_aggregated_results,
    print_interpretability_results,
)
from splade.evaluation.explainers import make_lime_explain_fn, make_ig_explain_fn
from splade.evaluation.faithfulness import UnigramSampler
from splade.models.splade import SpladeModel
from splade.training.finetune import finetune_splade_for_ffidelity
from splade.utils.cuda import set_seed, DEVICE
from splade.config.load import load_config
from splade.config.schema import Config
from splade.training.loop import train_model
from splade.inference import score_model, explain_model
import torch

class PredictorWrapper:
    """Wrapper to make SpladeModel compatible with EmbeddingPredictor protocol."""
    def __init__(self, model, tokenizer, max_length, batch_size):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

    def predict_proba(self, texts):
        from splade.inference import predict_proba_model
        return predict_proba_model(self.model, self.tokenizer, texts, self.max_length, self.batch_size)

    def get_embeddings(self, texts):
        """Return (embeddings, attention_mask) tensors for embedding-level soft metrics."""
        encoding = self.tokenizer(
            texts, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)
        _model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        with torch.inference_mode():
            embeddings = _model.get_embeddings(input_ids)
        return embeddings, attention_mask

    def predict_proba_from_embeddings(self, embeddings, attention_mask):
        """Forward pass from pre-computed embeddings."""
        _model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        with torch.inference_mode():
            logits, _ = _model.forward_from_embeddings(embeddings, attention_mask)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs.cpu().tolist()

def run_benchmark(config: Config) -> list:
    """Run multi-seed benchmark defined by config."""
    all_seed_results = []

    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "resolved_config.yaml"), "w") as f:
        yaml.dump(asdict(config), f)

    for seed in config.evaluation.seeds:
        print(f"\n" + "#" * 40)
        print(f"RUNNING BENCHMARK WITH SEED {seed}")
        print("#" * 40)

        set_seed(seed)

        train_texts, train_labels, test_texts, test_labels, num_labels = load_dataset_by_name(
            config.data.dataset_name,
            config.data.train_samples,
            config.data.test_samples,
            seed=seed,
        )
        config.data.num_labels = num_labels
        config.training.seed = seed

        # Split train into train/val to avoid test-set leakage (G3)
        val_size = min(200, len(train_texts) // 5)
        val_texts_split = train_texts[-val_size:]
        val_labels_split = train_labels[-val_size:]
        train_texts_split = train_texts[:-val_size]
        train_labels_split = train_labels[:-val_size]

        tokenizer = AutoTokenizer.from_pretrained(config.model.name)
        model = SpladeModel(config.model.name, config.data.num_labels).to(DEVICE)

        if config.model.compile:
            model = torch.compile(model, mode=config.model.compile_mode)

        train_model(model, tokenizer, train_texts_split, train_labels_split,
                    config.training, config.model, config.data,
                    val_texts=val_texts_split, val_labels=val_labels_split)

        accuracy = score_model(
            model,
            tokenizer,
            test_texts,
            test_labels,
            config.data.max_length,
            config.training.batch_size,
            config.data.num_labels
        )
        print(f"Model Accuracy: {accuracy:.4f}")
        mask_token = tokenizer.mask_token

        sampler = UnigramSampler(test_texts, seed=seed)
        print("\nFine-tuning model copy for F-Fidelity...")

        fine_tuned_model = finetune_splade_for_ffidelity(
            model,
            tokenizer,
            train_texts_split,
            train_labels_split,
            beta=config.evaluation.ffidelity_beta,
            ft_epochs=config.evaluation.ffidelity_ft_epochs,
            ft_lr=config.evaluation.ffidelity_ft_lr,
            batch_size=config.evaluation.ffidelity_ft_batch_size,
            mask_token=mask_token,
            seed=seed,
            max_length=config.data.max_length
        )

        predictor = PredictorWrapper(model, tokenizer, config.data.max_length, config.evaluation.batch_size)
        ft_predictor = PredictorWrapper(fine_tuned_model, tokenizer, config.data.max_length, config.evaluation.batch_size)

        attacks = [
            WordNetAttack(max_changes=config.evaluation.adversarial_max_changes),
            TextFoolerAttack(predictor, max_changes=config.evaluation.adversarial_max_changes),
            CharacterAttack(max_changes=config.evaluation.adversarial_max_changes),
        ]

        def splade_explain_fn(text, top_k):
            return explain_model(model, tokenizer, text, config.data.max_length, top_k=top_k, input_only=True)

        def random_explain_fn(text, top_k):
            import random as _rng
            words = text.split()
            if not words:
                return []
            rng_local = _rng.Random(seed)
            scored = [(w, rng_local.random()) for w in words]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]

        results = []
        for explainer_name in config.evaluation.explainers:
            if explainer_name == "splade":
                fn = splade_explain_fn
                tok = tokenizer
            elif explainer_name == "random":
                fn = random_explain_fn
                tok = None
            elif explainer_name == "lime":
                fn = make_lime_explain_fn(
                    predictor,
                    num_samples=config.evaluation.lime_num_samples,
                    seed=seed,
                )
                tok = None
            elif explainer_name == "ig":
                fn = make_ig_explain_fn(
                    model, tokenizer, config.data.max_length,
                    n_steps=config.evaluation.ig_n_steps,
                )
                tok = tokenizer
            else:
                raise ValueError(f"Unknown explainer: {explainer_name}")

            result = benchmark_explainer(
                predictor,
                explainer_name.upper(),
                fn,
                test_texts,
                config.evaluation,
                mask_token,
                attacks=attacks,
                sampler=sampler,
                ftuned_clf=ft_predictor,
                tokenizer=tok,
                max_length=config.data.max_length,
            )
            result.accuracy = accuracy
            results.append(result)

        all_seed_results.append(results)

    if len(config.evaluation.seeds) > 1:
        aggregated = aggregate_results(all_seed_results)
        print_aggregated_results(aggregated)
        import json
        with open(os.path.join(config.output_dir, "metrics_aggregated.json"), "w") as f:
            json.dump(aggregated, f, indent=2)
    else:
        print_interpretability_results(all_seed_results[0], config.evaluation)

    return all_seed_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canonical interpretability benchmark via YAML")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    run_benchmark(config)


if __name__ == "__main__":
    main()
