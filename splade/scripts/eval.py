import argparse
import json
import os
from dataclasses import asdict

import torch
import yaml
from transformers import AutoTokenizer

from splade.config.load import load_config
from splade.config.schema import Config
from splade.data.loader import infer_max_length, load_dataset_by_name
from splade.evaluation.adversarial import (CharacterAttack, TextFoolerAttack,
                                           WordNetAttack)
from splade.evaluation.benchmark import (aggregate_results,
                                         benchmark_explainer,
                                         print_aggregated_results,
                                         print_interpretability_results)
from splade.evaluation.constants import ADVERSARIAL_MAX_CHANGES, FFIDELITY_BETA
from splade.evaluation.faithfulness import UnigramSampler
from splade.inference import (explain_model, explain_model_batch,
                              predict_proba_model, score_model)
from splade.models.splade import SpladeModel
from splade.training.finetune import finetune_splade_for_ffidelity
from splade.training.loop import train_model
from splade.training.optim import _infer_batch_size
from splade.utils.cuda import COMPUTE_DTYPE, DEVICE, set_seed


class PredictorWrapper:
    def __init__(self, model, tokenizer, max_length, batch_size):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

    def predict_proba(self, texts):
        return predict_proba_model(self.model, self.tokenizer, texts, self.max_length, self.batch_size)

    def get_embeddings(self, texts):
        encoding = self.tokenizer(
            texts, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            embeddings = self.model._orig_mod.get_embeddings(input_ids)
        return embeddings, attention_mask

    def predict_proba_from_embeddings(self, embeddings, attention_mask):
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            logits, _ = self.model._orig_mod.forward_from_embeddings(embeddings, attention_mask)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs.cpu().tolist()

    def predict_proba_from_embeddings_tensor(self, embeddings, attention_mask):
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            logits, _ = self.model._orig_mod.forward_from_embeddings(embeddings, attention_mask)
        return torch.nn.functional.softmax(logits, dim=-1)


def run_benchmark(config: Config) -> list:
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

        tokenizer = AutoTokenizer.from_pretrained(config.model.name)
        max_length = infer_max_length(train_texts, tokenizer)
        batch_size = _infer_batch_size(config.model.name, max_length)
        eval_batch_size = min(batch_size * 4, 128)
        ft_batch_size = min(batch_size * 2, 64)

        print(f"Auto-inferred: max_length={max_length}, train_batch={batch_size}, "
              f"eval_batch={eval_batch_size}, ft_batch={ft_batch_size}")

        val_size = min(200, len(train_texts) // 5)
        val_texts_split = train_texts[-val_size:]
        val_labels_split = train_labels[-val_size:]
        train_texts_split = train_texts[:-val_size]
        train_labels_split = train_labels[:-val_size]

        model = SpladeModel(config.model.name, num_labels).to(DEVICE)
        model = torch.compile(model, mode="reduce-overhead")

        train_model(
            model, tokenizer, train_texts_split, train_labels_split,
            model_name=config.model.name, num_labels=num_labels,
            val_texts=val_texts_split, val_labels=val_labels_split,
        )

        accuracy = score_model(
            model, tokenizer, test_texts, test_labels,
            max_length, batch_size, num_labels,
        )
        print(f"Model Accuracy: {accuracy:.4f}")
        mask_token = tokenizer.mask_token

        sampler = UnigramSampler(test_texts, seed=seed)
        print("\nFine-tuning model copy for F-Fidelity...")

        fine_tuned_model = finetune_splade_for_ffidelity(
            model, tokenizer, train_texts_split, train_labels_split,
            beta=FFIDELITY_BETA, batch_size=ft_batch_size,
            mask_token=mask_token, seed=seed, max_length=max_length,
        )

        predictor = PredictorWrapper(model, tokenizer, max_length, eval_batch_size)
        ft_predictor = PredictorWrapper(fine_tuned_model, tokenizer, max_length, eval_batch_size)

        attacks = [
            WordNetAttack(max_changes=ADVERSARIAL_MAX_CHANGES),
            TextFoolerAttack(predictor, max_changes=ADVERSARIAL_MAX_CHANGES),
            CharacterAttack(max_changes=ADVERSARIAL_MAX_CHANGES),
        ]

        def splade_explain_fn(text, top_k):
            return explain_model(model, tokenizer, text, max_length, top_k=top_k, input_only=True)

        def splade_batch_explain_fn(texts, top_k):
            return explain_model_batch(
                model, tokenizer, texts, max_length,
                top_k=top_k, input_only=True, batch_size=eval_batch_size,
            )

        result = benchmark_explainer(
            predictor,
            "SPLADE",
            splade_explain_fn,
            splade_batch_explain_fn,
            test_texts,
            mask_token=mask_token,
            seed=seed,
            attacks=attacks,
            sampler=sampler,
            ftuned_clf=ft_predictor,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        result.accuracy = accuracy
        results = [result]

        all_seed_results.append(results)

    if len(config.evaluation.seeds) > 1:
        aggregated = aggregate_results(all_seed_results)
        print_aggregated_results(aggregated)
        with open(os.path.join(config.output_dir, "metrics_aggregated.json"), "w") as f:
            json.dump(aggregated, f, indent=2)
    else:
        print_interpretability_results(all_seed_results[0])

    return all_seed_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canonical interpretability benchmark via YAML")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    run_benchmark(config)


if __name__ == "__main__":
    main()
