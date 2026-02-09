"""NER experiment entry point: CoNLL-2003 with LexicalSAE.

Usage:
    python -m splade.scripts.run_ner --config experiments/ner/conll2003.yaml
"""

import argparse
import json
import os

import torch
from transformers import AutoTokenizer

from splade.config.ner_schema import load_ner_config
from splade.data.ner_loader import (
    CONLL2003_LABEL_NAMES,
    CONLL2003_NUM_LABELS,
    infer_ner_max_length,
    load_conll2003,
    tokenize_and_align_dataset,
)
from splade.evaluation.sequence_mechanistic import (
    print_sequence_mechanistic_results,
    run_sequence_mechanistic_evaluation,
)
from splade.models.lexical_sae import LexicalSAE
from splade.training.optim import _infer_batch_size
from splade.training.sequence_loop import train_sequence_model
from splade.utils.cuda import DEVICE, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_ner_config(args.config)

    for seed in config.evaluation.seeds:
        print(f"\n{'='*60}")
        print(f"NER Experiment: {config.experiment_name} (seed={seed})")
        print(f"{'='*60}")

        set_seed(seed)

        # Load CoNLL-2003
        train_tokens, train_tags, test_tokens, test_tags = load_conll2003(
            config.data.train_samples, config.data.test_samples, seed=seed,
        )
        print(f"Loaded CoNLL-2003: {len(train_tokens)} train, {len(test_tokens)} test")

        # Tokenize
        tokenizer = AutoTokenizer.from_pretrained(config.model.name)
        max_length = infer_ner_max_length(
            train_tokens, tokenizer, model_name=config.model.name,
        )
        # Sequence model holds [B,L,V] — much more memory than pooled [B,V].
        # Use explicit config batch_size if set; otherwise auto-infer with
        # conservative scaling for the [B*L, V] loss tensor overhead.
        if config.training.batch_size is not None:
            batch_size = config.training.batch_size
            print(f"Config override: max_length={max_length}, batch_size={batch_size}")
        else:
            base_bs = _infer_batch_size(config.model.name, max_length)
            batch_size = max(4, base_bs // (max(1, max_length // 16)))
            print(f"Auto-inferred: max_length={max_length}, batch_size={batch_size}")

        train_ids, train_masks, train_labels = tokenize_and_align_dataset(
            train_tokens, train_tags, tokenizer, max_length,
        )
        test_ids, test_masks, test_labels = tokenize_and_align_dataset(
            test_tokens, test_tags, tokenizer, max_length,
        )

        # Convert to tensors
        train_ids_t = torch.tensor(train_ids, dtype=torch.long)
        train_masks_t = torch.tensor(train_masks, dtype=torch.long)
        train_labels_t = torch.tensor(train_labels, dtype=torch.long)
        test_ids_t = torch.tensor(test_ids, dtype=torch.long)
        test_masks_t = torch.tensor(test_masks, dtype=torch.long)
        test_labels_t = torch.tensor(test_labels, dtype=torch.long)

        # Validation split
        val_size = min(200, len(train_ids_t) // 5)
        val_ids = train_ids_t[-val_size:]
        val_masks = train_masks_t[-val_size:]
        val_labels = train_labels_t[-val_size:]
        train_ids_t = train_ids_t[:-val_size]
        train_masks_t = train_masks_t[:-val_size]
        train_labels_t = train_labels_t[:-val_size]

        # Create model
        model = LexicalSAE(
            config.model.name, CONLL2003_NUM_LABELS,
        ).to(DEVICE)

        # Train
        grad_accum = config.training.gradient_accumulation_steps
        print(f"Gradient accumulation steps: {grad_accum} (effective batch size: {batch_size * grad_accum})")

        centroid_tracker = train_sequence_model(
            model, train_ids_t, train_masks_t, train_labels_t,
            model_name=config.model.name,
            num_labels=CONLL2003_NUM_LABELS,
            val_input_ids=val_ids,
            val_attention_masks=val_masks,
            val_token_labels=val_labels,
            batch_size=batch_size,
            target_accuracy=config.training.target_accuracy,
            sparsity_target=config.training.sparsity_target,
            warmup_fraction=config.training.warmup_fraction,
            gradient_accumulation_steps=grad_accum,
        )

        # Evaluate
        results = run_sequence_mechanistic_evaluation(
            model, test_ids_t, test_masks_t, test_labels_t,
            tokenizer, label_names=CONLL2003_LABEL_NAMES,
            centroid_tracker=centroid_tracker,
        )
        print_sequence_mechanistic_results(
            results, model=model, tokenizer=tokenizer,
            input_ids=test_ids_t, attention_masks=test_masks_t,
            token_labels=test_labels_t, label_names=CONLL2003_LABEL_NAMES,
            centroid_tracker=centroid_tracker,
        )

        # Save results
        os.makedirs(config.output_dir, exist_ok=True)
        output_path = os.path.join(config.output_dir, f"ner_seed{seed}.json")
        with open(output_path, "w") as f:
            json.dump({
                "experiment_name": config.experiment_name,
                "seed": seed,
                "model": config.model.name,
                "num_labels": CONLL2003_NUM_LABELS,
                "max_length": max_length,
                "batch_size": batch_size,
                "token_accuracy": results.token_accuracy,
                "entity_f1": results.entity_f1,
                "dla_verification_error": results.dla_verification_error,
                "mean_active_dims": results.mean_active_dims,
                "classification_report": results.classification_report,
            }, f, indent=2)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
