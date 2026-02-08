import argparse
import json
import os
from dataclasses import asdict

import torch
import yaml
from transformers import AutoTokenizer

from splade.config.load import load_config
from splade.data.loader import infer_max_length, load_dataset_by_name
from splade.inference import score_model
from splade.models.splade import SpladeModel
from splade.training.loop import train_model
from splade.training.optim import _infer_batch_size
from splade.utils.cuda import DEVICE, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SPLADE via YAML config")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.evaluation.seeds[0]
    set_seed(seed)

    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "resolved_config.yaml"), "w") as f:
        yaml.dump(asdict(config), f)

    train_texts, train_labels, test_texts, test_labels, num_labels = load_dataset_by_name(
        config.data.dataset_name,
        config.data.train_samples,
        config.data.test_samples,
        seed=seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    max_length = infer_max_length(train_texts, tokenizer)
    batch_size = _infer_batch_size(config.model.name, max_length)

    model = SpladeModel(config.model.name, num_labels).to(DEVICE)
    model = torch.compile(model, mode="reduce-overhead")

    train_model(
        model, tokenizer, train_texts, train_labels,
        model_name=config.model.name, num_labels=num_labels,
        val_texts=test_texts, val_labels=test_labels,
    )

    accuracy = score_model(
        model, tokenizer, test_texts, test_labels,
        max_length, batch_size, num_labels,
    )
    print(f"\nTest accuracy: {accuracy:.4f}")

    metrics_path = os.path.join(config.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": accuracy}, f, indent=2)

    checkpoint_path = os.path.join(config.output_dir, "model.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
