"""CLI entry point for canonical SPLADE training via YAML config."""

import argparse
import os
import torch
import yaml
from dataclasses import asdict
from transformers import AutoTokenizer

from splade.data.loader import load_sst2_data
from splade.models.splade import SpladeModel
from splade.utils.cuda import set_seed, DEVICE
from splade.config.load import load_config
from splade.training.loop import train_model
from splade.inference import score_model

def main() -> None:
    parser = argparse.ArgumentParser(description="Train SPLADE via YAML config")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.training.seed)
    
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "resolved_config.yaml"), "w") as f:
        yaml.dump(asdict(config), f)

    train_texts, train_labels, test_texts, test_labels, num_labels = load_sst2_data(
        config.data.train_samples,
        config.data.test_samples,
        seed=config.training.seed,
    )
    config.data.num_labels = num_labels

    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    model = SpladeModel(config.model.name, config.data.num_labels).to(DEVICE)
    
    if config.model.compile:
        model = torch.compile(model, mode=config.model.compile_mode)

    train_model(model, tokenizer, train_texts, train_labels, config.training, config.model, config.data,
                val_texts=test_texts, val_labels=test_labels)

    accuracy = score_model(
        model, 
        tokenizer, 
        test_texts, 
        test_labels, 
        config.data.max_length, 
        config.training.batch_size, 
        config.data.num_labels
    )
    print(f"\nTest accuracy: {accuracy:.4f}")

    metrics_path = os.path.join(config.output_dir, "metrics.json")
    import json
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": accuracy}, f, indent=2)

    checkpoint_path = os.path.join(config.output_dir, "model.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
