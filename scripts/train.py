"""Train a SPLADE classifier.

Thin entry point — all reusable logic lives in src/.
"""

import argparse
import os

import torch

from src.data import load_benchmark_data, load_hatexplain
from src.models import SPLADEClassifier
from src.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train a SPLADE classifier")
    parser.add_argument("--dataset", type=str, default="sst2", choices=["sst2", "ag_news", "imdb", "hatexplain"])
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--test-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=None, help="Fixed epochs (default: early stopping)")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    set_seed(args.seed)

    # Load data
    if args.dataset == "hatexplain":
        train_texts, train_labels, _, num_labels = load_hatexplain("train", args.train_samples)
        test_texts, test_labels, _, _ = load_hatexplain("test", args.test_samples)
    else:
        train_texts, train_labels, test_texts, test_labels, num_labels = load_benchmark_data(
            args.dataset, args.train_samples, args.test_samples,
        )

    # Train
    clf = SPLADEClassifier(num_labels=num_labels, batch_size=args.batch_size)
    clf.fit(train_texts, train_labels, epochs=args.epochs, max_epochs=args.max_epochs)

    # Evaluate
    accuracy = clf.score(test_texts, test_labels)
    print(f"\nTest accuracy: {accuracy:.4f}")

    # Save checkpoint
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, f"splade_{args.dataset}_seed{args.seed}.pt")
    torch.save(clf.model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
