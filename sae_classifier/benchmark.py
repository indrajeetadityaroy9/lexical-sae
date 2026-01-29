"""Classification benchmark for SAE classifier."""

import time
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from sae_classifier.data import load_classification_data
from sae_classifier.models import SAEClassifier, set_seed


@dataclass
class BenchmarkMetrics:
    """Classification benchmark metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_macro: float
    f1_weighted: float
    train_time: float
    inference_time: float
    samples_per_second: float


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    """Compute classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def benchmark_dataset(
    dataset: str,
    subset: str | None = None,
    train_samples: int = 5000,
    test_samples: int = 1000,
    epochs: int = 3,
    sae_expansion: int = 8,
    sae_k: int = 32,
    seed: int = 42,
    device: str = "auto",
    text_column: str = "text",
) -> BenchmarkMetrics:
    """Run benchmark on a single dataset.

    Args:
        dataset: HuggingFace dataset name
        subset: Dataset subset (e.g., "sst2" for glue)
        train_samples: Number of training samples
        test_samples: Number of test samples
        epochs: Training epochs
        sae_expansion: SAE expansion factor
        sae_k: SAE sparsity level
        seed: Random seed
        device: Device to use
        text_column: Name of text column

    Returns:
        BenchmarkMetrics with results
    """
    set_seed(seed)

    # Load data
    print(f"\nLoading {dataset}" + (f"/{subset}" if subset else "") + "...")
    texts, labels, num_labels = load_classification_data(
        dataset, subset=subset, split="train", max_samples=train_samples,
        text_column=text_column
    )

    # Try different test splits
    try:
        test_texts, test_labels, _ = load_classification_data(
            dataset, subset=subset, split="test", max_samples=test_samples,
            text_column=text_column
        )
    except:
        test_texts, test_labels, _ = load_classification_data(
            dataset, subset=subset, split="validation", max_samples=test_samples,
            text_column=text_column
        )

    print(f"Train: {len(texts)}, Test: {len(test_texts)}, Labels: {num_labels}")

    # Train
    clf = SAEClassifier(
        num_labels=num_labels,
        sae_expansion=sae_expansion,
        sae_k=sae_k,
        epochs=epochs,
        device=device,
    )

    start_time = time.time()
    clf.fit(texts, labels)
    train_time = time.time() - start_time

    # Predict
    start_time = time.time()
    predictions = clf.predict(test_texts)
    inference_time = time.time() - start_time

    # Metrics
    metrics = compute_metrics(test_labels, predictions)

    return BenchmarkMetrics(
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1_macro=metrics["f1_macro"],
        f1_weighted=metrics["f1_weighted"],
        train_time=train_time,
        inference_time=inference_time,
        samples_per_second=len(test_texts) / inference_time,
    )


def print_metrics(name: str, metrics: BenchmarkMetrics) -> None:
    """Print benchmark metrics."""
    print(f"\n{'=' * 50}")
    print(f"Dataset: {name}")
    print("=" * 50)
    print(f"Accuracy:     {metrics.accuracy:.4f}")
    print(f"Precision:    {metrics.precision:.4f}")
    print(f"Recall:       {metrics.recall:.4f}")
    print(f"F1 (macro):   {metrics.f1_macro:.4f}")
    print(f"F1 (weighted):{metrics.f1_weighted:.4f}")
    print(f"Train time:   {metrics.train_time:.1f}s")
    print(f"Throughput:   {metrics.samples_per_second:.1f} samples/s")


def run_benchmark(
    datasets: list[str] | None = None,
    epochs: int = 3,
    device: str = "auto",
) -> dict[str, BenchmarkMetrics]:
    """Run benchmark on multiple datasets.

    Args:
        datasets: List of dataset names (default: sst2, imdb)
        epochs: Training epochs
        device: Device to use

    Returns:
        Dictionary mapping dataset name to metrics
    """
    if datasets is None:
        datasets = ["sst2", "imdb"]

    results = {}

    for dataset in datasets:
        if dataset == "sst2":
            metrics = benchmark_dataset(
                "glue", subset="sst2", epochs=epochs, device=device,
                text_column="sentence"
            )
        elif dataset == "imdb":
            metrics = benchmark_dataset(
                "imdb", epochs=epochs, device=device
            )
        elif dataset == "ag_news":
            metrics = benchmark_dataset(
                "ag_news", epochs=epochs, device=device
            )
        else:
            metrics = benchmark_dataset(dataset, epochs=epochs, device=device)

        print_metrics(dataset, metrics)
        results[dataset] = metrics

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"{'Dataset':<15} {'Accuracy':<12} {'F1':<12}")
    print("-" * 40)
    for name, m in results.items():
        print(f"{name:<15} {m.accuracy:<12.4f} {m.f1_macro:<12.4f}")

    return results


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="SAE classifier benchmark")
    parser.add_argument("--dataset", default="sst2", help="Dataset to benchmark")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")

    args = parser.parse_args()

    if args.all:
        run_benchmark(["sst2", "imdb"], epochs=args.epochs, device=args.device)
    else:
        run_benchmark([args.dataset], epochs=args.epochs, device=args.device)


if __name__ == "__main__":
    main()
