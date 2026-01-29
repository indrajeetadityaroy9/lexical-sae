"""SPLADE classifier benchmark."""

import time
import argparse

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

from splade_classifier.models import SPLADEClassifier
from splade_classifier.data import load_classification_data

# Benchmark dataset configurations
_BENCHMARK_DATASETS = {
    "imdb": {"text_column": "text", "test_split": "test"},
    "sst2": {"dataset": "glue", "subset": "sst2", "text_column": "sentence", "test_split": "validation"},
    "ag_news": {"text_column": "text", "test_split": "test"},
}


def _load_benchmark_data(
    dataset: str,
    train_samples: int,
    test_samples: int,
) -> tuple[list[str], list[int], list[str], list[int], int]:
    """Load train and test data for benchmarking.

    Args:
        dataset: Dataset name ('ag_news', 'sst2', 'imdb')
        train_samples: Max training samples
        test_samples: Max test samples

    Returns:
        Tuple of (train_texts, train_labels, test_texts, test_labels, num_labels)
    """
    cfg = _BENCHMARK_DATASETS[dataset]
    hf_dataset = cfg.get("dataset", dataset)
    subset = cfg.get("subset")
    text_column = cfg["text_column"]
    test_split = cfg["test_split"]

    train_texts, train_labels, num_labels = load_classification_data(
        dataset=hf_dataset, split="train", max_samples=train_samples,
        subset=subset, text_column=text_column,
    )

    test_texts, test_labels, _ = load_classification_data(
        dataset=hf_dataset, split=test_split, max_samples=test_samples,
        subset=subset, text_column=text_column,
    )

    return train_texts, train_labels, test_texts, test_labels, num_labels


def compute_metrics(y_true: list[int], y_pred: list[int], y_proba: list[list[float]], num_labels: int) -> dict:
    """Compute comprehensive classification metrics."""
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    average_types = ['macro', 'micro', 'weighted']
    for avg in average_types:
        metrics[f'precision_{avg}'] = precision_score(y_true, y_pred, average=avg, zero_division=0)
        metrics[f'recall_{avg}'] = recall_score(y_true, y_pred, average=avg, zero_division=0)
        metrics[f'f1_{avg}'] = f1_score(y_true, y_pred, average=avg, zero_division=0)

    metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

    if num_labels == 2:
        y_score = [p[1] for p in y_proba]
        metrics['roc_auc'] = roc_auc_score(y_true, y_score)
    else:
        y_proba_array = np.array(y_proba)
        metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_proba_array, multi_class='ovr')

    return metrics


def benchmark_splade(
    dataset: str,
    train_samples: int,
    test_samples: int,
    epochs: int,
    batch_size: int,
) -> dict:
    """Run SPLADE classifier benchmark on a dataset."""
    print(f"\n{'='*60}")
    print(f"SPLADE Classifier - {dataset.upper()}")
    print(f"{'='*60}")

    print("Loading data...")
    train_texts, train_labels, test_texts, test_labels, num_labels = _load_benchmark_data(
        dataset, train_samples, test_samples
    )

    print(f"Train: {len(train_texts)}, Test: {len(test_texts)}, Classes: {num_labels}")
    print(f"Train label distribution: {np.bincount(train_labels).tolist()}")
    print(f"Test label distribution: {np.bincount(test_labels).tolist()}")

    clf = SPLADEClassifier(
        num_labels=num_labels,
        batch_size=batch_size,
        learning_rate=2e-5,
    )

    print(f"\nTraining for {epochs} epochs...")
    train_start = time.time()
    clf.fit(train_texts, train_labels, epochs=epochs)
    train_time = time.time() - train_start

    print("Running inference...")
    # Warmup runs
    for _ in range(3):
        clf.predict(test_texts[:min(100, len(test_texts))])

    infer_start = time.time()
    y_proba = clf.predict_proba(test_texts)
    y_pred = [max(range(len(p)), key=lambda i: p[i]) for p in y_proba]
    infer_time = time.time() - infer_start

    metrics = compute_metrics(test_labels, y_pred, y_proba, num_labels)
    metrics['train_time_seconds'] = train_time
    metrics['inference_time_seconds'] = infer_time
    metrics['samples_per_second'] = len(test_texts) / infer_time

    return metrics


def print_metrics(metrics: dict, name: str = "SPLADE"):
    """Pretty-print benchmark metrics."""
    print(f"\n{name} Results:")
    print("-" * 40)
    print(f"  Accuracy:           {metrics['accuracy']*100:.2f}%")
    print(f"  F1 (macro):         {metrics['f1_macro']*100:.2f}%")
    print(f"  F1 (weighted):      {metrics['f1_weighted']*100:.2f}%")
    print(f"  Precision (macro):  {metrics['precision_macro']*100:.2f}%")
    print(f"  Recall (macro):     {metrics['recall_macro']*100:.2f}%")

    if 'roc_auc' in metrics:
        print(f"  ROC-AUC:            {metrics['roc_auc']*100:.2f}%")
    elif 'roc_auc_ovr' in metrics:
        print(f"  ROC-AUC (OvR):      {metrics['roc_auc_ovr']*100:.2f}%")

    print(f"\n  Train time:         {metrics['train_time_seconds']:.1f}s")
    print(f"  Inference time:     {metrics['inference_time_seconds']:.2f}s")
    print(f"  Throughput:         {metrics['samples_per_second']:.0f} samples/s")

    print(f"\n  Per-class F1:       {[f'{f*100:.1f}%' for f in metrics['f1_per_class']]}")
    print(f"  Confusion Matrix:")
    for row in metrics['confusion_matrix']:
        print(f"    {row}")


def run_benchmark(dataset: str, train_samples: int, test_samples: int, epochs: int, batch_size: int) -> dict:
    """Run benchmark on a single dataset."""
    metrics = benchmark_splade(dataset, train_samples, test_samples, epochs, batch_size)
    print_metrics(metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Benchmark SPLADE classifier")
    parser.add_argument("--dataset", type=str, default="all", choices=["ag_news", "sst2", "imdb", "all"])
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--test-samples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    datasets = ["ag_news", "sst2", "imdb"] if args.dataset == "all" else [args.dataset]

    all_results = {}
    for dataset in datasets:
        metrics = run_benchmark(
            dataset=dataset,
            train_samples=args.train_samples,
            test_samples=args.test_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        all_results[dataset] = metrics

    if len(datasets) > 1:
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"\n{'Dataset':<12} {'Accuracy':>12} {'F1 (macro)':>12} {'Throughput':>12}")
        print("-" * 50)
        for ds in datasets:
            m = all_results[ds]
            print(f"{ds:<12} {m['accuracy']*100:>11.1f}% {m['f1_macro']*100:>11.1f}% {m['samples_per_second']:>10.0f}/s")


if __name__ == "__main__":
    main()
