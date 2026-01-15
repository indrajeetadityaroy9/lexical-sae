"""
Benchmark comparing sklearn TF-IDF vs Neural SPLADE classifier.

Supports two data sources:
    Option A - Local files:
        python -m src.benchmark --train_path data/train.csv --test_path data/test.csv

    Option B - HuggingFace datasets:
        python -m src.benchmark --dataset imdb --epochs 3
        python -m src.benchmark --dataset ag_news --epochs 3

Both implementations use sklearn-compatible APIs:

    # Sklearn
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(texts)
    clf = LogisticRegression().fit(X_train, y_train)
    preds = clf.predict(X_test)

    # SPLADE (equally simple!)
    clf = SPLADEClassifier()
    clf.fit(train_texts, train_labels)
    preds = clf.predict(test_texts)
"""

import time
import argparse
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.models import SPLADEClassifier
from src.data import load_classification_data
from src.utils import load_stopwords, simple_tokenizer, validate_data_sources


def train_sklearn_baseline(train_texts, train_labels, test_texts, test_labels, num_classes=2):
    """Train and evaluate sklearn TF-IDF + Logistic Regression baseline."""
    print("\n" + "="*60)
    print("SKLEARN TF-IDF BASELINE")
    print("="*60)

    start_time = time.time()

    # 1. Vectorize
    stopwords = load_stopwords()
    tokenizer_func = lambda text: simple_tokenizer(text, stopwords)
    vectorizer = TfidfVectorizer(max_features=5000, tokenizer=tokenizer_func, token_pattern=None)

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    # 2. Train Classifier
    if num_classes > 2:
        clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
    else:
        clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, train_labels)

    train_time = time.time() - start_time

    # 3. Evaluate
    start_inf = time.time()
    preds = clf.predict(X_test)
    inf_time = time.time() - start_inf

    acc = accuracy_score(test_labels, preds)
    if num_classes > 2:
        f1 = f1_score(test_labels, preds, average='macro')
    else:
        f1 = f1_score(test_labels, preds)
    sparsity = 100.0 * (1.0 - X_test.nnz / (X_test.shape[0] * X_test.shape[1]))

    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Sparsity:  {sparsity:.2f}%")
    print(f"Train time: {train_time:.2f}s")
    print(f"Inference:  {inf_time*1000:.2f}ms")

    return {
        "model": "Sklearn TF-IDF",
        "accuracy": acc,
        "f1": f1,
        "sparsity": sparsity,
        "train_time_s": train_time,
        "inference_ms": inf_time * 1000
    }


def train_splade_classifier(
    train_texts, train_labels, test_texts, test_labels,
    num_labels=1, class_names=None, epochs=5, model_path=None
):
    """Train and evaluate Neural SPLADE classifier."""
    print("\n" + "="*60)
    print("NEURAL SPLADE CLASSIFIER")
    print("="*60)

    # Create classifier with sklearn-like API
    clf = SPLADEClassifier(
        num_labels=num_labels,
        class_names=class_names,
        batch_size=16,
        learning_rate=2e-5,
        flops_lambda=1e-4,
        verbose=True
    )

    # Load pre-trained model or train from scratch
    if model_path and os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        clf.load(model_path)
    else:
        print(f"Training for {epochs} epochs...")
        start_time = time.time()
        clf.fit(train_texts, train_labels, epochs=epochs)
        train_time = time.time() - start_time
        print(f"Train time: {train_time:.2f}s")

        # Save model
        if model_path:
            clf.save(model_path)
            print(f"Model saved to {model_path}")

    # Evaluate
    start_inf = time.time()
    preds = clf.predict(test_texts)
    inf_time = time.time() - start_inf

    acc = accuracy_score(test_labels, preds)
    if num_labels > 1:
        f1 = f1_score(test_labels, preds, average='macro')
    else:
        f1 = f1_score(test_labels, preds)
    sparsity = clf.get_sparsity(test_texts)

    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Sparsity:  {sparsity:.2f}%")
    print(f"Inference:  {inf_time*1000:.2f}ms")

    return {
        "model": "Neural SPLADE",
        "accuracy": acc,
        "f1": f1,
        "sparsity": sparsity,
        "inference_ms": inf_time * 1000
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark sklearn TF-IDF vs Neural SPLADE classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark on local CSV/TSV files
  python -m src.benchmark --train_path data/train.csv --test_path data/test.csv

  # Benchmark on HuggingFace dataset
  python -m src.benchmark --dataset imdb --epochs 3
  python -m src.benchmark --dataset ag_news --epochs 3

  # Quick benchmark with limited samples
  python -m src.benchmark --dataset yelp_polarity --max_samples 5000 --epochs 2
        """
    )

    # Data source options
    data_group = parser.add_argument_group('Data Source (choose one)')
    data_group.add_argument('--train_path', type=str, default=None,
                           help='Path to training CSV/TSV file')
    data_group.add_argument('--test_path', type=str, default=None,
                           help='Path to test CSV/TSV file')
    data_group.add_argument('--dataset', type=str, default=None,
                           help='HuggingFace dataset name (e.g., imdb, ag_news)')

    # Options
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Limit samples per split')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to save/load SPLADE model')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Training epochs for SPLADE')
    parser.add_argument('--skip_sklearn', action='store_true',
                       help='Skip sklearn baseline')
    parser.add_argument('--skip_splade', action='store_true',
                       help='Skip SPLADE classifier')

    args = parser.parse_args()

    # Validate inputs
    has_local, has_hf = validate_data_sources(
        args.train_path, args.test_path, args.dataset,
        raise_on_error=False, print_error=True
    )
    if not has_local and not has_hf:
        return

    # Load data
    print("Loading data...")
    if has_local:
        print(f"  Source: Local files")
        print(f"  Train: {args.train_path}")
        print(f"  Test: {args.test_path}")
        train_texts, train_labels, train_meta = load_classification_data(
            file_path=args.train_path,
            max_samples=args.max_samples,
        )
        test_texts, test_labels, test_meta = load_classification_data(
            file_path=args.test_path,
            max_samples=args.max_samples,
        )
        num_classes = max(train_meta['num_labels'], test_meta['num_labels'])
        class_names = None
    else:
        print(f"  Source: HuggingFace dataset '{args.dataset}'")
        train_texts, train_labels, train_meta = load_classification_data(
            dataset=args.dataset,
            split="train",
            max_samples=args.max_samples,
        )
        test_texts, test_labels, test_meta = load_classification_data(
            dataset=args.dataset,
            split="test",
            max_samples=args.max_samples,
        )
        num_classes = train_meta['num_labels']
        class_names = train_meta.get('class_names')

    # For SPLADE: use num_labels=1 for binary (BCE), num_labels=N for multi-class
    num_labels = 1 if num_classes == 2 else num_classes

    print(f"  Train samples: {len(train_texts)}")
    print(f"  Test samples: {len(test_texts)}")
    print(f"  Num classes: {num_classes}")
    if class_names:
        print(f"  Classes: {class_names}")

    results = []

    # Run sklearn baseline
    if not args.skip_sklearn:
        results.append(train_sklearn_baseline(
            train_texts, train_labels,
            test_texts, test_labels,
            num_classes=num_classes
        ))

    # Run SPLADE
    if not args.skip_splade:
        results.append(train_splade_classifier(
            train_texts, train_labels,
            test_texts, test_labels,
            num_labels=num_labels,
            class_names=class_names,
            epochs=args.epochs,
            model_path=args.model_path
        ))

    # Summary
    if results:
        print("\n" + "="*60)
        print("FINAL COMPARISON")
        print("="*60)
        df = pd.DataFrame(results)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
