"""
Benchmark comparing sklearn TF-IDF vs Neural SPLADE classifier.

Both implementations now use the same sklearn-compatible API:

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
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.models import SPLADEClassifier
from src.utils import load_stopwords, simple_tokenizer


def load_data(file_path):
    """Load text data from TSV file."""
    df = pd.read_csv(file_path, sep='\t', header=None, names=['id', 'review', 'label'])
    return df['review'].tolist(), df['label'].values


def train_sklearn_baseline(train_texts, train_labels, test_texts, test_labels):
    """Train and evaluate sklearn TF-IDF + Logistic Regression baseline."""
    print("\n" + "="*60)
    print("SKLEARN TF-IDF BASELINE")
    print("="*60)

    start_time = time.time()

    # 1. Vectorize
    stopwords = load_stopwords()
    tokenizer_func = lambda text: simple_tokenizer(text, stopwords)
    vectorizer = TfidfVectorizer(max_features=1000, tokenizer=tokenizer_func, token_pattern=None)

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    # 2. Train Classifier
    clf = LogisticRegression()
    clf.fit(X_train, train_labels)

    train_time = time.time() - start_time

    # 3. Evaluate
    start_inf = time.time()
    preds = clf.predict(X_test)
    inf_time = time.time() - start_inf

    acc = accuracy_score(test_labels, preds)
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


def train_splade_classifier(train_texts, train_labels, test_texts, test_labels, epochs=5, model_path=None):
    """Train and evaluate Neural SPLADE classifier."""
    print("\n" + "="*60)
    print("NEURAL SPLADE CLASSIFIER")
    print("="*60)

    # Create classifier with sklearn-like API
    clf = SPLADEClassifier(
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
    parser = argparse.ArgumentParser(description="Benchmark sklearn vs SPLADE")
    parser.add_argument('--data_dir', type=str, default='Data')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to save/load SPLADE model')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--skip_sklearn', action='store_true')
    parser.add_argument('--skip_splade', action='store_true')
    args = parser.parse_args()

    # Load data
    train_path = os.path.join(args.data_dir, 'movie_reviews_train.txt')
    test_path = os.path.join(args.data_dir, 'movie_reviews_test.txt')

    train_texts, train_labels = load_data(train_path)
    test_texts, test_labels = load_data(test_path)

    print(f"Train samples: {len(train_texts)}")
    print(f"Test samples:  {len(test_texts)}")

    results = []

    # Run sklearn baseline
    if not args.skip_sklearn:
        results.append(train_sklearn_baseline(
            train_texts, train_labels,
            test_texts, test_labels
        ))

    # Run SPLADE
    if not args.skip_splade:
        results.append(train_splade_classifier(
            train_texts, train_labels,
            test_texts, test_labels,
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
