"""
Extract SPLADE sparse vectors from a trained model for SAE training.

Supports two data sources:
    Option A - Local files:
        python -m src.extract_vectors --model_path models/model.pth --train_path data/train.csv --test_path data/test.csv

    Option B - HuggingFace datasets:
        python -m src.extract_vectors --model_path models/model.pth --dataset imdb
"""

import sys
import argparse
import os

import torch
from tqdm import tqdm

from src.models import SPLADEClassifier
from src.data import load_classification_data
from src.utils import validate_data_sources


def extract_vectors(
    model_path: str,
    train_path: str = None,
    test_path: str = None,
    dataset: str = None,
    batch_size: int = 32,
    max_samples: int = None,
):
    """
    Extract sparse vectors from all samples in train and test sets.

    Args:
        model_path: Path to trained model weights
        train_path: Path to training CSV/TSV file (Option A)
        test_path: Path to test CSV/TSV file (Option A)
        dataset: HuggingFace dataset name (Option B)
        batch_size: Batch size for extraction
        max_samples: Limit samples per split

    Returns:
        dict with 'train_vectors', 'test_vectors', 'train_labels', 'test_labels', 'vocab_size'
    """
    # Load model
    print(f"Loading model from {model_path}...")
    clf = SPLADEClassifier(batch_size=batch_size, verbose=False)
    clf.load(model_path)
    print(f"Model loaded (num_labels={clf.num_labels})")
    print(f"Using device: {clf.device}")

    # Load data
    print("\nLoading data...")
    has_local, has_hf = validate_data_sources(train_path, test_path, dataset)

    if has_local:
        print(f"  Source: Local files")
        print(f"  Train: {train_path}")
        print(f"  Test: {test_path}")
        train_texts, train_labels, _ = load_classification_data(
            file_path=train_path, max_samples=max_samples
        )
        test_texts, test_labels, _ = load_classification_data(
            file_path=test_path, max_samples=max_samples
        )
    else:
        print(f"  Source: HuggingFace dataset '{dataset}'")
        train_texts, train_labels, _ = load_classification_data(
            dataset=dataset, split="train", max_samples=max_samples
        )
        test_texts, test_labels, _ = load_classification_data(
            dataset=dataset, split="test", max_samples=max_samples
        )

    print(f"  Train samples: {len(train_texts)}")
    print(f"  Test samples: {len(test_texts)}")

    # Extract vectors using SPLADEClassifier.transform()
    print("\nExtracting training vectors...")
    train_vectors = clf.transform(train_texts)

    print("Extracting test vectors...")
    test_vectors = clf.transform(test_texts)

    # Convert labels to tensors
    train_labels = torch.tensor(train_labels)
    test_labels = torch.tensor(test_labels)

    # Compute statistics
    train_sparsity = (train_vectors == 0).float().mean().item() * 100
    test_sparsity = (test_vectors == 0).float().mean().item() * 100
    train_nnz = (train_vectors != 0).sum(dim=1).float().mean().item()
    test_nnz = (test_vectors != 0).sum(dim=1).float().mean().item()

    print(f"\n=== Extraction Statistics ===")
    print(f"Train vectors: {train_vectors.shape}")
    print(f"Test vectors: {test_vectors.shape}")
    print(f"Train sparsity: {train_sparsity:.2f}%")
    print(f"Test sparsity: {test_sparsity:.2f}%")
    print(f"Train avg non-zeros: {train_nnz:.1f}")
    print(f"Test avg non-zeros: {test_nnz:.1f}")

    return {
        'train_vectors': train_vectors,
        'test_vectors': test_vectors,
        'train_labels': train_labels,
        'test_labels': test_labels,
        'vocab_size': train_vectors.shape[1],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract SPLADE sparse vectors from a trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from local files
  python -m src.extract_vectors --model_path models/model.pth --train_path data/train.csv --test_path data/test.csv

  # Extract from HuggingFace dataset
  python -m src.extract_vectors --model_path models/model.pth --dataset imdb

  # Extract with sample limit
  python -m src.extract_vectors --model_path models/model.pth --dataset ag_news --max_samples 5000
        """
    )

    # Data source options
    data_group = parser.add_argument_group('Data Source (choose one)')
    data_group.add_argument('--train_path', type=str, default=None,
                           help='Path to training CSV/TSV file')
    data_group.add_argument('--test_path', type=str, default=None,
                           help='Path to test CSV/TSV file')
    data_group.add_argument('--dataset', type=str, default=None,
                           help='HuggingFace dataset name')

    # Options
    parser.add_argument('--model_path', type=str, default='models/model.pth',
                       help='Path to trained model')
    parser.add_argument('--output_path', type=str, default='outputs/vectors.pt',
                       help='Path to save extracted vectors')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Limit samples per split')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')

    args = parser.parse_args()

    # Validate inputs
    has_local, has_hf = validate_data_sources(
        args.train_path, args.test_path, args.dataset,
        raise_on_error=False, print_error=True
    )
    if not has_local and not has_hf:
        sys.exit(1)

    # Create output directory
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)

    # Extract vectors
    results = extract_vectors(
        model_path=args.model_path,
        train_path=args.train_path,
        test_path=args.test_path,
        dataset=args.dataset,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )

    # Save
    torch.save(results, args.output_path)
    print(f"\nVectors saved to {args.output_path}")
