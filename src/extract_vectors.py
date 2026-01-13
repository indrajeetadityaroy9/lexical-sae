"""
Extract SPLADE sparse vectors from a trained model for SAE training.

Usage:
    python -m src.extract_vectors --model_path models/model.pth --data_dir Data --output_path outputs/vectors.pt
"""

import torch
import argparse
import os
from tqdm import tqdm

from src.models import DistilBERTSparseClassifier
from src.data import get_data_loaders


def extract_vectors(model_path: str, data_dir: str, batch_size: int = 32):
    """
    Extract sparse vectors from all samples in train and test sets.

    Args:
        model_path: Path to trained model weights
        data_dir: Directory containing data files
        batch_size: Batch size for extraction

    Returns:
        dict with 'train_vectors', 'test_vectors', 'train_labels', 'test_labels', 'train_texts', 'test_texts'
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = DistilBERTSparseClassifier()
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")

    # Load data
    train_loader, test_loader = get_data_loaders(
        os.path.join(data_dir, 'movie_reviews_train.txt'),
        os.path.join(data_dir, 'movie_reviews_test.txt'),
        batch_size=batch_size
    )

    def extract_from_loader(loader, desc="Extracting"):
        all_vectors = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label']

                # Forward pass to get sparse vectors
                _, sparse_vec = model(input_ids, attention_mask)

                all_vectors.append(sparse_vec.cpu())
                all_labels.append(labels)

        return torch.cat(all_vectors, dim=0), torch.cat(all_labels, dim=0)

    print("\nExtracting training vectors...")
    train_vectors, train_labels = extract_from_loader(train_loader, "Train")

    print("\nExtracting test vectors...")
    test_vectors, test_labels = extract_from_loader(test_loader, "Test")

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
        'vocab_size': train_vectors.shape[1]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/model.pth')
    parser.add_argument('--data_dir', type=str, default='Data')
    parser.add_argument('--output_path', type=str, default='outputs/vectors.pt')
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Extract vectors
    results = extract_vectors(args.model_path, args.data_dir, args.batch_size)

    # Save
    torch.save(results, args.output_path)
    print(f"\nVectors saved to {args.output_path}")
