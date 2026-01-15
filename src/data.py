"""
Data loading utilities for SPLADE classifier.

Provides a unified interface for loading text classification data from:
- Local CSV/TSV files (custom datasets)
- HuggingFace datasets (standard benchmarks)

Example usage:
    # Option A: Local files
    texts, labels, meta = load_classification_data(file_path="data/train.csv")

    # Option B: HuggingFace datasets
    texts, labels, meta = load_classification_data(dataset="ag_news", split="train")

    # Using DataLoaders
    train_loader, test_loader, meta = get_data_loaders(
        train_path="data/train.csv",  # OR dataset="ag_news"
        test_path="data/test.csv",
    )
"""

from typing import Tuple, Optional, List, Dict, Any, Union
import os
import random

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


# =============================================================================
# Supported HuggingFace Datasets Registry
# =============================================================================

SUPPORTED_DATASETS: Dict[str, Dict[str, Any]] = {
    # Binary classification
    "imdb": {
        "hf_name": "imdb",
        "text_column": "text",
        "label_column": "label",
        "num_labels": 2,
        "class_names": ["Negative", "Positive"],
    },
    "sst2": {
        "hf_name": "glue",
        "subset": "sst2",
        "text_column": "sentence",
        "label_column": "label",
        "num_labels": 2,
        "class_names": ["Negative", "Positive"],
    },
    "yelp_polarity": {
        "hf_name": "yelp_polarity",
        "text_column": "text",
        "label_column": "label",
        "num_labels": 2,
        "class_names": ["Negative", "Positive"],
    },
    "rotten_tomatoes": {
        "hf_name": "rotten_tomatoes",
        "text_column": "text",
        "label_column": "label",
        "num_labels": 2,
        "class_names": ["Negative", "Positive"],
    },

    # Multi-class classification
    "ag_news": {
        "hf_name": "ag_news",
        "text_column": "text",
        "label_column": "label",
        "num_labels": 4,
        "class_names": ["World", "Sports", "Business", "Sci/Tech"],
    },
    "yelp_review_full": {
        "hf_name": "yelp_review_full",
        "text_column": "text",
        "label_column": "label",
        "num_labels": 5,
        "class_names": ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
    },
    "dbpedia_14": {
        "hf_name": "fancyzhx/dbpedia_14",
        "text_column": "content",
        "label_column": "label",
        "num_labels": 14,
        "class_names": None,
    },
    "amazon_polarity": {
        "hf_name": "amazon_polarity",
        "text_column": "content",
        "label_column": "label",
        "num_labels": 2,
        "class_names": ["Negative", "Positive"],
    },
}


# =============================================================================
# Unified Data Loading Functions
# =============================================================================

def load_classification_data(
    # Option A: Local file
    file_path: Optional[str] = None,
    text_column: str = "text",
    label_column: str = "label",
    # Option B: HuggingFace dataset
    dataset: Optional[str] = None,
    split: str = "train",
    # Common options
    max_samples: Optional[int] = None,
) -> Tuple[List[str], List[int], Dict[str, Any]]:
    """
    Load text classification data from local files OR HuggingFace datasets.

    Args:
        file_path: Path to local CSV/TSV file (Option A)
        text_column: Column name containing text (for local files)
        label_column: Column name containing labels (for local files)
        dataset: HuggingFace dataset name (Option B)
        split: Dataset split ("train", "test", "validation")
        max_samples: Limit number of samples (for debugging)

    Returns:
        Tuple of (texts, labels, metadata)
        - texts: List[str] of input texts
        - labels: List[int] of integer labels
        - metadata: Dict with num_labels, class_names, dataset info

    Examples:
        # Option A: Local CSV/TSV file
        texts, labels, meta = load_classification_data(file_path="data/train.csv")

        # Option B: HuggingFace dataset
        texts, labels, meta = load_classification_data(dataset="ag_news", split="train")

    Raises:
        ValueError: If neither file_path nor dataset is specified
    """
    if file_path is not None:
        # Option A: Load from local file
        texts, labels, metadata = _load_from_file(
            file_path, text_column, label_column
        )
    elif dataset is not None:
        # Option B: Load from HuggingFace
        texts, labels, metadata = _load_from_huggingface(dataset, split)
    else:
        raise ValueError(
            "Must specify either 'file_path' (for local files) or "
            "'dataset' (for HuggingFace datasets)"
        )

    # Sample if requested
    if max_samples is not None and len(texts) > max_samples:
        random.seed(42)  # Fixed seed for reproducibility
        indices = random.sample(range(len(texts)), max_samples)
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]
        metadata["num_samples"] = max_samples
    else:
        metadata["num_samples"] = len(texts)

    return texts, labels, metadata


def _load_from_file(
    file_path: str,
    text_column: str,
    label_column: str,
) -> Tuple[List[str], List[int], Dict[str, Any]]:
    """Load data from a local CSV/TSV file."""
    # Auto-detect separator based on file extension
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".tsv", ".txt"):
        sep = "\t"
    else:
        sep = ","

    # Try loading with header first, fall back to no header
    try:
        df = pd.read_csv(file_path, sep=sep)
        if text_column not in df.columns:
            # Try without header (legacy TSV format: id, text, label)
            df = pd.read_csv(
                file_path, sep=sep, header=None,
                names=["id", text_column, label_column]
            )
    except Exception:
        # Fall back to no header
        df = pd.read_csv(
            file_path, sep=sep, header=None,
            names=["id", text_column, label_column]
        )

    texts = df[text_column].astype(str).tolist()
    labels = df[label_column].astype(int).tolist()

    # Infer num_labels
    unique_labels = set(labels)
    num_labels = max(unique_labels) + 1

    metadata = {
        "source": "file",
        "file_path": file_path,
        "num_labels": num_labels,
        "class_names": None,
    }

    return texts, labels, metadata


def _load_from_huggingface(
    dataset_name: str,
    split: str,
) -> Tuple[List[str], List[int], Dict[str, Any]]:
    """Load data from HuggingFace datasets."""
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets library is required for loading HuggingFace datasets. "
            "Install with: pip install datasets"
        )

    # Get config from registry or use defaults
    if dataset_name in SUPPORTED_DATASETS:
        config = SUPPORTED_DATASETS[dataset_name]
        hf_name = config["hf_name"]
        subset = config.get("subset")
        text_column = config["text_column"]
        label_column = config["label_column"]
        num_labels = config["num_labels"]
        class_names = config.get("class_names")
    else:
        # Assume it's a direct HuggingFace dataset name
        hf_name = dataset_name
        subset = None
        text_column = "text"
        label_column = "label"
        num_labels = None
        class_names = None

    # Load dataset
    if subset:
        ds = hf_load_dataset(hf_name, subset, split=split)
    else:
        ds = hf_load_dataset(hf_name, split=split)

    # Extract texts and labels
    texts = list(ds[text_column])
    labels = list(ds[label_column])

    # Ensure labels are integers
    labels = [int(l) for l in labels]

    # Infer num_labels if not in registry
    if num_labels is None:
        num_labels = max(labels) + 1

    metadata = {
        "source": "huggingface",
        "dataset": dataset_name,
        "split": split,
        "num_labels": num_labels,
        "class_names": class_names,
    }

    return texts, labels, metadata


def list_supported_datasets() -> List[str]:
    """Return list of supported HuggingFace dataset names."""
    return list(SUPPORTED_DATASETS.keys())


# =============================================================================
# PyTorch Dataset Class
# =============================================================================

class TextClassificationDataset(Dataset):
    """
    PyTorch Dataset for text classification with transformer tokenization.

    Args:
        texts: List of text strings
        labels: List of integer labels
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        num_labels: Number of classes (1 for binary, >1 for multi-class)
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 128,
        num_labels: int = 1,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_labels = num_labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Return appropriate label dtype based on classification type
        if self.num_labels == 1:
            # Binary classification: float for BCE loss
            label_tensor = torch.tensor(label, dtype=torch.float32)
        else:
            # Multi-class classification: long for CrossEntropy loss
            label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": label_tensor,
        }


# =============================================================================
# DataLoader Factory
# =============================================================================

def get_data_loaders(
    # Option A: Local files
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    text_column: str = "text",
    label_column: str = "label",
    # Option B: HuggingFace dataset
    dataset: Optional[str] = None,
    # Common options
    batch_size: int = 32,
    max_length: int = 128,
    num_workers: int = 0,
    tokenizer_name: str = "distilbert-base-uncased",
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Create train and test DataLoaders for text classification.

    Supports two modes:
    1. Local files: Specify train_path and test_path
    2. HuggingFace: Specify dataset name

    Args:
        train_path: Path to training CSV/TSV file (Option A)
        test_path: Path to test CSV/TSV file (Option A)
        text_column: Column name for text (local files)
        label_column: Column name for labels (local files)
        dataset: HuggingFace dataset name (Option B)
        batch_size: Batch size for DataLoaders
        max_length: Maximum sequence length for tokenization
        num_workers: Number of workers for data loading
        tokenizer_name: HuggingFace tokenizer name
        max_train_samples: Limit training samples (for debugging)
        max_test_samples: Limit test samples (for debugging)

    Returns:
        (train_loader, test_loader, metadata) tuple

    Examples:
        # Option A: Local files
        train_loader, test_loader, meta = get_data_loaders(
            train_path="data/train.csv",
            test_path="data/test.csv",
        )

        # Option B: HuggingFace dataset
        train_loader, test_loader, meta = get_data_loaders(
            dataset="ag_news",
            max_train_samples=10000,
        )

    Raises:
        ValueError: If neither local files nor dataset is specified
    """
    # Validate inputs
    has_local_files = train_path is not None and test_path is not None
    has_hf_dataset = dataset is not None

    if not has_local_files and not has_hf_dataset:
        raise ValueError(
            "Must specify either (train_path, test_path) for local files "
            "or 'dataset' for HuggingFace datasets"
        )

    if has_local_files and has_hf_dataset:
        raise ValueError(
            "Cannot specify both local files and HuggingFace dataset. "
            "Choose one option."
        )

    # Load data
    if has_local_files:
        # Option A: Local files
        train_texts, train_labels, train_meta = load_classification_data(
            file_path=train_path,
            text_column=text_column,
            label_column=label_column,
            max_samples=max_train_samples,
        )
        test_texts, test_labels, test_meta = load_classification_data(
            file_path=test_path,
            text_column=text_column,
            label_column=label_column,
            max_samples=max_test_samples,
        )
        # Use the larger num_labels from train/test
        num_labels = max(train_meta["num_labels"], test_meta["num_labels"])
        class_names = None
        source = f"files: {train_path}, {test_path}"
    else:
        # Option B: HuggingFace dataset
        train_texts, train_labels, train_meta = load_classification_data(
            dataset=dataset,
            split="train",
            max_samples=max_train_samples,
        )
        test_texts, test_labels, test_meta = load_classification_data(
            dataset=dataset,
            split="test",
            max_samples=max_test_samples,
        )
        num_labels = train_meta["num_labels"]
        class_names = train_meta.get("class_names")
        source = f"huggingface: {dataset}"

    # For binary classification with 2 classes, use num_labels=1 (BCE loss)
    model_num_labels = 1 if num_labels == 2 else num_labels

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Create datasets
    train_dataset = TextClassificationDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=max_length,
        num_labels=model_num_labels,
    )

    test_dataset = TextClassificationDataset(
        texts=test_texts,
        labels=test_labels,
        tokenizer=tokenizer,
        max_length=max_length,
        num_labels=model_num_labels,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Build metadata
    metadata = {
        "source": source,
        "num_labels": model_num_labels,
        "num_classes": num_labels,
        "class_names": class_names,
        "train_samples": len(train_texts),
        "test_samples": len(test_texts),
        "batch_size": batch_size,
        "max_length": max_length,
    }

    return train_loader, test_loader, metadata
