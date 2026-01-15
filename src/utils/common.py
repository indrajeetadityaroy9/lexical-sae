"""
Common utilities for the SPLADE project.

Consolidates duplicated patterns across CLI scripts:
- Device initialization
- Data source validation
"""

from typing import Tuple, Optional

import torch

from src.data import list_supported_datasets


def get_device() -> torch.device:
    """
    Get the best available device (CUDA if available, else CPU).

    Returns:
        torch.device: CUDA device if available, otherwise CPU
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validate_data_sources(
    train_path: Optional[str],
    test_path: Optional[str],
    dataset: Optional[str],
    raise_on_error: bool = True,
    print_error: bool = False,
) -> Tuple[bool, bool]:
    """
    Validate that exactly one data source is specified.

    Args:
        train_path: Path to training CSV/TSV file
        test_path: Path to test CSV/TSV file
        dataset: HuggingFace dataset name
        raise_on_error: Raise ValueError if no valid source
        print_error: Print error message to stdout

    Returns:
        Tuple of (has_local_files, has_huggingface_dataset)

    Raises:
        ValueError: If raise_on_error=True and no valid source specified
    """
    has_local = train_path is not None and test_path is not None
    has_hf = dataset is not None

    if not has_local and not has_hf:
        if print_error:
            print("\nError: Must specify either:")
            print("  --train_path and --test_path  (for local CSV/TSV files)")
            print("  --dataset                      (for HuggingFace datasets)")
            print(f"\nSupported HuggingFace datasets: {list_supported_datasets()}")
        if raise_on_error:
            raise ValueError(
                "Must specify either (train_path, test_path) for local files "
                "or 'dataset' for HuggingFace datasets"
            )

    return has_local, has_hf


__all__ = [
    "get_device",
    "validate_data_sources",
]
