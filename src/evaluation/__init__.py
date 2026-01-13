"""
Evaluation module for SPLADE models.

Provides metrics computation and benchmark evaluation:
- Standard classification metrics (accuracy, F1, sparsity)
- BEIR benchmark for information retrieval evaluation
"""

from .metrics import evaluate
from .beir_eval import (
    BEIRDataset,
    BEIRResults,
    SPLADERetriever,
    evaluate_on_beir,
    evaluate_on_multiple_datasets,
    BEIR_DATASETS,
    BEIR_QUICK_DATASETS,
)

__all__ = [
    # Classification metrics
    "evaluate",
    # BEIR benchmark
    "BEIRDataset",
    "BEIRResults",
    "SPLADERetriever",
    "evaluate_on_beir",
    "evaluate_on_multiple_datasets",
    "BEIR_DATASETS",
    "BEIR_QUICK_DATASETS",
]
