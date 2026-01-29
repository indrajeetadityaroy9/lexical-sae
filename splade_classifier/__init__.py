"""SPLADE classifier - Sparse lexical text classification with interpretability."""

from splade_classifier.models import SPLADEClassifier, set_seed
from splade_classifier.data import load_classification_data, load_hatexplain, rationale_agreement
from splade_classifier.faithfulness import comprehensiveness, sufficiency, monotonicity, aopc

__all__ = [
    "SPLADEClassifier",
    "set_seed",
    "load_classification_data",
    "load_hatexplain",
    "rationale_agreement",
    "comprehensiveness",
    "sufficiency",
    "monotonicity",
    "aopc",
]
