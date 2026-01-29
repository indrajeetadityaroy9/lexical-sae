"""SAE-based interpretable text classifier.

This package provides Sparse Autoencoder (SAE) based interpretability
for transformer text classifiers. SAE features are learned from model
activations and provide faithful explanations by construction.
"""

__version__ = "0.1.0"

# Core SAE components
from sae_classifier.sae import (
    BatchTopKSAE,
    TopKSAE,
    create_sae,
    SAETrainer,
    collect_activations,
    load_sae,
)

# Data loading
from sae_classifier.data import (
    load_classification_data,
    load_hatexplain,
    rationale_agreement,
)

# Faithfulness metrics
from sae_classifier.faithfulness import (
    comprehensiveness,
    sufficiency,
    monotonicity,
    aopc,
)

# Intervention metrics
from sae_classifier.interventions import (
    activation_patching,
    feature_ablation_curve,
    feature_necessity,
    feature_sufficiency,
    steering_effectiveness,
    compute_intervention_metrics,
)

# Classifier
from sae_classifier.models import SAEClassifier, set_seed

# Utilities
from sae_classifier.adaptive import (
    StableOps,
    compute_base_lr,
    AdaptiveLRScheduler,
)

__all__ = [
    # Classifier
    "SAEClassifier",
    "set_seed",
    # SAE
    "BatchTopKSAE",
    "TopKSAE",
    "create_sae",
    "SAETrainer",
    "collect_activations",
    "load_sae",
    # Data
    "load_classification_data",
    "load_hatexplain",
    "rationale_agreement",
    # Faithfulness metrics
    "comprehensiveness",
    "sufficiency",
    "monotonicity",
    "aopc",
    # Intervention metrics
    "activation_patching",
    "feature_ablation_curve",
    "feature_necessity",
    "feature_sufficiency",
    "steering_effectiveness",
    "compute_intervention_metrics",
    # Utilities
    "StableOps",
    "compute_base_lr",
    "AdaptiveLRScheduler",
]
