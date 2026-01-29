"""Baseline explainers for interpretability comparison."""

from splade_classifier.baselines.attention import AttentionExplainer
from splade_classifier.baselines.lime_explainer import LIMEExplainer
from splade_classifier.baselines.shap_explainer import SHAPExplainer
from splade_classifier.baselines.integrated_gradients import IntegratedGradientsExplainer

__all__ = [
    "AttentionExplainer",
    "LIMEExplainer",
    "SHAPExplainer",
    "IntegratedGradientsExplainer",
]
