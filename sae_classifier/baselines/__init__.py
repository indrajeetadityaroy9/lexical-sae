"""Baseline explainers for interpretability comparison.

Note: LIME, SHAP, and Integrated Gradients require optional dependencies.
Install with: pip install lime shap captum
"""

from sae_classifier.baselines.base import BaseExplainer
from sae_classifier.baselines.attention import AttentionExplainer

# Lazy imports for optional dependencies
def __getattr__(name):
    if name == "LIMEExplainer":
        from sae_classifier.baselines.lime_explainer import LIMEExplainer
        return LIMEExplainer
    elif name == "SHAPExplainer":
        from sae_classifier.baselines.shap_explainer import SHAPExplainer
        return SHAPExplainer
    elif name == "IntegratedGradientsExplainer":
        from sae_classifier.baselines.integrated_gradients import IntegratedGradientsExplainer
        return IntegratedGradientsExplainer
    elif name == "SPLADEExplainer":
        from sae_classifier.baselines.splade_explainer import SPLADEExplainer
        return SPLADEExplainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BaseExplainer",
    "AttentionExplainer",
    "LIMEExplainer",
    "SHAPExplainer",
    "IntegratedGradientsExplainer",
    "SPLADEExplainer",
]
