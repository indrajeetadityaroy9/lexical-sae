"""Minimal configuration schema.

All paper-mandated constants live in splade/training/constants.py and
splade/evaluation/constants.py. This schema contains only the values
that genuinely vary across experiments: dataset identity, model backbone,
evaluation seeds, and which explainers to benchmark.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    dataset_name: str = "sst2"
    train_samples: int = 2000
    test_samples: int = 200


@dataclass
class ModelConfig:
    name: str = "distilbert-base-uncased"


@dataclass
class EvaluationConfig:
    seeds: List[int] = field(default_factory=lambda: [42])
    explainers: List[str] = field(default_factory=lambda: [
        "splade", "lime", "ig", "gradient_shap", "attention",
        "saliency", "deeplift", "random",
    ])


@dataclass
class Config:
    experiment_name: str
    output_dir: str
    data: DataConfig
    model: ModelConfig
    evaluation: EvaluationConfig
