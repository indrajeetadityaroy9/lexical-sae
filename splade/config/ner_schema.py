"""NER experiment configuration schema.

Mirrors splade/config/schema.py structure but with NER-specific defaults.
"""

from dataclasses import dataclass, field

import yaml


@dataclass
class NERDataConfig:
    dataset_name: str = "conll2003"
    train_samples: int = -1  # -1 = use full split
    test_samples: int = -1


@dataclass
class NERModelConfig:
    name: str = "answerdotai/ModernBERT-base"


@dataclass
class NERTrainingConfig:
    target_accuracy: float | None = None  # GECO tau override; None = auto
    sparsity_target: float = 0.1
    warmup_fraction: float = 0.2
    batch_size: int | None = None  # None = auto-infer
    gradient_accumulation_steps: int = 1


@dataclass
class NEREvaluationConfig:
    seeds: list[int] = field(default_factory=lambda: [42])


@dataclass
class NERConfig:
    experiment_name: str
    output_dir: str
    data: NERDataConfig
    model: NERModelConfig
    evaluation: NEREvaluationConfig
    training: NERTrainingConfig = field(default_factory=NERTrainingConfig)


def load_ner_config(path: str) -> NERConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return NERConfig(
        experiment_name=raw["experiment_name"],
        output_dir=raw["output_dir"],
        data=NERDataConfig(**raw.get("data", {})),
        model=NERModelConfig(**raw.get("model", {})),
        evaluation=NEREvaluationConfig(**raw.get("evaluation", {})),
        training=NERTrainingConfig(**raw.get("training", {})),
    )
