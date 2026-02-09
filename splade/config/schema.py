from dataclasses import dataclass, field


@dataclass
class DataConfig:
    dataset_name: str = "sst2"
    train_samples: int = -1  # -1 = use full split
    test_samples: int = -1   # -1 = use full split


@dataclass
class ModelConfig:
    name: str = "answerdotai/ModernBERT-base"


@dataclass
class TrainingConfig:
    target_accuracy: float | None = None  # GECO tau override; None = auto from warmup
    sparsity_target: float = 0.1          # circuit_fraction (top k% of active dims)
    warmup_fraction: float = 0.2          # fraction of training for CE-only warmup


@dataclass
class MechanisticConfig:
    circuit_fraction: float = 0.1
    sae_comparison: bool = False


@dataclass
class EvaluationConfig:
    seeds: list[int] = field(default_factory=lambda: [42])


@dataclass
class Config:
    experiment_name: str
    output_dir: str
    data: DataConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    mechanistic: MechanisticConfig = field(default_factory=MechanisticConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
