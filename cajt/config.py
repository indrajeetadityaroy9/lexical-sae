from dataclasses import dataclass, field

import yaml


@dataclass
class DataConfig:
    dataset_name: str = "banking77"
    train_samples: int = -1  # -1 = use full split
    test_samples: int = -1   # -1 = use full split

    def __post_init__(self):
        if self.train_samples != -1 and self.train_samples <= 0:
            raise ValueError(f"train_samples must be -1 (full) or > 0, got {self.train_samples}")
        if self.test_samples != -1 and self.test_samples <= 0:
            raise ValueError(f"test_samples must be -1 (full) or > 0, got {self.test_samples}")


@dataclass
class ModelConfig:
    name: str = "answerdotai/ModernBERT-base"


@dataclass
class TrainingConfig:
    sparsity_target: float = 0.1
    warmup_fraction: float = 0.2
    pooling: str = "max"  # "max" or "attention"
    learning_rate: float = 3e-4
    batch_size: int | None = None       # None = auto-infer from GPU memory
    max_epochs: int = 50
    early_stop_patience: int = 10
    label_smoothing: float = 0.1
    val_fraction: int = 10              # 1/N of training data
    min_val_per_class: int = 5
    num_workers: int | None = None      # None = auto (min(cpu_count, 16))
    prefetch_factor: int = 4

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if not (0 < self.sparsity_target < 1):
            raise ValueError(f"sparsity_target must be in (0, 1), got {self.sparsity_target}")
        if self.warmup_fraction < 0:
            raise ValueError(f"warmup_fraction must be >= 0, got {self.warmup_fraction}")
        if self.pooling not in ("max", "attention"):
            raise ValueError(f"pooling must be 'max' or 'attention', got '{self.pooling}'")
        if self.max_epochs < 1:
            raise ValueError(f"max_epochs must be >= 1, got {self.max_epochs}")


@dataclass
class VPEConfig:
    enabled: bool = False
    token_ids: list[int] = field(default_factory=list)
    num_senses: int = 4

    def __post_init__(self):
        if self.num_senses < 1:
            raise ValueError(f"num_senses must be >= 1, got {self.num_senses}")


@dataclass
class EvalConfig:
    run_sparsity_frontier: bool = False
    run_transcoder_comparison: bool = False
    run_disentanglement: bool = False
    train_dense_baseline: bool = False
    spurious_token_ids: list[int] = field(default_factory=list)


@dataclass
class LongContextConfig:
    target_word_counts: list[int] = field(default_factory=lambda: [50, 100, 200, 500, 1000, 2000])
    max_token_lengths: list[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024, 2048])


@dataclass
class Config:
    experiment_name: str
    output_dir: str
    data: DataConfig
    model: ModelConfig
    experiment_type: str = "experiment"
    seed: int = 42
    training: TrainingConfig = field(default_factory=TrainingConfig)
    vpe: VPEConfig = field(default_factory=VPEConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    long_context: LongContextConfig = field(default_factory=LongContextConfig)


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    experiment_type = raw.get("experiment_type", "experiment")

    return Config(
        experiment_name=raw["experiment_name"],
        output_dir=raw["output_dir"],
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        experiment_type=experiment_type,
        seed=raw.get("seed", 42),
        training=TrainingConfig(**raw.get("training", {})),
        vpe=VPEConfig(**raw.get("vpe", {})),
        evaluation=EvalConfig(**raw.get("evaluation", {})),
        long_context=LongContextConfig(**raw.get("long_context", {})),
    )
