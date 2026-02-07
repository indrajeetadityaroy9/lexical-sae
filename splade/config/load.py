import warnings

import yaml
import os
from splade.config.schema import Config, DataConfig, ModelConfig, TrainingConfig, EvaluationConfig

def load_config(path: str) -> Config:
    """Load configuration from a YAML file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw_config = yaml.safe_load(f)

    # Validate top-level keys
    required_sections = {"experiment_name", "output_dir", "data", "model", "training", "evaluation"}
    missing = required_sections - set(raw_config.keys())
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")

    # Helper to load section with defaults
    def load_section(cls, data):
        # Filter keys that are not in the dataclass
        valid_keys = set(cls.__annotations__.keys())
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        unknown = set(data.keys()) - valid_keys
        if unknown:
            warnings.warn(f"Unknown config keys in {cls.__name__}: {unknown}")
        return cls(**filtered)

    return Config(
        experiment_name=raw_config["experiment_name"],
        output_dir=raw_config["output_dir"],
        data=load_section(DataConfig, raw_config["data"]),
        model=load_section(ModelConfig, raw_config["model"]),
        training=load_section(TrainingConfig, raw_config["training"]),
        evaluation=load_section(EvaluationConfig, raw_config["evaluation"]),
    )
