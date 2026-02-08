"""Load configuration from YAML files."""

import warnings

import yaml
import os
from splade.config.schema import Config, DataConfig, ModelConfig, EvaluationConfig


def load_config(path: str) -> Config:
    """Load configuration from a YAML file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw_config = yaml.safe_load(f)

    required_sections = {"experiment_name", "output_dir", "data", "model", "evaluation"}
    missing = required_sections - set(raw_config.keys())
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")

    def load_section(cls, data):
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
        evaluation=load_section(EvaluationConfig, raw_config["evaluation"]),
    )
