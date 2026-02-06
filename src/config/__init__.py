"""Configuration: evaluation parameters and experiment configs."""

from src.config.eval_config import EvalConfig


def load_experiment_config(path: str) -> dict:
    """Load a YAML experiment configuration file."""
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)
