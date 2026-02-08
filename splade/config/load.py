import yaml

from splade.config.schema import (Config, DataConfig, EvaluationConfig,
                                  ModelConfig)


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw_config = yaml.safe_load(f)

    return Config(
        experiment_name=raw_config["experiment_name"],
        output_dir=raw_config["output_dir"],
        data=DataConfig(**raw_config["data"]),
        model=ModelConfig(**raw_config["model"]),
        evaluation=EvaluationConfig(**raw_config["evaluation"]),
    )
