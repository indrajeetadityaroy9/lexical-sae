import glob
import os

import pytest

from splade.config.load import load_config

EXPERIMENTS_DIR = "experiments"

def get_experiment_yamls():
    return glob.glob(os.path.join(EXPERIMENTS_DIR, "**", "*.yaml"), recursive=True)

@pytest.mark.parametrize("yaml_path", get_experiment_yamls())
def test_experiment_yaml_loads(yaml_path):
    config = load_config(yaml_path)
    assert config.experiment_name is not None
    assert config.data.train_samples > 0
