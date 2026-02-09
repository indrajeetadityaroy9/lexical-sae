import glob
import os

import pytest

from splade.config.load import load_config
from splade.config.ner_schema import load_ner_config

EXPERIMENTS_DIR = "experiments"
NER_DIR = os.path.join(EXPERIMENTS_DIR, "ner")

def get_experiment_yamls():
    all_yamls = glob.glob(os.path.join(EXPERIMENTS_DIR, "**", "*.yaml"), recursive=True)
    return [y for y in all_yamls if not y.startswith(NER_DIR)]

def get_ner_yamls():
    return glob.glob(os.path.join(NER_DIR, "**", "*.yaml"), recursive=True)

@pytest.mark.parametrize("yaml_path", get_experiment_yamls())
def test_experiment_yaml_loads(yaml_path):
    config = load_config(yaml_path)
    assert config.experiment_name is not None
    assert config.data.train_samples > 0 or config.data.train_samples == -1

@pytest.mark.parametrize("yaml_path", get_ner_yamls())
def test_ner_yaml_loads(yaml_path):
    config = load_ner_config(yaml_path)
    assert config.experiment_name is not None
    assert config.data.train_samples > 0 or config.data.train_samples == -1
