"""
Experiment configuration schemas using Pydantic.

Provides type-safe, validated configuration for all aspects of training,
model architecture, sparsity regularization, and interpretability analysis.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional, List


class ModelConfig(BaseModel):
    """Configuration for model architecture."""

    backbone: Literal["distilbert", "mistral", "llama"] = Field(
        default="distilbert",
        description="Base transformer backbone"
    )
    model_name: str = Field(
        default="distilbert-base-uncased",
        description="HuggingFace model identifier"
    )
    max_seq_length: int = Field(
        default=128,
        ge=16,
        le=4096,
        description="Maximum sequence length for tokenization"
    )
    use_lora: bool = Field(
        default=False,
        description="Use LoRA for efficient fine-tuning"
    )
    lora_r: int = Field(
        default=16,
        ge=4,
        le=64,
        description="LoRA rank"
    )
    lora_alpha: int = Field(
        default=32,
        description="LoRA alpha scaling factor"
    )
    lora_dropout: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="LoRA dropout rate"
    )
    lora_target_modules: List[str] = Field(
        default=["q_proj", "v_proj"],
        description="Target modules for LoRA"
    )


class SparsityConfig(BaseModel):
    """Configuration for sparsity regularization."""

    regularizer: Literal["flops", "l1", "block_l1"] = Field(
        default="flops",
        description="Type of sparsity regularizer"
    )
    lambda_reg: float = Field(
        default=1e-4,
        ge=0.0,
        description="Regularization strength"
    )
    target_sparsity: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Target sparsity level (fraction of zeros)"
    )
    structured: bool = Field(
        default=False,
        description="Use structured (2:4) sparsity"
    )
    block_size: int = Field(
        default=4,
        ge=1,
        description="Block size for block L1 regularization"
    )
    sparsity_warmup_epochs: int = Field(
        default=0,
        ge=0,
        description="Epochs to linearly increase sparsity penalty"
    )


class SAEConfig(BaseModel):
    """Configuration for Sparse Autoencoder interpretability."""

    enabled: bool = Field(
        default=True,
        description="Whether to train SAE for interpretability"
    )
    num_features: int = Field(
        default=16384,
        ge=1024,
        description="Number of SAE latent features"
    )
    expansion_factor: int = Field(
        default=4,
        ge=1,
        description="SAE hidden dim = input_dim * expansion_factor"
    )
    sparsity_coefficient: float = Field(
        default=1e-3,
        ge=0.0,
        description="L1 penalty on SAE activations"
    )
    activation: Literal["relu", "topk", "gelu"] = Field(
        default="topk",
        description="SAE activation function"
    )
    k: int = Field(
        default=32,
        ge=1,
        description="Number of active features for TopK activation"
    )
    learning_rate: float = Field(
        default=1e-4,
        description="SAE-specific learning rate"
    )
    epochs: int = Field(
        default=10,
        ge=1,
        description="SAE training epochs"
    )
    batch_size: int = Field(
        default=256,
        ge=1,
        description="SAE training batch size"
    )


class TrainingConfig(BaseModel):
    """Configuration for model training."""

    batch_size: int = Field(
        default=32,
        ge=1,
        description="Training batch size"
    )
    epochs: int = Field(
        default=10,
        ge=1,
        description="Number of training epochs"
    )
    lr: float = Field(
        default=2e-5,
        gt=0,
        description="Learning rate"
    )
    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
        description="AdamW weight decay"
    )
    warmup_steps: int = Field(
        default=100,
        ge=0,
        description="Learning rate warmup steps"
    )
    warmup_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Warmup as fraction of total steps (overrides warmup_steps if > 0)"
    )
    grad_clip: float = Field(
        default=1.0,
        ge=0.0,
        description="Gradient clipping norm (0 to disable)"
    )
    use_wandb: bool = Field(
        default=False,
        description="Enable Weights & Biases logging"
    )
    wandb_project: str = Field(
        default="splade-phd",
        description="W&B project name"
    )
    save_every_n_epochs: int = Field(
        default=1,
        ge=1,
        description="Checkpoint save frequency"
    )
    eval_every_n_steps: int = Field(
        default=0,
        ge=0,
        description="Evaluation frequency (0 = only at epoch end)"
    )
    fp16: bool = Field(
        default=False,
        description="Use FP16 mixed precision training"
    )
    bf16: bool = Field(
        default=False,
        description="Use BF16 mixed precision training"
    )


class DataConfig(BaseModel):
    """Configuration for data loading."""

    dataset: str = Field(
        default="movie_reviews",
        description="Dataset name"
    )
    data_dir: str = Field(
        default="Data",
        description="Directory containing data files"
    )
    train_file: str = Field(
        default="movie_reviews_train.txt",
        description="Training data filename"
    )
    test_file: str = Field(
        default="movie_reviews_test.txt",
        description="Test data filename"
    )
    dev_file: Optional[str] = Field(
        default="movie_reviews_dev.txt",
        description="Development/validation data filename"
    )
    num_workers: int = Field(
        default=4,
        ge=0,
        description="DataLoader worker processes"
    )


class BEIRConfig(BaseModel):
    """Configuration for BEIR benchmark evaluation."""

    enabled: bool = Field(
        default=False,
        description="Enable BEIR evaluation"
    )
    datasets: List[str] = Field(
        default=["nfcorpus", "scifact", "fiqa"],
        description="BEIR datasets to evaluate on"
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size for BEIR encoding"
    )
    top_k: int = Field(
        default=100,
        ge=1,
        description="Number of documents to retrieve"
    )


class DistillationConfig(BaseModel):
    """Configuration for knowledge distillation."""

    enabled: bool = Field(
        default=False,
        description="Enable knowledge distillation"
    )
    teacher_model: str = Field(
        default="mistralai/Mistral-7B-v0.1",
        description="Teacher model for distillation"
    )
    temperature: float = Field(
        default=4.0,
        ge=1.0,
        description="Distillation temperature"
    )
    alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Balance between KD loss and task loss"
    )
    feature_distill: bool = Field(
        default=True,
        description="Whether to match sparse vectors"
    )
    feature_weight: float = Field(
        default=0.1,
        ge=0.0,
        description="Weight for feature matching loss"
    )


class ExperimentConfig(BaseModel):
    """Master configuration combining all sub-configs."""

    name: str = Field(
        default="baseline",
        description="Experiment name for logging"
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    output_dir: str = Field(
        default="outputs",
        description="Directory for outputs and checkpoints"
    )

    model: ModelConfig = Field(default_factory=ModelConfig)
    sparsity: SparsityConfig = Field(default_factory=SparsityConfig)
    sae: SAEConfig = Field(default_factory=SAEConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    beir: BEIRConfig = Field(default_factory=BEIRConfig)
    distillation: DistillationConfig = Field(default_factory=DistillationConfig)

    class Config:
        extra = "forbid"  # Raise error on unknown fields


def load_config(config_path: str) -> ExperimentConfig:
    """Load configuration from YAML file."""
    from omegaconf import OmegaConf

    yaml_config = OmegaConf.load(config_path)
    dict_config = OmegaConf.to_container(yaml_config, resolve=True)
    return ExperimentConfig(**dict_config)


def save_config(config: ExperimentConfig, config_path: str) -> None:
    """Save configuration to YAML file."""
    from omegaconf import OmegaConf
    import json

    # Convert Pydantic model to dict, then to OmegaConf
    config_dict = json.loads(config.model_dump_json())
    omega_conf = OmegaConf.create(config_dict)
    OmegaConf.save(omega_conf, config_path)
