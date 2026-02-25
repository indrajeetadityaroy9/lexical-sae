"""Sparse Autoencoder baseline for comparison with vocabulary-grounded decomposition.

Reference: Cunningham et al. (2023) "Sparse Autoencoders Find Highly Interpretable
Features in Language Models" (arXiv:2309.08600).
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from cajt.baselines.common import (
    BASELINE_OVERCOMPLETENESS,
    train_sparse_baseline,
)
from cajt.runtime import autocast, DEVICE


class SimpleSAE(nn.Module):
    """Overcomplete sparse autoencoder with L1 penalty on hidden activations."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, hidden_activations)."""
        hidden = torch.relu(self.encoder(x))
        reconstruction = self.decoder(hidden)
        return reconstruction, hidden

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x))


def train_sae_on_splade(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
) -> SimpleSAE:
    """Train an SAE on BERT hidden states at the vocabulary projection layer.

    Collects transformed hidden states (input to the output projection layer)
    then trains an overcomplete SAE to reconstruct them.
    """


    # Collect hidden states (kept on GPU) via architecture-agnostic hook
    all_hidden = []
    for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
        with torch.inference_mode(), autocast():
            transformed = model.get_mlm_head_input(input_ids, attention_mask)
        # Use CLS token representation, keep on GPU
        all_hidden.append(transformed[:, 0, :].float())

    hidden_states = torch.cat(all_hidden, dim=0)
    input_dim = hidden_states.shape[-1]
    hidden_dim = input_dim * BASELINE_OVERCOMPLETENESS

    sae = SimpleSAE(input_dim, hidden_dim).to(DEVICE)
    train_sparse_baseline(sae, TensorDataset(hidden_states))
    return sae


def compare_sae_with_dla(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels: list[int],
    tokenizer,
) -> dict:
    """Train SAE and compare with DLA in terms of active feature counts."""


    print("Training SAE on hidden states...")
    sae = train_sae_on_splade(model, input_ids_list, attention_mask_list)

    with torch.inference_mode():
        classifier_weight = model.classifier_fc2.weight
        vocab_projector_weight = model.backbone.get_output_embeddings().weight

    dla_active_counts = []
    sae_active_counts = []

    for input_ids, attention_mask, label in zip(input_ids_list, attention_mask_list, labels):
        with torch.inference_mode(), autocast():
            sparse_seq, *_ = model(input_ids, attention_mask)
            sparse_vector = model.to_pooled(sparse_seq, attention_mask)
            transformed = model.get_mlm_head_input(input_ids, attention_mask)
            cls_hidden = transformed[:, 0, :]

        dla_active = int((sparse_vector[0] > 0).sum().item())
        dla_active_counts.append(dla_active)

        sae_attrib = compute_sae_attribution(
            sae, cls_hidden, classifier_weight, label, vocab_projector_weight,
        )
        sae_active = int((sae_attrib.abs() > 1e-6).sum().item())
        sae_active_counts.append(sae_active)

    # SAE reconstruction error
    all_hidden = []
    for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
        with torch.inference_mode(), autocast():
            transformed = model.get_mlm_head_input(input_ids, attention_mask)
        all_hidden.append(transformed[:, 0, :].float())

    hidden_tensor = torch.cat(all_hidden, dim=0)
    with torch.inference_mode():
        reconstruction, _ = sae(hidden_tensor)
        recon_error = float(torch.nn.functional.mse_loss(
            reconstruction, hidden_tensor,
        ).item())

        all_features = sae.encode(hidden_tensor)
        ever_active = (all_features > 0).any(dim=0)
        dead_frac = 1.0 - ever_active.float().mean().item()

    return {
        "dla_active_tokens": sum(dla_active_counts) / len(dla_active_counts),
        "sae_active_features": sum(sae_active_counts) / len(sae_active_counts),
        "reconstruction_error": recon_error,
        "sae_dead_feature_fraction": dead_frac,
        "sae_hidden_dim": int(all_features.shape[1]),
    }


def compute_sae_attribution(
    sae: SimpleSAE,
    hidden_states: torch.Tensor,
    classifier_weight: torch.Tensor,
    class_idx: int,
    vocab_projector_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute attribution through SAE features.

    Projects hidden states through SAE encoder to get sparse features,
    then computes how each SAE feature contributes to the classifier output
    via the full pathway: SAE decoder -> vocab projector -> classifier.

    All computation stays on GPU. Returns a 1-D attribution tensor.
    """
    with torch.inference_mode():
        sae_features = sae.encode(hidden_states.to(DEVICE))

    decoder_weight = sae.decoder.weight  # [hidden_size, hidden_dim]
    sae_features_flat = sae_features.squeeze()  # [hidden_dim]

    # Full pathway: SAE feature -> decoder (hidden_dim -> hidden_size)
    #   -> output projection (hidden_size -> vocab_size) -> classifier_fc2 row
    proj = decoder_weight.T @ vocab_projector_weight.T @ classifier_weight[class_idx]  # [hidden_dim]
    attribution = sae_features_flat * proj

    return attribution
