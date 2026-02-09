"""Sparse Autoencoder baseline for comparison with vocabulary-grounded decomposition.

Reference: Cunningham et al. (2023) "Sparse Autoencoders Find Highly Interpretable
Features in Language Models" (arXiv:2309.08600).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from splade.utils.cuda import COMPUTE_DTYPE, DEVICE, unwrap_compiled


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
    overcompleteness = 4
    l1_coeff = 1e-3
    lr = 1e-3
    epochs = 50
    batch_size = 64
    _model = unwrap_compiled(model)

    # Collect hidden states (kept on GPU) via architecture-agnostic hook
    all_hidden = []
    for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            transformed = _model._get_mlm_head_input(input_ids, attention_mask)
        # Use CLS token representation, keep on GPU
        all_hidden.append(transformed[:, 0, :].float())

    hidden_states = torch.cat(all_hidden, dim=0)
    input_dim = hidden_states.shape[-1]
    hidden_dim = input_dim * overcompleteness

    sae = SimpleSAE(input_dim, hidden_dim).to(DEVICE)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    loader = DataLoader(
        TensorDataset(hidden_states),
        batch_size=batch_size,
        shuffle=True,
    )

    sae.train()
    for epoch in range(epochs):
        for (batch,) in loader:
            optimizer.zero_grad()
            reconstruction, hidden = sae(batch)
            recon_loss = nn.functional.mse_loss(reconstruction, batch)
            l1_loss = hidden.abs().mean()
            loss = recon_loss + l1_coeff * l1_loss
            loss.backward()
            optimizer.step()

    sae.eval()
    return sae


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
    # Note: SAE operates on hidden states before the ReLU MLP, so we use
    # classifier_fc2.weight directly (not W_eff) for this comparison.
    proj = decoder_weight.T @ vocab_projector_weight.T @ classifier_weight[class_idx]  # [hidden_dim]
    attribution = sae_features_flat * proj

    return attribution
