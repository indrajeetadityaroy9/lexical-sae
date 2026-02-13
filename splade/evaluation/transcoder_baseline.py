"""Transcoder baseline: maps input hidden states to output MLM logits.

Unlike SAE (which reconstructs the same representation space), a transcoder
maps pre-projection hidden states to post-projection MLM logits — matching
the Anthropic Circuit Tracing (2025) framing.

Reference: Anthropic "Circuit Tracing" (2025).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from splade.utils.cuda import COMPUTE_DTYPE, DEVICE, unwrap_compiled


class SimpleTranscoder(nn.Module):
    """Transcoder: encodes hidden states to sparse latents, decodes to MLM logit space."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = torch.relu(self.encoder(x))
        output = self.decoder(latent)
        return output, latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x))


def run_transcoder_comparison(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels: list[int],
) -> dict:
    """Train transcoder and evaluate against LexicalSAE's built-in decomposition.

    Returns:
        {
            "reconstruction_mse": float,
            "mean_active_features": float,
            "dead_feature_fraction": float,
            "transcoder_hidden_dim": int,
        }
    """
    _model = unwrap_compiled(model)
    overcompleteness = 4
    l1_coeff = 1e-3
    lr = 1e-3
    epochs = 50
    batch_size = 64

    # Collect paired (hidden_state, mlm_logit) data
    all_hidden = []
    all_logits = []
    for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            hidden = _model._get_mlm_head_input(input_ids, attention_mask)
            mlm_out = _model._backbone_forward(attention_mask, input_ids=input_ids).logits
        all_hidden.append(hidden[:, 0, :].float())
        all_logits.append(mlm_out[:, 0, :].float())

    hidden_states = torch.cat(all_hidden, dim=0)
    target_logits = torch.cat(all_logits, dim=0)

    input_dim = hidden_states.shape[-1]
    output_dim = target_logits.shape[-1]
    hidden_dim = input_dim * overcompleteness

    tc = SimpleTranscoder(input_dim, output_dim, hidden_dim).to(DEVICE)
    optimizer = torch.optim.Adam(tc.parameters(), lr=lr)

    loader = DataLoader(
        TensorDataset(hidden_states, target_logits),
        batch_size=batch_size, shuffle=True,
    )

    tc.train()
    for _ in range(epochs):
        for h_batch, t_batch in loader:
            optimizer.zero_grad()
            reconstruction, latent = tc(h_batch)
            recon_loss = nn.functional.mse_loss(reconstruction, t_batch)
            l1_loss = latent.abs().mean()
            loss = recon_loss + l1_coeff * l1_loss
            loss.backward()
            optimizer.step()

    tc.eval()

    # Evaluate
    with torch.inference_mode():
        reconstruction, latent = tc(hidden_states)
        recon_mse = nn.functional.mse_loss(reconstruction, target_logits).item()

        all_latent = tc.encode(hidden_states)
        mean_active = (all_latent > 0).float().sum(dim=-1).mean().item()
        ever_active = (all_latent > 0).any(dim=0)
        dead_frac = 1.0 - ever_active.float().mean().item()

    return {
        "reconstruction_mse": recon_mse,
        "mean_active_features": mean_active,
        "dead_feature_fraction": dead_frac,
        "transcoder_hidden_dim": hidden_dim,
    }
