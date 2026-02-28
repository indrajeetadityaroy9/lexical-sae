"""Sparsity frontier: L0 vs CE loss Pareto curve via threshold sweeping."""


import torch
import torch.nn.functional as F

from src.data.activation_store import ActivationStore
from src.data.patching import run_patched_forward
from src.sae import StratifiedSAE
from src.whitening.whitener import SoftZCAWhitener


@torch.no_grad()
def compute_sparsity_frontier(
    sae: StratifiedSAE,
    whitener: SoftZCAWhitener,
    store: ActivationStore,
    multipliers: list[float] | None = None,
    n_batches: int = 20,
) -> list[dict[str, float]]:
    """Sweep threshold multipliers to trace the L0-vs-CE frontier."""
    if multipliers is None:
        multipliers = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0, 3.0]

    device = next(sae.parameters()).device

    orig_log_threshold = sae.log_threshold.data.clone()

    results = []

    for mult in multipliers:
        sae.log_threshold.data = orig_log_threshold + torch.tensor(mult).log()

        total_l0 = 0.0
        total_mse = 0.0
        total_ce_orig = 0.0
        total_ce_patched = 0.0
        total_tokens = 0
        total_activations = 0

        token_iter = store._token_generator()

        for _ in range(n_batches):
            tokens = next(token_iter).to(device)
            B, S = tokens.shape

            orig_logits, patched_logits, x_flat, x_hat_flat, _, gate_mask = (
                run_patched_forward(store, sae, whitener, tokens)
            )

            labels = tokens[:, 1:]
            ce_orig = F.cross_entropy(
                orig_logits[:, :-1].reshape(-1, orig_logits.shape[-1]).float(),
                labels.reshape(-1),
            )
            ce_patched = F.cross_entropy(
                patched_logits[:, :-1].reshape(-1, patched_logits.shape[-1]).float(),
                labels.reshape(-1),
            )

            n_tok = B * (S - 1)
            n_act = x_flat.shape[0]
            total_l0 += gate_mask.sum(dim=1).mean().item() * n_act
            total_mse += (x_flat - x_hat_flat).pow(2).sum(dim=1).mean().item() * n_act
            total_ce_orig += ce_orig.item() * n_tok
            total_ce_patched += ce_patched.item() * n_tok
            total_tokens += n_tok
            total_activations += n_act

        point = {
            "multiplier": mult,
            "l0": total_l0 / total_activations,
            "ce_loss_increase": (total_ce_patched - total_ce_orig) / total_tokens,
            "mse": total_mse / total_activations,
        }
        results.append(point)

    sae.log_threshold.data = orig_log_threshold

    return results
