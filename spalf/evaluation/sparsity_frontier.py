"""Sparsity frontier: L0 vs Loss Recovered Pareto curve via threshold sweeping."""


import json

import torch
import torch.nn.functional as F

from spalf.data.activation_store import ActivationStore
from spalf.data.patching import run_patched_forward, run_zero_ablation_forward
from spalf.evaluation.convergence import WelfordMean
from spalf.model import StratifiedSAE
from spalf.whitening.whitener import SoftZCAWhitener

_DEFAULT_MULTIPLIERS: list[float] = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0, 3.0]


@torch.no_grad()
def compute_sparsity_frontier(
    sae: StratifiedSAE,
    whitener: SoftZCAWhitener,
    store: ActivationStore,
) -> list[dict[str, float]]:
    """Sweep threshold multipliers to trace the L0-vs-Loss-Recovered frontier.

    Self-terminating: each phase (baseline + per-multiplier) runs until the CE estimate
    reaches 2% relative standard error (minimum 500K tokens, hard cap 10M per phase).
    """
    device = next(sae.parameters()).device
    multipliers = _DEFAULT_MULTIPLIERS

    # --- Baseline: zero-ablation and original CE ---
    total_ce_zero = 0.0
    total_ce_orig = 0.0
    total_tokens_baseline = 0
    tracker_baseline = WelfordMean()

    baseline_iter = store._token_generator()
    while True:
        tokens = next(baseline_iter).to(device)
        B, S = tokens.shape
        labels = tokens[:, 1:]
        n_tok = B * (S - 1)

        model = store.get_model()
        orig_logits = model(tokens).logits
        zero_logits = run_zero_ablation_forward(store, tokens)

        orig_flat = orig_logits[:, :-1].reshape(-1, orig_logits.shape[-1]).float()
        zero_flat = zero_logits[:, :-1].reshape(-1, zero_logits.shape[-1]).float()
        labels_flat = labels.reshape(-1)

        ce_orig = F.cross_entropy(orig_flat, labels_flat).item()
        ce_zero = F.cross_entropy(zero_flat, labels_flat).item()

        total_ce_orig += ce_orig * n_tok
        total_ce_zero += ce_zero * n_tok
        total_tokens_baseline += n_tok

        tracker_baseline.update(ce_zero - ce_orig)

        if total_tokens_baseline >= 500_000 and tracker_baseline.relative_se < 0.02:
            break
        if total_tokens_baseline >= 10_000_000:
            break

    avg_ce_orig = total_ce_orig / total_tokens_baseline
    avg_ce_zero = total_ce_zero / total_tokens_baseline
    denominator = avg_ce_zero - avg_ce_orig

    print(
        json.dumps(
            {
                "event": "frontier_baseline",
                "ce_orig": avg_ce_orig,
                "ce_zero": avg_ce_zero,
                "baseline_tokens": total_tokens_baseline,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    # --- Sweep multipliers ---
    orig_log_threshold = sae.log_threshold.data.clone()
    results = []

    for mult in multipliers:
        sae.log_threshold.data = orig_log_threshold + torch.tensor(mult).log()

        total_l0 = 0.0
        total_mse = 0.0
        total_ce_patched = 0.0
        total_tokens = 0
        total_activations = 0
        tracker_point = WelfordMean()

        token_iter = store._token_generator()

        while True:
            tokens = next(token_iter).to(device)
            B, S = tokens.shape

            orig_logits, patched_logits, x_flat, x_hat_flat, _, gate_mask = (
                run_patched_forward(store, sae, whitener, tokens)
            )

            labels = tokens[:, 1:]
            n_tok = B * (S - 1)
            n_act = x_flat.shape[0]

            ce_patched = F.cross_entropy(
                patched_logits[:, :-1].reshape(-1, patched_logits.shape[-1]).float(),
                labels.reshape(-1),
            )

            total_l0 += gate_mask.sum(dim=1).mean().item() * n_act
            total_mse += (x_flat - x_hat_flat).pow(2).sum(dim=1).mean().item() * n_act
            total_ce_patched += ce_patched.item() * n_tok
            total_tokens += n_tok
            total_activations += n_act

            tracker_point.update(ce_patched.item())

            if total_tokens >= 500_000 and tracker_point.relative_se < 0.02:
                break
            if total_tokens >= 10_000_000:
                break

        avg_ce_patched = total_ce_patched / total_tokens
        loss_recovered = (avg_ce_zero - avg_ce_patched) / denominator

        point = {
            "multiplier": mult,
            "l0": total_l0 / total_activations,
            "ce_loss_increase": avg_ce_patched - avg_ce_orig,
            "loss_recovered": loss_recovered,
            "mse": total_mse / total_activations,
            "tokens_per_point": total_tokens,
        }
        results.append(point)

    sae.log_threshold.data = orig_log_threshold

    return results
