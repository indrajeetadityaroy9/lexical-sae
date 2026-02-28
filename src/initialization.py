"""SAE initialization: matched-filter encoder, decoder init, threshold calibration."""

import json

import torch
from torch import Tensor

from src.sae import StratifiedSAE
from src.whitening.whitener import SoftZCAWhitener


def initialize_sae(
    sae: StratifiedSAE,
    whitener: SoftZCAWhitener,
    W_vocab: Tensor,
    activation_sample: Tensor,
    L0_target: int,
    c_epsilon: float = 0.1,
) -> None:
    """Initialize SAE weights, thresholds, and bandwidths."""
    d, V = W_vocab.shape
    F = sae.F
    device = W_vocab.device

    with torch.no_grad():
        sae.W_dec_A.copy_(W_vocab)

        free_cols = torch.randn(d, sae.F_free, device=device)
        free_cols = free_cols / free_cols.norm(dim=0, keepdim=True)
        sae.W_dec_B.copy_(free_cols)

        if whitener.is_low_rank:
            # Low-rank inverse is applied column-wise to avoid materializing dense dxd matrices.
            W_enc_A = torch.zeros(V, d, device=device)
            for j in range(V):
                col = W_vocab[:, j].unsqueeze(0)
                W_enc_A[j] = (whitener.inverse(col) - whitener.mean).squeeze(0)
            sae.W_enc.data[:V] = W_enc_A
        else:
            W_white_inv = whitener.W_white_inv
            sae.W_enc.data[:V] = (W_white_inv @ W_vocab).T

        W_enc_A = sae.W_enc.data[:V]
        W_enc_B = torch.randn(sae.F_free, d, device=device) / (d**0.5)

        n_orthogonal = min(sae.F_free, d - V)
        if n_orthogonal > 0:
            Q, _ = torch.linalg.qr(W_enc_A.T)

            for i in range(n_orthogonal):
                row = W_enc_B[i]
                proj = Q @ (Q.T @ row)
                row = row - proj
                W_enc_B[i] = row / row.norm()

        for i in range(n_orthogonal, sae.F_free):
            W_enc_B[i] /= W_enc_B[i].norm()

        sae.W_enc.data[V:] = W_enc_B

        _calibrate_thresholds(sae, whitener, activation_sample, L0_target, c_epsilon)

    print(
        json.dumps(
            {
                "event": "sae_initialized",
                "d": d,
                "F": F,
                "V": V,
                "F_free": sae.F_free,
                "L0_target": L0_target,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _calibrate_thresholds(
    sae: StratifiedSAE,
    whitener: SoftZCAWhitener,
    activation_sample: Tensor,
    L0_target: int,
    c_epsilon: float,
) -> None:
    """Calibrate JumpReLU thresholds and bandwidths from an activation sample."""
    F = sae.F
    device = activation_sample.device

    x_tilde = whitener.forward(activation_sample)

    pre_act = x_tilde @ sae.W_enc.T + sae.b_enc

    quantile = 1.0 - L0_target / F
    thresholds = torch.quantile(pre_act, quantile, dim=0)
    sae.log_threshold.data = thresholds.log()

    # Restrict IQR to a threshold-local window so gamma tracks jump-region curvature.
    std_all = pre_act.std(dim=0)
    lower = thresholds - 2.0 * std_all
    upper = thresholds + 2.0 * std_all
    mask = (pre_act > lower.unsqueeze(0)) & (pre_act < upper.unsqueeze(0))
    masked = pre_act.clone()
    masked[~mask] = float("nan")
    q75 = torch.nanquantile(masked, 0.75, dim=0)
    q25 = torch.nanquantile(masked, 0.25, dim=0)
    iqr = q75 - q25
    iqr = torch.where(iqr.isnan(), std_all, iqr)  # Preserve finite gamma for degenerate local samples.

    # Moreau bandwidth for feature j: gamma_j = (c_epsilon * IQR_j)^2 / 2.
    gammas = (c_epsilon * iqr).pow(2) / 2.0

    sae.gamma.copy_(gammas)

    print(
        json.dumps(
            {
                "event": "thresholds_calibrated",
                "median_theta": thresholds.median().item(),
                "median_gamma": gammas.median().item(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
