"""SAE initialization: matched-filter encoder, decoder init, threshold calibration (§7.2)."""

from __future__ import annotations

import torch
from torch import Tensor

from src.model.sae import StratifiedSAE
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
    F_free = F - V
    device = W_vocab.device

    with torch.no_grad():
        sae.W_dec_A.copy_(W_vocab)

        free_cols = torch.randn(d, F_free, device=device)
        free_cols = free_cols / free_cols.norm(dim=0, keepdim=True)
        sae.W_dec_B.copy_(free_cols)

        if whitener.is_low_rank:
            # Apply inverse whitening to vocab columns; W_white_inv symmetry preserves orientation.
            W_enc_A = torch.zeros(V, d, device=device)
            for j in range(V):
                col = W_vocab[:, j].unsqueeze(0)
                W_enc_A[j] = (whitener.inverse(col) - whitener.mean).squeeze(0)
            sae.W_enc.data[:V] = W_enc_A
        else:
            W_white_inv = whitener.W_white_inv
            sae.W_enc.data[:V] = (W_white_inv @ W_vocab).T

        W_enc_A = sae.W_enc.data[:V]
        W_enc_B = torch.randn(F_free, d, device=device) / (d**0.5)

        n_orthogonal = min(F_free, d - V)
        if n_orthogonal > 0:
            Q, _ = torch.linalg.qr(W_enc_A.T)

            for i in range(n_orthogonal):
                row = W_enc_B[i]
                proj = Q @ (Q.T @ row)
                row = row - proj
                W_enc_B[i] = row / row.norm()

        for i in range(n_orthogonal, F_free):
            W_enc_B[i] /= W_enc_B[i].norm()

        sae.W_enc.data[V:] = W_enc_B

        _calibrate_thresholds(sae, whitener, activation_sample, L0_target, c_epsilon)

    print(f"SAE initialized: d={d}, F={F}, V={V}, F_free={F_free}, L0_target={L0_target}")


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
    sae.jumprelu.log_threshold.data = thresholds.log()

    gammas = torch.zeros(F, device=device)

    for j in range(F):
        col = pre_act[:, j]
        theta_j = thresholds[j].item()
        std_j = col.std().item()

        window = 2.0 * std_j
        mask = (col > theta_j - window) & (col < theta_j + window)
        near_threshold = col[mask]

        iqr = torch.quantile(near_threshold, 0.75) - torch.quantile(near_threshold, 0.25)

        # gamma_j = (c_epsilon * IQR_j)^2 / 2  (Moreau envelope bandwidth)
        gammas[j] = (c_epsilon * iqr) ** 2 / 2.0

    sae.jumprelu.gamma.copy_(gammas)

    print(f"Thresholds calibrated: median_θ={thresholds.median().item():.4f}, median_γ={gammas.median().item():.6f}")
