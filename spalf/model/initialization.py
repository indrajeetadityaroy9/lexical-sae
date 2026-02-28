"""SAE initialization: matched-filter encoder, decoder init, threshold calibration."""

import json

import torch
from torch import Tensor

from spalf.model.sae import StratifiedSAE
from spalf.whitening.whitener import SoftZCAWhitener

TYPE_CHECKING = False
if TYPE_CHECKING:
    from spalf.config import CalibrationResult
    from spalf.data.activation_store import ActivationStore


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


def initialize_from_calibration(
    cal: "CalibrationResult",
    store: "ActivationStore",
) -> StratifiedSAE:
    """Create and initialize the SAE from calibration outputs.

    Also measures initial orthogonality to set cal.tau_ortho (mutated in-place).
    """
    from spalf.constants import C_EPSILON
    from spalf.model.constraints import compute_orthogonality_violation

    device = cal.W_vocab.device

    sae = StratifiedSAE(cal.d, cal.F, cal.V).to(device)

    samples = []
    n_needed = min(max(100 * cal.F // cal.L0_target, 10_000), store.batch_size * 20)
    while sum(s.shape[0] for s in samples) < n_needed:
        samples.append(store.next_batch())
    activation_sample = torch.cat(samples, dim=0)[:n_needed].to(device)

    initialize_sae(
        sae=sae,
        whitener=cal.whitener,
        W_vocab=cal.W_vocab,
        activation_sample=activation_sample,
        L0_target=cal.L0_target,
        c_epsilon=C_EPSILON,
    )

    # Set tau_ortho from initialized geometry to keep the first constraint scale data-driven.
    with torch.no_grad():
        batch_size = min(activation_sample.shape[0], 4096)
        x_sample = activation_sample[:batch_size]
        x_tilde = cal.whitener.forward(x_sample)
        _, z_init, _, _, _ = sae(x_tilde)
        raw_ortho = compute_orthogonality_violation(
            z_init, sae.W_dec_A, sae.W_dec_B, 0.0, sae.gamma
        ).item()
    cal.tau_ortho = max(raw_ortho, 1.0 / cal.d)

    return sae
