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
    """Initialize SAE parameters following §7.2 steps 7-12.

    Steps:
    7.  W_enc^(A) = W_vocab^T · U(Λ+αI)^{+1/2}U^T  (matched-filter)
    8.  W_enc^(B): rows ~ N(0, I/d), Gram-Schmidt vs W_enc^(A)
    9.  W_dec^(A) = W_vocab
    10. W_dec^(B): cols ~ N(0, I/d), unit-normalized
    11. θ_j = Quantile(W_enc · W_white · x, 1 - L0_target/F)
    12. ε_j = c_ε · IQR(pre_act_j near θ_j)

    Args:
        sae: StratifiedSAE to initialize in-place.
        whitener: Frozen Soft-ZCA whitener.
        W_vocab: [d, V] unembedding matrix (columns = vocabulary directions).
        activation_sample: [N, d] sample of raw activations for threshold calibration.
        L0_target: Target number of active features per input.
        c_epsilon: Bandwidth scaling factor (default 0.1).
    """
    d, V = W_vocab.shape
    F = sae.F
    F_free = F - V
    device = W_vocab.device

    with torch.no_grad():
        # --- Step 9: Anchored decoder = W_vocab ---
        sae.W_dec_A.copy_(W_vocab)

        # --- Step 10: Free decoder = random unit-norm columns ---
        free_cols = torch.randn(d, F_free, device=device)
        free_cols = free_cols / free_cols.norm(dim=0, keepdim=True)
        sae.W_dec_B.copy_(free_cols)

        # --- Step 7: Matched-filter encoder for anchored features ---
        # W_enc^(A) = W_vocab^T · W_white^{-1}
        # = W_vocab^T · U(Λ+αI)^{+1/2}U^T
        if whitener.is_low_rank:
            # Low-rank inverse: apply inverse transform to each vocab column
            # W_enc_A[j, :] = W_vocab[:, j]^T @ W_white_inv
            W_enc_A = torch.zeros(V, d, device=device)
            for j in range(V):
                w = W_vocab[:, j]  # [d]
                W_enc_A[j] = whitener.inverse(w.unsqueeze(0)).squeeze(0) - whitener.mean
            # Actually: w_enc = w_vocab^T @ W_white_inv
            # whitener.inverse(x̃) = W_white_inv @ x̃ + μ
            # So W_white_inv @ w_vocab[:, j] = whitener.inverse(w_vocab[:, j]) - μ
            # And w_enc_A[j] = (whitener.inverse(w_vocab[:, j].unsqueeze(0)) - μ).squeeze(0)
            # But we want w_vocab^T @ W_white_inv, which is the transpose
            # w_enc_A = W_vocab^T @ W_white_inv = (W_white_inv^T @ W_vocab)^T
            # Since W_white_inv is symmetric: = (W_white_inv @ W_vocab)^T = W_vocab^T @ W_white_inv
            # For low-rank: apply inverse to columns of W_vocab
            W_enc_A_corrected = torch.zeros(V, d, device=device)
            for j in range(V):
                col = W_vocab[:, j].unsqueeze(0)  # [1, d] treat as "whitened" input
                unwhitened = whitener.inverse(col) - whitener.mean  # [1, d]
                W_enc_A_corrected[j] = unwhitened.squeeze(0)
            sae.W_enc.data[:V] = W_enc_A_corrected
        else:
            # Full: W_enc^(A) = W_vocab^T @ W_white_inv
            W_white_inv = whitener.W_white_inv  # [d, d]
            sae.W_enc.data[:V] = (W_white_inv @ W_vocab).T  # [V, d]

        # --- Step 8: Free encoder = random rows, Gram-Schmidt vs anchored ---
        W_enc_A = sae.W_enc.data[:V]  # [V, d]
        W_enc_B = torch.randn(F_free, d, device=device) / (d**0.5)

        # Gram-Schmidt: orthogonalize first (d-V) free rows against anchored
        n_orthogonal = min(F_free, d - V)
        if n_orthogonal > 0:
            # Build orthonormal basis from anchored encoder rows
            Q, _ = torch.linalg.qr(W_enc_A.T)  # [d, min(V,d)]
            n_basis = Q.shape[1]

            for i in range(n_orthogonal):
                row = W_enc_B[i]  # [d]
                # Project out the anchored subspace
                proj = Q @ (Q.T @ row)  # [d]
                row = row - proj
                # Normalize
                norm = row.norm()
                if norm > 1e-8:
                    W_enc_B[i] = row / norm
                else:
                    # Degenerate: just use random unit vector
                    W_enc_B[i] = torch.randn(d, device=device)
                    W_enc_B[i] /= W_enc_B[i].norm()

        # Remaining rows (F_free > d-V): just random unit-norm
        for i in range(n_orthogonal, F_free):
            W_enc_B[i] /= W_enc_B[i].norm()

        sae.W_enc.data[V:] = W_enc_B

        # --- Steps 11-12: Threshold and bandwidth calibration ---
        _calibrate_thresholds(sae, whitener, activation_sample, L0_target, c_epsilon)

    print(f"SAE initialized: d={d}, F={F}, V={V}, F_free={F_free}, L0_target={L0_target}")


def _calibrate_thresholds(
    sae: StratifiedSAE,
    whitener: SoftZCAWhitener,
    activation_sample: Tensor,
    L0_target: int,
    c_epsilon: float,
) -> None:
    """Calibrate JumpReLU thresholds and bandwidths from activation sample.

    Step 11: θ_j = Quantile_j(pre_act, 1 - L0_target/F)
    Step 12: ε_j = c_ε · IQR(pre_act_j near θ_j)
    """
    F = sae.F
    device = activation_sample.device

    # Whiten sample
    x_tilde = whitener.forward(activation_sample)  # [N, d]

    # Compute pre-activations
    pre_act = x_tilde @ sae.W_enc.T + sae.b_enc  # [N, F]

    # Step 11: Per-feature quantile threshold
    quantile = 1.0 - L0_target / F
    thresholds = torch.quantile(pre_act, quantile, dim=0)  # [F]
    sae.jumprelu.log_threshold.data = thresholds.log()

    # Step 12: Per-feature adaptive bandwidth
    # IQR of pre-activations near θ_j (within 2σ window)
    epsilons = torch.zeros(F, device=device)

    for j in range(F):
        col = pre_act[:, j]
        theta_j = thresholds[j].item()
        std_j = col.std().item()

        # Window around threshold
        window = 2.0 * std_j
        mask = (col > theta_j - window) & (col < theta_j + window)
        near_threshold = col[mask]

        if near_threshold.numel() >= 10:
            q75 = torch.quantile(near_threshold, 0.75)
            q25 = torch.quantile(near_threshold, 0.25)
            iqr = q75 - q25
        else:
            # Fallback: global IQR
            q75 = torch.quantile(col, 0.75)
            q25 = torch.quantile(col, 0.25)
            iqr = q75 - q25

        epsilons[j] = c_epsilon * iqr

    sae.jumprelu.epsilon.copy_(epsilons)

    print(f"Thresholds calibrated: median_θ={thresholds.median().item():.4f}, median_ε={epsilons.median().item():.6f}")
