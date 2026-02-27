"""Calibration routines."""

from __future__ import annotations

import math

import torch

from src.config import CalibrationResult, SPALFConfig
from src.constants import C_EPSILON, C_ORTHO, DELTA_COV, DELTA_DRIFT, DELTA_RANK, KAPPA_TARGET
from src.data.activation_store import ActivationStore
from src.model.initialization import initialize_sae
from src.model.sae import StratifiedSAE
from src.whitening.covariance import OnlineCovariance
from src.whitening.whitener import SoftZCAWhitener


def run_calibration(
    config: SPALFConfig,
    store: ActivationStore,
) -> CalibrationResult:
    """Build covariance, whitener, vocabulary slice, and constraint thresholds."""
    d = store.d_model
    print(f"Starting calibration phase: d={d}")

    F = config.F if config.F > 0 else 32 * d
    L0_target = config.L0_target if config.L0_target is not None else math.ceil(F / 400)

    cov = OnlineCovariance(d, delta_cov=DELTA_COV)

    while not cov.has_converged():
        batch = store.next_batch()
        cov.update(batch.cpu())

        if cov.n_samples > 50_000_000:
            print("Covariance not converged after 50M samples, proceeding anyway")
            break

    print(f"Covariance converged after {cov.n_samples} samples (or limit reached)")

    whitener = SoftZCAWhitener.from_covariance(
        cov,
        kappa_target=KAPPA_TARGET,
        delta_rank=DELTA_RANK,
    )
    whitener = whitener.to(store.device)

    W_vocab_full = store.get_unembedding_matrix()
    if config.V_cap is not None and config.V_cap < W_vocab_full.shape[1]:
        norms = W_vocab_full.norm(dim=0)
        _, top_indices = norms.topk(config.V_cap)
        top_indices = top_indices.sort().values
        W_vocab = W_vocab_full[:, top_indices]
        print(f"Capped vocabulary from {W_vocab_full.shape[1]} to {config.V_cap} (V_cap)")
    else:
        W_vocab = W_vocab_full

    V = W_vocab.shape[1]

    tau_faith = (1.0 - config.R2_target) * d
    tau_drift = DELTA_DRIFT**2 * W_vocab.pow(2).sum().item()
    tau_ortho = C_ORTHO / d

    print(
        f"Calibration complete: alpha={whitener.alpha:.6f}, k={whitener.effective_rank}, "
        f"V={V}, F={F}, L0_target={L0_target}"
    )

    return CalibrationResult(
        whitener=whitener,
        W_vocab=W_vocab,
        d=d,
        V=V,
        F=F,
        L0_target=L0_target,
        tau_faith=tau_faith,
        tau_drift=tau_drift,
        tau_ortho=tau_ortho,
    )


def initialize_from_calibration(
    cal: CalibrationResult,
    store: ActivationStore,
) -> StratifiedSAE:
    """Create and initialize the SAE from calibration outputs."""
    device = cal.W_vocab.device

    sae = StratifiedSAE(cal.d, cal.F, cal.V).to(device)

    samples = []
    n_needed = min(50_000, store.batch_size * 20)
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

    return sae
