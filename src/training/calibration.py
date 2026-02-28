"""Calibration routines."""


import json
import math

import torch

from src.config import CalibrationResult, SPALFConfig
from src.constants import (
    C_EPSILON,
    DELTA_DRIFT,
    DELTA_RANK,
    KAPPA_TARGET,
)
from src.data.activation_store import ActivationStore
from src.initialization import initialize_sae
from src.sae import StratifiedSAE
from src.whitening.covariance import OnlineCovariance
from src.whitening.whitener import SoftZCAWhitener


def run_calibration(
    config: SPALFConfig,
    store: ActivationStore,
) -> CalibrationResult:
    """Build covariance, whitener, vocabulary slice, and constraint thresholds."""
    d = store.d_model
    print(json.dumps({"event": "calibration_start", "d": d}, sort_keys=True), flush=True)

    F = config.F if config.F > 0 else 32 * d
    L0_target = config.L0_target if config.L0_target is not None else math.ceil(F / 400)

    cov = OnlineCovariance(d)
    cov_max_samples = min(100 * d * d, 200_000_000)

    while not cov.has_converged():
        batch = store.next_batch()
        cov.update(batch)

        if cov.n_samples > cov_max_samples:
            print(
                json.dumps(
                    {
                        "event": "covariance_limit_reached",
                        "max_samples": cov_max_samples,
                        "n_samples": cov.n_samples,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            break

    print(
        json.dumps(
            {
                "event": "covariance_ready",
                "converged": cov.has_converged(),
                "n_samples": cov.n_samples,
            },
            sort_keys=True,
        ),
        flush=True,
    )

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
        print(
            json.dumps(
                {
                    "event": "vocab_capped",
                    "vocab_size_full": W_vocab_full.shape[1],
                    "vocab_size_kept": config.V_cap,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        W_vocab = W_vocab_full

    V = W_vocab.shape[1]

    tau_faith = (1.0 - config.R2_target) * d
    tau_drift = DELTA_DRIFT**2 * W_vocab.pow(2).sum().item()
    tau_ortho = 0.0

    print(
        json.dumps(
            {
                "event": "calibration_ready",
                "alpha": whitener.alpha,
                "effective_rank": whitener.effective_rank,
                "V": V,
                "F": F,
                "L0_target": L0_target,
                "tau_faith": tau_faith,
                "tau_drift": tau_drift,
            },
            sort_keys=True,
        ),
        flush=True,
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
    """Create and initialize the SAE from calibration outputs.

    Also measures initial orthogonality to set cal.tau_ortho (mutated in-place).
    """
    from src.optimization.constraints import compute_orthogonality_violation

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
