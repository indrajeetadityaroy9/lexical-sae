"""Calibration routines."""


import json
import math

from spalf.config import CalibrationResult, SPALFConfig
from spalf.constants import DELTA_DRIFT
from spalf.data.activation_store import ActivationStore
from spalf.whitening.covariance import OnlineCovariance
from spalf.whitening.whitener import SoftZCAWhitener


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
    cov_max_samples = 20 * cov.check_interval

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

    whitener = SoftZCAWhitener.from_covariance(cov)
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
                "rho_oas": whitener.rho_oas,
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
