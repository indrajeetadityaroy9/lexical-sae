"""Training metrics logging."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StepMetrics:
    """Metrics collected at each training step."""

    step: int = 0
    phase: int = 1

    l0_mean: float = 0.0
    l0_corr: float = 0.0
    lagrangian: float = 0.0

    v_faith: float = 0.0
    v_drift: float = 0.0
    v_ortho: float = 0.0

    v_fast_faith: float = 0.0
    v_fast_drift: float = 0.0
    v_fast_ortho: float = 0.0

    lambda_faith: float = 0.0
    lambda_drift: float = 0.0
    lambda_ortho: float = 0.0
    rho_faith: float = 0.0
    rho_drift: float = 0.0
    rho_ortho: float = 0.0
    omega_o: float = 0.0

    lambda_disc: float = 0.0
    disc_correction: float = 0.0

    mse: float = 0.0
    r_squared: float = 0.0

    kl_div: float = 0.0


class MetricsLogger:
    """Logs training metrics at regular intervals."""

    def __init__(self, log_interval: int = 100, **_kwargs) -> None:
        self.log_interval = log_interval

    def log_step(self, metrics: StepMetrics) -> None:
        if metrics.step % self.log_interval != 0:
            return
        m = metrics
        print(
            f"[P{m.phase}] step={m.step} | L0={m.l0_mean:.1f} | "
            f"loss={m.lagrangian:.4f} | MSE={m.mse:.4f} | R2={m.r_squared:.4f} | "
            f"v=[{m.v_fast_faith:+.4f}, {m.v_fast_drift:+.4f}, {m.v_fast_ortho:+.4f}] | "
            f"lam=[{m.lambda_faith:.3f}, {m.lambda_drift:.3f}, {m.lambda_ortho:.3f}] | "
            f"rho=[{m.rho_faith:.3f}, {m.rho_drift:.3f}, {m.rho_ortho:.3f}] | "
            f"omega={m.omega_o:.3f}"
        )

    def log_phase_transition(self, step: int, reason: str) -> None:
        print(f"Phase transition at step {step}: {reason}")

    def log_calibration(self, alpha: float, effective_rank: int, V: int) -> None:
        print(f"Calibration: alpha={alpha:.6f}, effective_rank={effective_rank}, V={V}")
