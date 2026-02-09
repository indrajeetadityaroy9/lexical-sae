"""GECO-style constrained optimization for CIS training.

Replaces the fixed lambda schedules with a single Lagrangian multiplier
on the classification CE constraint. The circuit losses (completeness,
separation, sharpness) + DF-FLOPS become the objective to minimize, and
CE is constrained to stay below a threshold derived from the warmup phase.

All tunable hyperparameters are derived from steps_per_epoch:
- tau_ce: set from 25th percentile of warmup CE (data-driven, no margin)
- eta: 2 / (steps_per_epoch + 1), so per-epoch lambda drift is ~constant
  regardless of dataset size
- EMA decay: 1 - eta, smoothing over one epoch of constraint violations
- log_lambda clamped to [-5, 5] for numerical stability

Reference: Rezende & Viola, "Taming VAEs" (arXiv:1810.00597).
           Lei et al., "Sparse Attention Post-Training" (arXiv:2512.05865).
"""

import math
import statistics

import torch


# Clamp log_lambda to prevent numerical instability.
# lambda ∈ [exp(-5), exp(5)] ≈ [0.007, 148.4] — a 20,000x dynamic range.
_LOG_LAMBDA_CLAMP = 5.0


class GECOController:
    """Single-constraint GECO: minimize circuit_objective subject to CE <= tau.

    The Lagrangian multiplier lambda_ce is maintained in log-space to ensure
    positivity. An EMA of the constraint violation smooths the dual update.

    Adaptive mechanisms:
    - tau_ce: 25th percentile of warmup CE values (data-driven)
    - eta: 2 / (steps_per_epoch + 1), dataset-size-invariant
    - EMA decay: 1 - eta, smoothing over one epoch
    - log_lambda clamped to [-5, 5] for stability
    """

    def __init__(self, steps_per_epoch: int = 13):
        # EMA window = steps_per_epoch: smooth over one full epoch.
        # eta = 2/(W+1): per-epoch lambda drift ≈ 2 * avg_constraint,
        # independent of dataset size.
        self._eta = 2.0 / (steps_per_epoch + 1)
        self.ema_decay = 1.0 - self._eta
        self._log_lambda = 0.0  # lambda starts at exp(0) = 1.0
        self._ema_constraint: float | None = None
        self._warmup_ces: list[float] = []
        self.tau_ce: float | None = None

    @property
    def lambda_ce(self) -> float:
        """Current Lagrangian multiplier (always positive via exp)."""
        return math.exp(self._log_lambda)

    def record_warmup_ce(self, ce_value: float) -> None:
        """Record a CE value during the warmup phase."""
        self._warmup_ces.append(ce_value)

    def finalize_warmup(self) -> float:
        """Set tau_ce from the 25th percentile of warmup CE values.

        Using a percentile is data-driven and robust to outliers —
        no manual tau_margin constant needed. The 25th percentile
        represents "achievable good CE", tighter than the mean but
        not as aggressive as the minimum.
        """
        recent = self._warmup_ces[-50:] if len(self._warmup_ces) >= 50 else self._warmup_ces
        if len(recent) >= 4:
            self.tau_ce = statistics.quantiles(recent, n=4)[0]  # 25th percentile
        else:
            self.tau_ce = sum(recent) / len(recent)
        return self.tau_ce

    def compute_loss(
        self,
        ce_loss: torch.Tensor,
        circuit_objective: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Lagrangian and update lambda via EMA-smoothed dual ascent.

        Uses a fixed step size derived from the EMA window (eta ≈ 0.095).
        The original GECO paper (Rezende & Viola 2018) uses fixed-eta SGD
        for the dual variable. Matching eta to the EMA timescale ensures
        the dual update and constraint smoothing operate at the same frequency.

        log_lambda is clamped to [-5, 5] to prevent numerical instability
        while preserving a 20,000x dynamic range for the multiplier.

        Returns: circuit_objective + lambda_ce * ce_loss
        """
        with torch.no_grad():
            ce_val = ce_loss.item()
            constraint = ce_val - self.tau_ce
            if self._ema_constraint is None:
                self._ema_constraint = constraint
            else:
                self._ema_constraint = (
                    self.ema_decay * self._ema_constraint
                    + (1.0 - self.ema_decay) * constraint
                )
            # Fixed-eta dual ascent with EMA smoothing
            self._log_lambda += self._eta * self._ema_constraint
            self._log_lambda = max(-_LOG_LAMBDA_CLAMP,
                                   min(_LOG_LAMBDA_CLAMP, self._log_lambda))

        return circuit_objective + self.lambda_ce * ce_loss

    def compute_loss_sparsity_aware(
        self,
        ce_loss: torch.Tensor,
        circuit_objective: torch.Tensor,
        active_dims_ratio: float,
        relaxation_factor: float = 0.5,
        max_relaxation: float = 1.5,
    ) -> torch.Tensor:
        """Sparsity-aware GECO: relax tau when model is dense.

        When active_dims >> target, tau is inflated to create a "safe harbor"
        that lets the optimizer sacrifice accuracy temporarily to find sparse
        solutions. As sparsity improves, tau tightens back to the baseline.

        Args:
            ce_loss: Current CE loss (scalar tensor).
            circuit_objective: Sum of circuit losses (scalar tensor).
            active_dims_ratio: current_active_dims / target_active_dims.
                >1 means model is denser than target.
            relaxation_factor: How aggressively to relax tau (0.5 = up to 50%).
            max_relaxation: Hard cap on tau multiplier (1.5 = tau can grow 50%).

        Returns: Lagrangian loss = circuit_objective + lambda * ce_loss
        """
        with torch.no_grad():
            # Dynamic tau: relax when dense, tighten as model sparsifies
            sparsity_penalty = math.log(max(1.0, active_dims_ratio))
            relaxation = 1.0 + relaxation_factor * min(1.0, sparsity_penalty)
            relaxation = min(relaxation, max_relaxation)
            dynamic_tau = self.tau_ce * relaxation

            ce_val = ce_loss.item()
            constraint = ce_val - dynamic_tau
            if self._ema_constraint is None:
                self._ema_constraint = constraint
            else:
                self._ema_constraint = (
                    self.ema_decay * self._ema_constraint
                    + (1.0 - self.ema_decay) * constraint
                )
            self._log_lambda += self._eta * self._ema_constraint
            self._log_lambda = max(-_LOG_LAMBDA_CLAMP,
                                   min(_LOG_LAMBDA_CLAMP, self._log_lambda))

            # Store for logging
            self._last_dynamic_tau = dynamic_tau
            self._last_relaxation = relaxation

        return circuit_objective + self.lambda_ce * ce_loss
