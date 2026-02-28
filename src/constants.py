"""Numerical constants for SPALF."""

DELTA_RANK: float = 1e-2  # Keep 99% covariance variance during whitening.
KAPPA_TARGET: float = 100.0  # Soft-ZCA condition-number cap.
C_EPSILON: float = 0.1  # Moreau transition width as a fraction of local IQR.
DELTA_DRIFT: float = 0.1  # Anchored decoder drift budget in Frobenius norm.
EPS_NUM: float = 1e-8
BETA_SLOW: float = 0.99  # Slow EMA timescale (~100 updates).
LAMBDA_DISC_MAX: float = 1.0  # Maximum discretization penalty weight.
