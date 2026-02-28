"""Numerical constants for SPALF."""

C_EPSILON: float = 0.1  # Moreau transition width as a fraction of local IQR.
DELTA_DRIFT: float = 0.1  # Anchored decoder drift budget in Frobenius norm.
EPS_NUM: float = 1e-8
BETA_SLOW: float = 0.99  # Slow EMA timescale (~100 updates).
LAMBDA_DISC_MAX: float = 1.0  # Maximum discretization penalty weight.
