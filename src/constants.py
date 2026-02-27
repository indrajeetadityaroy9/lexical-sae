"""Fixed structural constants for the SPALF framework.

These values are derived from the methodology (sections 1-7) and are not
user-configurable. Changing them constitutes a different algorithm.
"""

# Covariance & whitening (section 1.2)
DELTA_COV: float = 1e-3
DELTA_RANK: float = 1e-2
KAPPA_TARGET: float = 100.0

# Constraint thresholds (section 3.2)
C_EPSILON: float = 0.1
C_ORTHO: float = 3.0
DELTA_DRIFT: float = 0.1
C_ETA: float = 1.0
EPS_NUM: float = 1e-8

# Control system (sections 4-5)
BETA_FAST: float = 0.9
BETA_SLOW: float = 0.99
OMEGA_O_INIT: float = 0.3
RHO_0: float = 1.0
SLOW_UPDATE_INTERVAL: int = 100

# Discretization (section 3.1)
S_DISC: float = 0.8
LAMBDA_DISC_MAX: float = 1.0

# Phase transition (section 7.3)
PHASE_TRANSITION_PATIENCE: int = 100
PHASE_TRANSITION_FALLBACK: float = 0.97

# Logging
LOG_INTERVAL: int = 100
