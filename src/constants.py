"""Fixed structural constants for SPALF."""

# Whitening / calibration
DELTA_RANK: float = 1e-2
KAPPA_TARGET: float = 100.0

# Moreau envelope / initialization
C_EPSILON: float = 0.1

# Constraint thresholds
DELTA_DRIFT: float = 0.1

# CAPU / control
C_ETA: float = 1.0
EPS_NUM: float = 1e-8

# EMA timescale (drives beta_fast, slow_update_interval, phase_transition_patience)
BETA_SLOW: float = 0.99

# ADRC observer
OMEGA_O_INIT: float = 0.3
RHO_0: float = 1.0

# Discretization
LAMBDA_DISC_MAX: float = 1.0

# Phase transition
PHASE_TRANSITION_FALLBACK: float = 0.97

# Frame energy
C_FRAME: float = 0.1

# Logging
LOG_INTERVAL: int = 100
