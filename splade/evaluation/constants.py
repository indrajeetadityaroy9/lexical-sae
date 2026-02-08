"""Paper-mandated constants for evaluation protocols.

Every value here defines a specific published evaluation method.
These are not hyperparameters — they are protocol definitions.
"""

# Standard evaluation grid for token-removal metrics
K_VALUES = (1, 5, 10, 20)
K_MAX = 20

# F-Fidelity (arXiv:2410.02970, Algorithm 1)
FFIDELITY_BETA = 0.1       # Per-position Bernoulli masking rate
FFIDELITY_N_SAMPLES = 50   # Stochastic mask samples per example
FFIDELITY_FT_EPOCHS = 30   # Fine-tuning epochs on masked inputs
FFIDELITY_FT_LR = 1e-4     # Fine-tuning peak learning rate

# Soft Perturbation Metrics (arXiv:2305.10496, Equations 4-5)
SOFT_METRIC_N_SAMPLES = 20  # Bernoulli embedding-dropout samples

# Normalized AOPC via beam search (arXiv:2408.08137)
NAOPC_BEAM_SIZE = 15  # Beam width for upper/lower bound search

# ERASER monotonicity (arXiv:1911.03429)
MONOTONICITY_STEPS = 10  # Incremental token-removal steps

# Adversarial sensitivity evaluation
ADVERSARIAL_MCP_THRESHOLD = 0.7  # Minimum confidence to include in evaluation
ADVERSARIAL_MAX_CHANGES = 3      # Maximum word substitutions per attack
ADVERSARIAL_TEST_SAMPLES = 50    # Subset size for adversarial evaluation

# LIME baseline (Ribeiro et al., 2016)
LIME_N_SAMPLES = 500  # Perturbation samples for LIME

# Integrated Gradients baseline (Sundararajan et al., 2017)
IG_N_STEPS = 50  # Interpolation steps from baseline to input

# GradientSHAP baseline (Lundberg & Lee, 2017)
GRADIENT_SHAP_N_SAMPLES = 50  # Stochastic baseline samples
