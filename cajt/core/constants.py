# Architecture (not hyperparameters)
CLASSIFIER_HIDDEN = 256

# JumpReLU STE bandwidth (Rajamanoharan et al. 2024, paper-specified)
# Controls sharpness of the sigmoid approximation to Heaviside in the backward pass.
# Smaller = sharper (closer to true Heaviside) but noisier gradients.
JUMPRELU_BANDWIDTH = 0.001

# Circuit loss internals
CIRCUIT_TEMPERATURE = 10.0

# Circuit mass fraction: select minimum features capturing this fraction
# of total attribution mass. Data-adaptive replacement for fixed circuit_fraction.
CIRCUIT_MASS_FRACTION = 0.9

# Evaluation infrastructure
EVAL_BATCH_SIZE = 32
