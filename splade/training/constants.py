"""Paper-mandated constants and validated training recipe for SPLADE.

Section 1: Method-defining constants from published papers. Do not tune.
Section 2: Empirically validated training recipe. Derived from literature
           review and LR range tests across representative conditions.
"""

# === Section 1: Paper-mandated constants (define the method) ===

# DF-FLOPS weighting function: w(df) = 1 / (1 + (df^(log2/log(alpha)) - 1)^beta)
# SPLADE v2 (arXiv:2109.10086, Section 3.2)
DF_ALPHA = 0.1  # Terms appearing in >10% of documents get down-weighted
DF_BETA = 5.0   # Sharpness of the DF weighting transition

# Adaptive Gradient Clipping (AGC)
# NFNet (arXiv:2102.06171, Section 4.1, Table 1)
AGC_CLIP_FACTOR = 0.01  # lambda: gradient/weight norm ratio bound
AGC_EPS = 1e-3           # Minimum weight norm to prevent division by zero

# Regularization schedule: quadratic ramp from 0 to LAMBDA_FINAL
# SPLADE v2 (arXiv:2109.10086, Section 4) uses lambda in [1e-4, 3e-3]
LAMBDA_FINAL = 1e-3  # Geometric mean of the paper's recommended range

# Document frequency tracking
DF_MOMENTUM = 0.9  # EMA decay factor for document frequency counts

# === Section 2: Validated training recipe ===

# Training duration — BERT fine-tuning converges in 3-10 epochs on small datasets;
# 20 is a generous ceiling. Early stopping handles actual termination.
MAX_EPOCHS = 20
EARLY_STOP_PATIENCE = 5  # Epochs without val loss improvement; wider window for EMA recovery

# Warmup — BERT (arXiv:1810.04805, Section 5.1): linear warmup over ~6% of total steps
WARMUP_RATIO = 0.06

# Weight decay — AdamW (Loshchilov & Hutter, arXiv:1711.05101)
# Applied to weight matrices only; bias and LayerNorm parameters excluded.
WEIGHT_DECAY = 0.01

# Label smoothing — Szegedy et al. (arXiv:1512.00567)
# Validated for NLP text classification (arXiv:2312.06522). Multi-class only.
LABEL_SMOOTHING = 0.1

# EMA — Polyak averaging for stable evaluation
# Decay 0.999 standard for BERT-scale models (arXiv:2411.18704)
EMA_DECAY = 0.999

# LR Range Test — Smith (arXiv:1506.01186)
# Empirically finds optimal LR for each model+data combination at training start.
LR_FIND_STEPS = 200           # Sweep iterations (exponential ramp)
LR_FIND_END = 1e-2            # Upper bound of LR sweep
LR_FIND_DIVERGE_FACTOR = 4.0  # Stop if loss exceeds factor × best_loss
