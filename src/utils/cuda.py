"""CUDA execution environment (H100-only)."""

import os
import random

# Must precede any CUDA allocation. Expandable segments (PyTorch 2.2+)
# reduce fragmentation from repeated small allocations during faithfulness
# evaluation (hundreds of single-text predict_proba calls).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

DEVICE = torch.device("cuda")
COMPUTE_DTYPE = torch.bfloat16


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --- H100 Tensor Core defaults ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

# Flash Attention 2 and memory-efficient SDP via H100 SM 9.0.
# Defaults on PyTorch 2.7/H100 but explicit to guard against upstream changes.
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
