"""H100-specific runtime configuration.

Hardware target: NVIDIA H100 PCIe (SM 9.0, 80 GB HBM3)
 - BF16 compute via torch.amp.autocast
 - TF32 for FP32 matmul and cuDNN convolutions
 - Flash / memory-efficient SDPA (math SDP disabled)
 - cuDNN auto-tuner for fixed-size inputs
 - Expandable CUDA memory segments
"""

import os
import random

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHINDUCTOR_COORDINATE_DESCENT_TUNING", "1")

import torch

assert torch.cuda.is_available(), "This codebase requires an NVIDIA GPU (H100)"

DEVICE = torch.device("cuda")
COMPUTE_DTYPE = torch.bfloat16

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# TF32: use Tensor Cores for FP32 operations (19-bit mantissa, sufficient for training)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# cuDNN auto-tuner: profile kernels on first call, reuse fastest (stable for fixed-size inputs)
torch.backends.cudnn.benchmark = True

# SDPA backend selection: flash and memory-efficient are native on H100;
# disable the slower math fallback so torch never silently degrades
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
