"""Runtime: CUDA device, bfloat16 autocast, seed management."""

import os
import random
from contextlib import contextmanager

import numpy
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHINDUCTOR_COORDINATE_DESCENT_TUNING", "1")

DEVICE = torch.device("cuda")
COMPUTE_DTYPE = torch.bfloat16

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)


@contextmanager
def autocast():
    """Autocast context manager — bfloat16 AMP on CUDA."""
    with torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
        yield


def set_seed(seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


