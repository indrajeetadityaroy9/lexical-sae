import os
import random

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHINDUCTOR_COORDINATE_DESCENT_TUNING", "1")

import numpy
import torch

DEVICE = torch.device("cuda")
COMPUTE_DTYPE = torch.bfloat16

def set_seed(seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

torch.backends.cudnn.benchmark = True

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)


def unwrap_compiled(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap a torch.compile'd model to access the original module."""
    return getattr(model, "_orig_mod", model)
