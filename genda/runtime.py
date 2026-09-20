"""Reproducibility helpers."""
import os
import random
import numpy as np
import torch

def seed_everything(seed: int = 2026):
    """
    Seed all random number generators for reproducibility.

    Args:
        seed (int): Random seed to use.
    """
    # Python built-in RNG
    random.seed(seed)

    # NumPy RNG
    np.random.seed(seed)

    # PyTorch RNGs
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # For deterministic behavior (slower, but more reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Environment variables (for hash-based sources of randomness)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
        ":4096:8"  # CUDA >= 10.2 for full determinism
    )


