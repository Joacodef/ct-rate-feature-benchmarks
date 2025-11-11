"""Reproducibility helpers shared across pipelines."""

import random
from typing import Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN when possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
