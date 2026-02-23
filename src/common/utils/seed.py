"""Reproducibility helpers shared across pipelines."""

import random
from typing import Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set deterministic random seeds across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value applied to all supported RNG backends.

    Input:
        Single integer seed used to initialize stochastic components in Python,
        NumPy, Torch CPU, and (when available) Torch CUDA.

    Returns:
        ``None``.

    Logic:
        1. Seed Python's built-in ``random`` module.
        2. Seed NumPy global RNG.
        3. Seed Torch CPU RNG and all CUDA devices when CUDA is available.
        4. Configure cuDNN for deterministic behavior by enabling
           ``deterministic`` and disabling ``benchmark``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN when possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
