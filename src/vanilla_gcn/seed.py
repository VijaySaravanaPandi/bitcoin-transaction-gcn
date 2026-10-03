"""
seed.py — Global reproducibility seed setter.

Sets random seeds for Python, NumPy, and PyTorch so that experiments are
deterministic across runs when the same seed is used.

Limitations
-----------
Full bit-for-bit determinism on GPU requires additional CUDA settings that
may significantly slow down training.  This module provides a reasonable
balance between reproducibility and performance.

Reference
---------
PyTorch reproducibility guide:
    https://pytorch.org/docs/stable/notes/randomness.html

Usage
-----
    from vanilla_gcn.seed import set_seed
    set_seed(42)
"""

from __future__ import annotations

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, *, deterministic: bool = False) -> None:
    """Set random seeds for full reproducibility.

    Parameters
    ----------
    seed:
        The integer seed value.  Default is ``42``.
    deterministic:
        If ``True``, configure PyTorch to use deterministic algorithms.
        This may reduce performance on GPU but guarantees reproducibility.

        .. note::
            Not all PyTorch operations have deterministic implementations.
            When enabled, PyTorch will raise ``RuntimeError`` for any
            operation that lacks a deterministic variant.

    Examples
    --------
    >>> set_seed(42)
    >>> set_seed(0, deterministic=True)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Available from PyTorch 1.8+
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            logger.warning(
                "torch.use_deterministic_algorithms is not available in this "
                "PyTorch version.  Skipping deterministic algorithm enforcement."
            )

    logger.debug("Seed set to %d (deterministic=%s)", seed, deterministic)
