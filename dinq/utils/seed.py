"""Seeding (Python, NumPy, torch, optional deterministic cuDNN) and git revision for provenance."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 0, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def git_sha(default: str = "unknown") -> str:
    """Short git SHA of the working tree, for run provenance."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or default
    except Exception:
        return default
