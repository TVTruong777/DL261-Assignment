"""Shared utilities: reproducibility, device selection, checkpointing, timing."""
from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Fix all relevant RNGs so runs are reproducible given the same config.

    Called once at the start of every training/evaluation run, using the
    `seed` value from the run's YAML config. This is what lets a reported
    result be reproduced from `configs/<name>.yaml` alone.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device(preferred: str = "auto") -> torch.device:
    """Resolve a device string from config into an actual torch.device.

    `preferred` may be "auto", "cpu", "cuda", or "cuda:0" etc.
    """
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Total parameter count — one of the mandatory comparison metrics."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


@contextmanager
def timer():
    """Context manager returning elapsed wall-clock seconds via `.elapsed`.

    Usage:
        with timer() as t:
            train_one_epoch(...)
        print(t.elapsed)
    """

    class _T:
        elapsed: float = 0.0

    t = _T()
    start = time.perf_counter()
    yield t
    t.elapsed = time.perf_counter() - start


@dataclass
class Checkpoint:
    epoch: int
    model_state: dict
    optimizer_state: dict
    best_metric: float
    config: dict


def save_checkpoint(path: str | Path, ckpt: Checkpoint) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": ckpt.epoch,
            "model_state": ckpt.model_state,
            "optimizer_state": ckpt.optimizer_state,
            "best_metric": ckpt.best_metric,
            "config": ckpt.config,
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {path}. See checkpoints/README.md for "
            f"download links or reconstruction instructions."
        )
    return torch.load(path, map_location=map_location)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_git_commit() -> str:
    """Best-effort short git commit hash, for results-traceability logging."""
    try:
        import subprocess

        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def append_experiment_log(log_path: str | Path, row: dict[str, Any]) -> None:
    """Append one row to results/experiment_log.csv, creating it if needed.

    This is what makes every reported number traceable back to a config,
    dataset split, checkpoint, experiment id, and commit — per the course's
    reproducibility requirement.
    """
    import csv

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
