"""Shared utilities extracted from the final Protocol A notebooks."""

from __future__ import annotations

import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Match notebook ``seed_everything`` exactly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(require_gpu: bool = False) -> torch.device:
    """Select CUDA when available; optionally require a GPU."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if require_gpu and device.type != "cuda":
        raise RuntimeError("GPU is required but CUDA is not available.")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return device


def amp_enabled(use_amp: bool, device: torch.device) -> bool:
    return bool(use_amp and device.type == "cuda")


def autocast_context(enabled: bool = False):
    """Notebook ``autocast_context``: CUDA float16 AMP or nullcontext."""
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload: Dict[str, Any], path: Union[str, Path]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_cfg_value(value: Any) -> Any:
    """Notebook ``_normalize_cfg_value`` for SSL checkpoint compatibility."""
    if isinstance(value, list):
        return tuple(value)
    return value
