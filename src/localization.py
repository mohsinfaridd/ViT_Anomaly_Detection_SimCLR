"""Localization maps from RAW patch kNN distances.

SCIENTIFIC RULE:
  LOCALIZATION USES RAW PATCH kNN DISTANCES.
  Position calibration MUST NOT be applied to localization maps.

Conversion: [N,196] -> [N,14,14] -> bilinear -> [N,224,224]
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from .config import ProtocolAConfig
from .data import build_mask_eval_transform


def patch_scores_to_maps(
    patch_scores: torch.Tensor,
    patch_grid: Tuple[int, int] = (14, 14),
    image_size: int = 224,
) -> np.ndarray:
    """
    Convert raw patch scores to full-resolution localization maps.

    Uses RAW patch kNN distances only — do not pass position-calibrated scores.
    """
    h, w = patch_grid
    maps = patch_scores.reshape(-1, 1, h, w).float()
    maps = F.interpolate(
        maps,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return maps[:, 0].numpy()


def load_eval_mask(row: pd.Series, cfg: ProtocolAConfig) -> np.ndarray:
    """Exact notebook ``load_eval_mask``."""
    if int(row["binary_label"]) == 0 or not str(row.get("mask_path", "")):
        return np.zeros((cfg.image_size, cfg.image_size), dtype=bool)
    mask_path = Path(str(row["mask_path"]))
    if not mask_path.is_file():
        return np.zeros((cfg.image_size, cfg.image_size), dtype=bool)
    transform = build_mask_eval_transform(cfg)
    with Image.open(mask_path) as mask:
        mask = mask.convert("L")
        tensor = transform(mask).squeeze(0)
    return tensor.numpy() > 0.5


def relative_defect_area(mask: np.ndarray) -> float:
    """Relative defect area = positive GT-mask pixels / total pixels."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return float("nan")
    return float(mask.sum()) / float(mask.size)


def category_common_visualization_limits(
    raw_patch_scores: Union[torch.Tensor, np.ndarray],
    low_quantile: float = 0.01,
    high_quantile: float = 0.99,
) -> Dict[str, float]:
    """
    Category-common visualization limits (Figure 6).

    Computed from the complete category raw patch-score population as the
    1st and 99th percentiles.

    IMPORTANT: these percentile limits affect visualization ONLY,
    NOT quantitative metrics (pixel ROC-AUC / AP / AUPRO).
    """
    if torch.is_tensor(raw_patch_scores):
        values = raw_patch_scores.detach().cpu().numpy().reshape(-1).astype(float)
    else:
        values = np.asarray(raw_patch_scores, dtype=float).reshape(-1)

    if values.size == 0:
        raise ValueError("No raw patch scores for visualization limits")
    if not np.all(np.isfinite(values)):
        raise ValueError("Non-finite raw patch scores for visualization limits")

    vmin = float(np.quantile(values, low_quantile))
    vmax = float(np.quantile(values, high_quantile))
    if not (np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin):
        raise ValueError(f"Invalid visualization range: {vmin}, {vmax}")

    return {
        "vmin": vmin,
        "vmax": vmax,
        "low_quantile": float(low_quantile),
        "high_quantile": float(high_quantile),
        "n_raw_patch_values": int(values.size),
        "affects_quantitative_metrics": False,
    }
