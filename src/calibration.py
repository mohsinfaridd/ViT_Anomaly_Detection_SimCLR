"""Position-wise and image-level calibration (exact notebook implementations).

Image decision rule: anomaly iff normalized image score > 1.
Primary quantile: q=0.95.
"""

from __future__ import annotations

from typing import Any, Dict, Union

import numpy as np
import torch

from .config import ProtocolAConfig


def fit_patch_position_calibrator(
    normal_patch_scores: torch.Tensor,
    robust_scale_eps: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """Robust normal calibration independently at each ViT patch position."""
    x = normal_patch_scores.detach().cpu().numpy().astype(np.float64)
    median = np.median(x, axis=0)
    mad = np.median(np.abs(x - median[None, :]), axis=0)
    scale = 1.4826 * mad

    global_scale = (
        float(np.median(scale[np.isfinite(scale) & (scale >= robust_scale_eps)]))
        if np.any(np.isfinite(scale) & (scale >= robust_scale_eps))
        else float(np.std(x))
    )
    if not np.isfinite(global_scale) or global_scale < robust_scale_eps:
        global_scale = 1.0
    bad = (~np.isfinite(scale)) | (scale < robust_scale_eps)
    scale[bad] = global_scale

    return {
        "median": median.astype(np.float32),
        "scale": scale.astype(np.float32),
    }


def apply_patch_position_calibrator(
    patch_scores: torch.Tensor,
    calibrator: Dict[str, Any],
    *,
    robust_scale_eps: float = 1e-6,
    clip_z: float = 15.0,
    positive_only: bool = False,
) -> torch.Tensor:
    """Apply per-position median/MAD z-score with notebook clipping."""
    median = torch.as_tensor(calibrator["median"], dtype=torch.float32).view(1, -1)
    scale = torch.as_tensor(calibrator["scale"], dtype=torch.float32).view(1, -1)
    z = (patch_scores.float().cpu() - median) / torch.clamp(scale, min=float(robust_scale_eps))
    clip = float(clip_z)
    if positive_only:
        z = torch.clamp(z, min=0.0, max=clip)
    else:
        z = torch.clamp(z, min=-clip, max=clip)
    return z


def fit_patch_position_calibrator_from_cfg(
    normal_patch_scores: torch.Tensor,
    cfg: ProtocolAConfig,
) -> Dict[str, np.ndarray]:
    return fit_patch_position_calibrator(
        normal_patch_scores, robust_scale_eps=cfg.robust_scale_eps
    )


def apply_patch_position_calibrator_from_cfg(
    patch_scores: torch.Tensor,
    calibrator: Dict[str, Any],
    cfg: ProtocolAConfig,
) -> torch.Tensor:
    return apply_patch_position_calibrator(
        patch_scores,
        calibrator,
        robust_scale_eps=cfg.robust_scale_eps,
        clip_z=cfg.position_calibration_clip_z,
        positive_only=cfg.position_calibration_positive_only,
    )


def position_calibrator_to_json(calibrator: Dict[str, Any]) -> Dict[str, list]:
    """Notebook ``_position_calibrator_to_json``."""
    return {
        "median": np.asarray(calibrator["median"], dtype=float).tolist(),
        "scale": np.asarray(calibrator["scale"], dtype=float).tolist(),
    }


def fit_quantile_margin_calibrator(
    normal_scores,
    threshold_quantile: float = 0.95,
    robust_scale_eps: float = 1e-6,
) -> Dict[str, float]:
    """Map normal median -> 0 and frozen normal q-threshold -> 1."""
    scores = np.asarray(normal_scores, dtype=float)
    median = float(np.median(scores))
    threshold = float(np.quantile(scores, threshold_quantile))
    scale = float(threshold - median)
    if not np.isfinite(scale) or scale < robust_scale_eps:
        scale = float(np.std(scores))
    if not np.isfinite(scale) or scale < robust_scale_eps:
        scale = 1.0
    return {"median": median, "threshold": threshold, "scale": scale}


def fit_quantile_margin_calibrator_from_cfg(normal_scores, cfg: ProtocolAConfig) -> Dict[str, float]:
    return fit_quantile_margin_calibrator(
        normal_scores,
        threshold_quantile=cfg.threshold_quantile,
        robust_scale_eps=cfg.robust_scale_eps,
    )


def quantile_margin_transform(scores, calibrator: Dict[str, float]) -> np.ndarray:
    """Exact notebook ``quantile_margin_transform``."""
    return (np.asarray(scores, dtype=float) - calibrator["median"]) / calibrator["scale"]


def predict_anomaly(normalized_scores, threshold: float = 1.0) -> np.ndarray:
    """Primary decision: anomaly iff normalized image score > 1."""
    return (np.asarray(normalized_scores, dtype=float) > float(threshold)).astype(int)
