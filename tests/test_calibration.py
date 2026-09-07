"""Unit tests for position and image calibration."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration import (
    apply_patch_position_calibrator,
    fit_patch_position_calibrator,
    fit_quantile_margin_calibrator,
    predict_anomaly,
    quantile_margin_transform,
)


def test_quantile_threshold_maps_to_one():
    rng = np.random.default_rng(0)
    scores = rng.normal(loc=2.0, scale=0.5, size=500)
    cal = fit_quantile_margin_calibrator(scores, threshold_quantile=0.95)
    transformed = quantile_margin_transform(np.array([cal["threshold"]]), cal)
    assert abs(float(transformed[0]) - 1.0) < 1e-9


def test_median_maps_to_zero():
    rng = np.random.default_rng(1)
    scores = rng.normal(loc=3.0, scale=1.0, size=400)
    cal = fit_quantile_margin_calibrator(scores, threshold_quantile=0.95)
    transformed = quantile_margin_transform(np.array([cal["median"]]), cal)
    assert abs(float(transformed[0]) - 0.0) < 1e-9


def test_decision_score_gt_one_is_anomaly():
    preds = predict_anomaly(np.array([0.0, 1.0, 1.0001, 2.5]))
    assert list(preds) == [0, 0, 1, 1]


def test_position_calibrator_shapes_and_clip():
    rng = np.random.default_rng(2)
    normal = torch.tensor(rng.normal(size=(40, 196)), dtype=torch.float32)
    cal = fit_patch_position_calibrator(normal)
    assert cal["median"].shape == (196,)
    assert cal["scale"].shape == (196,)
    z = apply_patch_position_calibrator(normal, cal, clip_z=15.0)
    assert z.shape == normal.shape
    assert float(z.min()) >= -15.0
    assert float(z.max()) <= 15.0
