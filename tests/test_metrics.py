"""Unit tests for metrics helpers."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics import binary_metrics, mean_and_sample_sd


REQUIRED_KEYS = {
    "accuracy",
    "balanced_accuracy",
    "anomaly_precision",
    "anomaly_recall",
    "anomaly_f1",
    "false_positive_rate",
    "roc_auc",
    "average_precision",
    "mcc",
    "TN",
    "FP",
    "FN",
    "TP",
}


def test_binary_metrics_keys():
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    y_score = np.array([0.1, 0.8, 0.9, 0.7, 0.4, 0.2])
    out = binary_metrics(y_true, y_pred, y_score)
    assert REQUIRED_KEYS.issubset(out.keys())


def test_sample_sd_uses_ddof_1():
    values = [1.0, 2.0, 3.0]
    stats = mean_and_sample_sd(values)
    expected = float(np.std(np.asarray(values, dtype=float), ddof=1))
    assert stats["n"] == 3
    assert abs(stats["mean"] - 2.0) < 1e-12
    assert abs(stats["std"] - expected) < 1e-12
    # Population std (ddof=0) would be different:
    pop = float(np.std(np.asarray(values, dtype=float), ddof=0))
    assert abs(stats["std"] - pop) > 1e-12


def test_sample_sd_single_value_is_nan():
    stats = mean_and_sample_sd([4.2])
    assert stats["n"] == 1
    assert math.isnan(stats["std"])
