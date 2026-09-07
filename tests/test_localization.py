"""Unit tests for localization map conversion."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.localization import category_common_visualization_limits, patch_scores_to_maps


def test_patch_map_196_to_224():
    scores = torch.randn(5, 196)
    maps = patch_scores_to_maps(scores, patch_grid=(14, 14), image_size=224)
    assert maps.shape == (5, 224, 224)


def test_visualization_limits_are_percentiles_only():
    scores = torch.arange(196, dtype=torch.float32).unsqueeze(0).repeat(10, 1)
    limits = category_common_visualization_limits(scores)
    assert limits["affects_quantitative_metrics"] is False
    assert limits["low_quantile"] == 0.01
    assert limits["high_quantile"] == 0.99
    assert limits["vmax"] > limits["vmin"]
