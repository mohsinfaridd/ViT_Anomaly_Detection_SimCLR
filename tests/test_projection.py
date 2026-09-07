"""Unit tests for deterministic patch projection."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features import make_patch_projection


def test_projection_shape_2304_to_256():
    proj = make_patch_projection(2304, 256, seed=42)
    assert proj.shape == (2304, 256)


def test_deterministic_same_seed():
    a = make_patch_projection(2304, 256, seed=42)
    b = make_patch_projection(2304, 256, seed=42)
    assert torch.equal(a, b)


def test_different_seed_differs():
    a = make_patch_projection(2304, 256, seed=42)
    b = make_patch_projection(2304, 256, seed=123)
    assert not torch.equal(a, b)


def test_orthonormal_columns():
    proj = make_patch_projection(2304, 256, seed=42).double()
    gram = proj.T @ proj
    assert torch.allclose(gram, torch.eye(256, dtype=torch.float64), atol=1e-6)
