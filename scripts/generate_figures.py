#!/usr/bin/env python3
"""Generate visualization helpers for Protocol A localization maps.

This script does not retrain models. It can compute category-common
1st/99th percentile visualization limits from raw patch score arrays.

Quantitative metrics are NOT affected by these visualization limits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.localization import category_common_visualization_limits, patch_scores_to_maps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Protocol A figure helpers")
    parser.add_argument(
        "--raw-scores",
        type=str,
        required=True,
        help="Path to a .pt/.npy tensor of raw patch scores [N,196] or [N,P]",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/figures",
        help="Directory for optional map exports",
    )
    parser.add_argument(
        "--export-maps",
        action="store_true",
        help="Also export bilinear-upsampled localization maps as .npy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.raw_scores)
    if path.suffix.lower() in {".pt", ".pth"}:
        scores = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(scores, dict):
            scores = scores.get("raw_patch_scores", scores.get("scores"))
        scores = torch.as_tensor(scores).float()
    else:
        scores = torch.as_tensor(np.load(path)).float()

    limits = category_common_visualization_limits(scores)
    print("Category-common visualization limits (NOT used for metrics):")
    for k, v in limits.items():
        print(f"  {k}: {v}")

    if args.export_maps:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        maps = patch_scores_to_maps(scores)
        out_path = out_dir / "localization_maps.npy"
        np.save(out_path, maps)
        print("Wrote:", out_path)


if __name__ == "__main__":
    main()
