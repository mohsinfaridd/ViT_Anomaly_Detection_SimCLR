#!/usr/bin/env python3
"""Run Protocol A from a YAML config.

Example:
  python scripts/run_protocol_a.py --config configs/seed42.yaml
  python scripts/run_protocol_a.py --config configs/seed42.yaml --data-root /path/to/mvtec
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path when run as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.protocol_a import run_protocol_a


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Protocol A (local-patch primary detector).")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to seed YAML config (e.g. configs/seed42.yaml)",
    )
    parser.add_argument("--data-root", type=str, default=None, help="Override MVTec data root")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override SSL checkpoint path")
    parser.add_argument("--output-root", type=str, default=None, help="Override output root")
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail if CUDA is unavailable",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        cfg.data_root = args.data_root
    if args.checkpoint:
        cfg.checkpoint_path = args.checkpoint
    if args.output_root:
        cfg.output_root = args.output_root
    if args.require_gpu:
        cfg.require_gpu = True

    print("Protocol A")
    print("  seed:", cfg.seed)
    print("  primary detector:", cfg.primary_detector)
    print("  aggregation:", cfg.final_fixed_patch_aggregation)
    print("  data_root:", cfg.resolve_data_root())
    print("  checkpoint:", cfg.resolve_checkpoint_path())
    print("  output_root:", cfg.resolve_output_root())

    result = run_protocol_a(cfg)
    metrics = result["primary_metrics"]
    print("\nPrimary local_patch metrics:")
    for key in (
        "balanced_accuracy",
        "anomaly_precision",
        "anomaly_recall",
        "anomaly_f1",
        "roc_auc",
        "average_precision",
        "mcc",
    ):
        print(f"  {key}: {metrics.get(key)}")
    print("Outputs written under:", result["output_root"])


if __name__ == "__main__":
    main()
