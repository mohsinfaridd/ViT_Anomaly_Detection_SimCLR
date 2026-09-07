#!/usr/bin/env python3
"""Generate manuscript Figures 4–7 that can be reproduced from repository outputs.

Example:
  python figure_scripts/generate_all.py
  python figure_scripts/generate_all.py --data-root /path/to/mvtec --checkpoint-root checkpoints

Figure 6 is skipped (with a clear message) when MVTec/checkpoint deps are absent,
unless a verified final Figure 6 asset can be installed into figures/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figure_scripts import io_utils
from figure_scripts.fig4_training_and_scores import build_figure as build_fig4
from figure_scripts.fig5_global_local_hybrid import build_figure as build_fig5
from figure_scripts.fig6_localization_examples import generate_fig6
from figure_scripts.fig7_pixel_roc_auc import build_figure as build_fig7
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    io_utils.add_common_figure_args(p, REPO_ROOT)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--checkpoint-root", type=Path, default=None)
    p.add_argument(
        "--skip-fig6",
        action="store_true",
        help="Skip Figure 6 entirely (even asset install)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    outputs_dir = Path(args.outputs_dir)
    io_utils.ensure_dir(out_dir)

    results = {}

    # Figures 4, 5, 7 from archived CSVs / digitized notebook exports
    for name, builder in [
        ("Fig4", build_fig4),
        ("Fig5", build_fig5),
        ("Fig7", build_fig7),
    ]:
        try:
            fig = builder(outputs_dir)
            paths = io_utils.save_figure_bundle(fig, out_dir, name)
            plt.close(fig)
            results[name] = ("ok", paths)
            print(f"{name}: saved")
        except Exception as exc:
            results[name] = ("error", str(exc))
            print(f"{name}: FAILED — {exc}")

    if args.skip_fig6:
        print("Figure 6 skipped: --skip-fig6")
        results["Fig6"] = ("skipped", "--skip-fig6")
    else:
        ok, message, paths = generate_fig6(
            output_dir=out_dir,
            outputs_dir=outputs_dir,
            data_root=args.data_root,
            checkpoint_root=args.checkpoint_root,
        )
        print(message)
        results["Fig6"] = ("ok" if ok else "skipped", paths or message)

    print("\nSummary:")
    status = 0
    for name in ("Fig4", "Fig5", "Fig6", "Fig7"):
        state, info = results[name]
        print(f"  {name}: {state}")
        if name != "Fig6" and state != "ok":
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
