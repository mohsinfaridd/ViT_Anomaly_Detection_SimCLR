#!/usr/bin/env python3
"""Figure 7 — Per-category raw-patch pixel ROC-AUC across three seeds.

Scientific status: three-seed aggregation (mean ± sample SD, ddof=1).
Metric: raw-patch PIXEL ROC-AUC only (not pixel AP / AUPRO / image ROC-AUC).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figure_scripts import io_utils, style


def load_fig7_table(outputs_dir: Path) -> pd.DataFrame:
    path = outputs_dir / "fig7_pixel_roc_auc_three_seed.csv"
    df = io_utils.load_csv(path, ["seed", "category", "pixel_roc_auc"])
    seeds = sorted(df["seed"].unique().tolist())
    if seeds != [42, 123, 2026]:
        raise ValueError(f"Expected seeds [42, 123, 2026], got {seeds}")
    n_cat = df["category"].nunique()
    if n_cat != 15:
        raise ValueError(f"Expected 15 MVTec categories, got {n_cat}")
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category, sub in df.groupby("category", sort=True):
        vals = 100.0 * sub.sort_values("seed")["pixel_roc_auc"].astype(float).to_numpy()
        if len(vals) != 3:
            raise ValueError(f"{category}: expected 3 seeds, got {len(vals)}")
        mean, sd = io_utils.mean_and_sample_sd(vals)
        rows.append(
            {
                "category": category,
                "mean_pct": mean,
                "sd_pct": sd,
                "seed42": vals[0],
                "seed123": vals[1],
                "seed2026": vals[2],
            }
        )
    out = pd.DataFrame(rows).sort_values("mean_pct", ascending=False).reset_index(drop=True)
    return out


def build_figure(outputs_dir: Path):
    style.apply_manuscript_style()
    df = load_fig7_table(outputs_dir)
    agg = aggregate(df)

    print("Figure 7 per-category mean pixel ROC-AUC (%):")
    for _, row in agg.iterrows():
        print(f"  {row['category']:12s} {row['mean_pct']:6.2f} ± {row['sd_pct']:5.2f}")

    fig, ax = plt.subplots(figsize=(style.MANUSCRIPT_WIDTH_IN, 4.0))
    y = np.arange(len(agg), dtype=float)
    highlight = agg["category"].eq("screw").to_numpy()
    colors = ["#b2182b" if h else "#2f6db3" for h in highlight]

    ax.barh(
        y,
        agg["mean_pct"],
        xerr=agg["sd_pct"],
        color=colors,
        edgecolor="#333333",
        linewidth=0.5,
        capsize=2.5,
        error_kw={"elinewidth": 0.9, "capthick": 0.9},
        zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ") for c in agg["category"]])
    ax.invert_yaxis()
    ax.set_xlabel("Pixel ROC-AUC (%)")
    ax.set_xlim(60, 100)
    handles = [
        Patch(facecolor="#2f6db3", edgecolor="#333333", label="Other categories"),
        Patch(
            facecolor="#b2182b",
            edgecolor="#333333",
            label="Screw (image-level failure case)",
        ),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    io_utils.add_common_figure_args(p, REPO_ROOT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fig = build_figure(Path(args.outputs_dir))
    paths = io_utils.save_figure_bundle(fig, Path(args.output_dir), "Fig7")
    plt.close(fig)
    print("Saved:", {k: str(v) for k, v in paths.items()})


if __name__ == "__main__":
    main()
