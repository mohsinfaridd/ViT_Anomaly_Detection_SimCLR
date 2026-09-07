#!/usr/bin/env python3
"""Figure 5 — Global / local-patch / hybrid image-level ablation (3 seeds).

Scientific status: three-seed aggregation (mean ± sample SD, ddof=1).

Roles:
  Global-only = ablation
  Local-patch = PRIMARY / proposed detector
  Hybrid (25% global) = secondary ablation (do not promote to primary)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figure_scripts import io_utils, style

METHOD_ORDER = ["global_only", "local_patch", "hybrid_fixed_g25"]
METHOD_LABELS = {
    "global_only": "Global-only (ablation)",
    "local_patch": "Local-patch (proposed)",
    "hybrid_fixed_g25": "Hybrid 25/75 (secondary)",
}
METHOD_COLORS = {
    "global_only": "#6b7c3a",
    "local_patch": "#2f6db3",
    "hybrid_fixed_g25": "#f0b27a",
}
METHOD_HATCH = {
    "global_only": "",
    "local_patch": "",
    "hybrid_fixed_g25": "///",
}

# metric_key -> (CSV column, display label, scale_to_percent)
METRICS = [
    ("roc_auc", "Image ROC-AUC", True),
    ("balanced_accuracy", "Balanced accuracy", True),
    ("anomaly_f1", "F1", True),
    ("anomaly_recall", "Recall", True),
]


def load_fig5_table(outputs_dir: Path) -> pd.DataFrame:
    path = outputs_dir / "fig5_global_local_hybrid_three_seed.csv"
    df = io_utils.load_csv(
        path,
        [
            "seed",
            "method",
            "balanced_accuracy",
            "anomaly_recall",
            "anomaly_f1",
            "roc_auc",
        ],
    )
    missing = [m for m in METHOD_ORDER if m not in set(df["method"])]
    if missing:
        raise ValueError(f"Figure 5 CSV missing methods: {missing}")
    seeds = sorted(df["seed"].unique().tolist())
    if seeds != [42, 123, 2026]:
        raise ValueError(f"Expected seeds [42, 123, 2026], got {seeds}")
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        sub = df[df["method"] == method]
        for col, label, as_pct in METRICS:
            vals = sub[col].astype(float).to_numpy()
            if as_pct:
                vals = 100.0 * vals
            mean, sd = io_utils.mean_and_sample_sd(vals)
            rows.append(
                {
                    "method": method,
                    "metric": label,
                    "mean": mean,
                    "sd": sd,
                    "seed_values": ";".join(f"{v:.10g}" for v in vals),
                }
            )
    return pd.DataFrame(rows)


def build_figure(outputs_dir: Path):
    style.apply_manuscript_style()
    df = load_fig5_table(outputs_dir)
    agg = aggregate(df)

    # Cross-check printout (exact notebook-derived means)
    print("Figure 5 three-seed means ± sample SD:")
    for metric in [m[1] for m in METRICS]:
        parts = []
        for method in METHOD_ORDER:
            row = agg[(agg["method"] == method) & (agg["metric"] == metric)].iloc[0]
            parts.append(f"{METHOD_LABELS[method]}={row['mean']:.2f}±{row['sd']:.2f}")
        print(f"  {metric}: " + " | ".join(parts))

    fig, ax = plt.subplots(figsize=(style.MANUSCRIPT_WIDTH_IN, 3.35))
    x = np.arange(len(METRICS), dtype=float)
    width = 0.24
    metric_labels = [m[1] for m in METRICS]

    for i, method in enumerate(METHOD_ORDER):
        means = []
        sds = []
        for label in metric_labels:
            row = agg[(agg["method"] == method) & (agg["metric"] == label)].iloc[0]
            means.append(row["mean"])
            sds.append(row["sd"])
        ax.bar(
            x + (i - 1) * width,
            means,
            width,
            yerr=sds,
            color=METHOD_COLORS[method],
            edgecolor="#333333",
            linewidth=0.6,
            hatch=METHOD_HATCH[method],
            capsize=3,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
            label=METHOD_LABELS[method],
            zorder=3,
        )

    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 90)
    ax.legend(frameon=False, loc="upper right")
    # No internal giant title; caption carries explanation.
    fig.tight_layout()
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    io_utils.add_common_figure_args(p, REPO_ROOT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fig = build_figure(Path(args.outputs_dir))
    paths = io_utils.save_figure_bundle(fig, Path(args.output_dir), "Fig5")
    plt.close(fig)
    print("Saved:", {k: str(v) for k, v in paths.items()})


if __name__ == "__main__":
    main()
