#!/usr/bin/env python3
"""Figure 4 — Seed-42 SSL training history + official-test score distributions.

Scientific status: representative Seed 42 only (not a three-seed average).

Panel (a): SimCLR NT-Xent train/val history (best checkpoint epoch 192).
Panel (b): Official-test primary LOCAL detector scores (local_normalized /
normalized_score alias), decision boundary at normalized score = 1.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figure_scripts import io_utils, style


def _load_ssl_history(outputs_dir: Path) -> pd.DataFrame:
    path = outputs_dir / "fig4_ssl_history_seed42.csv"
    df = io_utils.load_csv(path, ["epoch", "train_nt_xent", "val_nt_xent"])
    return df.sort_values("epoch")


def _load_raw_scores(outputs_dir: Path) -> pd.DataFrame | None:
    """Prefer per-image local_normalized scores when archived."""
    candidates = [
        outputs_dir / "fig4_test_scores_seed42.csv",
        outputs_dir / "protocolA_official_test_predictions.csv",
        outputs_dir / "seed42" / "protocolA_official_test_predictions.csv",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        score_col = None
        for c in ("local_normalized", "normalized_score"):
            if c in df.columns:
                score_col = c
                break
        if score_col is None or "binary_label" not in df.columns:
            continue
        out = df[["binary_label", score_col]].rename(columns={score_col: "local_normalized"})
        print(f"Figure 4 panel (b): using raw scores from {path}")
        return out
    return None


def _load_or_build_histogram(outputs_dir: Path) -> pd.DataFrame:
    hist_path = outputs_dir / "fig4_test_score_histogram_seed42.csv"
    if hist_path.is_file():
        df = pd.read_csv(hist_path)
        if {"class", "bin_left", "bin_right", "count"}.issubset(df.columns) and len(df):
            if set(df["class"].unique()) - {"Normal", "Anomaly", "unknown"}:
                pass
            if "unknown" not in set(df["class"].unique()) or (
                "Normal" in set(df["class"].unique()) and "Anomaly" in set(df["class"].unique())
            ):
                if "Normal" in set(df["class"].unique()):
                    return df

    # Rebuild from the notebook-exported SVG if present beside the repo / via env.
    svg_candidates = [
        outputs_dir / "ssl_assets" / "protocolA_test_score_distribution.svg",
        REPO_ROOT / "figures_source" / "protocolA_test_score_distribution.svg",
    ]
    # Optional sibling Manuscript tree (local archive only; not hardcoded into logic paths beyond search)
    manuscript_svg = (
        REPO_ROOT.parent.parent
        / "Manuscript"
        / "FIGURES"
        / "protocolA_test_score_distribution.svg"
    )
    svg_candidates.append(manuscript_svg)

    svg_path = next((p for p in svg_candidates if p.is_file()), None)
    if svg_path is None:
        raise FileNotFoundError(
            "Figure 4 panel (b) needs outputs/fig4_test_scores_seed42.csv "
            "(columns: binary_label, local_normalized) or a digitized histogram CSV "
            "outputs/fig4_test_score_histogram_seed42.csv."
        )

    df = _digitize_score_histogram_svg(svg_path)
    io_utils.ensure_dir(outputs_dir)
    df.to_csv(hist_path, index=False)
    print(f"Figure 4 panel (b): digitized histogram from {svg_path} -> {hist_path}")
    return df


def _digitize_score_histogram_svg(svg_path: Path) -> pd.DataFrame:
    text = svg_path.read_text(encoding="utf-8")
    ax_x, ax_y = 42.180312, 20.038594
    ax_w = 239.170313 - 42.180312
    ax_h = 161.438594 - 20.038594
    xlim = (-1.0, 6.0)
    ylim = (0.0, 120.0)

    rows = []
    for m in re.finditer(r'<path d="([^"]+)"[^>]*style="([^"]*)"', text):
        d, style_attr = m.group(1), m.group(2)
        if "fill: #" not in style_attr:
            continue
        if "1f77b4" in style_attr:
            cls = "Normal"
        elif "ff7f0e" in style_attr:
            cls = "Anomaly"
        else:
            continue
        pts = [
            (float(a), float(b))
            for _, a, b in re.findall(r"([ML])\s*([-\d.]+)\s+([-\d.]+)", d)
        ]
        if len(pts) < 4:
            continue
        xs = np.asarray([p[0] for p in pts], float)
        ys = np.asarray([p[1] for p in pts], float)
        w = xs.max() - xs.min()
        h = ys.max() - ys.min()
        if not (0.5 < w < 12 and h > 0.5):
            continue
        left = xlim[0] + (xs.min() - ax_x) / ax_w * (xlim[1] - xlim[0])
        right = xlim[0] + (xs.max() - ax_x) / ax_w * (xlim[1] - xlim[0])
        top = ylim[1] - (ys.min() - ax_y) / ax_h * (ylim[1] - ylim[0])
        bottom = ylim[1] - (ys.max() - ax_y) / ax_h * (ylim[1] - ylim[0])
        count = max(0.0, top - max(bottom, 0.0))
        rows.append(
            {
                "class": cls,
                "bin_left": left,
                "bin_right": right,
                "count": count,
                "provenance": f"digitized_from_svg:{svg_path.name}",
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No histogram bars parsed from {svg_path}")
    return df


def build_figure(outputs_dir: Path):
    style.apply_manuscript_style()
    hist = None
    scores = _load_raw_scores(outputs_dir)
    ssl = _load_ssl_history(outputs_dir)
    if scores is None:
        hist = _load_or_build_histogram(outputs_dir)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(style.MANUSCRIPT_WIDTH_IN * 0.52, 6.4),
        constrained_layout=True,
    )

    ax = axes[0]
    ax.plot(ssl["epoch"], ssl["train_nt_xent"], label="Train", color="#1f77b4")
    ax.plot(ssl["epoch"], ssl["val_nt_xent"], label="Validation", color="#ff7f0e")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NT-Xent loss")
    ax.set_title("(a) SimCLR training history", loc="left", fontsize=style.TITLE_FONT)
    ax.legend(frameon=True, fancybox=False, edgecolor="#bbbbbb")
    ax.set_xlim(0, 200)

    ax = axes[1]
    if scores is not None:
        normal = scores.loc[scores["binary_label"] == 0, "local_normalized"].to_numpy()
        anomaly = scores.loc[scores["binary_label"] == 1, "local_normalized"].to_numpy()
        ax.hist(normal, bins=35, alpha=0.6, label="Normal", color="#1f77b4")
        ax.hist(anomaly, bins=35, alpha=0.6, label="Anomaly", color="#ff7f0e")
    else:
        assert hist is not None
        for cls, color in [("Normal", "#1f77b4"), ("Anomaly", "#ff7f0e")]:
            sub = hist[hist["class"] == cls]
            widths = (sub["bin_right"] - sub["bin_left"]).to_numpy()
            ax.bar(
                sub["bin_left"],
                sub["count"],
                width=widths,
                align="edge",
                alpha=0.6,
                color=color,
                label=cls,
                linewidth=0,
            )
    ax.axvline(1.0, linestyle="--", color="#1f77b4", label="Frozen threshold")
    ax.set_xlabel("Normalized anomaly score")
    ax.set_ylabel("Images")
    ax.set_title("(b) Official-test anomaly scores", loc="left", fontsize=style.TITLE_FONT)
    ax.legend(frameon=True, fancybox=False, edgecolor="#bbbbbb")

    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    io_utils.add_common_figure_args(p, REPO_ROOT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fig = build_figure(Path(args.outputs_dir))
    paths = io_utils.save_figure_bundle(fig, Path(args.output_dir), "Fig4")
    plt.close(fig)
    print("Saved:", {k: str(v) for k, v in paths.items()})


if __name__ == "__main__":
    main()
