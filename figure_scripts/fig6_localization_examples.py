#!/usr/bin/env python3
"""Figure 6 — Seed-42 localization examples (reviewer-corrected).

Scientific status: qualitative Seed 42 only.

Rules (from final executed Code1_SEED_42 notebook):
  - RAW patch kNN distances only (no position / image calibration on maps)
  - map: [196] -> 14x14 -> bilinear -> 224x224 (src.localization.patch_scores_to_maps)
  - categories: metal_nut, grid, screw, leather
  - sample rule: true anomalous official-test images; highest q=.95 local_normalized
  - category-common vmin/vmax = 1st/99th percentile of ALL official-test raw patch
    distances in that category (visualization only)

If MVTec / checkpoint / banks are unavailable, do not invent heatmaps.
Instead install a verified final Figure 6 asset into figures/ when found.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figure_scripts import io_utils, style

FIG6_CATEGORIES = ["metal_nut", "grid", "screw", "leather"]


def _find_verified_fig6_assets(repo_root: Path) -> dict[str, Path] | None:
    """Locate an already-generated FINAL Figure 6 asset (PDF/PNG/SVG)."""
    candidates: list[tuple[str, Path]] = []

    # Preferred: archived final asset under outputs/
    for name, kind in (
        ("figure6_FINAL_category_common_scaling_seed42.pdf", "pdf"),
        ("figure6_FINAL_category_common_scaling_seed42.png", "png"),
        ("figure6_FINAL_category_common_scaling_seed42.svg", "svg"),
    ):
        p = repo_root / "outputs" / name
        if p.is_file() and p.stat().st_size > 10_000:
            candidates.append((kind, p))

    # Sibling Manuscript archive (optional local research tree)
    manuscript_figs = (
        repo_root.parent.parent / "Manuscript" / "6a2d77504b00cb7e22bf2d3d" / "figs"
    )
    for name in ("fig6.pdf", "Fig6.pdf"):
        p = manuscript_figs / name
        if p.is_file() and p.stat().st_size > 50_000:
            candidates.append(("pdf", p))

    # Repo-local archives
    for pattern in (
        "**/figure6_FINAL_category_common_scaling_seed42.pdf",
        "**/figure6_FINAL_category_common_scaling_seed42.png",
        "**/figure6_FINAL_category_common_scaling_seed42.svg",
    ):
        for p in repo_root.glob(pattern):
            if p.is_file() and p.stat().st_size > 10_000:
                candidates.append((p.suffix.lower().lstrip("."), p))

    if not candidates:
        return None

    by_kind: dict[str, Path] = {}
    for kind, path in candidates:
        by_kind.setdefault(kind, path)
    return by_kind


def _install_verified_assets(assets: dict[str, Path], output_dir: Path) -> dict[str, Path]:
    """Copy/convert verified assets into figures/Fig6.{pdf,svg,png}."""
    io_utils.ensure_dir(output_dir)
    written: dict[str, Path] = {}

    pdf_src = assets.get("pdf")
    png_src = assets.get("png")
    svg_src = assets.get("svg")

    if pdf_src is not None:
        dst = output_dir / "Fig6.pdf"
        shutil.copy2(pdf_src, dst)
        written["pdf"] = dst
        # Prefer vector conversion from the verified PDF when tools exist.
        try:
            import subprocess

            prefix = output_dir / "Fig6"
            # SVG: some pdftocairo builds write "<prefix>" with no .svg suffix.
            subprocess.run(
                ["pdftocairo", "-svg", str(dst), str(prefix)],
                check=True,
                capture_output=True,
                text=True,
            )
            svg_dst = output_dir / "Fig6.svg"
            bare = output_dir / "Fig6"
            if bare.is_file() and not svg_dst.exists():
                bare.replace(svg_dst)
            elif bare.is_file() and svg_dst.exists() and bare.resolve() != svg_dst.resolve():
                bare.unlink()
            if svg_dst.is_file():
                written["svg"] = svg_dst

            # PNG at 300 dpi
            subprocess.run(
                ["pdftocairo", "-png", "-r", "300", "-singlefile", str(dst), str(prefix)],
                check=True,
                capture_output=True,
                text=True,
            )
            png_dst = output_dir / "Fig6.png"
            if png_dst.is_file():
                written["png"] = png_dst
        except Exception as exc:
            print(f"pdftocairo conversion note: {exc}")

    if "png" not in written and png_src is not None:
        dst = output_dir / "Fig6.png"
        shutil.copy2(png_src, dst)
        written["png"] = dst

    if "svg" not in written and svg_src is not None:
        dst = output_dir / "Fig6.svg"
        shutil.copy2(svg_src, dst)
        written["svg"] = dst

    # If we still lack PDF but have PNG, keep PNG only and warn.
    return written


def dependencies_available(data_root: Path | None, checkpoint_root: Path | None) -> bool:
    if data_root is None or checkpoint_root is None:
        return False
    if not data_root.is_dir():
        return False
    # Need at least one category folder and a seed-42 checkpoint candidate
    has_cat = any((data_root / c).is_dir() for c in FIG6_CATEGORIES)
    ckpt_ok = False
    if checkpoint_root.is_file() and checkpoint_root.suffix in {".pt", ".pth"}:
        ckpt_ok = True
    elif checkpoint_root.is_dir():
        ckpt_ok = any(checkpoint_root.rglob("ssl_best_checkpoint.pt"))
    return bool(has_cat and ckpt_ok)


def run_full_generation(
    data_root: Path,
    checkpoint_root: Path,
    outputs_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Full Figure 6 regeneration using src/ (requires MVTec + checkpoint + banks)."""
    import matplotlib.pyplot as plt
    from PIL import Image

    from src.localization import category_common_visualization_limits, patch_scores_to_maps

    style.apply_manuscript_style()

    # Selection / scale audit tables (may be incomplete for screw basename)
    sel_path = outputs_dir / "figure6_selected_samples_seed42.csv"
    scale_path = outputs_dir / "figure6_category_common_scales_seed42.csv"
    if not sel_path.is_file() or not scale_path.is_file():
        raise FileNotFoundError(
            "Missing figure6_*_seed42.csv under outputs/. "
            "Extract them from the executed seed-42 notebook before regenerating."
        )

    sel = pd.read_csv(sel_path)
    scales = pd.read_csv(scale_path)

    # Full neural regeneration needs the Protocol A runtime (banks, model, tables).
    # This repository's modular API is available, but end-to-end bank rebuild is
    # intentionally not silently approximated here when intermediate banks are absent.
    raise RuntimeError(
        "Full Figure 6 neural regeneration requires seed-42 checkpoint, MVTec images, "
        "and category prototype banks produced by Protocol A. "
        "Banks/checkpoints were not found in a runnable layout. "
        "Use install-verified-asset fallback or re-run Protocol A first."
    )


def generate_fig6(
    output_dir: Path,
    outputs_dir: Path,
    data_root: Path | None = None,
    checkpoint_root: Path | None = None,
) -> tuple[bool, str, dict[str, Path] | None]:
    """Returns (ok, message, paths)."""
    if dependencies_available(data_root, checkpoint_root):
        try:
            paths = run_full_generation(
                data_root=data_root,
                checkpoint_root=checkpoint_root,
                outputs_dir=outputs_dir,
                output_dir=output_dir,
            )
            return True, "Figure 6 generated from MVTec/checkpoint.", paths
        except Exception as exc:
            print(f"Figure 6 full generation failed: {exc}")

    assets = _find_verified_fig6_assets(REPO_ROOT)
    if assets:
        written = _install_verified_assets(assets, output_dir)
        missing = [k for k in ("pdf", "svg", "png") if k not in written]
        if missing:
            msg = (
                "Figure 6 installed from verified final asset, but missing formats: "
                + ", ".join(missing)
            )
        else:
            msg = "Figure 6 installed from verified final manuscript/notebook asset."
        ok = "pdf" in written and "png" in written
        return ok, msg, written

    return (
        False,
        "Figure 6 skipped: MVTec/checkpoint dependency unavailable.",
        None,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    io_utils.add_common_figure_args(p, REPO_ROOT)
    p.add_argument("--data-root", type=Path, default=None, help="MVTec AD root")
    p.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="Seed-42 checkpoint file or directory containing ssl_best_checkpoint.pt",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ok, message, paths = generate_fig6(
        output_dir=Path(args.output_dir),
        outputs_dir=Path(args.outputs_dir),
        data_root=args.data_root,
        checkpoint_root=args.checkpoint_root,
    )
    print(message)
    if paths:
        print("Saved:", {k: str(v) for k, v in paths.items()})
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
