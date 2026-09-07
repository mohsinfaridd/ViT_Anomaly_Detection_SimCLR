"""I/O helpers for paper figure scripts (paths, CSV checks, save, sample SD)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def repo_root_from_file(file_path: str | Path) -> Path:
    """figure_scripts/<module>.py -> repository root."""
    return Path(file_path).resolve().parents[1]


def default_outputs_dir(repo_root: Path) -> Path:
    return repo_root / "outputs"


def default_figures_dir(repo_root: Path) -> Path:
    return repo_root / "figures"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}. Have: {list(df.columns)}")


def sample_sd(values: Sequence[float]) -> float:
    """Sample standard deviation with ddof=1 (notebook multi-seed summary)."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size <= 1:
        return float("nan")
    return float(np.std(arr, ddof=1))


def mean_and_sample_sd(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    return float(np.mean(arr)), sample_sd(arr)


def load_csv(path: Path, required_columns: Sequence[str] | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if required_columns:
        require_columns(df, required_columns, str(path.name))
    return df


def save_figure_bundle(fig, out_dir: Path, stem: str) -> dict[str, Path]:
    """Save PDF (manuscript), SVG (editable), and 300-dpi PNG."""
    ensure_dir(out_dir)
    save_kwargs = {
        "bbox_inches": "tight",
        "pad_inches": 0.03,
        "facecolor": "white",
    }
    paths = {
        "pdf": out_dir / f"{stem}.pdf",
        "svg": out_dir / f"{stem}.svg",
        "png": out_dir / f"{stem}.png",
    }
    fig.savefig(paths["pdf"], format="pdf", **save_kwargs)
    fig.savefig(paths["svg"], format="svg", **save_kwargs)
    fig.savefig(paths["png"], format="png", dpi=300, **save_kwargs)
    for kind, path in paths.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"Failed to write non-empty {kind}: {path}")
    return paths


def add_common_figure_args(parser: argparse.ArgumentParser, repo_root: Path) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_figures_dir(repo_root),
        help="Directory for Fig*.pdf/svg/png (default: <repo>/figures)",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=default_outputs_dir(repo_root),
        help="Directory with reproducibility CSVs (default: <repo>/outputs)",
    )
