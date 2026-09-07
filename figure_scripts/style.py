"""Shared manuscript figure style for Figures 4–7.

Scientific status (do not imply all figures are three-seed averages):
  Figure 4 — representative Seed 42 only
  Figure 5 — three-seed aggregation
  Figure 6 — qualitative Seed 42 only
  Figure 7 — three-seed aggregation
"""

from __future__ import annotations

import warnings

import matplotlib as mpl
from matplotlib import font_manager

MANUSCRIPT_WIDTH_IN = 6.30

BASE_FONT = 10.5
LABEL_FONT = 11.0
TITLE_FONT = 11.0
TICK_FONT = 9.5
LEGEND_FONT = 9.5
COLORBAR_FONT = 9.0

LINEWIDTH = 1.25
AXES_LINEWIDTH = 0.8
SAVE_PAD_INCHES = 0.03

SERIF_STACK = [
    "Times New Roman",
    "Times",
    "Nimbus Roman",
    "Liberation Serif",
    "DejaVu Serif",
]


def times_new_roman_available() -> bool:
    names = {f.name for f in font_manager.fontManager.ttflist}
    return "Times New Roman" in names


def apply_manuscript_style() -> bool:
    """Apply shared rcParams. Returns True if Times New Roman is installed."""
    available = times_new_roman_available()
    if not available:
        warnings.warn(
            "Times New Roman is not installed; using a Times-compatible serif fallback.",
            UserWarning,
            stacklevel=2,
        )
        print(
            "WARNING: Times New Roman is not installed in this runtime. "
            "Figures request Times New Roman first and will fall back to a "
            "Times-compatible serif font."
        )
    else:
        print("Figure font: Times New Roman")

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": SERIF_STACK,
            "font.size": BASE_FONT,
            "axes.titlesize": TITLE_FONT,
            "axes.labelsize": LABEL_FONT,
            "xtick.labelsize": TICK_FONT,
            "ytick.labelsize": TICK_FONT,
            "legend.fontsize": LEGEND_FONT,
            "figure.titlesize": TITLE_FONT,
            "axes.linewidth": AXES_LINEWIDTH,
            "lines.linewidth": LINEWIDTH,
            "lines.markersize": 5.0,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
        }
    )
    return available
