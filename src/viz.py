"""Shared chart styling for the notebooks.

One palette, defined once, so every figure in the write-up reads as one system.
The values below were checked with a colorblind-safety validator rather than
chosen by eye:

  categorical  worst adjacent CVD dE 9.1, normal-vision dE 22.9  (>= 8 / >= 15)
  ordinal      monotone lightness, single hue, light end 2.06:1 vs surface

Figures render on the light notebook surface, so only the light steps are
defined here. Two consequences of the contrast check are baked into the
helpers: aqua and yellow sit below 3:1 against the surface, so bars carry
visible value labels rather than relying on fill alone.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- palette ---------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Categorical slots, in fixed order. Never cycle past the end -- a 5th series
# folds into "Other" or becomes a small multiple.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

# Fit tiers are ordered worst -> best, so they get an ordinal ramp in one hue,
# not four arbitrary categorical colors.
TIER_COLORS = {
    "No Fit": "#86b6ef",
    "Partial Fit": "#2a78d6",
    "Good Fit": "#104281",
}

# Full sequential blue, for continuous magnitude (the confusion matrix).
SEQUENTIAL = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SEQUENTIAL_CMAP = mpl.colors.LinearSegmentedColormap.from_list("viz_blue", SEQUENTIAL)


def use_style() -> None:
    """Apply the shared rcParams. Call once at the top of a notebook."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "600",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "legend.frameon": False,
        "figure.dpi": 120,
        "lines.linewidth": 2,
        "patch.linewidth": 0,
    })


def despine(ax, *, keep=("left", "bottom")) -> None:
    """Drop chart junk: only the axes that carry a scale stay."""
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


def bar_labels(ax, bars, fmt: str = "{:.0f}", *, pad: int = 3) -> None:
    """Direct value labels.

    Not decoration: two palette slots fall below 3:1 against the surface, and
    the validator's relief rule requires visible labels wherever they are used.
    Labels wear ink, never the series color.
    """
    ax.bar_label(bars, fmt=fmt, padding=pad, color=INK_SECONDARY, fontsize=9)
