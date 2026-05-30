"""Shared plotting style + path bootstrap for the figure scripts."""

import os
import sys

# make the package importable when scripts are run directly
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# muted, print-friendly palette
C_PERP = "#1b6ca8"     # blue
C_PARA = "#c1453b"     # red
C_DELTA = "#2a7d4f"    # green
C_NEUTRAL = "#444444"
C_ACCENT = "#d08c00"   # amber


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save(fig, name):
    apply = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(apply + ".png", bbox_inches="tight", dpi=300)
    fig.savefig(apply + ".pdf", bbox_inches="tight", dpi=300)
    print(f"  wrote figures/{name}.png  and  .pdf")
    return apply
