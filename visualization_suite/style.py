"""Matplotlib styling utilities for high-quality figures."""
from __future__ import annotations

import matplotlib.pyplot as plt


def set_style() -> None:
    """Apply a consistent style suitable for thesis figures."""
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })
    plt.style.use("seaborn-v0_8")


def enable_latex() -> None:
    """Enable LaTeX rendering in matplotlib."""
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"


def save_high_res(fig, path: str, dpi: int = 300) -> None:
    """Save ``fig`` to ``path`` with ``dpi`` resolution."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
