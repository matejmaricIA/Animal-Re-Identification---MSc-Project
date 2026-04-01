#!/usr/bin/env python3
"""
Generate a slide-friendly HITL-NIS sampling illustration.

This is intentionally *simple* and "presentation-first": fixed layout,
minimal text, no reliance on any project artifacts.

Outputs:
  - docs/Final Thesis/Figures/<name>.png
  - docs/Final Thesis/Figures/<name>.pdf
  - visualization_suite/output/presentation/<name>.png (copy)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


COL_BLUE = "#4E79A7"
COL_ORANGE = "#F28E2B"
COL_GREEN = "#59A14F"
COL_TEXT = "#111827"
COL_MUTED = "#6B7280"
COL_BG = "#FFFFFF"
COL_PANEL = "#F8FAFC"
COL_BORDER = "#E5E7EB"
COL_DASH = "#9CA3AF"
COL_DIFF = "#D62728"


def _panel(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=COL_PANEL,
            edgecolor=COL_BORDER,
            linewidth=1.2,
        )
    )


def _header(ax, title: str, subtitle: str | None = None) -> None:
    ax.text(0.04, 0.93, title, color=COL_TEXT, weight="bold", va="top")
    if subtitle:
        ax.text(0.04, 0.86, subtitle, color=COL_MUTED, va="top")


def _caption(ax, txt: str) -> None:
    ax.text(0.04, 0.05, txt, color=COL_MUTED, va="bottom")


def _component_box(
    ax,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    color: str,
    label: str,
) -> tuple[float, float, float, float]:
    # Rounded component container. Using boxes (not overlapping circles) keeps things legible at slide scale.
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.06",
            facecolor=color,
            edgecolor=color,
            alpha=0.10,
            linewidth=1.6,
            zorder=0,
        )
    )
    ax.text(x + 0.03, y + h - 0.04, label, color=COL_MUTED, weight="bold", fontsize=11, va="top")
    return x, y, w, h


def _nodes_in_box(
    ax,
    *,
    box: tuple[float, float, float, float],
    color: str,
    rel_pts: list[tuple[float, float]],
    size: float = 150,
) -> list[tuple[float, float]]:
    x, y, w, h = box
    pts: list[tuple[float, float]] = []
    for rx, ry in rel_pts:
        px = x + w * rx
        py = y + h * ry
        pts.append((px, py))
    ax.scatter(
        [p[0] for p in pts],
        [p[1] for p in pts],
        s=size,
        c=color,
        edgecolors="white",
        linewidths=1.8,
        zorder=3,
    )
    return pts


def _node_u(ax, *, xy: tuple[float, float], size: float = 220) -> None:
    ax.scatter([xy[0]], [xy[1]], s=size, c=COL_TEXT, edgecolors="white", linewidths=2.0, zorder=6)
    ax.text(xy[0], xy[1], "u", color="white", weight="bold", fontsize=10, ha="center", va="center", zorder=7)


def _arc(
    ax,
    a: tuple[float, float],
    b: tuple[float, float],
    *,
    color: str = COL_DASH,
    lw: float = 1.8,
    alpha: float = 0.9,
    rad: float = 0.2,
    dashed: bool = True,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-",
            linewidth=lw,
            color=color,
            alpha=alpha,
            linestyle=(0, (3, 3)) if dashed else "solid",
            zorder=2,
        )
    )


def _arrow(
    ax,
    a: tuple[float, float],
    b: tuple[float, float],
    *,
    color: str,
    lw: float = 3.0,
    alpha: float = 0.95,
    rad: float = 0.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=lw,
            color=color,
            alpha=alpha,
            zorder=5,
        )
    )


def _human_icon(ax, x: float, y: float, *, s: float = 0.08) -> None:
    # Simple head + body + laptop. Works well at small sizes.
    ax.add_patch(Circle((x, y + s * 0.65), s * 0.28, facecolor=COL_BORDER, edgecolor=COL_DASH, linewidth=1.0))
    ax.add_patch(
        Rectangle((x - s * 0.20, y + s * 0.20), s * 0.40, s * 0.30, facecolor=COL_BORDER, edgecolor=COL_DASH, linewidth=1.0)
    )
    ax.add_patch(
        Rectangle((x - s * 0.35, y - s * 0.05), s * 0.70, s * 0.25, facecolor="white", edgecolor=COL_DASH, linewidth=1.0)
    )


def _save_outputs(fig, *, out_base: Path, dpi: int) -> tuple[Path, Path]:
    out_png = out_base.with_suffix(".png")
    out_pdf = out_base.with_suffix(".pdf")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    fig.savefig(out_pdf)
    return out_png, out_pdf


def make_figure(*, n_images: int | None, n_vertices: int | None, n_neighbors: int | None) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
        }
    )

    # 4 panels -> bigger per-panel real estate, fewer overlaps.
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), facecolor=COL_BG)

    # Shared component layouts (boxes do not overlap, so it's readable when scaled down).
    box_A = (0.08, 0.40, 0.36, 0.32)
    box_B = (0.56, 0.40, 0.36, 0.32)
    box_C = (0.32, 0.14, 0.36, 0.22)

    # Relative node placements within a box (0..1 range).
    pts_A = [(0.30, 0.65), (0.72, 0.62), (0.35, 0.30), (0.74, 0.28)]
    pts_B = [(0.30, 0.60), (0.78, 0.56), (0.58, 0.26)]
    pts_C = [(0.30, 0.68), (0.76, 0.62), (0.50, 0.45), (0.28, 0.22), (0.78, 0.22)]

    # Panel (a) Graph view
    ax = axes[0]
    _panel(ax)
    _header(ax, "Graph view", "Components = individuals")
    A = _component_box(ax, x=box_A[0], y=box_A[1], w=box_A[2], h=box_A[3], color=COL_BLUE, label="A")
    B = _component_box(ax, x=box_B[0], y=box_B[1], w=box_B[2], h=box_B[3], color=COL_ORANGE, label="B")
    C = _component_box(ax, x=box_C[0], y=box_C[1], w=box_C[2], h=box_C[3], color=COL_GREEN, label="C")
    _nodes_in_box(ax, box=A, color=COL_BLUE, rel_pts=pts_A)
    _nodes_in_box(ax, box=B, color=COL_ORANGE, rel_pts=pts_B)
    _nodes_in_box(ax, box=C, color=COL_GREEN, rel_pts=pts_C)
    _caption(ax, "(a) Images grouped by individual")

    # Panel (b) Sample u uniformly
    ax = axes[1]
    _panel(ax)
    _header(ax, "Sampling (outer)", r"Pick $u\sim Q(u)$ (uniform)")
    A = _component_box(ax, x=box_A[0], y=box_A[1], w=box_A[2], h=box_A[3], color=COL_BLUE, label="A")
    B = _component_box(ax, x=box_B[0], y=box_B[1], w=box_B[2], h=box_B[3], color=COL_ORANGE, label="B")
    C = _component_box(ax, x=box_C[0], y=box_C[1], w=box_C[2], h=box_C[3], color=COL_GREEN, label="C")
    A_pts_abs = _nodes_in_box(ax, box=A, color=COL_BLUE, rel_pts=pts_A)
    _nodes_in_box(ax, box=B, color=COL_ORANGE, rel_pts=pts_B)
    _nodes_in_box(ax, box=C, color=COL_GREEN, rel_pts=pts_C)
    # Example sampled u (just a visual example; sampling is uniform over all vertices).
    u_xy = A_pts_abs[0]
    ax.add_patch(Circle(u_xy, 0.03, facecolor="none", edgecolor=COL_TEXT, linewidth=2.0, zorder=5))
    ax.text(0.04, 0.76, r"Here:  $Q(u)=1/|V|$", color=COL_MUTED)
    _caption(ax, "(b) Pick one anchor image")

    # Panel (c) Sample neighbors using q(v|u) from model similarity
    ax = axes[2]
    _panel(ax)
    _header(ax, "Sampling (inner)", r"Pick $v\sim q(v\mid u)$ from Re-ID similarity")
    A = _component_box(ax, x=box_A[0], y=box_A[1], w=box_A[2], h=box_A[3], color=COL_BLUE, label="A")
    B = _component_box(ax, x=box_B[0], y=box_B[1], w=box_B[2], h=box_B[3], color=COL_ORANGE, label="B")
    A_pts_abs = _nodes_in_box(ax, box=A, color=COL_BLUE, rel_pts=pts_A)
    B_pts_abs = _nodes_in_box(ax, box=B, color=COL_ORANGE, rel_pts=pts_B)
    _component_box(ax, x=box_C[0], y=box_C[1], w=box_C[2], h=box_C[3], color=COL_GREEN, label="C")
    _nodes_in_box(ax, box=box_C, color=COL_GREEN, rel_pts=pts_C)

    # Place u as a black node at a stable location (not overlapping arrows/labels).
    u_xy = (A[0] + A[2] * 0.55, A[1] + A[3] * 0.70)
    _node_u(ax, xy=u_xy)

    v_same = A_pts_abs[1]  # in same component
    v_diff = B_pts_abs[0]  # in different component
    _arrow(ax, u_xy, v_same, color=COL_BLUE, lw=4.8, rad=0.08)
    _arrow(ax, u_xy, v_diff, color=COL_DASH, lw=2.6, alpha=0.9, rad=0.0)
    ax.text(v_same[0] - 0.06, v_same[1] + 0.02, "high q", color=COL_BLUE, weight="bold", fontsize=10)
    ax.text(v_diff[0] - 0.02, v_diff[1] - 0.10, "low q", color=COL_MUTED, fontsize=10)
    _caption(ax, "(c) Query a few likely neighbors")

    # Panel (d) Annotate and estimate (make it explicit)
    ax = axes[3]
    _panel(ax)
    _header(ax, "Annotate + estimate", "Repeat and average contributions")

    # Pair query cartoon
    u0 = (0.18, 0.60)
    v0 = (0.34, 0.60)
    ax.scatter([u0[0]], [u0[1]], s=240, c=COL_TEXT, edgecolors="white", linewidths=2.0)
    ax.text(u0[0], u0[1], "u", color="white", weight="bold", fontsize=10, ha="center", va="center")
    ax.scatter([v0[0]], [v0[1]], s=220, c=COL_BLUE, edgecolors="white", linewidths=2.0)
    ax.text(v0[0], v0[1], "v", color="white", weight="bold", fontsize=10, ha="center", va="center")
    _arrow(ax, (u0[0] + 0.03, u0[1]), (v0[0] - 0.03, v0[1]), color=COL_DASH, lw=2.2)
    _human_icon(ax, 0.52, 0.52, s=0.20)
    ax.text(0.44, 0.40, "annotator:\n same / diff", color=COL_MUTED, fontsize=10, ha="center")

    # Repeat -> average -> output
    ax.text(0.10, 0.28, r"Repeat for $N$ anchors", color=COL_MUTED)
    if n_vertices is not None and n_neighbors is not None:
        ax.text(0.10, 0.22, rf"Example:  $N={int(n_vertices)}$,  $M={int(n_neighbors)}$ pairs each", color=COL_MUTED, fontsize=10)

    # Simple contribution bars
    bar_x = 0.16
    bar_y = 0.12
    bar_w = 0.04
    heights = [0.10, 0.16, 0.07, 0.13, 0.09]
    for i, h in enumerate(heights):
        ax.add_patch(Rectangle((bar_x + i * (bar_w + 0.015), bar_y), bar_w, h, facecolor=COL_BORDER, edgecolor=COL_DASH, linewidth=1.0))
    ax.text(bar_x, bar_y + 0.20, "contrib(u)", color=COL_MUTED, fontsize=10)

    # Arrow to output
    _arrow(ax, (0.43, 0.16), (0.72, 0.16), color=COL_DASH, lw=2.2)
    ax.text(0.74, 0.18, r"Output:", color=COL_MUTED, fontsize=11)
    ax.text(0.74, 0.12, r"$\hat K\ \pm\ CI$", color=COL_TEXT, weight="bold", fontsize=14)
    if n_images is not None:
        ax.text(0.74, 0.06, rf"$n={int(n_images)}$", color=COL_MUTED, fontsize=10)

    _caption(ax, "(d) Population estimate")

    fig.subplots_adjust(wspace=0.08, left=0.02, right=0.99, top=0.98, bottom=0.10)
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="nis_sampling_steps", help="Output base name (no extension).")
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--n_images", type=int, default=2078)
    p.add_argument("--n_vertices", type=int, default=150)
    p.add_argument("--n_neighbors", type=int, default=20)
    p.add_argument(
        "--no_numbers",
        action="store_true",
        help="Do not print n/N/M in the last panel.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    n_images = None if args.no_numbers else int(args.n_images)
    n_vertices = None if args.no_numbers else int(args.n_vertices)
    n_neighbors = None if args.no_numbers else int(args.n_neighbors)

    fig = make_figure(n_images=n_images, n_vertices=n_vertices, n_neighbors=n_neighbors)

    docs_out_base = Path("docs/Final Thesis/Figures") / str(args.name)
    out_png, _out_pdf = _save_outputs(fig, out_base=docs_out_base, dpi=int(args.dpi))

    # Copy PNG into visualization_suite output folder for convenience.
    pres_dir = Path("visualization_suite/output/presentation")
    pres_dir.mkdir(parents=True, exist_ok=True)
    (pres_dir / out_png.name).write_bytes(out_png.read_bytes())

    plt.close(fig)
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()
