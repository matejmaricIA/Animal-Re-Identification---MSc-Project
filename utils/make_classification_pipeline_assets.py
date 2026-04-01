from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from constants import INLIER_THRESHOLD, NORMALIZED_THRESHOLD_DIVISOR
from geometric_verification import (
    match_features_by_descriptors,
    match_features_lightglue,
    match_features_loftr,
    normalize_coordinates,
)


def _json_default(obj):
    """Convert numpy/path types to plain JSON-serializable Python values."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _read_bgr(path: str | Path | None) -> np.ndarray | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return img


def _fit_square(image: np.ndarray, size: int, fill: tuple[int, int, int] = (18, 18, 18)) -> tuple[np.ndarray, tuple[float, int, int]]:
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        canvas = np.full((size, size, 3), fill, dtype=np.uint8)
        return canvas, (1.0, 0, 0)
    scale = min(size / float(w), size / float(h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), fill, dtype=np.uint8)
    dx = (size - nw) // 2
    dy = (size - nh) // 2
    canvas[dy : dy + nh, dx : dx + nw] = resized
    return canvas, (scale, dx, dy)


def _map_points(points: np.ndarray, transform: tuple[float, int, int], *, x_offset: int = 0, y_offset: int = 0) -> np.ndarray:
    scale, dx, dy = transform
    mapped = np.zeros_like(points, dtype=np.float32)
    mapped[:, 0] = points[:, 0] * scale + dx + x_offset
    mapped[:, 1] = points[:, 1] * scale + dy + y_offset
    return mapped


def _coerce_keypoints_xy(keypoints: np.ndarray | None) -> np.ndarray:
    """Return keypoints as (N, 2) float32, handling shapes like (1,N,2)."""
    if keypoints is None:
        return np.empty((0, 2), dtype=np.float32)
    pts = np.asarray(keypoints, dtype=np.float32)
    if pts.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    pts = pts.squeeze()
    if pts.ndim == 1:
        if pts.shape[0] < 2:
            return np.empty((0, 2), dtype=np.float32)
        pts = pts.reshape(-1, 2)
    elif pts.ndim > 2:
        pts = pts.reshape(-1, pts.shape[-1])
    if pts.ndim != 2 or pts.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float32)
    pts = pts[:, :2].astype(np.float32, copy=False)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    return pts


def _prepare_keypoints_for_image(keypoints: np.ndarray | None, image: np.ndarray) -> np.ndarray:
    """Convert keypoints to image pixel coordinates when needed."""
    pts = _coerce_keypoints_xy(keypoints)
    if len(pts) == 0:
        return pts
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return np.empty((0, 2), dtype=np.float32)

    x = pts[:, 0]
    y = pts[:, 1]
    # Some extractors store normalized [0,1] coordinates.
    if np.min(x) >= 0.0 and np.min(y) >= 0.0 and np.max(x) <= 1.5 and np.max(y) <= 1.5:
        pts = pts.copy()
        pts[:, 0] = pts[:, 0] * float(max(1, w - 1))
        pts[:, 1] = pts[:, 1] * float(max(1, h - 1))

    pts = pts.copy()
    pts[:, 0] = np.clip(pts[:, 0], 0.0, float(max(0, w - 1)))
    pts[:, 1] = np.clip(pts[:, 1], 0.0, float(max(0, h - 1)))
    return pts


def _sample_points(points: np.ndarray, max_points: int, *, seed: int = 42) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=int(max_points), replace=False)
    return points[np.sort(idx)]


def _draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    *,
    max_points: int = 350,
    point_radius: int = 3,
    ring_radius: int = 4,
) -> np.ndarray:
    out = image.copy()
    pts = _prepare_keypoints_for_image(keypoints, image)
    if len(pts) == 0:
        return out
    pts = _sample_points(pts, max_points, seed=42)

    # Darken the base image slightly so keypoints pop on busy textures.
    out = cv2.addWeighted(out, 0.82, np.zeros_like(out), 0.18, 0.0)

    # Glow layer for a more pronounced (presentation-friendly) keypoint visualization.
    glow = np.zeros_like(out)
    glow_color = (255, 255, 0)  # cyan (BGR)
    glow_r = int(max(6, ring_radius * 3))
    for x, y in pts:
        cv2.circle(glow, (int(x), int(y)), glow_r, glow_color, -1, cv2.LINE_AA)
    sigma = float(max(2.0, ring_radius * 1.4))
    glow = cv2.GaussianBlur(glow, (0, 0), sigmaX=sigma, sigmaY=sigma)
    out = cv2.addWeighted(out, 1.0, glow, 0.35, 0.0)

    # Crisp rings + centers on top of the glow.
    for x, y in pts:
        p = (int(x), int(y))
        cv2.circle(out, p, int(ring_radius + 2), (0, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(out, p, int(ring_radius), (245, 245, 245), 2, cv2.LINE_AA)
        cv2.circle(out, p, int(point_radius), glow_color, -1, cv2.LINE_AA)
    return out


def _draw_descriptor_signature(descriptors: np.ndarray, width: int = 960, height: int = 300) -> np.ndarray:
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    cv2.putText(canvas, "Local Descriptor Signature", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)

    if descriptors is None or descriptors.size == 0:
        cv2.putText(canvas, "No descriptors available", (20, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2, cv2.LINE_AA)
        return canvas

    vec = np.mean(np.abs(descriptors), axis=0).astype(np.float32)
    if vec.size > 96:
        vec = vec[:96]
    vmax = float(np.max(vec)) if vec.size else 1.0
    vec = vec / (vmax + 1e-9)

    chart_x0 = 20
    chart_y0 = 58
    chart_w = width - 40
    chart_h = height - 110
    cv2.rectangle(canvas, (chart_x0, chart_y0), (chart_x0 + chart_w, chart_y0 + chart_h), (225, 225, 225), 1, cv2.LINE_AA)

    n = max(1, len(vec))
    bar_w = max(1, chart_w // n)
    for i, v in enumerate(vec):
        x0 = chart_x0 + i * bar_w
        x1 = min(chart_x0 + chart_w - 1, x0 + bar_w - 1)
        y1 = chart_y0 + chart_h - 1
        y0 = int(round(y1 - v * (chart_h - 6)))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (90, 150, 60), -1, cv2.LINE_AA)

    cv2.putText(
        canvas,
        f"dims={int(descriptors.shape[1])}  points={int(descriptors.shape[0])}",
        (20, height - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _draw_vector_signature(
    vector: np.ndarray | None,
    *,
    title: str,
    width: int = 960,
    height: int = 300,
    color: tuple[int, int, int] = (150, 110, 40),
    show_dims: bool = True,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    cv2.putText(canvas, title, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)

    vec = np.asarray(vector if vector is not None else np.empty((0,), dtype=np.float32), dtype=np.float32).reshape(-1)
    if vec.size == 0:
        cv2.putText(canvas, "No vector available", (20, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2, cv2.LINE_AA)
        return canvas

    vec = np.abs(vec)
    if vec.size > 128:
        vec = vec[:128]
    vmax = float(np.max(vec)) if vec.size else 1.0
    vec = vec / (vmax + 1e-9)

    chart_x0 = 20
    chart_y0 = 58
    chart_w = width - 40
    chart_h = height - 110
    cv2.rectangle(canvas, (chart_x0, chart_y0), (chart_x0 + chart_w, chart_y0 + chart_h), (225, 225, 225), 1, cv2.LINE_AA)

    n = max(1, len(vec))
    bar_w = max(1, chart_w // n)
    for i, v in enumerate(vec):
        x0 = chart_x0 + i * bar_w
        x1 = min(chart_x0 + chart_w - 1, x0 + bar_w - 1)
        y1 = chart_y0 + chart_h - 1
        y0 = int(round(y1 - v * (chart_h - 6)))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, -1, cv2.LINE_AA)

    if show_dims:
        cv2.putText(
            canvas,
            f"dims={int(np.asarray(vector).reshape(-1).shape[0])}",
            (20, height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _normalize_for_plot(values: list[float | None]) -> list[float]:
    arr = np.asarray([np.nan if v is None else float(v) for v in values], dtype=np.float64)
    finite = np.isfinite(arr)
    out = np.zeros_like(arr, dtype=np.float64)
    if not np.any(finite):
        return out.tolist()
    lo = float(np.nanmin(arr[finite]))
    hi = float(np.nanmax(arr[finite]))
    if hi - lo < 1e-12:
        out[finite] = 0.5
    else:
        out[finite] = (arr[finite] - lo) / (hi - lo)
    return out.tolist()


def _normalize_relative_to_best(values: list[float | None]) -> list[float]:
    """Scale each value by the best score in the list (no min-max normalization)."""
    arr = np.asarray([np.nan if v is None else float(v) for v in values], dtype=np.float64)
    finite = np.isfinite(arr)
    out = np.zeros_like(arr, dtype=np.float64)
    if not np.any(finite):
        return out.tolist()
    vals = arr[finite]
    best = float(np.nanmax(vals))
    if best > 1e-12:
        out[finite] = np.clip(vals / best, 0.0, 1.0)
        return out.tolist()
    abs_best = float(np.nanmax(np.abs(vals)))
    if abs_best > 1e-12:
        out[finite] = np.clip(np.abs(vals) / abs_best, 0.0, 1.0)
    return out.tolist()


def _vector_profile(vector: np.ndarray | None, *, max_dims: int = 128) -> np.ndarray:
    vec = np.asarray(vector if vector is not None else np.empty((0,), dtype=np.float32), dtype=np.float32).reshape(-1)
    if vec.size == 0:
        return np.empty((0,), dtype=np.float32)
    vec = np.abs(vec)
    if vec.size > max_dims:
        vec = vec[:max_dims]
    vmax = float(np.max(vec)) if vec.size else 1.0
    vec = vec / (vmax + 1e-9)
    return vec.astype(np.float32)


def _draw_profile_row(
    canvas: np.ndarray,
    profile: np.ndarray,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    color: tuple[int, int, int],
) -> None:
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (225, 225, 225), -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (190, 190, 190), 1, cv2.LINE_AA)
    if profile.size == 0:
        cv2.putText(canvas, "missing", (x + 8, y + h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
        return
    n = int(profile.size)
    for i, v in enumerate(profile):
        x0 = x + int(round((i / max(1, n)) * w))
        y1 = y + h - 2
        y0 = int(round(y1 - float(v) * (h - 4)))
        cv2.line(canvas, (x0, y1), (x0, y0), color, 1, cv2.LINE_AA)


def _draw_vector_database_panel(
    *,
    query_id: str,
    query_class: str | None,
    query_vector: np.ndarray | None,
    ranked_entries: list[dict],
    db_vectors: dict[str, np.ndarray] | None,
    title: str,
    top_k: int,
    width: int = 1700,
    max_dims: int = 128,
    color: tuple[int, int, int] = (120, 120, 120),
) -> np.ndarray:
    rows = list(ranked_entries[: max(0, int(top_k))])
    header_h = 72
    row_h = 56
    bottom_h = 20
    height = header_h + row_h * (1 + len(rows)) + bottom_h
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    cv2.putText(canvas, title, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)

    left_w = 540
    plot_x = left_w
    plot_w = width - left_w - 20
    plot_h = 34

    q_class = "-" if query_class is None else str(query_class)
    q_text = f"Q {query_id}  class={q_class}"
    y = header_h
    cv2.putText(canvas, q_text, (14, y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1, cv2.LINE_AA)
    q_profile = _vector_profile(query_vector, max_dims=max_dims)
    _draw_profile_row(canvas, q_profile, x=plot_x, y=y + 8, w=plot_w, h=plot_h, color=color)

    db_vectors = db_vectors or {}
    for i, e in enumerate(rows, start=1):
        y = header_h + i * row_h
        tid = str(e.get("train_id", ""))
        cls = "-" if e.get("label") is None else str(e.get("label"))
        score_raw = e.get("score")
        score = _fmt_score(float(score_raw), precision=3) if score_raw is not None else "-"
        txt = f"#{i} {tid}  class={cls}  sim={score}"
        cv2.putText(canvas, txt, (14, y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (35, 35, 35), 1, cv2.LINE_AA)
        prof = _vector_profile(db_vectors.get(tid), max_dims=max_dims)
        _draw_profile_row(canvas, prof, x=plot_x, y=y + 8, w=plot_w, h=plot_h, color=color)

    return canvas


def _draw_fusion_breakdown(
    entries: list[dict],
    *,
    top_k: int,
    width: int = 1700,
    overview_mode: bool = False,
) -> np.ndarray:
    rows = list(entries[: max(0, int(top_k))])
    if not rows:
        canvas = np.full((180, width, 3), 248, dtype=np.uint8)
        cv2.putText(canvas, "Tier-2 Fusion Breakdown", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.putText(canvas, "No candidates available", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2, cv2.LINE_AA)
        return canvas

    g_vals = [r.get("global_score") for r in rows]
    f_vals = [r.get("fisher_score") for r in rows]
    t2_vals = [r.get("tier2_score") for r in rows]
    if overview_mode:
        g_norm = _normalize_relative_to_best(g_vals)
        f_norm = _normalize_relative_to_best(f_vals)
        t2_norm = _normalize_relative_to_best(t2_vals)

        header_h = 96
        row_h = 56
        bottom_h = 18
        height = header_h + row_h * len(rows) + bottom_h
        canvas = np.full((height, width, 3), 248, dtype=np.uint8)
        cv2.putText(canvas, "Tier-2 Fusion Overview", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "",
            (20, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )

        left_w = 130
        col_gap = 18
        usable_w = width - left_w - 20 - 2 * col_gap
        col_w = max(120, usable_w // 3)
        x_g = left_w
        x_f = x_g + col_w + col_gap
        x_t2 = x_f + col_w + col_gap
        y0 = header_h

        headers = [(x_g, "Global", (214, 136, 69)), (x_f, "Fisher", (64, 160, 255)), (x_t2, "Fused", (89, 169, 76))]
        for x, name, c in headers:
            cv2.putText(canvas, name, (x + 4, y0 - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.74, c, 2, cv2.LINE_AA)

        pad_x = 8
        pad_y = 9
        value_w = 84
        for i, row in enumerate(rows):
            y = y0 + i * row_h
            cy = y + row_h // 2
            tid = str(row.get("train_id", ""))
            flag = row.get("_is_correct")
            if flag is True:
                mark, mark_color = "C", (30, 140, 30)
            elif flag is False:
                mark, mark_color = "W", (20, 20, 180)
            else:
                mark, mark_color = "?", (90, 90, 90)
            cv2.putText(canvas, f"#{i+1}", (14, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (35, 35, 35), 2, cv2.LINE_AA)
            cv2.putText(canvas, mark, (64, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.78, mark_color, 2, cv2.LINE_AA)
            cv2.putText(canvas, tid[:10], (84, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (90, 90, 90), 1, cv2.LINE_AA)

            for x_col, rel, raw, color in [
                (x_g, g_norm[i], g_vals[i], (214, 136, 69)),
                (x_f, f_norm[i], f_vals[i], (64, 160, 255)),
                (x_t2, t2_norm[i], t2_vals[i], (89, 169, 76)),
            ]:
                box_x = x_col + pad_x
                box_y = y + pad_y
                box_w = col_w - 2 * pad_x - value_w
                box_h = row_h - 2 * pad_y
                cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (225, 225, 225), -1, cv2.LINE_AA)
                cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (195, 195, 195), 1, cv2.LINE_AA)
                fill_w = int(round(float(np.clip(rel, 0.0, 1.0)) * box_w))
                if fill_w > 0:
                    cv2.rectangle(canvas, (box_x, box_y), (box_x + fill_w, box_y + box_h), color, -1, cv2.LINE_AA)
                raw_txt = _fmt_score(raw, precision=3) if raw is not None else "-"
                cv2.putText(
                    canvas,
                    raw_txt,
                    (box_x + box_w + 10, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (45, 45, 45),
                    1,
                    cv2.LINE_AA,
                )

        return canvas

    g_norm = _normalize_for_plot(g_vals)
    f_norm = _normalize_for_plot(f_vals)
    t2_norm = _normalize_for_plot(t2_vals)

    header_h = 72
    row_h = 38
    bottom_h = 24
    height = header_h + row_h * len(rows) + bottom_h
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)

    cv2.putText(canvas, "Tier-2 Fusion Breakdown (Global + Fisher -> Tier-2)", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)

    left_w = 560
    col_w = 340
    col_gap = 16
    x_g = left_w
    x_f = x_g + col_w + col_gap
    x_t2 = x_f + col_w + col_gap
    bar_pad_x = 10
    bar_w = col_w - 120
    bar_h = 12
    y0 = header_h

    for x, name in [(x_g, "Global"), (x_f, "Fisher"), (x_t2, "Tier-2 fused")]:
        cv2.putText(canvas, name, (x + 8, y0 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (40, 40, 40), 1, cv2.LINE_AA)

    for i, row in enumerate(rows):
        y = y0 + i * row_h
        cy = y + row_h // 2
        tid = str(row.get("train_id", ""))
        cls = "-" if row.get("label") is None else str(row.get("label"))
        src = str(row.get("source", "-"))
        left_txt = f"#{i+1} {tid}  class={cls}  src={src}"
        cv2.putText(canvas, left_txt, (14, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (30, 30, 30), 1, cv2.LINE_AA)

        specs = [
            (x_g, g_norm[i], g_vals[i], (214, 136, 69)),
            (x_f, f_norm[i], f_vals[i], (64, 160, 255)),
            (x_t2, t2_norm[i], t2_vals[i], (89, 169, 76)),
        ]
        for x_col, nrm, raw, color in specs:
            by = cy - bar_h // 2
            cv2.rectangle(canvas, (x_col + bar_pad_x, by), (x_col + bar_pad_x + bar_w, by + bar_h), (225, 225, 225), -1, cv2.LINE_AA)
            fill_w = int(round(float(np.clip(nrm, 0.0, 1.0)) * bar_w))
            if fill_w > 0:
                cv2.rectangle(canvas, (x_col + bar_pad_x, by), (x_col + bar_pad_x + fill_w, by + bar_h), color, -1, cv2.LINE_AA)
            cv2.rectangle(canvas, (x_col + bar_pad_x, by), (x_col + bar_pad_x + bar_w, by + bar_h), (190, 190, 190), 1, cv2.LINE_AA)
            txt = _fmt_score(raw, precision=4) if raw is not None else "-"
            cv2.putText(canvas, txt, (x_col + bar_pad_x + bar_w + 12, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1, cv2.LINE_AA)

    return canvas


def _compute_inlier_mask(matched_q: np.ndarray, matched_d: np.ndarray) -> np.ndarray:
    if matched_q is None or matched_d is None:
        return np.zeros((0,), dtype=bool)
    if len(matched_q) < 4 or len(matched_d) < 4:
        return np.zeros((len(matched_q),), dtype=bool)

    qn = normalize_coordinates(np.asarray(matched_q, dtype=np.float32))
    dn = normalize_coordinates(np.asarray(matched_d, dtype=np.float32))

    try:
        _, mask = cv2.findHomography(
            qn.reshape(-1, 1, 2),
            dn.reshape(-1, 1, 2),
            cv2.RANSAC,
            ransacReprojThreshold=float(INLIER_THRESHOLD) / float(NORMALIZED_THRESHOLD_DIVISOR),
        )
    except cv2.error:
        mask = None

    if mask is None:
        return np.zeros((len(matched_q),), dtype=bool)
    return mask.ravel().astype(bool)


def _compute_matches(
    query_desc: np.ndarray,
    query_kp: np.ndarray,
    cand_desc: np.ndarray,
    cand_kp: np.ndarray,
    *,
    matcher: str,
    method: str,
    query_image_path: str | None,
    candidate_image_path: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matcher = (matcher or "ratio").lower()

    if matcher == "loftr" and query_image_path and candidate_image_path:
        _, mq, md = match_features_loftr(query_image_path, candidate_image_path)
        mq = np.asarray(mq, dtype=np.float32)
        md = np.asarray(md, dtype=np.float32)
        inliers = _compute_inlier_mask(mq, md)
        return mq, md, inliers

    if matcher == "lightglue":
        _, mq, md = match_features_lightglue(query_desc, cand_desc, query_kp, cand_kp, method=method)
        mq = np.asarray(mq, dtype=np.float32)
        md = np.asarray(md, dtype=np.float32)
        if len(mq) == 0:
            _, mq, md = match_features_by_descriptors(query_desc, cand_desc, query_kp, cand_kp)
            mq = np.asarray(mq, dtype=np.float32)
            md = np.asarray(md, dtype=np.float32)
        inliers = _compute_inlier_mask(mq, md)
        return mq, md, inliers

    _, mq, md = match_features_by_descriptors(query_desc, cand_desc, query_kp, cand_kp)
    mq = np.asarray(mq, dtype=np.float32)
    md = np.asarray(md, dtype=np.float32)
    inliers = _compute_inlier_mask(mq, md)
    return mq, md, inliers


def _draw_match_panel(
    query_img: np.ndarray,
    cand_img: np.ndarray,
    query_pts: np.ndarray,
    cand_pts: np.ndarray,
    inlier_mask: np.ndarray,
    *,
    side: int,
    max_display_matches: int = 15,
    overview_mode: bool = False,
) -> np.ndarray:
    q_canvas, q_t = _fit_square(query_img, side)
    d_canvas, d_t = _fit_square(cand_img, side)

    header_h = 72 if overview_mode else 0
    panel = np.full((side + header_h, side * 2, 3), 14, dtype=np.uint8)
    if overview_mode:
        panel[:header_h, :] = 244
    panel[header_h:, :side] = q_canvas
    panel[header_h:, side:] = d_canvas

    if len(query_pts) == 0:
        if overview_mode:
            cv2.putText(panel, "MATCHES=0  INLIERS=0", (16, header_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (30, 30, 30), 2, cv2.LINE_AA)
            cv2.putText(panel, "No matches", (20, header_h + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2, cv2.LINE_AA)
        else:
            cv2.putText(panel, "No matches", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2, cv2.LINE_AA)
        return panel

    q_arr = _coerce_keypoints_xy(query_pts)
    d_arr = _coerce_keypoints_xy(cand_pts)
    n = min(len(q_arr), len(d_arr))
    if n == 0:
        if overview_mode:
            cv2.putText(panel, "MATCHES=0  INLIERS=0", (16, header_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (30, 30, 30), 2, cv2.LINE_AA)
            cv2.putText(panel, "No matches", (20, header_h + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2, cv2.LINE_AA)
        else:
            cv2.putText(panel, "No matches", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2, cv2.LINE_AA)
        return panel
    q_arr = q_arr[:n]
    d_arr = d_arr[:n]

    mask = np.asarray(inlier_mask).astype(bool).ravel() if inlier_mask is not None else np.zeros((0,), dtype=bool)
    if len(mask) < n:
        padded = np.zeros((n,), dtype=bool)
        if len(mask) > 0:
            padded[: len(mask)] = mask[: len(mask)]
        mask = padded
    else:
        mask = mask[:n]

    inlier_count = int(np.sum(mask))
    total = int(n)

    draw_idx = np.arange(n)
    if max_display_matches > 0 and n > max_display_matches:
        rng = np.random.default_rng(42)
        draw_idx = np.sort(rng.choice(n, size=int(max_display_matches), replace=False))

    q_draw = q_arr[draw_idx]
    d_draw = d_arr[draw_idx]
    q_m = _map_points(np.asarray(q_draw, dtype=np.float32), q_t, y_offset=header_h)
    d_m = _map_points(np.asarray(d_draw, dtype=np.float32), d_t, x_offset=side, y_offset=header_h)

    line_color = (0, 210, 0)  # green
    dot_color = (0, 0, 255)   # red
    for pq, pd in zip(q_m, d_m):
        p1 = (int(pq[0]), int(pq[1]))
        p2 = (int(pd[0]), int(pd[1]))
        cv2.line(panel, p1, p2, line_color, 1, cv2.LINE_AA)
        cv2.circle(panel, p1, 3, dot_color, -1, cv2.LINE_AA)
        cv2.circle(panel, p2, 3, dot_color, -1, cv2.LINE_AA)

    title = f"MATCHES={total}  INLIERS={inlier_count}"
    if overview_mode:
        font = 1.1
        thick = 2
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font, thick)
        tx = max(16, (panel.shape[1] - tw) // 2)
        ty = max(th + 8, header_h - 16)
        cv2.putText(panel, title, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font, (28, 28, 28), thick, cv2.LINE_AA)
    else:
        cv2.putText(panel, f"matches={total}  inliers={inlier_count}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2, cv2.LINE_AA)
    return panel


def _correctness_label(entry: dict) -> str:
    v = entry.get("_is_correct", None)
    if v is True:
        return "CORRECT"
    if v is False:
        return "WRONG"
    return "UNKNOWN"


def _compose_strip(
    *,
    query_img: np.ndarray,
    query_id: str,
    query_class: str | None,
    entries: list[dict],
    train_image_paths: dict[str, str],
    title: str,
    detail_fn: Callable[[dict], str],
    panel_size: int,
    top_k: int,
    overview_mode: bool = False,
) -> np.ndarray:
    entries = list(entries[: max(0, int(top_k))])
    gap = 12 if overview_mode else 10
    header_h = 66 if overview_mode else 52
    footer_h = 56 if overview_mode else 66
    tile_h = panel_size
    n_tiles = 1 + len(entries)
    width = gap + n_tiles * (panel_size + gap)
    height = header_h + tile_h + footer_h

    canvas = np.full((height, width, 3), 246, dtype=np.uint8)
    title_y = 42 if overview_mode else 34
    title_scale = 1.05 if overview_mode else 0.9
    cv2.putText(canvas, title, (14, title_y), cv2.FONT_HERSHEY_SIMPLEX, title_scale, (20, 20, 20), 2, cv2.LINE_AA)

    q_tile, _ = _fit_square(query_img, panel_size, fill=(30, 30, 30))
    x0 = gap
    y0 = header_h
    canvas[y0 : y0 + panel_size, x0 : x0 + panel_size] = q_tile
    cv2.rectangle(canvas, (x0, y0), (x0 + panel_size, y0 + panel_size), (66, 66, 66), 1, cv2.LINE_AA)
    if overview_mode:
        cv2.putText(canvas, "QUERY", (x0 + 6, y0 + panel_size + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.84, (35, 35, 35), 2, cv2.LINE_AA)
    else:
        q_label = f"Q: {query_id}"
        q_cls = "-" if query_class is None else str(query_class)
        cv2.putText(canvas, q_label, (x0 + 6, y0 + panel_size + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (35, 35, 35), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"class={q_cls}", (x0 + 6, y0 + panel_size + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (55, 55, 55), 1, cv2.LINE_AA)

    for i, entry in enumerate(entries, start=1):
        cx = gap + i * (panel_size + gap)
        image = _read_bgr(train_image_paths.get(str(entry.get("train_id", ""))))
        if image is None:
            tile = np.full((panel_size, panel_size, 3), 60, dtype=np.uint8)
            cv2.putText(tile, "missing", (18, panel_size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2, cv2.LINE_AA)
        else:
            tile, _ = _fit_square(image, panel_size, fill=(30, 30, 30))

        canvas[y0 : y0 + panel_size, cx : cx + panel_size] = tile
        border_color = (96, 96, 96)
        border_thickness = 1
        if overview_mode:
            correct = entry.get("_is_correct", None)
            if correct is True:
                border_color = (55, 168, 40)
                border_thickness = 3
            elif correct is False:
                border_color = (35, 35, 220)
                border_thickness = 3
            else:
                border_color = (120, 120, 120)
                border_thickness = 2
        cv2.rectangle(
            canvas,
            (cx, y0),
            (cx + panel_size, y0 + panel_size),
            border_color,
            border_thickness,
            cv2.LINE_AA,
        )

        if overview_mode:
            txt = f"#{i}  {detail_fn(entry)}"
            cv2.putText(canvas, txt, (cx + 6, y0 + panel_size + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (35, 35, 35), 2, cv2.LINE_AA)
        else:
            rid = str(entry.get("train_id", ""))
            label = entry.get("label")
            class_id = "-" if label is None else str(label)
            txt1 = f"#{i} {rid}"
            txt2 = f"class={class_id}"
            txt3 = detail_fn(entry)
            cv2.putText(canvas, txt1, (cx + 4, y0 + panel_size + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (35, 35, 35), 1, cv2.LINE_AA)
            cv2.putText(canvas, txt2, (cx + 4, y0 + panel_size + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (35, 35, 35), 1, cv2.LINE_AA)
            cv2.putText(canvas, txt3, (cx + 4, y0 + panel_size + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (55, 55, 55), 1, cv2.LINE_AA)

    return canvas


def _save_image(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")
    return str(path)


def _resolve_union_entries(result: dict) -> list[dict]:
    global_map = {str(x.get("train_id")): x for x in result.get("global_ranked", [])}
    fisher_map = {str(x.get("train_id")): x for x in result.get("fisher_ranked", [])}
    global_ids = set(global_map.keys())
    fisher_ids = set(fisher_map.keys())
    tier2_map = {str(x.get("train_id")): x for x in result.get("tier2_ranked", [])}

    entries = []
    for train_id in result.get("union_ids", []):
        tid = str(train_id)
        src_g = tid in global_ids
        src_f = tid in fisher_ids
        source = "G+F" if src_g and src_f else ("G" if src_g else ("F" if src_f else "-"))
        tier2_item = tier2_map.get(tid, {})
        g_item = global_map.get(tid, {})
        f_item = fisher_map.get(tid, {})
        tier2_raw = tier2_item.get("tier2_score", None)
        tier2_score = float(tier2_raw) if tier2_raw is not None else None
        g_raw = g_item.get("score", None)
        g_score = float(g_raw) if g_raw is not None else None
        f_raw = f_item.get("score", None)
        f_score = float(f_raw) if f_raw is not None else None
        label = tier2_item.get("label", g_item.get("label", f_item.get("label")))
        entries.append(
            {
                "train_id": tid,
                "source": source,
                "global_score": g_score,
                "fisher_score": f_score,
                "tier2_score": tier2_score,
                "label": label,
            }
        )
    return entries


def _resolve_union_entries_display(result: dict, *, top_k: int) -> list[dict]:
    """Resolve a *display* union: Global top-K + Fisher top-K (deduped, ordered)."""
    k = max(0, int(top_k))
    global_ranked = result.get("global_ranked", []) or []
    fisher_ranked = result.get("fisher_ranked", []) or []

    global_top = [str(x.get("train_id")) for x in global_ranked[:k] if x.get("train_id") is not None]
    fisher_top = [str(x.get("train_id")) for x in fisher_ranked[:k] if x.get("train_id") is not None]
    union_ids = list(dict.fromkeys(global_top + fisher_top))

    global_map = {str(x.get("train_id")): x for x in global_ranked if x.get("train_id") is not None}
    fisher_map = {str(x.get("train_id")): x for x in fisher_ranked if x.get("train_id") is not None}
    tier2_map = {str(x.get("train_id")): x for x in result.get("tier2_ranked", []) or [] if x.get("train_id") is not None}
    global_set = set(global_top)
    fisher_set = set(fisher_top)

    entries: list[dict] = []
    for tid in union_ids:
        src_g = tid in global_set
        src_f = tid in fisher_set
        source = "G+F" if src_g and src_f else ("G" if src_g else ("F" if src_f else "-"))

        tier2_item = tier2_map.get(tid, {})
        g_item = global_map.get(tid, {})
        f_item = fisher_map.get(tid, {})

        tier2_raw = tier2_item.get("tier2_score", None)
        tier2_score = float(tier2_raw) if tier2_raw is not None else None
        g_raw = g_item.get("score", None)
        g_score = float(g_raw) if g_raw is not None else None
        f_raw = f_item.get("score", None)
        f_score = float(f_raw) if f_raw is not None else None
        label = tier2_item.get("label", g_item.get("label", f_item.get("label")))

        entries.append(
            {
                "train_id": tid,
                "source": source,
                "global_score": g_score,
                "fisher_score": f_score,
                "tier2_score": tier2_score,
                "label": label,
            }
        )
    return entries


def _fmt_score(val: float | None, precision: int = 3) -> str:
    if val is None:
        return "-"
    return f"{float(val):.{precision}f}"


def _with_correctness(entries: list[dict], query_class: str | None) -> list[dict]:
    q = None if query_class is None else str(query_class)
    out: list[dict] = []
    for e in entries:
        row = dict(e)
        label = row.get("label")
        if q is None or label is None:
            row["_is_correct"] = None
        else:
            row["_is_correct"] = str(label) == q
        out.append(row)
    return out


def _select_predicted_entry(result: dict) -> dict | None:
    """Pick the top predicted candidate, preferring final Tier-3 ranking."""
    for key in ("tier3_ranked", "tier2_ranked", "global_ranked", "fisher_ranked"):
        entries = result.get(key, [])
        if entries:
            top = entries[0]
            if isinstance(top, dict) and top.get("train_id") is not None:
                out = dict(top)
                out["_source_rank"] = key
                return out
    return None


def _draw_predicted_image_tile(
    *,
    image: np.ndarray | None,
    panel_size: int,
    train_id: str | None,
    class_id: str | None,
    source_rank: str | None,
    overview_mode: bool = False,
) -> np.ndarray:
    tile = np.full((panel_size, panel_size, 3), 40, dtype=np.uint8)
    if image is not None:
        tile = _fit_square(image, panel_size, fill=(30, 30, 30))[0]
    if overview_mode:
        return tile
    cv2.rectangle(tile, (0, 0), (panel_size - 1, panel_size - 1), (96, 96, 96), 1, cv2.LINE_AA)
    info_bg_h = 62
    cv2.rectangle(tile, (0, panel_size - info_bg_h), (panel_size, panel_size), (0, 0, 0), -1)
    tid = "-" if train_id is None else str(train_id)
    cls = "-" if class_id is None else str(class_id)
    src = "-" if source_rank is None else str(source_rank).replace("_ranked", "")
    cv2.putText(tile, "Predicted Top-1", (10, panel_size - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(tile, f"id={tid}  class={cls}  src={src}", (10, panel_size - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (230, 230, 230), 1, cv2.LINE_AA)
    return tile


def build_assets_from_funnel(
    *,
    out_dir: Path,
    query_id: str,
    query_class: str | None,
    query_raw_path: str | None,
    query_processed_path: str,
    query_segmented_path: str | None,
    train_processed_paths: dict[str, str],
    query_keypoints: np.ndarray,
    query_descriptors: np.ndarray,
    query_global_embedding: np.ndarray | None,
    query_fisher_vector: np.ndarray | None,
    train_global_embeddings: dict[str, np.ndarray] | None,
    train_fisher_vectors: dict[str, np.ndarray] | None,
    train_keypoints: dict[str, np.ndarray],
    train_descriptors: dict[str, np.ndarray],
    result: dict,
    gv_matcher: str,
    gv_features: str,
    image_paths: dict[str, str] | None = None,
    top_k: int = 8,
    panel_size: int = 320,
    overview_mode: bool = False,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    query_processed = _read_bgr(query_processed_path)
    if query_processed is None:
        raise FileNotFoundError(f"Query processed image not found: {query_processed_path}")

    query_raw = _read_bgr(query_raw_path) if query_raw_path else None
    if query_raw is None:
        query_raw = query_processed.copy()

    query_seg = _read_bgr(query_segmented_path) if query_segmented_path else None
    if query_seg is None:
        query_seg = query_processed.copy()

    query_local = _draw_keypoints(
        query_processed,
        query_keypoints,
        point_radius=4 if overview_mode else 3,
        ring_radius=6 if overview_mode else 4,
    )
    fisher_sig = _draw_descriptor_signature(np.asarray(query_descriptors, dtype=np.float32))
    global_sig = _draw_vector_signature(
        query_global_embedding,
        title="Global Embedding Signature",
        color=(176, 132, 50),
        show_dims=not overview_mode,
    )
    fisher_vec_sig = _draw_vector_signature(
        query_fisher_vector,
        title="Fisher Vector Signature",
        color=(70, 170, 230),
        show_dims=not overview_mode,
    )

    assets: dict[str, str] = {}
    assets["query_original"] = _save_image(out_dir / "asset_query_original.png", _fit_square(query_raw, panel_size)[0])
    assets["query_segmented"] = _save_image(out_dir / "asset_query_segmented.png", _fit_square(query_seg, panel_size)[0])
    assets["query_local_keypoints"] = _save_image(out_dir / "asset_query_local_keypoints.png", _fit_square(query_local, panel_size)[0])
    assets["query_local_descriptor_signature"] = _save_image(
        out_dir / "asset_query_local_descriptor_signature.png",
        fisher_sig,
    )
    assets["query_global_signature"] = _save_image(out_dir / "asset_query_global_signature.png", global_sig)
    assets["query_fisher_vector_signature"] = _save_image(out_dir / "asset_query_fisher_vector_signature.png", fisher_vec_sig)

    global_entries = _with_correctness(result.get("global_ranked", []), query_class)
    fisher_entries = _with_correctness(result.get("fisher_ranked", []), query_class)
    tier2_entries = _with_correctness(result.get("tier2_ranked", []), query_class)
    tier3_entries = _with_correctness(result.get("tier3_ranked", []), query_class)

    global_strip = _compose_strip(
        query_img=query_processed,
        query_id=query_id,
        query_class=query_class,
        entries=global_entries,
        train_image_paths=train_processed_paths,
        title="Tier-1 Global shortlist",
        detail_fn=(
            # In overview mode we keep text minimal (scores may be on different scales
            # across signals and can distract in slides).
            (lambda e: f"{_correctness_label(e)}")
            if overview_mode
            else (lambda e: f"score={float(e.get('score', 0.0)):.3f}")
        ),
        panel_size=panel_size,
        top_k=top_k,
        overview_mode=overview_mode,
    )
    assets["tier1_global_strip"] = _save_image(out_dir / "asset_tier1_global_strip.png", global_strip)

    fisher_strip = _compose_strip(
        query_img=query_processed,
        query_id=query_id,
        query_class=query_class,
        entries=fisher_entries,
        train_image_paths=train_processed_paths,
        title="Tier-1 Fisher shortlist",
        detail_fn=(
            (lambda e: f"{_correctness_label(e)}")
            if overview_mode
            else (lambda e: f"score={float(e.get('score', 0.0)):.3f}")
        ),
        panel_size=panel_size,
        top_k=top_k,
        overview_mode=overview_mode,
    )
    assets["tier1_fisher_strip"] = _save_image(out_dir / "asset_tier1_fisher_strip.png", fisher_strip)

    # Union strip (presentation-friendly): show Global top-K + Fisher top-K (deduped).
    # This makes the union stage visually match the idea of "K from each signal".
    union_entries_full = _resolve_union_entries(result)
    union_entries_display = _resolve_union_entries_display(result, top_k=top_k)
    union_entries_overview = _with_correctness(union_entries_display, query_class)
    tier2_fusion_entries = []
    union_map = {str(x.get("train_id")): x for x in union_entries_full}
    for item in result.get("tier2_ranked", []):
        tid = str(item.get("train_id"))
        merged = dict(union_map.get(tid, {"train_id": tid, "source": "-", "global_score": None, "fisher_score": None}))
        merged["tier2_score"] = float(item.get("tier2_score", merged.get("tier2_score", 0.0)))
        merged["label"] = item.get("label", merged.get("label"))
        if query_class is None or merged.get("label") is None:
            merged["_is_correct"] = None
        else:
            merged["_is_correct"] = str(merged.get("label")) == str(query_class)
        tier2_fusion_entries.append(merged)

    union_strip = _compose_strip(
        query_img=query_processed,
        query_id=query_id,
        query_class=query_class,
        entries=union_entries_overview,
        train_image_paths=train_processed_paths,
        title="Tier-1 Union shortlist",
        detail_fn=(
            (lambda e: f"{_correctness_label(e)}")
            if overview_mode
            else (
                lambda e: (
                    f"src={e.get('source','-')}  "
                    f"g={_fmt_score(e.get('global_score'))}  "
                    f"f={_fmt_score(e.get('fisher_score'))}  "
                    f"t2={_fmt_score(e.get('tier2_score'))}"
                )
            )
        ),
        panel_size=panel_size,
        top_k=max(0, len(union_entries_overview)),
        overview_mode=overview_mode,
    )
    assets["tier1_union_strip"] = _save_image(out_dir / "asset_tier1_union_strip.png", union_strip)

    fusion_panel = _draw_fusion_breakdown(
        tier2_fusion_entries,
        top_k=top_k,
        overview_mode=overview_mode,
    )
    assets["tier2_fusion_breakdown"] = _save_image(out_dir / "asset_tier2_fusion_breakdown.png", fusion_panel)

    global_db_panel = _draw_vector_database_panel(
        query_id=query_id,
        query_class=query_class,
        query_vector=query_global_embedding,
        ranked_entries=global_entries,
        db_vectors=train_global_embeddings,
        title="Global Embedding DB (Query vs Top Retrieved DB Vectors)",
        top_k=top_k,
        color=(176, 132, 50),
    )
    assets["global_vector_database_panel"] = _save_image(
        out_dir / "asset_global_vector_database_panel.png",
        global_db_panel,
    )

    fisher_db_panel = _draw_vector_database_panel(
        query_id=query_id,
        query_class=query_class,
        query_vector=query_fisher_vector,
        ranked_entries=fisher_entries,
        db_vectors=train_fisher_vectors,
        title="Fisher Vector DB (Query vs Top Retrieved DB Vectors)",
        top_k=top_k,
        color=(70, 170, 230),
    )
    assets["fisher_vector_database_panel"] = _save_image(
        out_dir / "asset_fisher_vector_database_panel.png",
        fisher_db_panel,
    )

    tier2_strip = _compose_strip(
        query_img=query_processed,
        query_id=query_id,
        query_class=query_class,
        entries=tier2_entries,
        train_image_paths=train_processed_paths,
        title="Tier-2 ranking (before GV)",
        detail_fn=(
            (lambda e: f"t2={float(e.get('tier2_score', 0.0)):.3f}  {_correctness_label(e)}")
            if overview_mode
            else (lambda e: f"t2={float(e.get('tier2_score', 0.0)):.3f}")
        ),
        panel_size=panel_size,
        top_k=top_k,
        overview_mode=overview_mode,
    )
    assets["tier2_rerank_strip"] = _save_image(out_dir / "asset_tier2_rerank_strip.png", tier2_strip)

    tier3_strip = _compose_strip(
        query_img=query_processed,
        query_id=query_id,
        query_class=query_class,
        entries=tier3_entries,
        train_image_paths=train_processed_paths,
        title="Tier-3 ranking (after GV)",
        detail_fn=(
            (lambda e: f"inl={int(e.get('n_inliers', 0))}  f={float(e.get('_fused_logit', 0.0)):.2f}")
            if overview_mode
            else (lambda e: f"inl={int(e.get('n_inliers', 0))}  f={float(e.get('_fused_logit', 0.0)):.2f}")
        ),
        panel_size=panel_size,
        top_k=top_k,
        overview_mode=overview_mode,
    )
    assets["tier3_rerank_strip"] = _save_image(out_dir / "asset_tier3_rerank_strip.png", tier3_strip)

    comp = np.full((tier2_strip.shape[0] * 2 + 8, max(tier2_strip.shape[1], tier3_strip.shape[1]), 3), 255, dtype=np.uint8)
    comp[: tier2_strip.shape[0], : tier2_strip.shape[1]] = tier2_strip
    comp[tier2_strip.shape[0] + 8 : tier2_strip.shape[0] + 8 + tier3_strip.shape[0], : tier3_strip.shape[1]] = tier3_strip
    assets["tier2_vs_tier3_comparison"] = _save_image(out_dir / "asset_tier2_vs_tier3_comparison.png", comp)

    predicted_entry = _select_predicted_entry(result)
    predicted_image_id = None
    predicted_image_class = None
    predicted_source_rank = None
    predicted_image = None
    if predicted_entry is not None:
        predicted_image_id = str(predicted_entry.get("train_id"))
        predicted_label_raw = predicted_entry.get("label", result.get("predicted_class"))
        predicted_image_class = None if predicted_label_raw is None else str(predicted_label_raw)
        predicted_source_rank = predicted_entry.get("_source_rank")
        predicted_image = _read_bgr(train_processed_paths.get(predicted_image_id))
    predicted_tile = _draw_predicted_image_tile(
        image=predicted_image,
        panel_size=panel_size,
        train_id=predicted_image_id,
        class_id=predicted_image_class,
        source_rank=predicted_source_rank,
        overview_mode=overview_mode,
    )
    assets["predicted_image"] = _save_image(out_dir / "asset_predicted_image.png", predicted_tile)

    gv_panel = None
    best = result.get("tier3_ranked", [])
    if best:
        best_id = str(best[0].get("train_id"))
        cand_path = train_processed_paths.get(best_id)
        cand_img = _read_bgr(cand_path)
        cand_desc = train_descriptors.get(best_id)
        cand_kp = train_keypoints.get(best_id)
        if cand_img is not None and cand_desc is not None and cand_kp is not None:
            query_img_path = image_paths.get(query_id) if image_paths else query_processed_path
            cand_img_path = image_paths.get(best_id) if image_paths else cand_path
            matched_q, matched_d, inlier_mask = _compute_matches(
                query_descriptors,
                query_keypoints,
                cand_desc,
                cand_kp,
                matcher=gv_matcher,
                method=gv_features,
                query_image_path=query_img_path,
                candidate_image_path=cand_img_path,
            )
            gv_panel = _draw_match_panel(
                query_processed,
                cand_img,
                matched_q,
                matched_d,
                inlier_mask,
                side=panel_size,
                overview_mode=overview_mode,
            )
            assets["gv_best_candidate_matches"] = _save_image(out_dir / "asset_gv_best_candidate_matches.png", gv_panel)

    manifest = {
        "query_id": str(query_id),
        "query_class": None if query_class is None else str(query_class),
        "predicted_class": result.get("predicted_class"),
        "predicted_image_id": predicted_image_id,
        "predicted_image_class": predicted_image_class,
        "predicted_image_source_rank": predicted_source_rank,
        "top_n": result.get("top_n", []),
        "gv_matcher": gv_matcher,
        "gv_features": gv_features,
        "top_k_assets": int(top_k),
        "panel_size": int(panel_size),
        "assets": assets,
        "fusion_view_entries": tier2_fusion_entries[: max(0, int(top_k))],
        "rankings": {
            "global": result.get("global_ranked", []),
            "fisher": result.get("fisher_ranked", []),
            "union_ids": result.get("union_ids", []),
            "tier2": result.get("tier2_ranked", []),
            "tier3": result.get("tier3_ranked", []),
        },
    }

    manifest_path = out_dir / "assets_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return manifest


if __name__ == "__main__":
    raise SystemExit(
        "This module is intended to be used from main.py single-query mode. "
        "Run main.py with --visualize_query_pipeline."
    )
