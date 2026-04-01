from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


def _require_cv2() -> None:
    if cv2 is None:
        raise SystemExit(
            "Missing dependency 'cv2'. Run with the project venv:\n"
            "  ./venv/bin/python utils/make_tier1_assets.py --input_dir <ASSETS_DIR>\n"
            "and ensure requirements are installed:\n"
            "  ./venv/bin/pip install -r requirements.txt"
        )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _resolve_existing_path(path_str: str, *, query_dir: Path) -> Path:
    p = Path(path_str)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(p)
        candidates.append(Path.cwd() / p)
        candidates.append(query_dir / p)
        candidates.append(query_dir / p.name)
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not resolve path from '{path_str}' (query_dir={query_dir})")


def _read_bgr(path: Path) -> np.ndarray:
    _require_cv2()
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def _resize_contain(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    _require_cv2()
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return img
    scale = min(max_w / float(w), max_h / float(h))
    scale = max(scale, 1e-6)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def _draw_text_box(
    canvas: np.ndarray,
    *,
    x: int,
    y: int,
    lines: list[str],
    font_scale: float,
    fg: tuple[int, int, int] = (25, 25, 25),
    bg: tuple[int, int, int] = (248, 248, 248),
    pad: int = 14,
    line_gap: int = 10,
    thickness: int = 2,
) -> None:
    _require_cv2()
    font = cv2.FONT_HERSHEY_SIMPLEX
    sizes = [cv2.getTextSize(t, font, font_scale, thickness)[0] for t in lines]
    w = max((s[0] for s in sizes), default=0)
    h_line = max((s[1] for s in sizes), default=0)
    h = len(lines) * h_line + max(0, len(lines) - 1) * line_gap
    x0 = int(x)
    y0 = int(y)
    x1 = min(canvas.shape[1] - 1, x0 + w + 2 * pad)
    y1 = min(canvas.shape[0] - 1, y0 + h + 2 * pad)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), bg, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (210, 210, 210), 2, cv2.LINE_AA)

    cy = y0 + pad + h_line
    for t in lines:
        cv2.putText(canvas, t, (x0 + pad, cy), font, font_scale, fg, thickness, cv2.LINE_AA)
        cy += h_line + line_gap


def _paste(canvas: np.ndarray, img: np.ndarray, x: int, y: int) -> None:
    h, w = img.shape[:2]
    x0 = int(x)
    y0 = int(y)
    x1 = min(canvas.shape[1], x0 + w)
    y1 = min(canvas.shape[0], y0 + h)
    if x0 >= x1 or y0 >= y1:
        return
    canvas[y0:y1, x0:x1] = img[: y1 - y0, : x1 - x0]


def _make_vertical_gradient(
    h: int,
    w: int,
    *,
    top_bgr: tuple[int, int, int] = (255, 255, 255),
    bottom_bgr: tuple[int, int, int] = (242, 246, 250),
) -> np.ndarray:
    """Create a subtle slide-like background (BGR)."""
    top = np.array(top_bgr, dtype=np.float32)[None, None, :]
    bot = np.array(bottom_bgr, dtype=np.float32)[None, None, :]
    t = np.linspace(0.0, 1.0, max(1, h), dtype=np.float32)[:, None, None]
    bg = (1.0 - t) * top + t * bot
    bg = np.clip(bg, 0.0, 255.0).astype(np.uint8)
    return np.repeat(bg, max(1, w), axis=1)


def _draw_rounded_rect(
    canvas: np.ndarray,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    radius: int,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw a rounded rectangle using basic OpenCV primitives."""
    _require_cv2()
    if radius <= 0:
        cv2.rectangle(canvas, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness, cv2.LINE_AA)
        return

    x0 = int(x)
    y0 = int(y)
    x1 = int(x + w)
    y1 = int(y + h)
    r = int(max(1, radius))
    r = int(min(r, max(1, (x1 - x0) // 2), max(1, (y1 - y0) // 2)))

    if thickness < 0:
        cv2.rectangle(canvas, (x0 + r, y0), (x1 - r, y1), color, -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x0, y0 + r), (x1, y1 - r), color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x0 + r, y0 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x1 - r, y0 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x0 + r, y1 - r), r, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x1 - r, y1 - r), r, color, -1, cv2.LINE_AA)
        return

    cv2.line(canvas, (x0 + r, y0), (x1 - r, y0), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x0 + r, y1), (x1 - r, y1), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x0, y0 + r), (x0, y1 - r), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x1, y0 + r), (x1, y1 - r), color, thickness, cv2.LINE_AA)

    cv2.ellipse(canvas, (x0 + r, y0 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x1 - r, y0 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x1 - r, y1 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x0 + r, y1 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)


def _draw_shadowed_rect(
    canvas: np.ndarray,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    fill: tuple[int, int, int] = (255, 255, 255),
    border: tuple[int, int, int] = (220, 220, 220),
    border_thickness: int = 2,
    shadow_dx: int = 10,
    shadow_dy: int = 12,
    shadow_alpha: float = 0.10,
    radius: int = 28,
) -> None:
    """Draw a simple card with a soft-ish shadow (no alpha channel required)."""
    _require_cv2()
    x0 = int(x)
    y0 = int(y)
    x1 = int(min(canvas.shape[1] - 1, x0 + int(w)))
    y1 = int(min(canvas.shape[0] - 1, y0 + int(h)))
    if x0 >= x1 or y0 >= y1:
        return

    if shadow_alpha > 0:
        overlay = canvas.copy()
        sx0 = min(canvas.shape[1] - 1, max(0, x0 + int(shadow_dx)))
        sy0 = min(canvas.shape[0] - 1, max(0, y0 + int(shadow_dy)))
        sx1 = min(canvas.shape[1] - 1, max(0, x1 + int(shadow_dx)))
        sy1 = min(canvas.shape[0] - 1, max(0, y1 + int(shadow_dy)))
        if sx0 < sx1 and sy0 < sy1:
            _draw_rounded_rect(
                overlay,
                x=int(sx0),
                y=int(sy0),
                w=int(sx1 - sx0),
                h=int(sy1 - sy0),
                radius=int(radius),
                color=(0, 0, 0),
                thickness=-1,
            )
            canvas[:] = cv2.addWeighted(overlay, float(shadow_alpha), canvas, 1.0 - float(shadow_alpha), 0.0)

    _draw_rounded_rect(
        canvas,
        x=int(x0),
        y=int(y0),
        w=int(x1 - x0),
        h=int(y1 - y0),
        radius=int(radius),
        color=fill,
        thickness=-1,
    )
    _draw_rounded_rect(
        canvas,
        x=int(x0),
        y=int(y0),
        w=int(x1 - x0),
        h=int(y1 - y0),
        radius=int(radius),
        color=border,
        thickness=int(border_thickness),
    )


def _generate_union_flow(
    *,
    query_original: np.ndarray | None,
    query_segmented: np.ndarray | None,
    query_keypoints: np.ndarray | None,
    global_strip: np.ndarray,
    fisher_strip: np.ndarray,
    union_strip: np.ndarray,
    out_path: Path,
    title: str,
    canvas_w: int,
    canvas_h: int,
    margin: int,
) -> None:
    _require_cv2()
    canvas = _make_vertical_gradient(int(canvas_h), int(canvas_w))

    title_scale = 1.15 if canvas_w >= 3000 else 0.95
    _draw_text_box(
        canvas,
        x=margin,
        y=margin // 2,
        lines=[title],
        font_scale=title_scale,
        fg=(20, 20, 20),
        bg=(245, 247, 250),
        pad=18,
        thickness=3,
    )

    y_top = int(margin * 1.8)
    avail_h = canvas_h - y_top - margin
    # Layout: three rows.
    steps_area_h = int(avail_h * 0.24)
    shortlist_area_h = int(avail_h * 0.30)
    merge_area_h = int(avail_h * 0.10)
    union_area_h = avail_h - steps_area_h - shortlist_area_h - merge_area_h

    # Card geometry.
    content_w = canvas_w - 2 * margin
    cards_gap_x = max(70, int(round(margin * 0.85)))
    card_w = int((content_w - cards_gap_x) // 2)
    card_h = int(shortlist_area_h)
    g_card_x = int(margin)
    f_card_x = int(margin + card_w + cards_gap_x)
    cards_y = int(y_top + steps_area_h)

    # Preprocessing tiles (top row).
    # Place them above the two Tier-1 cards so arrows read naturally:
    # Input (left) -> Background removal (center) -> Keypoints (right).
    tile_gap = max(40, int(round(margin * 0.55)))
    tile_size = int(min(380, max(180, int(round(steps_area_h * 0.80)))))
    g_center_x = g_card_x + card_w // 2
    f_center_x = f_card_x + card_w // 2
    mid_center_x = int((g_center_x + f_center_x) // 2)
    in_center_x = int(g_center_x)
    bg_center_x = int(mid_center_x)
    kp_center_x = int(f_center_x)

    # Keep tiles fully on-canvas.
    min_cx = margin + tile_size // 2
    max_cx = canvas_w - margin - tile_size // 2
    in_center_x = int(np.clip(in_center_x, min_cx, max_cx))
    bg_center_x = int(np.clip(bg_center_x, min_cx, max_cx))
    kp_center_x = int(np.clip(kp_center_x, min_cx, max_cx))

    tiles_y = int(y_top + max(0, (steps_area_h - tile_size) // 2))

    def _make_step_tile(img: np.ndarray | None, label: str) -> np.ndarray:
        tile = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
        if img is not None:
            fit = _resize_contain(img, tile_size, tile_size)
            _paste(tile, fit, (tile_size - fit.shape[1]) // 2, (tile_size - fit.shape[0]) // 2)
        cv2.rectangle(tile, (0, 0), (tile_size - 1, tile_size - 1), (210, 210, 210), 2, cv2.LINE_AA)
        # Minimal label banner for each step.
        banner_h = max(34, int(round(tile_size * 0.10)))
        cv2.rectangle(tile, (0, 0), (tile_size, banner_h), (18, 18, 18), -1)
        cv2.putText(
            tile,
            label,
            (10, int(round(banner_h * 0.72))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75 if tile_size >= 260 else 0.65,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
        return tile

    arrow_color = (60, 60, 60)
    thickness = 5 if canvas_w >= 3000 else 3
    tip = 0.018

    # Accent colors (BGR) for a cleaner, presentation-friendly look.
    # Global: blue, Fisher: orange, Union: teal.
    c_global = (223, 108, 45)  # RGB #2D6CDF
    c_fisher = (97, 162, 244)  # RGB #F4A261
    c_union = (143, 157, 42)   # RGB #2A9D8F

    # Draw cards for the two Tier-1 shortlists.
    _draw_shadowed_rect(
        canvas,
        x=g_card_x,
        y=cards_y,
        w=card_w,
        h=card_h,
        fill=(255, 255, 255),
        border=(222, 222, 222),
        border_thickness=2,
        shadow_dx=10,
        shadow_dy=12,
        shadow_alpha=0.10,
        radius=32,
    )
    _draw_shadowed_rect(
        canvas,
        x=f_card_x,
        y=cards_y,
        w=card_w,
        h=card_h,
        fill=(255, 255, 255),
        border=(222, 222, 222),
        border_thickness=2,
        shadow_dx=10,
        shadow_dy=12,
        shadow_alpha=0.10,
        radius=32,
    )

    # Card insets: reserve a dedicated header region for the accent bar so strips never
    # overlap it (presentation-friendly).
    inner_pad_lr = max(26, int(round(margin * 0.40)))
    inner_pad_bottom = inner_pad_lr
    bar_inset = max(16, int(round(margin * 0.20)))
    bar_h = max(10, int(round(card_h * 0.035)))
    header_gap = max(18, int(round(margin * 0.22)))
    inner_pad_top = inner_pad_lr + bar_inset + bar_h + header_gap

    inner_w = max(1, card_w - 2 * inner_pad_lr)
    inner_h = max(1, card_h - inner_pad_top - inner_pad_bottom)

    global_fit = _resize_contain(global_strip, inner_w, inner_h)
    fisher_fit = _resize_contain(fisher_strip, inner_w, inner_h)
    g_h, g_w = global_fit.shape[:2]
    f_h, f_w = fisher_fit.shape[:2]
    g_x = int(g_card_x + inner_pad_lr + (inner_w - g_w) // 2)
    f_x = int(f_card_x + inner_pad_lr + (inner_w - f_w) // 2)
    g_y = int(cards_y + inner_pad_top + (inner_h - g_h) // 2)
    f_y = int(cards_y + inner_pad_top + (inner_h - f_h) // 2)
    _paste(canvas, global_fit, g_x, g_y)
    _paste(canvas, fisher_fit, f_x, f_y)

    # Accent bars (drawn after paste so they stay crisp on top).
    cv2.rectangle(
        canvas,
        (int(g_card_x + bar_inset), int(cards_y + bar_inset)),
        (int(g_card_x + card_w - bar_inset), int(cards_y + bar_inset + bar_h)),
        c_global,
        -1,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        canvas,
        (int(f_card_x + bar_inset), int(cards_y + bar_inset)),
        (int(f_card_x + card_w - bar_inset), int(cards_y + bar_inset + bar_h)),
        c_fisher,
        -1,
        cv2.LINE_AA,
    )

    # Union card (bottom row).
    union_y = int(y_top + steps_area_h + shortlist_area_h + merge_area_h)
    union_w = int(canvas_w - 2 * margin)
    union_h = int(union_area_h)
    _draw_shadowed_rect(
        canvas,
        x=margin,
        y=union_y,
        w=union_w,
        h=union_h,
        fill=(255, 255, 255),
        border=(222, 222, 222),
        border_thickness=2,
        shadow_dx=10,
        shadow_dy=12,
        shadow_alpha=0.10,
        radius=32,
    )
    union_inner_pad_lr = inner_pad_lr
    union_inner_pad_bottom = inner_pad_bottom
    union_inner_pad_top = inner_pad_lr + bar_inset + bar_h + header_gap
    union_fit = _resize_contain(
        union_strip,
        union_w - 2 * union_inner_pad_lr,
        union_h - union_inner_pad_top - union_inner_pad_bottom,
    )
    u_h, u_w = union_fit.shape[:2]
    u_x = int(margin + union_inner_pad_lr + ((union_w - 2 * union_inner_pad_lr) - u_w) // 2)
    u_y = int(union_y + union_inner_pad_top + ((union_h - union_inner_pad_top - union_inner_pad_bottom) - u_h) // 2)
    _paste(canvas, union_fit, u_x, u_y)
    cv2.rectangle(
        canvas,
        (int(margin + bar_inset), int(union_y + bar_inset)),
        (int(margin + union_w - bar_inset), int(union_y + bar_inset + bar_h)),
        c_union,
        -1,
        cv2.LINE_AA,
    )

    # Preprocessing tiles.
    input_tile = _make_step_tile(query_original, "Input image")
    seg_tile = _make_step_tile(query_segmented, "Background removal")
    kp_tile = _make_step_tile(query_keypoints, "Keypoints")

    in_x = int(in_center_x - tile_size // 2)
    bg_x = int(bg_center_x - tile_size // 2)
    kp_x = int(kp_center_x - tile_size // 2)
    _paste(canvas, input_tile, in_x, tiles_y)
    _paste(canvas, seg_tile, bg_x, tiles_y)
    _paste(canvas, kp_tile, kp_x, tiles_y)

    # Arrows between preprocessing steps (left-to-right).
    in_mid_r = (in_x + tile_size, tiles_y + tile_size // 2)
    bg_mid_l = (bg_x, tiles_y + tile_size // 2)
    bg_mid_r = (bg_x + tile_size, tiles_y + tile_size // 2)
    kp_mid_l = (kp_x, tiles_y + tile_size // 2)
    cv2.arrowedLine(canvas, in_mid_r, bg_mid_l, arrow_color, thickness, cv2.LINE_AA, tipLength=tip)
    cv2.arrowedLine(canvas, bg_mid_r, kp_mid_l, arrow_color, thickness, cv2.LINE_AA, tipLength=tip)

    # Drop arrows from preprocessing to Tier-1 cards (clean, non-crossing).
    bg_bot = (bg_x + tile_size // 2, tiles_y + tile_size)
    kp_bot = (kp_x + tile_size // 2, tiles_y + tile_size)
    g_top = (g_card_x + card_w // 2, cards_y)
    f_top = (f_card_x + card_w // 2, cards_y)
    elbow_y = int(cards_y - max(26, int(round(margin * 0.22))))
    # Background removal -> Global (elbow connector).
    cv2.line(canvas, bg_bot, (bg_bot[0], elbow_y), c_global, thickness, cv2.LINE_AA)
    cv2.line(canvas, (bg_bot[0], elbow_y), (g_top[0], elbow_y), c_global, thickness, cv2.LINE_AA)
    cv2.arrowedLine(canvas, (g_top[0], elbow_y), g_top, c_global, thickness, cv2.LINE_AA, tipLength=tip)
    # Keypoints -> Fisher.
    cv2.arrowedLine(canvas, kp_bot, f_top, c_fisher, thickness, cv2.LINE_AA, tipLength=tip)

    # Merge arrows (two inputs -> union).
    g_mid = (g_card_x + card_w // 2, cards_y + card_h)
    f_mid = (f_card_x + card_w // 2, cards_y + card_h)
    merge_pt = (int((g_mid[0] + f_mid[0]) // 2), int(cards_y + card_h + merge_area_h // 2))
    u_top = (margin + union_w // 2, union_y)

    cv2.line(canvas, g_mid, merge_pt, c_global, thickness, cv2.LINE_AA)
    cv2.line(canvas, f_mid, merge_pt, c_fisher, thickness, cv2.LINE_AA)
    # After merging, continue in Union color.
    cv2.arrowedLine(canvas, merge_pt, u_top, c_union, thickness, cv2.LINE_AA, tipLength=tip)

    # Small merge node for readability.
    cv2.circle(canvas, merge_pt, 9 if canvas_w >= 3000 else 6, (60, 60, 60), -1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), canvas)
    if not ok:
        raise RuntimeError(f"Failed to write image: {out_path}")


def _generate_union_venn(
    *,
    out_path: Path,
    title: str,
    global_n: int,
    fisher_n: int,
    inter_n: int,
    union_n: int,
    canvas_w: int,
    canvas_h: int,
) -> None:
    _require_cv2()
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    _draw_text_box(
        canvas,
        x=60,
        y=40,
        lines=[title],
        font_scale=1.0,
        fg=(20, 20, 20),
        bg=(245, 247, 250),
        pad=16,
        thickness=3,
    )

    c1 = (canvas_w // 2 - 240, canvas_h // 2 + 30)
    c2 = (canvas_w // 2 + 240, canvas_h // 2 + 30)
    radius = 320

    overlay = canvas.copy()
    cv2.circle(overlay, c1, radius, (70, 170, 230), -1, cv2.LINE_AA)   # Fisher-ish blue
    cv2.circle(overlay, c2, radius, (176, 132, 50), -1, cv2.LINE_AA)   # Global-ish gold
    canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0.0)
    cv2.circle(canvas, c1, radius, (70, 170, 230), 6, cv2.LINE_AA)
    cv2.circle(canvas, c2, radius, (176, 132, 50), 6, cv2.LINE_AA)

    left_only = max(0, int(global_n) - int(inter_n))
    right_only = max(0, int(fisher_n) - int(inter_n))

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Global shortlist", (c2[0] - 250, c2[1] - radius - 30), font, 1.0, (30, 30, 30), 3, cv2.LINE_AA)
    cv2.putText(canvas, "Fisher shortlist", (c1[0] - 250, c1[1] - radius - 30), font, 1.0, (30, 30, 30), 3, cv2.LINE_AA)

    cv2.putText(canvas, f"{left_only}", (c2[0] + 90, c2[1] + 10), font, 1.6, (20, 20, 20), 4, cv2.LINE_AA)
    cv2.putText(canvas, f"{right_only}", (c1[0] - 140, c1[1] + 10), font, 1.6, (20, 20, 20), 4, cv2.LINE_AA)
    cv2.putText(canvas, f"{inter_n}", (canvas_w // 2 - 45, c1[1] + 10), font, 1.6, (20, 20, 20), 4, cv2.LINE_AA)

    _draw_text_box(
        canvas,
        x=60,
        y=canvas_h - 180,
        lines=[
            "Tier-1 union shortlist = ordered union of Global top-K + Fisher top-K (deduped)",
        ],
        font_scale=0.75,
        fg=(35, 35, 35),
        bg=(245, 247, 250),
        pad=16,
        thickness=2,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), canvas)
    if not ok:
        raise RuntimeError(f"Failed to write image: {out_path}")


def _compute_tier1_sets(manifest: dict) -> tuple[list[str], list[str], list[str], int]:
    rankings = manifest.get("rankings", {}) or {}
    global_ranked = rankings.get("global", []) or []
    fisher_ranked = rankings.get("fisher", []) or []
    top_k = int(manifest.get("top_k_assets", 8) or 0)

    # For presentation, interpret "Tier-1 shortlist" as the top-K shown on the slide,
    # not the full internal retrieval list (which can be much larger).
    if top_k > 0:
        global_ranked = global_ranked[:top_k]
        fisher_ranked = fisher_ranked[:top_k]

    global_ids = [str(x.get("train_id")) for x in global_ranked if x.get("train_id") is not None]
    fisher_ids = [str(x.get("train_id")) for x in fisher_ranked if x.get("train_id") is not None]
    union_ids = list(dict.fromkeys(global_ids + fisher_ids))
    return global_ids, fisher_ids, union_ids, top_k


def _process_query_dir(query_dir: Path, *, overwrite: bool, canvas_w: int, canvas_h: int, venn_w: int, venn_h: int, margin: int) -> None:
    manifest_path = query_dir / "assets_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = _read_json(manifest_path)
    assets = manifest.get("assets", {}) or {}

    def get_asset_path(key: str, fallback_name: str) -> Path:
        if key in assets and isinstance(assets[key], str) and assets[key].strip():
            return _resolve_existing_path(assets[key], query_dir=query_dir)
        cand = query_dir / fallback_name
        if cand.exists():
            return cand
        raise FileNotFoundError(f"Missing required asset '{key}' and fallback '{cand}'.")

    global_strip_path = get_asset_path("tier1_global_strip", "asset_tier1_global_strip.png")
    fisher_strip_path = get_asset_path("tier1_fisher_strip", "asset_tier1_fisher_strip.png")
    union_strip_path = get_asset_path("tier1_union_strip", "asset_tier1_union_strip.png")

    def get_optional_asset_path(key: str, fallback_name: str) -> Path | None:
        try:
            return get_asset_path(key, fallback_name)
        except Exception:
            return None

    query_original_path = get_optional_asset_path("query_original", "asset_query_original.png")
    query_segmented_path = get_optional_asset_path("query_segmented", "asset_query_segmented.png")
    query_keypoints_path = get_optional_asset_path("query_local_keypoints", "asset_query_local_keypoints.png")

    global_strip = _read_bgr(global_strip_path)
    fisher_strip = _read_bgr(fisher_strip_path)
    union_strip = _read_bgr(union_strip_path)
    query_original = _read_bgr(query_original_path) if query_original_path else None
    query_segmented = _read_bgr(query_segmented_path) if query_segmented_path else None
    query_keypoints = _read_bgr(query_keypoints_path) if query_keypoints_path else None

    global_ids, fisher_ids, union_ids, top_k = _compute_tier1_sets(manifest)
    g_set = set(global_ids)
    f_set = set(fisher_ids)
    i_set = g_set & f_set
    u_set = set(union_ids) if union_ids else (g_set | f_set)

    out_flow = query_dir / "asset_tier1_union_flow.png"
    out_venn = query_dir / "asset_tier1_union_venn.png"
    if (not overwrite) and (out_flow.exists() or out_venn.exists()):
        return

    _generate_union_flow(
        query_original=query_original,
        query_segmented=query_segmented,
        query_keypoints=query_keypoints,
        global_strip=global_strip,
        fisher_strip=fisher_strip,
        union_strip=union_strip,
        out_path=out_flow,
        title="Tier-1 Candidate Retrieval: Global + Fisher -> Union",
        canvas_w=int(canvas_w),
        canvas_h=int(canvas_h),
        margin=int(margin),
    )

    _generate_union_venn(
        out_path=out_venn,
        title=f"Tier-1 Union Overlap (K={max(0, int(top_k))})" if top_k else "Tier-1 Union Overlap",
        global_n=len(g_set),
        fisher_n=len(f_set),
        inter_n=len(i_set),
        union_n=len(u_set),
        canvas_w=int(venn_w),
        canvas_h=int(venn_h),
    )

    assets = manifest.setdefault("assets", {})
    assets["tier1_union_flow"] = str(out_flow)
    assets["tier1_union_venn"] = str(out_venn)
    manifest["tier1_union_counts"] = {
        "global": int(len(g_set)),
        "fisher": int(len(f_set)),
        "intersection": int(len(i_set)),
        "union": int(len(u_set)),
    }
    _write_json(manifest_path, manifest)


def _iter_query_dirs(input_dir: Path) -> list[Path]:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        return []
    if (input_dir / "assets_manifest.json").exists():
        return [input_dir]

    batch_manifest_path = input_dir / "batch_manifest.json"
    if batch_manifest_path.exists():
        batch = _read_json(batch_manifest_path)
        out = []
        for q in batch.get("queries", []) or []:
            p = q.get("output_dir")
            if not p:
                continue
            out.append(Path(p))
        return out

    # Fallback: scan direct children.
    out = []
    for child in input_dir.iterdir():
        if child.is_dir() and (child / "assets_manifest.json").exists():
            out.append(child)
    return out


def main() -> None:
    _require_cv2()
    parser = argparse.ArgumentParser(description="Generate Tier-1 union flow + overlap visuals from existing pipeline assets.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("docs") / "Final Thesis" / "Figures" / "pipeline_assets",
        help="Either a single query folder (contains assets_manifest.json) or an assets root containing per-query subfolders.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated Tier-1 visuals.")
    parser.add_argument("--canvas_w", type=int, default=3840, help="Width for union flow image (16:9 recommended).")
    parser.add_argument("--canvas_h", type=int, default=2160, help="Height for union flow image (16:9 recommended).")
    parser.add_argument("--venn_w", type=int, default=1920, help="Width for Venn overlap image.")
    parser.add_argument("--venn_h", type=int, default=1080, help="Height for Venn overlap image.")
    parser.add_argument("--margin", type=int, default=80, help="Canvas margin (pixels).")
    args = parser.parse_args()

    query_dirs = _iter_query_dirs(args.input_dir)
    if not query_dirs:
        raise SystemExit(f"No query folders found under: {args.input_dir}")

    processed = 0
    for qd in query_dirs:
        try:
            _process_query_dir(
                Path(qd),
                overwrite=bool(args.overwrite),
                canvas_w=int(args.canvas_w),
                canvas_h=int(args.canvas_h),
                venn_w=int(args.venn_w),
                venn_h=int(args.venn_h),
                margin=int(args.margin),
            )
            processed += 1
        except Exception as e:
            print(f"[WARN] Failed for {qd}: {e}")

    print(f"Processed {processed} query folders.")


if __name__ == "__main__":
    main()
