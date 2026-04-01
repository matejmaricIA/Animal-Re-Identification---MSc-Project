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
            "  ./venv/bin/python utils/make_tier3_assets.py --input_dir <ASSETS_DIR>\n"
            "and ensure requirements are installed:\n"
            "  ./venv/bin/pip install -r requirements.txt"
        )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _resolve_existing_path(path_str: str, *, query_dir: Path) -> Path:
    p = Path(path_str)
    candidates: list[Path] = []
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


def _paste(canvas: np.ndarray, img: np.ndarray, x: int, y: int) -> None:
    h, w = img.shape[:2]
    x0 = int(x)
    y0 = int(y)
    x1 = min(canvas.shape[1], x0 + w)
    y1 = min(canvas.shape[0], y0 + h)
    if x0 >= x1 or y0 >= y1:
        return
    canvas[y0:y1, x0:x1] = img[: y1 - y0, : x1 - x0]


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


def _make_vertical_gradient(
    h: int,
    w: int,
    *,
    top_bgr: tuple[int, int, int] = (255, 255, 255),
    bottom_bgr: tuple[int, int, int] = (242, 246, 250),
) -> np.ndarray:
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


def _draw_accent_bar(
    canvas: np.ndarray,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    color: tuple[int, int, int],
) -> None:
    _require_cv2()
    if w <= 0 or h <= 0:
        return
    cv2.rectangle(canvas, (int(x), int(y)), (int(x + w), int(y + h)), color, -1, cv2.LINE_AA)


def _generate_tier3_gv_flow(
    *,
    tier2_strip: np.ndarray,
    gv_matches: np.ndarray,
    tier3_strip: np.ndarray,
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

    # Compact "cards" that hug the actual image content (instead of spanning the full slide).
    # This reads better in slides and keeps the flow clean.
    y_top = int(margin * 1.55)
    avail_w = int(canvas_w - 2 * margin)
    avail_h = int(canvas_h - y_top - margin)

    is_large = bool(canvas_w >= 3000)
    pad_lr = 26 if is_large else 20
    bar_inset = 14 if is_large else 12
    bar_h = 10 if is_large else 8
    header_gap = 14 if is_large else 12
    gap_h = max(56, int(round(margin * (0.70 if is_large else 0.60))))
    inner_pad_top = pad_lr + bar_inset + bar_h + header_gap
    inner_pad_bottom = pad_lr
    overhead_h = inner_pad_top + inner_pad_bottom

    def _resize_by_scale(img: np.ndarray, scale: float) -> np.ndarray:
        _require_cv2()
        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            return img
        scale = float(max(scale, 1e-6))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        if new_w == w and new_h == h:
            return img
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(img, (new_w, new_h), interpolation=interp)

    # Compute a single scale factor so all three panels fit vertically on a 16:9 canvas,
    # while also respecting the available width.
    img_h_sum = float(tier2_strip.shape[0] + gv_matches.shape[0] + tier3_strip.shape[0])
    overhead_total = float(3 * overhead_h + 2 * gap_h)
    max_img_w = float(max(tier2_strip.shape[1], gv_matches.shape[1], tier3_strip.shape[1]))

    height_room = max(1.0, float(avail_h) - overhead_total)
    width_room = max(1.0, float(avail_w) - 2.0 * float(pad_lr))
    s_h = height_room / max(1.0, img_h_sum)
    s_w = width_room / max(1.0, max_img_w)
    scale = float(min(s_h, s_w))
    # Nudge down slightly so rounding doesn't overflow.
    scale = float(max(0.05, scale * 0.995))

    tier2_fit = _resize_by_scale(tier2_strip, scale)
    gv_fit = _resize_by_scale(gv_matches, scale)
    tier3_fit = _resize_by_scale(tier3_strip, scale)

    def _card_dims(img: np.ndarray) -> tuple[int, int]:
        h, w = img.shape[:2]
        return int(w + 2 * pad_lr), int(h + overhead_h)

    card1_w, card1_h = _card_dims(tier2_fit)
    card2_w, card2_h = _card_dims(gv_fit)
    card3_w, card3_h = _card_dims(tier3_fit)

    total_h = int(card1_h + gap_h + card2_h + gap_h + card3_h)
    # Center the flow vertically within the available region.
    top_y = int(y_top + max(0, (avail_h - total_h) // 2))
    gv_y = int(top_y + card1_h + gap_h)
    bot_y = int(gv_y + card2_h + gap_h)

    x_mid = int(canvas_w // 2)
    card1_x = int(x_mid - card1_w // 2)
    card2_x = int(x_mid - card2_w // 2)
    card3_x = int(x_mid - card3_w // 2)

    # Accent colors (BGR).
    c_fused = (89, 169, 76)   # fused green used elsewhere
    c_gv = (194, 87, 126)     # purple-ish for GV stage (RGB #7E57C2)

    # Cards.
    _draw_shadowed_rect(
        canvas,
        x=card1_x,
        y=top_y,
        w=card1_w,
        h=card1_h,
        fill=(255, 255, 255),
        border=(222, 222, 222),
        border_thickness=2,
        shadow_dx=8,
        shadow_dy=10,
        shadow_alpha=0.08,
        radius=28,
    )
    _draw_shadowed_rect(
        canvas,
        x=card2_x,
        y=gv_y,
        w=card2_w,
        h=card2_h,
        fill=(255, 255, 255),
        border=(222, 222, 222),
        border_thickness=2,
        shadow_dx=8,
        shadow_dy=10,
        shadow_alpha=0.08,
        radius=28,
    )
    _draw_shadowed_rect(
        canvas,
        x=card3_x,
        y=bot_y,
        w=card3_w,
        h=card3_h,
        fill=(255, 255, 255),
        border=(222, 222, 222),
        border_thickness=2,
        shadow_dx=8,
        shadow_dy=10,
        shadow_alpha=0.08,
        radius=28,
    )

    # Paste images flush into their "hugging" cards.
    _paste(canvas, tier2_fit, int(card1_x + pad_lr), int(top_y + inner_pad_top))
    _paste(canvas, gv_fit, int(card2_x + pad_lr), int(gv_y + inner_pad_top))
    _paste(canvas, tier3_fit, int(card3_x + pad_lr), int(bot_y + inner_pad_top))

    # Accent bars.
    _draw_accent_bar(
        canvas,
        x=int(card1_x + bar_inset),
        y=int(top_y + bar_inset),
        w=int(card1_w - 2 * bar_inset),
        h=int(bar_h),
        color=c_fused,
    )
    _draw_accent_bar(
        canvas,
        x=int(card2_x + bar_inset),
        y=int(gv_y + bar_inset),
        w=int(card2_w - 2 * bar_inset),
        h=int(bar_h),
        color=c_gv,
    )
    _draw_accent_bar(
        canvas,
        x=int(card3_x + bar_inset),
        y=int(bot_y + bar_inset),
        w=int(card3_w - 2 * bar_inset),
        h=int(bar_h),
        color=c_fused,
    )

    # Connectors (Tier-2 -> GV -> Tier-3).
    arrow_thickness = 6 if canvas_w >= 3000 else 4
    tip = 0.018

    p1 = (x_mid, int(top_y + card1_h))
    p2 = (x_mid, int(gv_y))
    p3 = (x_mid, int(gv_y + card2_h))
    p4 = (x_mid, int(bot_y))

    cv2.arrowedLine(canvas, p1, p2, c_fused, arrow_thickness, cv2.LINE_AA, tipLength=tip)
    cv2.arrowedLine(canvas, p3, p4, c_gv, arrow_thickness, cv2.LINE_AA, tipLength=tip)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), canvas)
    if not ok:
        raise RuntimeError(f"Failed to write image: {out_path}")


def _process_query_dir(query_dir: Path, *, overwrite: bool, canvas_w: int, canvas_h: int, margin: int) -> None:
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

    tier2_strip_path = get_asset_path("tier2_rerank_strip", "asset_tier2_rerank_strip.png")
    gv_matches_path = get_asset_path("gv_best_candidate_matches", "asset_gv_best_candidate_matches.png")
    tier3_strip_path = get_asset_path("tier3_rerank_strip", "asset_tier3_rerank_strip.png")

    tier2_strip = _read_bgr(tier2_strip_path)
    gv_matches = _read_bgr(gv_matches_path)
    tier3_strip = _read_bgr(tier3_strip_path)

    out_flow = query_dir / "asset_tier3_gv_flow.png"
    if (not overwrite) and out_flow.exists():
        return

    _generate_tier3_gv_flow(
        tier2_strip=tier2_strip,
        gv_matches=gv_matches,
        tier3_strip=tier3_strip,
        out_path=out_flow,
        title="Tier-3: Geometric Verification (GV) re-ranks Tier-2",
        canvas_w=int(canvas_w),
        canvas_h=int(canvas_h),
        margin=int(margin),
    )

    assets = manifest.setdefault("assets", {})
    assets["tier3_gv_flow"] = str(out_flow)
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

    out = []
    for child in input_dir.iterdir():
        if child.is_dir() and (child / "assets_manifest.json").exists():
            out.append(child)
    return out


def main() -> None:
    _require_cv2()
    parser = argparse.ArgumentParser(description="Generate Tier-3 GV flow visual from existing pipeline assets.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("visualization_suite") / "output" / "presentation",
        help="Either a single query folder (contains assets_manifest.json) or an assets root containing per-query subfolders.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated Tier-3 visuals.")
    parser.add_argument("--canvas_w", type=int, default=3840, help="Width for Tier-3 GV flow image (16:9 recommended).")
    parser.add_argument("--canvas_h", type=int, default=2160, help="Height for Tier-3 GV flow image (16:9 recommended).")
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
                margin=int(args.margin),
            )
            processed += 1
        except Exception as e:
            print(f"[WARN] Failed for {qd}: {e}")

    print(f"Processed {processed} query folders.")


if __name__ == "__main__":
    main()
