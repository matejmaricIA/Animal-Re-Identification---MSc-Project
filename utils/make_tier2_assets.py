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
            "  ./venv/bin/python utils/make_tier2_assets.py --input_dir <ASSETS_DIR>\n"
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
    colors: list[tuple[int, int, int]],
) -> None:
    _require_cv2()
    if w <= 0 or h <= 0:
        return
    if not colors:
        colors = [(140, 140, 140)]

    n = max(1, len(colors))
    seg_w = max(1, int(round(w / float(n))))
    for i, c in enumerate(colors):
        x0 = x + i * seg_w
        x1 = x + w if i == n - 1 else min(x + w, x0 + seg_w)
        cv2.rectangle(canvas, (int(x0), int(y)), (int(x1), int(y + h)), c, -1, cv2.LINE_AA)


def _generate_tier2_fusion_flow(
    *,
    union_strip: np.ndarray,
    fusion_breakdown: np.ndarray,
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
    gap_h = max(90, int(round(avail_h * 0.07)))
    union_h = int(round(avail_h * 0.40))
    fusion_h = max(1, avail_h - union_h - gap_h)

    card_x = margin
    card_w = canvas_w - 2 * margin

    union_y = y_top
    fusion_y = union_y + union_h + gap_h

    # Accent colors (BGR).
    c_global = (223, 108, 45)  # RGB #2D6CDF
    c_fisher = (97, 162, 244)  # RGB #F4A261
    c_fused = (89, 169, 76)    # BGR used in fusion breakdown
    c_union = (143, 157, 42)   # RGB #2A9D8F

    # Cards.
    _draw_shadowed_rect(
        canvas,
        x=card_x,
        y=union_y,
        w=card_w,
        h=union_h,
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
        x=card_x,
        y=fusion_y,
        w=card_w,
        h=fusion_h,
        fill=(255, 255, 255),
        border=(222, 222, 222),
        border_thickness=2,
        shadow_dx=10,
        shadow_dy=12,
        shadow_alpha=0.10,
        radius=32,
    )

    # Inner content region (reserve header for accent bar).
    inner_pad_lr = max(26, int(round(margin * 0.40)))
    inner_pad_bottom = inner_pad_lr
    bar_inset = max(16, int(round(margin * 0.20)))
    bar_h = max(10, int(round(min(union_h, fusion_h) * 0.035)))
    header_gap = max(18, int(round(margin * 0.22)))
    inner_pad_top = inner_pad_lr + bar_inset + bar_h + header_gap

    # Union strip.
    union_inner_w = max(1, card_w - 2 * inner_pad_lr)
    union_inner_h = max(1, union_h - inner_pad_top - inner_pad_bottom)
    union_fit = _resize_contain(union_strip, union_inner_w, union_inner_h)
    u_h, u_w = union_fit.shape[:2]
    u_x = int(card_x + inner_pad_lr + (union_inner_w - u_w) // 2)
    u_y = int(union_y + inner_pad_top + (union_inner_h - u_h) // 2)
    _paste(canvas, union_fit, u_x, u_y)
    _draw_accent_bar(
        canvas,
        x=int(card_x + bar_inset),
        y=int(union_y + bar_inset),
        w=int(card_w - 2 * bar_inset),
        h=int(bar_h),
        colors=[c_union],
    )

    # Fusion breakdown.
    fusion_inner_w = max(1, card_w - 2 * inner_pad_lr)
    fusion_inner_h = max(1, fusion_h - inner_pad_top - inner_pad_bottom)
    fusion_fit = _resize_contain(fusion_breakdown, fusion_inner_w, fusion_inner_h)
    f_h, f_w = fusion_fit.shape[:2]
    f_x = int(card_x + inner_pad_lr + (fusion_inner_w - f_w) // 2)
    f_y = int(fusion_y + inner_pad_top + (fusion_inner_h - f_h) // 2)
    _paste(canvas, fusion_fit, f_x, f_y)
    # Tri-color accent bar: Global (blue), Fisher (orange), Fused (green).
    _draw_accent_bar(
        canvas,
        x=int(card_x + bar_inset),
        y=int(fusion_y + bar_inset),
        w=int(card_w - 2 * bar_inset),
        h=int(bar_h),
        colors=[c_global, c_fisher, c_fused],
    )

    # Connector arrow (Union -> Fusion).
    arrow_thickness = 6 if canvas_w >= 3000 else 4
    tip = 0.018
    start = (int(card_x + card_w // 2), int(union_y + union_h))
    end = (int(card_x + card_w // 2), int(fusion_y))
    mid_y = int((start[1] + end[1]) // 2)
    # Subtle "color turn": union teal -> fused green.
    cv2.line(canvas, start, (start[0], mid_y), c_union, arrow_thickness, cv2.LINE_AA)
    cv2.arrowedLine(canvas, (start[0], mid_y), end, c_fused, arrow_thickness, cv2.LINE_AA, tipLength=tip)
    cv2.circle(canvas, (start[0], mid_y), 8 if canvas_w >= 3000 else 6, (60, 60, 60), -1, cv2.LINE_AA)

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

    union_strip_path = get_asset_path("tier1_union_strip", "asset_tier1_union_strip.png")
    fusion_breakdown_path = get_asset_path("tier2_fusion_breakdown", "asset_tier2_fusion_breakdown.png")

    union_strip = _read_bgr(union_strip_path)
    fusion_breakdown = _read_bgr(fusion_breakdown_path)

    out_flow = query_dir / "asset_tier2_fusion_flow.png"
    if (not overwrite) and out_flow.exists():
        return

    _generate_tier2_fusion_flow(
        union_strip=union_strip,
        fusion_breakdown=fusion_breakdown,
        out_path=out_flow,
        title="Tier-2 Fusion: Rerank the Tier-1 Union shortlist",
        canvas_w=int(canvas_w),
        canvas_h=int(canvas_h),
        margin=int(margin),
    )

    assets = manifest.setdefault("assets", {})
    assets["tier2_fusion_flow"] = str(out_flow)
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
    parser = argparse.ArgumentParser(description="Generate Tier-2 fusion flow visual from existing pipeline assets.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("visualization_suite") / "output" / "presentation",
        help="Either a single query folder (contains assets_manifest.json) or an assets root containing per-query subfolders.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated Tier-2 visuals.")
    parser.add_argument("--canvas_w", type=int, default=3840, help="Width for Tier-2 fusion flow image (16:9 recommended).")
    parser.add_argument("--canvas_h", type=int, default=2160, help="Height for Tier-2 fusion flow image (16:9 recommended).")
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

