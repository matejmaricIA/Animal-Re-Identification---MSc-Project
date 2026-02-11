#!/usr/bin/env python3
"""
Generate a single PNG figure showing all processed images for a random ELPephants identity.

Why: ELPephants is visually challenging; this collage makes that apparent at a glance.

Input:
  - data/elpephants/processed_metadata.csv
  - processed images under data/elpephants/dataset/<identity>/<image_id>.jpg

Output:
  - visualization_suite/output/elpephants_identity_<identity>_n<N>_seed<seed>.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visualization_suite import io, style


def _resize_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    scale = float(max_side) / float(max(h, w))
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)


def _choose_grid(n: int) -> tuple[int, int]:
    if n <= 0:
        return 1, 1

    best_cols = 1
    best_rows = n
    best_empty = best_rows * best_cols - n
    best_squareness = abs(best_rows - best_cols)

    for cols in range(1, n + 1):
        rows = int(np.ceil(n / cols))
        empty = rows * cols - n
        squareness = abs(rows - cols)

        if empty < best_empty:
            best_cols, best_rows = cols, rows
            best_empty = empty
            best_squareness = squareness
            continue

        if empty == best_empty and squareness < best_squareness:
            best_cols, best_rows = cols, rows
            best_squareness = squareness
            continue

        if empty == best_empty and squareness == best_squareness and rows < best_rows:
            best_cols, best_rows = cols, rows

    return best_rows, best_cols


def _resolve_processed_image_path(row: dict) -> Path:
    identity = str(row["identity"]).strip()
    image_id = str(row["image_id"]).strip()

    base = Path(str(row.get("processed_path", "")).strip())
    if not base:
        base = PROJECT_ROOT / "data" / "elpephants" / "dataset" / identity
    elif not base.is_absolute():
        base = (PROJECT_ROOT / base).resolve()

    return base / f"{image_id}.jpg"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a collage PNG for a random ELPephants identity (processed images only)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed (default: random).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output PNG path (default: visualization_suite/output/<auto-name>.png).",
    )
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
    rng = np.random.default_rng(seed)

    metadata_path = PROJECT_ROOT / "data" / "elpephants" / "processed_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing metadata: {metadata_path}. Did you run utils/import_elpephants.py?"
        )

    df = pd.read_csv(metadata_path, dtype={"image_id": str, "identity": str})
    if "identity" not in df.columns or "image_id" not in df.columns:
        raise ValueError("Expected columns 'identity' and 'image_id' in processed_metadata.csv.")

    identities = df["identity"].astype(str).map(str.strip)
    identities = identities[(identities != "") & (identities.str.lower() != "nan")]
    unique_ids = identities.unique().tolist()
    if not unique_ids:
        raise RuntimeError("No identities found in ELPephants metadata.")

    chosen_identity = str(rng.choice(unique_ids))
    rows_df = df[df["identity"].astype(str) == chosen_identity].copy()
    rows_df = rows_df.sample(frac=1.0, random_state=int(seed) & 0xFFFF_FFFF)

    images: list[np.ndarray] = []
    missing: int = 0
    max_side = 640

    for row in rows_df.to_dict(orient="records"):
        img_path = _resolve_processed_image_path(row)
        if not img_path.exists():
            missing += 1
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            missing += 1
            continue
        images.append(_resize_max_side(img, max_side))

    if not images:
        raise RuntimeError(f"No processed images found for identity {chosen_identity}.")

    grid_rows, grid_cols = _choose_grid(len(images))

    style.set_style()
    cell_in = 2.6
    fig, axes = plt.subplots(
        grid_rows,
        grid_cols,
        figsize=(grid_cols * cell_in, grid_rows * cell_in),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for idx, img in enumerate(images):
        ax = axes_flat[idx]
        ax.imshow(io.bgr_to_rgb(img))
        ax.axis("off")

    for ax in axes_flat[len(images) :]:
        ax.axis("off")

    fig.tight_layout(pad=0.0)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.01, hspace=0.01)

    out_path: Path
    if args.out:
        out_path = Path(args.out).expanduser()
        if not out_path.is_absolute():
            out_path = (PROJECT_ROOT / out_path).resolve()
    else:
        out_dir = PROJECT_ROOT / "visualization_suite" / "output"
        out_path = out_dir / f"elpephants_identity_{chosen_identity}_n{len(images)}_seed{seed}.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print(f"seed={seed}")
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
