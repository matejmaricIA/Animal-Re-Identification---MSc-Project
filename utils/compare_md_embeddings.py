#!/usr/bin/env python3
"""Compare MegaDescriptor embeddings between baseline and main pipelines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
import timm

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from megadescriptor import load_megadescriptor_l_384


def resolve_dataset_dir(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.exists():
        return direct
    name_lower = name.lower()
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() == name_lower:
            return child
    return None


def build_baseline_paths(baseline_root: Path, dataset: str) -> pd.DataFrame:
    meta_root = baseline_root / "metadata" / "datasets"
    img_root = baseline_root / "images" / "size-518"

    meta_dir = resolve_dataset_dir(meta_root, dataset)
    if meta_dir is None:
        raise FileNotFoundError(f"Baseline metadata not found for dataset: {dataset}")
    meta_csv = meta_dir / "metadata.csv"
    if not meta_csv.exists():
        raise FileNotFoundError(f"Missing baseline metadata CSV: {meta_csv}")

    df = pd.read_csv(meta_csv, dtype={"image_id": str, "identity": str})
    df["image_id"] = df["image_id"].astype(str)

    img_dir = resolve_dataset_dir(img_root, meta_dir.name)
    if img_dir is None:
        raise FileNotFoundError(f"Baseline image dir not found for dataset: {meta_dir.name}")

    df["abs_path"] = df["path"].astype(str).apply(lambda p: str((img_dir / p).resolve()))
    return df


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def baseline_model(device: torch.device):
    model = timm.create_model("hf-hub:BVRA/wildlife-mega-L-384", pretrained=True)
    model.to(device).eval()
    preprocess = T.Compose(
        [
            T.Resize((384, 384)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return model, preprocess


def embed(model, preprocess, device, paths: list[str]) -> np.ndarray:
    out = []
    with torch.inference_mode():
        for p in paths:
            img = load_image(p)
            tens = preprocess(img).unsqueeze(0).to(device)
            emb = model(tens)
            if isinstance(emb, (list, tuple)):
                emb = emb[0]
            out.append(emb.squeeze().cpu().numpy())
    return np.vstack(out)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_n = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (a_n * b_n).sum(axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare MegaDescriptor embeddings between baseline and main pipelines."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument(
        "--baseline-root",
        default=str(repo_root / "test-scripts" / "wildlife-tools-data"),
        help="Baseline data root (default: test-scripts/wildlife-tools-data)",
    )
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument(
        "--image-ids",
        default="",
        help="Comma-separated list of image_id to compare (overrides sampling)",
    )
    args = parser.parse_args()

    baseline_root = Path(args.baseline_root)
    df = build_baseline_paths(baseline_root, args.dataset)

    if args.image_ids.strip():
        target_ids = [s.strip() for s in args.image_ids.split(",") if s.strip()]
        df = df[df["image_id"].isin(target_ids)].copy()
    else:
        df = df.sample(n=min(args.num_samples, len(df)), random_state=args.seed).copy()

    if df.empty:
        print("No images selected.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_model, base_pre = baseline_model(device)
    main_model, main_pre = load_megadescriptor_l_384(device)

    paths = df["abs_path"].tolist()
    ids = df["image_id"].tolist()

    base_emb = embed(base_model, base_pre, device, paths)
    main_emb = embed(main_model, main_pre, device, paths)

    cos = cosine_similarity(base_emb, main_emb)
    l2 = np.linalg.norm(base_emb - main_emb, axis=1)
    max_abs = np.max(np.abs(base_emb - main_emb), axis=1)

    print("\nPer-image comparison:")
    for img_id, c, d, m in zip(ids, cos, l2, max_abs):
        print(f"  {img_id}: cosine={c:.6f} l2={d:.6e} max_abs={m:.6e}")

    print("\nSummary:")
    print(f"  cosine: min={cos.min():.6f} mean={cos.mean():.6f} max={cos.max():.6f}")
    print(f"  l2:     min={l2.min():.6e} mean={l2.mean():.6e} max={l2.max():.6e}")
    print(f"  max_abs:min={max_abs.min():.6e} mean={max_abs.mean():.6e} max={max_abs.max():.6e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
