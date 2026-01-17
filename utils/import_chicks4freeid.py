#!/usr/bin/env python3
"""
Import the Chicks4FreeID dataset from Hugging Face into this repo's expected format.

This script:
  - downloads Chicks4FreeID via `datasets.load_dataset`
  - saves images under `data/chicks4freeid/original_data/`
  - writes `data/chicks4freeid/processed_metadata.csv` with columns:
      image_id, identity, path, dataset, split

After running, you can evaluate different descriptors with:
  - MegaDescriptor only:
      python main.py --train --ds chicks4freeid --use_global_embedding --embedding_model megadescriptor-l-384 --use_fisher false
  - Fisher only (example with DISK):
      python main.py --train --ds chicks4freeid --use_fisher true --use_global_embedding false --method disk
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def _coerce_pil_image(value: Any):
    from PIL import Image

    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        maybe_path = value.get("path")
        if maybe_path:
            return Image.open(maybe_path)
        maybe_bytes = value.get("bytes")
        if maybe_bytes:
            import io

            return Image.open(io.BytesIO(maybe_bytes))
        return None
    # numpy array fallback
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return Image.fromarray(value)
    except Exception:
        pass
    return None


def _extract_image(example: dict[str, Any]):
    from PIL import Image

    for key in ("crop", "image", "img"):
        if key in example:
            img = _coerce_pil_image(example.get(key))
            if img is not None:
                return img
    for v in example.values():
        if isinstance(v, Image.Image):
            return v
        img = _coerce_pil_image(v)
        if img is not None:
            return img
    raise KeyError(f"Could not find an image field in example keys: {sorted(example.keys())}")


def _iter_splits(dataset: Any, preferred: Iterable[str]) -> list[str]:
    keys = list(getattr(dataset, "keys", lambda: [])())
    found = [k for k in preferred if k in keys]
    return found or keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Chicks4FreeID into ./data/chicks4freeid/")
    parser.add_argument(
        "--dataset_name",
        default="chicks4freeid",
        help="Target dataset folder name under ./data/ (default: chicks4freeid)",
    )
    parser.add_argument(
        "--hf_repo",
        default="dariakern/Chicks4FreeID",
        help="Hugging Face dataset repo (default: dariakern/Chicks4FreeID)",
    )
    parser.add_argument(
        "--hf_config",
        default="chicken-re-id-all-visibility",
        help="Dataset configuration name (default: chicken-re-id-all-visibility)",
    )
    parser.add_argument(
        "--image_format",
        choices=("png", "jpg", "jpeg"),
        default="png",
        help="Format to save original images (default: png for lossless)",
    )
    parser.add_argument(
        "--limit_train",
        type=int,
        default=None,
        help="Optional cap on the number of train samples to export",
    )
    parser.add_argument(
        "--limit_test",
        type=int,
        default=None,
        help="Optional cap on the number of test samples to export",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing ./data/<dataset_name>/original_data and metadata",
    )
    args = parser.parse_args()

    from datasets import load_dataset

    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = project_root / "data" / args.dataset_name
    originals_dir = dataset_dir / "original_data"
    metadata_path = dataset_dir / "processed_metadata.csv"

    if args.force:
        if originals_dir.exists():
            shutil.rmtree(originals_dir)
        if metadata_path.exists():
            metadata_path.unlink()

    originals_dir.mkdir(parents=True, exist_ok=True)

    if metadata_path.exists() and not args.force:
        raise SystemExit(
            f"{metadata_path} already exists. Re-run with --force to overwrite, or choose a different --dataset_name."
        )

    print(f"Downloading dataset from Hugging Face: {args.hf_repo} ({args.hf_config})")
    dataset = load_dataset(args.hf_repo, args.hf_config)

    rows: list[dict[str, Any]] = []
    split_order = _iter_splits(dataset, preferred=("train", "test", "validation"))

    for split in split_order:
        split_ds = dataset[split]
        limit = None
        if split == "train":
            limit = args.limit_train
        elif split == "test":
            limit = args.limit_test

        print(f"Exporting split={split} (rows={len(split_ds)})")
        for idx, example in enumerate(split_ds):
            if limit is not None and idx >= limit:
                break

            if "identity" not in example:
                raise KeyError(f"Expected 'identity' in example, got keys: {sorted(example.keys())}")
            identity = example["identity"]

            image = _extract_image(example).convert("RGB")
            image_id = f"{split}_{idx:06d}"

            out_path = originals_dir / f"{image_id}.{args.image_format}"
            if args.image_format in {"jpg", "jpeg"}:
                image.save(out_path, quality=95, optimize=True)
            else:
                image.save(out_path)

            rows.append(
                {
                    "image_id": str(image_id),
                    "identity": identity,
                    "path": str(out_path.relative_to(project_root)),
                    "dataset": args.dataset_name,
                    "split": split,
                }
            )

    if not rows:
        raise SystemExit("No samples exported. Check that the HF dataset and config are correct.")

    df = pd.DataFrame(rows)
    df["image_id"] = df["image_id"].astype(str)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(metadata_path, index=False)

    print(f"\nSaved metadata: {metadata_path}")
    print(f"Saved originals: {originals_dir}")
    print("\nNext:")
    print(
        f"  python main.py --train --ds {args.dataset_name} --use_global_embedding --embedding_model megadescriptor-l-384 --use_fisher false"
    )


if __name__ == "__main__":
    main()

