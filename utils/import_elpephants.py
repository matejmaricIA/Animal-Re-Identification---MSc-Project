#!/usr/bin/env python3
"""
Import the ELPephants dataset into this repo's expected format.

This script:
  - expects the dataset to be manually downloaded (see the dataset site)
  - optionally extracts a provided archive into ./data/<dataset>/original_data/
  - writes ./data/<dataset>/processed_metadata.csv with columns:
      image_id, identity, path, dataset, split

Usage:
  python utils/import_elpephants.py --archive /path/to/ELPephant.zip
  python utils/import_elpephants.py --source /path/to/extracted/ELPephants
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _assign_split_by_identity(df: pd.DataFrame, test_ratio: float, seed: int) -> pd.Series:
    if test_ratio <= 0:
        return pd.Series(["train"] * len(df), index=df.index)
    rng = np.random.default_rng(seed)
    split = pd.Series(index=df.index, dtype="object")
    for identity, group in df.groupby("identity"):
        idx = group.index.to_list()
        if len(idx) <= 1:
            split.loc[idx] = "train"
            continue
        n_test = max(1, int(round(len(idx) * test_ratio)))
        test_idx = set(rng.choice(idx, size=n_test, replace=False).tolist())
        split.loc[idx] = ["test" if i in test_idx else "train" for i in idx]
    return split


def _has_images(root: Path) -> bool:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return any(p.is_file() and p.suffix.lower() in exts for p in root.rglob("*"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Import ELPephants into ./data/elpephants/")
    parser.add_argument(
        "--dataset_name",
        default="elpephants",
        help="Target dataset folder name under ./data/ (default: elpephants)",
    )
    parser.add_argument(
        "--archive",
        help="Path to the ELPephants archive (zip). If provided, it will be extracted.",
    )
    parser.add_argument(
        "--source",
        help="Path to an already extracted ELPephants folder to copy into ./data/<dataset>/original_data/",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing ./data/<dataset>/original_data and metadata",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Fallback test split ratio if no official split is found (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the fallback split (default: 42)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = project_root / "data" / args.dataset_name
    raw_dir = dataset_dir / "original_data"
    metadata_path = dataset_dir / "processed_metadata.csv"

    if args.force:
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        if metadata_path.exists():
            metadata_path.unlink()

    if metadata_path.exists() and not args.force:
        raise SystemExit(
            f"{metadata_path} already exists. Re-run with --force to overwrite, or choose a different --dataset_name."
        )

    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.archive:
        if raw_dir.exists() and any(raw_dir.iterdir()) and not args.force:
            raise SystemExit(
                f"{raw_dir} is not empty. Use --force to replace it before extracting {args.archive}."
            )
        shutil.unpack_archive(args.archive, raw_dir)
    elif args.source:
        source_path = Path(args.source).expanduser().resolve()
        if not source_path.exists():
            raise SystemExit(f"Source path does not exist: {source_path}")
        if raw_dir.exists() and any(raw_dir.iterdir()) and not args.force:
            raise SystemExit(
                f"{raw_dir} is not empty. Use --force to replace it before copying {source_path}."
            )
        shutil.copytree(source_path, raw_dir, dirs_exist_ok=True)

    if not _has_images(raw_dir):
        raise SystemExit(
            "No images found under original_data. Provide --archive or --source, or place images in "
            f"{raw_dir} and re-run."
        )

    from wildlife_datasets import datasets

    ds = datasets.ELPephants(str(raw_dir), check_files=False)
    df = ds.metadata.copy()
    df["image_id"] = df["image_id"].astype(str)

    rel_root = raw_dir.relative_to(project_root)
    df["path"] = df["path"].apply(lambda p: str(rel_root / p))
    df["dataset"] = args.dataset_name

    if "split" not in df.columns:
        if "original_split" in df.columns:
            df["split"] = df["original_split"].replace({"val": "test"}).fillna("train")
        else:
            df["split"] = _assign_split_by_identity(df, args.test_ratio, args.seed)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(metadata_path, index=False)

    print(f"\nSaved metadata: {metadata_path}")
    print(f"Saved originals: {raw_dir}")
    print("\nNext:")
    print(f"  python main.py --train --ds {args.dataset_name} --use_global_embedding --embedding_model megadescriptor-l-384 --use_fisher false")


if __name__ == "__main__":
    main()
