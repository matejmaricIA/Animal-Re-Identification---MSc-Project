#!/usr/bin/env python3
"""
Import the SeaStarReID2023 dataset into this repo's expected format.

This script:
  - downloads SeaStarReID2023 via wildlife_datasets
  - stores images under ./data/<dataset>/original_data/
  - writes ./data/<dataset>/processed_metadata.csv with columns:
      image_id, identity, path, dataset, split

Usage:
  python utils/import_seastarreid2023.py
  python utils/import_seastarreid2023.py --dataset_name seastarreid2023 --force_download
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Import SeaStarReID2023 into ./data/seastarreid2023/")
    parser.add_argument(
        "--dataset_name",
        default="seastarreid2023",
        help="Target dataset folder name under ./data/ (default: seastarreid2023)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing ./data/<dataset>/original_data and metadata",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Force re-download even if the dataset already exists",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Test split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the split (default: 42)",
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

    from wildlife_datasets import datasets

    print(f"Downloading SeaStarReID2023 into {raw_dir}")
    datasets.SeaStarReID2023.get_data(str(raw_dir), force=args.force_download)

    ds = datasets.SeaStarReID2023(str(raw_dir), check_files=False)
    df = ds.metadata.copy()
    df["image_id"] = df["image_id"].astype(str)

    rel_root = raw_dir.relative_to(project_root)
    df["path"] = df["path"].apply(lambda p: str(rel_root / p))
    df["dataset"] = args.dataset_name

    if "split" not in df.columns:
        df["split"] = _assign_split_by_identity(df, args.test_ratio, args.seed)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(metadata_path, index=False)

    print(f"\nSaved metadata: {metadata_path}")
    print(f"Saved originals: {raw_dir}")
    print("\nNext:")
    print(f"  python main.py --train --ds {args.dataset_name} --use_global_embedding --embedding_model megadescriptor-l-384 --use_fisher false")


if __name__ == "__main__":
    main()
