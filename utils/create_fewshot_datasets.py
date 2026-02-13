#!/usr/bin/env python3
"""
Create few-shot dataset variants by capping train images per identity.

Example:
  python utils/create_fewshot_datasets.py --datasets atrw sealid --max_train_per_identity 3

This writes:
  data/atrw_fewshot/processed_metadata.csv
  data/sealid_fewshot/processed_metadata.csv

By default only non-test rows are capped. Test rows are kept unchanged.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utility_functions import load_dataset


REQUIRED_COLUMNS = ("image_id", "identity", "split")


def _validate_columns(df: pd.DataFrame, dataset_name: str) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset '{dataset_name}' is missing required columns: {missing}. "
            "Expected at least image_id, identity, split."
        )


def _sample_per_identity(
    df: pd.DataFrame,
    max_per_identity: int | None,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if max_per_identity is None:
        return df
    if max_per_identity <= 0:
        raise ValueError("max_per_identity must be >= 1 when provided.")
    if df.empty:
        return df

    kept_groups: list[pd.DataFrame] = []
    for _, group in df.groupby("identity", sort=False):
        keep_n = min(len(group), max_per_identity)
        if len(group) > keep_n:
            seed = int(rng.integers(0, 2**31 - 1))
            group = group.sample(n=keep_n, replace=False, random_state=seed)
        kept_groups.append(group)
    return pd.concat(kept_groups, ignore_index=False)


def create_fewshot_dataset(
    source_dataset: str,
    *,
    suffix: str,
    max_train_per_identity: int,
    max_test_per_identity: int | None,
    force: bool,
    seed: int,
) -> None:
    target_dataset = f"{source_dataset}{suffix}"
    target_dir = Path("data") / target_dataset
    target_metadata = target_dir / "processed_metadata.csv"

    if target_dir.exists():
        if not force:
            raise FileExistsError(
                f"Target dataset already exists: {target_dir}. "
                "Use --force to overwrite."
            )
        shutil.rmtree(target_dir)

    df = load_dataset(source_dataset)
    if df is None or df.empty:
        raise ValueError(f"Source dataset '{source_dataset}' is empty or unavailable.")
    _validate_columns(df, source_dataset)

    df = df.copy()
    df["image_id"] = df["image_id"].astype(str)
    df["identity"] = df["identity"].astype(str)
    split_lower = df["split"].astype(str).str.strip().str.lower()
    df["_orig_index"] = np.arange(len(df), dtype=np.int64)

    train_df = df[split_lower != "test"].copy()
    test_df = df[split_lower == "test"].copy()

    rng = np.random.default_rng(seed)
    train_kept = _sample_per_identity(train_df, max_train_per_identity, rng)
    test_kept = _sample_per_identity(test_df, max_test_per_identity, rng)

    out_df = (
        pd.concat([train_kept, test_kept], ignore_index=False)
        .sort_values("_orig_index")
        .drop(columns=["_orig_index"])
        .copy()
    )
    out_df["dataset"] = target_dataset

    target_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(target_metadata, index=False)

    source_train = len(train_df)
    source_test = len(test_df)
    out_train = int((out_df["split"].astype(str).str.lower() != "test").sum())
    out_test = int((out_df["split"].astype(str).str.lower() == "test").sum())

    print(f"\nCreated few-shot dataset: {target_dataset}")
    print(f"Source: {source_dataset}")
    print(
        "Train rows: "
        f"{source_train} -> {out_train} (max {max_train_per_identity} per identity)"
    )
    if max_test_per_identity is None:
        print(f"Test rows: {source_test} -> {out_test} (unchanged)")
    else:
        print(
            "Test rows: "
            f"{source_test} -> {out_test} (max {max_test_per_identity} per identity)"
        )
    print(f"Saved metadata: {target_metadata}")
    print(f"Run with: python main.py --train --ds {target_dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create <dataset>_fewshot variants by capping rows per identity."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Source dataset names (e.g., atrw sealid).",
    )
    parser.add_argument(
        "--max_train_per_identity",
        type=int,
        default=3,
        help="Maximum number of non-test rows kept per identity (default: 3).",
    )
    parser.add_argument(
        "--max_test_per_identity",
        type=int,
        default=None,
        help="Optional cap for test rows per identity (default: keep all test rows).",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_fewshot",
        help="Suffix appended to each source dataset name (default: _fewshot).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for per-identity sampling (default: 42).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing target dataset folder.",
    )
    args = parser.parse_args()

    if not args.suffix:
        raise ValueError("--suffix cannot be empty.")
    if args.max_train_per_identity <= 0:
        raise ValueError("--max_train_per_identity must be >= 1.")
    if args.max_test_per_identity is not None and args.max_test_per_identity <= 0:
        raise ValueError("--max_test_per_identity must be >= 1 when provided.")

    for source in args.datasets:
        source = str(source).strip()
        if not source:
            continue
        create_fewshot_dataset(
            source,
            suffix=args.suffix,
            max_train_per_identity=args.max_train_per_identity,
            max_test_per_identity=args.max_test_per_identity,
            force=args.force,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
