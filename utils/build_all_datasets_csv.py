#!/usr/bin/env python3
"""Build a per-image all_datasets.csv from the configured dataset list.

Rules:
- Use datasets from constants.MD_DATASET_SPLITS.
- For MD-trained datasets: random ClosedSetSplit (0.8, seed=666).
- For MD-untrained datasets: use official WReID split column.
- Default to closed-set (drop test identities not in train); optional --open-set.
- Copy missing images into data/<dataset>/dataset/ and rewrite paths to local copies.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from constants import MD_DATASET_SPLITS, WILD_DATASET_PATH
from wildlife_datasets import splits
from wildlife_datasets.datasets import WildlifeReID10k


IDENTITY_SKIP = "unknown"
SPLIT_RATIO = 0.8
SPLIT_SEED = 666


def _normalize_path(value: str) -> str:
    path = str(value).strip().replace("\\", "/")
    if path.startswith("./"):
        path = path[2:]
    return path


def _dataset_label_from_metadata(df_all: pd.DataFrame, key: str) -> str | None:
    if "dataset" not in df_all.columns:
        return None
    matches = df_all[df_all["dataset"].str.lower() == key]
    if matches.empty:
        return None
    return str(matches["dataset"].iloc[0])


def _resolve_source_path(
    path_value: str,
    dataset_key: str,
    project_root: Path,
    wreid_root: Path,
) -> Path:
    path_value = _normalize_path(path_value)
    src_path = Path(path_value)
    if src_path.is_absolute():
        return src_path
    if dataset_key != "elpephants":
        return wreid_root / path_value
    return project_root / path_value


def _dest_relative_path(dataset_label: str, path_value: str) -> Path:
    path_value = _normalize_path(path_value)
    rel_path = Path(path_value)
    if rel_path.is_absolute():
        return Path("data") / dataset_label / "dataset" / rel_path.name

    prefix = Path("data") / dataset_label
    try:
        rel_path = rel_path.relative_to(prefix)
    except ValueError:
        pass
    return Path("data") / dataset_label / "dataset" / rel_path


def _copy_if_missing(src: Path, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _apply_closed_set(df: pd.DataFrame, open_set: bool) -> pd.DataFrame:
    if open_set:
        return df
    train_ids = set(df.loc[df["split"] == "train", "identity"].astype(str))
    mask = ~((df["split"] == "test") & ~df["identity"].isin(train_ids))
    return df[mask].copy()


def _ensure_split_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["split"] = df["split"].astype(str).str.lower()
    df.loc[df["split"] != "test", "split"] = "train"
    return df


def build_all_datasets_csv(output_csv: Path, open_set: bool, copy_images: bool) -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data"
    wreid_root = project_root / WILD_DATASET_PATH

    wreid = WildlifeReID10k(str(wreid_root), check_files=False)
    wreid_df = wreid.metadata.copy()
    wreid_df["image_id"] = wreid_df["image_id"].astype(str)
    wreid_df["identity"] = wreid_df["identity"].astype(str)

    rows = []

    for dataset_key, meta in MD_DATASET_SPLITS.items():
        dataset_key = str(dataset_key).strip().lower()
        if dataset_key == "elpephants":
            dataset_label = "elpephants"
            local_csv = data_root / dataset_label / "processed_metadata.csv"
            if not local_csv.exists():
                print(f"Skipping {dataset_label}: missing {local_csv}")
                continue
            df = pd.read_csv(local_csv, dtype={"image_id": str, "identity": str})
        else:
            dataset_label = _dataset_label_from_metadata(wreid_df, dataset_key)
            if dataset_label is None:
                print(f"Skipping {dataset_key}: not found in WReID metadata")
                continue
            df = wreid_df[wreid_df["dataset"].str.lower() == dataset_key].copy()

        if "identity" not in df.columns:
            print(f"Skipping {dataset_label}: missing identity column")
            continue

        df = df[df["identity"].astype(str).str.lower() != IDENTITY_SKIP].copy()
        df["identity"] = df["identity"].astype(str)
        df["image_id"] = df["image_id"].astype(str)

        if meta["trained_on"]:
            df = df.reset_index(drop=True)
            splitter = splits.ClosedSetSplit(
                SPLIT_RATIO, identity_skip=IDENTITY_SKIP, seed=SPLIT_SEED
            )
            idx_train, idx_test = splitter.split(df)[0]
            df["split"] = "train"
            df.loc[df.index[idx_test], "split"] = "test"
        else:
            if "split" not in df.columns:
                print(f"Skipping {dataset_label}: missing split column")
                continue
        df = _ensure_split_column(df)
        df = _apply_closed_set(df, open_set)

        if copy_images:
            dest_root = data_root / dataset_label / "dataset"
            for i, row in df.iterrows():
                src = _resolve_source_path(
                    row["path"], dataset_key, project_root, wreid_root
                )
                dest_rel = _dest_relative_path(dataset_label, row["path"])
                dest = project_root / dest_rel
                if not dest.exists():
                    if src.exists():
                        _copy_if_missing(src, dest)
                    else:
                        print(f"Missing source: {src}")
                df.at[i, "path"] = dest_rel.as_posix()

        df["dataset"] = dataset_label
        df["split_type"] = meta["split_type"]
        df["trained_on"] = meta["trained_on"]
        df["random_split"] = meta["random_split"]

        rows.append(
            df[
                [
                    "dataset",
                    "image_id",
                    "identity",
                    "path",
                    "split",
                    "split_type",
                    "trained_on",
                    "random_split",
                ]
            ].copy()
        )

    if not rows:
        print("No datasets processed.")
        return

    out_df = pd.concat(rows, ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Wrote {len(out_df)} rows to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build data/all_datasets.csv for full training runs."
    )
    parser.add_argument(
        "--output",
        default="data/all_datasets.csv",
        help="Output CSV path (default: data/all_datasets.csv)",
    )
    parser.add_argument(
        "--open-set",
        action="store_true",
        help="Keep test identities not seen in train.",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Do not copy images into data/<dataset>/dataset/.",
    )
    args = parser.parse_args()

    build_all_datasets_csv(
        output_csv=Path(args.output),
        open_set=args.open_set,
        copy_images=not args.no_copy,
    )


if __name__ == "__main__":
    main()
