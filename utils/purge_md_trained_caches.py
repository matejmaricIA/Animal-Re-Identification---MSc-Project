#!/usr/bin/env python3
"""Purge cached descriptors/embeddings for MD-trained datasets."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from constants import MD_DATASET_SPLITS


PATTERNS = (
    "feature_descriptors_*",
    "pca_model_*",
    "gmm_model_*",
    "fisher_vectors_*",
    "global_embeddings_*",
)


def resolve_dataset_dir(data_root: Path, name: str) -> Path | None:
    direct = data_root / name
    if direct.exists():
        return direct
    lower_name = name.lower()
    for child in data_root.iterdir():
        if child.is_dir() and child.name.lower() == lower_name:
            return child
    return None


def collect_paths(dataset_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in PATTERNS:
        paths.extend(dataset_dir.glob(pattern))
    return sorted(set(paths))


def delete_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Purge cached descriptors/embeddings for MD-trained datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help=(
            "Optional dataset names to purge (case-insensitive). "
            "Default: all datasets with trained_on=True in constants.MD_DATASET_SPLITS."
        ),
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Data root (default: data)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files (default: dry-run).",
    )
    args = parser.parse_args()

    data_root = (repo_root / args.data_root).resolve()
    if args.datasets:
        targets = [str(name).strip() for name in args.datasets if str(name).strip()]
        mode = "explicit"
    else:
        targets = [
            name
            for name, meta in MD_DATASET_SPLITS.items()
            if meta.get("trained_on")
        ]
        mode = "trained_on"

    if not targets:
        print("No datasets selected.")
        return 0

    print(f"Data root: {data_root}")
    print(f"Selection mode: {mode}")

    total_paths: list[Path] = []
    for name in targets:
        dataset_dir = resolve_dataset_dir(data_root, name)
        if dataset_dir is None:
            print(f"[SKIP] Dataset not found under data root: {name}")
            continue
        paths = collect_paths(dataset_dir)
        if not paths:
            print(f"[OK] No cached artifacts for {dataset_dir.name}")
            continue
        print(f"[FOUND] {dataset_dir.name}: {len(paths)} paths")
        total_paths.extend(paths)

    if not total_paths:
        print("Nothing to purge.")
        return 0

    for path in total_paths:
        action = "DELETE" if args.apply else "DRY-RUN"
        print(f"[{action}] {path}")
        if args.apply:
            delete_path(path)

    if args.apply:
        print(f"Deleted {len(total_paths)} paths.")
    else:
        print(f"Dry-run complete ({len(total_paths)} paths). Use --apply to delete.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
