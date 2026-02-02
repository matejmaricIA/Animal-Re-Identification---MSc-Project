#!/usr/bin/env python3
"""Delete data/<dataset>/ folders for MD-trained datasets."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from constants import MD_DATASET_SPLITS


def resolve_dataset_dir(data_root: Path, name: str) -> Path | None:
    direct = data_root / name
    if direct.exists():
        return direct
    lower_name = name.lower()
    for child in data_root.iterdir():
        if child.is_dir() and child.name.lower() == lower_name:
            return child
    return None


def delete_dataset_dir(path: Path) -> None:
    shutil.rmtree(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Delete data/<dataset>/ folders for MD dataset subsets."
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Data root (default: data)",
    )
    parser.add_argument(
        "--trained-on",
        choices=["true", "false", "all"],
        default="true",
        help="Filter datasets by MD trained_on flag (default: true).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete folders (default: dry-run).",
    )
    args = parser.parse_args()

    data_root = (repo_root / args.data_root).resolve()
    trained_flag = args.trained_on.lower()
    if trained_flag == "all":
        targets = list(MD_DATASET_SPLITS.keys())
    else:
        want_trained = trained_flag == "true"
        targets = [
            name
            for name, meta in MD_DATASET_SPLITS.items()
            if meta.get("trained_on") == want_trained
        ]

    if not targets:
        print(f"No datasets found for trained_on={args.trained_on}.")
        return 0

    print(f"Data root: {data_root}")
    print(f"Target datasets (trained_on={args.trained_on}): {len(targets)}")

    dataset_dirs: list[Path] = []
    for name in targets:
        dataset_dir = resolve_dataset_dir(data_root, name)
        if dataset_dir is None:
            print(f"[SKIP] Dataset not found under data root: {name}")
            continue
        dataset_dirs.append(dataset_dir)

    if not dataset_dirs:
        print("Nothing to purge.")
        return 0

    for path in dataset_dirs:
        action = "DELETE" if args.apply else "DRY-RUN"
        print(f"[{action}] {path}")
        if args.apply:
            delete_dataset_dir(path)

    if args.apply:
        print(f"Deleted {len(dataset_dirs)} dataset folders.")
    else:
        print(
            f"Dry-run complete ({len(dataset_dirs)} folders). Use --apply to delete."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
