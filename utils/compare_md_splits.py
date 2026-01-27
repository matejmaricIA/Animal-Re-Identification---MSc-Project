#!/usr/bin/env python3
"""Compare splits between MegaDescriptor combined_all.csv and local all_datasets.csv.

Matching is done by dataset name + image basename (filename).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple


def _norm(value: str | None) -> str:
    return (value or "").strip()


def _basename(path: str) -> str:
    tokens = [p for p in _norm(path).replace("\\", "/").split("/") if p and p != "."]
    return tokens[-1] if tokens else ""


def _find_col(header: Iterable[str], candidates: Iterable[str]) -> str | None:
    candidates = {c.lower() for c in candidates}
    for name in header:
        if name is None:
            continue
        if str(name).lower() in candidates:
            return name
    return None


def _load_index(csv_path: Path) -> Tuple[
    Dict[str, Dict[str, Dict[str, str]]],
    Dict[str, int],
    Dict[str, int],
]:
    """Load CSV into dataset->key->payload maps.

    Returns:
        dataset_map, duplicates_per_dataset, total_rows_per_dataset
    """
    dataset_map: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
    duplicates: Dict[str, int] = defaultdict(int)
    totals: Dict[str, int] = defaultdict(int)

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []

        dataset_col = _find_col(header, {"dataset"})
        path_col = _find_col(header, {"path", "filepath", "file_path", "img_path"})
        split_col = _find_col(header, {"split", "set", "partition"})

        if dataset_col is None or path_col is None or split_col is None:
            raise ValueError(
                f"Missing required columns in {csv_path}. "
                f"Found header={header}"
            )

        for row in reader:
            dataset = _norm(row.get(dataset_col)).lower()
            path = _norm(row.get(path_col))
            split = _norm(row.get(split_col)).lower()
            key = _basename(path).lower()
            if not dataset or not key:
                continue
            totals[dataset] += 1
            if key in dataset_map[dataset]:
                duplicates[dataset] += 1
                # Keep first occurrence, but count duplicates.
                continue
            dataset_map[dataset][key] = {
                "split": split,
                "path": path,
            }

    return dataset_map, duplicates, totals


def _summarize(
    combined_map: Dict[str, Dict[str, Dict[str, str]]],
    local_map: Dict[str, Dict[str, Dict[str, str]]],
    combined_totals: Dict[str, int],
    local_totals: Dict[str, int],
    max_examples: int,
    dataset_filter: set[str] | None,
) -> None:
    datasets = sorted(set(combined_map.keys()) | set(local_map.keys()))
    if dataset_filter:
        datasets = [d for d in datasets if d in dataset_filter]

    header = (
        "dataset",
        "local_total",
        "combined_total",
        "matched",
        "mismatched",
        "missing_in_combined",
        "missing_in_local",
    )
    print(",".join(header))

    grand = {k: 0 for k in header[1:]}
    for ds in datasets:
        local_keys = local_map.get(ds, {})
        combined_keys = combined_map.get(ds, {})
        matched = 0
        mismatched = 0
        missing_in_combined = 0
        missing_in_local = 0
        examples = []

        for key, local_row in local_keys.items():
            comb_row = combined_keys.get(key)
            if comb_row is None:
                missing_in_combined += 1
                continue
            if local_row["split"] == comb_row["split"]:
                matched += 1
            else:
                mismatched += 1
                if len(examples) < max_examples:
                    examples.append(
                        (
                            key,
                            local_row["split"],
                            comb_row["split"],
                            local_row["path"],
                            comb_row["path"],
                        )
                    )

        for key in combined_keys.keys():
            if key not in local_keys:
                missing_in_local += 1

        row = (
            ds,
            str(local_totals.get(ds, 0)),
            str(combined_totals.get(ds, 0)),
            str(matched),
            str(mismatched),
            str(missing_in_combined),
            str(missing_in_local),
        )
        print(",".join(row))

        grand["local_total"] += local_totals.get(ds, 0)
        grand["combined_total"] += combined_totals.get(ds, 0)
        grand["matched"] += matched
        grand["mismatched"] += mismatched
        grand["missing_in_combined"] += missing_in_combined
        grand["missing_in_local"] += missing_in_local

        if examples:
            print(f"# examples mismatched in {ds} (basename, local_split, combined_split)")
            for ex in examples:
                print(f"# {ex[0]} | local={ex[1]} | combined={ex[2]}")

    total_row = (
        "TOTAL",
        str(grand["local_total"]),
        str(grand["combined_total"]),
        str(grand["matched"]),
        str(grand["mismatched"]),
        str(grand["missing_in_combined"]),
        str(grand["missing_in_local"]),
    )
    print(",".join(total_row))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare splits between combined_all.csv and all_datasets.csv "
            "using dataset+image basename matching."
        )
    )
    parser.add_argument(
        "--combined",
        default="docs/combined_all.csv",
        help="Path to combined_all.csv (default: docs/combined_all.csv)",
    )
    parser.add_argument(
        "--local",
        default="data/all_datasets.csv",
        help="Path to local all_datasets.csv (default: data/all_datasets.csv)",
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Optional comma-separated dataset filter (case-insensitive).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Max mismatched examples per dataset (default: 5).",
    )
    args = parser.parse_args()

    combined_path = Path(args.combined)
    local_path = Path(args.local)

    if not combined_path.exists():
        raise SystemExit(f"Missing combined_all.csv at {combined_path}")
    if not local_path.exists():
        raise SystemExit(f"Missing all_datasets.csv at {local_path}")

    dataset_filter = None
    if args.dataset.strip():
        dataset_filter = {d.strip().lower() for d in args.dataset.split(",") if d.strip()}

    combined_map, combined_dupes, combined_totals = _load_index(combined_path)
    local_map, local_dupes, local_totals = _load_index(local_path)

    _summarize(
        combined_map,
        local_map,
        combined_totals,
        local_totals,
        max_examples=args.max_examples,
        dataset_filter=dataset_filter,
    )

    dup_total_combined = sum(combined_dupes.values())
    dup_total_local = sum(local_dupes.values())
    if dup_total_combined or dup_total_local:
        print("# duplicate basenames detected (kept first occurrence)")
        if dup_total_combined:
            print(f"# combined duplicates: {dup_total_combined}")
        if dup_total_local:
            print(f"# local duplicates: {dup_total_local}")


if __name__ == "__main__":
    main()
