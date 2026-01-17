#!/usr/bin/env python3
"""
Checks for potential pretraining leakage when evaluating MegaDescriptor.

It downloads (or reuses a cached copy of) the MegaDescriptor training metadata
(`combined_all.csv`) and compares it to your local dataset split metadata
(`processed_metadata.csv`), reporting any TEST-split images that appear in the
MegaDescriptor TRAIN metadata.

This is useful when your local train/test split differs from the split used by
the MegaDescriptor authors (seed=666 closed-set 80/20).
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

MEGADESCRIPTOR_TRAIN_URL = (
    "https://raw.githubusercontent.com/WildlifeDatasets/wildlife-tools/main/"
    "baselines/data/metadata/combined/combined_all.csv"
)


def _norm(value: Optional[str]) -> str:
    return (value or "").strip()


def _path_parts(path: str) -> List[str]:
    path = _norm(path).replace("\\", "/")
    return [p for p in path.split("/") if p and p != "."]


def _path_suffix(path: str, parts: int) -> str:
    tokens = _path_parts(path)
    if not tokens:
        return ""
    if parts <= 0:
        return "/".join(tokens)
    return "/".join(tokens[-parts:])


def _basename(path: str) -> str:
    tokens = _path_parts(path)
    return tokens[-1] if tokens else ""


def _token_from_row(
    match_mode: str,
    *,
    path_value: str,
    image_id_value: str,
    path_suffix_parts: int,
) -> str:
    if match_mode == "image_id":
        return _norm(image_id_value).lower()
    if match_mode == "basename":
        return _basename(path_value).lower()
    # default: suffix
    return _path_suffix(path_value, path_suffix_parts).lower()


def _read_csv_indices(header: List[str]) -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(header)}


def _load_local_split_keys(
    metadata_csv: Path,
    *,
    split_col: str,
    test_split_values: Set[str],
    dataset_col: str,
    path_col: str,
    identity_col: str,
    image_id_col: str,
    match_mode: str,
    path_suffix_parts: int,
    default_dataset: str,
) -> Tuple[Set[str], Set[str], Dict[str, Dict[str, str]]]:
    train_keys: Set[str] = set()
    test_keys: Set[str] = set()
    test_rows: Dict[str, Dict[str, str]] = {}

    with metadata_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"Empty CSV: {metadata_csv}")

        col = _read_csv_indices(header)
        if split_col not in col:
            raise ValueError(f"Missing '{split_col}' column in {metadata_csv}")
        if path_col not in col:
            raise ValueError(f"Missing '{path_col}' column in {metadata_csv}")

        for row in reader:
            split_value = _norm(row[col[split_col]]).lower()
            dataset_value = _norm(row[col[dataset_col]]) if dataset_col in col else default_dataset
            dataset_value_norm = dataset_value.lower()

            path_value = row[col[path_col]]
            image_id_value = row[col[image_id_col]] if image_id_col in col else ""

            token = _token_from_row(
                match_mode,
                path_value=path_value,
                image_id_value=image_id_value,
                path_suffix_parts=path_suffix_parts,
            )
            if not token:
                continue

            key = f"{dataset_value_norm}|{token}"
            if split_value in test_split_values:
                test_keys.add(key)
                if key not in test_rows:
                    test_rows[key] = {
                        "dataset": dataset_value,
                        "split": split_value,
                        "path": _norm(path_value),
                        "identity": _norm(row[col[identity_col]]) if identity_col in col else "",
                        "image_id": _norm(image_id_value),
                        "source_csv": str(metadata_csv),
                    }
            else:
                train_keys.add(key)

    return train_keys, test_keys, test_rows


def _ensure_md_train_csv(path: Path, *, url: str, allow_download: bool) -> None:
    if path.exists():
        return
    if not allow_download:
        raise FileNotFoundError(
            f"MegaDescriptor training metadata not found at {path}. "
            f"Provide --md-train-csv or enable --download."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MegaDescriptor training metadata to: {path}", file=sys.stderr)
    with urllib.request.urlopen(url) as resp, path.open("wb") as out:
        shutil.copyfileobj(resp, out)


def _load_md_train_keys_for_local(
    md_train_csv: Path,
    *,
    local_keys: Set[str],
    dataset_col: str,
    path_col: str,
    identity_col: str,
    image_id_col: str,
    split_col: str,
    train_split_values: Set[str],
    match_mode: str,
    path_suffix_parts: int,
) -> Tuple[Set[str], Dict[str, Dict[str, str]]]:
    md_train_keys: Set[str] = set()
    md_rows: Dict[str, Dict[str, str]] = {}

    with md_train_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"Empty CSV: {md_train_csv}")

        col = _read_csv_indices(header)
        for required in (dataset_col, path_col):
            if required not in col:
                raise ValueError(f"Missing '{required}' column in {md_train_csv}")

        has_split = split_col in col
        for row in reader:
            if has_split:
                split_value = _norm(row[col[split_col]]).lower()
                if split_value not in train_split_values:
                    continue

            dataset_value = _norm(row[col[dataset_col]])
            dataset_value_norm = dataset_value.lower()

            path_value = row[col[path_col]]
            image_id_value = row[col[image_id_col]] if image_id_col in col else ""

            token = _token_from_row(
                match_mode,
                path_value=path_value,
                image_id_value=image_id_value,
                path_suffix_parts=path_suffix_parts,
            )
            if not token:
                continue

            key = f"{dataset_value_norm}|{token}"
            if key not in local_keys:
                continue

            md_train_keys.add(key)
            if key not in md_rows:
                md_rows[key] = {
                    "dataset": dataset_value,
                    "path": _norm(path_value),
                    "identity": _norm(row[col[identity_col]]) if identity_col in col else "",
                    "image_id": _norm(image_id_value),
                    "source_csv": str(md_train_csv),
                }

    return md_train_keys, md_rows


def _dataset_from_key(key: str) -> str:
    return key.split("|", 1)[0]


def _write_leak_report_csv(
    report_path: Path,
    *,
    leak_keys: Iterable[str],
    local_test_rows: Dict[str, Dict[str, str]],
    md_rows: Dict[str, Dict[str, str]],
    match_mode: str,
    path_suffix_parts: int,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "match_mode",
                "path_suffix_parts",
                "match_token",
                "local_split",
                "local_path",
                "local_identity",
                "local_image_id",
                "local_source_csv",
                "megadescriptor_train_path",
                "megadescriptor_train_identity",
                "megadescriptor_train_image_id",
                "megadescriptor_source_csv",
            ]
        )

        for key in sorted(leak_keys):
            dataset, token = key.split("|", 1)
            local = local_test_rows.get(key, {})
            md = md_rows.get(key, {})
            writer.writerow(
                [
                    dataset,
                    match_mode,
                    path_suffix_parts,
                    token,
                    local.get("split", ""),
                    local.get("path", ""),
                    local.get("identity", ""),
                    local.get("image_id", ""),
                    local.get("source_csv", ""),
                    md.get("path", ""),
                    md.get("identity", ""),
                    md.get("image_id", ""),
                    md.get("source_csv", ""),
                ]
            )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description=(
            "Detect potential MegaDescriptor pretraining leakage by checking whether any "
            "TEST-split images in your local metadata also appear in the MegaDescriptor TRAIN metadata."
        )
    )
    parser.add_argument(
        "--ds",
        action="append",
        default=[],
        help="Dataset name; loads ./data/<ds>/processed_metadata.csv (repeatable).",
    )
    parser.add_argument(
        "--metadata-csv",
        action="append",
        default=[],
        help="Path to a local processed_metadata.csv file (repeatable).",
    )
    parser.add_argument(
        "--scan-data-dir",
        action="store_true",
        help="Scan ./data/*/processed_metadata.csv when no --ds/--metadata-csv are given.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(repo_root / "data"),
        help="Data directory to scan (default: ./data).",
    )
    parser.add_argument(
        "--split-col",
        default="split",
        help="Column name with split labels (default: split).",
    )
    parser.add_argument(
        "--test-split",
        nargs="+",
        default=["test"],
        help="Values treated as TEST split (case-insensitive). Everything else counts as TRAIN.",
    )
    parser.add_argument(
        "--dataset-col",
        default="dataset",
        help="Dataset column name (default: dataset).",
    )
    parser.add_argument(
        "--path-col",
        default="path",
        help="Path column name (default: path).",
    )
    parser.add_argument(
        "--identity-col",
        default="identity",
        help="Identity column name (default: identity).",
    )
    parser.add_argument(
        "--image-id-col",
        default="image_id",
        help="Image id column name (default: image_id).",
    )
    parser.add_argument(
        "--match-mode",
        choices=["suffix", "basename", "image_id"],
        default="suffix",
        help=(
            "How to match images between your metadata and combined_all.csv. "
            "'suffix' matches on the last N path parts (recommended), "
            "'basename' matches on filename only, "
            "'image_id' matches on image_id column."
        ),
    )
    parser.add_argument(
        "--path-suffix-parts",
        type=int,
        default=4,
        help="Number of path components to keep when --match-mode suffix is used (default: 4).",
    )
    parser.add_argument(
        "--md-train-csv",
        default="",
        help="Path to MegaDescriptor combined_all.csv (if omitted, uses cached download).",
    )
    parser.add_argument(
        "--md-train-url",
        default=MEGADESCRIPTOR_TRAIN_URL,
        help="URL for combined_all.csv (default: WildlifeDatasets/wildlife-tools).",
    )
    parser.add_argument(
        "--cache-path",
        default=str(repo_root / "data" / "external" / "megadescriptor" / "combined_all.csv"),
        help="Where to cache the downloaded combined_all.csv (default: ./data/external/megadescriptor/combined_all.csv).",
    )
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download combined_all.csv when missing (default: true).",
    )
    parser.add_argument(
        "--report-csv",
        default="",
        help="Write leaked rows to this CSV path (recommended for inspection).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=15,
        help="Number of leaked examples to print (default: 15).",
    )

    args = parser.parse_args()

    test_split_values = {v.lower() for v in args.test_split}

    # Collect metadata CSVs.
    metadata_paths: List[Path] = []
    for ds in args.ds:
        ds = _norm(ds)
        if not ds:
            continue
        candidates = [
            Path(args.data_dir) / ds / "processed_metadata.csv",
            Path(args.data_dir) / ds.lower() / "processed_metadata.csv",
        ]
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            print(f"[WARN] No processed_metadata.csv found for --ds {ds}", file=sys.stderr)
            continue
        metadata_paths.append(found)

    for p in args.metadata_csv:
        p = Path(p)
        if not p.exists():
            print(f"[WARN] --metadata-csv not found: {p}", file=sys.stderr)
            continue
        metadata_paths.append(p)

    if not metadata_paths:
        if not args.scan_data_dir and not args.ds and not args.metadata_csv:
            args.scan_data_dir = True
        if args.scan_data_dir:
            data_dir = Path(args.data_dir)
            metadata_paths = sorted(data_dir.glob("*/processed_metadata.csv"))

    if not metadata_paths:
        print(
            "No metadata CSVs found. Provide --ds <NAME> or --metadata-csv <PATH>, "
            "or use --scan-data-dir.",
            file=sys.stderr,
        )
        return 2

    # Load local splits.
    local_train_keys: Set[str] = set()
    local_test_keys: Set[str] = set()
    local_test_rows: Dict[str, Dict[str, str]] = {}

    for metadata_csv in metadata_paths:
        default_dataset = metadata_csv.parent.name
        try:
            tr, te, te_rows = _load_local_split_keys(
                metadata_csv,
                split_col=args.split_col,
                test_split_values=test_split_values,
                dataset_col=args.dataset_col,
                path_col=args.path_col,
                identity_col=args.identity_col,
                image_id_col=args.image_id_col,
                match_mode=args.match_mode,
                path_suffix_parts=args.path_suffix_parts,
                default_dataset=default_dataset,
            )
        except Exception as e:
            print(f"[WARN] Skipping {metadata_csv}: {e}", file=sys.stderr)
            continue

        local_train_keys |= tr
        local_test_keys |= te
        for k, v in te_rows.items():
            local_test_rows.setdefault(k, v)

    if not local_train_keys and not local_test_keys:
        print("No train/test keys loaded from the provided metadata.", file=sys.stderr)
        return 2

    local_all_keys = local_train_keys | local_test_keys

    # Load MegaDescriptor training keys relevant to our local data.
    md_csv = Path(args.md_train_csv) if args.md_train_csv else Path(args.cache_path)
    try:
        _ensure_md_train_csv(md_csv, url=args.md_train_url, allow_download=args.download)
    except Exception as e:
        print(f"Failed to obtain MegaDescriptor training metadata: {e}", file=sys.stderr)
        return 2

    try:
        md_train_keys, md_rows = _load_md_train_keys_for_local(
            md_csv,
            local_keys=local_all_keys,
            dataset_col="dataset",
            path_col="path",
            identity_col="identity",
            image_id_col="image_id",
            split_col="split",
            train_split_values={"train"},
            match_mode=args.match_mode,
            path_suffix_parts=args.path_suffix_parts,
        )
    except Exception as e:
        print(f"Failed reading MegaDescriptor metadata CSV: {e}", file=sys.stderr)
        return 2

    leak_keys = local_test_keys & md_train_keys

    # Summaries.
    datasets_local = sorted({_dataset_from_key(k) for k in local_all_keys})
    print(f"Checked datasets (local): {len(datasets_local)}")
    print(f"Local unique keys: {len(local_all_keys):,} (train={len(local_train_keys):,}, test={len(local_test_keys):,})")
    print(f"Matched MegaDescriptor-train keys: {len(md_train_keys):,} (within your local key universe)")
    print(f"LEAKED test keys: {len(leak_keys):,}")

    if leak_keys:
        # Per-dataset leakage counts.
        leak_by_ds: Dict[str, int] = {}
        test_by_ds: Dict[str, int] = {}
        for k in local_test_keys:
            ds = _dataset_from_key(k)
            test_by_ds[ds] = test_by_ds.get(ds, 0) + 1
        for k in leak_keys:
            ds = _dataset_from_key(k)
            leak_by_ds[ds] = leak_by_ds.get(ds, 0) + 1

        print("\nPer-dataset leakage (test ∩ MegaDescriptor-train):")
        for ds in sorted(leak_by_ds, key=lambda d: leak_by_ds[d], reverse=True):
            leak_n = leak_by_ds[ds]
            test_n = test_by_ds.get(ds, 0)
            frac = (leak_n / test_n * 100.0) if test_n else 0.0
            print(f"- {ds}: {leak_n:,} / {test_n:,} ({frac:.1f}%)")

        # Example rows.
        print("\nExamples:")
        for i, key in enumerate(sorted(leak_keys)[: max(0, args.max_examples)]):
            local = local_test_rows.get(key, {})
            md = md_rows.get(key, {})
            print(f"- {key}")
            if local.get("path"):
                print(f"  local: {local.get('path')} (id={local.get('identity','')})")
            if md.get("path"):
                print(f"  md   : {md.get('path')} (id={md.get('identity','')})")

    if args.report_csv:
        report_path = Path(args.report_csv)
        _write_leak_report_csv(
            report_path,
            leak_keys=leak_keys,
            local_test_rows=local_test_rows,
            md_rows=md_rows,
            match_mode=args.match_mode,
            path_suffix_parts=args.path_suffix_parts,
        )
        print(f"\nWrote leak report: {report_path}")

    # Exit code: non-zero when leakage is detected (useful for CI).
    return 1 if leak_keys else 0


if __name__ == "__main__":
    raise SystemExit(main())

