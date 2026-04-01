#!/usr/bin/env python3
"""Build deterministic 80/10/10 splits for WildlifeReID-10k v7.

Reads the local WildlifeReID-10k metadata.csv and writes a new CSV that includes:
- an "open-set enabled" split column:
    split_open: train | val | test_known | test_new
- a "closed-set" view:
    split_closed: train | val | test | ignore

Key properties:
- Open-set identities (test_new) are selected per dataset as a fraction of identities.
- Remaining identities are split per-identity, group-aware:
    - Time-aware if timestamps are sufficiently present (oldest->train, middle->val, newest->test_known)
    - Similarity-aware otherwise, using cluster_id groups (no group crosses splits)
  Missing timestamps are assigned to train to avoid leakage.
  Missing cluster_id values are treated as singleton groups (per-image).
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_INPUT = "data/wildlifedatasets/wildlifereid-10k/versions/7/metadata.csv"
DEFAULT_OUTPUT = "data/wreid10k_splits_80_10_10.csv"


@dataclass(frozen=True)
class Ratios:
    train: float
    val: float
    test: float


def _sha1_hex(value: str, n: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:n]


def _norm_str(value: object) -> str:
    return str(value).strip()


def _parse_dates(series: pd.Series) -> pd.Series:
    # Parse dates; invalid formats become NaT.
    return pd.to_datetime(series, errors="coerce", utc=True)


def _allocate_counts(n: int, ratios: Ratios) -> tuple[int, int, int]:
    """Allocate (n_train, n_val, n_test) with best-effort 80/10/10.

    Ensures:
    - if n == 1: all train
    - if n >= 2: at least 1 in train and 1 in test
    - val is non-empty only when n is large enough
    """
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 0, 1
    if n == 3:
        # Prioritize a non-empty test set; val would eliminate it.
        return 2, 0, 1

    n_train = int(np.floor(ratios.train * n))
    n_val = int(np.floor(ratios.val * n))

    # Encourage at least one val when possible.
    if n >= 4:
        n_val = max(1, n_val)

    # Enforce at least one train.
    n_train = max(1, n_train)

    n_test = n - n_train - n_val
    if n_test < 1:
        # Reduce train first, then val, to make room for test.
        need = 1 - n_test
        take = min(need, max(0, n_train - 1))
        n_train -= take
        need -= take
        if need > 0 and n_val > 0:
            take2 = min(need, n_val)
            n_val -= take2
            need -= take2
        n_test = n - n_train - n_val

    # Final guard.
    if n_test < 1:
        n_test = 1
        if n_train > 1:
            n_train -= 1
        elif n_val > 0:
            n_val -= 1

    return int(n_train), int(n_val), int(n_test)


def _choose_open_set_identities(
    identities: list[str],
    rng: np.random.Generator,
    frac: float,
    min_ids: int,
) -> set[str]:
    if not identities:
        return set()
    n_total = len(identities)
    n_open = int(np.round(float(frac) * n_total))
    n_open = max(int(min_ids), n_open)
    n_open = min(n_open, n_total)
    if n_open <= 0:
        return set()
    picked = rng.choice(np.array(identities, dtype=object), size=n_open, replace=False)
    return {str(x) for x in picked.tolist()}


def _split_time_aware_identity(group: pd.DataFrame, ratios: Ratios) -> pd.Series:
    """Return split labels (train/val/test_known) for one identity group."""
    group = group.copy()
    if "_date_parsed" not in group.columns:
        raise ValueError("Missing required column _date_parsed for time-aware split.")

    # Missing/invalid timestamps are assigned to train (safe).
    split = pd.Series(index=group.index, data="train", dtype=str)
    valid = group[~group["_date_parsed"].isna()].copy()
    if valid.empty:
        return split

    valid["_day"] = valid["_date_parsed"].dt.date
    days = sorted({d for d in valid["_day"].tolist() if d is not None})
    n_train, n_val, _n_test = _allocate_counts(len(days), ratios)
    train_days = set(days[:n_train])
    val_days = set(days[n_train : n_train + n_val])
    test_days = set(days[n_train + n_val :])

    idx_val = valid.index[valid["_day"].isin(val_days)]
    idx_test = valid.index[valid["_day"].isin(test_days)]
    split.loc[idx_val] = "val"
    split.loc[idx_test] = "test_known"
    return split


def _split_cluster_aware_identity(
    group: pd.DataFrame,
    *,
    ratios: Ratios,
    rng: np.random.Generator,
) -> pd.Series:
    """Return split labels (train/val/test_known) for one identity group."""
    group = group.copy()

    def _cluster_token(row: pd.Series) -> str:
        cid = row.get("cluster_id", None)
        if cid is None:
            return f"singleton:{row['image_id']}"
        cid_str = str(cid).strip()
        if not cid_str or cid_str.lower() in {"nan", "none"}:
            return f"singleton:{row['image_id']}"
        return cid_str

    group["_cluster"] = group.apply(_cluster_token, axis=1)
    clusters = sorted(set(group["_cluster"].astype(str).tolist()))
    n = len(clusters)
    if n <= 1:
        return pd.Series(index=group.index, data="train", dtype=str)

    n_train, n_val, _n_test = _allocate_counts(n, ratios)
    perm = rng.permutation(np.array(clusters, dtype=object)).tolist()
    train_clusters = set(perm[:n_train])
    val_clusters = set(perm[n_train : n_train + n_val])
    test_clusters = set(perm[n_train + n_val :])

    split = pd.Series(index=group.index, data="train", dtype=str)
    split.loc[group.index[group["_cluster"].isin(val_clusters)]] = "val"
    split.loc[group.index[group["_cluster"].isin(test_clusters)]] = "test_known"
    return split


def _per_dataset_strategy(sub: pd.DataFrame, date_threshold: float) -> str:
    parsed = sub.get("_date_parsed", None)
    if parsed is None:
        parsed = _parse_dates(sub["date"])
    frac = float((~parsed.isna()).mean()) if len(parsed) else 0.0
    return "time" if frac >= float(date_threshold) else "cluster"


def _multi_day_identity_frac(sub: pd.DataFrame) -> float:
    parsed = sub.get("_date_parsed", None)
    if parsed is None:
        parsed = _parse_dates(sub["date"])
    tmp = sub.copy()
    tmp["_date_parsed"] = parsed
    tmp = tmp[~tmp["_date_parsed"].isna()].copy()
    if tmp.empty:
        return 0.0
    tmp["_day"] = tmp["_date_parsed"].dt.date
    counts = tmp.groupby("identity")["_day"].nunique()
    return float((counts >= 2).mean()) if len(counts) else 0.0


def _summarize_dataset(ds: str, split_open: pd.Series) -> None:
    counts = split_open.value_counts().to_dict()
    keys = ["train", "val", "test_known", "test_new"]
    row = ", ".join([f"{k}={int(counts.get(k, 0))}" for k in keys])
    print(f"{ds}: {row}")


def build_splits(
    input_csv: Path,
    output_csv: Path,
    *,
    ratios: Ratios,
    open_set_id_frac: float,
    min_open_set_ids: int,
    seed: int,
    date_threshold: float,
    min_multi_day_frac: float,
) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")

    df = pd.read_csv(input_csv, low_memory=False, dtype={"identity": str, "dataset": str, "path": str})
    required = {"identity", "dataset", "path", "date", "cluster_id"}
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    df = df.copy()
    df["identity"] = df["identity"].astype(str)
    df["dataset"] = df["dataset"].astype(str)
    df["path"] = df["path"].astype(str)

    # Stable, file-safe ID derived from dataset+path.
    df["image_id"] = [
        _sha1_hex(f"{_norm_str(ds)}|{_norm_str(p)}", n=16)
        for ds, p in zip(df["dataset"].tolist(), df["path"].tolist())
    ]

    rng = np.random.default_rng(int(seed))

    split_open_out = pd.Series(index=df.index, data="train", dtype=str)
    strategy_out = pd.Series(index=df.index, data="", dtype=str)
    is_open_identity_out = pd.Series(index=df.index, data=False, dtype=bool)

    for ds in sorted(df["dataset"].unique().tolist()):
        sub = df[df["dataset"] == ds].copy()
        sub["_date_parsed"] = _parse_dates(sub["date"])
        identities = sorted(sub["identity"].astype(str).unique().tolist())

        ds_rng = np.random.default_rng(int(seed) + int(_sha1_hex(ds, n=8), 16) % (2**31 - 1))
        open_ids = _choose_open_set_identities(
            identities=identities,
            rng=ds_rng,
            frac=open_set_id_frac,
            min_ids=min_open_set_ids,
        )

        open_mask = sub["identity"].astype(str).isin(open_ids)
        open_idx = sub.index[open_mask].tolist()
        split_open_out.loc[open_idx] = "test_new"
        is_open_identity_out.loc[open_idx] = True

        known = sub[~open_mask]
        if known.empty:
            strategy_out.loc[sub.index] = "open_only"
            _summarize_dataset(ds, split_open_out.loc[sub.index])
            continue

        # Prefer time-aware when timestamps are present and there are meaningful multi-day identities.
        base_strategy = _per_dataset_strategy(known, date_threshold=date_threshold)
        if base_strategy == "time":
            frac_multi = _multi_day_identity_frac(known)
            strategy = "time" if frac_multi >= float(min_multi_day_frac) else "cluster"
        else:
            strategy = "cluster"
        strategy_out.loc[sub.index] = strategy

        # Split the known identities.
        if strategy == "time":
            for ident, group in known.groupby("identity", sort=True):
                labels = _split_time_aware_identity(group, ratios)
                split_open_out.loc[group.index] = labels
        else:
            for ident, group in known.groupby("identity", sort=True):
                # Per-identity RNG derived from dataset+identity for determinism.
                ident_seed = int(seed) + int(_sha1_hex(f"{ds}|{ident}", n=8), 16) % (2**31 - 1)
                ident_rng = np.random.default_rng(ident_seed)
                labels = _split_cluster_aware_identity(group, ratios=ratios, rng=ident_rng)
                split_open_out.loc[group.index] = labels

        _summarize_dataset(ds, split_open_out.loc[sub.index])

    split_closed_out = split_open_out.replace(
        {
            "test_known": "test",
            "test_new": "ignore",
        }
    )

    out = df.copy()
    out["split_open"] = split_open_out.astype(str)
    out["split_closed"] = split_closed_out.astype(str)
    out["split_strategy"] = strategy_out.astype(str)
    out["is_open_set_identity"] = is_open_identity_out.astype(bool)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def _validate_ratios(train: float, val: float, test: float) -> Ratios:
    total = float(train) + float(val) + float(test)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Invalid split ratios.")
    train_n = float(train) / total
    val_n = float(val) / total
    test_n = float(test) / total
    return Ratios(train=train_n, val=val_n, test=test_n)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic group-aware 80/10/10 splits for WildlifeReID-10k v7."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help=f"Input metadata CSV (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output CSV (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio (default: 0.8)")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Val ratio (default: 0.1)")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio (default: 0.1)")
    parser.add_argument("--open-id-frac", type=float, default=0.05, help="Open-set identity fraction per dataset (default: 0.05)")
    parser.add_argument("--min-open-ids", type=int, default=0, help="Minimum open-set identities per dataset (default: 0)")
    parser.add_argument("--seed", type=int, default=666, help="Random seed (default: 666)")
    parser.add_argument("--date-threshold", type=float, default=0.9, help="Use time-aware split if date non-null fraction >= threshold (default: 0.9)")
    parser.add_argument(
        "--min-multi-day-frac",
        type=float,
        default=0.05,
        help="Require at least this fraction of identities with >=2 unique days to use time-aware (default: 0.05)",
    )
    args = parser.parse_args()

    ratios = _validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    out = build_splits(
        input_csv=Path(args.input),
        output_csv=Path(args.output),
        ratios=ratios,
        open_set_id_frac=float(args.open_id_frac),
        min_open_set_ids=int(args.min_open_ids),
        seed=int(args.seed),
        date_threshold=float(args.date_threshold),
        min_multi_day_frac=float(args.min_multi_day_frac),
    )

    print(f"Wrote {len(out)} rows to {args.output}")
    print("split_open counts:", out["split_open"].value_counts().to_dict())
    print("split_closed counts:", out["split_closed"].value_counts().to_dict())


if __name__ == "__main__":
    main()
