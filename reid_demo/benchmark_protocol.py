"""Versioned, identity-disjoint development protocols. No model dependencies."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reid_demo.benchmark import fingerprint

VERSION = "identity_pools_v1"
POOLS = ("fitting", "calibration", "selection")
LINKS = ("encounter_id", "source_sha256", "image_sha256")


def _components(frame, node_column):
    parent = {str(v): str(v) for v in frame[node_column].unique()}

    def root(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for column in LINKS:
        first = {}
        for node, value in zip(frame[node_column].astype(str), frame[column]):
            if pd.isna(value) or not str(value).strip():
                continue
            value = str(value)
            if value in first:
                a, b = root(node), root(first[value])
                parent[max(a, b)] = min(a, b)
            else:
                first[value] = node
    groups = defaultdict(list)
    for node in sorted(parent):
        groups[root(node)].append(node)
    return sorted(groups.values(), key=lambda group: tuple(group))


def _inventory(frame):
    dates = pd.to_datetime(frame.timestamp, errors="coerce", utc=True)
    return {
        "images": len(frame), "identities": int(frame.identity.nunique()),
        "encounters": int(frame.encounter_id.nunique()),
        "sources": {str(k): int(v) for k, v in frame.groupby("source").size().items()},
        "missing_dates": int(dates.isna().sum()),
        "first_date": dates.min().isoformat() if dates.notna().any() else None,
        "last_date": dates.max().isoformat() if dates.notna().any() else None,
    }


def _assign_pools(train, groups, fractions, seed):
    # Largest groups first; a seeded tie order is stable under metadata row shuffles.
    rng = np.random.default_rng(seed)
    tie_order = rng.permutation(len(groups))
    order = sorted(tie_order, key=lambda i: -len(groups[i]))
    sources = sorted(train.source.unique())
    vectors = []
    for group in groups:
        rows = train[train.identity.isin(group)]
        vectors.append(np.asarray([len(group)] + [
            rows.loc[rows.source == source, "identity"].nunique() for source in sources], dtype=float))
    total = np.sum(vectors, axis=0)
    targets = np.asarray(fractions)[:, None] * total
    counts = np.zeros_like(targets)
    assigned = {pool: [] for pool in POOLS}
    for index in order:
        vector = vectors[index]
        delta = np.sum(((counts + vector - targets) ** 2 - (counts - targets) ** 2)
                       / np.maximum(targets, 1), axis=1)
        role = int(np.argmin(delta))
        assigned[POOLS[role]].append(groups[index])
        counts[role] += vector
    return {pool: sorted(value) for pool, value in assigned.items()}


def _evaluation_partition(rows, groups, *, seed, unknown_fraction, query_fraction):
    dates = pd.to_datetime(rows.timestamp, errors="coerce", utc=True)
    # Compute atomic encounter/duplicate groups before excluding missing dates.
    units = _components(rows, "encounter_id")
    eligible = rows.loc[dates.notna()].copy()
    eligible["_date"] = dates[dates.notna()]
    excluded = [{"image_id": str(rid), "reason": "missing_or_invalid_timestamp"}
                for rid in sorted(rows.loc[dates.isna(), "image_id"])]
    eligible_groups = [g for g in groups if eligible.identity.isin(g).any()]
    if len(eligible_groups) < 2:
        raise ValueError("Each evaluation pool needs at least two dated identity groups.")
    rng = np.random.default_rng(seed)
    count = min(len(eligible_groups) - 1, max(1, round(len(eligible_groups) * unknown_fraction)))
    unknown = {ident for i in rng.permutation(len(eligible_groups))[:count]
               for ident in eligible_groups[i]}
    known = eligible[~eligible.identity.isin(unknown)]
    unit_rows = {}
    by_identity = defaultdict(set)
    bounds = {}
    for index, events in enumerate(units):
        block = known[known.encounter_id.isin(events)]
        if block.empty:
            continue
        unit_rows[index] = block
        for identity, part in block.groupby("identity"):
            by_identity[identity].add(index)
            bounds[index, identity] = (part["_date"].min(), part["_date"].max())
    targets = {identity: min(len(ids) - 1, max(1, int(np.ceil(len(ids) * query_fraction))))
               for identity, ids in by_identity.items()}
    query_units = set()
    order = sorted(unit_rows, key=lambda i: (
        -unit_rows[i]["_date"].min().value, tuple(sorted(unit_rows[i].image_id))))
    for index in order:
        identities = sorted(unit_rows[index].identity.unique())
        if not any(len(by_identity[i] & query_units) < targets[i] for i in identities):
            continue
        # Never remove the final gallery unit or leave a later unit in the gallery.
        allowed = True
        for identity in identities:
            remaining = by_identity[identity] - query_units - {index}
            if not remaining or any(bounds[j, identity][1] > bounds[index, identity][0]
                                    for j in remaining):
                allowed = False
                break
        if allowed:
            query_units.add(index)
    query_ids = set(eligible.loc[eligible.identity.isin(unknown), "image_id"])
    for index in query_units:
        query_ids.update(unit_rows[index].image_id)
    query = eligible[eligible.image_id.isin(query_ids)]
    gallery = eligible[~eligible.image_id.isin(query_ids)]
    known_count = int(query.identity.isin(gallery.identity).sum())
    unknown_count = len(query) - known_count
    if gallery.empty or not known_count or not unknown_count:
        raise ValueError("Cannot form nonempty gallery, known queries and unknown queries; "
                         "increase pool coverage or choose a different protocol seed.")
    for column in LINKS:
        overlap = set(gallery[column].dropna()) & set(query[column].dropna())
        overlap.discard("")
        if overlap:
            raise ValueError(f"Atomic partition violation: {column}")
    return {
        "gallery_ids": sorted(gallery.image_id), "query_ids": sorted(query.image_id),
        "excluded": excluded, "unknown_identities": sorted(unknown),
        "gallery": _inventory(gallery), "query": _inventory(query),
        "known_queries": known_count, "unknown_queries": unknown_count,
        "known_identities_without_queries": sorted(set(gallery.identity) - set(query.identity)),
    }


def build_protocol(frame, metadata_sha256, *, seed=42, fitting_fraction=0.6,
                   calibration_fraction=0.2, unknown_fraction=0.2, query_fraction=0.25,
                   split="time_open"):
    if split != "time_open":
        raise ValueError("Identity-pool protocols currently support time_open only.")
    fractions = [float(fitting_fraction), float(calibration_fraction),
                 1 - float(fitting_fraction) - float(calibration_fraction)]
    if (not all(np.isfinite(f) and 0 < f < 1 for f in fractions)
            or not 0 < unknown_fraction < 1 or not 0 < query_fraction < 1):
        raise ValueError("Invalid protocol fractions.")
    required = {"image_id", "identity", "source", "timestamp", "split-time_open", *LINKS}
    if not required.issubset(frame) or frame.image_id.duplicated().any():
        raise ValueError("Protocol requires unique prepared image IDs and provenance columns.")
    train = frame[frame["split-time_open"] == "train"].sort_values("image_id").copy()
    for column in ("image_id", "identity", "source", "encounter_id"):
        if train[column].isna().any() or (train[column].astype(str).str.strip() == "").any():
            raise ValueError(f"Missing required training metadata: {column}")
        train[column] = train[column].astype(str)
    groups = _components(train, "identity")
    if len(groups) < 5:
        raise ValueError("At least five independent identity groups are needed for three pools.")
    assigned = _assign_pools(train, groups, fractions, seed)
    pools = {}
    for offset, pool in enumerate(POOLS):
        identities = {i for group in assigned[pool] for i in group}
        rows = train[train.identity.isin(identities)]
        if rows.empty:
            raise ValueError(f"Empty {pool} pool; group constraints prevent this split.")
        entry = {"identity_groups": assigned[pool], "image_ids": sorted(rows.image_id),
                 "inventory": _inventory(rows)}
        if pool != "fitting":
            entry.update(_evaluation_partition(
                rows, assigned[pool], seed=seed + offset, unknown_fraction=unknown_fraction,
                query_fraction=query_fraction))
        pools[pool] = entry
    result = {
        "schema": 1, "version": VERSION, "metadata_sha256": metadata_sha256,
        "settings": {"seed": seed, "split": split, "fitting_fraction": fitting_fraction,
                     "calibration_fraction": calibration_fraction, "unknown_fraction": unknown_fraction,
                     "query_fraction": query_fraction},
        "requested_pool_fractions": dict(zip(POOLS, fractions)),
        "training_inventory": _inventory(train), "identity_groups": groups, "pools": pools,
        "scope": "official train only; independent identities within this experiment; upstream exposure unknown",
    }
    result["fingerprint"] = fingerprint(result)
    return result


def load_protocol(path, frame, metadata_sha256):
    data = json.loads(Path(path).read_text())
    payload = {k: v for k, v in data.items() if k != "fingerprint"}
    if (data.get("version") != VERSION or data.get("metadata_sha256") != metadata_sha256
            or data.get("fingerprint") != fingerprint(payload)):
        raise ValueError("Incompatible or damaged protocol plan / metadata.")
    # Reconstruct to reject re-signed membership edits, omitted exclusions and stale inventories.
    expected = build_protocol(frame, metadata_sha256, **data["settings"])
    if data != expected:
        raise ValueError("Protocol memberships or audit do not match deterministic construction.")
    return data


def protocol_frames(plan, frame, partition):
    if partition not in ("calibration", "selection"):
        raise ValueError("Choose calibration or selection.")
    indexed = frame.set_index("image_id", drop=False)
    pool = plan["pools"][partition]
    fitting = indexed.loc[plan["pools"]["fitting"]["image_ids"]].reset_index(drop=True)
    gallery = indexed.loc[pool["gallery_ids"]].reset_index(drop=True)
    query = indexed.loc[pool["query_ids"]].reset_index(drop=True)
    audit = {k: pool[k] for k in ("gallery", "query", "excluded", "known_queries", "unknown_queries")}
    audit.update(partition=partition, protocol_fingerprint=plan["fingerprint"],
                 unknown_query_images=pool["unknown_queries"], excluded_query_ids=[
                     row["image_id"] for row in pool["excluded"]])
    return fitting, gallery, query, audit


def protocol_binding(plan):
    return {"version": plan["version"], "fingerprint": plan["fingerprint"],
            "fitting_ids": plan["pools"]["fitting"]["image_ids"],
            "calibration_gallery_ids": plan["pools"]["calibration"]["gallery_ids"],
            "calibration_query_ids": plan["pools"]["calibration"]["query_ids"]}


def validate_calibration(data, plan, signals):
    expected = protocol_binding(plan)
    if (data.get("stage") != "development" or data.get("partition") != "calibration"
            or data.get("protocol") != expected
            or data.get("gallery_ids") != expected["calibration_gallery_ids"]
            or data.get("query_ids") != expected["calibration_query_ids"]):
        raise ValueError("Thresholds must come from this protocol's calibration partition.")
    thresholds = data.get("thresholds", {})
    if (not isinstance(thresholds, dict) or set(signals) - thresholds.keys()
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)
                   for v in thresholds.values())):
        raise ValueError("Calibration thresholds are missing requested signals or are nonfinite.")
    return data

