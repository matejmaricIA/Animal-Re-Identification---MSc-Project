"""Reproducible protocols and metrics for the CzechLynx benchmark.

Prediction functions never receive query identities. Fit/query separation, cache
provenance, and score definitions are explicit rather than inferred from filenames.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
SIGNALS = ("global", "global+fisher", "full-funnel")


def file_digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def normalized(matrix):
    x = np.asarray(matrix, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Features must be a finite two-dimensional matrix.")
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)



def validated_unit_matrix(matrix):
    """Validate unit (or zero) rows in bounded blocks without copying large caches."""
    x = np.asarray(matrix, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("Features must be a two-dimensional matrix.")
    for start in range(0, len(x), 256):
        block = x[start:start + 256]
        norms = np.linalg.norm(block, axis=1)
        if not np.isfinite(block).all() or not np.all((norms == 0) | np.isclose(norms, 1, atol=2e-4)):
            raise ValueError("Pre-normalized features must have finite unit or zero rows.")
    return x

def save_array_cache(path, matrix, ids, spec):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = list(map(str, ids))
    x = np.asarray(matrix, dtype=np.float32)
    if x.ndim != 2 or len(x) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError("Invalid cache shape or repeated IDs.")
    if not np.isfinite(x).all():
        raise ValueError("Cannot cache nonfinite features.")
    with path.with_name(path.name + ".tmp").open("wb") as stream:
        np.savez(stream, matrix=x, image_ids=np.asarray(ids, dtype=str))
    os.replace(path.with_name(path.name + ".tmp"), path)
    write_json(path.with_suffix(".json"), {
        "schema": SCHEMA_VERSION, "spec": spec, "fingerprint": fingerprint(spec),
        "sha256": file_digest(path), "shape": list(x.shape),
    })


def load_array_cache(path, ids, spec):
    path = Path(path)
    meta = path.with_suffix(".json")
    if not path.exists() and not meta.exists():
        return None
    if not path.exists() or not meta.exists():
        raise ValueError(f"Incomplete cache: {path}")
    info = json.loads(meta.read_text())
    if info.get("fingerprint") != fingerprint(spec) or info.get("sha256") != file_digest(path):
        raise ValueError(f"Incompatible or damaged cache: {path}")
    with np.load(path, allow_pickle=False) as cache:
        x = cache["matrix"]
        actual_ids = cache["image_ids"].tolist()
    if actual_ids != list(map(str, ids)) or list(x.shape) != info["shape"]:
        raise ValueError("Cache ID ordering/shape differs from requested data.")
    if x.ndim != 2 or len(x) != len(ids) or not np.isfinite(x).all():
        raise ValueError("Invalid cached features.")
    return x


def sample_encounters(frame, *, max_identities=None, max_images_per_identity=None, seed=42):
    """Seeded identity/encounter sampling; whole encounters may exceed the image cap.

    A shared encounter involving multiple identities is retained as a unit.
    """
    if max_identities is not None and max_identities < 1:
        raise ValueError("max_identities must be positive")
    if max_images_per_identity is not None and max_images_per_identity < 1:
        raise ValueError("max_images_per_identity must be positive")
    df = frame.sort_values("image_id").copy()
    rng = np.random.default_rng(seed)
    identities = np.asarray(sorted(df.identity.unique()), dtype=object)
    if max_identities is not None and len(identities) > max_identities:
        identities = rng.choice(identities, max_identities, replace=False)
        df = df[df.identity.isin(identities)]
    if max_images_per_identity is not None:
        chosen = set()
        for identity, group in df.groupby("identity", sort=True):
            events = np.asarray(sorted(group.encounter_id.unique()), dtype=object)
            rng.shuffle(events)
            n = 0
            for event in events:
                chosen.add(event)
                n += int((group.encounter_id == event).sum())
                if n >= max_images_per_identity:
                    break
        df = df[df.encounter_id.isin(chosen)]
    return df.sort_values("image_id").reset_index(drop=True)


def partition_overlaps(gallery, query):
    result = {}
    for key in ("image_id", "encounter_id", "image_sha256", "source_sha256"):
        if key in gallery and key in query:
            overlap = sorted(set(gallery[key].dropna()) & set(query[key].dropna()))
            if overlap:
                result[key] = overlap
    return result


def separate_partitions(gallery, query, *, overlap_policy="error"):
    """Keep original split columns, and record any explicit query-side exclusions."""
    overlaps = partition_overlaps(gallery, query)
    excluded = []
    if overlaps:
        if overlap_policy != "purge-query":
            raise ValueError(f"Gallery/query overlap: { {k: len(v) for k, v in overlaps.items()} }")
        bad = pd.Series(False, index=query.index)
        for column, values in overlaps.items():
            bad |= query[column].isin(values)
        excluded = query.loc[bad, "image_id"].astype(str).tolist()
        query = query.loc[~bad].copy()
    if gallery.empty or query.empty:
        raise ValueError("Gallery and query must remain nonempty.")
    return gallery.reset_index(drop=True), query.reset_index(drop=True), {
        "overlap_policy": overlap_policy,
        "overlap_counts": {k: len(v) for k, v in overlaps.items()},
        "excluded_query_ids": excluded,
    }


def make_protocol(frame, *, split="time_closed", stage="development", seed=42,
                  query_fraction=0.25, unknown_fraction=0.2, overlap_policy="error"):
    """Development uses only official train; test preserves official train/test.

    Development partitions by encounter across identities, including mixed-animal
    encounters. Missing-date encounters go to the fitting/gallery partition.
    """
    column = "split-" + split
    if column not in frame:
        raise ValueError(f"Missing official split column: {column}")
    if split == "geo_aware":
        raise ValueError("Geographic identities are disjoint; use a separately specified within-region protocol.")
    train = frame[frame[column] == "train"].copy()
    if stage == "test":
        gallery, query = train, frame[frame[column] == "test"].copy()
    elif stage == "development":
        if not 0 < query_fraction < 1 or not 0 <= unknown_fraction < 1:
            raise ValueError("Invalid development fractions.")
        identities = sorted(train.identity.unique())
        rng = np.random.default_rng(seed)
        n_unknown = max(1, int(len(identities) * unknown_fraction)) if len(identities) >= 5 and unknown_fraction else 0
        unknown = set(rng.choice(identities, n_unknown, replace=False).tolist())
        query_events = set(train.loc[train.identity.isin(unknown), "encounter_id"])
        # Each known identity contributes later encounters, where possible.
        for identity, group in train[~train.identity.isin(unknown)].groupby("identity", sort=True):
            events = group.groupby("encounter_id").timestamp.min().dropna().sort_values(kind="stable")
            events = events[~events.index.isin(query_events)]
            if len(events) >= 2:
                n = min(len(events) - 1, max(1, int(np.ceil(len(events) * query_fraction))))
                query_events.update(events.index[-n:])
        query = train[train.encounter_id.isin(query_events)].copy()
        gallery = train[~train.encounter_id.isin(query_events)].copy()
    else:
        raise ValueError(f"Unsupported stage: {stage}")
    gallery, query, audit = separate_partitions(gallery, query, overlap_policy=overlap_policy)
    audit.update({
        "stage": stage, "split_column": column, "seed": seed,
        "gallery_images": len(gallery), "query_images": len(query),
        "gallery_identities": int(gallery.identity.nunique()),
        "query_identities": int(query.identity.nunique()),
        "unknown_query_images": int((~query.identity.isin(gallery.identity)).sum()),
        "missing_dates": int(frame.timestamp.isna().sum()),
    })
    return gallery, query, audit


def rank_queries(query_ids, gallery_ids, query_global, gallery_global, *,
                 signal="global", query_fisher=None, gallery_fisher=None,
                 shortlist_size=50, rerank_size=10, depth=50, chunk_size=64,
                 verifier=None, features_normalized=False):
    """Yield rankings without access to either query or gallery identity labels.

    Global+Fisher uses clipped cosine mean. Full-funnel reranks only the leading
    candidates with the existing -inliers*log(1-tier2) rule, retaining the tail.
    """
    if signal not in SIGNALS or min(shortlist_size, depth, chunk_size) < 1:
        raise ValueError("Invalid ranking configuration.")
    if signal == "full-funnel" and (verifier is None or rerank_size < 1):
        raise ValueError("Full-funnel requires a verifier and positive rerank_size.")
    qids, gids = list(map(str, query_ids)), list(map(str, gallery_ids))
    if not gids or len(set(qids)) != len(qids) or len(set(gids)) != len(gids):
        raise ValueError("Empty gallery or repeated image IDs.")
    validate = validated_unit_matrix if features_normalized else normalized
    qg, gg = validate(query_global), validate(gallery_global)
    if len(qg) != len(qids) or len(gg) != len(gids) or qg.shape[1] != gg.shape[1]:
        raise ValueError("Global matrix/ID mismatch.")
    qf = gf = None
    if signal != "global":
        if query_fisher is None or gallery_fisher is None:
            raise ValueError("Fisher matrices are required.")
        qf, gf = validate(query_fisher), validate(gallery_fisher)
        if len(qf) != len(qids) or len(gf) != len(gids) or qf.shape[1] != gf.shape[1]:
            raise ValueError("Fisher matrix/ID mismatch.")
    for start in range(0, len(qids), chunk_size):
        global_block = qg[start:start + chunk_size] @ gg.T
        fisher_block = qf[start:start + chunk_size] @ gf.T if qf is not None else None
        for offset, raw_global in enumerate(global_block):
            qi = start + offset
            valid = np.asarray([gid != qids[qi] for gid in gids])
            gscore = np.where(valid, raw_global, -np.inf)
            gorder = np.argsort(-gscore, kind="stable")
            gorder = gorder[valid[gorder]]
            if signal == "global":
                order, scores = gorder, gscore
                candidate_order = gorder[:shortlist_size]
            else:
                raw_fisher = fisher_block[offset]
                fscore = np.where(valid, raw_fisher, -np.inf)
                forder = np.argsort(-fscore, kind="stable")
                forder = forder[valid[forder]]
                candidates = sorted(set(gorder[:shortlist_size]) | set(forder[:shortlist_size]))
                scores = (np.clip(raw_global, 0, 1) + np.clip(raw_fisher, 0, 1)) / 2
                scores = np.where(valid, scores, -np.inf)
                order = np.asarray(sorted(candidates, key=lambda j: (-float(scores[j]), gids[j])), dtype=int)
                candidate_order = order.copy()
            verified = []
            head_scores = {}
            if signal == "full-funnel":
                head = order[:rerank_size].tolist()
                for j in head:
                    evidence = dict(verifier(qids[qi], gids[j]))
                    attempted = bool(evidence.get("attempted", False))
                    n = max(0, int(evidence.get("n_inliers", 0)))
                    base = float(scores[j])
                    power = float(-n * np.log(np.clip(1 - base, 1e-12, 1.0))) if attempted else None
                    verified.append({"image_id": gids[j], "attempted": attempted,
                                     "n_inliers": n, "reason": evidence.get("reason", "")})
                    head_scores[j] = power
                # Skipped evidence retains its position; only attempted entries rerank.
                positions = [p for p, j in enumerate(head) if head_scores[j] is not None]
                attempted_order = sorted((head[p] for p in positions),
                                         key=lambda j: (-head_scores[j], -float(scores[j]), gids[j]))
                for position, j in zip(positions, attempted_order):
                    head[position] = j
                order = np.asarray(head + order[len(head):].tolist(), dtype=int)
            ranked = order[:depth]
            best = int(ranked[0]) if len(ranked) else None
            confidence = None if best is None else float(scores[best])
            if signal == "full-funnel" and best is not None and head_scores.get(best) is not None:
                confidence = head_scores[best]
            yield {
                "query_image_id": qids[qi], "candidate_ids": [gids[j] for j in ranked],
                "candidate_scores": [float(scores[j]) for j in ranked],
                "shortlist_ids": [gids[j] for j in candidate_order],
                "confidence": confidence, "verification": verified,
            }


def retrieval_metrics(rankings, query, gallery, *, depth=50, threshold=None):
    """AP@K denominator is min(number of relevant gallery images, K).

    Unknown queries have undefined AP and are excluded from known-query accuracy,
    while their counts and false acceptance are reported explicitly.
    """
    qrows = query.set_index("image_id")
    labels = gallery.set_index("image_id").identity.astype(str).to_dict()
    label_counts = gallery.identity.astype(str).value_counts().to_dict()
    records = []
    if len(rankings) != len(query) or set(r["query_image_id"] for r in rankings) != set(query.image_id):
        raise ValueError("Predictions do not cover the query partition exactly.")
    for pred in rankings:
        qid = pred["query_image_id"]
        identity = str(qrows.loc[qid, "identity"])
        ids = pred["candidate_ids"][:depth]
        if len(set(ids)) != len(ids) or any(i not in labels or i == qid for i in ids):
            raise ValueError("Invalid candidate IDs.")
        positive_count = label_counts.get(identity, 0) - int(labels.get(qid) == identity)
        known = positive_count > 0
        relevance = np.asarray([labels[i] == identity for i in ids], dtype=float)
        ap = None
        if known:
            precision = np.cumsum(relevance) / np.arange(1, len(ids) + 1)
            ap = float(np.sum(precision * relevance) / min(positive_count, depth))
        accepted = bool(pred["confidence"] is not None and pred["confidence"] >= threshold) if threshold is not None else None
        records.append({
            **pred, "identity": identity, "known": known,
            "top1_identity": labels[ids[0]] if ids else None,
            "top1_correct": bool(len(relevance) and relevance[0]),
            "top5_correct": bool(np.any(relevance[:5])),
            "shortlist_hit": bool(any(labels[i] == identity for i in pred["shortlist_ids"])),
            "ap_at_k": ap, "accepted": accepted,
            "encounter_id": str(qrows.loc[qid, "encounter_id"]),
        })
    known_rows = [r for r in records if r["known"]]
    unknown_rows = [r for r in records if not r["known"]]
    mean = lambda rows, key: float(np.mean([r[key] for r in rows])) if rows else None
    attempts = [v for r in records for v in r["verification"]]
    metrics = {
        "queries": len(records), "gallery": len(gallery), "known_queries": len(known_rows),
        "unknown_queries": len(unknown_rows), "top1": mean(known_rows, "top1_correct"),
        "top5": mean(known_rows, "top5_correct"), "map_at_k": mean(known_rows, "ap_at_k"),
        "retrieval_depth": depth, "ap_denominator": "min(relevant_gallery_images, K)",
        "shortlist_recall": mean(known_rows, "shortlist_hit"),
        "gv_requested_pairs": len(attempts),
        "gv_attempted_pairs": sum(v["attempted"] for v in attempts),
        "gv_skipped_pairs": sum(not v["attempted"] for v in attempts),
        "threshold": threshold,
    }
    if known_rows:
        per_id = pd.DataFrame(known_rows).groupby("identity").top1_correct.mean()
        metrics["macro_identity_top1"] = float(per_id.mean())
        per_encounter = pd.DataFrame(known_rows).groupby("encounter_id").top1_correct.mean()
        metrics["macro_encounter_top1"] = float(per_encounter.mean())
    if threshold is not None:
        metrics["unknown_false_accept_rate"] = mean(unknown_rows, "accepted")
        metrics["known_false_reject_rate"] = 1 - mean(known_rows, "accepted") if known_rows else None
        metrics["known_correct_accept_rate"] = (
            float(np.mean([r["accepted"] and r["top1_correct"] for r in known_rows]))
            if known_rows else None)
    return metrics, records


def rejection_threshold(records, *, target_false_accept_rate=0.05):
    """Development-only threshold; caller must provide provenance when using it."""
    if not 0 <= target_false_accept_rate < 1:
        raise ValueError("Invalid false acceptance target.")
    unknown = sorted(float(r["confidence"]) for r in records
                     if not r["known"] and r["confidence"] is not None)
    if not unknown:
        raise ValueError("Threshold selection requires unknown development queries.")
    allowed = int(np.floor(target_false_accept_rate * len(unknown)))
    boundary = unknown[len(unknown) - allowed - 1]
    return float(np.nextafter(boundary, np.inf))


def clustering_metrics(ids, labels, identities):
    from reid_demo.eval import standard_metrics
    ids, labels = list(ids), np.asarray(labels, dtype=int)
    if len(ids) != len(labels) or len(set(ids)) != len(ids):
        raise ValueError("Invalid clustering assignment IDs.")
    truth = [str(identities[i]) for i in ids]
    # Noise represents unresolved separate observations, not one giant individual.
    mapped = labels.copy()
    next_label = max([int(v) for v in labels if v >= 0], default=-1) + 1
    for i in np.flatnonzero(labels < 0):
        mapped[i] = next_label
        next_label += 1
    groups = {}
    appearances = {}
    for identity, label in zip(truth, labels):
        if label >= 0:
            groups.setdefault(int(label), set()).add(identity)
            appearances.setdefault(identity, set()).add(int(label))
    return {
        **standard_metrics(truth, mapped.tolist()),
        "images": len(ids), "true_individuals": len(set(truth)),
        "found_clusters": len(groups), "unassigned_images": int(np.sum(labels < 0)),
        "merge_errors": sum(len(g) > 1 for g in groups.values()),
        "split_errors": sum(len(g) > 1 for g in appearances.values()),
        "noise_metric_policy": "one distinct predicted label per unassigned image",
    }
