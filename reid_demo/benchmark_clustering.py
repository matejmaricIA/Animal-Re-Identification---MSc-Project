"""Cache-only, full-collection CzechLynx clustering and count diagnostics.

Inference accepts IDs and vectors, never identity truth or a requested count.
Calibration profiles are exploratory; selection requires an exported frozen choice.
No detection store, neural model, or source cache is written by this module.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from reid_demo.benchmark import file_digest, fingerprint, write_json
from reid_demo.benchmark_protocol import load_protocol, protocol_binding


ROOT = Path(__file__).resolve().parents[1]
VERSION = "full_collection_global_controls_v1"
CONFIG_KIND = "czechlynx_cluster_config"
ALGORITHMS = ("legacy_neighbour", "threshold", "consistent")
DEFAULT_BUDGETS = (15, 50)
DEFAULT_THRESHOLDS = (0.35, 0.45, 0.55, 0.65, 0.75, 0.85)
SCORE_DECIMALS = 7
NOTES = [
    "Development collection discovery; upstream pretrained-model exposure remains unresolved.",
    "Component counts are provisional photographed-image group counts, not park abundance.",
    "Singleton proposals are not confirmed new animals. Unresolved images are not counted as animals.",
    "Unknown flank metadata does not establish opposite-flank linkage.",
    "Encounters can contain multiple animals; encounter IDs never impose same-animal links.",
    "Near-monochrome is a colour proxy, not verified night/infrared.",
    "Calibration threshold sensitivity is not a confidence interval or a guaranteed count bound.",
    "Image pairs and repeated encounter frames are correlated.",
]


@dataclass
class Collection:
    frame: pd.DataFrame
    plan: dict
    reference: dict
    binding: dict
    provenance: dict


@dataclass
class ClusteringResult:
    image_ids: list
    labels: np.ndarray
    evidence: list
    audit: dict


def implementation_binding():
    names = (
        "reid_demo/benchmark_clustering.py", "tools/evaluate_czechlynx_clusters.py",
        "reid_demo/benchmark.py", "reid_demo/benchmark_protocol.py",
        "reid_demo/cluster.py", "reid_demo/eval.py",
        "reid_demo/benchmark_diagnostics.py", "reid_demo/benchmark_masks.py",
    )
    versions = {}
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "networkx",
                 "igraph", "leidenalg", "Pillow", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"version": VERSION, "code": {name: file_digest(ROOT / name) for name in names},
            "dependencies": versions, "cosine_rounding_decimals": SCORE_DECIMALS}


def _read_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _require_hash(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"Invalid SHA-256 for {name}.")
    return value


def load_collection(metadata, protocol_plan, reference_dir, partition):
    """Reconstruct all eligible pool membership and validate its completed reference."""
    if partition not in ("calibration", "selection"):
        raise ValueError("Choose calibration or selection.")
    metadata, reference_dir = Path(metadata), Path(reference_dir)
    frame = pd.read_csv(metadata, dtype={"image_id": str, "identity": str})
    required = {"image_id", "identity", "path", "image_sha256", "source_sha256",
                "source", "encounter_id", "timestamp", "input_mode", "split-time_open"}
    if (not required.issubset(frame) or frame.empty or frame.image_id.isna().any()
            or frame.image_id.duplicated().any() or frame.identity.isna().any()
            or frame.input_mode.nunique() != 1):
        raise ValueError("Use unique, validated prepared CzechLynx metadata.")
    metadata_hash = file_digest(metadata)
    prepared_path = metadata.with_name("manifest.json")
    prepared = _read_json(prepared_path)
    if prepared.get("prepared_metadata_sha256") != metadata_hash:
        raise ValueError("Prepared metadata does not match its manifest.")
    plan = load_protocol(protocol_plan, frame, metadata_hash)
    pool = plan["pools"][partition]
    ids = sorted(set(pool["gallery_ids"]) | set(pool["query_ids"]))
    if len(ids) != len(pool["gallery_ids"]) + len(pool["query_ids"]):
        raise ValueError("Gallery/query collection membership overlaps.")
    reference_path = reference_dir / "manifest.json"
    reference = _read_json(reference_path)
    if (reference.get("status") != "complete" or reference.get("stage") != "development"
            or reference.get("split") != "time_open" or reference.get("partition") != partition
            or reference.get("metadata_sha256") != metadata_hash
            or reference.get("protocol") != protocol_binding(plan)
            or reference.get("gallery_ids") != pool["gallery_ids"]
            or reference.get("query_ids") != pool["query_ids"]
            or reference.get("audit", {}).get("excluded") != pool["excluded"]
            or reference.get("prepared_data") != prepared
            or reference.get("config", {}).get("signals") != ["global"]):
        raise ValueError("Reference must be the matching completed phase 2 global run.")
    indexed = frame.set_index("image_id", drop=False)
    subset = indexed.loc[ids].reset_index(drop=True)
    spec = reference.get("global_model", {})
    dimension = spec.get("dimension")
    if (not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1
            or not spec.get("kind") or spec.get("input_mode") != subset.input_mode.iloc[0]
            or ("expected_dimension" in spec and spec["expected_dimension"] != dimension)):
        raise ValueError("Reference global model specification is invalid.")
    _require_hash(spec.get("weights_sha256"), "model weights")
    if spec["input_mode"] == "masked":
        from reid_demo.benchmark_masks import MASKED_PREPROCESSING
        if (spec.get("masked_preprocessing") != MASKED_PREPROCESSING
                or prepared.get("masked_preprocessing") != MASKED_PREPROCESSING
                or not {"mask_path", "mask_sha256"}.issubset(subset)):
            raise ValueError("Reference/preparation masking specification differs.")
    files = {"manifest.json": file_digest(reference_path)}
    for role in ("gallery", "query"):
        path = reference_dir / f"{role}.csv"
        saved = pd.read_csv(path, dtype={"image_id": str, "identity": str})
        expected = indexed.loc[pool[f"{role}_ids"]].reset_index(drop=True)
        if list(saved.columns) != list(expected.columns) or not saved.equals(expected):
            raise ValueError(f"Reference {role} rows differ from the prepared protocol.")
        files[path.name] = file_digest(path)
    cache_spec = {key: value for key, value in spec.items() if key != "dimension"}
    binding = {
        "protocol_fingerprint": plan["fingerprint"], "metadata_sha256": metadata_hash,
        "prepared_manifest_sha256": file_digest(prepared_path), "global_model": spec,
        "vector_cache_spec_fingerprint": fingerprint(cache_spec),
        "implementation": implementation_binding(),
    }
    provenance = {
        "metadata": str(metadata.resolve()), "protocol_plan": str(Path(protocol_plan).resolve()),
        "reference_dir": str(reference_dir.resolve()), "reference_files": files,
        "image_ids": ids, "gallery_ids": pool["gallery_ids"], "query_ids": pool["query_ids"],
        "excluded": pool["excluded"], "partition": partition,
        "seed": int(plan["settings"]["seed"]),
    }
    return Collection(subset, plan, reference, binding, provenance)


def load_frozen_vectors(frame, model_spec, cache_dir):
    """Load bytes through SQLite read-only mode, validating every requested record."""
    spec = {key: value for key, value in model_spec.items() if key != "dimension"}
    key = fingerprint(spec)
    path = Path(cache_dir) / f"global_{key[:20]}.sqlite"
    if not path.is_file():
        raise ValueError(f"Frozen vector cache is missing: {path}")
    dim = model_spec.get("dimension")
    if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
        raise ValueError("Invalid frozen vector dimension.")
    rows = frame.to_dict("records")
    matrix = np.empty((len(rows), dim), dtype=np.float32)
    missing = []
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        stored = conn.execute("SELECT value FROM metadata WHERE key='spec'").fetchone()
        if stored is None or stored[0] != key:
            raise ValueError("Frozen vector cache model/preprocessing specification differs.")
        for i, row in enumerate(rows):
            rid = row["image_id"]
            expected_hash = _require_hash(row.get("image_sha256"), rid)
            if file_digest(row["path"]) != expected_hash:
                raise ValueError(f"Image changed since preparation: {rid}")
            if row.get("input_mode") == "masked":
                mask_hash = _require_hash(row.get("mask_sha256"), f"{rid} mask")
                if file_digest(row["mask_path"]) != mask_hash:
                    raise ValueError(f"Mask changed since preparation: {rid}")
            record = conn.execute(
                "SELECT input_hash, dim, digest, data FROM vectors WHERE image_id=?", (rid,)
            ).fetchone()
            if record is None:
                missing.append(rid)
                continue
            input_hash, stored_dim, digest, data = record
            if (input_hash != expected_hash or stored_dim != dim or len(data) != dim * 4
                    or hashlib.sha256(data).hexdigest() != digest):
                raise ValueError(f"Changed input or damaged frozen vector: {rid}")
            vector = np.frombuffer(data, dtype=np.float32)
            if not np.isfinite(vector).all():
                raise ValueError(f"Nonfinite frozen vector: {rid}")
            matrix[i] = vector
    except sqlite3.Error as exc:
        raise ValueError(f"Cannot read frozen vector cache {path}: {exc}") from exc
    finally:
        conn.close()
    if missing:
        raise ValueError(f"Missing {len(missing)} frozen vectors; no extraction performed: "
                         + ", ".join(missing))
    return matrix, {"path": str(path.resolve()), "spec_fingerprint": key,
                    "required": len(rows), "loaded": len(rows), "missing_ids": [],
                    "dimension": dim, "access": "sqlite_mode_ro"}


def inference_inputs(ids, matrix):
    ids = list(map(str, ids))
    x = np.asarray(matrix, dtype=np.float64)
    if (not ids or len(set(ids)) != len(ids) or x.ndim != 2 or x.shape[0] != len(ids)
            or x.shape[1] == 0 or not np.isfinite(x).all()):
        raise ValueError("Clustering requires unique IDs and finite aligned vectors.")
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    ids, x = [ids[i] for i in order], x[order].astype(np.float64)
    norms = np.linalg.norm(x, axis=1)
    valid = norms > 0
    x[valid] /= norms[valid, None]
    return ids, x, valid


def _positive_int(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def validate_parameters(parameters):
    p = dict(parameters)
    algorithm = p.get("algorithm")
    if algorithm not in ALGORITHMS:
        raise ValueError("Unknown clustering algorithm.")
    _positive_int(p.get("knn_k"), "knn_k")
    if not isinstance(p.get("seed"), int) or isinstance(p.get("seed"), bool) or p["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer.")
    expected = {"algorithm", "knn_k", "seed"}
    if algorithm == "legacy_neighbour":
        expected |= {"resolution", "backend"}
        if (p.get("backend") not in ("leiden", "networkx_louvain")
                or isinstance(p.get("resolution"), bool)
                or not isinstance(p.get("resolution"), (float, int))
                or not np.isfinite(p["resolution"]) or p["resolution"] <= 0):
            raise ValueError("Invalid existing neighbour-graph configuration.")
    else:
        expected.add("threshold")
        t = p.get("threshold")
        if isinstance(t, bool) or not isinstance(t, (float, int)) or not np.isfinite(t) or not 0 <= t <= 1:
            raise ValueError("Cosine threshold must be finite and in [0, 1].")
    if set(p) != expected:
        raise ValueError("Unexpected clustering options; target counts are not allowed.")
    return p


def legacy_backend():
    try:
        import igraph  # noqa: F401
        import leidenalg  # noqa: F401
        return "leiden"
    except ImportError:
        import networkx  # noqa: F401
        return "networkx_louvain"


def profile_id(parameters):
    p = validate_parameters(parameters)
    if p["algorithm"] == "legacy_neighbour":
        return f"legacy_k{p['knn_k']}_{p['backend']}"
    value = format(p["threshold"], ".12g").replace(".", "p")
    return f"{p['algorithm']}_k{p['knn_k']}_t{value}_{fingerprint(p)[:6]}"


def calibration_profiles(seed, budgets=DEFAULT_BUDGETS, thresholds=DEFAULT_THRESHOLDS):
    budgets, thresholds = tuple(budgets), tuple(thresholds)
    if not budgets or not thresholds or len(set(budgets)) != len(budgets) or len(set(thresholds)) != len(thresholds):
        raise ValueError("Provide nonempty, distinct budgets and thresholds.")
    profiles = [{"algorithm": "legacy_neighbour", "knn_k": 15, "resolution": 1.0,
                 "backend": legacy_backend(), "seed": seed}]
    profiles.extend({"algorithm": algorithm, "knn_k": int(k), "threshold": float(t), "seed": seed}
                    for algorithm in ("threshold", "consistent") for k in budgets for t in thresholds)
    return [validate_parameters(p) for p in profiles]


def candidate_neighbours(ids, matrix, budgets, *, chunk_size=64):
    """Union of directional top-k neighbours, sorted by score and stable image ID."""
    _positive_int(chunk_size, "chunk_size")
    budgets = sorted(set(_positive_int(k, "candidate budget") for k in budgets))
    if not budgets:
        return {}
    ids, x, valid = inference_inputs(ids, matrix)
    ranks = {}
    k = min(max(budgets), max(0, int(valid.sum()) - 1))
    for start in range(0, len(ids), chunk_size):
        scores = np.clip(np.round(x[start:start + chunk_size] @ x.T, SCORE_DECIMALS), -1, 1)
        scores[:, ~valid] = -np.inf
        for offset, row in enumerate(scores):
            i = start + offset
            if not valid[i]:
                continue
            row[i] = -np.inf
            for rank, j in enumerate(np.argsort(-row, kind="stable")[:k]):
                a, b = sorted((i, int(j)))
                old = ranks.get((a, b))
                ranks[a, b] = (float(row[j]), min(rank, old[1]) if old else rank)
    ordered = sorted(((a, b, score, rank) for (a, b), (score, rank) in ranks.items()),
                     key=lambda edge: (-edge[2], ids[edge[0]], ids[edge[1]]))
    return {budget: [(a, b, score) for a, b, score, rank in ordered if rank < budget]
            for budget in budgets}


def cluster_control(ids, matrix, parameters, *, edges=None, chunk_size=64, evidence_limit=5000):
    p = validate_parameters(parameters)
    _positive_int(chunk_size, "chunk_size")
    if not isinstance(evidence_limit, int) or isinstance(evidence_limit, bool) or evidence_limit < 0:
        raise ValueError("evidence_limit must be nonnegative.")
    ids, x, valid = inference_inputs(ids, matrix)
    labels = np.full(len(ids), -1, dtype=int)
    audit = {"algorithm": p["algorithm"], "invalid_zero_vectors": int((~valid).sum()),
             "evidence_limit": evidence_limit}
    if p["algorithm"] == "legacy_neighbour":
        if legacy_backend() != p["backend"]:
            raise ValueError("Frozen baseline implementation is unavailable in this environment.")
        from reid_demo.cluster import cluster_embeddings
        selected = np.flatnonzero(valid)
        if len(selected):
            result = cluster_embeddings(dict(zip([ids[i] for i in selected], x[selected])),
                                        backend="leiden", backend_params={"knn_k": p["knn_k"],
                                        "resolution": p["resolution"], "seed": p["seed"]}, seed=p["seed"])
            raw = result.labels.copy()
            fresh = max(raw, default=-1) + 1
            for i in np.flatnonzero(raw < 0):
                raw[i], fresh = fresh, fresh + 1
            labels[selected] = raw
        audit.update(backend=p["backend"], candidate_pairs=None, retained_edge_rows=0,
                     edge_evidence_policy="Existing baseline internals; no reconstructed edge evidence.")
        return ClusteringResult(ids, labels, [], audit)
    edges = edges if edges is not None else candidate_neighbours(
        ids, x, [p["knn_k"]], chunk_size=chunk_size)[p["knn_k"]]
    parent = list(range(len(ids)))
    members = {i: [i] for i in np.flatnonzero(valid)}
    decisions, evidence = Counter(), []
    incompatible = {}
    cached_checks, cross_blocks = 0, 0

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b, score in edges:
        if (not 0 <= a < b < len(ids) or not valid[a] or not valid[b]
                or not np.isfinite(score) or not -1 <= score <= 1):
            raise ValueError("Invalid candidate edge.")
        ra, rb = root(a), root(b)
        checked_min = None
        if score < p["threshold"]:
            decision = "below_threshold"
        elif ra == rb:
            decision = "already_connected"
        else:
            decision = "merged"
            if p["algorithm"] == "consistent":
                pair = tuple(sorted((ra, rb)))
                if pair in incompatible:
                    # Components only grow: an existing failed cross-pair stays a witness.
                    checked_min = incompatible[pair]
                    decision = "inconsistent_components"
                    cached_checks += 1
                else:
                    left, right = members[ra], members[rb]
                    for start in range(0, len(left), chunk_size):
                        for stop in range(0, len(right), chunk_size):
                            cross = np.clip(np.round(x[left[start:start + chunk_size]]
                                                    @ x[right[stop:stop + chunk_size]].T,
                                                    SCORE_DECIMALS), -1, 1)
                            cross_blocks += 1
                            minimum = float(cross.min())
                            checked_min = minimum if checked_min is None else min(checked_min, minimum)
                            if minimum < p["threshold"]:
                                decision = "inconsistent_components"
                                incompatible[pair] = checked_min
                                break
                        if decision != "merged":
                            break
        decisions[decision] += 1
        if len(evidence) < evidence_limit:
            evidence.append({"image_id_a": ids[a], "image_id_b": ids[b], "cosine_similarity": score,
                             "decision": decision, "component_size_a": len(members[ra]),
                             "component_size_b": len(members[rb]), "checked_min_similarity": checked_min})
        if decision == "merged":
            lo, hi = sorted((ra, rb))
            parent[hi] = lo
            members[lo] = sorted(members[lo] + members.pop(hi))
    groups = sorted(members.values(), key=lambda group: ids[min(group)])
    for label, group in enumerate(groups):
        labels[group] = label
    audit.update(candidate_pairs=len(edges), decisions=dict(decisions), retained_edge_rows=len(evidence),
                 truncated_edge_rows=len(edges) - len(evidence),
                 edge_evidence_policy="First score-ordered decisions; counts cover all candidates.",
                 cached_inconsistent_checks=cached_checks, cross_similarity_blocks=cross_blocks,
                 cosine_rounding_decimals=SCORE_DECIMALS)
    return ClusteringResult(ids, labels, evidence, audit)


def _value(value):
    return None if pd.isna(value) or not str(value).strip() else str(value)


def _camera(row):
    source, trap = _value(row.get("source")), _value(row.get("trap_id"))
    return f"{source}:{trap}" if source and trap else None


def _choose_two(n):
    return n * (n - 1) // 2


def _cross_group_pairs(rows, field):
    eligible = [r for r in rows if r.get(field) is not None]
    truth = Counter(r["identity"] for r in eligible)
    truth_axis = Counter((r["identity"], r[field]) for r in eligible)
    assigned = [r for r in eligible if r["component_id"] >= 0]
    predicted = Counter(r["component_id"] for r in assigned)
    predicted_axis = Counter((r["component_id"], r[field]) for r in assigned)
    correct = Counter((r["component_id"], r["identity"]) for r in assigned)
    correct_axis = Counter((r["component_id"], r["identity"], r[field]) for r in assigned)
    pairs = lambda counter: sum(_choose_two(n) for n in counter.values())
    denominator = pairs(predicted) - pairs(predicted_axis)
    positives = pairs(truth) - pairs(truth_axis)
    true_positives = pairs(correct) - pairs(correct_axis)
    return {"eligible_images": len(eligible), "true_pairs": positives,
            "predicted_pairs": denominator, "correct_pairs": true_positives,
            "precision": true_positives / denominator if denominator else None,
            "recall": true_positives / positives if positives else None}


def evaluate_clustering(result, frame, *, appearance=None):
    """Truth enters here, after label-free inference; unresolved is not a counted animal."""
    from reid_demo.eval import standard_metrics
    appearance = appearance or {}
    if (not result.image_ids or len(set(result.image_ids)) != len(result.image_ids)
            or frame.image_id.duplicated().any() or set(frame.image_id) != set(result.image_ids)
            or len(result.labels) != len(result.image_ids)):
        raise ValueError("Evaluation must cover exactly the inferred collection.")
    indexed = frame.set_index("image_id")
    rows, groups, identities = [], defaultdict(list), defaultdict(list)
    sizes = Counter(int(label) for label in result.labels if label >= 0)
    for rid, label in zip(result.image_ids, result.labels):
        source = indexed.loc[rid].to_dict()
        mono = appearance.get(rid, {}).get("near_monochrome")
        row = {"image_id": rid, "identity": str(source["identity"]), "component_id": int(label),
               "component_size": sizes[int(label)] if label >= 0 else 0,
               "encounter_id": _value(source.get("encounter_id")), "camera": _camera(source),
               "source": _value(source.get("source")), "orientation": _value(source.get("orientation")),
               "appearance_proxy": "unknown" if mono is None else "near_monochrome" if mono else "colour",
               "path": source.get("path")}
        rows.append(row)
        identities[row["identity"]].append(row)
        if label >= 0:
            groups[int(label)].append(row)
    breakdown = []
    for label, members in sorted(groups.items()):
        breakdown.append({"component_id": label, "images": len(members),
                          "identities": sorted({r["identity"] for r in members}),
                          "encounters": len({r["encounter_id"] for r in members if r["encounter_id"]}),
                          "cameras": len({r["camera"] for r in members if r["camera"]})})
    by_label = {r["component_id"]: r for r in breakdown}
    identity_rows = []
    for identity, members in sorted(identities.items()):
        components = sorted({r["component_id"] for r in members if r["component_id"] >= 0})
        recovered = sum(r["component_id"] >= 0 and
                        len(by_label[r["component_id"]]["identities"]) == 1 for r in members)
        identity_rows.append({"identity": identity, "images": len(members),
                              "encounters": len({r["encounter_id"] for r in members if r["encounter_id"]}),
                              "cameras": len({r["camera"] for r in members if r["camera"]}),
                              "components": components, "component_count": len(components),
                              "unresolved_images": sum(r["component_id"] < 0 for r in members),
                              "images_in_pure_components": recovered,
                              "pure_component_image_coverage": recovered / len(members),
                              "recovered_as_one_pure_component": len(components) == 1 and recovered == len(members)})
    truth = [r["identity"] for r in rows]
    metric_labels = result.labels.copy()
    fresh = max(metric_labels, default=-1) + 1
    for i in np.flatnonzero(metric_labels < 0):
        metric_labels[i], fresh = fresh, fresh + 1
    unresolved = int(np.sum(result.labels < 0))
    count, true_count = len(groups), len(identities)
    signed_error = count - true_count if unresolved == 0 else None
    standard = standard_metrics(truth, metric_labels.tolist())
    true_pairs = sum(_choose_two(len(members)) for members in identities.values())
    predicted_pairs = sum(_choose_two(size) for size in sizes.values())
    correct_pairs = sum(_choose_two(n) for n in Counter(
        (r["component_id"], r["identity"]) for r in rows if r["component_id"] >= 0).values())
    precision = correct_pairs / predicted_pairs if predicted_pairs else None
    recall = correct_pairs / true_pairs if true_pairs else None
    standard.update(pairwise_precision=precision, pairwise_recall=recall,
                    pairwise_f1=(2 * precision * recall / (precision + recall)
                                 if precision is not None and recall is not None and precision + recall
                                 else 0.0 if precision is not None and recall is not None else None))
    metrics = {
        **standard, "images": len(rows),
        "true_same_pairs": true_pairs, "predicted_same_pairs": predicted_pairs,
        "correct_same_pairs": correct_pairs,
        "true_individuals": true_count, "provisional_component_count": count,
        "complete_count_estimate": count if unresolved == 0 else None,
        "count_error_signed": signed_error,
        "count_error_absolute": abs(signed_error) if signed_error is not None else None,
        "count_error_relative": abs(signed_error) / true_count if signed_error is not None else None,
        "multi_image_groups": sum(r["images"] > 1 for r in breakdown),
        "singleton_proposals": sum(r["images"] == 1 for r in breakdown),
        "unresolved_images": unresolved, "assigned_image_coverage": (len(rows) - unresolved) / len(rows),
        "merge_errors": sum(len(r["identities"]) > 1 for r in breakdown),
        "split_errors": sum(r["component_count"] > 1 for r in identity_rows),
        "component_size_histogram": dict(sorted(Counter(sizes.values()).items())),
        "groups_with_cross_encounter_corroboration": sum(r["encounters"] > 1 for r in breakdown),
        "observations_without_cross_encounter_corroboration": unresolved + sum(
            r["images"] for r in breakdown if r["encounters"] <= 1),
        "groups_with_cross_camera_corroboration": sum(r["cameras"] > 1 for r in breakdown),
        "cross_encounter_pairs": _cross_group_pairs(rows, "encounter_id"),
        "cross_camera_pairs": _cross_group_pairs(rows, "camera"),
        "cross_appearance_proxy_pairs": _cross_group_pairs(
            [{**r, "appearance_proxy": None if r["appearance_proxy"] == "unknown"
              else r["appearance_proxy"]} for r in rows], "appearance_proxy"),
        "orientation_counts": dict(Counter(r["orientation"] or "unknown" for r in rows)),
        "appearance_proxy_counts": dict(Counter(r["appearance_proxy"] for r in rows)),
        "missing_camera_images": sum(r["camera"] is None for r in rows),
        "macro_identity_pure_component_coverage": sum(
            r["pure_component_image_coverage"] for r in identity_rows) / len(identity_rows),
        "macro_identity_exact_group_recovery": sum(
            r["recovered_as_one_pure_component"] for r in identity_rows) / len(identity_rows),
        "per_identity": identity_rows, "components": breakdown,
        "metric_unresolved_policy": "Unique metric label per unresolved image; never a count imputation.",
        "pair_metric_policy": "Undefined denominators report null, not vacuously perfect precision/recall.",
        "count_policy": "Nonnegative components include singleton proposals; count error withheld if unresolved.",
        "corroboration_policy": "Structural metadata diversity, not independent proof of identity.",
    }
    for row in rows:
        row["evidence_status"] = ("unresolved_zero_vector" if row["component_id"] < 0
                                  else "singleton_proposal" if row["component_size"] == 1
                                  else "provisional_multi_image_group")
    return metrics, rows


def visual_examples(rows, *, per_kind=3):
    """Select examples deterministically from full assignments, after inference."""
    by_group, by_identity = defaultdict(list), defaultdict(list)
    for row in rows:
        by_identity[row["identity"]].append(row)
        if row["component_id"] >= 0:
            by_group[row["component_id"]].append(row)
    examples = []
    groups = sorted(by_group.values(), key=lambda rs: (-len(rs), rs[0]["image_id"]))
    for members in groups:
        a = members[0]
        b = next((r for r in members if r["identity"] != a["identity"]), None)
        if b is not None:
            examples.append({"kind": "false_merge", "images": [a["image_id"], b["image_id"]]})
        if sum(e["kind"] == "false_merge" for e in examples) >= per_kind:
            break
    for identity in sorted(by_identity):
        members = by_identity[identity]
        assigned = [r for r in members if r["component_id"] >= 0]
        if assigned:
            a = assigned[0]
            b = next((r for r in assigned if r["component_id"] != a["component_id"]), None)
            if b is not None and sum(e["kind"] == "fragmentation" for e in examples) < per_kind:
                examples.append({"kind": "fragmentation", "images": [a["image_id"], b["image_id"]]})
        for kind, field, eligible in (
            ("cross_camera", "camera", [r for r in members if r["camera"]]),
            ("cross_appearance_proxy", "appearance_proxy",
             [r for r in members if r["appearance_proxy"] != "unknown"]),
        ):
            if eligible and sum(e["kind"] == kind for e in examples) < per_kind:
                a = eligible[0]
                b = next((r for r in eligible if r[field] != a[field]), None)
                if b is not None:
                    examples.append({"kind": kind, "images": [a["image_id"], b["image_id"]]})
    for kind, eligible in (
        ("isolated", [r for r in rows if r["component_size"] == 1]),
        ("unresolved", [r for r in rows if r["component_id"] < 0]),
    ):
        examples.extend({"kind": kind, "images": [r["image_id"]]} for r in eligible[:per_kind])
    return examples


def _write_csv(path, rows, columns):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _shell(title, body):
    return ("<!doctype html><html lang='en'><meta charset='utf-8'>"
            f"<title>{html.escape(title)}</title>"
            "<style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:16px}"
            "table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:8px;text-align:left}"
            ".pair{display:flex;flex-wrap:wrap;gap:18px}.pair figure{margin:8px;width:42%}"
            "img{max-width:100%}pre{white-space:pre-wrap}a{color:#185790}</style>"
            f"<h1>{html.escape(title)}</h1>{body}</html>")


def write_profile_report(path, result, parameters, metrics, rows):
    from PIL import Image
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / "parameters.json", parameters)
    write_json(path / "metrics.json", metrics)
    write_json(path / "edge_audit.json", result.audit)
    assignment_columns = ("image_id", "component_id", "component_size", "evidence_status",
                          "encounter_id", "camera", "source", "orientation", "appearance_proxy", "identity")
    _write_csv(path / "assignments.csv", rows, assignment_columns)
    _write_csv(path / "edges.csv", result.evidence,
               ("image_id_a", "image_id_b", "cosine_similarity", "decision", "component_size_a",
                "component_size_b", "checked_min_similarity"))
    examples = visual_examples(rows)
    write_json(path / "visual_examples.json", {"policy": "Deterministic first IDs/largest mixed groups; 3 per kind.",
                                               "examples": examples})
    by_id = {row["image_id"]: row for row in rows}
    thumbs = path / "thumbs"
    thumbs.mkdir(exist_ok=True)
    body = "<p>All assignments are provisional. Identity truth below is evaluation evidence.</p>"
    keys = ("images", "true_individuals", "provisional_component_count", "count_error_signed",
            "count_error_absolute", "pairwise_precision", "pairwise_recall", "adjusted_rand_index",
            "merge_errors", "split_errors", "singleton_proposals", "unresolved_images")
    body += "<table><tr><th>Measure</th><th>Value</th></tr>" + "".join(
        f"<tr><td>{html.escape(key)}</td><td>{html.escape(str(metrics[key]))}</td></tr>" for key in keys) + "</table>"
    body += "<p>Count error is withheld when unresolved observations exist. "
    body += "No guaranteed count bounds or statistical confidence interval are supplied.</p>"
    body += "<p><a href='assignments.csv'>Assignments and evaluation truth</a> · "
    body += "<a href='edges.csv'>Bounded edge evidence</a> · <a href='metrics.json'>Full diagnostics</a></p>"
    for example in examples:
        body += f"<h2>{html.escape(example['kind'])}</h2><div class='pair'>"
        for rid in example["images"]:
            row = by_id[rid]
            filename = hashlib.sha256(rid.encode()).hexdigest()[:24] + ".jpg"
            with Image.open(row["path"]) as original:
                thumbnail = original.convert("RGB")
                thumbnail.thumbnail((480, 360))
                thumbnail.save(thumbs / filename, quality=90)
            caption = (f"{rid}; reference {row['identity']}; component {row['component_id']}; "
                       f"camera {row['camera']}; {row['appearance_proxy']}")
            body += (f"<figure><img src='thumbs/{filename}' alt='{html.escape(caption, quote=True)}'>"
                     f"<figcaption>{html.escape(caption)}</figcaption></figure>")
        body += "</div>"
    body += "<h2>Limitations</h2><ul>" + "".join(f"<li>{html.escape(n)}</li>" for n in NOTES) + "</ul>"
    (path / "index.html").write_text(_shell("Full-collection clustering diagnostic", body))
    names = ("parameters.json", "metrics.json", "assignments.csv", "edges.csv", "edge_audit.json",
             "visual_examples.json", "index.html")
    artifact_names = [*names, *(str(p.relative_to(path)) for p in sorted(thumbs.glob("*.jpg")))]
    return {name: file_digest(path / name) for name in artifact_names}


def write_overview(path, profiles, partition):
    path = Path(path)
    columns = ("profile_id", "algorithm", "knn_k", "threshold", "resolution", "backend",
               "provisional_component_count", "count_error_signed", "count_error_absolute",
               "pairwise_precision", "pairwise_recall", "adjusted_rand_index", "merge_errors",
               "split_errors", "singleton_proposals", "unresolved_images", "seconds")
    flat = [{**{k: p["metrics"].get(k) for k in columns}, **p["parameters"],
             "profile_id": name, "seconds": p["seconds"]}
            for name, p in profiles.items()]
    _write_csv(path / "profiles.csv", flat, columns)
    write_json(path / "metrics.json", {name: p["metrics"] for name, p in profiles.items()})
    body = f"<p>Partition: {html.escape(partition)}. Full eligible gallery/query union; no target count supplied.</p>"
    if partition == "calibration":
        body += "<p>Review count error, false merges and fragmentation together before exporting a configuration.</p>"
        body += "<p>Sensitivity plots describe calibration settings; they are not confidence intervals.</p>"
    else:
        body += "<p>One frozen calibration configuration; selection performs no threshold fitting or profiling.</p>"
    shown = ("provisional_component_count", "count_error_signed", "pairwise_precision", "pairwise_recall",
             "adjusted_rand_index", "merge_errors", "split_errors", "singleton_proposals", "unresolved_images")
    body += "<table><tr><th>Profile</th>" + "".join(f"<th>{html.escape(k)}</th>" for k in shown) + "</tr>"
    for name, p in profiles.items():
        body += f"<tr><td><a href='profiles/{name}/index.html'>{html.escape(name)}</a></td>"
        body += "".join(f"<td>{html.escape(str(p['metrics'][k]))}</td>" for k in shown) + "</tr>"
    body += "</table>"
    if partition == "calibration" and any(p["parameters"]["algorithm"] != "legacy_neighbour" for p in profiles.values()):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for algorithm in ("threshold", "consistent"):
            budgets = sorted({p["parameters"]["knn_k"] for p in profiles.values()
                              if p["parameters"]["algorithm"] == algorithm})
            for budget in budgets:
                entries = sorted((p for p in profiles.values() if p["parameters"]["algorithm"] == algorithm
                                  and p["parameters"]["knn_k"] == budget),
                                 key=lambda p: p["parameters"]["threshold"])
                thresholds = [p["parameters"]["threshold"] for p in entries]
                for ax, metric in zip(axes, ("count_error_signed", "pairwise_precision", "pairwise_recall")):
                    values = [p["metrics"][metric] if p["metrics"][metric] is not None else np.nan for p in entries]
                    ax.plot(thresholds, values, marker="o", label=f"{algorithm}, k={budget}")
                    ax.set_xlabel("Cosine threshold")
                    ax.set_ylabel(metric)
                    ax.grid(alpha=.25)
        axes[0].axhline(0, color="grey", linewidth=1)
        axes[-1].legend(fontsize=8)
        fig.suptitle("Calibration sensitivity — not count confidence intervals")
        fig.tight_layout()
        fig.savefig(path / "calibration_sensitivity.png", dpi=160)
        plt.close(fig)
        body += "<img src='calibration_sensitivity.png' alt='Calibration threshold sensitivity'>"
    body += "<h2>Limitations</h2><ul>" + "".join(f"<li>{html.escape(n)}</li>" for n in NOTES) + "</ul>"
    (path / "index.html").write_text(_shell("CzechLynx clustering and provisional counts", body))


def appearance_evidence(frame, source_cache_dir, report_dir):
    from reid_demo.benchmark_diagnostics import VERSION as APPEARANCE_VERSION, measure_appearance
    values, cached_count = {}, 0
    for row in frame.to_dict("records"):
        spec = {"version": APPEARANCE_VERSION, "image": row["image_sha256"],
                "mask": row.get("mask_sha256") if row["input_mode"] == "masked" else None,
                "input_mode": row["input_mode"]}
        path = Path(source_cache_dir) / "appearance" / (fingerprint(spec) + ".json")
        if path.exists():
            saved = _read_json(path)
            if saved.get("spec") != spec or saved.get("digest") != fingerprint(saved.get("values")):
                raise ValueError(f"Damaged existing appearance evidence: {row['image_id']}")
            value = saved["values"]
            cached_count += 1
        else:
            value = measure_appearance(row)
        if not isinstance(value, dict) or value.get("near_monochrome") not in (True, False, None):
            raise ValueError("Invalid appearance evidence.")
        values[row["image_id"]] = value
    write_json(Path(report_dir) / "appearance.json", {
        "version": APPEARANCE_VERSION, "source_cached_images": cached_count,
        "new_pixel_measurements": len(frame) - cached_count,
        "policy": "Existing evidence read only; new measurements stored in this report only.",
        "values": values,
    })
    return values


def check_output_directory(out_dir, protected, *, run_fingerprint=None):
    out = Path(out_dir).resolve()
    for value in protected:
        path = Path(value).resolve()
        if out == path or out.is_relative_to(path) or path.is_relative_to(out):
            raise ValueError(f"Output overlaps a protected source/calibration path: {path}")
    if out.exists() and any(out.iterdir()):
        manifest = out / "manifest.json"
        if not manifest.is_file() or run_fingerprint is None:
            raise ValueError("Output directory is not an empty clustering report directory.")
        old = _read_json(manifest)
        if old.get("kind") != VERSION or old.get("run_fingerprint") != run_fingerprint:
            raise ValueError("Output belongs to a different run; choose a new directory.")


def validated_completed_report(directory):
    directory = Path(directory).resolve()
    manifest = _read_json(directory / "manifest.json")
    spec = manifest.get("run_spec", {})
    if (manifest.get("schema") != 1 or manifest.get("kind") != VERSION
            or manifest.get("status") != "complete" or not spec
            or manifest.get("run_fingerprint") != fingerprint(spec)
            or manifest.get("binding") != spec.get("binding")
            or manifest.get("partition") != spec.get("partition")
            or manifest.get("collection") != spec.get("collection")
            or manifest.get("binding", {}).get("implementation") != implementation_binding()
            or not manifest.get("profiles") or not manifest.get("artifacts")):
        raise ValueError("Clustering report is incomplete, damaged or from a different implementation.")
    expected = {profile_id(p): validate_parameters(p) for p in spec.get("profile_parameters", [])}
    actual = {name: p.get("parameters") for name, p in manifest["profiles"].items()}
    if expected != actual:
        raise ValueError("Saved profile configurations differ from the run contract.")
    for name, digest in manifest["artifacts"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or file_digest(path) != digest:
            raise ValueError(f"Changed or missing report artifact: {name}")
    for name, profile in manifest["profiles"].items():
        if profile_id(profile["parameters"]) != name or not profile.get("artifacts"):
            raise ValueError("Invalid saved profile ID/evidence.")
        root = directory / "profiles" / name
        for filename, digest in profile["artifacts"].items():
            path = (root / filename).resolve()
            if not path.is_relative_to(root) or not path.is_file() or file_digest(path) != digest:
                raise ValueError(f"Changed or missing profile artifact: {name}/{filename}")
        if (_read_json(root / "parameters.json") != profile["parameters"]
                or _read_json(root / "metrics.json") != profile["metrics"]):
            raise ValueError("Profile metrics/configuration differ from their manifest.")
    return manifest


def export_configuration(calibration_dir, chosen_profile, config_out):
    directory = Path(calibration_dir).resolve()
    manifest = validated_completed_report(directory)
    if manifest["partition"] != "calibration":
        raise ValueError("Configuration export requires a calibration report, never selection.")
    if chosen_profile not in manifest["profiles"]:
        raise ValueError("Choose an existing profile ID from profiles.csv.")
    source = manifest["collection"]
    collection = load_collection(source["metadata"], source["protocol_plan"], source["reference_dir"], "calibration")
    if collection.binding != manifest["binding"] or collection.provenance != source:
        raise ValueError("Calibration source inputs changed since the completed report.")
    payload = {
        "schema": 1, "kind": CONFIG_KIND, "version": VERSION, "partition": "calibration",
        "profile_id": chosen_profile, "parameters": manifest["profiles"][chosen_profile]["parameters"],
        "binding": manifest["binding"], "calibration_collection_ids": source["image_ids"],
        "calibration_report_dir": str(directory),
        "calibration_manifest_sha256": file_digest(directory / "manifest.json"),
        "calibration_run_fingerprint": manifest["run_fingerprint"],
    }
    payload["fingerprint"] = fingerprint(payload)
    path = Path(config_out).resolve()
    # Export can add a file to its own calibration directory, but cannot replace evidence.
    protected = {"manifest.json", *manifest["artifacts"]}
    if path == directory or (path.is_relative_to(directory) and (
            str(path.relative_to(directory)) in protected or path.is_relative_to(directory / "profiles"))):
        raise ValueError("Configuration export cannot overwrite calibration evidence.")
    for source_dir in (source["reference_dir"], Path(source["metadata"]).parent,
                       manifest["run_spec"]["cache_dir"]):
        if path.is_relative_to(Path(source_dir).resolve()):
            raise ValueError("Configuration export cannot write into source artifacts.")
    if path.exists():
        if not path.is_file() or _read_json(path) != payload:
            raise ValueError("Configuration output already exists with different content.")
        return payload
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)
    return payload


def load_frozen_configuration(path, collection):
    data = _read_json(path)
    payload = {k: v for k, v in data.items() if k != "fingerprint"}
    if (data.get("schema") != 1 or data.get("kind") != CONFIG_KIND or data.get("version") != VERSION
            or data.get("partition") != "calibration" or data.get("fingerprint") != fingerprint(payload)
            or data.get("binding") != collection.binding):
        raise ValueError("Selection requires this protocol/model's exported clustering configuration.")
    pool = collection.plan["pools"]["calibration"]
    expected_ids = sorted(set(pool["gallery_ids"]) | set(pool["query_ids"]))
    if data.get("calibration_collection_ids") != expected_ids:
        raise ValueError("Configuration came from a different calibration collection.")
    directory = Path(data["calibration_report_dir"])
    if file_digest(directory / "manifest.json") != data.get("calibration_manifest_sha256"):
        raise ValueError("Calibration report changed after configuration export.")
    manifest = validated_completed_report(directory)
    name = data.get("profile_id")
    if (manifest["partition"] != "calibration" or manifest["binding"] != collection.binding
            or manifest["run_fingerprint"] != data.get("calibration_run_fingerprint")
            or manifest["collection"]["image_ids"] != expected_ids
            or name not in manifest["profiles"]
            or manifest["profiles"][name]["parameters"] != data.get("parameters")):
        raise ValueError("Frozen configuration was not exported from the selected calibration profile.")
    if data["parameters"]["seed"] != collection.provenance["seed"]:
        raise ValueError("Configuration seed differs from the protocol.")
    return data
