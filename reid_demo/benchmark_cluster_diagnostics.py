"""Calibration-only pair and fixed-candidate-graph diagnostics.

Candidates are reconstructed by the existing label-free builder. Truth is used
only by the evaluators below. Oracle components are diagnostic fragments, never
deployable identities or estimates of park abundance. No neural inference or
source-cache writes occur here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import html
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reid_demo.benchmark import file_digest, fingerprint, write_json
from reid_demo.benchmark_clustering import (
    NOTES as CLUSTER_NOTES, ROOT, SCORE_DECIMALS, _camera, _positive_int, _value,
    check_output_directory, inference_inputs, load_collection, validated_completed_report,
)
from reid_demo.benchmark_diagnostics import VERSION as APPEARANCE_VERSION


VERSION = "calibration_pair_connectivity_v1"
HISTOGRAM_EDGES = np.round(np.linspace(-1, 1, 101), 2)
EXAMPLES_PER_KIND = 8
GROUPS = (
    "all", "within_encounter", "across_encounter", "missing_encounter",
    "within_camera", "across_camera", "missing_camera", "colour_colour",
    "near_monochrome_near_monochrome", "across_appearance_proxy", "missing_appearance_proxy",
)
NOTES = [*CLUSTER_NOTES,
    "Calibration diagnostics only; no selection or official test observations are evaluated.",
    "Oracle connectivity uses reference truth to filter a fixed candidate graph; it is not achieved accuracy.",
    "Oracle fragment counts do not bound all future methods and are not deployed animal counts.",
    "Zero vectors are unscorable. Their true positive pairs remain in recall denominators.",
    "Direct-edge and within-component precision have different populations; their difference is descriptive.",
    "Histograms include scorable unordered image pairs only; macro recall also reports eligible identity counts.",
    "Review examples are deterministic score extremes, not a representative sample or a complete audit.",
]


@dataclass
class CalibrationSource:
    directory: Path
    manifest: dict
    collection: object
    appearance: dict
    thresholds_by_budget: dict
    binding: dict

    @property
    def thresholds(self):
        return sorted({t for values in self.thresholds_by_budget.values() for t in values})


@dataclass
class EvaluationInputs:
    image_ids: list
    vectors: np.ndarray
    valid: np.ndarray
    frame: pd.DataFrame
    identity_codes: np.ndarray
    identities: list
    encounter_codes: np.ndarray
    camera_codes: np.ndarray
    appearance_codes: np.ndarray


def implementation_binding():
    names = ("reid_demo/benchmark_cluster_diagnostics.py", "tools/diagnose_czechlynx_clusters.py")
    return {"version": VERSION, "code": {name: file_digest(ROOT / name) for name in names},
            "threadpoolctl_version": importlib.metadata.version("threadpoolctl"),
            "histogram_edges": HISTOGRAM_EDGES.tolist(), "examples_per_kind": EXAMPLES_PER_KIND,
            "cosine_rounding_decimals": SCORE_DECIMALS}


def load_calibration(directory):
    """Reject non-calibration roles before reconstructing source membership."""
    directory = Path(directory).resolve()
    header = json.loads((directory / "manifest.json").read_text())
    if not isinstance(header, dict) or header.get("partition") != "calibration":
        raise ValueError("Diagnostics require a completed calibration report; selection is forbidden.")
    manifest = validated_completed_report(directory)
    source = manifest["collection"]
    collection = load_collection(source["metadata"], source["protocol_plan"],
                                 source["reference_dir"], "calibration")
    if collection.binding != manifest["binding"] or collection.provenance != source:
        raise ValueError("Calibration source/model/input contracts changed since the report.")
    saved = pd.read_csv(directory / "collection.csv", dtype={"image_id": str, "identity": str})
    if list(saved.columns) != list(collection.frame.columns) or not saved.equals(collection.frame):
        raise ValueError("Calibration collection rows differ from the reconstructed protocol.")
    appearance = json.loads((directory / "appearance.json").read_text())
    values = appearance.get("values", {})
    if (appearance.get("version") != APPEARANCE_VERSION
            or set(values) != set(collection.frame.image_id)
            or any(not isinstance(v, dict) or "near_monochrome" not in v
                   or (v["near_monochrome"] is not None and not isinstance(v["near_monochrome"], bool))
                   for v in values.values())):
        raise ValueError("Calibration appearance evidence is incompatible or incomplete.")
    grid = defaultdict(set)
    for profile in manifest["profiles"].values():
        p = profile["parameters"]
        if p["algorithm"] != "legacy_neighbour":
            grid[p["knn_k"]].add(float(p["threshold"]))
    if not grid:
        raise ValueError("Calibration contains no threshold/consistent candidate controls.")
    binding = {"calibration_manifest_sha256": file_digest(directory / "manifest.json"),
               "calibration_run_fingerprint": manifest["run_fingerprint"],
               "protocol_plan_sha256": file_digest(source["protocol_plan"]),
               "source_binding": collection.binding, "collection": source,
               "diagnostic_implementation": implementation_binding()}
    return CalibrationSource(directory, manifest, collection, values,
                             {k: sorted(values) for k, values in sorted(grid.items())}, binding)


def check_diagnostic_output(out, source):
    frame, provenance = source.collection.frame, source.collection.provenance
    protected = [source.directory, provenance["reference_dir"],
                 source.manifest["run_spec"]["cache_dir"], Path(provenance["metadata"]).parent,
                 Path(provenance["protocol_plan"])]
    protected.extend({str(Path(p).parent) for p in frame.path})
    if "mask_path" in frame:
        protected.extend({str(Path(p).parent) for p in frame.mask_path.dropna()})
    # A diagnostic always starts in a new/empty directory, including after failure.
    check_output_directory(out, protected)


def prepare_evaluation(image_ids, matrix, frame, appearance=None):
    """Align truth/context after label-free candidate inference, retaining zero rows."""
    ids, x, valid = inference_inputs(image_ids, matrix)
    if (not {"image_id", "identity"}.issubset(frame) or frame.image_id.duplicated().any()
            or set(frame.image_id) != set(ids) or frame.identity.isna().any()
            or frame.identity.astype(str).str.strip().eq("").any()):
        raise ValueError("Evaluation metadata must cover exactly the inferred IDs and valid reference identities.")
    frame = frame.set_index("image_id", drop=False).loc[ids].reset_index(drop=True)
    labels = frame.identity.astype(str)
    identities = sorted(set(labels))
    codes = {identity: i for i, identity in enumerate(identities)}
    rows = frame.to_dict("records")
    encode = lambda values: pd.factorize(pd.Series(values, dtype=object), sort=True)[0]
    appearance = appearance or {}
    colour = []
    for rid in ids:
        value = appearance.get(rid, {}).get("near_monochrome")
        if value is not None and not isinstance(value, bool):
            raise ValueError("Appearance groups require bool/null colour proxies.")
        colour.append(-1 if value is None else int(value))
    return EvaluationInputs(ids, x, valid, frame, np.asarray([codes[v] for v in labels]), identities,
                            encode([_value(r.get("encounter_id")) for r in rows]),
                            encode([_camera(r) for r in rows]), np.asarray(colour))


def _thresholds(values):
    values = list(values)
    if (not values or len(set(values)) != len(values)
            or any(isinstance(t, bool) or not isinstance(t, (int, float, np.integer, np.floating))
                   or not np.isfinite(t) or not 0 <= t <= 1 for t in values)):
        raise ValueError("Use distinct finite thresholds in [0, 1].")
    return sorted(float(t) for t in values)


def _relations(inputs, a, b):
    yield "all", np.ones(len(a), dtype=bool)
    for name, codes in (("encounter", inputs.encounter_codes), ("camera", inputs.camera_codes)):
        known = (codes[a] >= 0) & (codes[b] >= 0)
        equal = codes[a] == codes[b]
        yield f"within_{name}", known & equal
        yield f"across_{name}", known & ~equal
        yield f"missing_{name}", ~known
    left, right = inputs.appearance_codes[a], inputs.appearance_codes[b]
    yield "colour_colour", (left == 0) & (right == 0)
    yield "near_monochrome_near_monochrome", (left == 1) & (right == 1)
    yield "across_appearance_proxy", (left >= 0) & (right >= 0) & (left != right)
    yield "missing_appearance_proxy", (left < 0) | (right < 0)


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


class PairAccumulator:
    """Stream unordered pairs; unscorable positives stay in recall denominators."""

    def __init__(self, thresholds, identities, *, positive_distribution=False):
        self.thresholds = _thresholds(thresholds)
        self.identities = identities
        n, bins = len(identities), len(HISTOGRAM_EDGES) - 1
        self.population_pairs = self.positive_pairs = self.scored_pairs = self.scored_positive_pairs = 0
        self.positive_by_identity = np.zeros(n, dtype=np.int64)
        self.tp = np.zeros(len(self.thresholds), dtype=np.int64)
        self.fp = self.tp.copy()
        self.identity_tp = np.zeros((len(self.thresholds), n), dtype=np.int64)
        self.hist_same = np.zeros(bins, dtype=np.int64)
        self.hist_different = self.hist_same.copy()
        self.positive_distribution = positive_distribution
        self.identity_histogram = np.zeros((n, bins), dtype=np.int64)
        self.identity_score_count = np.zeros(n, dtype=np.int64)
        self.identity_score_sum = np.zeros(n, dtype=np.int64)
        self.identity_score_min = np.full(n, np.inf)
        self.identity_score_max = np.full(n, -np.inf)

    def observe(self, scores, same, valid, identities):
        n = len(self.identities)
        self.population_pairs += len(scores)
        self.positive_pairs += int(np.sum(same))
        self.positive_by_identity += np.bincount(identities[same], minlength=n)
        scores, identities, same = scores[valid], identities[valid], same[valid]
        self.scored_pairs += len(scores)
        self.scored_positive_pairs += int(np.sum(same))
        self.hist_same += np.histogram(scores[same], bins=HISTOGRAM_EDGES)[0]
        self.hist_different += np.histogram(scores[~same], bins=HISTOGRAM_EDGES)[0]
        for i, threshold in enumerate(self.thresholds):
            accepted = scores >= threshold
            positive = accepted & same
            self.tp[i] += int(np.sum(positive))
            self.fp[i] += int(np.sum(accepted & ~same))
            self.identity_tp[i] += np.bincount(identities[positive], minlength=n)
        if self.positive_distribution:
            labels, values = identities[same], scores[same]
            self.identity_score_count += np.bincount(labels, minlength=n)
            # Sum the declared quantized scores exactly, independent of block order.
            np.add.at(self.identity_score_sum, labels,
                      np.rint(values * 10 ** SCORE_DECIMALS).astype(np.int64))
            np.minimum.at(self.identity_score_min, labels, values)
            np.maximum.at(self.identity_score_max, labels, values)
            bins = len(HISTOGRAM_EDGES) - 1
            positions = np.minimum(np.searchsorted(HISTOGRAM_EDGES, values, side="right") - 1, bins - 1)
            self.identity_histogram += np.bincount(labels * bins + positions,
                                                   minlength=n * bins).reshape(n, bins)

    def threshold_rows(self, group):
        rows = []
        eligible = self.positive_by_identity > 0
        for i, threshold in enumerate(self.thresholds):
            tp, fp = int(self.tp[i]), int(self.fp[i])
            precision, recall = _ratio(tp, tp + fp), _ratio(tp, self.positive_pairs)
            rows.append({"group": group, "threshold": threshold,
                         "population_pairs": self.population_pairs, "true_same_pairs": self.positive_pairs,
                         "scored_pairs": self.scored_pairs, "scored_same_pairs": self.scored_positive_pairs,
                         "unscorable_pairs": self.population_pairs - self.scored_pairs,
                         "unscorable_same_pairs": self.positive_pairs - self.scored_positive_pairs,
                         "accepted_same_pairs": tp, "accepted_different_pairs": fp,
                         "precision": precision, "recall": recall,
                         "f1": (2 * precision * recall / (precision + recall) if precision + recall else 0.0)
                         if precision is not None and recall is not None else None,
                         "macro_identity_positive_recall": float(np.mean(
                             self.identity_tp[i, eligible] / self.positive_by_identity[eligible]))
                         if eligible.any() else None,
                         "macro_recall_eligible_identities": int(eligible.sum())})
        return rows


def all_pair_diagnostics(inputs, thresholds, *, chunk_size=64, examples_per_kind=EXAMPLES_PER_KIND,
                         progress=None):
    """Evaluate every unordered reference pair in bounded score blocks, without fitting."""
    _positive_int(chunk_size, "chunk_size")
    _positive_int(examples_per_kind, "examples_per_kind")
    accumulators = {group: PairAccumulator(thresholds, inputs.identities,
                                           positive_distribution=group == "all") for group in GROUPS}
    n = len(inputs.image_ids)
    low_pairs, processed = [], 0
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        block = np.clip(np.round(inputs.vectors[start:stop] @ inputs.vectors.T, SCORE_DECIMALS), -1, 1)
        a, b = np.nonzero(np.arange(n)[None, :] > np.arange(start, stop)[:, None])
        scores, a = block[a, b], a + start
        valid = inputs.valid[a] & inputs.valid[b]
        same = inputs.identity_codes[a] == inputs.identity_codes[b]
        for group, mask in _relations(inputs, a, b):
            accumulators[group].observe(scores[mask], same[mask], valid[mask], inputs.identity_codes[a[mask]])
        eligible = np.flatnonzero(same & valid)
        chosen = eligible[np.argsort(scores[eligible], kind="stable")[:examples_per_kind]]
        low_pairs.extend((int(a[i]), int(b[i]), float(scores[i])) for i in chosen)
        low_pairs = sorted(low_pairs, key=lambda e: (e[2], inputs.image_ids[e[0]], inputs.image_ids[e[1]]))[:examples_per_kind]
        processed += len(scores)
        if progress and (start // chunk_size % 16 == 0 or stop == n):
            progress(processed, n * (n - 1) // 2)
    total = accumulators["all"]
    if (total.population_pairs != n * (n - 1) // 2
            or int(total.hist_same.sum()) != total.scored_positive_pairs
            or int(total.hist_different.sum()) != total.scored_pairs - total.scored_positive_pairs):
        raise ValueError("Pair coverage/histogram counts do not cover the complete scored population.")
    return accumulators, low_pairs


def validate_candidate_edges(inputs, edges):
    """Validate the full undirected union rather than its bounded saved prefix."""
    previous, seen = None, set()
    for a, b, score in edges:
        if (isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_))
                or not isinstance(a, (int, np.integer)) or not isinstance(b, (int, np.integer))
                or not 0 <= a < b < len(inputs.image_ids) or not inputs.valid[a] or not inputs.valid[b]
                or not np.isfinite(score) or not -1 <= score <= 1 or (a, b) in seen):
            raise ValueError("Invalid, duplicate or unscorable candidate edge.")
        key = (-score, inputs.image_ids[a], inputs.image_ids[b])
        if previous is not None and key < previous:
            raise ValueError("Candidate edges differ from deterministic score/ID order.")
        previous = key
        seen.add((a, b))


def verify_candidates(source, inputs, candidates):
    if set(candidates) != set(source.thresholds_by_budget):
        raise ValueError("Reconstructed candidate budgets differ from calibration.")
    for edges in candidates.values():
        validate_candidate_edges(inputs, edges)
    for name, profile in source.manifest["profiles"].items():
        if profile["parameters"]["algorithm"] == "legacy_neighbour":
            continue
        edges = candidates[profile["parameters"]["knn_k"]]
        audit = profile["edge_audit"]
        if (audit.get("candidate_pairs") != len(edges)
                or audit.get("invalid_zero_vectors") != int((~inputs.valid).sum())):
            raise ValueError(f"Candidate/zero-vector counts differ from calibration: {name}")
        with (source.directory / "profiles" / name / "edges.csv").open(newline="") as stream:
            saved = list(csv.DictReader(stream))
        if len(saved) != audit.get("retained_edge_rows") or len(saved) > len(edges):
            raise ValueError(f"Retained candidate evidence differs from calibration: {name}")
        for row, (a, b, score) in zip(saved, edges):
            if (row["image_id_a"] != inputs.image_ids[a] or row["image_id_b"] != inputs.image_ids[b]
                    or float(row["cosine_similarity"]) != score):
                raise ValueError(f"Reconstructed candidate scores/IDs differ from calibration: {name}")
    return {"budgets": {str(k): len(v) for k, v in candidates.items()},
            "saved_candidate_counts_and_retained_score_prefixes_match": True,
            "policy": "Full reconstructed unions evaluated; saved prefixes are bounded evidence only."}


def oracle_connectivity(inputs, edges, *, threshold=None):
    """Truth-filtered components of a fixed graph, never a deployable inference result."""
    if threshold is not None:
        _thresholds([threshold])
    parent = np.arange(len(inputs.image_ids))
    degree = np.zeros(len(parent), dtype=int)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    for a, b, score in edges:
        if threshold is not None and score < threshold:
            continue
        if inputs.identity_codes[a] != inputs.identity_codes[b]:
            continue
        ra, rb = root(a), root(b)
        parent[max(ra, rb)] = min(ra, rb)
        degree[a] += 1
        degree[b] += 1
    labels = np.asarray([root(i) if inputs.valid[i] else -1 for i in range(len(parent))])
    rows = []
    for code, identity in enumerate(inputs.identities):
        selected = np.flatnonzero(inputs.identity_codes == code)
        valid = selected[inputs.valid[selected]]
        counts = Counter(int(labels[i]) for i in valid)
        rows.append({"identity": identity, "threshold": threshold, "images": len(selected),
                     "valid_images": len(valid), "unresolved_images": len(selected) - len(valid),
                     "oracle_fragments": len(counts), "component_sizes": sorted(counts.values(), reverse=True),
                     "largest_component_images": max(counts.values(), default=0),
                     "fully_connected": len(counts) == 1 and len(valid) == len(selected),
                     "valid_images_without_correct_neighbour": int(np.sum(degree[valid] == 0))})
    complete = sum(r["unresolved_images"] for r in rows) == 0
    multi = [r for r in rows if r["images"] > 1]
    summary = {"threshold": threshold, "identities": len(rows),
               "fully_connected_identities": sum(r["fully_connected"] for r in rows),
               "fraction_fully_connected_identities": _ratio(sum(r["fully_connected"] for r in rows), len(rows)),
               "multi_image_identities": len(multi),
               "fully_connected_multi_image_identities": sum(r["fully_connected"] for r in multi),
               "fraction_fully_connected_multi_image_identities": _ratio(sum(r["fully_connected"] for r in multi), len(multi)),
               "oracle_fragments": sum(r["oracle_fragments"] for r in rows),
               "complete_oracle_fragment_total": sum(r["oracle_fragments"] for r in rows) if complete else None,
               "unresolved_images": sum(r["unresolved_images"] for r in rows),
               "valid_images_without_correct_neighbour": sum(r["valid_images_without_correct_neighbour"] for r in rows),
               "multi_image_identity_valid_images_without_correct_neighbour": sum(
                   r["valid_images_without_correct_neighbour"] for r in multi),
               "policy": "Reference-truth filtering of this fixed candidate graph only; not deployable identities/counts."}
    return summary, rows, labels


def candidate_diagnostics(inputs, candidates, thresholds_by_budget, pair_accumulators, *, source_profiles=None,
                          examples_per_kind=EXAMPLES_PER_KIND):
    """Direct-edge metrics and reference-truth connectivity after candidate inference."""
    _positive_int(examples_per_kind, "examples_per_kind")
    if set(candidates) != set(thresholds_by_budget) or not candidates:
        raise ValueError("Candidates and existing threshold grids must have matching budgets.")
    threshold_rows, connectivity_rows, connectivity_summaries = [], [], []
    unthresholded_labels = {}
    for budget, edges in sorted(candidates.items()):
        _positive_int(budget, "candidate budget")
        validate_candidate_edges(inputs, edges)
        thresholds = _thresholds(thresholds_by_budget[budget])
        accumulators = {group: PairAccumulator(thresholds, inputs.identities) for group in GROUPS}
        # Bounded edge chunks avoid materialising every subgroup mask at once.
        for start in range(0, len(edges), 4096):
            chunk = edges[start:start + 4096]
            a = np.asarray([e[0] for e in chunk], dtype=int)
            b = np.asarray([e[1] for e in chunk], dtype=int)
            scores = np.asarray([e[2] for e in chunk])
            same = inputs.identity_codes[a] == inputs.identity_codes[b]
            for group, mask in _relations(inputs, a, b):
                accumulators[group].observe(scores[mask], same[mask], np.ones(int(mask.sum()), dtype=bool),
                                             inputs.identity_codes[a[mask]])
        for group, acc in accumulators.items():
            population = pair_accumulators[group]
            for i, row in enumerate(acc.threshold_rows(group)):
                threshold_rows.append({"knn_k": budget, "group": group, "threshold": row["threshold"],
                    "candidate_pairs": acc.population_pairs, "true_candidate_pairs": acc.positive_pairs,
                    "accepted_same_edges": int(acc.tp[i]), "accepted_different_edges": int(acc.fp[i]),
                    "direct_edge_precision": row["precision"],
                    "candidate_positive_recall": _ratio(int(acc.tp[i]), population.positive_pairs),
                    "true_population_pairs": population.positive_pairs,
                    "macro_identity_positive_recall": float(np.mean(
                        acc.identity_tp[i, population.positive_by_identity > 0]
                        / population.positive_by_identity[population.positive_by_identity > 0]))
                        if (population.positive_by_identity > 0).any() else None,
                    "macro_recall_eligible_identities": int((population.positive_by_identity > 0).sum())})
        for threshold in [None, *thresholds]:
            summary, rows, labels = oracle_connectivity(inputs, edges, threshold=threshold)
            connectivity_summaries.append({"knn_k": budget, **summary})
            connectivity_rows.extend({"knn_k": budget, **row} for row in rows)
            if threshold is None:
                unthresholded_labels[budget] = labels
    comparisons = []
    direct = {(r["knn_k"], r["threshold"]): r for r in threshold_rows if r["group"] == "all"}
    for name, profile in (source_profiles or {}).items():
        p, m = profile["parameters"], profile["metrics"]
        edge = direct.get((p["knn_k"], p.get("threshold")))
        comparisons.append({"profile_id": name, "algorithm": p["algorithm"], "knn_k": p["knn_k"],
            "threshold": p.get("threshold"), "direct_edge_precision": edge["direct_edge_precision"] if edge else None,
            "admitted_candidate_edges": edge["accepted_same_edges"] + edge["accepted_different_edges"] if edge else None,
            "component_pair_precision": m["pairwise_precision"], "component_pair_recall": m["pairwise_recall"],
            "provisional_component_count": m["provisional_component_count"],
            "merge_errors": m["merge_errors"], "split_errors": m["split_errors"],
            "comparison_policy": "Candidate edges before consistency checks; component pairs include indirect links."
                if edge else "Legacy community baseline has no absolute score cutoff; direct comparison undefined."})
    largest = max(candidates)
    false_pairs = []
    for edge in candidates[largest]:
        if inputs.identity_codes[edge[0]] != inputs.identity_codes[edge[1]]:
            false_pairs.append(edge)
            if len(false_pairs) == examples_per_kind:
                break
    disconnected = []
    labels = unthresholded_labels[largest]
    for code, identity in enumerate(inputs.identities):
        eligible = np.flatnonzero((inputs.identity_codes == code) & inputs.valid)
        if len(eligible):
            a = int(eligible[0])
            b = next((int(i) for i in eligible if labels[i] != labels[a]), None)
            if b is not None:
                score = float(np.clip(np.round(inputs.vectors[a] @ inputs.vectors[b], SCORE_DECIMALS), -1, 1))
                disconnected.append((a, b, score))
                if len(disconnected) == examples_per_kind:
                    break
    return {"candidate_thresholds": threshold_rows, "oracle_connectivity": connectivity_summaries,
            "identity_connectivity": connectivity_rows, "component_comparisons": comparisons,
            "false_candidate_pairs": false_pairs, "disconnected_pairs": disconnected,
            "review_candidate_budget": largest}


def pair_review_rows(inputs, low_pairs, candidate_results):
    """Expose reference truth and context only in post-inference review examples."""
    examples = []
    for kind, pairs in (("high_score_different_identity_candidate", candidate_results["false_candidate_pairs"]),
                        ("low_score_same_identity_pair", low_pairs),
                        ("disconnected_same_identity_candidate_groups", candidate_results["disconnected_pairs"])):
        for a, b, score in pairs:
            endpoints = []
            for i in (a, b):
                row = inputs.frame.iloc[i].to_dict()
                endpoints.append({"image_id": inputs.image_ids[i], "reference_identity": str(row["identity"]),
                    "encounter_id": _value(row.get("encounter_id")), "camera": _camera(row),
                    "orientation": _value(row.get("orientation")) or "unknown",
                    "appearance_proxy": "unknown" if inputs.appearance_codes[i] < 0 else
                        "near_monochrome" if inputs.appearance_codes[i] else "colour",
                    "night_ir_label": "unknown", "path": str(row.get("path", "")),
                    "image_sha256": row.get("image_sha256")})
            examples.append({"kind": kind, "cosine_similarity": score, "endpoints": endpoints,
                             "candidate_budget": candidate_results["review_candidate_budget"]
                                 if kind != "low_score_same_identity_pair" else None,
                             "connectivity_threshold": None})
    return examples


def _write_table(path, rows, fields):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()}
                         for row in rows)


def write_diagnostic_report(out, inputs, accumulators, candidate_results, low_pairs):
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out)
    all_pairs = accumulators["all"]
    pair_rows = [row for group, acc in accumulators.items() for row in acc.threshold_rows(group)]
    summary = {"images": len(inputs.image_ids), "reference_individuals": len(inputs.identities),
               "valid_vectors": int(inputs.valid.sum()), "unresolved_images": int((~inputs.valid).sum()),
               "unordered_population_pairs": all_pairs.population_pairs,
               "scored_pairs": all_pairs.scored_pairs,
               "unscorable_pairs": all_pairs.population_pairs - all_pairs.scored_pairs,
               "pair_thresholds": pair_rows, "candidate_thresholds": candidate_results["candidate_thresholds"],
               "oracle_connectivity": candidate_results["oracle_connectivity"],
               "component_comparisons": candidate_results["component_comparisons"], "notes": NOTES}
    summary["metadata_coverage"] = {
        "missing_encounter_images": int(np.sum(inputs.encounter_codes < 0)),
        "missing_camera_images": int(np.sum(inputs.camera_codes < 0)),
        "missing_appearance_proxy_images": int(np.sum(inputs.appearance_codes < 0)),
        "orientation_counts": dict(Counter(_value(row.get("orientation")) or "unknown"
                                           for row in inputs.frame.to_dict("records"))),
        "night_ir_policy": "No validated actual day/night/IR labels; appearance groups are colour proxies only.",
    }
    write_json(out / "summary.json", summary)
    _write_table(out / "pair_thresholds.csv", pair_rows, list(pair_rows[0]))
    histogram_rows, identity_histograms, identity_rows = [], [], []
    for group, acc in accumulators.items():
        for bin_index in range(len(HISTOGRAM_EDGES) - 1):
            histogram_rows.append({"group": group, "bin_left": float(HISTOGRAM_EDGES[bin_index]),
                "bin_right": float(HISTOGRAM_EDGES[bin_index + 1]),
                "same_identity_pairs": int(acc.hist_same[bin_index]),
                "different_identity_pairs": int(acc.hist_different[bin_index])})
    for i, identity in enumerate(inputs.identities):
        total = int(np.sum(inputs.identity_codes == i))
        valid = int(np.sum((inputs.identity_codes == i) & inputs.valid))
        count = int(all_pairs.identity_score_count[i])
        for j, threshold in enumerate(all_pairs.thresholds):
            identity_rows.append({"identity": identity, "images": total, "valid_vectors": valid,
                "unresolved_images": total - valid, "true_same_pairs": int(all_pairs.positive_by_identity[i]),
                "scored_same_pairs": count,
                "score_mean": _ratio(int(all_pairs.identity_score_sum[i]), count * 10 ** SCORE_DECIMALS),
                "score_min": float(all_pairs.identity_score_min[i]) if count else None,
                "score_max": float(all_pairs.identity_score_max[i]) if count else None,
                "threshold": threshold, "accepted_same_pairs": int(all_pairs.identity_tp[j, i]),
                "positive_recall": _ratio(int(all_pairs.identity_tp[j, i]), int(all_pairs.positive_by_identity[i]))})
        for j in range(len(HISTOGRAM_EDGES) - 1):
            identity_histograms.append({"identity": identity, "bin_left": float(HISTOGRAM_EDGES[j]),
                "bin_right": float(HISTOGRAM_EDGES[j + 1]), "same_identity_pairs": int(all_pairs.identity_histogram[i, j])})
    tables = (("pair_histograms.csv", histogram_rows), ("identity_pair_statistics.csv", identity_rows),
              ("identity_pair_histograms.csv", identity_histograms),
              ("candidate_thresholds.csv", candidate_results["candidate_thresholds"]),
              ("oracle_connectivity.csv", candidate_results["identity_connectivity"]),
              ("component_comparisons.csv", candidate_results["component_comparisons"]))
    for name, rows in tables:
        _write_table(out / name, rows, list(rows[0]) if rows else ["profile_id"])
    examples = pair_review_rows(inputs, low_pairs, candidate_results)
    write_json(out / "review_pairs.json", {"examples_per_kind": EXAMPLES_PER_KIND,
        "policy": {"high_score_different_identity_candidate": "Highest score then lexical IDs, largest saved budget.",
                   "low_score_same_identity_pair": "Lowest score then lexical IDs over all scorable same-identity pairs.",
                   "disconnected_same_identity_candidate_groups": "First lexical identity/IDs in separate unthresholded oracle components, largest saved budget."},
        "examples": examples, "notes": NOTES})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    centres = (HISTOGRAM_EDGES[:-1] + HISTOGRAM_EDGES[1:]) / 2
    for ax, group in zip(axes, ("all", "across_camera", "across_appearance_proxy")):
        for label, counts in (("same identity", accumulators[group].hist_same),
                              ("different identity", accumulators[group].hist_different)):
            if counts.sum():
                ax.plot(centres, counts / counts.sum(), label=label)
        ax.set_title(group.replace("_", " "))
        ax.set_xlabel("Raw cosine similarity")
        ax.set_ylabel("Fraction of scorable class pairs per bin")
        ax.set_xlim(-1, 1)
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=8)
    fig.suptitle("Calibration pair-score distributions; colour groups are not verified night/IR")
    fig.tight_layout()
    fig.savefig(out / "pair_score_histograms.png", dpi=160)
    plt.close(fig)
    body = (f"<p>Calibration only: {summary['images']} photographs, {summary['reference_individuals']} reference identities, "
            f"{summary['unresolved_images']} unresolved vectors.</p>"
            "<p>Oracle connectivity removes incorrect edges using reference truth. Its fragments are diagnostic, "
            "not achieved identity recovery or deployable animal counts.</p>"
            "<p>Zero-vector pairs are absent from score histograms and stay in true-pair recall denominators. "
            "Missing metadata has separate groups; image pairs are correlated.</p>"
            "<img style='max-width:100%' src='pair_score_histograms.png' alt='Calibration same/different pair score distributions'>")
    body += "<h2>Overall pair thresholds</h2><table><tr><th>Threshold</th><th>Precision</th><th>Recall</th><th>Macro positive recall</th></tr>"
    for row in all_pairs.threshold_rows("all"):
        body += "<tr>" + "".join(f"<td>{html.escape(str(row[k]))}</td>" for k in
                                ("threshold", "precision", "recall", "macro_identity_positive_recall")) + "</tr>"
    body += "</table><h2>Oracle connectivity of the fixed candidate graph</h2>"
    body += "<table><tr><th>k</th><th>Threshold</th><th>Connected identities</th><th>Diagnostic fragments</th><th>Unresolved</th></tr>"
    for row in candidate_results["oracle_connectivity"]:
        body += "<tr>" + "".join(f"<td>{html.escape(str(row[k]))}</td>" for k in
                                ("knn_k", "threshold", "fully_connected_identities", "oracle_fragments", "unresolved_images")) + "</tr>"
    body += "</table><h2>Data</h2><ul>"
    for name in ("summary.json", "pair_thresholds.csv", *(name for name, _ in tables), "review_pairs.json"):
        body += f"<li><a href='{name}'>{name}</a></li>"
    body += "</ul><h2>Deterministic review examples</h2><p>Score extremes; bounded review, not a representative sample.</p>"
    thumbs = out / "thumbs"
    thumbs.mkdir()
    for example in examples:
        body += f"<h3>{html.escape(example['kind'])}; cosine {example['cosine_similarity']}</h3><div class='pair'>"
        for row in example["endpoints"]:
            if file_digest(row["path"]) != row["image_sha256"]:
                raise ValueError(f"Review image changed since vector validation: {row['image_id']}")
            filename = fingerprint(row["image_id"])[:24] + ".jpg"
            with Image.open(row["path"]) as original:
                tile = original.convert("RGB")
                tile.thumbnail((480, 360))
                tile.save(thumbs / filename, quality=90)
            caption = (f"{row['image_id']}; reference {row['reference_identity']}; camera {row['camera']}; "
                       f"{row['appearance_proxy']}; flank {row['orientation']}; night/IR unknown")
            body += (f"<figure><img src='thumbs/{filename}' alt='{html.escape(caption, quote=True)}'>"
                     f"<figcaption>{html.escape(caption)}</figcaption></figure>")
        body += "</div>"
    body += "<h2>Limitations</h2><ul>" + "".join(f"<li>{html.escape(n)}</li>" for n in NOTES) + "</ul>"
    (out / "index.html").write_text("<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>CzechLynx calibration pair diagnostics</title><style>"
        "body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:16px}"
        "table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:8px}"
        ".pair{display:flex;flex-wrap:wrap;gap:18px}.pair figure{margin:8px;width:42%}img{max-width:100%}"
        "</style><h1>CzechLynx calibration pair and connectivity diagnostics</h1>" + body + "</html>")
    return summary, {str(p.relative_to(out)): file_digest(p) for p in sorted(out.rglob("*"))
                     if p.is_file() and p.name != "manifest.json"}
