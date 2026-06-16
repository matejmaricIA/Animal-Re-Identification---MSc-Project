"""reid_demo.eval — clustering evaluation harness (T07).

This module is the **scorecard** of the open-set lynx re-ID demo. Upstream tickets
detect (T02), classify (T03), embed (T04) and CLUSTER (T05) each crop, writing the
discovered ``cluster_id`` (plus ``cluster_conf`` / ``is_candidate_new``) back into the
shared T01 store. Labeled datasets (LeopardID2022 / ATRW) additionally carry a
ground-truth ``gt_identity`` (and ``orientation``) populated by T02. T07 reads BOTH the
predicted cluster labels and the ground-truth identities **READ-ONLY** and reports how
well the clustering matched reality — in plain language a park biologist understands AND
in standard ML clustering metrics.

WHAT THIS MODULE DOES (and only this):
  * Loads, for one ``dataset``, the crops that have BOTH a cluster assignment
    (``cluster_id IS NOT NULL``) AND a ground-truth label (``gt_identity IS NOT NULL``)
    through the T01 public API (``reid_demo.store`` only — never raw SQL).
  * Builds aligned predicted (cluster) and true (GT) label arrays, under an optional
    flank-aware GT convention (the DATA_CONTRACT 3-bucket policy, D4):
    ``left``->``left``, ``right``->``right``, ``{front,back,down,unknown,'',NULL}``->
    ``other``. When flank-aware, the effective GT label is ``f"{gt_identity}|{bucket}"``.
  * Computes BOTH metric families:
      - Plain-language / biologist metrics: true individual count, discovered cluster
        count, % photos correctly grouped, merge errors, split errors, candidate-new
        precision/recall.
      - Standard ML metrics: homogeneity, completeness, V-measure, ARI, AMI, plus a
        BCubed-style pairwise precision/recall/F1 (computed from group-size sums, NOT by
        materializing O(n^2) pairs).
  * Assembles per-individual and per-cluster breakdown tables for T06/T09.
  * Writes exactly ONE JSON report ``evaluations/clustering/<dataset>_<tag>.json`` whose
    top-level keys are the ``ClusteringReport`` field names; optionally a per-individual
    CSV and a one-page HTML summary.
  * Exposes a CLI: ``python -m reid_demo.eval --dataset LeopardID2022 [...]``.

It does NOT cluster, embed, detect, classify, render catalogues, or modify any existing
repo file. It is a pure READ-ONLY consumer of the store.

DEFINITIONS (reproducible, documented contract):
  * **pct_photos_correctly_grouped** (0..100, NEVER a 0..1 fraction): for each predicted
    cluster, its "majority true label" is the most common ``gt_label`` among its photos;
    a photo is "correctly grouped" iff its ``gt_label`` equals its cluster's majority
    true label. The percentage is ``(#correct) / (#evaluated) * 100``. With
    ``include_noise=True`` the whole ``cluster_id == -1`` noise bucket is treated as ONE
    pseudo-cluster for this calc; with ``include_noise=False`` noise rows are dropped.
  * **Merge errors:** predicted clusters whose photos span >1 distinct ``gt_label``. The
    merged true-label sets are recorded in ``merged_individual_groups``.
  * **Split errors:** distinct ``gt_label`` values whose photos appear in >1 distinct
    ``cluster_id`` (noise excluded from the "appears in" set unless ``include_noise``).
  * **Candidate-new precision/recall:** a GT "singleton" is a ``gt_label`` with exactly 1
    photo in the evaluated set. Precision = fraction of ``is_candidate_new==1`` rows whose
    ``gt_label`` is a singleton. Recall = fraction of GT-singleton photos flagged
    ``is_candidate_new==1``. With zero flags or zero singletons the respective metric is
    ``None`` (not 0 / NaN).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
)

from reid_demo import store

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Default directory for clustering eval outputs (parallel to closed-set
#: constants.EVALUATION_DIR='./evaluations/full_evals'; we never collide with it).
DEFAULT_OUT_DIR: str = "evaluations/clustering"

#: DBSCAN noise / unassigned label (DATA_CONTRACT: cluster_id == -1).
NOISE_LABEL: int = -1

#: Orientation values that map to each flank bucket (DATA_CONTRACT D4). Anything not
#: explicitly 'left'/'right' (including front/back/down/unknown/''/NULL) -> 'other'.
_FLANK_BUCKETS = {"left": "left", "right": "right"}


def flank_bucket(orientation: Optional[str]) -> str:
    """Collapse a raw orientation to the T05-matching {left, right, other} bucket (D4).

    ``left`` -> ``left``, ``right`` -> ``right``, and EVERYTHING else
    (``front``, ``back``, ``down``, ``unknown``, ``''``, ``None``) -> ``other``.
    """
    if orientation is None:
        return "other"
    return _FLANK_BUCKETS.get(str(orientation).strip().lower(), "other")


# --------------------------------------------------------------------------- #
# JSON-safe scalar coercion
# --------------------------------------------------------------------------- #

def _py(value: Any) -> Any:
    """Recursively cast numpy/pandas scalars to plain python for json.dump.

    Floats are rounded to 4 decimals for readability (full precision stays on the
    dataclass). ``None`` passes through unchanged.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    # numpy integers / python ints
    if isinstance(value, int):
        return int(value)
    # numpy / python floats (numpy scalars expose .item via float())
    if isinstance(value, float):
        if value != value:  # NaN -> None (never emit NaN into JSON)
            return None
        return round(float(value), 4)
    if isinstance(value, dict):
        return {str(_py(k) if not isinstance(k, str) else k): _py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_py(v) for v in value]
    # numpy scalar types (np.int64, np.float64, np.bool_) without importing numpy here
    item = getattr(value, "item", None)
    if callable(item):
        return _py(item())
    return value


# --------------------------------------------------------------------------- #
# Report dataclass
# --------------------------------------------------------------------------- #

@dataclass
class ClusteringReport:
    """A fully populated clustering scorecard for one (dataset, tag, flank convention).

    Top-level JSON keys equal these field names (snake_case). The headline contract
    T10 depends on is the trio ``pct_photos_correctly_grouped`` (0..100 scale),
    ``n_true_individuals`` (int) and ``n_found_clusters`` (int).
    """

    dataset: str
    tag: str
    flank_aware: bool

    # ---- counts (plain language) ----
    n_photos_total: int
    n_photos_labeled: int
    n_photos_clustered: int
    n_photos_evaluated: int
    n_photos_noise: int
    n_true_individuals: int
    n_found_clusters: int

    # ---- plain-language quality ----
    pct_photos_correctly_grouped: float
    n_merge_errors: int
    n_split_errors: int
    merged_individual_groups: List[List[str]]
    split_individuals: List[str]

    # ---- candidate-new (singleton) quality ----
    n_candidate_new: int
    candidate_new_precision: Optional[float]
    candidate_new_recall: Optional[float]

    # ---- standard ML metrics ----
    homogeneity: float
    completeness: float
    v_measure: float
    adjusted_rand_index: float
    adjusted_mutual_info: float
    pairwise_precision: float
    pairwise_recall: float
    pairwise_f1: float

    # ---- breakdown tables (records, JSON-serializable) ----
    per_individual: List[Dict[str, Any]] = field(default_factory=list)
    per_cluster: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a fully JSON-serializable dict (numpy scalars cast, floats rounded)."""
        return _py(asdict(self))

    def plain_language_summary(self) -> str:
        """Multi-line biologist-readable summary string (the pitch sentence)."""
        flank = "yes" if self.flank_aware else "no"
        lines = [
            f"Dataset: {self.dataset} (flank-aware: {flank})",
            (
                f"Photos evaluated: {self.n_photos_evaluated}  |  "
                f"Known individuals: {self.n_true_individuals}  |  "
                f"Found individuals: {self.n_found_clusters}"
            ),
            f"Correctly grouped: {self.pct_photos_correctly_grouped:.1f}% of photos",
            f"Merge mistakes: {self.n_merge_errors} (different cats grouped together)",
            f"Split mistakes: {self.n_split_errors} (one cat spread across multiple groups)",
        ]
        prec = self.candidate_new_precision
        rec = self.candidate_new_recall
        prec_s = "n/a" if prec is None else f"{prec:.2f}"
        rec_s = "n/a" if rec is None else f"{rec:.2f}"
        lines.append(
            f"Candidate-new flags: {self.n_candidate_new} "
            f"(precision {prec_s}, recall {rec_s})"
        )
        lines.append(
            f"Standard metrics: V-measure {self.v_measure:.2f} | "
            f"ARI {self.adjusted_rand_index:.2f} | AMI {self.adjusted_mutual_info:.2f}"
        )
        if self.n_photos_noise:
            lines.append(
                f"(Plus {self.n_photos_noise} photos the system was unsure about / noise.)"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Frame loading
# --------------------------------------------------------------------------- #

def _derive_gt_label(gt_identity: Any, orientation: Any, flank_aware: bool) -> Optional[str]:
    """Compute the effective ground-truth label for a row.

    Returns None when ``gt_identity`` is null/empty (unlabeled). With ``flank_aware``
    the label is ``f"{gt_identity}|{flank_bucket(orientation)}"``.
    """
    if gt_identity is None:
        return None
    # pandas may surface NULL as float NaN
    if isinstance(gt_identity, float) and gt_identity != gt_identity:
        return None
    gt = str(gt_identity)
    if gt == "":
        return None
    if not flank_aware:
        return gt
    return f"{gt}|{flank_bucket(orientation)}"


def load_eval_frame(conn, dataset: str, *, flank_aware: bool = False):
    """Read this dataset's rows from the T01 store and return a pandas DataFrame.

    Columns include at least ``record_id``, ``cluster_id``, ``gt_identity``,
    ``orientation``, ``is_candidate_new``, ``crop_path``, ``species``, ``review_status``,
    plus a derived ``gt_label`` column = ``gt_identity`` (flank_aware=False) or
    ``f"{gt_identity}|{flank_bucket}"`` (flank_aware=True), where ``flank_bucket`` maps
    orientation to the T05-matching {left, right, other} convention.

    Uses ``reid_demo.store.to_dataframe`` only. Does NOT drop unlabeled rows here (the
    caller decides via ``build_label_arrays``).
    """
    df = store.to_dataframe(conn, dataset=dataset)
    # to_dataframe always returns COLUMNS in order; ensure the derived label even when
    # the frame is empty (so downstream .empty/column access never KeyErrors).
    if df.empty:
        df["gt_label"] = []
        return df
    df = df.copy()
    df["gt_label"] = [
        _derive_gt_label(gt, orient, flank_aware)
        for gt, orient in zip(df["gt_identity"], df["orientation"])
    ]
    return df


def build_label_arrays(
    df, *, include_noise: bool = True
) -> Tuple[List[str], List[int], List[str]]:
    """Return aligned ``(y_true, y_pred, record_ids)`` over evaluable rows.

    Evaluable = rows where ``gt_label`` is not null AND ``cluster_id`` is not null. When
    ``include_noise=False``, rows with ``cluster_id == -1`` (noise) are dropped. ``y_true``
    are ``gt_label`` strings; ``y_pred`` are cluster ints; ``record_ids`` are the row ids.
    """
    y_true: List[str] = []
    y_pred: List[int] = []
    record_ids: List[str] = []
    if df is None or len(df) == 0:
        return y_true, y_pred, record_ids
    for rid, gt_label, cid in zip(df["record_id"], df["gt_label"], df["cluster_id"]):
        if gt_label is None:
            continue
        if isinstance(gt_label, float) and gt_label != gt_label:
            continue
        if cid is None:
            continue
        if isinstance(cid, float) and cid != cid:  # NaN cluster_id == "not clustered"
            continue
        cid_int = int(cid)
        if not include_noise and cid_int == NOISE_LABEL:
            continue
        y_true.append(str(gt_label))
        y_pred.append(cid_int)
        record_ids.append(str(rid))
    return y_true, y_pred, record_ids


# --------------------------------------------------------------------------- #
# Standard ML metrics
# --------------------------------------------------------------------------- #

def _pairwise_bcubed(y_true: List[str], y_pred: List[int]) -> Tuple[float, float, float]:
    """BCubed-style pairwise precision/recall/F1 via the contingency/group-size formula.

    Over all unordered pairs of evaluated photos, a pair is "predicted same" iff same
    ``cluster_id`` and "true same" iff same ``gt_label``.

        TP = sum over (true label x predicted cluster) cells of C(n_ij, 2)
        TP+FP = sum over predicted clusters of C(n_pred, 2)   (predicted-same pairs)
        TP+FN = sum over true labels of C(n_true, 2)          (true-same pairs)

    Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = harmonic mean. This is O(n) in the
    number of (label, cluster) cells, never O(n^2). Degenerate denominators (a labeling
    with no same-pairs) yield precision/recall = 1.0 by convention (vacuously perfect).
    """
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, 0.0

    def _c2(k: int) -> int:
        return k * (k - 1) // 2

    cell_counts: Counter = Counter()
    pred_counts: Counter = Counter()
    true_counts: Counter = Counter()
    for t, p in zip(y_true, y_pred):
        cell_counts[(t, p)] += 1
        pred_counts[p] += 1
        true_counts[t] += 1

    tp = sum(_c2(c) for c in cell_counts.values())
    pred_pairs = sum(_c2(c) for c in pred_counts.values())  # TP + FP
    true_pairs = sum(_c2(c) for c in true_counts.values())  # TP + FN

    precision = 1.0 if pred_pairs == 0 else tp / pred_pairs
    recall = 1.0 if true_pairs == 0 else tp / true_pairs
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


def standard_metrics(y_true: List[str], y_pred: List[int]) -> Dict[str, float]:
    """Return the standard clustering metrics dict.

    Exact keys: ``homogeneity``, ``completeness``, ``v_measure``,
    ``adjusted_rand_index``, ``adjusted_mutual_info``, ``pairwise_precision``,
    ``pairwise_recall``, ``pairwise_f1``. Uses ``sklearn.metrics``; pairwise is the
    BCubed-style group-size formula. ARI/AMI may be slightly negative (allowed).
    """
    if len(y_true) == 0:
        # No data: define everything as 0.0 (callers guard against empty before here).
        keys = (
            "homogeneity", "completeness", "v_measure", "adjusted_rand_index",
            "adjusted_mutual_info", "pairwise_precision", "pairwise_recall", "pairwise_f1",
        )
        return {k: 0.0 for k in keys}

    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    p, r, f1 = _pairwise_bcubed(y_true, y_pred)
    return {
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "v_measure": float(v_measure),
        "adjusted_rand_index": float(ari),
        "adjusted_mutual_info": float(ami),
        "pairwise_precision": float(p),
        "pairwise_recall": float(r),
        "pairwise_f1": float(f1),
    }


# --------------------------------------------------------------------------- #
# Plain-language metrics
# --------------------------------------------------------------------------- #

def _candidate_new_arrays(df, record_ids: List[str]) -> List[int]:
    """Align ``is_candidate_new`` (0/1) to the evaluated ``record_ids`` order."""
    flag_by_id: Dict[str, int] = {}
    for rid, flag in zip(df["record_id"], df["is_candidate_new"]):
        if flag is None or (isinstance(flag, float) and flag != flag):
            flag_by_id[str(rid)] = 0
        else:
            flag_by_id[str(rid)] = int(flag)
    return [flag_by_id.get(rid, 0) for rid in record_ids]


def plain_language_metrics(y_true: List[str], y_pred: List[int], df) -> Dict[str, Any]:
    """Return the biologist-facing metric block.

    Keys: ``n_true_individuals``, ``n_found_clusters``, ``pct_photos_correctly_grouped``,
    ``n_merge_errors``, ``n_split_errors``, ``merged_individual_groups``,
    ``split_individuals``, ``n_candidate_new``, ``candidate_new_precision``,
    ``candidate_new_recall``, plus the per-individual/per-cluster breakdown tables and
    ``n_photos_noise``.

    ``df`` is the eval frame (used for ``is_candidate_new`` alignment); it is filtered to
    the evaluated ``record_ids`` by position via ``build_label_arrays`` upstream.
    """
    n = len(y_true)

    # ---- distinct true individuals / found clusters (non-noise) ----
    true_labels = set(y_true)
    found_clusters = {c for c in y_pred if c != NOISE_LABEL}
    n_true_individuals = len(true_labels)
    n_found_clusters = len(found_clusters)

    # ---- per-cluster grouping (noise treated as its own pseudo-cluster) ----
    cluster_to_labels: Dict[int, Counter] = defaultdict(Counter)
    label_to_clusters: Dict[str, set] = defaultdict(set)
    for t, p in zip(y_true, y_pred):
        cluster_to_labels[p][t] += 1
        label_to_clusters[t].add(p)

    # ---- pct_photos_correctly_grouped (majority-correct, 0..100) ----
    correct = 0
    for p, label_counts in cluster_to_labels.items():
        majority_label, _ = label_counts.most_common(1)[0]
        correct += label_counts[majority_label]
    pct_correct = 100.0 * correct / n if n else 0.0

    # ---- merge errors: clusters spanning >1 true label (noise excluded) ----
    merged_groups: List[List[str]] = []
    n_merge_errors = 0
    for p in sorted(cluster_to_labels, key=lambda c: (c == NOISE_LABEL, c)):
        if p == NOISE_LABEL:
            continue
        labels_here = sorted(cluster_to_labels[p].keys())
        if len(labels_here) > 1:
            n_merge_errors += 1
            merged_groups.append(labels_here)

    # ---- split errors: true labels in >1 distinct cluster (noise excluded) ----
    split_individuals: List[str] = []
    for t in sorted(label_to_clusters):
        clusters_here = {c for c in label_to_clusters[t] if c != NOISE_LABEL}
        if len(clusters_here) > 1:
            split_individuals.append(t)
    n_split_errors = len(split_individuals)

    # ---- candidate-new precision/recall ----
    record_ids = list(df["record_id"]) if (df is not None and len(df)) else []
    # Re-derive evaluated record_ids in the SAME order as y_true via the frame is not
    # reliable; instead the caller passes a frame already restricted to evaluated rows.
    cand_flags = _candidate_new_arrays(df, _evaluated_record_ids(df, y_true, y_pred))
    gt_singletons = {t for t, c in Counter(y_true).items() if c == 1}
    is_singleton = [1 if t in gt_singletons else 0 for t in y_true]

    n_candidate_new = sum(cand_flags)
    n_singletons = sum(is_singleton)

    if n_candidate_new == 0:
        candidate_new_precision: Optional[float] = None
    else:
        tp_prec = sum(1 for f, s in zip(cand_flags, is_singleton) if f == 1 and s == 1)
        candidate_new_precision = tp_prec / n_candidate_new

    if n_singletons == 0:
        candidate_new_recall: Optional[float] = None
    else:
        tp_rec = sum(1 for f, s in zip(cand_flags, is_singleton) if f == 1 and s == 1)
        candidate_new_recall = tp_rec / n_singletons

    # ---- breakdown tables ----
    per_cluster = _build_per_cluster(cluster_to_labels)
    per_individual = _build_per_individual(label_to_clusters, cluster_to_labels)

    n_photos_noise = sum(1 for p in y_pred if p == NOISE_LABEL)

    return {
        "n_true_individuals": n_true_individuals,
        "n_found_clusters": n_found_clusters,
        "pct_photos_correctly_grouped": float(pct_correct),
        "n_merge_errors": n_merge_errors,
        "n_split_errors": n_split_errors,
        "merged_individual_groups": merged_groups,
        "split_individuals": split_individuals,
        "n_candidate_new": int(n_candidate_new),
        "candidate_new_precision": candidate_new_precision,
        "candidate_new_recall": candidate_new_recall,
        "per_individual": per_individual,
        "per_cluster": per_cluster,
        "n_photos_noise": int(n_photos_noise),
    }


def _evaluated_record_ids(df, y_true: List[str], y_pred: List[int]) -> List[str]:
    """Recover the evaluated record_ids in y_true/y_pred order from the frame.

    ``plain_language_metrics`` receives a frame already restricted to the evaluated rows
    (in the same order build_label_arrays produced), so we rebuild the id list by
    re-running the same evaluable filter. This keeps the candidate-new alignment exact
    regardless of how the frame was sliced.
    """
    ids: List[str] = []
    if df is None or len(df) == 0:
        return ids
    for rid, gt_label, cid in zip(df["record_id"], df["gt_label"], df["cluster_id"]):
        if gt_label is None:
            continue
        if isinstance(gt_label, float) and gt_label != gt_label:
            continue
        if cid is None:
            continue
        if isinstance(cid, float) and cid != cid:
            continue
        ids.append(str(rid))
    return ids


def _build_per_cluster(cluster_to_labels: Dict[int, Counter]) -> List[Dict[str, Any]]:
    """One row per discovered cluster (noise included as cluster_id == -1).

    Row keys: ``cluster_id``, ``n_photos``, ``dominant_gt_label``, ``purity``,
    ``n_true_individuals``, ``is_merge``.
    """
    rows: List[Dict[str, Any]] = []
    for p in sorted(cluster_to_labels, key=lambda c: (c == NOISE_LABEL, c)):
        label_counts = cluster_to_labels[p]
        n_photos = sum(label_counts.values())
        dominant_label, dominant_n = label_counts.most_common(1)[0]
        n_true = len(label_counts)
        rows.append({
            "cluster_id": int(p),
            "n_photos": int(n_photos),
            "dominant_gt_label": str(dominant_label),
            "purity": float(dominant_n / n_photos) if n_photos else 0.0,
            "n_true_individuals": int(n_true),
            "is_merge": bool(n_true > 1 and p != NOISE_LABEL),
        })
    return rows


def _build_per_individual(
    label_to_clusters: Dict[str, set], cluster_to_labels: Dict[int, Counter]
) -> List[Dict[str, Any]]:
    """One row per true individual.

    Row keys: ``gt_label``, ``n_photos``, ``n_clusters`` (distinct cluster_id the photos
    fell into, INCLUDING noise), ``dominant_cluster`` (the cluster holding the most of its
    photos), ``is_split`` (n distinct NON-noise clusters > 1).
    """
    # photos per (label) and per (label, cluster)
    label_cluster_counts: Dict[str, Counter] = defaultdict(Counter)
    for p, label_counts in cluster_to_labels.items():
        for t, c in label_counts.items():
            label_cluster_counts[t][p] += c

    rows: List[Dict[str, Any]] = []
    for t in sorted(label_to_clusters):
        cluster_counts = label_cluster_counts[t]
        n_photos = sum(cluster_counts.values())
        clusters_here = set(cluster_counts.keys())
        n_clusters = len(clusters_here)
        # dominant cluster = cluster holding most photos of this individual; tie-break on
        # cluster id ascending for determinism.
        dominant_cluster = max(
            cluster_counts.items(), key=lambda kv: (kv[1], -kv[0])
        )[0]
        non_noise = {c for c in clusters_here if c != NOISE_LABEL}
        rows.append({
            "gt_label": str(t),
            "n_photos": int(n_photos),
            "n_clusters": int(n_clusters),
            "dominant_cluster": int(dominant_cluster),
            "is_split": bool(len(non_noise) > 1),
        })
    return rows


# --------------------------------------------------------------------------- #
# Top-level orchestration
# --------------------------------------------------------------------------- #

def evaluate_clustering(
    conn,
    dataset: str,
    *,
    flank_aware: bool = False,
    tag: str = "default",
    include_noise: bool = True,
) -> ClusteringReport:
    """Load frame, build arrays, compute both metric families, assemble breakdowns.

    Returns a fully populated ``ClusteringReport``. Raises ``ValueError`` with a clear
    message if the evaluated set is empty (no rows with both ``gt_identity`` and
    ``cluster_id``).
    """
    df = load_eval_frame(conn, dataset, flank_aware=flank_aware)

    n_photos_total = int(len(df))
    if n_photos_total:
        n_photos_labeled = int(sum(1 for g in df["gt_label"] if g is not None))
        n_photos_clustered = int(
            sum(
                1
                for c in df["cluster_id"]
                if c is not None and not (isinstance(c, float) and c != c)
            )
        )
    else:
        n_photos_labeled = 0
        n_photos_clustered = 0

    y_true, y_pred, record_ids = build_label_arrays(df, include_noise=include_noise)

    if len(y_true) == 0:
        raise ValueError(
            f"No evaluated rows: need both gt_identity and cluster_id for dataset={dataset}"
        )

    # Restrict the frame to the evaluated rows (in build_label_arrays' order) so the
    # candidate-new alignment in plain_language_metrics is exact.
    eval_frame = _restrict_frame(df, record_ids, include_noise=include_noise)

    std = standard_metrics(y_true, y_pred)
    plain = plain_language_metrics(y_true, y_pred, eval_frame)

    n_photos_evaluated = int(len(y_true))
    n_photos_noise = int(plain["n_photos_noise"])

    report = ClusteringReport(
        dataset=dataset,
        tag=tag,
        flank_aware=bool(flank_aware),
        n_photos_total=n_photos_total,
        n_photos_labeled=n_photos_labeled,
        n_photos_clustered=n_photos_clustered,
        n_photos_evaluated=n_photos_evaluated,
        n_photos_noise=n_photos_noise,
        n_true_individuals=int(plain["n_true_individuals"]),
        n_found_clusters=int(plain["n_found_clusters"]),
        pct_photos_correctly_grouped=float(plain["pct_photos_correctly_grouped"]),
        n_merge_errors=int(plain["n_merge_errors"]),
        n_split_errors=int(plain["n_split_errors"]),
        merged_individual_groups=plain["merged_individual_groups"],
        split_individuals=plain["split_individuals"],
        n_candidate_new=int(plain["n_candidate_new"]),
        candidate_new_precision=plain["candidate_new_precision"],
        candidate_new_recall=plain["candidate_new_recall"],
        homogeneity=std["homogeneity"],
        completeness=std["completeness"],
        v_measure=std["v_measure"],
        adjusted_rand_index=std["adjusted_rand_index"],
        adjusted_mutual_info=std["adjusted_mutual_info"],
        pairwise_precision=std["pairwise_precision"],
        pairwise_recall=std["pairwise_recall"],
        pairwise_f1=std["pairwise_f1"],
        per_individual=plain["per_individual"],
        per_cluster=plain["per_cluster"],
    )
    return report


def _restrict_frame(df, record_ids: List[str], *, include_noise: bool):
    """Return the eval frame restricted to ``record_ids`` (preserving derived gt_label).

    Used so candidate-new alignment sees exactly the evaluated rows. If pandas filtering
    is unavailable for any reason we fall back to the full frame (the alignment helper
    re-filters anyway), but normally we slice to keep semantics tight.
    """
    if df is None or len(df) == 0:
        return df
    keep = set(record_ids)
    mask = [str(rid) in keep for rid in df["record_id"]]
    restricted = df[mask].copy()
    return restricted


# --------------------------------------------------------------------------- #
# Report writers
# --------------------------------------------------------------------------- #

def save_report(
    report: ClusteringReport,
    out_dir: str = DEFAULT_OUT_DIR,
    *,
    write_csv: bool = False,
    write_html: bool = False,
) -> str:
    """Write the SINGLE JSON report ``<dataset>_<tag>.json`` (always) and optional CSV/HTML.

    Returns the JSON path. Creates ``out_dir``. The JSON top-level keys equal the
    ``ClusteringReport`` field names (snake_case); T10 reads the headline straight out of
    this one file. When ``write_csv`` the per-individual table is dumped to
    ``<dataset>_<tag>.csv``; when ``write_html`` a one-page summary is written to
    ``<dataset>_<tag>.html``.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = f"{report.dataset}_{report.tag}"

    json_path = out_path / f"{stem}.json"
    payload = report.to_dict()
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    if write_csv:
        _write_per_individual_csv(out_path / f"{stem}.csv", report.per_individual)

    if write_html:
        _write_html_summary(out_path / f"{stem}.html", report)

    return str(json_path)


def _write_per_individual_csv(path: Path, per_individual: List[Dict[str, Any]]) -> None:
    """Dump the per-individual breakdown table to CSV (stdlib csv, stable column order)."""
    import csv as _csv

    fieldnames = ["gt_label", "n_photos", "n_clusters", "dominant_cluster", "is_split"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_individual:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _write_html_summary(path: Path, report: ClusteringReport) -> None:
    """Write a minimal one-page HTML summary (no external deps; plain string template)."""
    import html as _html

    def esc(x: Any) -> str:
        return _html.escape(str(x))

    summary = esc(report.plain_language_summary()).replace("\n", "<br>\n")

    def table(rows: List[Dict[str, Any]], cols: List[str]) -> str:
        if not rows:
            return "<p>(none)</p>"
        head = "".join(f"<th>{esc(c)}</th>" for c in cols)
        body = ""
        for r in rows:
            body += "<tr>" + "".join(f"<td>{esc(r.get(c))}</td>" for c in cols) + "</tr>"
        return f"<table border='1' cellpadding='4'><tr>{head}</tr>{body}</table>"

    per_ind_cols = ["gt_label", "n_photos", "n_clusters", "dominant_cluster", "is_split"]
    per_clu_cols = ["cluster_id", "n_photos", "dominant_gt_label", "purity",
                    "n_true_individuals", "is_merge"]

    doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Clustering report — {esc(report.dataset)}_{esc(report.tag)}</title>"
        "<style>body{font-family:sans-serif;margin:2em;}"
        "table{border-collapse:collapse;margin:1em 0;}"
        "th{background:#eee;}</style></head><body>"
        f"<h1>Clustering evaluation — {esc(report.dataset)} "
        f"(<code>{esc(report.tag)}</code>)</h1>"
        f"<p>{summary}</p>"
        "<h2>Per-individual</h2>" + table(report.per_individual, per_ind_cols) +
        "<h2>Per-cluster</h2>" + table(report.per_cluster, per_clu_cols) +
        "</body></html>"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reid_demo.eval",
        description="Clustering evaluation harness for the open-set re-ID demo (T07).",
    )
    parser.add_argument("--dataset", required=True, help="dataset name to evaluate")
    parser.add_argument("--db", default=store.DEFAULT_DB_PATH, help="SQLite store path")
    parser.add_argument("--flank-aware", action="store_true",
                        help="use the {left,right,other} flank-bucketed GT label convention")
    parser.add_argument("--tag", default="default", help="tag for the output filename")
    parser.add_argument("--no-noise", action="store_true",
                        help="exclude cluster_id == -1 (noise) rows from the evaluation")
    parser.add_argument("--csv", action="store_true",
                        help="also write the per-individual table as CSV")
    parser.add_argument("--html", action="store_true",
                        help="also write a one-page HTML summary")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help="output directory for the report files")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint. Prints the plain-language summary, writes the JSON, prints its path.

    Exit 0 on success; non-zero with a clear message if the evaluated set is empty or the
    dataset is absent from the store.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        conn = store.connect(args.db, create=False)
    except Exception as exc:  # store not initialized / missing file
        print(f"ERROR: could not open store at {args.db!r}: {exc}", file=sys.stderr)
        return 2

    try:
        report = evaluate_clustering(
            conn,
            args.dataset,
            flank_aware=args.flank_aware,
            tag=args.tag,
            include_noise=not args.no_noise,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(report.plain_language_summary())
    json_path = save_report(
        report,
        out_dir=args.out_dir,
        write_csv=args.csv,
        write_html=args.html,
    )
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
