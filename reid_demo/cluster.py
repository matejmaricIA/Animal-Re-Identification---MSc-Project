"""reid_demo.cluster — open-set, flank-aware clustering engine (T05).

This is the DECISION LAYER of the lynx re-ID demo. Upstream tickets detect (T02),
classify (T03) and embed (T04) each crop; this engine takes the *unlabeled* pile of
MegaDescriptor global embeddings and discovers **how many distinct individuals are
present** (the count is unknown — there is no fixed gallery), assigns each crop a
``cluster_id`` + per-crop ``cluster_conf``, and flags un-matched crops as candidate-new
individuals. Results are written back into the T01 store; downstream tickets (T06
catalogue / T07 eval / T08 review / T10 runner) read them from the store.

WHAT THIS MODULE DOES (and only this):
  * Loads crop embeddings via the T04 API ``reid_demo.embed.get_embedding_matrix(
    normalize=True)`` (or, for tests, a single ``--embeddings`` pickle).
  * Clusters per FLANK BUCKET — the DATA_CONTRACT 3-bucket policy (D4): ``left`` and
    ``right`` (spot-bearing flanks) cluster in SEPARATE buckets; ``{front, back, down,
    unknown, ''}`` POOL into a single ``other`` bucket. Buckets are iterated in
    DETERMINISTIC sorted order (``left``, ``other``, ``right``) so cluster ids are
    reproducible/idempotent, and every non-negative id is globally unique across buckets.
  * Two cosine backends (D-default DBSCAN, plus threshold-based agglomerative).
  * Computes a per-crop confidence and flags candidate-new singletons.
  * Writes ``cluster_id`` / ``cluster_conf`` / ``is_candidate_new`` back via
    ``reid_demo.store.update_cluster`` — and NOTHING else.

EMBEDDING CONTRACT (D2 — authoritative on T04, not here): stored vectors are
MODEL-NATIVE dimension (1536 for the base ``megadescriptor-l-384``, 384 only for a
``linear_l2`` checkpoint) and are RAW / NOT L2-normalized. We obtain the matrix via
``get_embedding_matrix(normalize=True)`` (which L2-normalizes at read time), READ THE
DIMENSION FROM THE MATRIX, and additionally re-normalize defensively before any cosine
math. There is NO hard-coded ``384`` anywhere in this module.

PLUGGABLE AFFINITY (D8): the core (``cluster_embeddings`` / ``cluster_by_flank``) and
the driver (``run_clustering``) accept an OPTIONAL precomputed pairwise affinity — a
square ``(N, N)`` similarity matrix (higher = more likely the SAME individual) aligned
to the call's SORTED ``record_id`` list, or a provider callable ``(sorted_ids,
normalized_embeddings) -> (N, N)``. When ``None`` (default) we build the global cosine
affinity internally — the M1 backbone, ``--signals global``. When supplied (by T12 via
the T10 runner for ``global+fisher`` / ``full-funnel``) we cluster on it instead. This
module NEVER imports ``reid_demo.fisher`` (T11) or ``reid_demo.fusion`` (T12); the
dependency direction is one-way. Whatever the source, clustering consumes
``distance = 1 - affinity`` so the result of clustering a supplied affinity is identical
to clustering its equivalent internally-built affinity.

CONFIDENCE DEFINITION (documented contract, in ``[0, 1]``):
  * Crop in a cluster of size >= 2: the MEAN cosine similarity of the crop to the OTHER
    members of its cluster, floored at 0 (``max(0, mean_sim)``). If a fitted
    ``ScoreCalibrator`` is supplied, that mean similarity is mapped to ``P(same)`` via
    ``calibrator.predict_proba`` and clipped to ``[0, 1]``.
  * Singleton (final cluster size 1) or noise (``cluster_id == -1``): confidence is
    EXACTLY ``0.0`` (an unmatched crop has no corroborating support).

CANDIDATE-NEW RULE (single authoritative rule — D5): any crop whose FINAL cluster is a
singleton (size 1) OR is labeled ``-1`` (DBSCAN noise) gets BOTH ``cluster_id = -1`` AND
``is_candidate_new = 1``; all others keep their ``>= 0`` id with ``is_candidate_new = 0``.
``is_candidate_new`` is the field downstream keys on; there is no "assign 0" for a lone
crop.

RE-RUN SAFETY (D5): T05 runs BEFORE T08 review. Re-running on a dataset recomputes
identical ``cluster_*`` for UNREVIEWED rows (deterministic/idempotent), but rows whose
``review_status != 'unreviewed'`` (human merges/splits) are PRESERVED — skipped from the
write and counted in ``n_review_preserved`` — unless ``force=True`` is passed.

Dependencies: stdlib + numpy + scikit-learn + ``reid_demo.store`` + (optionally, only if
``--calibrator`` is given) ``calibration.ScoreCalibrator``. No new third-party deps; no
image files, model loading, GPU, or network.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from reid_demo import store
from reid_demo.store import (
    connect,
    query_records,
    update_cluster,
)


# --------------------------------------------------------------------------- #
# Module-level constants (exact names — downstream tickets import these)
# --------------------------------------------------------------------------- #

DEFAULT_BACKEND: str = "dbscan"                  # one of CLUSTER_BACKENDS
CLUSTER_BACKENDS: set = {"dbscan", "agglomerative"}
DEFAULT_EPS: float = 0.30                         # DBSCAN cosine-distance eps (analyze_folder.py default)
DEFAULT_MIN_SAMPLES: int = 2                      # >=2 so a lone crop is noise -> candidate-new
DEFAULT_DISTANCE_THRESHOLD: float = 0.30          # agglomerative cosine-distance cut
NOISE_LABEL: int = -1                             # matches sklearn DBSCAN / T01 cluster_id == -1

#: The 3 DATA_CONTRACT flank buckets, in deterministic iteration order (D4).
FLANK_BUCKETS: Tuple[str, ...] = ("left", "other", "right")
_SPOT_FLANKS = {"left", "right"}

#: Confidence assigned to a singleton / noise crop (documented constant).
SINGLETON_CONFIDENCE: float = 0.0


# --------------------------------------------------------------------------- #
# Pluggable affinity (D8) typing
# --------------------------------------------------------------------------- #

# An OPTIONAL precomputed pairwise affinity. Either a (N, N) similarity matrix aligned to
# the SORTED image_ids, or a provider callable (sorted_ids, normalized_embeddings) ->
# (N, N) similarity. Higher = more likely the SAME individual. None (default) => build the
# global cosine affinity internally (backbone behavior).
AffinityProvider = Callable[[List[str], np.ndarray], np.ndarray]
Affinity = Union[np.ndarray, AffinityProvider]


# --------------------------------------------------------------------------- #
# Result dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class CropClustering:
    """Result of clustering one flank bucket (or the merged global result)."""

    image_ids: List[str]            # crop record_ids, aligned with the arrays below
    labels: np.ndarray              # int cluster_id per crop; -1 = noise/unassigned
    confidences: np.ndarray         # float [0,1] per crop
    is_candidate_new: np.ndarray    # int 0/1 per crop


@dataclass
class ClusterRunSummary:
    dataset: Optional[str]
    backend: str
    params: Dict[str, Any]
    n_crops: int
    n_clusters_total: int       # count of distinct cluster_id >= 0
    n_individuals: int          # clusters of size >= 2 (a confirmed individual)
    n_candidate_new: int        # crops flagged is_candidate_new == 1
    n_noise: int                # crops with cluster_id == -1
    per_flank: Dict[str, Dict[str, int]]   # bucket -> {crops, clusters, candidate_new}
    n_review_preserved: int     # rows skipped because review_status != 'unreviewed'
    sentence: str               # human-readable one-liner


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #

def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows with an epsilon guard (mirrors the repo idiom). Zero rows stay
    zero. Defensive: stored vectors are RAW, and a supplied matrix may not be unit-norm."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12).astype(np.float32)
    return (matrix / norms).astype(np.float32)


def _stack_sorted(embeddings: Dict[str, np.ndarray]) -> Tuple[List[str], np.ndarray]:
    """Sort keys (stable, deterministic) and stack into an (N, D) float32 matrix.

    The dimension D is read from the vectors themselves (never hard-coded). All vectors
    must share one dimension; a mismatch raises ValueError.
    """
    ids = sorted(embeddings.keys())
    if not ids:
        return [], np.empty((0, 0), dtype=np.float32)
    first = np.asarray(embeddings[ids[0]], dtype=np.float32).reshape(-1)
    dim = int(first.shape[0])
    mat = np.zeros((len(ids), dim), dtype=np.float32)
    for i, rid in enumerate(ids):
        vec = np.asarray(embeddings[rid], dtype=np.float32).reshape(-1)
        if vec.shape[0] != dim:
            raise ValueError(
                f"embedding dim mismatch for {rid!r}: expected {dim}, got {vec.shape[0]}"
            )
        mat[i] = vec
    return ids, mat


def _flank_bucket(orientation: Optional[str]) -> str:
    """Map a raw orientation to its DATA_CONTRACT bucket (D4).

    ``left -> 'left'``, ``right -> 'right'``; ``{front, back, down, unknown, '', None}``
    and any non-canonical value pool into ``'other'``.
    """
    if orientation in _SPOT_FLANKS:
        return orientation
    return "other"


# --------------------------------------------------------------------------- #
# Affinity construction / validation (D8)
# --------------------------------------------------------------------------- #

def _affinity_to_distance(
    sorted_ids: List[str],
    matrix: np.ndarray,
    affinity: Optional[Affinity],
) -> np.ndarray:
    """Build the (N, N) cosine-DISTANCE matrix the backends consume.

    ``affinity`` semantics (D8, signal-agnostic): higher value = more likely the SAME
    individual. ``distance = 1 - affinity``. When ``affinity is None`` we build the global
    cosine SIMILARITY internally (``normalized @ normalized.T``) so the default backbone
    path and any externally-supplied equivalent affinity yield identical labels.
    """
    n = matrix.shape[0]
    if affinity is None:
        sim = matrix @ matrix.T
    elif callable(affinity):
        sim = np.asarray(affinity(sorted_ids, matrix), dtype=np.float64)
        _validate_affinity_matrix(sim, n, source="affinity provider")
    else:
        sim = np.asarray(affinity, dtype=np.float64)
        _validate_affinity_matrix(sim, n, source="affinity matrix")

    dist = 1.0 - sim
    # Cosine distance is non-negative; clip float noise and force an exact-zero diagonal so
    # every point is its own neighbour (DBSCAN min_samples counts self).
    dist = np.clip(dist, 0.0, None)
    dist = 0.5 * (dist + dist.T)          # defensive symmetrization
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float64)


def _validate_affinity_matrix(sim: np.ndarray, n: int, *, source: str) -> None:
    """Raise ValueError unless ``sim`` is a square, ~symmetric (N, N) matrix."""
    if sim.ndim != 2 or sim.shape != (n, n):
        raise ValueError(
            f"{source} must be a square ({n}, {n}) matrix aligned to the sorted ids; "
            f"got shape {sim.shape}."
        )
    if not np.allclose(sim, sim.T, atol=1e-3):
        raise ValueError(f"{source} must be (approximately) symmetric.")


# --------------------------------------------------------------------------- #
# Confidence (pure)
# --------------------------------------------------------------------------- #

def assignment_confidence(
    embeddings: np.ndarray,          # (N, D); re-normalized defensively before cosine
    labels: np.ndarray,              # (N,) cluster ids, -1 = noise
    *,
    calibrator: Optional["ScoreCalibrator"] = None,
) -> np.ndarray:
    """Per-crop confidence in ``[0, 1]`` (see the module docstring for the contract).

    For a crop in a cluster of size >= 2: the mean cosine similarity to the OTHER members
    of its cluster, floored at 0; optionally mapped to ``P(same)`` through ``calibrator``.
    For a singleton (size 1) or noise (``-1``): exactly ``SINGLETON_CONFIDENCE`` (``0.0``).
    Deterministic given fixed inputs.
    """
    labels = np.asarray(labels)
    n = labels.shape[0]
    conf = np.zeros(n, dtype=np.float64)
    if n == 0:
        return conf

    X = _l2_normalize_rows(np.asarray(embeddings, dtype=np.float32)).astype(np.float64)

    # Group member indices by (non-negative) label.
    members_by_label: Dict[int, List[int]] = {}
    for i, lab in enumerate(labels.tolist()):
        if lab >= 0:
            members_by_label.setdefault(int(lab), []).append(i)

    for lab, idxs in members_by_label.items():
        if len(idxs) < 2:
            continue  # singleton -> confidence stays 0.0
        sub = X[idxs]                       # (m, D)
        sims = sub @ sub.T                  # (m, m) cosine sims (rows are unit-norm)
        m = len(idxs)
        # mean similarity to the OTHER members = (row_sum - self_sim) / (m - 1)
        row_sums = sims.sum(axis=1)
        self_sims = np.diag(sims)
        mean_others = (row_sums - self_sims) / (m - 1)
        for local, gi in enumerate(idxs):
            conf[gi] = max(0.0, float(mean_others[local]))

    if calibrator is not None:
        # Map the raw mean-similarity of cluster members to P(same). Singletons/noise stay
        # at SINGLETON_CONFIDENCE (an unmatched crop has no support to calibrate).
        member_mask = np.array(
            [labels[i] >= 0 and len(members_by_label.get(int(labels[i]), [])) >= 2
             for i in range(n)],
            dtype=bool,
        )
        if member_mask.any():
            calibrated = np.asarray(
                calibrator.predict_proba(conf[member_mask]), dtype=np.float64
            ).reshape(-1)
            conf[member_mask] = np.clip(calibrated, 0.0, 1.0)

    return np.clip(conf, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Core clustering of one homogeneous bucket (pure, no DB)
# --------------------------------------------------------------------------- #

def cluster_embeddings(
    embeddings: Dict[str, np.ndarray],
    *,
    backend: str = DEFAULT_BACKEND,
    eps: float = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    calibrator: Optional["ScoreCalibrator"] = None,
    affinity: Optional[Affinity] = None,
    seed: int = 42,
) -> CropClustering:
    """Cluster a single homogeneous group of embeddings (one bucket).

    Embeddings are model-native dim (1536 base / 384 linear_l2 checkpoint) and are NOT
    assumed pre-normalized — they are re-normalized defensively before cosine math, and D
    is read from the array. Ids are sorted for a stable, deterministic order.

    Affinity (D8, PLUGGABLE): if ``affinity`` is None (default) build the GLOBAL cosine
    affinity internally (``--signals global`` backbone). A precomputed ``(N, N)`` matrix
    MUST be aligned to this call's SORTED ids; a provider callable is invoked as
    ``affinity(sorted_ids, normalized_embeddings) -> (N, N)``. A supplied affinity is a
    same-individual similarity (higher = more similar); the cosine-distance backends
    consume ``distance = 1 - affinity``. Such affinities come from T12 (fused
    global+Fisher, calibrated; or GV-refined for full-funnel) via the T10 runner — this
    module NEVER imports T11/T12. A matrix affinity is validated (square, ~symmetric,
    sized N); otherwise ValueError.

    Returns labels (>= 0 individuals, ``-1`` noise/singletons), per-crop confidence in
    ``[0, 1]``, and candidate-new flags (singletons and noise -> ``cluster_id = -1`` AND
    ``is_candidate_new = 1``). No DB, no I/O, no import of T11/T12.
    """
    if backend not in CLUSTER_BACKENDS:
        raise ValueError(
            f"unknown backend {backend!r}; must be one of {sorted(CLUSTER_BACKENDS)}"
        )

    ids, mat = _stack_sorted(embeddings)
    n = len(ids)

    # Empty / single-crop short-circuits (avoid backend edge cases; a lone crop is always a
    # candidate-new singleton per D5).
    if n == 0:
        return CropClustering([], np.array([], dtype=int),
                              np.array([], dtype=float), np.array([], dtype=int))
    mat = _l2_normalize_rows(mat)
    if n == 1:
        return CropClustering(
            ids,
            np.array([NOISE_LABEL], dtype=int),
            np.array([SINGLETON_CONFIDENCE], dtype=float),
            np.array([1], dtype=int),
        )

    dist = _affinity_to_distance(ids, mat, affinity)
    raw_labels = _run_backend(
        dist, backend=backend, eps=eps, min_samples=min_samples,
        distance_threshold=distance_threshold,
    )

    final_labels = _collapse_singletons_and_relabel(raw_labels)
    confidences = assignment_confidence(mat, final_labels, calibrator=calibrator)
    candidate_new = (final_labels == NOISE_LABEL).astype(int)

    return CropClustering(ids, final_labels, confidences, candidate_new)


def _run_backend(
    dist: np.ndarray,
    *,
    backend: str,
    eps: float,
    min_samples: int,
    distance_threshold: float,
) -> np.ndarray:
    """Run the selected backend on a precomputed cosine-DISTANCE matrix.

    Both backends are deterministic for a fixed input order. ``metric='precomputed'`` makes
    the DBSCAN path identical to ``metric='cosine'`` on unit-norm vectors while letting a
    pluggable affinity flow through the SAME code path.
    """
    from sklearn.cluster import DBSCAN, AgglomerativeClustering

    if backend == "dbscan":
        model = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
        return model.fit(dist).labels_.astype(int)

    # agglomerative: threshold-based, no pre-specified n_clusters. average linkage is the
    # only meaningful choice for a precomputed cosine-distance matrix (ward needs Euclidean).
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    return model.fit(dist).labels_.astype(int)


def _collapse_singletons_and_relabel(raw_labels: np.ndarray) -> np.ndarray:
    """Apply the candidate-new rule and relabel survivors to consecutive ids.

    Any cluster of size 1 (e.g. an agglomerative outlier) OR a ``-1`` (DBSCAN noise) crop
    collapses to ``NOISE_LABEL``. Surviving clusters (size >= 2) are relabeled to a
    consecutive ``0..m-1`` (ascending original id) so that ``cluster_by_flank`` can offset
    cleanly without id gaps. Deterministic.
    """
    raw_labels = np.asarray(raw_labels, dtype=int)
    sizes = Counter(raw_labels.tolist())
    survivors = sorted(
        lab for lab in sizes if lab != NOISE_LABEL and sizes[lab] >= 2
    )
    remap = {old: new for new, old in enumerate(survivors)}
    out = np.full(raw_labels.shape, NOISE_LABEL, dtype=int)
    for i, lab in enumerate(raw_labels.tolist()):
        if lab in remap:
            out[i] = remap[lab]
    return out


# --------------------------------------------------------------------------- #
# Flank-aware clustering (3-bucket policy, D4)
# --------------------------------------------------------------------------- #

def cluster_by_flank(
    embeddings: Dict[str, np.ndarray],
    orientations: Dict[str, Optional[str]],
    *,
    backend: str = DEFAULT_BACKEND,
    eps: float = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    calibrator: Optional["ScoreCalibrator"] = None,
    affinity: Optional[Affinity] = None,
    flank_policy: str = "separate",
    seed: int = 42,
) -> CropClustering:
    """Cluster crops with the DATA_CONTRACT 3-bucket flank policy (D4).

    ``flank_policy='separate'`` (default): bucket crops into ``{left, right, other}`` —
    ``left``/``right`` cluster separately and ``{front, back, down, unknown, ''}`` pool
    into ``other`` — then cluster each non-empty bucket independently in DETERMINISTIC
    sorted order (``left``, ``other``, ``right``) and concatenate with GLOBALLY UNIQUE
    cluster ids (each bucket's ``>= 0`` labels offset past the previous; noise stays
    ``-1``). Left and right of the same animal therefore NEVER share a cluster_id.

    ``flank_policy='ignore'``: cluster all crops together (flank-blind). This may merge
    the left+right flanks of one animal into a single cluster — acceptable for
    front-on tigers (ATRW) or ablations, WRONG for lynx/leopard — hence ``separate`` is
    the default.

    Affinity (D8): forwarded to each bucket's ``cluster_embeddings`` call. A provider
    callable is invoked per bucket on that bucket's SORTED ids + normalized embeddings (so
    cross-flank pairs never enter a within-bucket affinity). A precomputed MATRIX affinity
    must be a single GLOBAL ``(N, N)`` similarity aligned to the FULL sorted id set; the
    within-bucket submatrix is sliced per bucket before clustering. None (default) builds
    global cosine per bucket. Never imports T11/T12.
    """
    if backend not in CLUSTER_BACKENDS:
        raise ValueError(
            f"unknown backend {backend!r}; must be one of {sorted(CLUSTER_BACKENDS)}"
        )
    if flank_policy not in ("separate", "ignore"):
        raise ValueError(
            f"unknown flank_policy {flank_policy!r}; must be 'separate' or 'ignore'"
        )

    # Global sorted id order (used both for the 'ignore' path and for slicing a matrix
    # affinity into per-bucket submatrices).
    global_ids = sorted(embeddings.keys())
    global_pos = {rid: i for i, rid in enumerate(global_ids)}

    # A matrix affinity is GLOBAL — validate once against the full id set so we can slice it.
    global_sim: Optional[np.ndarray] = None
    if affinity is not None and not callable(affinity):
        global_sim = np.asarray(affinity, dtype=np.float64)
        _validate_affinity_matrix(global_sim, len(global_ids), source="affinity matrix")

    if flank_policy == "ignore":
        # One group; a global matrix affinity is already aligned to sorted(embeddings).
        return cluster_embeddings(
            embeddings, backend=backend, eps=eps, min_samples=min_samples,
            distance_threshold=distance_threshold, calibrator=calibrator,
            affinity=global_sim if global_sim is not None else affinity, seed=seed,
        )

    # Bucket the ids (deterministic).
    buckets: Dict[str, List[str]] = {b: [] for b in FLANK_BUCKETS}
    for rid in global_ids:
        buckets[_flank_bucket(orientations.get(rid))].append(rid)

    out_ids: List[str] = []
    out_labels: List[int] = []
    out_conf: List[float] = []
    out_cand: List[int] = []
    next_id = 0

    for bucket in FLANK_BUCKETS:                 # deterministic sorted order
        bids = buckets[bucket]
        if not bids:
            continue
        sub_emb = {rid: embeddings[rid] for rid in bids}

        # Resolve the per-bucket affinity.
        sub_affinity: Optional[Affinity]
        if global_sim is not None:
            idx = [global_pos[rid] for rid in sorted(bids)]
            sub_affinity = global_sim[np.ix_(idx, idx)]
        else:
            sub_affinity = affinity   # None or a provider callable (invoked per bucket)

        cc = cluster_embeddings(
            sub_emb, backend=backend, eps=eps, min_samples=min_samples,
            distance_threshold=distance_threshold, calibrator=calibrator,
            affinity=sub_affinity, seed=seed,
        )

        labels = cc.labels.copy()
        mask = labels >= 0
        labels[mask] += next_id
        n_clusters = int(len(set(cc.labels[cc.labels >= 0].tolist())))
        next_id += n_clusters

        out_ids.extend(cc.image_ids)
        out_labels.extend(labels.tolist())
        out_conf.extend(cc.confidences.tolist())
        out_cand.extend(cc.is_candidate_new.tolist())

    return CropClustering(
        out_ids,
        np.asarray(out_labels, dtype=int),
        np.asarray(out_conf, dtype=float),
        np.asarray(out_cand, dtype=int),
    )


# --------------------------------------------------------------------------- #
# Store-integrated driver
# --------------------------------------------------------------------------- #

def _load_calibrator(calibrator_path: Optional[str]) -> Optional["ScoreCalibrator"]:
    """Optionally load a pre-fit ScoreCalibrator (consume only — never fit; fitting needs
    GT and is T07's concern). Lazy import keeps the module light when unused."""
    if not calibrator_path:
        return None
    from calibration import ScoreCalibrator
    return ScoreCalibrator.load(calibrator_path)


def _resolve_embeddings(
    conn,
    records,
    *,
    dataset: Optional[str],
    embeddings_path: Optional[str],
) -> Dict[str, np.ndarray]:
    """Build a ``{record_id -> vector}`` dict for the given records.

    Default path: obtain the matrix via the T04 API ``get_embedding_matrix(normalize=True)``
    (dim read from the matrix; never assume 384/unit-norm) and restrict to ``records``.
    Override path (``embeddings_path``): load that single pickle and key by each record's
    ``embedding_ref``. Records whose vector cannot be resolved are simply omitted (the
    caller counts them as skipped).
    """
    wanted = {r.record_id: r for r in records if r.embedding_ref is not None}
    if not wanted:
        return {}

    if embeddings_path is not None:
        from reid_demo.embed import load_embeddings
        cache = load_embeddings(embeddings_path)
        emb: Dict[str, np.ndarray] = {}
        for rid, r in wanted.items():
            ref = r.embedding_ref
            if ref in cache:
                emb[rid] = np.asarray(cache[ref], dtype=np.float32).reshape(-1)
        return emb

    from reid_demo.embed import get_embedding_matrix
    matrix, ids = get_embedding_matrix(conn, dataset=dataset, normalize=True)
    pos = {rid: i for i, rid in enumerate(ids)}
    return {rid: matrix[pos[rid]] for rid in wanted if rid in pos}


def run_clustering(
    db_path: str = None,
    *,
    dataset: Optional[str] = None,
    embeddings_path: Optional[str] = None,
    backend: str = DEFAULT_BACKEND,
    eps: float = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    flank_policy: str = "separate",
    calibrator_path: Optional[str] = None,
    species_filter: Optional[str] = None,
    require_embedding: bool = True,
    force: bool = False,
    dry_run: bool = False,
    affinity: Optional[Affinity] = None,   # D8: precomputed affinity OR provider (T10 wires T12 in)
    seed: int = 42,
) -> ClusterRunSummary:
    """Read records from the T01 store, cluster per the 3-bucket flank policy, and (unless
    ``dry_run``) write ``cluster_id`` / ``cluster_conf`` / ``is_candidate_new`` back via
    ``reid_demo.store.update_cluster``. Returns a :class:`ClusterRunSummary`.

    Rows are filtered by ``dataset`` and (when given) the store's ``species`` column
    (D7 — NOT ``species_kept``). Rows missing an embedding are skipped and counted. Re-run
    safety (D5): rows with ``review_status != 'unreviewed'`` are PRESERVED (not
    overwritten, reported in ``n_review_preserved``) unless ``force=True``. Among unreviewed
    rows the run is deterministic/idempotent.

    Affinity (D8): if given (a precomputed ``(N, N)`` similarity aligned to the SORTED ids
    of the clustered set, or a provider callable) clustering uses it instead of the internal
    global cosine. The T10 runner computes this from T12 for ``global+fisher`` /
    ``full-funnel``; the default backbone passes ``None``.
    """
    if backend not in CLUSTER_BACKENDS:
        raise ValueError(
            f"unknown backend {backend!r}; must be one of {sorted(CLUSTER_BACKENDS)}"
        )
    if flank_policy not in ("separate", "ignore"):
        raise ValueError(
            f"unknown flank_policy {flank_policy!r}; must be 'separate' or 'ignore'"
        )

    db_path = db_path or store.DEFAULT_DB_PATH
    calibrator = _load_calibrator(calibrator_path)

    params: Dict[str, Any] = {
        "eps": eps,
        "min_samples": min_samples,
        "distance_threshold": distance_threshold,
        "flank_policy": flank_policy,
        "species_filter": species_filter,
        "calibrator": calibrator_path,
        "force": force,
        "dry_run": dry_run,
        "seed": seed,
        "affinity_supplied": affinity is not None,
    }

    conn = connect(db_path)
    try:
        records = query_records(
            conn, dataset=dataset, species=species_filter, order_by="record_id"
        )

        # Re-run safety: which rows may be (re)written?
        reviewed = [r for r in records if r.review_status != "unreviewed"]
        if force:
            to_cluster = list(records)
            n_review_preserved = 0
        else:
            to_cluster = [r for r in records if r.review_status == "unreviewed"]
            n_review_preserved = len(reviewed)

        embeddings = _resolve_embeddings(
            conn, to_cluster, dataset=dataset, embeddings_path=embeddings_path
        )
        n_skipped_no_emb = len(to_cluster) - len(embeddings)
        params["n_skipped_no_embedding"] = n_skipped_no_emb

        if require_embedding and to_cluster and not embeddings:
            raise RuntimeError(
                f"no embeddings found for {len(to_cluster)} candidate record(s) in "
                f"dataset={dataset!r}; run T04 embedding first or pass --embeddings."
            )

        orientations = {r.record_id: r.orientation for r in to_cluster}
        result = cluster_by_flank(
            embeddings, orientations, backend=backend, eps=eps,
            min_samples=min_samples, distance_threshold=distance_threshold,
            calibrator=calibrator, affinity=affinity, flank_policy=flank_policy, seed=seed,
        )

        if not dry_run:
            for rid, lab, cf, cand in zip(
                result.image_ids, result.labels.tolist(),
                result.confidences.tolist(), result.is_candidate_new.tolist(),
            ):
                update_cluster(conn, rid, int(lab), float(cf), int(cand))

        summary = _build_summary(
            dataset=dataset, backend=backend, params=params, result=result,
            orientations=orientations, n_review_preserved=n_review_preserved,
        )
        return summary
    finally:
        conn.close()


def _build_summary(
    *,
    dataset: Optional[str],
    backend: str,
    params: Dict[str, Any],
    result: CropClustering,
    orientations: Dict[str, Optional[str]],
    n_review_preserved: int,
) -> ClusterRunSummary:
    labels = result.labels
    n_crops = len(result.image_ids)

    # Cluster sizes over the FINAL (globally-unique) labels.
    sizes = Counter(int(l) for l in labels.tolist() if l >= 0)
    n_clusters_total = len(sizes)
    n_individuals = sum(1 for s in sizes.values() if s >= 2)
    n_candidate_new = int(result.is_candidate_new.sum())
    n_noise = int((labels == NOISE_LABEL).sum())

    # Per-flank breakdown keyed by the 3 buckets (always present, even if empty).
    per_flank: Dict[str, Dict[str, int]] = {
        b: {"crops": 0, "clusters": 0, "candidate_new": 0} for b in FLANK_BUCKETS
    }
    bucket_labels: Dict[str, set] = {b: set() for b in FLANK_BUCKETS}
    for rid, lab, cand in zip(
        result.image_ids, labels.tolist(), result.is_candidate_new.tolist()
    ):
        b = _flank_bucket(orientations.get(rid))
        per_flank[b]["crops"] += 1
        per_flank[b]["candidate_new"] += int(cand)
        if lab >= 0:
            bucket_labels[b].add(int(lab))
    for b in FLANK_BUCKETS:
        per_flank[b]["clusters"] = len(bucket_labels[b])

    sentence = (
        f"Found {n_clusters_total} individual{'' if n_clusters_total == 1 else 's'} "
        f"among {n_crops} crop{'' if n_crops == 1 else 's'}; "
        f"{n_candidate_new} candidate-new singleton{'' if n_candidate_new == 1 else 's'}"
    )
    if n_review_preserved:
        sentence += f" ({n_review_preserved} reviewed row(s) preserved)"
    sentence += "."

    return ClusterRunSummary(
        dataset=dataset,
        backend=backend,
        params=params,
        n_crops=n_crops,
        n_clusters_total=n_clusters_total,
        n_individuals=n_individuals,
        n_candidate_new=n_candidate_new,
        n_noise=n_noise,
        per_flank=per_flank,
        n_review_preserved=n_review_preserved,
        sentence=sentence,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reid_demo.cluster",
        description="T05 open-set, flank-aware clustering engine for the lynx re-ID demo.",
    )
    parser.add_argument("--db", default=None,
                        help=f"store DB path (default {store.DEFAULT_DB_PATH}).")
    parser.add_argument("--dataset", default=None,
                        help="cluster only records in this dataset (None = all).")
    parser.add_argument("--backend", default=DEFAULT_BACKEND,
                        help=f"clustering backend {sorted(CLUSTER_BACKENDS)} "
                             f"(default {DEFAULT_BACKEND}).")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS,
                        help=f"DBSCAN cosine-distance eps (default {DEFAULT_EPS}).")
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES,
                        help=f"DBSCAN min_samples (default {DEFAULT_MIN_SAMPLES}).")
    parser.add_argument("--distance-threshold", type=float, default=DEFAULT_DISTANCE_THRESHOLD,
                        help=f"agglomerative cosine-distance cut (default "
                             f"{DEFAULT_DISTANCE_THRESHOLD}).")
    parser.add_argument("--flank-policy", default="separate", choices=("separate", "ignore"),
                        help="'separate' (default; left/right/other) or 'ignore' (flank-blind).")
    parser.add_argument("--species", default=None,
                        help="filter rows by the store's `species` column (D7; NOT species_kept).")
    parser.add_argument("--calibrator", default=None,
                        help="optional pre-fit ScoreCalibrator .pkl for cluster_conf.")
    parser.add_argument("--embeddings", default=None,
                        help="override: a single embeddings .pkl used for all records.")
    parser.add_argument("--seed", type=int, default=42, help="determinism seed (default 42).")
    parser.add_argument("--force", action="store_true",
                        help="re-cluster reviewed rows too (default preserves them).")
    parser.add_argument("--dry-run", action="store_true",
                        help="compute + print the summary but write nothing to the store.")
    parser.add_argument("--json", action="store_true",
                        help="print the summary as machine-readable JSON.")
    return parser


def _main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Validate the backend up front so a bad value fails fast (and without needing a DB).
    if args.backend not in CLUSTER_BACKENDS:
        print(
            f"[cluster] FATAL: unknown backend {args.backend!r}; "
            f"must be one of {sorted(CLUSTER_BACKENDS)}",
            file=sys.stderr,
        )
        return 2

    db_path = args.db or store.DEFAULT_DB_PATH
    try:
        summary = run_clustering(
            db_path=db_path,
            dataset=args.dataset,
            embeddings_path=args.embeddings,
            backend=args.backend,
            eps=args.eps,
            min_samples=args.min_samples,
            distance_threshold=args.distance_threshold,
            flank_policy=args.flank_policy,
            calibrator_path=args.calibrator,
            species_filter=args.species,
            force=args.force,
            dry_run=args.dry_run,
            seed=args.seed,
        )
    except Exception as exc:
        print(f"[cluster] FATAL: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(dataclasses.asdict(summary), default=str))
    else:
        print(summary.sentence)
        print(
            f"  backend={summary.backend} crops={summary.n_crops} "
            f"clusters={summary.n_clusters_total} candidate_new={summary.n_candidate_new} "
            f"noise={summary.n_noise} review_preserved={summary.n_review_preserved}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
