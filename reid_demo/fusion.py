"""reid_demo.fusion — multi-signal fusion + geometric-verification reranking (T12).

The ACCURACY layer that sits ON TOP of the M1 global backbone (T04 embed -> T05
cluster), never replacing it (D8, binding). It produces two products the downstream
pipeline consumes:

  (a) ``build_fused_affinity(...)`` — a calibrated GLOBAL + FISHER pairwise affinity
      matrix (and a pluggable ``affinity_provider``) that T05 ingests via its
      precomputed-affinity argument. Per-pair value = the MEAN of calibrated ``P(same)``
      over the present signals (global cosine + Fisher cosine), exactly the Tier-2 recipe
      from ``predict.rank_by_local_score`` (calibrated mean, or clipped-raw-mean when no
      calibrator is shipped). Flank gating (D4): ``left``<->``right`` pairs are forced to
      ``0.0`` under ``flank_policy='separate'``.

  (b) ``gv_rerank(...)`` — a BUDGET-CAPPED geometric-verification reranker over a
      borderline-pair shortlist (NEVER N^2). For each shortlist pair it calls
      ``geometric_verification.compute_geometric_similarity`` once and turns the inlier
      count into a ``geom_score``. ``refine_affinity_with_gv`` then nudges the affinity
      (additive boost for strong GV, suppress for zero-inlier borderline) — the single
      seam where GV changes clustering. Without LightGlue/torch the reranker degrades to a
      no-op (the demo is never blocked on GV).

Plus a store-integrated driver ``run_fusion``, sidecar ``.npz`` / ``.json`` artifacts
under ``data/reid_demo/fusion/``, and a CLI.

BOUNDARY (D8): this module is imported one-way. ``reid_demo.cluster`` (T05) MUST NOT
import this module (no cycle). We CONSUME (never fit/modify):
``calibration.ScoreCalibrator.predict_proba``,
``geometric_verification.compute_geometric_similarity``,
``utils.distance_utils.fisher_distance``,
``reid_demo.embed.get_embedding_matrix`` (T04) and
``reid_demo.fisher.get_fisher_matrix`` (T11). T11 has NO ``get_local_features`` — the two
``_t11_*`` shims below adapt T11's read API for the Fisher matrix and for per-crop
keypoints/descriptors; tests monkeypatch them so no real T11/LightGlue/GPU is needed.

Heavy deps (torch / lightglue / h5py via the GV + T11 paths) are LAZY-imported inside the
functions that need them, mirroring ``reid_demo.embed`` / ``reid_demo.fisher``, so this
module imports cleanly under plain ``python3`` with none of them installed. The pure cores
(``build_fused_affinity`` / ``select_borderline_pairs`` / ``gv_rerank`` /
``refine_affinity_with_gv``) take numpy arrays + dicts only — fully testable with GV
stubbed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from reid_demo import store
from reid_demo.store import connect, query_records


# --------------------------------------------------------------------------- #
# Module-level constants (exact names — downstream tickets / T10 import these)
# --------------------------------------------------------------------------- #

DEFAULT_SIGNALS: str = "global+fisher"
SIGNAL_SETS: set = {"global+fisher", "full-funnel"}
DEFAULT_CALIBRATION_METHOD: str = "isotonic_pchip"

BORDERLINE_LOW: float = 0.35
BORDERLINE_HIGH: float = 0.65

DEFAULT_GV_PAIR_BUDGET: int = 2000
DEFAULT_GV_MATCHER: str = "lightglue"
DEFAULT_GV_METHOD: str = "disk"

GV_INLIER_BOOST: float = 0.20
GV_BORDERLINE_SUPPRESS: float = 0.20

FUSION_DIR: str = "data/reid_demo/fusion"

#: Mirrors constants.MIN_INLIERS / constants.ALPHA so tests need not import constants.py
#: (and so the module is import-clean even if constants.py grows heavy deps). Derived from
#: the repo defaults; geometric_verification.compute_geometric_similarity uses its OWN
#: constants internally — these are the values T12 reasons about (refinement threshold).
MIN_INLIERS: int = 10
ALPHA: float = 0.35

#: The 3 DATA_CONTRACT flank buckets (D4); ``other`` is compatible with everything.
_SPOT_FLANKS = {"left", "right"}


# --------------------------------------------------------------------------- #
# Result dataclasses (exact field order — downstream / JSON sidecar rely on it)
# --------------------------------------------------------------------------- #

@dataclass
class PairScore:
    """Geometric-verification score for ONE borderline pair (the T08 review signal)."""

    record_id_a: str
    record_id_b: str
    fused_prob: float            # the pre-GV fused affinity for the pair (in [0,1])
    n_inliers: int               # RANSAC inliers from compute_geometric_similarity (0 if GV no-op)
    gv_prob: Optional[float]     # calibrated P(same) from a GV calibrator, else None
    geom_score: float            # final GV-derived score in [0,1] (sort key; higher = stronger)
    bucket: str                  # provenance: "band" | "candidate_merge" | "band+candidate_merge"
    reason: str                  # short human-readable note (e.g. "gv_inliers=23")


@dataclass
class FusionResult:
    """Outcome of a ``run_fusion`` run."""

    dataset: Optional[str]
    signals: str
    record_ids: List[str]              # the SORTED record_id order of the affinity matrix
    affinity_path: Optional[str]       # .npz sidecar (None on --dry-run)
    pairs_path: Optional[str]          # pairs .json sidecar (None unless GV ran + persisted)
    n_crops: int
    n_borderline_pairs: int
    n_pairs_capped: int                # pairs dropped because the shortlist exceeded the budget
    gv_ran: bool                       # True iff GV actually executed (LightGlue/torch present)
    params: Dict[str, Any]
    sentence: str                      # human-readable one-liner


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #

def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows with an epsilon guard (mirrors the repo idiom). Zero rows stay
    zero. Defensive: a supplied matrix may not already be unit-norm."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12).astype(np.float32)
    return (matrix / norms).astype(np.float32)


def _flank_bucket(orientation: Optional[str]) -> str:
    """Map a raw orientation to its DATA_CONTRACT bucket (D4): ``left``/``right`` stay,
    everything else (``front``/``back``/``down``/``unknown``/``''``/``None``) -> ``other``."""
    if orientation in _SPOT_FLANKS:
        return orientation
    return "other"


def _cross_flank(b_a: str, b_b: str) -> bool:
    """True iff the two flank buckets are the spot-bearing pair ``{left, right}`` — the
    only combination that is incompatible (``other`` is compatible with everything)."""
    return {b_a, b_b} == _SPOT_FLANKS


def _calibrate_one(cal, raw: float) -> float:
    """Map one raw similarity to P(same) via ``cal.predict_proba([raw])[0]`` when a
    calibrator is supplied, else clip the raw value into ``[0, 1]`` (the Tier-2
    clipped-raw fallback from ``predict.rank_by_local_score`` lines 112-120)."""
    if cal is not None:
        return float(np.clip(float(np.asarray(cal.predict_proba([float(raw)])).reshape(-1)[0]), 0.0, 1.0))
    return float(np.clip(float(raw), 0.0, 1.0))


# --------------------------------------------------------------------------- #
# (a) Pure core: build the fused (global + Fisher) affinity matrix
# --------------------------------------------------------------------------- #

def build_fused_affinity(
    record_ids: Sequence[str],
    global_matrix: Optional[np.ndarray],
    fisher_matrix: Optional[np.ndarray],
    orientations: Optional[Dict[str, Optional[str]]] = None,
    *,
    calibrators: Optional[Dict[str, Any]] = None,
    flank_policy: str = "separate",
) -> np.ndarray:
    """Build the calibrated GLOBAL + FISHER pairwise affinity matrix (pure; no DB/model).

    ``record_ids`` is the SHARED order (typically already sorted) that BOTH matrices are
    aligned to. ``global_matrix`` / ``fisher_matrix`` are ``(N, D)`` per-signal embedding
    matrices; each is L2-normalized defensively and turned into a cosine-similarity matrix
    ``S = M @ M.T``. Either matrix may be ``None`` (signal absent) — at least one must be
    present.

    Per pair the affinity is the MEAN of the calibrated ``P(same)`` over the PRESENT
    signals, each mapped via ``cal.predict_proba([s])[0]`` (when a calibrator is supplied
    for that signal under key ``'global'`` / ``'fisher'``) or clipped-raw ``np.clip(s,0,1)``
    when no calibrator — exactly the Tier-2 recipe of ``predict.rank_by_local_score``.

    The result is ``(N, N)`` float32, symmetric, in ``[0, 1]``, with the diagonal forced
    to ``1.0`` (a crop is identical to itself). Under ``flank_policy='separate'`` any
    ``left``<->``right`` pair (per the D4 ``{left, right, other}`` map; ``other`` compatible
    with all) is forced to exactly ``0.0``. ``flank_policy='ignore'`` applies NO gating.

    Deterministic given fixed inputs.
    """
    if flank_policy not in ("separate", "ignore"):
        raise ValueError(
            f"unknown flank_policy {flank_policy!r}; must be 'separate' or 'ignore'"
        )

    ids = list(record_ids)
    n = len(ids)
    if n == 0:
        return np.empty((0, 0), dtype=np.float32)

    calibrators = calibrators or {}

    # Collect per-signal cosine-similarity matrices (defensively normalized).
    sims: List[Tuple[str, np.ndarray]] = []
    for key, mat in (("global", global_matrix), ("fisher", fisher_matrix)):
        if mat is None:
            continue
        arr = np.asarray(mat, dtype=np.float32)
        if arr.size == 0:
            continue
        if arr.ndim != 2 or arr.shape[0] != n:
            raise ValueError(
                f"{key} matrix must be (N, D) aligned to record_ids (N={n}); "
                f"got shape {arr.shape}."
            )
        norm = _l2_normalize_rows(arr)
        sims.append((key, (norm @ norm.T).astype(np.float64)))

    if not sims:
        raise ValueError(
            "build_fused_affinity needs at least one of global_matrix / fisher_matrix."
        )

    # Calibrate each signal's cosine elementwise, then mean over present signals.
    # Vectorize the calibrator over the flattened upper-triangle entries for speed, then
    # rebuild the symmetric matrix.
    iu = np.triu_indices(n, k=1)
    accum = np.zeros(iu[0].shape[0], dtype=np.float64)
    for key, S in sims:
        raw = S[iu]
        cal = calibrators.get(key)
        if cal is not None:
            proba = np.asarray(cal.predict_proba(raw), dtype=np.float64).reshape(-1)
            cal_vals = np.clip(proba, 0.0, 1.0)
        else:
            cal_vals = np.clip(raw, 0.0, 1.0)
        accum += cal_vals
    mean_vals = accum / float(len(sims))

    fused = np.zeros((n, n), dtype=np.float64)
    fused[iu] = mean_vals
    fused = fused + fused.T            # symmetrize (diagonal still 0 here)
    np.fill_diagonal(fused, 1.0)        # a crop is identical to itself

    # Flank gating (D4): force left<->right pairs to 0.0 under 'separate'.
    if flank_policy == "separate" and orientations is not None:
        buckets = np.array([_flank_bucket(orientations.get(rid)) for rid in ids])
        left_mask = buckets == "left"
        right_mask = buckets == "right"
        if left_mask.any() and right_mask.any():
            cross = np.outer(left_mask, right_mask) | np.outer(right_mask, left_mask)
            fused[cross] = 0.0

    fused = np.clip(fused, 0.0, 1.0)
    return fused.astype(np.float32)


def affinity_provider(
    record_ids: Sequence[str],
    fisher_matrix: Optional[np.ndarray],
    orientations: Optional[Dict[str, Optional[str]]] = None,
    *,
    calibrators: Optional[Dict[str, Any]] = None,
    flank_policy: str = "separate",
) -> Callable[[List[str], np.ndarray], np.ndarray]:
    """Build a T05-compatible pluggable-affinity provider.

    T05's pluggable-affinity contract (D8) is a callable
    ``(sorted_ids, normalized_embeddings) -> (N, N)`` similarity, invoked by
    ``reid_demo.cluster`` either globally (``flank_policy='ignore'``) or PER FLANK BUCKET
    (``'separate'`` — the provider receives only that bucket's ids + normalized global
    embeddings). This factory captures the Fisher matrix / orientations / calibrators and
    returns such a callable; on each call it slices the captured Fisher rows to the ids T05
    passes (so the fused affinity is built on EXACTLY the records of that call) and runs
    ``build_fused_affinity`` with the call's ``normalized_embeddings`` as the global signal.

    Because T05 already buckets by flank before invoking the provider, the provider itself
    is called with same-bucket ids only; we still pass ``flank_policy`` through so a single
    cross-flank slice (defensive) is gated. Returns a same-individual SIMILARITY (higher =
    more similar) exactly as T05 expects.
    """
    base_ids = list(record_ids)
    base_pos = {rid: i for i, rid in enumerate(base_ids)}
    fisher = None if fisher_matrix is None else np.asarray(fisher_matrix, dtype=np.float32)

    def _provider(sorted_ids: List[str], normalized_embeddings: np.ndarray) -> np.ndarray:
        call_ids = list(sorted_ids)
        global_sub = np.asarray(normalized_embeddings, dtype=np.float32)
        # Slice the captured Fisher rows to the records of THIS call (skip if any id is
        # unknown to the captured order — then Fisher is simply absent for this call).
        fisher_sub: Optional[np.ndarray] = None
        if fisher is not None and all(rid in base_pos for rid in call_ids):
            idx = [base_pos[rid] for rid in call_ids]
            fisher_sub = fisher[idx]
        return build_fused_affinity(
            call_ids,
            global_sub,
            fisher_sub,
            orientations,
            calibrators=calibrators,
            flank_policy=flank_policy,
        )

    return _provider


# --------------------------------------------------------------------------- #
# Borderline-pair selection (the cost guard — NEVER N^2 returned)
# --------------------------------------------------------------------------- #

def select_borderline_pairs(
    record_ids: Sequence[str],
    affinity: np.ndarray,
    orientations: Optional[Dict[str, Optional[str]]] = None,
    *,
    low: float = BORDERLINE_LOW,
    high: float = BORDERLINE_HIGH,
    prelim_labels: Optional[Dict[str, int]] = None,
    budget: int = DEFAULT_GV_PAIR_BUDGET,
    flank_policy: str = "separate",
    seed: int = 42,
) -> Tuple[List[Tuple[str, str, str]], int]:
    """Select the borderline pairs GV should verify — the budget-capped cost guard.

    A pair ``(a, b)`` (``a < b`` by index) is a candidate when EITHER:
      * its affinity lies in the borderline band ``[low, high]`` (``bucket='band'``), OR
      * ``prelim_labels`` marks it a candidate-merge: both crops are assigned (label >= 0)
        but to DIFFERENT preliminary clusters (``bucket='candidate_merge'``). A pair that
        is both gets ``bucket='band+candidate_merge'``.

    Cross-flank pairs (``left``<->``right`` under ``flank_policy='separate'``) are EXCLUDED
    — they can never merge. Candidates are ordered by ``|affinity - 0.5|`` ASCENDING (the
    most uncertain pairs first, where GV buys the most) and TRUNCATED to ``budget``.

    Returns ``(pairs, n_capped)`` where ``pairs`` is a list of ``(record_id_a,
    record_id_b, bucket)`` tuples (length ``<= budget``) and ``n_capped`` is how many
    candidates were dropped by the budget cut. NEVER returns all ``N(N-1)/2`` pairs unless
    the band/merge sets genuinely cover them. Deterministic (ties broken by index order).
    """
    if flank_policy not in ("separate", "ignore"):
        raise ValueError(
            f"unknown flank_policy {flank_policy!r}; must be 'separate' or 'ignore'"
        )
    ids = list(record_ids)
    n = len(ids)
    aff = np.asarray(affinity, dtype=np.float64)
    if n < 2:
        return [], 0
    if aff.shape != (n, n):
        raise ValueError(
            f"affinity must be ({n}, {n}) aligned to record_ids; got {aff.shape}."
        )

    buckets_f = (
        [_flank_bucket(orientations.get(rid)) for rid in ids]
        if (orientations is not None and flank_policy == "separate")
        else None
    )

    # Preliminary cluster id per record (for candidate-merge detection).
    plabels = None
    if prelim_labels is not None:
        plabels = [prelim_labels.get(rid, None) for rid in ids]

    candidates: List[Tuple[float, int, int, str]] = []  # (|aff-0.5|, i, j, bucket)
    for i in range(n):
        for j in range(i + 1, n):
            # Cross-flank exclusion.
            if buckets_f is not None and _cross_flank(buckets_f[i], buckets_f[j]):
                continue

            a = float(aff[i, j])
            in_band = low <= a <= high

            is_merge = False
            if plabels is not None:
                la, lb = plabels[i], plabels[j]
                if la is not None and lb is not None and la >= 0 and lb >= 0 and la != lb:
                    is_merge = True

            if not (in_band or is_merge):
                continue

            if in_band and is_merge:
                bucket = "band+candidate_merge"
            elif in_band:
                bucket = "band"
            else:
                bucket = "candidate_merge"

            candidates.append((abs(a - 0.5), i, j, bucket))

    # Order by |aff-0.5| ascending; ties deterministic by (i, j).
    candidates.sort(key=lambda t: (t[0], t[1], t[2]))

    n_total = len(candidates)
    budget = int(budget) if budget is not None else n_total
    if budget < 0:
        budget = 0
    kept = candidates[:budget]
    n_capped = max(0, n_total - len(kept))

    pairs = [(ids[i], ids[j], bucket) for (_d, i, j, bucket) in kept]
    return pairs, n_capped


# --------------------------------------------------------------------------- #
# (b) GV reranker over the shortlist (budget-capped; NEVER N^2)
# --------------------------------------------------------------------------- #

def _lightglue_available() -> bool:
    """True iff torch + lightglue import cleanly (the GV matcher can actually run).

    Mirrors ``geometric_verification._LIGHTGLUE_AVAILABLE`` but evaluated lazily here so
    importing ``reid_demo.fusion`` never imports torch. Any failure -> GV is a no-op."""
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    try:
        import lightglue  # noqa: F401
    except Exception:
        return False
    return True


def gv_rerank(
    pairs: Sequence[Tuple[str, str, str]],
    keypoints: Optional[Dict[str, Any]],
    descriptors: Optional[Dict[str, Any]],
    *,
    affinity_lookup: Optional[Dict[Tuple[str, str], float]] = None,
    fisher_distance_lookup: Optional[Dict[Tuple[str, str], float]] = None,
    gv_calibrator: Optional[Any] = None,
    use_lightglue: bool = True,
    method: str = DEFAULT_GV_METHOD,
    gv_matcher: Optional[str] = DEFAULT_GV_MATCHER,
    buckets: Optional[Dict[Tuple[str, str], str]] = None,
    budget: int = DEFAULT_GV_PAIR_BUDGET,
) -> List[PairScore]:
    """Geometric-verification rerank over a borderline shortlist — NEVER N^2.

    For EACH pair ``(a, b, bucket)`` in ``pairs`` (already capped upstream; truncated to
    ``budget`` here as a hard guard) call
    ``geometric_verification.compute_geometric_similarity(query_desc, query_kp, db_desc,
    db_kp, feature_distance, min_inliers=MIN_INLIERS, use_lightglue=..., method=...,
    gv_matcher=...)`` EXACTLY ONCE and read its ``n_inliers``. ``feature_distance`` is the
    pair's ``fisher_distance_lookup`` value (or ``1.0`` when absent — a neutral max
    distance). ``geom_score`` is ``gv_cal.predict_proba([log1p(n_inliers)])[0]`` (the
    ``train_late_fusion`` ``gv`` signal) when a GV calibrator is supplied, else
    ``min(n_inliers / 50, 1.0)`` (the ``geometric_verification._I90 = 50`` normalization).

    Graceful no-op: when LightGlue/torch are unavailable, OR a crop's keypoints/descriptors
    are missing, the pair yields ``n_inliers=0``, ``gv_prob=None`` and
    ``geom_score=fused_prob`` (the pre-GV affinity, so a missing GV signal never demotes a
    pair) WITHOUT raising. ``compute_geometric_similarity`` is then NOT called for that pair
    (so it is skippable in tests). When GV is available it IS called exactly once per pair
    — assert ``len(pairs)`` calls.

    Returns a ``List[PairScore]`` sorted ASCENDING by ``geom_score`` (the T08 review queue
    order: weakest geometric support first).
    """
    pair_list = list(pairs)
    budget = int(budget) if budget is not None else len(pair_list)
    if budget >= 0:
        pair_list = pair_list[:budget]

    affinity_lookup = affinity_lookup or {}
    fisher_distance_lookup = fisher_distance_lookup or {}
    buckets = buckets or {}

    gv_live = use_lightglue and _lightglue_available()

    out: List[PairScore] = []
    for entry in pair_list:
        a, b = entry[0], entry[1]
        bucket = entry[2] if len(entry) > 2 else buckets.get((a, b), buckets.get((b, a), "band"))

        fused_prob = float(
            affinity_lookup.get((a, b), affinity_lookup.get((b, a), 0.0))
        )

        q_kp = keypoints.get(a) if keypoints is not None else None
        d_kp = keypoints.get(b) if keypoints is not None else None
        q_desc = descriptors.get(a) if descriptors is not None else None
        d_desc = descriptors.get(b) if descriptors is not None else None
        have_feats = all(x is not None for x in (q_kp, d_kp, q_desc, d_desc))

        if not gv_live or not have_feats:
            # Graceful no-op: GV contributes nothing; keep the fused prob as the score.
            reason = "gv_unavailable" if not gv_live else "missing_local_features"
            out.append(PairScore(
                record_id_a=a, record_id_b=b, fused_prob=fused_prob,
                n_inliers=0, gv_prob=None, geom_score=fused_prob,
                bucket=bucket, reason=reason,
            ))
            continue

        # Feature distance for the GV combine (lower = more similar). Default 1.0 (neutral
        # max) when no Fisher distance is supplied for the pair.
        fd = float(fisher_distance_lookup.get((a, b), fisher_distance_lookup.get((b, a), 1.0)))

        from geometric_verification import compute_geometric_similarity  # lazy: pulls torch
        _final_distance, n_inliers = compute_geometric_similarity(
            q_desc, q_kp, d_desc, d_kp, fd,
            min_inliers=MIN_INLIERS,
            use_lightglue=use_lightglue,
            method=method,
            gv_matcher=gv_matcher,
        )
        n_inliers = int(n_inliers)

        if gv_calibrator is not None:
            gv_prob = float(np.clip(
                float(np.asarray(gv_calibrator.predict_proba([float(np.log1p(max(0, n_inliers)))])).reshape(-1)[0]),
                0.0, 1.0,
            ))
            geom_score = gv_prob
        else:
            gv_prob = None
            geom_score = float(min(n_inliers / 50.0, 1.0))

        out.append(PairScore(
            record_id_a=a, record_id_b=b, fused_prob=fused_prob,
            n_inliers=n_inliers, gv_prob=gv_prob, geom_score=geom_score,
            bucket=bucket, reason=f"gv_inliers={n_inliers}",
        ))

    out.sort(key=lambda ps: (ps.geom_score, ps.record_id_a, ps.record_id_b))
    return out


# --------------------------------------------------------------------------- #
# Boundary refinement — the ONLY seam where GV changes clustering
# --------------------------------------------------------------------------- #

def refine_affinity_with_gv(
    affinity: np.ndarray,
    record_ids: Sequence[str],
    pair_scores: Sequence[PairScore],
    *,
    boost: float = GV_INLIER_BOOST,
    suppress: float = GV_BORDERLINE_SUPPRESS,
    min_inliers: int = MIN_INLIERS,
) -> np.ndarray:
    """Return a NEW ``(N, N)`` affinity with the GV pair scores folded in (input NOT
    mutated).

    For each ``PairScore`` whose record ids are both present in ``record_ids``:
      * ``n_inliers >= min_inliers`` (strong geometric support) -> ``+boost`` (clamped to
        ``<= 1.0``) on BOTH symmetric entries.
      * ``n_inliers == 0`` (zero-inlier borderline; GV actively disagreed) -> ``-suppress``
        (clamped to ``>= 0.0``).
      * otherwise (weak-but-nonzero inliers) -> unchanged (no confident signal either way).

    The diagonal and every pair NOT in ``pair_scores`` are left exactly as in the input.
    Simple clamped additive nudge — no power-formula underflow. Symmetric in, symmetric out.
    """
    ids = list(record_ids)
    pos = {rid: i for i, rid in enumerate(ids)}
    out = np.array(affinity, dtype=np.float32, copy=True)  # NEW matrix (never mutate input)

    for ps in pair_scores:
        i = pos.get(ps.record_id_a)
        j = pos.get(ps.record_id_b)
        if i is None or j is None or i == j:
            continue
        if ps.n_inliers >= min_inliers:
            delta = float(boost)
        elif ps.n_inliers == 0:
            delta = -float(suppress)
        else:
            continue
        new_val = float(np.clip(out[i, j] + delta, 0.0, 1.0))
        out[i, j] = new_val
        out[j, i] = new_val

    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# Sidecar persistence (.npz affinity + .json pairs)
# --------------------------------------------------------------------------- #

def _affinity_npz_path(out_dir: str, dataset: Optional[str], signals: str) -> str:
    ds = dataset if dataset else "all"
    sig = signals.replace("+", "+")  # keep as-is; signals are constrained to SIGNAL_SETS
    return os.path.join(out_dir, f"{ds}_{sig}.npz")


def _pairs_json_path(out_dir: str, dataset: Optional[str], signals: str) -> str:
    ds = dataset if dataset else "all"
    return os.path.join(out_dir, f"{ds}_{signals}_pairs.json")


def _save_affinity(path: str, affinity: np.ndarray, record_ids: Sequence[str]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez(
        path,
        affinity=np.asarray(affinity, dtype=np.float32),
        record_ids=np.asarray(list(record_ids), dtype=object).astype(str),
    )


def load_affinity(path: str) -> Tuple[np.ndarray, List[str]]:
    """Load an affinity ``.npz`` sidecar -> ``(matrix (N,N) float32, record_ids list[str])``.

    Round-trips ``_save_affinity`` / the ``run_fusion`` output: the ``record_ids`` order is
    EXACTLY the SORTED order T05 must pass through alongside the matrix.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"affinity sidecar not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        matrix = np.asarray(data["affinity"], dtype=np.float32)
        record_ids = [str(x) for x in data["record_ids"].tolist()]
    return matrix, record_ids


def _save_pairs_json(path: str, pair_scores: Sequence[PairScore]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    # Already sorted ascending by geom_score upstream; re-sort defensively for the file
    # guarantee.
    ordered = sorted(pair_scores, key=lambda ps: (ps.geom_score, ps.record_id_a, ps.record_id_b))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([asdict(ps) for ps in ordered], fh, indent=2)


# --------------------------------------------------------------------------- #
# T11 read-API shims (T11 has NO get_local_features) — monkeypatched in tests
# --------------------------------------------------------------------------- #

def _t11_fisher_matrix(
    conn, *, dataset: Optional[str], normalize: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """Shim over T11's ``reid_demo.fisher.get_fisher_matrix`` -> ``(matrix, ids)``.

    Isolated so tests can MONKEYPATCH it with a synthetic Fisher matrix (no real T11 /
    feature pipeline / GPU). Lazy-imports T11 so plain imports stay light.
    """
    from reid_demo.fisher import get_fisher_matrix
    return get_fisher_matrix(conn, dataset=dataset, normalize=normalize)


def _t11_local_features(
    conn, record_ids: Sequence[str], *, dataset: Optional[str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Shim that resolves per-crop ``(keypoints, descriptors)`` for the GV reranker.

    T11 (``reid_demo.fisher``) caches local descriptors + keypoints in per-(dataset, method)
    HDF5 files (``descriptors.h5`` / ``keypoints.h5``, keyed by ``record_id``) under its
    descriptor dir. This shim reads those HDF5 caches for the requested ``record_ids`` and
    returns ``(keypoints_by_id, descriptors_by_id)``. It is isolated + lazy so tests can
    monkeypatch it with synthetic dicts (no real T11 / h5py / LightGlue needed), and so a
    missing cache degrades gracefully (returns ``({}, {})`` -> ``gv_rerank`` no-ops).
    """
    try:
        from reid_demo.fisher import _descriptor_dir, DEFAULT_METHOD  # lazy
    except Exception:
        return {}, {}

    desc_dir = _descriptor_dir(dataset, DEFAULT_METHOD)
    desc_h5 = os.path.join(desc_dir, "descriptors.h5")
    kp_h5 = os.path.join(desc_dir, "keypoints.h5")
    if not (os.path.exists(desc_h5) and os.path.exists(kp_h5)):
        return {}, {}

    try:
        import h5py  # lazy
    except Exception:
        return {}, {}

    wanted = set(record_ids)
    keypoints: Dict[str, Any] = {}
    descriptors: Dict[str, Any] = {}
    try:
        with h5py.File(kp_h5, "r") as f:
            for k in f.keys():
                if k in wanted:
                    keypoints[k] = np.asarray(f[k])
        with h5py.File(desc_h5, "r") as f:
            for k in f.keys():
                if k in wanted:
                    descriptors[k] = np.asarray(f[k])
    except Exception:
        return {}, {}
    return keypoints, descriptors


# --------------------------------------------------------------------------- #
# Calibrator loading (consume only — never fit)
# --------------------------------------------------------------------------- #

def _load_calibrators(calibrators_dir: Optional[str]) -> Dict[str, Any]:
    """Load pre-fit per-signal ``ScoreCalibrator``s from a directory (consume only).

    Looks for ``{signal}.pkl`` (or ``cal_{signal}.pkl``) for ``signal`` in
    ``{global, fisher, gv}``. Missing files are simply absent (the affinity falls back to
    clipped-raw for that signal). Returns ``{}`` when ``calibrators_dir`` is None or no
    calibrators are found. NEVER fits (that needs GT — T07's job).
    """
    if not calibrators_dir:
        return {}
    out: Dict[str, Any] = {}
    try:
        from calibration import ScoreCalibrator  # lazy
    except Exception:
        return {}
    for sig in ("global", "fisher", "gv"):
        for name in (f"{sig}.pkl", f"cal_{sig}.pkl"):
            p = os.path.join(calibrators_dir, name)
            if os.path.exists(p):
                try:
                    out[sig] = ScoreCalibrator.load(p)
                except Exception:
                    pass
                break
    return out


# --------------------------------------------------------------------------- #
# Store-integrated driver
# --------------------------------------------------------------------------- #

def _aligned_signal_matrices(
    conn,
    *,
    dataset: Optional[str],
    species_filter: Optional[str],
) -> Tuple[List[str], np.ndarray, np.ndarray, Dict[str, Optional[str]]]:
    """Load + ALIGN the global and Fisher matrices to ONE shared sorted record_id order.

    Returns ``(record_ids, global_matrix, fisher_matrix, orientations)`` where both
    matrices are row-aligned to ``record_ids`` (the SORTED intersection of the two signal
    id sets). Rows missing EITHER signal are warned about and skipped. ``species_filter``
    (D7 — the ``species`` column, NOT ``species_kept``) restricts the record set.
    """
    from reid_demo.embed import get_embedding_matrix  # lazy

    g_mat, g_ids = get_embedding_matrix(conn, dataset=dataset, normalize=True)
    f_mat, f_ids = _t11_fisher_matrix(conn, dataset=dataset, normalize=True)

    g_pos = {rid: i for i, rid in enumerate(g_ids)}
    f_pos = {rid: i for i, rid in enumerate(f_ids)}

    # Optional species filter (D7): keep only ids whose store `species` matches exactly.
    allowed: Optional[set] = None
    if species_filter is not None:
        recs = query_records(conn, dataset=dataset, species=species_filter, order_by="record_id")
        allowed = {r.record_id for r in recs}

    shared = sorted(set(g_pos) & set(f_pos))
    if allowed is not None:
        shared = [rid for rid in shared if rid in allowed]

    g_only = set(g_pos) - set(f_pos)
    f_only = set(f_pos) - set(g_pos)
    if g_only or f_only:
        warnings.warn(
            f"[fusion] aligning global+fisher: {len(g_only)} record(s) have only a global "
            f"signal and {len(f_only)} have only a fisher signal; both skipped "
            f"(intersection has {len(shared)} record(s)).",
            stacklevel=2,
        )

    if not shared:
        empty = np.empty((0, 0), dtype=np.float32)
        return [], empty, empty, {}

    g_aligned = np.stack([g_mat[g_pos[rid]] for rid in shared], axis=0).astype(np.float32)
    f_aligned = np.stack([f_mat[f_pos[rid]] for rid in shared], axis=0).astype(np.float32)

    # Orientations for flank gating (from the store).
    orientations: Dict[str, Optional[str]] = {}
    recs = query_records(conn, dataset=dataset, order_by="record_id")
    by_id = {r.record_id: r for r in recs}
    for rid in shared:
        r = by_id.get(rid)
        orientations[rid] = r.orientation if r is not None else None

    return shared, g_aligned, f_aligned, orientations


def run_fusion(
    db_path: Optional[str] = None,
    *,
    dataset: Optional[str] = None,
    signals: str = DEFAULT_SIGNALS,
    species_filter: Optional[str] = None,
    calibrators_dir: Optional[str] = None,
    borderline_low: float = BORDERLINE_LOW,
    borderline_high: float = BORDERLINE_HIGH,
    gv_budget: int = DEFAULT_GV_PAIR_BUDGET,
    gv_matcher: str = DEFAULT_GV_MATCHER,
    method: str = DEFAULT_GV_METHOD,
    flank_policy: str = "separate",
    out_dir: str = FUSION_DIR,
    dry_run: bool = False,
    seed: int = 42,
) -> FusionResult:
    """Store-integrated driver: build the fused affinity (+ optional GV rerank) and persist.

    Steps:
      1. Load + align global (T04) and Fisher (T11) matrices to ONE shared sorted
         ``record_id`` order; warn+skip rows missing either signal. Apply ``species_filter``
         (D7).
      2. ``build_fused_affinity`` over the aligned matrices (calibrated when
         ``calibrators_dir`` ships ``global``/``fisher`` calibrators; clipped-raw otherwise).
      3. For ``signals='full-funnel'``: ``select_borderline_pairs`` (bounded shortlist),
         resolve T11 keypoints/descriptors via ``_t11_local_features``, ``gv_rerank`` the
         shortlist, and ``refine_affinity_with_gv``. ``gv_ran`` is True only if GV actually
         executed (LightGlue/torch present). For ``signals='global+fisher'`` GV never runs.
      4. Unless ``dry_run``: write the ``.npz`` affinity sidecar (affinity + record_ids in
         SORTED order — the order T05 expects) and, when GV produced pair scores, the
         ``_pairs.json`` (sorted ascending by ``geom_score``). Writes NO ``detections``
         columns — ``cluster_id`` stays NULL.

    Returns a :class:`FusionResult`. Deterministic given a fixed ``dataset`` + ``seed``.
    """
    if signals not in SIGNAL_SETS:
        raise ValueError(
            f"unknown signals {signals!r}; must be one of {sorted(SIGNAL_SETS)}"
        )
    if flank_policy not in ("separate", "ignore"):
        raise ValueError(
            f"unknown flank_policy {flank_policy!r}; must be 'separate' or 'ignore'"
        )

    db_path = db_path or store.DEFAULT_DB_PATH

    params: Dict[str, Any] = {
        "signals": signals,
        "species_filter": species_filter,
        "calibrators_dir": calibrators_dir,
        "borderline_low": borderline_low,
        "borderline_high": borderline_high,
        "gv_budget": gv_budget,
        "gv_matcher": gv_matcher,
        "method": method,
        "flank_policy": flank_policy,
        "dry_run": dry_run,
        "seed": seed,
    }

    calibrators = _load_calibrators(calibrators_dir)

    conn = connect(db_path)
    try:
        record_ids, g_mat, f_mat, orientations = _aligned_signal_matrices(
            conn, dataset=dataset, species_filter=species_filter
        )
        n_crops = len(record_ids)

        if n_crops == 0:
            sentence = (
                f"Fused affinity: 0 crops with both global+fisher signals "
                f"(dataset={dataset!r}, signals={signals})."
            )
            return FusionResult(
                dataset=dataset, signals=signals, record_ids=[], affinity_path=None,
                pairs_path=None, n_crops=0, n_borderline_pairs=0, n_pairs_capped=0,
                gv_ran=False, params=params, sentence=sentence,
            )

        affinity = build_fused_affinity(
            record_ids, g_mat, f_mat, orientations,
            calibrators=calibrators, flank_policy=flank_policy,
        )

        n_borderline_pairs = 0
        n_pairs_capped = 0
        gv_ran = False
        pair_scores: List[PairScore] = []

        if signals == "full-funnel":
            pairs, n_pairs_capped = select_borderline_pairs(
                record_ids, affinity, orientations,
                low=borderline_low, high=borderline_high,
                budget=gv_budget, flank_policy=flank_policy, seed=seed,
            )
            n_borderline_pairs = len(pairs)
            if n_pairs_capped:
                print(
                    f"[fusion] gv shortlist capped: kept {n_borderline_pairs}, "
                    f"dropped {n_pairs_capped} (budget={gv_budget}).",
                    file=sys.stderr,
                )

            # Resolve T11 local features + a Fisher-distance lookup for the shortlist.
            keypoints, descriptors = _t11_local_features(conn, record_ids, dataset=dataset)
            affinity_lookup = {
                (a, b): float(affinity[record_ids.index(a), record_ids.index(b)])
                for (a, b, _bucket) in pairs
            }
            fisher_distance_lookup = _shortlist_fisher_distances(record_ids, f_mat, pairs)

            gv_cal = calibrators.get("gv")
            pair_scores = gv_rerank(
                pairs, keypoints, descriptors,
                affinity_lookup=affinity_lookup,
                fisher_distance_lookup=fisher_distance_lookup,
                gv_calibrator=gv_cal,
                use_lightglue=True, method=method, gv_matcher=gv_matcher,
                budget=gv_budget,
            )
            gv_ran = _lightglue_available() and bool(keypoints) and bool(descriptors)
            if gv_ran:
                affinity = refine_affinity_with_gv(
                    affinity, record_ids, pair_scores,
                    boost=GV_INLIER_BOOST, suppress=GV_BORDERLINE_SUPPRESS,
                    min_inliers=MIN_INLIERS,
                )

        affinity_path: Optional[str] = None
        pairs_path: Optional[str] = None
        if not dry_run:
            affinity_path = _affinity_npz_path(out_dir, dataset, signals)
            _save_affinity(affinity_path, affinity, record_ids)
            if signals == "full-funnel" and pair_scores:
                pairs_path = _pairs_json_path(out_dir, dataset, signals)
                _save_pairs_json(pairs_path, pair_scores)

        sentence = (
            f"Fused {signals} affinity over {n_crops} crop"
            f"{'' if n_crops == 1 else 's'}"
        )
        if signals == "full-funnel":
            sentence += (
                f"; {n_borderline_pairs} borderline pair"
                f"{'' if n_borderline_pairs == 1 else 's'} reranked"
                f"{' (GV ran)' if gv_ran else ' (GV no-op)'}"
            )
            if n_pairs_capped:
                sentence += f", {n_pairs_capped} capped"
        sentence += "."

        return FusionResult(
            dataset=dataset, signals=signals, record_ids=record_ids,
            affinity_path=affinity_path, pairs_path=pairs_path,
            n_crops=n_crops, n_borderline_pairs=n_borderline_pairs,
            n_pairs_capped=n_pairs_capped, gv_ran=gv_ran,
            params=params, sentence=sentence,
        )
    finally:
        conn.close()


def _shortlist_fisher_distances(
    record_ids: Sequence[str],
    fisher_matrix: Optional[np.ndarray],
    pairs: Sequence[Tuple[str, str, str]],
) -> Dict[Tuple[str, str], float]:
    """Compute ``fisher_distance`` (1 - cosine) for ONLY the shortlist pairs.

    Uses ``utils.distance_utils.fisher_distance`` on the aligned Fisher rows; returns ``{}``
    when no Fisher matrix is present (GV then uses the neutral 1.0 default). Bounded by the
    shortlist (never N^2).
    """
    if fisher_matrix is None or np.asarray(fisher_matrix).size == 0 or not pairs:
        return {}
    from utils.distance_utils import fisher_distance  # lazy

    pos = {rid: i for i, rid in enumerate(record_ids)}
    fm = np.asarray(fisher_matrix, dtype=np.float32)
    out: Dict[Tuple[str, str], float] = {}
    for (a, b, _bucket) in pairs:
        ia, ib = pos.get(a), pos.get(b)
        if ia is None or ib is None:
            continue
        va, vb = fm[ia], fm[ib]
        if float(np.linalg.norm(va)) < 1e-12 or float(np.linalg.norm(vb)) < 1e-12:
            out[(a, b)] = 1.0
        else:
            out[(a, b)] = float(fisher_distance(va, vb))
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reid_demo.fusion",
        description="T12 multi-signal fusion + GV reranking for the lynx re-ID demo.",
    )
    parser.add_argument("--db", default=None,
                        help=f"store DB path (default {store.DEFAULT_DB_PATH}).")
    parser.add_argument("--dataset", default=None,
                        help="dataset selector to scope which records to fuse.")
    parser.add_argument("--signals", default=DEFAULT_SIGNALS,
                        help=f"signal set {sorted(SIGNAL_SETS)} (default {DEFAULT_SIGNALS}).")
    parser.add_argument("--species", default=None,
                        help="filter rows by the store's `species` column (D7; NOT species_kept).")
    parser.add_argument("--calibrators-dir", default=None,
                        help="dir of pre-fit per-signal ScoreCalibrator .pkl files (consume only).")
    parser.add_argument("--borderline-low", type=float, default=BORDERLINE_LOW,
                        help=f"borderline band low edge (default {BORDERLINE_LOW}).")
    parser.add_argument("--borderline-high", type=float, default=BORDERLINE_HIGH,
                        help=f"borderline band high edge (default {BORDERLINE_HIGH}).")
    parser.add_argument("--gv-budget", type=int, default=DEFAULT_GV_PAIR_BUDGET,
                        help=f"max GV pairs (default {DEFAULT_GV_PAIR_BUDGET}).")
    parser.add_argument("--gv-matcher", default=DEFAULT_GV_MATCHER,
                        help=f"GV matcher (default {DEFAULT_GV_MATCHER}).")
    parser.add_argument("--method", default=DEFAULT_GV_METHOD,
                        help=f"local-feature method for GV (default {DEFAULT_GV_METHOD}).")
    parser.add_argument("--flank-policy", default="separate", choices=("separate", "ignore"),
                        help="'separate' (default; left/right gated) or 'ignore'.")
    parser.add_argument("--out-dir", default=FUSION_DIR,
                        help=f"sidecar output dir (default {FUSION_DIR}).")
    parser.add_argument("--seed", type=int, default=42, help="determinism seed (default 42).")
    parser.add_argument("--dry-run", action="store_true",
                        help="compute + report but write no sidecar files.")
    parser.add_argument("--json", action="store_true",
                        help="print the FusionResult as machine-readable JSON.")
    return parser


def _main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Validate the signal set up front so a bad value fails fast (no DB needed).
    if args.signals not in SIGNAL_SETS:
        print(
            f"[fusion] FATAL: unknown signals {args.signals!r}; "
            f"must be one of {sorted(SIGNAL_SETS)}",
            file=sys.stderr,
        )
        return 2

    db_path = args.db or store.DEFAULT_DB_PATH
    try:
        result = run_fusion(
            db_path=db_path,
            dataset=args.dataset,
            signals=args.signals,
            species_filter=args.species,
            calibrators_dir=args.calibrators_dir,
            borderline_low=args.borderline_low,
            borderline_high=args.borderline_high,
            gv_budget=args.gv_budget,
            gv_matcher=args.gv_matcher,
            method=args.method,
            flank_policy=args.flank_policy,
            out_dir=args.out_dir,
            dry_run=args.dry_run,
            seed=args.seed,
        )
    except Exception as exc:
        print(f"[fusion] FATAL: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(dataclasses.asdict(result), default=str))
    else:
        print(result.sentence)
        print(
            f"  signals={result.signals} crops={result.n_crops} "
            f"borderline={result.n_borderline_pairs} capped={result.n_pairs_capped} "
            f"gv_ran={result.gv_ran} affinity={result.affinity_path}"
        )

    # "Nothing to fuse" is worth surfacing (no overlapping global+fisher signals).
    if result.n_crops == 0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
