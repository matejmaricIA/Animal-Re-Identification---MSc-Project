"""reid_demo.review — Human-in-the-loop (HITL) review tool (T08).

The clustering engine (T05) is imperfect: some cluster assignments are borderline
(low ``cluster_conf``) and some crops are singletons that matched nothing (candidate
NEW individuals, ``is_candidate_new == 1`` / ``cluster_id == -1``). Forcing a
non-technical park biologist to inspect *everything* defeats the purpose. T08 surfaces
ONLY the most uncertain decisions, presents them as simple side-by-side
"Is this the same individual?" questions, captures the human's answer, persists it
back to the store via the T01 review API, and re-applies confirmed merges / splits /
new-individuals to the ``cluster_id`` field so that downstream consumers (catalogue
T06, eval T07, report T09, runner T10) reflect the corrected clustering.

Design constraints honoured here (see ``reid_demo/DATA_CONTRACT.md`` + STATUS_BOARD
D1-D8):

* ``cluster_id >= 0``  => a multi-crop discovered individual.
* ``cluster_id == -1 AND is_candidate_new == 1`` => singleton OR DBSCAN noise (D5).
  We route singletons on ``is_candidate_new``, NOT by recomputing anything.
* ``cluster_id IS NULL`` => clustering has not run for that row; we skip it.
* Flank policy (D4): spot-bearing flanks ``{left, right}`` are individually
  re-identifiable; ``{front, back, down, unknown, '', NULL}`` pool into ``other``.
  With ``respect_flanks=True`` we NEVER pair a known ``left`` against a known
  ``right``; ``other``/``unknown``/``None`` is compatible with anything.

ALL store access goes through ``reid_demo.store`` functions — never raw SQL, never
opening the SQLite file directly. The only fields T08 writes (via
``store.update_review``) are ``review_status``, ``review_note`` and ``cluster_id``.

The must-ship core is purely headless / scriptable:
``build_review_queue``, ``apply_decisions`` (+ the decisions-JSON round-trip),
``review_status_summary`` and ``build_pair_image``. ``serve_review_ui`` is an
OPTIONAL thin stdlib ``http.server`` front-end that funnels every decision through
``apply_decisions`` and adds no merge/split logic of its own.

No third-party web framework is imported anywhere (Flask is not installed). Heavy
optional deps (PIL, numpy) are imported lazily so the module imports cleanly without
them, and this module does NOT import the T11/T12 Fisher / fusion modules (its hard
deps stay {T01, T02, T05}); the optional T12 GV/affinity scores arrive purely as a
caller-supplied ``pair_scores`` mapping.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from reid_demo import store as _store
from reid_demo.store import (
    DEFAULT_DB_PATH,
    DetectionRecord,
)


# --------------------------------------------------------------------------- #
# Module-level constants (exact names — downstream / tests import these)
# --------------------------------------------------------------------------- #

#: Max review items surfaced by default.
DEFAULT_QUEUE_SIZE: int = 30

#: Assignments with ``cluster_conf`` strictly below this are review candidates.
LOW_CONF_THRESHOLD: float = 0.6

# Decision verbs (the only allowed values in a decision's ``answer`` field):
DECISION_SAME: str = "same"            # the two crops ARE the same individual
DECISION_DIFFERENT: str = "different"  # the two crops are DIFFERENT individuals
DECISION_NEW: str = "new"              # (singleton) confirm this crop is a NEW individual
DECISION_SKIP: str = "skip"            # reviewer unsure / defer; no store change

DECISIONS: set = {DECISION_SAME, DECISION_DIFFERENT, DECISION_NEW, DECISION_SKIP}

#: Directory (relative to the repo root) where review-session JSON artifacts land.
#: Mirrors ``constants.py`` style: derive the repo root from this file's location so
#: the path is stable no matter the CWD.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REVIEWS_DIR: str = os.path.join(_REPO_ROOT, "data", "reid_demo", "reviews")

#: Flank buckets that are individually re-identifiable and must never be auto-paired
#: across (D4). ``other``/``unknown``/``None`` is compatible with anything.
_KNOWN_FLANKS = {"left", "right"}


# --------------------------------------------------------------------------- #
# Data shapes (field names mirrored EXACTLY in the JSON — see ticket)
# --------------------------------------------------------------------------- #

@dataclass
class ReviewItem:
    """One review question shown to the human.

    ``item_id`` is a stable id of the form ``pair__<rid_a>__<rid_b>`` or
    ``singleton__<rid>`` so items and decisions can be matched on round-trip.
    """

    item_id: str
    kind: str                              # "pair" | "singleton"
    dataset: str
    record_id_a: str                       # primary crop under review (always set)
    record_id_b: Optional[str] = None      # comparison crop for "pair"; None for singleton
    cluster_id_a: Optional[int] = None
    cluster_id_b: Optional[int] = None
    cluster_conf: Optional[float] = None   # the confidence that triggered review (lower = more uncertain)
    orientation_a: Optional[str] = None
    orientation_b: Optional[str] = None
    species_a: Optional[str] = None
    species_b: Optional[str] = None
    crop_path_a: Optional[str] = None
    crop_path_b: Optional[str] = None
    reason: str = ""                       # human-readable why-this-is-here

    def to_json(self) -> dict:
        """Plain JSON-serialisable dict (round-trips through ``ReviewItem(**d)``)."""
        return asdict(self)


@dataclass
class ReviewDecision:
    """A human's answer to one :class:`ReviewItem`."""

    item_id: str
    answer: str                            # one of DECISIONS
    note: Optional[str] = None             # free text -> review_note
    target_cluster_id: Optional[int] = None  # singleton "belongs to existing cluster"

    def to_json(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Internal helpers — selection / ordering
# --------------------------------------------------------------------------- #

def _make_pair_item_id(rid_a: str, rid_b: str) -> str:
    return f"pair__{rid_a}__{rid_b}"


def _make_singleton_item_id(rid: str) -> str:
    return f"singleton__{rid}"


def _flanks_compatible(orient_a: Optional[str], orient_b: Optional[str]) -> bool:
    """True unless one side is a known ``left`` and the other a known ``right``.

    ``other``/``unknown``/``None``/``''`` is treated as compatible with anything (D4).
    """
    a = orient_a if orient_a in _KNOWN_FLANKS else None
    b = orient_b if orient_b in _KNOWN_FLANKS else None
    if a is None or b is None:
        return True
    return a == b


def _conf_sort_key(conf: Optional[float]) -> Tuple[int, float]:
    """Sort key so NULL ``cluster_conf`` sorts FIRST (most uncertain).

    Returns ``(0, -inf)`` for None so it precedes any real confidence, otherwise
    ``(1, conf)`` for plain ascending order.
    """
    if conf is None:
        return (0, float("-inf"))
    return (1, float(conf))


def _pair_score_lookup(
    pair_scores: Optional[Mapping[Tuple[str, str], float]],
    rid_a: str,
    rid_b: str,
) -> Optional[float]:
    """Order-insensitive lookup of a (a,b) score; tolerant of a None/partial map."""
    if not pair_scores:
        return None
    try:
        if (rid_a, rid_b) in pair_scores:
            return pair_scores[(rid_a, rid_b)]
        if (rid_b, rid_a) in pair_scores:
            return pair_scores[(rid_b, rid_a)]
    except TypeError:
        # A non-mapping or unhashable-key object was passed; degrade gracefully.
        return None
    return None


def _pick_anchor(
    cluster_members: Sequence[DetectionRecord],
    uncertain: DetectionRecord,
    *,
    respect_flanks: bool,
) -> Optional[DetectionRecord]:
    """Pick the cluster's exemplar (anchor) to compare ``uncertain`` against.

    Strategy: among the OTHER members of the same cluster that pass the flank filter,
    pick the highest-``cluster_conf`` one (ties broken by ``record_id`` for
    determinism). Returns None if no compatible other member exists (caller then
    downgrades the item to a singleton-style question).
    """
    candidates = [
        m for m in cluster_members
        if m.record_id != uncertain.record_id
        and (not respect_flanks or _flanks_compatible(uncertain.orientation, m.orientation))
    ]
    if not candidates:
        return None

    def key(m: DetectionRecord) -> Tuple[float, str]:
        conf = m.cluster_conf if m.cluster_conf is not None else float("-inf")
        # Highest conf first => negate; tie-break ascending record_id.
        return (-float(conf), m.record_id)

    candidates.sort(key=key)
    return candidates[0]


# --------------------------------------------------------------------------- #
# build_review_queue
# --------------------------------------------------------------------------- #

def build_review_queue(
    conn,
    *,
    dataset: str,
    queue_size: int = DEFAULT_QUEUE_SIZE,
    low_conf_threshold: float = LOW_CONF_THRESHOLD,
    include_singletons: bool = True,
    respect_flanks: bool = True,
    pair_scores: Optional[Mapping[Tuple[str, str], float]] = None,
) -> List[ReviewItem]:
    """Read clustered records for ``dataset`` and return a bounded, priority-ordered
    list of :class:`ReviewItem`.

    PAIR items: for each low-confidence assignment (``cluster_conf < low_conf_threshold``
    AND ``cluster_id >= 0``), pick the SAME-cluster highest-confidence member that
    passes the flank filter as ``record_id_b`` (the exemplar). When ``respect_flanks``
    is True, a known ``left`` is never paired with a known ``right``; if no compatible
    cluster-mate exists the item is downgraded to a SINGLETON-style question.

    SINGLETON items: every record with ``is_candidate_new == 1`` (the authoritative
    signal; equivalently ``cluster_id == -1``). Keyed on the flag — never recomputed.

    Rows whose ``review_status != 'unreviewed'`` are excluded so a re-run never
    re-asks. ``cluster_id IS NULL`` rows are skipped (clustering has not run for them).

    Ordering: ascending ``cluster_conf`` (most uncertain first; NULL conf with a real
    cluster sorts first), singletons appended after the pairs, then truncated to
    ``queue_size``.

    ``pair_scores`` (D8, OPTIONAL): a caller-supplied
    ``{(record_id_a, record_id_b) -> score}`` map of T12 geometric-verification /
    fused-affinity probabilities. When present it ORDERS the pair items so the most
    informative reviews surface first — highest GV-vs-affinity disagreement
    (``abs(gv_prob - cluster_conf)``) and lowest borderline confidence — WITHOUT
    changing which items are queued. It is a pure ordering hint with no hard
    dependency on T11/T12; a None / partial / malformed map never changes WHICH items
    are queued and never crashes the builder.
    """
    # ---- Pull every clustered-or-candidate row for this dataset (single read). ----
    records = _store.query_records(conn, dataset=dataset, order_by="record_id")

    by_id: Dict[str, DetectionRecord] = {r.record_id: r for r in records}

    # Group cluster members by cluster_id (for anchor selection). Only real clusters.
    members_by_cluster: Dict[int, List[DetectionRecord]] = {}
    for r in records:
        if r.cluster_id is not None and r.cluster_id >= 0:
            members_by_cluster.setdefault(r.cluster_id, []).append(r)

    pair_items: List[ReviewItem] = []
    singleton_items: List[ReviewItem] = []

    for r in records:
        # Skip already-reviewed rows (idempotent queue across re-runs).
        if r.review_status is not None and r.review_status != "unreviewed":
            continue

        cid = r.cluster_id

        # cluster_id IS NULL => clustering never ran for this row; skip entirely.
        if cid is None:
            continue

        # ---- SINGLETON: candidate-new (DBSCAN noise OR 1-crop singleton). ----
        # Key on is_candidate_new (authoritative); cluster_id == -1 is the equivalent.
        if r.is_candidate_new == 1 or cid == -1:
            if include_singletons:
                singleton_items.append(
                    ReviewItem(
                        item_id=_make_singleton_item_id(r.record_id),
                        kind="singleton",
                        dataset=dataset,
                        record_id_a=r.record_id,
                        record_id_b=None,
                        cluster_id_a=cid,
                        cluster_id_b=None,
                        cluster_conf=r.cluster_conf,
                        orientation_a=r.orientation,
                        species_a=r.species,
                        crop_path_a=r.crop_path,
                        reason="candidate new individual",
                    )
                )
            continue

        # ---- PAIR: low-confidence assignment in a real cluster (cluster_id >= 0). ----
        conf = r.cluster_conf
        is_low = (conf is None) or (float(conf) < float(low_conf_threshold))
        if not is_low:
            continue

        anchor = _pick_anchor(
            members_by_cluster.get(cid, []),
            r,
            respect_flanks=respect_flanks,
        )
        if anchor is None:
            # Cluster has only this (uncertain) crop, or no flank-compatible mate.
            # Downgrade to a singleton-style "is this its own individual?" question.
            singleton_items.append(
                ReviewItem(
                    item_id=_make_singleton_item_id(r.record_id),
                    kind="singleton",
                    dataset=dataset,
                    record_id_a=r.record_id,
                    record_id_b=None,
                    cluster_id_a=cid,
                    cluster_id_b=None,
                    cluster_conf=conf,
                    orientation_a=r.orientation,
                    species_a=r.species,
                    crop_path_a=r.crop_path,
                    reason=(
                        "low-confidence assignment "
                        f"({_fmt_conf(conf)}) — no flank-compatible cluster-mate"
                    ),
                )
            )
            continue

        pair_items.append(
            ReviewItem(
                item_id=_make_pair_item_id(r.record_id, anchor.record_id),
                kind="pair",
                dataset=dataset,
                record_id_a=r.record_id,
                record_id_b=anchor.record_id,
                cluster_id_a=cid,
                cluster_id_b=anchor.cluster_id,
                cluster_conf=conf,
                orientation_a=r.orientation,
                orientation_b=anchor.orientation,
                species_a=r.species,
                species_b=anchor.species,
                crop_path_a=r.crop_path,
                crop_path_b=anchor.crop_path,
                reason=f"low-confidence assignment ({_fmt_conf(conf)})",
            )
        )

    # ---- Base ordering: ascending cluster_conf (NULL-with-cluster first). ----
    # Stable tie-break on record_id_a keeps the queue deterministic for tests.
    pair_items.sort(key=lambda it: (_conf_sort_key(it.cluster_conf), it.record_id_a))
    singleton_items.sort(key=lambda it: (_conf_sort_key(it.cluster_conf), it.record_id_a))

    # ---- (D8, OPTIONAL) GV/affinity re-ordering of the PAIR items only. ----
    if pair_scores:
        pair_items = _reorder_by_pair_scores(pair_items, pair_scores)

    # Singletons are appended AFTER pairs (per the contract), then hard-cap.
    queue = pair_items + singleton_items
    if queue_size is not None and queue_size >= 0:
        queue = queue[:queue_size]
    return queue


def _reorder_by_pair_scores(
    pair_items: List[ReviewItem],
    pair_scores: Mapping[Tuple[str, str], float],
) -> List[ReviewItem]:
    """Re-rank pair items by GV-vs-affinity disagreement, then borderline confidence.

    Disagreement is ``abs(gv_prob - cluster_conf)`` against the same ``cluster_conf``
    the queue already carries: a pair with high affinity but a low GV probability (a
    likely false merge) or low affinity but a high GV probability (a likely missed
    merge) floats to the top. Pairs absent from the map keep their ``cluster_conf``
    rank and sit after all scored pairs. Never adds, drops, or rewrites items; never
    crashes on a partial map.
    """
    annotated: List[Tuple[Tuple[int, float, float, str], ReviewItem]] = []
    for it in pair_items:
        gv = _pair_score_lookup(pair_scores, it.record_id_a, it.record_id_b or "")
        if gv is None:
            # Unscored: rank AFTER scored ones, keep ascending-conf backbone order.
            conf = it.cluster_conf if it.cluster_conf is not None else float("-inf")
            key = (1, 0.0, float(conf), it.record_id_a)
        else:
            conf = it.cluster_conf if it.cluster_conf is not None else 0.0
            disagreement = abs(float(gv) - float(conf))
            # Highest disagreement first (negate), then lowest confidence first.
            key = (0, -disagreement, float(conf), it.record_id_a)
        annotated.append((key, it))
    annotated.sort(key=lambda pair: pair[0])
    return [it for _key, it in annotated]


def _fmt_conf(conf: Optional[float]) -> str:
    return "n/a" if conf is None else f"{float(conf):.2f}"


# --------------------------------------------------------------------------- #
# apply_decisions
# --------------------------------------------------------------------------- #

def _max_cluster_id(conn, dataset: str) -> int:
    """Highest non-negative ``cluster_id`` currently present for ``dataset``.

    Returns -1 if no non-negative cluster exists (so ``+1`` yields the first id 0).
    Uses ``store.count_by`` (no raw SQL) which returns ``{cluster_id: count}``.
    """
    counts = _store.count_by(conn, "cluster_id", dataset=dataset)
    ids = [cid for cid in counts.keys() if cid is not None and cid >= 0]
    return max(ids) if ids else -1


def apply_decisions(
    conn,
    items: Sequence[ReviewItem],
    decisions: Sequence[ReviewDecision],
    *,
    dataset: str,
    session_path: Optional[str] = None,
) -> dict:
    """Apply human decisions to the store via ``store.update_review`` ONLY, and write
    an auditable review-session JSON that round-trips through ``--apply``.

    Mapping rules (every per-record old->new mutation is recorded in the summary):

    PAIR item:
      * ``same``      -> if a and b are already in the same cluster: confirm in place
                         (status ``confirmed``). If in DIFFERENT clusters: MERGE — keep
                         the SMALLER existing cluster id, reassign EVERY member of the
                         larger id to it (status ``merged``).
      * ``different`` -> SPLIT ``record_id_a`` out into a fresh
                         ``cluster_id = max(existing for dataset) + 1`` (status
                         ``split``).
      * ``skip``      -> no store change (recorded in the session only).

    SINGLETON item:
      * ``new``                      -> assign ``record_id_a`` a fresh
                                        ``cluster_id = max+1`` (status ``confirmed``).
                                        ``is_candidate_new`` is intentionally NOT
                                        cleared (D5/D6c) — it is not a T08-writeable
                                        field; downstream treats any reviewed row as
                                        resolved.
      * ``same`` + ``target_cluster_id`` -> reassign ``record_id_a`` into the target
                                        cluster (status ``merged``).
      * ``different``                -> status ``rejected`` (no cluster change).
      * ``skip``                     -> no-op.

    An ``answer`` not in :data:`DECISIONS` raises ``ValueError``. Idempotent: applying
    the same decisions twice produces no net change (a merge whose members are already
    merged is a no-op; a confirm/new on an already-resolved row is a no-op).
    """
    items_by_id: Dict[str, ReviewItem] = {it.item_id: it for it in items}

    # Decisions matched to items by item_id; an item with no decision is a skip.
    decisions_by_id: Dict[str, ReviewDecision] = {}
    for d in decisions:
        if d.answer not in DECISIONS:
            raise ValueError(
                f"invalid answer {d.answer!r} for item {d.item_id!r}; "
                f"must be one of {sorted(DECISIONS)}"
            )
        decisions_by_id[d.item_id] = d

    mutations: List[dict] = []
    applied_decisions: List[dict] = []

    merges_applied = 0
    splits_applied = 0
    new_individuals_confirmed = 0
    skips = 0

    # Track the running max cluster id so multiple splits/news in one batch stay unique.
    next_free = _max_cluster_id(conn, dataset) + 1

    for item in items:
        decision = decisions_by_id.get(item.item_id)
        answer = decision.answer if decision is not None else DECISION_SKIP
        note = decision.note if decision is not None else None
        target = decision.target_cluster_id if decision is not None else None

        applied_decisions.append(
            {
                "item_id": item.item_id,
                "answer": answer,
                "note": note,
                "target_cluster_id": target,
            }
        )

        if answer == DECISION_SKIP:
            skips += 1
            continue

        if item.kind == "pair":
            muts, kind = _apply_pair(
                conn, item, answer, note, dataset, lambda: next_free
            )
            mutations.extend(muts)
            if kind == "merge" and muts:
                merges_applied += 1
            elif kind == "split" and muts:
                splits_applied += 1
                next_free += 1  # consumed a fresh id
        else:  # singleton
            muts, kind = _apply_singleton(
                conn, item, answer, note, target, dataset, lambda: next_free
            )
            mutations.extend(muts)
            if kind == "new" and muts:
                new_individuals_confirmed += 1
                next_free += 1
            elif kind == "merge" and muts:
                merges_applied += 1

    summary = {
        "dataset": dataset,
        "created_at": _now_str(),
        "queue_size": len(items),
        "low_conf_threshold": LOW_CONF_THRESHOLD,
        "items": [it.to_json() for it in items],
        "decisions": applied_decisions,
        "mutations": mutations,
        "counts": {
            "items_reviewed": len(items),
            "decisions_applied": len(applied_decisions),
            "merges_applied": merges_applied,
            "splits_applied": splits_applied,
            "new_individuals_confirmed": new_individuals_confirmed,
            "skips": skips,
        },
    }

    out_path = _write_session(summary, dataset, session_path)
    summary["session_path"] = out_path
    return summary


def _apply_pair(
    conn,
    item: ReviewItem,
    answer: str,
    note: Optional[str],
    dataset: str,
    next_free_fn,
) -> Tuple[List[dict], str]:
    """Apply one PAIR decision. Returns (mutations, kind) where kind in
    {'merge', 'split', 'confirm', 'noop'}."""
    rec_a = _store.get_record(conn, item.record_id_a)
    if rec_a is None:
        return ([], "noop")

    if answer == DECISION_SAME:
        rec_b = _store.get_record(conn, item.record_id_b) if item.record_id_b else None
        cid_a = rec_a.cluster_id
        cid_b = rec_b.cluster_id if rec_b is not None else None

        # Same cluster already (or b unknown): confirm a in place (idempotent).
        if rec_b is None or cid_a == cid_b:
            # True no-op guard: if a is already resolved into this cluster (a previous
            # apply confirmed/merged it here), re-applying must not flip its status.
            # This keeps a cross-cluster MERGE idempotent when its session JSON (which
            # still carries the pre-merge cluster ids) is replayed: both crops now read
            # the same live cluster id, and we must leave the 'merged' status intact.
            if rec_a.review_status in ("confirmed", "merged"):
                return ([], "confirm")
            muts = _set_review(
                conn, rec_a, status="confirmed", note=note, new_cluster_id=None
            )
            return (muts, "confirm")

        # Cross-cluster MERGE: smaller id wins; reassign EVERY member of the larger id.
        if cid_a is None or cid_b is None:
            # One side never clustered — confirm a into b's cluster if b is real.
            if cid_b is not None and cid_b >= 0:
                muts = _set_review(
                    conn, rec_a, status="merged", note=note, new_cluster_id=cid_b
                )
                return (muts, "merge")
            muts = _set_review(conn, rec_a, status="confirmed", note=note)
            return (muts, "confirm")

        keep = min(cid_a, cid_b)
        drop = max(cid_a, cid_b)
        muts = _merge_clusters(conn, dataset, keep=keep, drop=drop, note=note)
        return (muts, "merge")

    if answer == DECISION_DIFFERENT:
        # SPLIT record_id_a out into a fresh cluster id.
        fresh = next_free_fn()
        if rec_a.cluster_id == fresh:
            # Already split there (idempotent re-apply guard).
            return ([], "noop")
        muts = _set_review(
            conn, rec_a, status="split", note=note, new_cluster_id=fresh
        )
        return (muts, "split")

    # answer == DECISION_NEW on a pair item is not meaningful; treat as confirm-in-place.
    if answer == DECISION_NEW:
        muts = _set_review(conn, rec_a, status="confirmed", note=note)
        return (muts, "confirm")

    return ([], "noop")


def _apply_singleton(
    conn,
    item: ReviewItem,
    answer: str,
    note: Optional[str],
    target_cluster_id: Optional[int],
    dataset: str,
    next_free_fn,
) -> Tuple[List[dict], str]:
    """Apply one SINGLETON decision. Returns (mutations, kind) where kind in
    {'new', 'merge', 'reject', 'noop'}."""
    rec_a = _store.get_record(conn, item.record_id_a)
    if rec_a is None:
        return ([], "noop")

    if answer == DECISION_NEW:
        # Idempotency: an already-resolved singleton with a real cluster id and a
        # non-unreviewed status is a no-op (re-applying must not allocate a new id).
        if (
            rec_a.cluster_id is not None
            and rec_a.cluster_id >= 0
            and rec_a.review_status == "confirmed"
        ):
            return ([], "noop")
        fresh = next_free_fn()
        muts = _set_review(
            conn, rec_a, status="confirmed", note=note, new_cluster_id=fresh
        )
        return (muts, "new")

    if answer == DECISION_SAME:
        # Reassign into an existing cluster the reviewer targeted.
        if target_cluster_id is not None and target_cluster_id >= 0:
            if rec_a.cluster_id == target_cluster_id and rec_a.review_status == "merged":
                return ([], "noop")
            muts = _set_review(
                conn, rec_a, status="merged", note=note, new_cluster_id=target_cluster_id
            )
            return (muts, "merge")
        # No target given — nothing to merge into; treat as a plain confirm of review.
        muts = _set_review(conn, rec_a, status="confirmed", note=note)
        return (muts, "noop")

    if answer == DECISION_DIFFERENT:
        # Reviewer says this candidate-new is NOT a (new) individual we should keep:
        # mark rejected, leave cluster_id as-is.
        if rec_a.review_status == "rejected":
            return ([], "noop")
        muts = _set_review(conn, rec_a, status="rejected", note=note)
        return (muts, "reject")

    return ([], "noop")


def _set_review(
    conn,
    record: DetectionRecord,
    *,
    status: str,
    note: Optional[str],
    new_cluster_id: Optional[int] = None,
) -> List[dict]:
    """Write one review update via ``store.update_review`` and return a mutation row.

    Records the old->new ``cluster_id`` and ``review_status``. ``new_cluster_id=None``
    leaves the cluster unchanged.
    """
    old_cluster = record.cluster_id
    old_status = record.review_status
    new_cluster = old_cluster if new_cluster_id is None else new_cluster_id

    _store.update_review(
        conn,
        record.record_id,
        status,
        review_note=note,
        cluster_id=new_cluster_id,  # None => leave cluster unchanged (store contract)
    )
    return [
        {
            "record_id": record.record_id,
            "old_cluster_id": old_cluster,
            "new_cluster_id": new_cluster,
            "old_status": old_status,
            "new_status": status,
        }
    ]


def _merge_clusters(
    conn,
    dataset: str,
    *,
    keep: int,
    drop: int,
    note: Optional[str],
) -> List[dict]:
    """Merge cluster ``drop`` into cluster ``keep`` (smaller-id-wins convention).

    Reassigns EVERY member crop of ``drop`` to ``keep`` via ``store.update_review``,
    not just the one under review. Returns one mutation row per reassigned record.
    Idempotent: if ``drop`` has no members (already merged) this is a no-op.
    """
    if keep == drop:
        return []
    members = _store.query_records(conn, dataset=dataset, cluster_id=drop)
    muts: List[dict] = []
    for m in members:
        muts.extend(
            _set_review(conn, m, status="merged", note=note, new_cluster_id=keep)
        )
    return muts


# --------------------------------------------------------------------------- #
# review_status_summary
# --------------------------------------------------------------------------- #

def review_status_summary(conn, *, dataset: str) -> dict:
    """Plain-language status counts for a non-technical user.

    Keys:
      * ``individuals_before``       — distinct real cluster ids the human would have
                                       seen pre-review = current distinct real ids +
                                       merges already applied (each merge collapsed one
                                       id) - splits/new ids that did not exist before.
        (Computed pragmatically from the current store + review_status tallies; see
        note below — ``individuals_after`` is the ground truth, ``_before`` is a
        best-effort reconstruction for the report headline.)
      * ``individuals_after``        — distinct ``cluster_id >= 0`` present now.
      * ``items_reviewed``           — rows with ``review_status != 'unreviewed'``.
      * ``merges_applied``           — rows with ``review_status == 'merged'``.
      * ``splits_applied``           — rows with ``review_status == 'split'``.
      * ``new_individuals_confirmed``— ``is_candidate_new == 1`` rows now
                                       ``review_status == 'confirmed'`` with a real id.
      * ``still_unreviewed``         — rows still ``review_status == 'unreviewed'``.
    """
    records = _store.query_records(conn, dataset=dataset)

    real_ids_now = {
        r.cluster_id for r in records
        if r.cluster_id is not None and r.cluster_id >= 0
    }
    individuals_after = len(real_ids_now)

    merged = sum(1 for r in records if r.review_status == "merged")
    split = sum(1 for r in records if r.review_status == "split")
    reviewed = sum(1 for r in records if r.review_status not in (None, "unreviewed"))
    still_unreviewed = sum(
        1 for r in records if r.review_status in (None, "unreviewed")
    )
    new_confirmed = sum(
        1 for r in records
        if r.is_candidate_new == 1
        and r.review_status == "confirmed"
        and r.cluster_id is not None
        and r.cluster_id >= 0
    )

    # merges_applied counts MERGE OPERATIONS, not reassigned crops: each merge collapses
    # exactly one cluster id. We approximate the operation count by the number of
    # distinct (merged) target clusters that absorbed foreign members. A robust,
    # store-only proxy is the drop in distinct ids attributable to merged rows: every
    # 'merged' row that no longer sits in its own singleton id. For the headline we use
    # the simpler, well-defined "number of clusters that gained a 'merged' member".
    merged_target_ids = {
        r.cluster_id for r in records
        if r.review_status == "merged" and r.cluster_id is not None and r.cluster_id >= 0
    }
    merges_applied = len(merged_target_ids)
    splits_applied = split  # each split row created one fresh id

    # individuals_before: reconstruct what the count was before review. Each merge
    # operation removed one id (two -> one); each split/new added one id. So:
    #   before = after + merges_applied - splits_applied - new_individuals_confirmed
    # Clamp at >= individuals_after's lower bound of 0.
    individuals_before = (
        individuals_after + merges_applied - splits_applied - new_confirmed
    )
    if individuals_before < 0:
        individuals_before = individuals_after

    return {
        "individuals_before": individuals_before,
        "individuals_after": individuals_after,
        "items_reviewed": reviewed,
        "merges_applied": merges_applied,
        "splits_applied": splits_applied,
        "new_individuals_confirmed": new_confirmed,
        "still_unreviewed": still_unreviewed,
    }


# --------------------------------------------------------------------------- #
# build_pair_image
# --------------------------------------------------------------------------- #

def _load_crop_pil(record_or_paths, max_side: int):
    """Best-effort load of a crop as a PIL.Image (RGB), with the full fallback chain:

        on-disk crop_path -> bbox-crop from source_image -> gray placeholder.

    ``record_or_paths`` is a dict-ish with keys ``crop_path``, ``source_image`` and
    optional ``bbox_*`` plus a ``label`` for the placeholder caption. NEVER raises.
    """
    from PIL import Image  # lazy

    crop_path = record_or_paths.get("crop_path")
    source_image = record_or_paths.get("source_image")
    label = record_or_paths.get("label", "?")

    # 1) on-disk crop file
    if crop_path and os.path.isfile(crop_path):
        try:
            img = Image.open(crop_path).convert("RGB")
            return _fit(img, max_side)
        except Exception:
            pass

    # 2) bbox-crop from the source image
    if source_image and os.path.isfile(source_image):
        try:
            src = Image.open(source_image).convert("RGB")
            W, H = src.size
            bx = record_or_paths.get("bbox_x")
            by = record_or_paths.get("bbox_y")
            bw = record_or_paths.get("bbox_w")
            bh = record_or_paths.get("bbox_h")
            if None not in (bx, by, bw, bh):
                left = max(0, min(W, int(round(bx * W))))
                top = max(0, min(H, int(round(by * H))))
                right = max(left + 1, min(W, int(round((bx + bw) * W))))
                bottom = max(top + 1, min(H, int(round((by + bh) * H))))
                return _fit(src.crop((left, top, right, bottom)), max_side)
            return _fit(src, max_side)
        except Exception:
            pass

    # 3) gray placeholder
    return _placeholder(label, max_side)


def _fit(img, max_side: int):
    """Downscale (preserving aspect) so the longest side <= max_side."""
    w, h = img.size
    longest = max(w, h)
    if longest > max_side and longest > 0:
        scale = max_side / float(longest)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return img


def _placeholder(label: str, max_side: int):
    """A gray square with the record id captioned — the terminal fallback."""
    from PIL import Image, ImageDraw  # lazy

    side = max(64, min(max_side, 256))
    img = Image.new("RGB", (side, side), color=(110, 110, 110))
    draw = ImageDraw.Draw(img)
    text = f"(no image)\n{label}"
    try:
        draw.text((6, side // 2 - 10), text, fill=(235, 235, 235))
    except Exception:
        pass
    return img


def _caption_for(item: ReviewItem, side: str) -> str:
    """Build a human-readable caption for side 'a' or 'b'."""
    if side == "a":
        return (
            f"{item.record_id_a}\n"
            f"cluster {item.cluster_id_a} · {item.orientation_a or 'unknown'}\n"
            f"{item.species_a or 'species?'}"
        )
    return (
        f"{item.record_id_b}\n"
        f"cluster {item.cluster_id_b} · {item.orientation_b or 'unknown'}\n"
        f"{item.species_b or 'species?'}"
    )


def build_pair_image(item: ReviewItem, *, max_side: int = 512):
    """Return a single ``PIL.Image`` showing crop A and crop B side by side (pair) or
    crop A alone (singleton), each captioned with cluster id / orientation / species.

    Robust fallbacks: a missing ``crop_path`` falls back to a bbox-crop from
    ``source_image``; a missing source falls back to a gray placeholder. NEVER raises.
    We resolve ``source_image`` / ``bbox_*`` from the store on demand (the ReviewItem
    only carries ``crop_path``) so the bbox fallback works for real records.
    """
    from PIL import Image, ImageDraw  # lazy

    pad = 8
    caption_h = 52

    def _meta(rid: Optional[str], crop_path: Optional[str], label: str):
        """Resolve source_image + bbox for the fallback chain from the store if we can."""
        info = {"crop_path": crop_path, "label": label,
                "source_image": None,
                "bbox_x": None, "bbox_y": None, "bbox_w": None, "bbox_h": None}
        # crop on disk? we don't need the store at all.
        if crop_path and os.path.isfile(crop_path):
            return info
        # Otherwise try to enrich with source_image+bbox via the store (best-effort).
        conn = _meta.conn
        if conn is not None and rid:
            try:
                rec = _store.get_record(conn, rid)
                if rec is not None:
                    info["source_image"] = rec.source_image
                    info["bbox_x"] = rec.bbox_x
                    info["bbox_y"] = rec.bbox_y
                    info["bbox_w"] = rec.bbox_w
                    info["bbox_h"] = rec.bbox_h
            except Exception:
                pass
        return info

    _meta.conn = _maybe_open_store_for_image(item)

    panels = []
    captions = []

    info_a = _meta(item.record_id_a, item.crop_path_a, item.record_id_a)
    panels.append(_load_crop_pil(info_a, max_side))
    captions.append(_caption_for(item, "a"))

    if item.kind == "pair" and item.record_id_b:
        info_b = _meta(item.record_id_b, item.crop_path_b, item.record_id_b)
        panels.append(_load_crop_pil(info_b, max_side))
        captions.append(_caption_for(item, "b"))

    # Normalize panel heights to the tallest panel.
    panel_h = max(p.size[1] for p in panels)
    norm_panels = []
    for p in panels:
        if p.size[1] != panel_h:
            scale = panel_h / float(p.size[1])
            p = p.resize((max(1, int(p.size[0] * scale)), panel_h))
        norm_panels.append(p)

    total_w = sum(p.size[0] for p in norm_panels) + pad * (len(norm_panels) + 1)
    total_h = panel_h + caption_h + pad * 2

    canvas = Image.new("RGB", (total_w, total_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    x = pad
    for p, cap in zip(norm_panels, captions):
        canvas.paste(p, (x, pad))
        try:
            draw.text((x + 2, pad + panel_h + 4), cap, fill=(20, 20, 20))
        except Exception:
            pass
        x += p.size[0] + pad

    return canvas


def _maybe_open_store_for_image(item: ReviewItem):
    """Open the default store read-only-ish for bbox fallback, or None on any failure.

    We only need this when a crop file is missing; opening the default DB is a best
    effort — never let it raise out of ``build_pair_image``.
    """
    # If both crop files exist there is no need to touch the store.
    have_a = bool(item.crop_path_a) and os.path.isfile(item.crop_path_a)
    have_b = (
        item.kind != "pair"
        or not item.record_id_b
        or (bool(item.crop_path_b) and os.path.isfile(item.crop_path_b))
    )
    if have_a and have_b:
        return None
    db = os.path.join(_REPO_ROOT, DEFAULT_DB_PATH)
    if not os.path.isfile(db):
        return None
    try:
        return _store.connect(db, create=False)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Session JSON helpers + decisions-JSON round-trip
# --------------------------------------------------------------------------- #

def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _write_session(summary: dict, dataset: str, session_path: Optional[str]) -> str:
    """Write the review-session JSON (audit trail + headless-replay input)."""
    if session_path is None:
        Path(REVIEWS_DIR).mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_path = os.path.join(REVIEWS_DIR, f"{dataset}_review_{stamp}.json")
    else:
        parent = os.path.dirname(os.path.abspath(session_path))
        if parent:
            Path(parent).mkdir(parents=True, exist_ok=True)
    with open(session_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return session_path


def load_decisions_json(path: str) -> Tuple[List[ReviewItem], List[ReviewDecision], str]:
    """Parse a decisions/session JSON file into (items, decisions, dataset).

    Reconstructs :class:`ReviewItem`/:class:`ReviewDecision` from the documented
    schema, ignoring any extra keys (e.g. the session's ``mutations``/``counts``).
    """
    with open(path, "r", encoding="utf-8") as fh:
        obj = json.load(fh)
    dataset = obj.get("dataset", "")
    item_fields = set(ReviewItem.__dataclass_fields__)
    dec_fields = set(ReviewDecision.__dataclass_fields__)
    items = [
        ReviewItem(**{k: v for k, v in i.items() if k in item_fields})
        for i in obj.get("items", [])
    ]
    decisions = [
        ReviewDecision(**{k: v for k, v in d.items() if k in dec_fields})
        for d in obj.get("decisions", [])
    ]
    return items, decisions, dataset


def apply_decisions_file(
    conn,
    path: str,
    *,
    session_path: Optional[str] = None,
    dataset: Optional[str] = None,
) -> dict:
    """Headless ``--apply`` entry point: load a decisions JSON and apply it.

    This is exactly what T10's end-to-end runner drives in batch with no human present.
    """
    items, decisions, file_dataset = load_decisions_json(path)
    ds = dataset or file_dataset
    if not ds:
        raise ValueError(f"decisions file {path!r} has no 'dataset' and none was given")
    return apply_decisions(conn, items, decisions, dataset=ds, session_path=session_path)


# --------------------------------------------------------------------------- #
# serve_review_ui — OPTIONAL thin stdlib http.server front-end (no Flask)
# --------------------------------------------------------------------------- #

def _item_image_data_uri(item: ReviewItem, *, max_side: int = 384) -> str:
    """Render ``build_pair_image`` and return a base64 PNG ``data:`` URI (best effort)."""
    import base64
    import io

    try:
        img = build_pair_image(item, max_side=max_side)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        # 1x1 transparent gif fallback so the page never breaks.
        return (
            "data:image/gif;base64,"
            "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
        )


_REVIEW_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Re-ID review</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;margin:24px;background:#fafafa;color:#222}
 #card{max-width:820px;margin:0 auto;background:#fff;border:1px solid #ddd;
       border-radius:10px;padding:20px;box-shadow:0 1px 4px rgba(0,0,0,.06)}
 img{max-width:100%;border:1px solid #ccc;border-radius:6px}
 .meta{color:#555;font-size:14px;margin:8px 0}
 button{font-size:16px;padding:10px 16px;margin:6px 6px 0 0;border-radius:8px;
        border:1px solid #bbb;cursor:pointer;background:#f3f3f3}
 button:hover{background:#e8e8e8}
 #note{width:100%;box-sizing:border-box;margin-top:10px;padding:8px;font-size:14px}
 .flank{color:#a40000;font-size:13px}
</style></head><body>
<div id="card">
 <h2>Are these the same individual?</h2>
 <div id="reason" class="meta"></div>
 <div><img id="img" alt="review crops"></div>
 <div class="flank">Left and right flanks are never auto-merged — opposite flanks are
   shown only for context, never as a "same?" question.</div>
 <textarea id="note" placeholder="Optional note (e.g. 'same rosette on shoulder')"></textarea>
 <div>
  <button onclick="decide('same')">Same animal</button>
  <button onclick="decide('different')">Different animal</button>
  <button onclick="decide('new')">New individual</button>
  <button onclick="decide('skip')">Not sure</button>
 </div>
 <div id="progress" class="meta"></div>
</div>
<script>
let QUEUE = __QUEUE_JSON__;
let i = 0;
function render(){
  if(i >= QUEUE.length){
    document.getElementById('card').innerHTML =
      '<h2>All done — thank you!</h2><p>You can close this tab.</p>';
    fetch('/finish', {method:'POST'});
    return;
  }
  const it = QUEUE[i];
  document.getElementById('reason').textContent =
    '(' + (it.kind) + ') ' + (it.reason || '');
  document.getElementById('img').src = it._img;
  document.getElementById('note').value = '';
  document.getElementById('progress').textContent =
    'Item ' + (i+1) + ' of ' + QUEUE.length;
}
function decide(answer){
  const it = QUEUE[i];
  const note = document.getElementById('note').value || null;
  fetch('/decide', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({item_id: it.item_id, answer: answer, note: note})})
   .then(r=>r.json()).then(_=>{ i++; render(); });
}
render();
</script>
</body></html>
"""


def serve_review_ui(
    conn_or_db_path,
    *,
    dataset: str,
    host: str = "127.0.0.1",
    port: int = 8765,
    queue_size: int = DEFAULT_QUEUE_SIZE,
    auto_open: bool = False,
) -> None:
    """OPTIONAL interactive front-end (stdlib ``http.server`` only — no Flask).

    Builds the queue, then serves a single vanilla-HTML page that walks it one item at
    a time with the two crops side by side (or one for singletons), the metadata, and
    plain-language buttons (Same animal / Different animal / New individual / Not sure)
    plus a note box. Each submit POSTs a single :class:`ReviewDecision` to ``/decide``,
    which funnels through :func:`apply_decisions` for that one item — the UI adds NO
    merge/split logic of its own. ``/finish`` (or queue exhaustion) shuts the server
    down. Blocks until then.

    Accepts either an open ``sqlite3.Connection`` or a ``db_path`` string; when a path
    is given a fresh connection is opened per request so writes commit cleanly.
    """
    import http.server
    import socketserver
    import threading

    if isinstance(conn_or_db_path, str):
        db_path: Optional[str] = conn_or_db_path
        build_conn = _store.connect(db_path)
    else:
        db_path = None
        build_conn = conn_or_db_path

    items = build_review_queue(
        build_conn, dataset=dataset, queue_size=queue_size
    )
    items_by_id = {it.item_id: it for it in items}

    # Pre-render images for the page (base64 data URIs) so no static serving is needed.
    queue_payload = []
    for it in items:
        d = it.to_json()
        d["_img"] = _item_image_data_uri(it)
        queue_payload.append(d)
    page = _REVIEW_PAGE.replace("__QUEUE_JSON__", json.dumps(queue_payload))

    shutdown_event = threading.Event()

    def _conn_for_request():
        if db_path is not None:
            return _store.connect(db_path)
        return build_conn

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence default logging
            pass

        def _send(self, code, body, content_type="application/json"):
            payload = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            else:
                self._send(404, json.dumps({"error": "not found"}))

        def do_POST(self):
            if self.path == "/finish":
                self._send(200, json.dumps({"ok": True, "done": True}))
                shutdown_event.set()
                return
            if self.path == "/decide":
                length = int(self.headers.get("Content-Length", 0) or 0)
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    payload = json.loads(raw or b"{}")
                except Exception:
                    self._send(400, json.dumps({"error": "bad json"}))
                    return
                item = items_by_id.get(payload.get("item_id"))
                if item is None:
                    self._send(404, json.dumps({"error": "unknown item_id"}))
                    return
                decision = ReviewDecision(
                    item_id=payload["item_id"],
                    answer=payload.get("answer", DECISION_SKIP),
                    note=payload.get("note"),
                    target_cluster_id=payload.get("target_cluster_id"),
                )
                conn = _conn_for_request()
                try:
                    apply_decisions(conn, [item], [decision], dataset=dataset)
                    self._send(200, json.dumps({"ok": True}))
                except Exception as exc:  # never crash the server on one bad apply
                    self._send(500, json.dumps({"error": str(exc)}))
                return
            self._send(404, json.dumps({"error": "not found"}))

    class _Server(socketserver.TCPServer):
        allow_reuse_address = True

    with _Server((host, port), Handler) as httpd:
        url = f"http://{host}:{port}"
        print(f"[review] serving review UI at {url}  (Ctrl-C or 'Finish' to stop)")
        if auto_open:
            try:
                import webbrowser
                webbrowser.open(url)
            except Exception:
                pass

        def _watch_shutdown():
            shutdown_event.wait()
            httpd.shutdown()

        watcher = threading.Thread(target=_watch_shutdown, daemon=True)
        watcher.start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[review] stopped.")
        finally:
            shutdown_event.set()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _queue_to_json(items: Sequence[ReviewItem], *, dataset: str,
                   queue_size: int, low_conf_threshold: float) -> dict:
    return {
        "dataset": dataset,
        "created_at": _now_str(),
        "queue_size": queue_size,
        "low_conf_threshold": low_conf_threshold,
        "items": [it.to_json() for it in items],
        "decisions": [],  # empty — the human / UI fills this in
    }


def _resolve_db(db: Optional[str]) -> str:
    return db if db else DEFAULT_DB_PATH


def _cmd_build_queue(args) -> int:
    conn = _store.connect(_resolve_db(args.db), create=False)
    items = build_review_queue(
        conn,
        dataset=args.dataset,
        queue_size=args.queue_size,
        low_conf_threshold=args.low_conf_threshold,
    )
    payload = _queue_to_json(
        items,
        dataset=args.dataset,
        queue_size=args.queue_size,
        low_conf_threshold=args.low_conf_threshold,
    )
    if args.out:
        parent = os.path.dirname(os.path.abspath(args.out))
        if parent:
            Path(parent).mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[review] wrote {len(items)} queue items -> {args.out}")
    else:
        print(json.dumps(payload, indent=2))
    return 0


def _cmd_apply(args) -> int:
    conn = _store.connect(_resolve_db(args.db), create=False)
    summary = apply_decisions_file(conn, args.apply, dataset=args.dataset)
    counts = summary.get("counts", {})
    print(f"[review] applied decisions from {args.apply}")
    print(f"         dataset:                 {summary.get('dataset')}")
    print(f"         items_reviewed:          {counts.get('items_reviewed')}")
    print(f"         merges_applied:          {counts.get('merges_applied')}")
    print(f"         splits_applied:          {counts.get('splits_applied')}")
    print(f"         new_individuals_confirmed: {counts.get('new_individuals_confirmed')}")
    print(f"         session_path:            {summary.get('session_path')}")
    return 0


def _cmd_status(args) -> int:
    conn = _store.connect(_resolve_db(args.db), create=False)
    summary = review_status_summary(conn, dataset=args.dataset)
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_serve(args) -> int:
    serve_review_ui(
        _resolve_db(args.db),
        dataset=args.dataset,
        host=args.host,
        port=args.port,
        queue_size=args.queue_size,
        auto_open=args.auto_open,
    )
    return 0


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="reid_demo.review",
        description="Human-in-the-loop review tool for open-set re-ID (T08).",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build-queue", action="store_true",
                      help="build & print/save the review queue (no store writes)")
    mode.add_argument("--apply", metavar="DECISIONS_JSON",
                      help="headless apply: read a decisions JSON and apply it")
    mode.add_argument("--status", action="store_true",
                      help="print the plain-language review_status_summary")
    mode.add_argument("--serve", action="store_true",
                      help="launch the optional local web UI (stdlib http.server)")

    parser.add_argument("--dataset", default=None, help="logical run name to scope to")
    parser.add_argument("--db", default=None,
                        help=f"DB path (default {DEFAULT_DB_PATH})")
    parser.add_argument("--queue-size", type=int, default=DEFAULT_QUEUE_SIZE,
                        help="max review items (default %(default)s)")
    parser.add_argument("--low-conf-threshold", type=float, default=LOW_CONF_THRESHOLD,
                        help="assignments below this cluster_conf are review candidates")
    parser.add_argument("--out", default=None,
                        help="(--build-queue) write the queue JSON to this path")
    parser.add_argument("--host", default="127.0.0.1", help="(--serve) bind host")
    parser.add_argument("--port", type=int, default=8765, help="(--serve) bind port")
    parser.add_argument("--auto-open", action="store_true",
                        help="(--serve) open the browser automatically")

    args = parser.parse_args(argv)

    if args.build_queue:
        if not args.dataset:
            parser.error("--build-queue requires --dataset")
        return _cmd_build_queue(args)
    if args.apply:
        return _cmd_apply(args)
    if args.status:
        if not args.dataset:
            parser.error("--status requires --dataset")
        return _cmd_status(args)
    if args.serve:
        if not args.dataset:
            parser.error("--serve requires --dataset")
        return _cmd_serve(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
