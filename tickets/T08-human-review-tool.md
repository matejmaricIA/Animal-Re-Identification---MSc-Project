# T08 — Human-in-the-loop review tool

> **Status:** 🔴 Not started · **Milestone:** M1 (Demo-ready)
> **Depends on:** T01, T02, T05 · **Blocks:** —
> **Owner:** _unassigned_
>
> Self-contained ticket for the lynx re-identification demo/pilot. See [STATUS_BOARD.md](../STATUS_BOARD.md) for the full plan, the binding design decisions, and the agent working protocol.

---

## Signal layering (D8 — optional GV-prioritized queue)

> **Amendment per binding decision D8 (see STATUS_BOARD.md).** The review-queue builder MAY consume per-pair scores from **T12** (geometric-verification / fused-affinity) to ORDER the queue — surfacing the highest GV-disagreement / lowest-confidence borderline pairs first. This is an **optional enhancement, not a hard dependency**: accept an optional `gv_scores` / `pair_scores` mapping (keyed by the same record-id pairs) and, when absent, fall back to global+Fisher affinity uncertainty or the global-only ordering already specified. T08 still runs AFTER T05 (D5), and re-running T05 must not wipe T08's `review_status` writes.

## Context

We are building a DEMO + PILOT MVP of an **open-set, individual-animal re-identification** system for Eurasian lynx (closest public analogs: spotted big cats — LeopardID2022 leopards, ATRW Amur tigers). The pipeline takes an unlabeled pile of animal crops, discovers how many DISTINCT individuals are present (unknown count), and flags singletons that match nothing as candidate NEW individuals.

The work is split into independent tickets (T01–T10) each handed to a separate AI coding agent. They all communicate **only** through a shared SQLite store and its single Python access module, `reid_demo/store.py` (defined by ticket **T01** — the data contract). You do **not** have the rest of that conversation; everything you need is below. Read `reid_demo/DATA_CONTRACT.md` in the repo first — it is the authoritative description of the store; this ticket cites it but the doc wins on any conflict.

**This is T08: the Human-in-the-Loop (HITL) review tool.** The clustering engine (T05) is imperfect: some merges are borderline (two crops it *thinks* are the same individual but isn't sure), and some crops are singletons that matched nothing (candidate NEW individuals). Forcing a non-technical park biologist to inspect *everything* defeats the purpose. **T08's job is to surface ONLY the most uncertain decisions** — the lowest-confidence cluster assignments / proposed merges, plus unmatched singletons — present them as simple side-by-side "Is this the same individual?" questions, capture the human's answer, **persist it back to the store** via the T01 review API, and **re-apply** confirmed merges/splits/rejections to the `cluster_id` field so that downstream consumers (catalogue T06, eval T07, report T09) reflect the corrected clustering.

The whole demo must be readable by a **non-technical park biologist**. So the review UI talks in animals and photos ("Are these two photos the same lynx? Yes / No / Not sure"), not in cosine distances. A core constraint of lynx/leopard re-ID: **left and right flanks are different spot patterns** — the store carries an `orientation` field (`left`/`right`/`front`/`back`/`down`/`unknown`); T08 must show it to the reviewer and never auto-merge across opposite flanks.

### Where T08 sits in the pipeline

```
... T05 clustering (runs FIRST) --writes--> cluster_id, cluster_conf, is_candidate_new  (store)
        (singletons AND DBSCAN noise => cluster_id = -1 AND is_candidate_new = 1)
                                              |
                                              v
        T08 (this ticket, runs AFTER T05): pick the N lowest-confidence assignments
        + every is_candidate_new == 1 singleton, ask the human "same individual?",
        write review_status (+ maybe new cluster_id)
                                              |
                                              v
   T06 catalogue / T07 eval / T09 report / T10 runner: READ the corrected clustering

NOTE: re-running T05 must NOT wipe T08's review_status writes — T05 preserves rows
where review_status != 'unreviewed' (or refuses without an explicit --force flag).
```

## Objective

Deliver a self-contained `reid_demo/review.py` module whose **must-ship core** is purely headless/scriptable, plus an OPTIONAL interactive front-end. Concretely:

**Must-ship core (D7a — required for this ticket to count as done):**

1. `build_review_queue` — **Selects a review queue**: from the store, pick the *lowest-confidence* cluster assignments and *all* candidate-new singletons for a given `dataset`, producing concrete "decision items" (mostly pairwise "same individual?" questions, plus per-singleton "is this a new individual or does it belong to an existing one?" questions). Cap the queue size so the biologist reviews a small, bounded number of items. **(D8, optional)** When a caller supplies precomputed T12 geometric-verification / fused-affinity pair scores, use them to **order** the queue so the most informative items (highest GV-vs-affinity disagreement, lowest-confidence borderline pairs) come first; this is a pure ordering hint and **degrades gracefully** to `cluster_conf`-based ordering when no scores are provided.
2. `apply_decisions` — **Persists decisions** back into the store using **only** the T01 review API (`update_review`), and **re-applies** them to `cluster_id` (merge two clusters into one id; split/reject a misassigned crop into its own cluster; confirm a singleton as a brand-new individual id).
3. **decisions-JSON round-trip** — the JSON written by `apply_decisions` (session artifact) can be re-fed to the headless `--apply` path to reproduce the same store state, deterministically and idempotently. This is the entry point T10's end-to-end runner drives in batch without a human present, and is what makes the whole thing testable.
4. `review_status_summary` — plain-language counts of what review did (merges, splits, new individuals, still-unreviewed), consumable by the CLI `--status` and by T09's report.
5. `build_pair_image` — **Renders side-by-side crops** for each item using the real crop image files referenced in the store, annotated with the human-readable metadata (species, timestamp, camera, orientation/flank, current cluster id), with robust fallbacks. (Pure image function; no UI required to exercise it.)

**Optional / nice-to-have (NOT required to close this ticket; build only if time permits):**

- An **interactive web front-end**: a static HTML page served by stdlib `http.server` (`serve_review_ui`) that walks the queue and captures `same` / `different` / `new` / `skip` answers.
- An **interactive notebook front-end**: an equivalent `ipywidgets` path inside Jupyter.

Both interactive front-ends are thin shells over the core — they MUST funnel every decision through `apply_decisions` and add no merge/split logic of their own. Neither introduces new pip dependencies (Flask is NOT installed; stdlib `http.server` + `ipywidgets` if present, with a non-interactive JSON-file fallback, are the only allowed UIs).

This ticket does **NOT** do clustering (T05), does NOT generate the final catalogue (T06), does NOT compute metrics (T07). It only reads cluster outputs from the store and writes human-corrected review fields + cluster reassignments back to the store.

### Ordering and re-run safety (D5)

T08 runs **AFTER** T05 clustering: T05 writes `cluster_id`, `cluster_conf`, and `is_candidate_new`, and only then does T08 surface uncertain items and write `review_status` (+ corrected `cluster_id`). T08 keys its singleton selection on **`is_candidate_new == 1`** (equivalently `cluster_id == -1`) — the single authoritative "this might be a brand-new individual" signal that both DBSCAN noise and 1-crop singletons carry. T08 must NOT recompute or second-guess that flag; it only reads it.

Because T05 may be re-run after a review pass, this is a documented hazard: **re-running T05 MUST NOT silently wipe T08's `review_status` writes.** T08's contract relies on T05 preserving any row whose `review_status != 'unreviewed'` (or refusing to overwrite without an explicit `--force`). T08 itself defends against double-asking by excluding already-reviewed rows from the queue (see `build_review_queue`).

## Scope

**Must-ship core (D7a):**
- A new module `reid_demo/review.py` with the public API in *Interface contract* below.
- A **review-queue builder** (`build_review_queue`): rank cluster assignments by ascending `cluster_conf`, take the lowest-confidence ones; collect all `is_candidate_new == 1` singletons (equivalently `cluster_id == -1`); emit a bounded list of `ReviewItem`s. Excludes already-reviewed rows so a re-run never re-asks. **(D8, optional)** Accept an OPTIONAL precomputed per-pair score map from T12 (GV inlier-based / fused-affinity probabilities) and, when present, prioritize the queue by GV-disagreement / lowest-confidence borderline pairs first; absence of these scores changes nothing (fall back to `cluster_conf` ordering). This is an ordering enhancement only — **no hard dependency on T11/T12**.
- A **decision applier** (`apply_decisions`) that maps human answers → store writes (`update_review`) and cluster reassignments (merge / split / new-individual) via `update_review(..., cluster_id=...)`.
- A **headless apply mode**: read a decisions JSON file and apply it, for batch/testing/T10 — the primary, human-free entry point.
- A **review session artifact** (JSON) recording the queue + decisions for auditability/reproducibility, structured so it **round-trips** through `--apply`.
- `review_status_summary`: plain-language counts of the review outcome for the CLI `--status` and T09's report.
- `build_pair_image`: side-by-side (pair) / single (singleton) crop renderer with on-disk-crop → bbox-crop → placeholder fallbacks.
- A CLI: `python -m reid_demo.review --build-queue ...`, `--apply ...`, `--status ...` (plus the optional `--serve ...`).
- Unit tests under `tests/test_review.py` exercising queue building + decision application + the round-trip against an in-memory/temp store (no human, no browser).

**Optional / nice-to-have (D7a — build only if time permits; absence does NOT fail the ticket):**
- A **minimal local web UI** (`serve_review_ui`; stdlib `http.server`, single page, vanilla HTML/JS, images inlined as base64 or served from disk) that walks the queue and POSTs decisions through `apply_decisions`.
- A **Jupyter notebook UI** path using `ipywidgets` (degrade gracefully to printing image paths + an instruction to fill a JSON file if `ipywidgets` is unavailable).

### Out
- The clustering algorithm, confidence calibration, or flank-aware grouping logic itself (that is **T05** — T08 only *reads* `cluster_id`/`cluster_conf`/`is_candidate_new`/`orientation` and *corrects* them).
- The data store / schema (that is **T01** — `reid_demo/store.py`; T08 imports it, never redefines fields or writes SQL directly).
- The polished visual catalogue / contact sheets (that is **T06**).
- Any ML metric computation or ground-truth scoring (that is **T07**).
- Any change to existing pipeline files (`main.py`, `global_embedding.py`, `nested_importance_sampling.py`, etc.). T08 only ADDS files under `reid_demo/` and `tests/`.
- Adding new pip dependencies. Use only what is already installed (verified present: `pandas`, `sklearn`, `Pillow`, `numpy`, `jupyter`; verified ABSENT: `flask`) plus the standard library.
- Authentication, multi-user concurrency, remote hosting — the UI is single-user on `localhost`.

## Inputs

- **The store**: a SQLite DB at `data/reid_demo/reid_demo.sqlite` (T01 default `DEFAULT_DB_PATH`), already populated by T02→T05 (T05 clustering having run **before** T08) so that rows for the target `dataset` have `cluster_id`, `cluster_conf`, and `is_candidate_new` set. Singletons and DBSCAN noise both arrive with `cluster_id == -1` and `is_candidate_new == 1` (T08 keys on the flag). T08 opens it via `reid_demo.store.connect(db_path)`. T08's `review_status` writes are expected to survive a subsequent T05 re-run (T05 preserves `review_status != 'unreviewed'` rows or refuses without `--force`); T08 does not need to defend against that beyond excluding already-reviewed rows from its queue.
- **Crop image files** on disk at each record's `crop_path` (e.g. `data/MedvednicaDS/animal_crops/02020401_crop1_conf92.jpg`, or `data/reid_demo/crops/IMG_0066__crop1.jpg`). T08 reads these to render side-by-side images. If a crop file is missing, fall back to cropping from `source_image` + normalized `bbox_*` using Pillow (the bbox is `[x,y,w,h]` normalized in `[0,1]`, top-left origin) — but never crash the queue on a single missing image.
- **(Headless mode only)** a decisions JSON file (schema in *Interface contract*) describing human answers to apply without a UI.
- **(Optional)** the same store may hold multiple runs distinguished by the `dataset` column (e.g. `MedvednicaDS`, `LeopardID2022`); all T08 operations are scoped to one `--dataset`.

T08 must not hard-depend on any file outside the store except the crop images the store points at.

## Outputs

- New file `reid_demo/review.py` (+ no change to `reid_demo/__init__.py` is required, but you MAY add `from .review import build_review_queue, apply_decisions, serve_review_ui` re-exports if you also leave the existing `__init__` exports intact — additive only).
- **Store side effects** (the real product): for each reviewed record, `review_status` set to one of `confirmed` / `rejected` / `merged` / `split` (never leave it inconsistent with the applied cluster change), `review_note` optionally set, and `cluster_id` reassigned where the human merged/split/created-new. All via T01's `update_review`.
- A **review session JSON** written to `data/reid_demo/reviews/<dataset>_review_<timestamp>.json` capturing: the queue items shown, the raw human decisions, and the resulting store mutations (old→new `cluster_id`, old→new `review_status`). This is the audit trail and the headless-replay input.
- A **served HTML page** (transient, in-memory; not necessarily written to disk) when `--serve` is used.
- `tests/test_review.py`.

## Interface contract

Downstream tickets (T06/T07/T09/T10) do **not** import T08 functions — they read the store. So T08's external contract is mainly **(a) the store mutations it produces** and **(b) the decisions-JSON / session-JSON formats** that T10 (end-to-end runner) replays. Provide exactly the following. The must-ship core (D7a) is `build_review_queue`, `apply_decisions`, the decisions-JSON round-trip, `review_status_summary`, and `build_pair_image`; `serve_review_ui` (and the notebook path) are OPTIONAL.

### Module-level constants

```python
DEFAULT_QUEUE_SIZE: int = 30          # max review items surfaced by default
LOW_CONF_THRESHOLD: float = 0.6       # assignments with cluster_conf below this are review candidates
# Decision verbs (the only allowed values in a decision's "answer" field):
DECISION_SAME: str = "same"           # the two crops ARE the same individual
DECISION_DIFFERENT: str = "different" # the two crops are DIFFERENT individuals
DECISION_NEW: str = "new"             # (singleton item) confirm this crop is a NEW individual
DECISION_SKIP: str = "skip"           # reviewer unsure / defer; no store change
DECISIONS: set[str] = {"same", "different", "new", "skip"}
```

### Data shapes (use `@dataclass`; mirror these field names in the JSON exactly)

```python
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ReviewItem:
    item_id: str                 # stable id for this question, e.g. "pair__<rid_a>__<rid_b>" or "singleton__<rid>"
    kind: str                    # "pair" | "singleton"
    dataset: str
    record_id_a: str             # primary crop under review (always set)
    record_id_b: Optional[str] = None   # the comparison crop for "pair"; None for "singleton"
    cluster_id_a: Optional[int] = None
    cluster_id_b: Optional[int] = None
    cluster_conf: Optional[float] = None   # the confidence that triggered review (lower = more uncertain)
    orientation_a: Optional[str] = None
    orientation_b: Optional[str] = None
    species_a: Optional[str] = None
    species_b: Optional[str] = None
    crop_path_a: Optional[str] = None
    crop_path_b: Optional[str] = None
    reason: str = ""             # human-readable why-this-is-here, e.g. "low-confidence assignment (0.42)" / "candidate new individual"

@dataclass
class ReviewDecision:
    item_id: str
    answer: str                  # one of DECISIONS
    note: Optional[str] = None   # free text -> review_note
    # For singleton "belongs to existing cluster" answers, the reviewer may target a cluster:
    target_cluster_id: Optional[int] = None
```

### Public functions (exact signatures)

```python
import sqlite3
from typing import Optional, List, Sequence, Mapping, Tuple

def build_review_queue(
    conn: sqlite3.Connection,
    *,
    dataset: str,
    queue_size: int = DEFAULT_QUEUE_SIZE,
    low_conf_threshold: float = LOW_CONF_THRESHOLD,
    include_singletons: bool = True,
    respect_flanks: bool = True,
    pair_scores: Optional[Mapping[Tuple[str, str], float]] = None,  # (D8, OPTIONAL) T12 GV/affinity
) -> List[ReviewItem]:
    """Read clustered records for `dataset` from the store and produce a bounded,
    priority-ordered list of ReviewItems:
      - PAIR items: for each low-confidence assignment (cluster_conf < low_conf_threshold,
        cluster_id >= 0), pick a representative crop from the SAME cluster (highest-conf
        member) as record_id_b so the human can confirm 'does this uncertain crop really
        belong with this cluster?'. When respect_flanks=True, only pair crops whose
        orientation matches (or where one side is 'unknown'/None); NEVER pair left vs right.
      - SINGLETON items: every record with is_candidate_new == 1 (the authoritative signal;
        equivalently cluster_id == -1 — both DBSCAN noise and 1-crop singletons carry this),
        asking 'is this a NEW individual?'. Key on the flag, do not recompute it.
    Skip records already reviewed (review_status != 'unreviewed') so re-runs don't re-ask.
    Order by ascending cluster_conf (most uncertain first); singletons appended after.
    Truncate to queue_size. Populate crop_path_*, orientation_*, species_* from the records.

    (D8, OPTIONAL) pair_scores: a caller-supplied mapping {(record_id_a, record_id_b) ->
    score} of T12 geometric-verification / fused-affinity probabilities for borderline /
    candidate-merge pairs (order-insensitive key lookup; try both orderings). When provided,
    it ORDERS the queue so the most informative items surface first — highest
    GV-vs-affinity disagreement (e.g. high global+Fisher affinity but few GV inliers, or
    vice-versa) and lowest-confidence borderline pairs ahead of the rest. This is purely an
    ordering hint layered on top of the cluster_conf ranking: it never adds, drops, or
    rewrites items, and is NOT a hard dependency. When pair_scores is None (the default, and
    whenever GV was not run), the queue degrades gracefully to global+Fisher affinity
    uncertainty if those scores are passed instead, or to plain ascending-cluster_conf
    ordering (current backbone behavior) when nothing is supplied."""

def apply_decisions(
    conn: sqlite3.Connection,
    items: Sequence[ReviewItem],
    decisions: Sequence[ReviewDecision],
    *,
    dataset: str,
    session_path: Optional[str] = None,
) -> dict:
    """Apply human decisions to the store via reid_demo.store.update_review ONLY.
    Mapping rules (record the old->new for every mutation in the returned summary):
      PAIR item:
        answer 'same'      -> record_id_a stays in cluster_id_b's cluster; set review_status
                              'confirmed' (if already same cluster) or 'merged' (if it had to
                              be reassigned to b's cluster). If a and b were in different
                              clusters, MERGE: reassign all crops of cluster_id_a to cluster_id_b
                              (smaller-id-wins convention documented below), each via update_review.
        answer 'different' -> SPLIT record_id_a out of its current cluster into a fresh
                              cluster_id (max(existing)+1); review_status 'split' (or 'rejected'
                              if the crop is removed from a cluster as a misassignment).
        answer 'skip'      -> no store change; record the skip in the session only.
      SINGLETON item:
        answer 'new'                 -> assign record_id_a a fresh cluster_id (max+1);
                                        review_status 'confirmed'. cluster_id + review_status
                                        are the cluster-membership truth. Note (D5): downstream
                                        T06/T07 key the "candidate-new" count on is_candidate_new,
                                        which T08 reads but does NOT write through update_review
                                        (T08's only writeable fields are review_status,
                                        review_note, cluster_id). So a human-confirmed new
                                        individual keeps is_candidate_new == 1 while gaining a
                                        non-negative cluster_id + review_status 'confirmed';
                                        downstream MUST treat a reviewed (review_status !=
                                        'unreviewed') candidate as resolved rather than
                                        double-counting it (see Open Questions / D6c).
        answer 'same' + target_cluster_id
                                     -> reassign record_id_a into target_cluster_id; status 'merged'.
        answer 'different'/'skip'    -> 'rejected'/no-op respectively.
    Use update_review(conn, record_id, review_status, review_note=note, cluster_id=new_id).
    Write a review-session JSON to session_path (or the default reviews dir) capturing
    items, decisions, and the list of (record_id, old_cluster_id, new_cluster_id,
    old_status, new_status) mutations. Return that summary dict."""

def serve_review_ui(   # OPTIONAL / nice-to-have (D7a) — not required to close this ticket
    conn_or_db_path,
    *,
    dataset: str,
    host: str = "127.0.0.1",
    port: int = 8765,
    queue_size: int = DEFAULT_QUEUE_SIZE,
    auto_open: bool = False,
) -> None:
    """OPTIONAL interactive front-end. Build the queue, start a stdlib http.server on host:port serving a single
    vanilla-HTML review page (no Flask). The page shows one ReviewItem at a time with
    the two crops side by side (or one crop for singletons), the metadata, and buttons
    Yes(same)/No(different)/New/Skip + an optional note box. On submit, the browser POSTs
    a ReviewDecision as JSON to an endpoint that calls apply_decisions for that single item.
    Blocks until the queue is exhausted or the user stops the server (Ctrl-C / a 'Finish'
    button that shuts the server down). Re-open conn fresh per request if a db_path was passed."""

def review_status_summary(conn: sqlite3.Connection, *, dataset: str) -> dict:
    """Plain-language status for a non-technical user, e.g.
    {'individuals_before': 24, 'individuals_after': 23, 'items_reviewed': 18,
     'merges_applied': 3, 'splits_applied': 1, 'new_individuals_confirmed': 5,
     'still_unreviewed': 6}. Counts derived from the store via store.count_by /
     query_records. Used by the CLI --status and consumable by T09's report."""

def build_pair_image(item: ReviewItem, *, max_side: int = 512):
    """Return a single PIL.Image (or numpy array) showing crop A and crop B side by side
    with captions (cluster id, orientation, species, timestamp). For singletons, returns
    crop A alone. Falls back to cropping from source_image+bbox if crop_path is missing.
    Used by both the web UI (encoded to base64 PNG) and the notebook UI."""
```

### Decisions-JSON file format (headless `--apply` input AND session artifact)

A single JSON object:

```json
{
  "dataset": "LeopardID2022",
  "created_at": "2026-06-09 14:00:00",
  "queue_size": 30,
  "low_conf_threshold": 0.6,
  "items": [
    {"item_id": "pair__leoA__crop1__leoB__crop1", "kind": "pair",
     "record_id_a": "leoA__crop1", "record_id_b": "leoB__crop1",
     "cluster_id_a": 7, "cluster_id_b": 4, "cluster_conf": 0.42,
     "orientation_a": "left", "orientation_b": "left",
     "species_a": "leopard", "species_b": "leopard",
     "crop_path_a": "...", "crop_path_b": "...",
     "reason": "low-confidence assignment (0.42)"}
  ],
  "decisions": [
    {"item_id": "pair__leoA__crop1__leoB__crop1", "answer": "same",
     "note": "same rosette pattern on shoulder", "target_cluster_id": null}
  ]
}
```

- `items` and `decisions` are matched by `item_id`. An item with no matching decision is treated as `skip`.
- `--apply <file>` reads this object, reconstructs `ReviewItem`s from `items`, `ReviewDecision`s from `decisions`, and calls `apply_decisions`. This is exactly what `apply_decisions` writes as its session artifact, so a served session can be replayed deterministically.

### Cluster-reassignment conventions (document in code + in the session JSON)

- **Merge** two clusters → keep the **smaller** existing `cluster_id`, reassign all members of the larger id to the smaller id.
- **Split / new individual** → new `cluster_id = (max existing cluster_id for this dataset) + 1` (so ids stay non-negative and unique within the dataset; never reuse `-1`).
- Never auto-merge across opposite flanks (`left`↔`right`); the queue builder must not even propose such pairs when `respect_flanks=True`.
- All reassignments go through `store.update_review(conn, record_id, review_status, review_note=..., cluster_id=new_id)` — T08 issues **no raw SQL**.

### CLI

```
python -m reid_demo.review --build-queue --dataset NAME [--db PATH] [--queue-size N]
        [--low-conf-threshold F] [--out queue.json]      # print/save the queue, no writes

python -m reid_demo.review --serve --dataset NAME [--db PATH] [--host H] [--port P]
        [--queue-size N]                                  # launch the local web UI

python -m reid_demo.review --apply decisions.json [--db PATH]   # headless apply (no human)

python -m reid_demo.review --status --dataset NAME [--db PATH]  # print review_status_summary
```

`--db` defaults to `reid_demo.store.DEFAULT_DB_PATH`.

## Existing code to reuse (real paths)

- **`reid_demo/store.py`** (T01, the ONLY way T08 touches data): import `connect`, `query_records`, `get_record`, `update_review`, `count_by`, `DEFAULT_DB_PATH`, `REVIEW_STATUSES`, `ORIENTATIONS`, `DetectionRecord`. Read `reid_demo/DATA_CONTRACT.md` for the 28-column schema; the columns T08 reads are `record_id, crop_path, source_image, bbox_x/y/w/h, species, timestamp, camera_id, orientation, cluster_id, cluster_conf, is_candidate_new, review_status, dataset`. The columns T08 writes (only through `update_review`) are `review_status, review_note, cluster_id`. **Do not** add columns or write other fields.
- **`/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/visualization_suite/io.py`** — `load_image(path)`, `bgr_to_rgb(image)`, `fig_to_image(fig)`, `save_image(path, image)`. Reuse for loading crops if convenient (note it returns BGR numpy via OpenCV; convert with `bgr_to_rgb` before PIL/base64). Pillow is also available directly (`from PIL import Image`).
- **`/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/visualization_suite/collage.py`** — `make_grid(images, titles, cols, figsize)` builds a labeled image grid with matplotlib; you may reuse it for `build_pair_image` (1×2 grid) instead of hand-rolling layout. Keep dependencies to what it already uses (matplotlib, numpy).
- **`/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/calibration.py`** — `ScoreCalibrator.predict_proba` shows that `cluster_conf` written by T05 is a calibrated P(same) in `[0,1]`; T08 treats `cluster_conf` as "higher = more confident, lower = more uncertain". Read-only context; do not call it.
- **`/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/nested_importance_sampling.py`** — its `oracle: Callable[[str,str]->int]` (1=same, 0=different) is the *same mental model* as T08's pairwise human decision. T08 does NOT call NIS, but if T10 later wants a human oracle, T08's pair-decision flow is the natural backing. Context only.
- **Standard library**: `sqlite3`, `http.server`, `json`, `base64`, `pathlib`, `datetime`, `webbrowser`, `dataclasses`, `argparse`. **Flask is not installed — do not import it.** For the notebook path, import `ipywidgets`/`IPython.display` lazily and degrade gracefully if absent.
- Crops live at e.g. `data/MedvednicaDS/animal_crops/02020401_crop1_conf92.jpg`; reference frames at `data/MedvednicaDS/animal_images/`. The store's `crop_path` is the canonical pointer — use it, don't reconstruct names.

## Implementation notes

- **Read everything through `query_records`/`get_record`; never open the SQLite file directly.** This keeps T08 immune to schema details and consistent with T01's validation (e.g. `update_review` validates `review_status ∈ REVIEW_STATUSES`).
- **Queue priority**: sort low-confidence assignments by ascending `cluster_conf`; treat NULL `cluster_conf` as most-uncertain (sort first) only if `cluster_id` is set. Append singletons after pairs. Hard-cap at `queue_size`. Skip any record whose `review_status != 'unreviewed'` so the human is never re-asked on a re-run (idempotent queue).
- **Optional GV/affinity ordering (D8)**: `build_review_queue` MAY accept a `pair_scores` mapping `{(record_id_a, record_id_b) -> score}` produced by T12 (the full-funnel run wiring in T10 computes it; the `--smoke`/global default does not). When present, use it to re-order — not re-populate — the pair items so the most informative reviews come first: rank by **GV-vs-affinity disagreement** (pairs where geometric verification and the fused global+Fisher affinity most disagree) and **lowest borderline confidence**. Compute disagreement against the same `cluster_conf`/affinity the queue already has (e.g. `abs(gv_prob - cluster_conf)`), so a pair with high affinity but few GV inliers (a likely false merge) or low affinity but many inliers (a likely missed merge) floats to the top of the queue. This is **optional and non-blocking**: T08 has **no import of and no hard dependency on T11/T12** (its hard deps stay {T01, T02, T05}); when `pair_scores` is `None` it falls back to global+Fisher affinity uncertainty if those were supplied, otherwise to plain ascending-`cluster_conf` ordering. Key lookup is order-insensitive (try `(a,b)` then `(b,a)`); pairs absent from the map keep their `cluster_conf` rank. Never let a missing/partial score map change WHICH items are queued or crash the builder — ordering only.
- **Choosing `record_id_b` for a pair**: from the same `cluster_id` as the uncertain crop, pick the **highest-`cluster_conf`** member that passes the flank filter; that is the cluster's "anchor"/exemplar the human compares against. If the cluster has only the uncertain crop itself, downgrade the item to a singleton ("is this its own new individual?").
- **Flank safety**: when `respect_flanks=True`, never form a pair whose `orientation_a` and `orientation_b` are opposite known flanks (`left` vs `right`). Treat `None`/`unknown` as compatible with anything (the field-data lynx case where flank isn't estimated yet). Document this in the served page so the biologist understands why some comparisons aren't offered.
- **Apply is the source of truth, UI is a thin shell**: the headless `--apply` path and `apply_decisions` are the must-ship core; the OPTIONAL web UI and notebook UI are thin shells that must funnel every decision through `apply_decisions` (single-item or batched). Do not duplicate the merge/split logic in the UI layer. This guarantees the headless `--apply` path and the interactive paths produce identical store states (and makes `tests/test_review.py` able to cover the real logic without a browser). Because the core is fully exercised headlessly, the ticket is complete even if neither interactive front-end is built.
- **Merge/split atomicity**: when merging cluster X into cluster Y, reassign *every* member crop of X (query `query_records(dataset=..., cluster_id=X)`), not just the one under review — otherwise the catalogue would show a half-merged individual. Record each per-record mutation in the session summary.
- **Web UI (OPTIONAL / nice-to-have, D7a)**: a single `BaseHTTPRequestHandler`. `GET /` serves the HTML+JS shell and the queue (as embedded JSON) with each crop image inlined as a base64 PNG data URI (so no separate static-file serving / path-escaping headaches). `POST /decide` receives one `ReviewDecision` JSON, calls `apply_decisions(conn, [item], [decision], dataset=...)`, returns the next item or a "done" signal. A `POST /finish` (or queue exhaustion) shuts the server down cleanly. Keep the page dependency-free (vanilla JS, no CDN). Buttons must be labeled in plain language: **"Same animal" / "Different animal" / "New individual" / "Not sure"**. Skip entirely if time is short — the headless core stands alone.
- **Notebook UI (OPTIONAL / nice-to-have, D7a)**: provide a function (can be `serve_review_ui` variant or a documented snippet at top of `review.py`) that, given a queue, renders each `build_pair_image` with `ipywidgets` Yes/No/New/Skip buttons and calls `apply_decisions`. If `ipywidgets` import fails, print the queue + crop paths and instruct the user to fill a decisions JSON and run `--apply` (never crash). Also skippable if time is short.
- **Image fallback**: if `crop_path` is missing on disk, crop from `source_image` using the normalized bbox (`left=bbox_x*W, top=bbox_y*H, right=(bbox_x+bbox_w)*W, bottom=(bbox_y+bbox_h)*H`, clamp to image bounds) with Pillow. If even `source_image` is missing, render a gray placeholder with the `record_id` as caption — never raise out of `build_pair_image`.
- **Determinism for tests**: queue building must be deterministic given the same store state (stable tie-breaking by `record_id`). `apply_decisions` must be a pure function of (store state, items, decisions) so tests can seed a store, apply a decisions list, and assert the resulting `cluster_id`/`review_status`.
- **No-op safety**: applying the same decisions JSON twice must not corrupt state — re-merging already-merged clusters is a no-op; re-confirming is a no-op. Make `apply_decisions` idempotent on already-applied decisions.
- **Constants/paths**: follow `constants.py` style (module-level path constants, `os.path.dirname(os.path.abspath(__file__))`); put the reviews output dir under `data/reid_demo/reviews/` and create it on demand. Do NOT edit `constants.py`.
- **Status board**: append a single line to `STATUS_BOARD.md` (create it if absent) marking T08 deliverables. Do not edit other tickets' status lines or any other repo file.

## Acceptance criteria

- [ ] `reid_demo/review.py` and `tests/test_review.py` exist; no existing repo file is modified except an additive line in `STATUS_BOARD.md` (and, optionally, additive re-export lines in `reid_demo/__init__.py` that do not remove existing exports).
- [ ] `python -c "from reid_demo.review import build_review_queue, apply_decisions, serve_review_ui, review_status_summary, build_pair_image, ReviewItem, ReviewDecision, DECISIONS, DEFAULT_QUEUE_SIZE, LOW_CONF_THRESHOLD, DECISION_SAME, DECISION_DIFFERENT, DECISION_NEW, DECISION_SKIP"` succeeds (every contracted name importable — `serve_review_ui` must at minimum exist as a callable even if the optional UI is left as a thin/stub implementation).
- [ ] **Must-ship core present**: `build_review_queue`, `apply_decisions`, the decisions-JSON round-trip, `review_status_summary`, and `build_pair_image` are all implemented and pass their tests. (The interactive web UI and notebook UI are OPTIONAL per D7a; their absence does NOT fail the ticket, and the UI-specific verify step below is manual/optional.)
- [ ] **Queue keys singletons on `is_candidate_new == 1`** (D5): every `is_candidate_new == 1` record (equivalently `cluster_id == -1`) becomes a singleton item; T08 reads the flag and does not recompute it.
- [ ] `reid_demo/review.py` imports **no** third-party web framework (grep for `flask`/`fastapi`/`django` returns nothing); any (optional) web UI uses stdlib `http.server` only.
- [ ] On a seeded temp store (build it in the test from `reid_demo.store` directly — several records across ≥2 clusters, some with `cluster_conf < LOW_CONF_THRESHOLD`, plus ≥1 record with `is_candidate_new=1`), `build_review_queue(conn, dataset=...)` returns a `List[ReviewItem]` that: (a) contains the low-confidence assignments ordered by ascending `cluster_conf`, (b) contains a singleton item for each `is_candidate_new=1` record, (c) never exceeds `queue_size`, (d) excludes records whose `review_status != 'unreviewed'`.
- [ ] With `respect_flanks=True`, no returned `ReviewItem` of kind `pair` has `orientation_a='left'` and `orientation_b='right'` (or vice-versa).
- [ ] **(D8, optional ordering)** `build_review_queue` accepts an optional `pair_scores` mapping and, when given, reorders the queue so highest GV-disagreement / lowest-confidence borderline pairs come first, WITHOUT changing which items are queued; with `pair_scores=None` (the default) the queue is identical to the pre-D8 ascending-`cluster_conf` ordering (graceful degradation). T08 imports neither T11 nor T12 — its hard deps remain {T01, T02, T05} (verifiable: `grep` for `reid_demo.fisher`/`reid_demo.fusion` in `reid_demo/review.py` returns nothing). A malformed/partial `pair_scores` map never crashes the builder.
- [ ] `apply_decisions` with a `same` decision on a cross-cluster pair MERGES the two clusters (every member of the larger-id cluster is reassigned to the smaller id; `review_status` becomes `merged`/`confirmed`) — verified by re-querying the store; cluster count drops by exactly 1.
- [ ] `apply_decisions` with a `different` decision SPLITS `record_id_a` into a fresh `cluster_id = max+1` and sets `review_status` to `split`/`rejected`.
- [ ] `apply_decisions` with a `new` decision on a singleton assigns a fresh non-negative `cluster_id` and sets `review_status='confirmed'`.
- [ ] All store writes go through `store.update_review`; an invalid `answer` not in `DECISIONS` raises `ValueError`; `apply_decisions` writes a session JSON whose structure matches the documented decisions-JSON format and can be re-fed to `--apply` (round-trip) producing the same store state (idempotent — applying twice changes nothing).
- [ ] `review_status_summary(conn, dataset=...)` returns the documented plain-language keys with correct integer counts on the seeded store (e.g. `individuals_before`, `individuals_after`, `merges_applied`, `new_individuals_confirmed`, `still_unreviewed`).
- [ ] `build_pair_image` returns an image object for a `pair` item and for a `singleton` item, and does NOT raise when `crop_path_a` points to a missing file (falls back to bbox-crop or placeholder).
- [ ] `python -m reid_demo.review --build-queue --dataset <NAME> --db <temp.sqlite> --out /tmp/q.json` writes a queue JSON and makes no store mutations (row contents unchanged before/after).
- [ ] `python -m reid_demo.review --apply /tmp/decisions.json --db <temp.sqlite>` applies decisions and prints a summary; `--status --dataset <NAME>` prints the summary dict.
- [ ] `python -m pytest tests/test_review.py -q` passes under the repo venv.

## How to verify

```bash
cd /home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project
source venv/bin/activate    # repo's active env (Python 3.12; flask NOT installed)

# 0. Contract surface importable
python -c "from reid_demo.review import build_review_queue, apply_decisions, serve_review_ui, review_status_summary, build_pair_image, ReviewItem, ReviewDecision, DECISIONS; print('imports OK')"

# 0b. No web framework dependency
! grep -RniE 'import (flask|fastapi|django)|from (flask|fastapi|django)' reid_demo/review.py && echo "stdlib-only UI OK"

# 1. Seed a store, cluster a few records by hand, build a queue, apply decisions headlessly
python - <<'PY'
import os, json
from reid_demo.store import (connect, upsert_records, update_cluster, query_records,
                             DetectionRecord, make_record_id, count_by)
from reid_demo.review import (build_review_queue, apply_decisions, review_status_summary,
                              ReviewDecision)

db = "/tmp/reid_review.sqlite"
if os.path.exists(db): os.remove(db)
conn = connect(db)

# two clusters: cluster 1 (confident) and cluster 2 (one shaky member), + one singleton
recs = []
def mk(stem, idx, ds="DemoDS", orient="left"):
    return DetectionRecord(
        record_id=make_record_id(stem, idx),
        source_image=f"data/x/{stem}.JPG", source_stem=stem, det_index=idx,
        crop_path=f"/tmp/{stem}__crop{idx}.jpg",
        bbox_x=0.1,bbox_y=0.1,bbox_w=0.2,bbox_h=0.2,
        species="leopard", orientation=orient, dataset=ds)
for s in ["A1","A2"]: recs.append(mk(s,1))
for s in ["B1"]:      recs.append(mk(s,1))
recs.append(mk("B2",1))          # shaky member of cluster 2
recs.append(mk("S1",1))          # singleton candidate-new
upsert_records(conn, recs)

update_cluster(conn, "A1__crop1", 1, 0.95)
update_cluster(conn, "A2__crop1", 1, 0.93)
update_cluster(conn, "B1__crop1", 2, 0.90)
update_cluster(conn, "B2__crop1", 2, 0.40)               # low-confidence -> review
update_cluster(conn, "S1__crop1", -1, 0.10, is_candidate_new=1)  # singleton

q = build_review_queue(conn, dataset="DemoDS")
print("queue size:", len(q))
for it in q: print(" ", it.kind, it.item_id, "conf=", it.cluster_conf, "reason=", it.reason)
assert any(it.kind=="singleton" for it in q), "expected a singleton item"
assert any(it.kind=="pair" and it.record_id_a=="B2__crop1" for it in q), "expected B2 low-conf pair"

# human says B2 actually belongs with cluster 2 (same) and S1 is a new individual
decisions = []
for it in q:
    if it.record_id_a == "B2__crop1":
        decisions.append(ReviewDecision(item_id=it.item_id, answer="same", note="same flank pattern"))
    if it.kind == "singleton":
        decisions.append(ReviewDecision(item_id=it.item_id, answer="new"))
summary = apply_decisions(conn, q, decisions, dataset="DemoDS", session_path="/tmp/sess.json")
print("apply summary:", json.dumps(summary, indent=2)[:400])

print("clusters now:", count_by(conn, "cluster_id", dataset="DemoDS"))
print("status:", review_status_summary(conn, dataset="DemoDS"))

# round-trip: re-apply the saved session -> no change
import copy
before = {r.record_id:(r.cluster_id, r.review_status) for r in query_records(conn, dataset="DemoDS")}
sess = json.load(open("/tmp/sess.json"))
from reid_demo.review import ReviewItem
items2 = [ReviewItem(**{k:v for k,v in i.items() if k in ReviewItem.__dataclass_fields__}) for i in sess["items"]]
decs2  = [ReviewDecision(**{k:v for k,v in d.items() if k in ReviewDecision.__dataclass_fields__}) for d in sess["decisions"]]
apply_decisions(conn, items2, decs2, dataset="DemoDS", session_path="/tmp/sess2.json")
after = {r.record_id:(r.cluster_id, r.review_status) for r in query_records(conn, dataset="DemoDS")}
assert before == after, "apply must be idempotent"
print("idempotent re-apply OK")
PY

# 2. CLI: build-queue makes no mutations; status prints summary
python -m reid_demo.review --build-queue --dataset DemoDS --db /tmp/reid_review.sqlite --out /tmp/q.json
python -m reid_demo.review --status     --dataset DemoDS --db /tmp/reid_review.sqlite

# 3. CLI: headless apply from a decisions JSON (use the session written above)
python -m reid_demo.review --apply /tmp/sess.json --db /tmp/reid_review.sqlite

# 4. Tests
python -m pytest tests/test_review.py -q

# 5. (manual / OPTIONAL — only if the nice-to-have web UI was built; D7a) click through a few items
#    python -m reid_demo.review --serve --dataset DemoDS --db /tmp/reid_review.sqlite --port 8765
#    then open http://127.0.0.1:8765 and verify side-by-side crops + Same/Different/New/Skip buttons,
#    that decisions persist (re-run --status shows reduced 'still_unreviewed'), and 'Finish' stops the server.
#    Skip this whole step if the optional UI was not implemented — the must-ship core (steps 0–4) stands alone.
```

## Open questions

1. **Pair exemplar choice**: the spec picks the highest-`cluster_conf` member of the same cluster as `record_id_b`. If T05 emits a per-cluster medoid/exemplar in `extra_json`, T08 should prefer it — confirm with T05 whether such an exemplar key exists; absent it, highest-conf member is the fallback (documented).
2. **Singleton "belongs to existing cluster"**: a singleton genuinely belonging to an *existing* individual needs a target cluster. In the must-ship headless core this is surfaced purely via `target_cluster_id` in the decisions JSON; the default singleton flow is just New vs Skip. The optional web UI MAY add a dropdown if built. Full "assign to existing" interactive UX is Phase 2.
3. **`is_candidate_new` after confirmation (RESOLVED — D5/D6c)**: the flag is NOT cleared by T08. T08's only `update_review`-writeable fields are `review_status`, `review_note`, `cluster_id`, so a human-confirmed `new` singleton keeps `is_candidate_new == 1` while gaining a non-negative `cluster_id` and `review_status == 'confirmed'`. Downstream (T06's candidate-new count is keyed on `is_candidate_new` per D6c) must treat any reviewed row (`review_status != 'unreviewed'`) as resolved rather than double-counting it. No `extra_json` / follow-up helper is needed from T08.
4. **Re-run safety (RESOLVED — D5)**: T05 clustering runs BEFORE T08, and a later T05 re-run must NOT wipe T08's `review_status` writes — T05 preserves rows where `review_status != 'unreviewed'` (or refuses without an explicit `--force`). T08's only obligation is to exclude already-reviewed rows from its queue (which `build_review_queue` does). No additional guard is required on the T08 side.
5. **Multi-flank individuals (RESOLVED — D4)**: spot-bearing flanks {left, right} cluster in separate buckets; {front, back, down, unknown, ''} are pooled into a single 'other' bucket — the canonical 3-bucket convention {left, right, other} owned by T05/the DATA_CONTRACT. T08 honours it by never proposing or applying a cross-flank merge (left↔right) when `respect_flanks=True`; treat 'unknown'/'other'/None as compatible with anything. T08 does not attempt to reconcile a left and right flank of the same physical animal into one individual (Phase 2).
6. **Concurrency**: single-user assumed. If two browser tabs are open (optional UI only), last-write-wins via `update_review`. Acceptable for the demo; note it.
