# T05 — Open-set clustering engine

> **Status:** 🔴 Not started · **Milestone:** M1 (Demo-ready)
> **Depends on:** T01, T04 · **Blocks:** T06, T07, T08, T10, T12
> **Owner:** _unassigned_
>
> Self-contained ticket for the lynx re-identification demo/pilot. See [STATUS_BOARD.md](../STATUS_BOARD.md) for the full plan, the binding design decisions, and the agent working protocol.

---

## Signal layering (D8 — pluggable affinity)

> **Amendment per binding decision D8 (see STATUS_BOARD.md).** This ticket is the GLOBAL BACKBONE and must build/run standalone with no dependency on T11/T12.
>
> - `cluster_embeddings(...)` (and the run driver) MUST accept an **optional precomputed pairwise affinity** — an `(N, N)` similarity matrix (or a provider callable returning one) aligned to the same ordered `record_id` list as the embedding matrix. When **not** supplied, build the global-cosine affinity internally exactly as specified below (current behaviour) — the default `--signals global` path.
> - When supplied (by **T12**, wired through **T10** for `--signals global+fisher` / `full-funnel`), cluster on that affinity instead. Do **not** import `reid_demo.fisher` (T11) or `reid_demo.fusion` (T12) — T10 computes the affinity and passes it in.
> - The embedding contract (D2), flank bucketing (D4), and the singleton/candidate-new + re-run-safety rules (D5) all still apply unchanged, whichever affinity is used.
> - **Acceptance (added):** clustering on a supplied affinity yields identical labels to clustering its equivalent internally-built affinity; the supplied-affinity path is covered by a test.

## Context

We are building a DEMO + PILOT MVP of an **open-set, individual-animal re-identification** system for Eurasian lynx (closest public analogs: spotted big cats — LeopardID2022 leopards, ATRW Amur tigers). The existing repo (`/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project`) does CLOSED-SET re-id (match a query against a KNOWN gallery via gallery-argmax). **This ticket pivots the decision layer to OPEN-SET CLUSTERING.**

Given an unlabeled pile of animal crops that have already been detected, classified, and embedded by upstream tickets, this engine must:
1. **Discover how many DISTINCT individuals are present** (the count is UNKNOWN — there is no fixed gallery).
2. Assign each crop a `cluster_id` (one cluster = one discovered individual) with a **per-crop confidence**.
3. **Flag singletons / unmatched crops as candidate NEW individuals** (`is_candidate_new = 1`).
4. Be **flank-aware**: lynx (and leopard) **left and right flanks are DIFFERENT spot patterns**. Crops with `orientation="left"` must NEVER be clustered with `orientation="right"`. Use the `orientation` field from the store. Per the DATA_CONTRACT 3-bucket policy, spot-bearing flanks `{left, right}` cluster in SEPARATE buckets, while `{front, back, down, unknown, ''}` are POOLED into a single `other` bucket (they are not individually re-identifiable from spot pattern, and pooling avoids inflating the individual count by splitting one animal across up to 5 buckets).

You are an autonomous coding agent. You have repo access but were NOT present for the planning conversation. Everything you need is in this ticket plus the **shared data contract (T01)** at `reid_demo/DATA_CONTRACT.md` and `reid_demo/store.py`. You obtain embeddings via the T04 embedding API (`get_embedding_matrix(normalize=True)`) and write `cluster_id` / `cluster_conf` / `is_candidate_new` back into the T01 store. **You do not detect, classify, embed, build catalogues, evaluate, or build UIs** — those are other tickets (T02/T03/T04/T06/T07/T08).

### Where this sits in the pipeline (one crop = one row)
```
raw image --(T02 MegaDetector)--> crop + bbox + camera + timestamp
          --(T03 SpeciesNet)----> species + species_conf (keep/drop filter)
          --(T04 Embedding)------> embedding_ref + embedding_path (key into a .pkl)
          --(T05 THIS TICKET)----> cluster_id + cluster_conf + is_candidate_new   <-- YOU ARE HERE
          --(T06 catalogue / T07 eval / T08 HITL / T09 report / T10 runner): downstream consumers
```

### Key domain facts you must honor
- **Embeddings** are MegaDescriptor global vectors stored RAW (NOT L2-normalized) at **model-native dimension** — 1536 for the base `megadescriptor-l-384`, and 384 ONLY for a `linear_l2` projection checkpoint. T04 is authoritative on the embedding contract. You MUST obtain your matrix via the T04 API `get_embedding_matrix(normalize=True)` and **read the dimension FROM the returned matrix** — never hard-code 384, never assume unit norm. `normalize=True` L2-normalizes for you so cosine similarity is the right metric. (See `global_embedding.py` and the embedding-contract section in T01/T04.)
- **The store** is the single source of truth. The relevant columns you read/write (full schema in `reid_demo/DATA_CONTRACT.md`):
  - read: `record_id` (TEXT PK), `embedding_ref` (TEXT, key into the embedding matrix), `embedding_path` (TEXT, path to the embedding artifact), `orientation` (one of `left`,`right`,`front`,`back`,`down`,`unknown`; empty-string `''` is normalized to `unknown` at ingest — never store NULL/`''` here once T02 has run), `dataset` (TEXT, run name), `species`.
  - write (via the store's `update_cluster`): `cluster_id` (INTEGER; `>=0` an individual, `-1` = noise/unassigned, sklearn DBSCAN convention), `cluster_conf` (REAL `[0,1]`), `is_candidate_new` (INTEGER 0/1).
- **Flank / orientation bucketing (DATA_CONTRACT 3-bucket policy)**: clustering operates over exactly THREE buckets — `left`, `right`, and `other`. `{left, right}` (spot-bearing flanks) each cluster on their own; `{front, back, down, unknown, ''}` are POOLED into the single `other` bucket. `''` is already normalized to `unknown` at ingest and therefore falls into `other`. Bucket iteration order is DETERMINISTIC (sorted bucket names: `left`, `other`, `right`) so global cluster ids are reproducible/idempotent.
- **`-1` semantics + candidate-new (single authoritative rule)**: follow sklearn DBSCAN — `cluster_id = -1` means "not confidently assigned to any individual". A crop that is a singleton (its final cluster has size 1) OR is DBSCAN noise gets BOTH `cluster_id = -1` AND `is_candidate_new = 1`. `is_candidate_new` is the field downstream keys on. There is NO "assign 0 for the 1-crop case" — a lone crop is always `cluster_id = -1, is_candidate_new = 1`.

### Pluggable affinity — T05 is the GLOBAL backbone; richer signals plug in (D8)

This engine is the **M1-core clustering backbone** and clusters on the **global MegaDescriptor embedding (T04) by default**, with **NO hard dependency on T11/T12**. By binding decision **D8 (multi-signal clustering)** the project layers richer similarity signals on TOP of this backbone — Fisher-vector cosine and geometric-verification reranking — selected at run time via T10's `--signals {global|global+fisher|full-funnel}` flag. So that T05 can serve as the consumer of those signals WITHOUT taking a dependency on them, the clustering affinity is **PLUGGABLE**:

- `cluster_embeddings(...)` (and `cluster_by_flank(...)`) accept an **OPTIONAL precomputed pairwise affinity** (a square similarity matrix) **or an affinity-provider callable**. When it is **not** supplied, T05 builds the **global cosine affinity internally exactly as today** (the default backbone behavior — nothing changes for `--signals global`).
- The supplied affinity, when present, is produced by **T12 (`reid_demo/fusion.py`)**: either a **fused global+Fisher** affinity (T04 + T11 cosine similarities, calibrated to `P(same)` via `calibration.py`) for `--signals global+fisher`, or that fused affinity **PLUS GV reranking** on a shortlist of borderline pairs for `--signals full-funnel`. T05 simply **clusters on whatever affinity it is handed**.
- **T05 MUST NOT import T11 (`fisher.py`) or T12 (`fusion.py`)** — the dependency direction is one-way: T12 depends on T05's pluggable interface, never the reverse. The **T10 runner** computes the T12 affinity when `--signals` requires it and passes it into T05; on the default `--smoke`/`global` path no affinity is supplied and T05 runs standalone. This keeps the demo from ever being blocked on the GV layer (the fused/GV path is a fast-follow accuracy layer, still M1).
- The affinity contract is signal-agnostic: a higher value means "more likely the same individual". Whether that number is raw global cosine (default), calibrated global+Fisher probability, or a GV-refined score is invisible to T05's clustering math — cosine-distance backends consume `distance = 1 - affinity` over the matched id order. T05's flank bucketing, candidate-new rule, confidence definition, and store writes are **unchanged** regardless of which signal set produced the affinity.

## Objective

Deliver `reid_demo/cluster.py`: a self-contained open-set clustering engine that loads crop embeddings (via the T01 store + the T04 embedding API `get_embedding_matrix(normalize=True)`), clusters them into an unknown number of individuals **per flank bucket** (`left`/`right`/`other`, deterministic sorted order), computes a per-crop assignment confidence, flags candidate-new singletons (`cluster_id=-1, is_candidate_new=1`), **writes results back into the store** via the T01 stage-write API, and prints a plain-language summary ("found N individuals among M crops; K candidate-new singletons"). It runs BEFORE T08 review and must NOT silently overwrite human review on re-run (see re-run safety below). Provide both a clean Python API and a CLI, plus unit tests.

## Scope

### In
- A clustering function that takes a `Dict[str, np.ndarray]` of embeddings (+ a parallel `orientation` map) and returns cluster labels, per-crop confidences, and candidate-new flags — **without touching the DB** (pure, testable core).
- A store-integrated driver that: reads records for a `dataset` from the T01 store (filtered by `species` per D7, NOT `species_kept`), obtains the embedding matrix via the T04 API `get_embedding_matrix(normalize=True)`, runs the core, and writes results back via `update_cluster`.
- **Flank-aware bucketing (3-bucket policy)**: cluster `left` and `right` in **separate buckets** and pool `{front, back, down, unknown, ''}` into a single `other` bucket; never merge across buckets. Iterate buckets in DETERMINISTIC sorted order (`left`, `other`, `right`). `cluster_id`s remain globally unique across buckets within a run.
- At least **two clustering backends** selectable by flag: (a) **DBSCAN** with cosine metric (reuse the proven `cluster_fisher_vectors` pattern from `deprecated/analyze_folder.py`), (b) an **agglomerative / threshold-based** backend (sklearn `AgglomerativeClustering`, average linkage, cosine, `distance_threshold`) that does NOT require pre-specifying the number of clusters. DBSCAN is the default.
- A **per-crop confidence** in `[0,1]` derived from intra-cluster cosine similarity (e.g. mean cosine similarity of the crop to its cluster's other members, or to the cluster medoid), with a defined value for singletons.
- **Candidate-new detection (single rule)**: any crop whose final cluster is a singleton (cluster of size 1) OR is labeled `-1` (noise) gets BOTH `cluster_id = -1` AND `is_candidate_new = 1`. All others `is_candidate_new = 0`.
- **Re-run safety**: T05 runs BEFORE T08 review. Re-running on a dataset must NOT silently overwrite human review — preserve rows whose `review_status != 'unreviewed'` unless an explicit, documented `--force` flag is passed.
- Optional, off-by-default **calibration** hook: accept a fitted `ScoreCalibrator` (`calibration.py`) to map raw cosine similarity → `P(same)` for the confidence value. When absent, fall back to raw cosine mapped to `[0,1]`.
- A **CLI** `python -m reid_demo.cluster ...` (signatures below), a `--force` flag (re-run safety), and a `--dry-run` mode that computes but does not write.
- Deterministic behavior given fixed inputs/seed.
- Unit tests in `tests/test_cluster.py`.

### Out (do NOT implement — other tickets own these)
- Detection / cropping / `megadetector` ingestion → **T02**.
- Species classification / filtering → **T03**.
- Embedding extraction (calling MegaDescriptor / `global_embedding.py`) → **T04**. You only *consume* the embeddings via the T04 `get_embedding_matrix(normalize=True)` API.
- The HTML/montage catalogue → **T06**.
- Evaluating clusters against ground truth (V-measure, ARI, individuals-found-vs-true) → **T07**. Do not import `gt_identity` for clustering; you must cluster **blind**.
- Human-in-the-loop review UI and re-applying human merges/splits → **T08**.
- Medvednica filtering report → **T09**; end-to-end runner / pitch bundle → **T10**.
- Any edit to existing pipeline files (`main.py`, `global_embedding.py`, `analyze_folder.py`, `calibration.py`, etc.). You ADD `reid_demo/cluster.py` and `tests/test_cluster.py` only (plus one additive line in `STATUS_BOARD.md`).
- Modifying the T01 schema or `reid_demo/store.py`.

## Inputs

1. **The T01 store** (SQLite), default path from `reid_demo.store.DEFAULT_DB_PATH` (`data/reid_demo/reid_demo.sqlite`). You read records via `reid_demo.store.query_records(...)` / `connect(...)`.
2. **The T04 embedding artifact(s)**, consumed via the T04 API `get_embedding_matrix(normalize=True)`, keyed by `embedding_ref` and located at the `embedding_path` stored on each record. Per the T04-authoritative contract: vectors are stored RAW at **model-native dimension (1536 base / 384 for a `linear_l2` checkpoint), NOT pre-normalized**; you call `get_embedding_matrix(normalize=True)` to L2-normalize and you **read the dimension from the returned matrix** (never hard-code 384, never assume unit norm). (You may also accept an explicit `--embeddings <path>` override for testing.)
3. **Per-record fields read from the store**: `record_id`, `embedding_ref`, `embedding_path`, `orientation`, `dataset`, `species`, `review_status` (for re-run safety).
4. **Tuning parameters** (CLI / kwargs): clustering backend, distance threshold / eps, min cluster size, calibrator path (optional), seed, `--force` (re-run safety).
5. You must NOT require: image files on disk, model loading, GPU, or network. This ticket operates **purely on precomputed embeddings + metadata**.

## Outputs

1. **`reid_demo/cluster.py`** providing the API + CLI below.
2. **Writes back into the T01 store** for each processed record: `cluster_id`, `cluster_conf`, `is_candidate_new` (via `reid_demo.store.update_cluster`). No other columns are modified by this ticket.
3. **A returned/printed summary object** (`ClusterResult` / dict) with at minimum: `n_crops`, `n_individuals` (number of distinct `cluster_id >= 0` that are non-singletons, plus a `n_clusters_total`), `n_candidate_new`, `n_noise`, per-bucket breakdown (keyed by the 3 buckets `left`/`right`/`other`), parameters used. Printed as a one-line human sentence AND as machine-readable JSON to stdout (`--json`).
4. **`tests/test_cluster.py`** (passes under the repo venv).
5. One additive line in `STATUS_BOARD.md` marking T05 deliverables. Do not author other tickets' rows.

## Interface contract (what downstream tickets may rely on)

Place these in `reid_demo/cluster.py`. **Do not rename.** Downstream (T06/T07/T08/T10) read results from the **store**, not from your internals; the API below is the supported surface for T10's runner and for tests.

### Module-level constants
```python
DEFAULT_BACKEND: str = "dbscan"                  # one of CLUSTER_BACKENDS
CLUSTER_BACKENDS: set[str] = {"dbscan", "agglomerative"}
DEFAULT_EPS: float = 0.30                         # DBSCAN cosine-distance eps (matches analyze_folder.py default)
DEFAULT_MIN_SAMPLES: int = 2                      # DBSCAN min_samples (>=2 so a lone crop is noise -> candidate-new)
DEFAULT_DISTANCE_THRESHOLD: float = 0.30          # agglomerative cosine-distance cut
NOISE_LABEL: int = -1                             # matches sklearn DBSCAN / T01 cluster_id == -1
```

### Core (pure, no DB) — fully unit-testable
```python
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

@dataclass
class CropClustering:
    """Result of clustering one flank bucket (or the merged global result)."""
    image_ids: List[str]            # crop record_ids, aligned with the arrays below
    labels: np.ndarray              # int cluster_id per crop; -1 = noise/unassigned
    confidences: np.ndarray         # float [0,1] per crop
    is_candidate_new: np.ndarray    # int 0/1 per crop

# Pluggable affinity (D8): an OPTIONAL precomputed pairwise affinity. Either a
# (N, N) similarity matrix aligned to the SORTED image_ids, or a provider callable
# (sorted_ids, normalized_embeddings) -> (N, N) similarity. Higher = more likely same
# individual. None (default) => build global cosine affinity internally (backbone behavior).
AffinityProvider = Callable[[List[str], np.ndarray], np.ndarray]
Affinity = Union[np.ndarray, AffinityProvider]

def cluster_embeddings(
    embeddings: Dict[str, np.ndarray],
    *,
    backend: str = DEFAULT_BACKEND,
    eps: float = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    calibrator: Optional["ScoreCalibrator"] = None,
    affinity: Optional[Affinity] = None,   # D8: precomputed pairwise affinity OR provider; None = global cosine
    seed: int = 42,
) -> CropClustering:
    """Cluster a single homogeneous group of embeddings (one bucket). Embeddings are
    model-native dim (1536 base / 384 linear_l2 checkpoint) and are NOT assumed
    pre-normalized — re-normalize defensively before cosine math. Read D from the array.

    Affinity (D8, PLUGGABLE): if `affinity` is None (default), build the GLOBAL cosine
    affinity internally exactly as today (the M1-core backbone — `--signals global`). If a
    precomputed (N, N) similarity matrix is given it MUST be aligned to this call's SORTED
    image_ids; if a provider callable is given it is invoked as `affinity(sorted_ids,
    normalized_embeddings) -> (N, N)`. A supplied affinity is treated as a same-individual
    similarity (higher = more similar); cosine-distance backends consume `distance = 1 -
    affinity`. Such an affinity is produced by T12 (fused global+Fisher, calibrated; or
    GV-refined for `full-funnel`) and passed in by the T10 runner — T05 NEVER imports
    T11/T12. Validate that a matrix affinity is square, symmetric-ish, and sized N; raise
    ValueError otherwise.

    Cosine metric (default backbone). Returns labels (>=0 individuals, -1 noise), per-crop
    confidence, and candidate-new flags (singletons and noise -> cluster_id=-1 AND
    is_candidate_new=1). No DB, no I/O, no import of T11/T12."""

def cluster_by_flank(
    embeddings: Dict[str, np.ndarray],
    orientations: Dict[str, Optional[str]],
    *,
    backend: str = DEFAULT_BACKEND,
    eps: float = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    calibrator: Optional["ScoreCalibrator"] = None,
    affinity: Optional[Affinity] = None,   # D8: precomputed pairwise affinity OR provider; None = global cosine
    flank_policy: str = "separate",   # 'separate' (default) | 'ignore'
    seed: int = 42,
) -> CropClustering:
    """Bucket crops into the 3 DATA_CONTRACT buckets {left, right, other} — left/right
    cluster separately and {front, back, down, unknown, ''} pool into 'other' — then
    cluster each non-empty bucket independently in DETERMINISTIC sorted bucket order
    ('left', 'other', 'right') and concatenate with GLOBALLY UNIQUE cluster_ids across
    buckets (offset each bucket's labels so ids never collide; noise stays -1). 'ignore'
    clusters all crops together (flank-blind; may merge left+right of one animal — WRONG
    for lynx/leopard, acceptable only for tigers/ablations).

    Affinity (D8): forwarded to each bucket's `cluster_embeddings` call. A provider callable
    is invoked per bucket on that bucket's SORTED ids + normalized embeddings (so cross-flank
    pairs never enter a within-bucket affinity). A precomputed MATRIX affinity, if supplied,
    must be a single GLOBAL (N, N) similarity aligned to the full sorted id set; T05 slices
    the within-bucket submatrix per bucket before clustering. None (default) builds global
    cosine per bucket — the backbone path. Still NEVER imports T11/T12."""

def assignment_confidence(
    embeddings: np.ndarray,          # (N, D); re-normalize defensively before cosine
    labels: np.ndarray,              # (N,) cluster ids, -1 = noise
    *,
    calibrator: Optional["ScoreCalibrator"] = None,
) -> np.ndarray:
    """Per-crop confidence in [0,1]: for a crop in a cluster of size >=2, the mean cosine
    similarity to the other members of its cluster (optionally passed through `calibrator`
    to get P(same)); for singletons and noise (-1), a defined low/zero confidence
    (document the exact value). Deterministic."""
```

### Store-integrated driver
```python
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
    per_flank: Dict[str, Dict[str, int]]   # bucket ('left'|'right'|'other') -> {crops, clusters, candidate_new}
    n_review_preserved: int     # rows skipped because review_status != 'unreviewed' (re-run safety)
    sentence: str               # human-readable one-liner

def run_clustering(
    db_path: str = None,                 # default reid_demo.store.DEFAULT_DB_PATH
    *,
    dataset: Optional[str] = None,       # filter records by dataset (None = all)
    embeddings_path: Optional[str] = None,  # override; else read each record's embedding_path
    backend: str = DEFAULT_BACKEND,
    eps: float = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    flank_policy: str = "separate",
    calibrator_path: Optional[str] = None,
    species_filter: Optional[str] = None,   # filter rows by the `species` column (D7); NOT species_kept
    require_embedding: bool = True,         # skip rows with NULL embedding_ref (warn count)
    force: bool = False,                    # re-cluster reviewed rows too (re-run safety; default preserves them)
    dry_run: bool = False,               # compute but DO NOT write to store
    seed: int = 42,
) -> ClusterRunSummary:
    """Read records (T01 query_records, filtered by the `species` column when species_filter
    is given — NOT species_kept), obtain the embedding matrix via the T04 API
    get_embedding_matrix(normalize=True) (read D from the matrix; do NOT assume 384/unit-norm),
    cluster by the 3-bucket flank policy in deterministic sorted order, and (unless dry_run)
    write cluster_id/cluster_conf/is_candidate_new back via reid_demo.store.update_cluster.
    Returns a summary. Rows missing an embedding are skipped (counted/warned).
    Re-run safety: rows with review_status != 'unreviewed' are PRESERVED (not overwritten)
    unless force=True; the count of preserved rows is reported. Among unreviewed rows the run
    is deterministic/idempotent: re-running recomputes identical cluster_* from scratch."""
```

### CLI
```
python -m reid_demo.cluster --dataset MedvednicaDS [--db <path>]
       [--backend dbscan|agglomerative] [--eps 0.30] [--min-samples 2]
       [--distance-threshold 0.30] [--flank-policy separate|ignore]
       [--species "eurasian lynx"] [--calibrator <path.pkl>]
       [--embeddings <path>] [--seed 42] [--force] [--dry-run] [--json]
```
- Exit 0 on success, non-zero on hard error (e.g. no embeddings found).
- Prints the human sentence by default; prints machine-readable JSON of `ClusterRunSummary` when `--json`.
- `--species` filters on the store's `species` column (D7), NOT `species_kept`.
- `--force` re-clusters reviewed rows too; without it, rows with `review_status != 'unreviewed'` are preserved.
- `--dry-run` computes and prints the summary but performs NO store writes.

### File-format guarantees for downstream
- After a non-dry run, for every processed (unreviewed, embedded) record in `dataset`, the store columns `cluster_id` (INTEGER, `>=0` or `-1`), `cluster_conf` (REAL in `[0,1]`), `is_candidate_new` (0/1) are populated. `(dataset, cluster_id)` groups crops into one individual (the join key T06/T07/T08 use).
- Every singleton/noise crop carries BOTH `cluster_id = -1` AND `is_candidate_new = 1` (the single authoritative candidate-new rule); `is_candidate_new` is the field downstream keys on. There is no "assign 0" for lone crops.
- Cluster ids are non-negative integers per individual, **globally unique within a `(dataset)` run** even across the 3 flank buckets; `-1` reserved for noise/singletons. Deterministic sorted bucket order makes ids reproducible across re-runs.
- Rows already reviewed by T08 (`review_status != 'unreviewed'`) are NOT overwritten unless `--force`; their cluster_* are left intact.
- No other store columns are written by this ticket.

## Existing code to reuse (REAL paths — read these, do not reinvent)

- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/deprecated/analyze_folder.py` — `cluster_fisher_vectors(fv_dict, eps=0.3, min_samples=1)` (lines 48-55) is the canonical DBSCAN-with-cosine pattern: `DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(np.stack(list(fv_dict.values())))`, returning `(labels_, image_ids)`. Reuse this pattern for the DBSCAN backend (it currently clusters Fisher vectors; you apply the identical recipe to the MegaDescriptor embeddings at their model-native dim — read D from the matrix, do NOT hard-code 384). NOTE: it defaults `min_samples=1`; you default to `2` so a lone crop becomes noise → candidate-new.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/utils/distance_utils.py` — `fisher_distance(v1, v2) -> 1 - cosine_sim(normalize(v1), normalize(v2))`. Same cosine-distance formulation you need; reuse for confidence/medoid computations (note: it L2-normalizes internally; embeddings are already normalized but re-normalizing is harmless).
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/calibration.py` — `ScoreCalibrator` (lines 12-70): `.fit(scores, labels)`, `.predict_proba(scores) -> [0,1]`, `.save(path)` / `.load(path)`. If `--calibrator` is given, `ScoreCalibrator.load(path)` and pass raw cosine similarities through `.predict_proba(...)` to produce `cluster_conf`. Do NOT fit calibrators here (that needs GT and is T07's concern); only optionally CONSUME a pre-fit one.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/global_embedding.py` — `load_or_build_global_embeddings(image_paths, cache_path, ...) -> Dict[str, np.ndarray]` (lines 191-210) and the cache format. You do NOT call this (that's T04); you obtain vectors via the T04 `get_embedding_matrix(normalize=True)` API keyed by `embedding_ref`. Stored vectors are RAW at model-native dim (1536 base / 384 linear_l2 checkpoint), NOT pre-normalized — read D from the matrix and let `normalize=True` handle L2.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/reid_demo/store.py` (produced by T01) — use `connect`, `query_records`, `update_cluster`, `DEFAULT_DB_PATH`, `DetectionRecord`, `COLUMNS`, and read `review_status` per record for re-run safety. Read `reid_demo/DATA_CONTRACT.md` (the column table + join rules + orientation value set / 3-bucket policy) before coding. The relevant write API is exactly: `update_cluster(conn, record_id, cluster_id, cluster_conf, is_candidate_new=0)`.
- The **T04 embedding API** `get_embedding_matrix(normalize=True)` is the supported way to obtain vectors: it returns the matrix + id order at the model-native dimension and (with `normalize=True`) L2-normalizes for you. Read D from the returned matrix; do NOT `pickle.load` raw vectors and assume 384/unit-norm.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/nested_importance_sampling.py` — `_l2_normalize_rows`, `_stack_vectors(vectors, image_ids)` (lines 17-48) show the exact idiom for stacking a `Dict[str, ndarray]` into an `(N, D)` L2-normalized matrix in a fixed id order. Reuse this idiom (or reimplement minimally) for any local similarity/medoid math so ordering is stable; re-normalize defensively since stored vectors are RAW.

## Implementation notes

- **Dependencies**: stdlib + `numpy` + `scikit-learn` (already in the repo venv; `DBSCAN`, `AgglomerativeClustering` from `sklearn.cluster`) + `reid_demo.store` + optionally `calibration.ScoreCalibrator`. No new third-party deps.
- **Cosine throughout**: stored embeddings are RAW (NOT pre-normalized) — obtain them via `get_embedding_matrix(normalize=True)` and additionally re-normalize defensively (`_l2_normalize_rows` idiom) so cosine distance `= 1 - dot`. Read the dimension D from the matrix; never hard-code 384. For DBSCAN pass `metric="cosine"`. For AgglomerativeClustering use `metric="cosine", linkage="average", distance_threshold=..., n_clusters=None`.
- **Stable ordering**: sort `image_ids` (e.g. by `record_id`) before stacking so labels are deterministic given a fixed seed; document that DBSCAN/agglomerative themselves are deterministic for fixed input order.
- **Flank-aware bucketing** (`cluster_by_flank`, default `flank_policy="separate"`) — the DATA_CONTRACT 3-bucket policy `{left, right, other}`:
  - Map each crop's `orientation` to a bucket: `left -> 'left'`, `right -> 'right'`, and `{front, back, down, unknown, ''}` (and any non-canonical/None value) `-> 'other'`. `''` is already normalized to `unknown` at ingest. Cluster each non-empty bucket independently with the chosen backend.
  - Iterate buckets in **deterministic sorted order** (`'left'`, `'other'`, `'right'`) so global cluster ids are reproducible/idempotent. Assign **globally unique** non-negative ids: keep a running `max_id`; offset each bucket's `>=0` labels by `(max_id + 1)`; leave `-1` as `-1`. This guarantees no id collisions across buckets while preserving the DBSCAN `-1` convention.
  - Rationale for pooling `other`: `front/back/down/unknown` are not individually re-identifiable from spot pattern, and pooling avoids inflating the individual count by splitting one animal across up to 5 buckets. Only `left`/`right` (spot-bearing flanks) stay separate.
  - `flank_policy="ignore"` clusters everything in one group (useful for ATRW/front-on tigers or ablations); document the tradeoff (may merge left+right of the same animal into one cluster — acceptable for tigers, WRONG for lynx/leopard, hence default `separate`).
- **Confidence definition** (must be documented in a module docstring AND in code comments):
  - For a crop in a cluster of size `>= 2`: confidence = mean cosine similarity to the other members of the same cluster (range mapped to `[0,1]` via `max(0, sim)`; if `calibrator` given, run that mean-sim — or each pairwise sim then average — through `calibrator.predict_proba` and clip to `[0,1]`).
  - For a singleton cluster (size 1) or noise (`-1`): confidence = `0.0` (an unmatched crop has no support). State this constant explicitly.
- **Candidate-new rule** (single authoritative rule, must match T01 semantics): a crop whose final cluster size is 1 OR whose label is `-1` gets BOTH `cluster_id = -1` AND `is_candidate_new = 1`; all others keep their `>=0` id with `is_candidate_new = 0`. Compute cluster sizes AFTER the global-offset merge so a singleton in any bucket is still a singleton globally, then collapse those singletons to `-1`. `is_candidate_new` is what downstream keys on. Do NOT assign `0` to lone crops.
- **`n_individuals` vs `n_clusters_total`**: `n_clusters_total` = count of distinct `cluster_id >= 0`. `n_individuals` = count of those whose size `>= 2` (a confirmed individual with corroborating photos). Report both; the human sentence uses `n_clusters_total` as "found X individuals" but the summary keeps the distinction.
- **Driver embedding lookup**: obtain vectors via the T04 API `get_embedding_matrix(normalize=True)` keyed by `record.embedding_ref` (usually `== record_id`), reading from `record.embedding_path`. Read the dimension D from the returned matrix; do not assume 384 or unit norm beyond what `normalize=True` guarantees. If `--embeddings` is passed, use that single artifact for all records (test/override path). Skip + count records whose `embedding_ref` is NULL or missing from the matrix; if `require_embedding` and none resolve, exit non-zero with a clear message.
- **Species filter (D7)**: when `species_filter` is given, restrict the input rows by the store's `species` column (e.g. `species == "eurasian lynx"`). Do NOT filter on `species_kept` — that is a different column owned upstream.
- **Re-run safety (idempotency + review preservation)**: among UNREVIEWED rows, re-running on the same `dataset` recomputes `cluster_*` from scratch and overwrites via `update_cluster` (deterministic ⇒ identical results; do not append). Rows with `review_status != 'unreviewed'` (human merges/splits from T08) are PRESERVED — skipped from the write, counted in `n_review_preserved` — unless `force=True`/`--force` is passed, which re-clusters them too. T05 runs BEFORE T08 in the T10 stage ordering; it must never silently clobber human review.
- **Empty / tiny inputs**: 0 crops → summary with all zeros, exit 0. 1 crop → it is a candidate-new singleton: `cluster_id = -1` AND `is_candidate_new = 1` (the single rule; never `0`).
- **Determinism / seed**: pass `seed` where stochastic (DBSCAN/agglomerative are deterministic here; seed mainly guards any tie-breaking / sampling you add). Same inputs ⇒ same labels.
- Keep the pure core (`cluster_embeddings`, `cluster_by_flank`, `assignment_confidence`) free of any DB or file I/O so tests can run on synthetic embeddings without a store.
- Add a concise module docstring stating: inputs (embedding dict + orientation map), the cosine/flank/confidence/candidate-new rules, and the store columns written. Add one additive line to `STATUS_BOARD.md`.

## Acceptance criteria (testable checklist)

- [ ] `reid_demo/cluster.py` and `tests/test_cluster.py` exist; no existing repo file is modified except one additive line in `STATUS_BOARD.md`.
- [ ] `python -c "from reid_demo.cluster import cluster_embeddings, cluster_by_flank, assignment_confidence, run_clustering, CropClustering, ClusterRunSummary, DEFAULT_BACKEND, CLUSTER_BACKENDS, DEFAULT_EPS, DEFAULT_MIN_SAMPLES, DEFAULT_DISTANCE_THRESHOLD, NOISE_LABEL"` succeeds (every contracted name importable).
- [ ] **Model-native dim, not pre-normalized**: the pure core reads the dimension D from the input matrix (works for both 384 and 1536) and does not assume unit norm — it re-normalizes defensively; the driver obtains its matrix via `get_embedding_matrix(normalize=True)`. No hard-coded `384` anywhere.
- [ ] **Recovers known clusters**: given synthetic embeddings forming 3 well-separated tight groups (e.g. 3 random unit centroids + small noise, 5 crops each), `cluster_embeddings(...)` returns exactly 3 non-negative clusters with the correct membership (verify via adjusted Rand index == 1.0 against the planted labels, or exact membership match).
- [ ] **Singleton → candidate-new**: a planted lone crop (far from all groups) gets `cluster_id == -1` AND `is_candidate_new == 1` and `cluster_conf == 0.0`.
- [ ] **Confidence in range & sensible**: all returned `confidences` are in `[0,1]`; crops in a tight cluster have confidence noticeably higher (e.g. `>= 0.8`) than singletons (`== 0.0`).
- [ ] **Flank separation (3-bucket policy)**: `cluster_by_flank` with `flank_policy="separate"` on inputs where the SAME embedding vector appears tagged `orientation="left"` and `orientation="right"` places them in DIFFERENT clusters (never the same `cluster_id`). Crops with `front`/`back`/`down`/`unknown`/`''` are POOLED into one `other` bucket (a `front` and a `back` of the same vector CAN co-cluster). With `flank_policy="ignore"` the left/right construction may co-cluster (documents the difference).
- [ ] **Deterministic bucket order / globally unique ids**: after `cluster_by_flank` over multiple buckets (iterated in sorted order `left`,`other`,`right`), every non-negative `cluster_id` is unique to one bucket (no id reused across buckets); `-1` preserved; the id assignment is reproducible across runs.
- [ ] **Backends**: both `backend="dbscan"` and `backend="agglomerative"` run and return valid `CropClustering` (labels length == n crops, no exception) on the synthetic 3-group fixture; an invalid backend raises `ValueError`.
- [ ] **Store round-trip**: seed a temp T01 DB (via `reid_demo.store`) with N records carrying `embedding_ref`/`embedding_path`/`orientation` and a matching embedding artifact; `run_clustering(db_path=..., dataset=...)` writes `cluster_id`/`cluster_conf`/`is_candidate_new` for every embedded unreviewed record (verify by re-querying the store); rows with NULL `embedding_ref` are skipped and counted, not written.
- [ ] **Re-run safety**: a row with `review_status != 'unreviewed'` keeps its existing `cluster_id` after a normal re-run (counted in `n_review_preserved`); with `--force`/`force=True` it is re-clustered.
- [ ] **Species filter (D7)**: `run_clustering(..., species_filter="eurasian lynx")` only clusters/writes rows whose `species` column equals that value (rows of other species untouched); the filter is on `species`, not `species_kept`.
- [ ] **`--dry-run` writes nothing**: after `run_clustering(..., dry_run=True)`, the store's `cluster_id` values are unchanged (still NULL) while the returned summary still reports a non-zero `n_clusters_total` for clusterable input.
- [ ] **Idempotency**: running `run_clustering` twice on the same dataset yields identical `cluster_id`/`cluster_conf` assignments (deterministic) and does not create duplicate rows.
- [ ] **Summary correctness**: `ClusterRunSummary` reports `n_crops`, `n_clusters_total`, `n_individuals` (size>=2), `n_candidate_new`, `n_noise`, `n_review_preserved`, and `per_flank` (keyed by the 3 buckets) consistent with the actual labels; `sentence` is a human-readable string mentioning individuals found and candidate-new count.
- [ ] **CLI**: `python -m reid_demo.cluster --dataset <ds> --db <temp.sqlite> --embeddings <pkl> --json` exits 0 and prints valid JSON of the summary; `--dry-run` performs no writes; unknown `--backend` exits non-zero.
- [ ] **Calibrator (optional path)**: if a fitted `ScoreCalibrator` pickle is supplied via `--calibrator`, confidences are produced through it (and remain in `[0,1]`); absence of a calibrator does not error.
- [ ] `tests/test_cluster.py` passes under the repo venv.

## How to verify

```bash
cd /home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project
source venv/bin/activate    # or the repo's active env

# 0. Contract surface importable
python -c "from reid_demo.cluster import cluster_embeddings, cluster_by_flank, assignment_confidence, run_clustering, CropClustering, ClusterRunSummary, DEFAULT_BACKEND, CLUSTER_BACKENDS, DEFAULT_EPS, DEFAULT_MIN_SAMPLES, DEFAULT_DISTANCE_THRESHOLD, NOISE_LABEL; print('OK')"

# 1. Pure core: recovers 3 planted clusters + 1 singleton, flank separation, confidence range
python - <<'PY'
import numpy as np
from sklearn.metrics import adjusted_rand_score
from reid_demo.cluster import cluster_embeddings, cluster_by_flank

rng = np.random.default_rng(0)
# Use model-native base dim (1536). Core must read D from the matrix, not hard-code 384.
# Vectors are RAW (NOT pre-normalized) on purpose; the core re-normalizes defensively.
DIM = 1536
centroids = rng.normal(size=(3, DIM))
emb, gt = {}, []
for ci,c in enumerate(centroids):
    for j in range(5):
        emb[f"id{ci}_{j}"] = (c + 0.01*rng.normal(size=DIM)).astype("float32"); gt.append(ci)
# a lone outlier far away (RAW, not normalized)
emb["loner"] = rng.normal(size=DIM).astype("float32"); gt.append(99)

res = cluster_embeddings(emb, backend="dbscan", eps=0.30, min_samples=2)
order = res.image_ids
gt_map = {f"id{ci}_{j}": ci for ci in range(3) for j in range(5)}; gt_map["loner"]=99
gt_aligned = [gt_map[i] for i in order]
print("ARI:", adjusted_rand_score(gt_aligned, res.labels.tolist()))
li = order.index("loner")
assert res.labels[li] == -1, "loner must be cluster_id -1"
assert res.is_candidate_new[li] == 1 and res.confidences[li] == 0.0, "loner must be candidate-new, conf 0"
assert ((res.confidences >= 0) & (res.confidences <= 1)).all()
print("singleton -> cluster_id=-1 AND candidate-new OK; confidences in [0,1]")

# flank: same vector under left vs right must NOT co-cluster; front/back POOL into 'other'
v = rng.normal(size=DIM).astype("float32")
emb2 = {"a_left": v, "b_left": v, "a_right": v, "b_right": v, "a_front": v, "b_back": v}
ori = {"a_left":"left","b_left":"left","a_right":"right","b_right":"right",
       "a_front":"front","b_back":"back"}
r2 = cluster_by_flank(emb2, ori, backend="dbscan", eps=0.30, min_samples=2, flank_policy="separate")
lab = dict(zip(r2.image_ids, r2.labels.tolist()))
assert lab["a_left"] == lab["b_left"]            # same flank -> same cluster
assert lab["a_left"] != lab["a_right"]           # different flank -> different cluster
assert lab["a_front"] == lab["b_back"]           # front+back pooled into 'other' -> may co-cluster
assert lab["a_front"] != lab["a_left"] and lab["a_front"] != lab["a_right"]  # 'other' is its own bucket
print("3-bucket flank separation OK:", lab)
PY

# 2. Store round-trip + dry-run + idempotency (needs T01 store)
python - <<'PY'
import os, pickle, tempfile, numpy as np
from reid_demo import store
from reid_demo.store import connect, DetectionRecord, make_record_id, get_record, update_cluster  # noqa
from reid_demo.cluster import run_clustering

tmp = tempfile.mkdtemp()
db = os.path.join(tmp, "t.sqlite")
pkl = os.path.join(tmp, "emb.pkl")

# model-native base dim; vectors stored RAW (NOT pre-normalized) — the driver normalizes
DIM = 1536
rng = np.random.default_rng(1)
c0, c1 = rng.normal(size=DIM), rng.normal(size=DIM)
emb = {}
conn = store.connect(db)
plan = {
    "L0":(c0,"left"),"L1":(c0,"left"),"R0":(c1,"right"),"R1":(c1,"right"),
    "S":(rng.normal(size=DIM),"left"),
}
for k,(stem,(c,ori)) in enumerate(plan.items()):
    rid = make_record_id(stem, 1)
    emb[rid] = (c + 0.01*rng.normal(size=DIM)).astype("float32")   # RAW, not normalized
    store.upsert_record(conn, DetectionRecord(
        record_id=rid, source_image=f"{stem}.jpg", source_stem=stem, det_index=1,
        crop_path=f"{stem}__crop1.jpg", bbox_x=0.1,bbox_y=0.1,bbox_w=0.2,bbox_h=0.2,
        embedding_ref=rid, embedding_path=pkl, orientation=ori, dataset="DemoDS"))
with open(pkl,"wb") as f: pickle.dump(emb, f)

# dry-run writes nothing
s_dry = run_clustering(db_path=db, dataset="DemoDS", embeddings_path=pkl, dry_run=True)
assert all(get_record(conn, rid).cluster_id is None for rid in emb), "dry-run must not write"
print("dry-run summary:", s_dry.sentence)

# real run writes cluster ids
s = run_clustering(db_path=db, dataset="DemoDS", embeddings_path=pkl, flank_policy="separate")
got = {rid: get_record(conn, rid) for rid in emb}
assert all(g.cluster_id is not None and 0.0 <= (g.cluster_conf or 0) <= 1.0 for g in got.values())
# left pair and right pair are different clusters
assert got[make_record_id("L0",1)].cluster_id != got[make_record_id("R0",1)].cluster_id
# the lone 'S' crop is a singleton -> cluster_id == -1 AND is_candidate_new == 1
s_rec = got[make_record_id("S",1)]
assert s_rec.cluster_id == -1 and s_rec.is_candidate_new == 1, "singleton must be -1 + candidate-new"
print("n_clusters_total:", s.n_clusters_total, "candidate_new:", s.n_candidate_new, "noise:", s.n_noise)

# idempotency
labels1 = {rid:get_record(conn,rid).cluster_id for rid in emb}
run_clustering(db_path=db, dataset="DemoDS", embeddings_path=pkl, flank_policy="separate")
labels2 = {rid:get_record(conn,rid).cluster_id for rid in emb}
assert labels1 == labels2, "clustering must be deterministic/idempotent"
print("idempotent OK")

# re-run safety: mark one row reviewed, pin a sentinel cluster_id, re-run WITHOUT --force
from reid_demo.store import update_review  # T08 review-write API
rid_rev = make_record_id("L0", 1)
update_cluster(conn, rid_rev, 999, 1.0, 0)              # human-pinned cluster id
update_review(conn, rid_rev, review_status="confirmed")
run_clustering(db_path=db, dataset="DemoDS", embeddings_path=pkl, flank_policy="separate")
assert get_record(conn, rid_rev).cluster_id == 999, "reviewed row must be preserved on re-run"
# with force=True it is re-clustered
run_clustering(db_path=db, dataset="DemoDS", embeddings_path=pkl, flank_policy="separate", force=True)
assert get_record(conn, rid_rev).cluster_id != 999, "--force must re-cluster reviewed rows"
print("re-run safety OK")
PY

# 3. CLI smoke (JSON output, exit code)
python -m reid_demo.cluster --dataset DemoDS --db /tmp/does_not_matter.sqlite --backend bogus ; echo "expect non-zero exit=$?"

# 4. Unit tests
python -m pytest tests/test_cluster.py -q
```

## Open questions

1. **Confidence formula**: this ticket specifies "mean cosine similarity to other cluster members, singletons = 0.0". T06 (catalogue) and T08 (HITL ordering) consume `cluster_conf` to surface the *lowest-confidence* merges. Is mean-intra-cluster-similarity the right ranking signal for T08, or should low-confidence be the *minimum* pairwise sim / the medoid distance? Confirm with T08; the contract value stays in `[0,1]` regardless.
2. **Default `eps` / `distance_threshold`**: `0.30` is inherited from `analyze_folder.py` (tuned on Fisher vectors). MegaDescriptor cosine geometry at the model-native dim (1536 base / 384 linear_l2) may want a different cut for leopard/lynx vs tiger. T10's runner should be free to sweep these; this ticket only provides sensible defaults + the knobs. Flag if T07's eval wants a recommended-eps output.
3. **`min_samples` default = 2**: chosen so a single unmatched crop becomes noise → candidate-new (the desired open-set behavior). `analyze_folder.py` used `1`. Confirm 2 is the intended demo behavior (a single photo of a never-seen lynx should be a "candidate new", not silently its own confident individual).
4. **Multi-artifact datasets**: the driver obtains vectors via `get_embedding_matrix(normalize=True)` keyed by `embedding_ref`/`embedding_path`, allowing a dataset whose crops are split across multiple embedding artifacts. Confirm T04 writes a single artifact per dataset (the common case) so the `--embeddings` override is sufficient for the demo.

> RESOLVED by binding decisions (no longer open): (a) Flank/orientation policy — fixed by **D4**: spot-bearing flanks `{left, right}` cluster separately; `{front, back, down, unknown, ''}` POOL into a single `other` bucket; buckets are iterated in deterministic sorted order. There is no separate "unknown-flank" bucket. (b) Singleton/candidate-new convention — fixed by **D5**: singletons AND noise get `cluster_id = -1` AND `is_candidate_new = 1` (no "assign 0"). (c) Embedding contract — fixed by **D2**: vectors are model-native dim, stored RAW; consume via `get_embedding_matrix(normalize=True)` and read D from the matrix.
