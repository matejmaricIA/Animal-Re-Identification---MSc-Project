# T07 — Clustering evaluation harness

> **Status:** 🔴 Not started · **Milestone:** M1 (Demo-ready)
> **Depends on:** T01, T05 · **Blocks:** T10
> **Owner:** _unassigned_
>
> Self-contained ticket for the lynx re-identification demo/pilot. See [STATUS_BOARD.md](../STATUS_BOARD.md) for the full plan, the binding design decisions, and the agent working protocol.

---

## Context

We are building a DEMO + PILOT MVP of an **open-set, individual-animal re-identification** system for Eurasian lynx (closest public analog: spotted big cats — **LeopardID2022** leopards, **ATRW** Amur tigers). The existing repo (`/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project`) does CLOSED-SET re-id (query vs known gallery). The demo pivots the *decision layer only* to OPEN-SET CLUSTERING: take an unlabeled pile of animal crops, discover how many DISTINCT individuals are present (unknown count), and flag singletons that match nothing as candidate NEW individuals.

The whole demo is a constellation of independent modules (T01–T10) handed to separate AI agents. They all read/write the **same** per-crop "detection record" through a single shared SQLite store and a single Python access module defined in **T01** (`reid_demo/store.py`). **You (T07) only READ from that store** — you never insert, embed, or cluster. Your job is the **scorecard**: given the cluster labels that **T05** (the clustering engine) wrote into the store, and ground-truth individual identities (`gt_identity`) that exist for LeopardID2022 / ATRW, compute and report **how well the clustering matched reality** — in BOTH plain language a park biologist understands AND standard ML clustering metrics.

This is the credibility number for the pitch. The biologist-facing claim we want to be able to make truthfully is: *"On 1431 leopard photos of 430 known individuals, the system found 412 individuals; 96% of photos were correctly grouped with their true individual; it accidentally merged 3 pairs of different cats and split 5 cats into two groups each."* You produce exactly those numbers, plus the academically standard ones (homogeneity / completeness / V-measure / ARI / AMI).

### Pipeline position

```
... T04 embed --> T05 cluster (writes cluster_id, cluster_conf, is_candidate_new into store)
                     |
                     v
   T07 (YOU): read cluster_id + gt_identity from store --> evaluation report (JSON + plain-language + per-individual tables)
```

### Real repo facts you must respect

- **Cluster labels and GT live in the T01 store** (`reid_demo/store.py`, table `detections`). Each crop row has: `record_id`, `dataset`, `cluster_id` (int; `>=0` = a discovered individual, `-1` = noise/unassigned, `NULL` until T05 runs), `cluster_conf`, `is_candidate_new`, `gt_identity` (the true individual id, `NULL` for unlabeled field data), `orientation` (`left`/`right`/`front`/`back`/`down`/`unknown`), `species`, `crop_path`, `review_status`. Full schema in `reid_demo/DATA_CONTRACT.md` (written by T01).
- **GT identities for LeopardID2022 / ATRW** originally come from the raw `WildlifeReID10k` metadata via `utility_functions.load_dataset(subset)` — for LeopardID2022 this carries columns `identity`, `orientation` (the flank: left/right/front/back/down), `species`, `split`, `dataset`. **T02 is the SOLE owner of `gt_identity` and `orientation` for labeled datasets**: its `ingest_wildlife_dataset(subset, ...)` adapter populates `gt_identity`, `orientation`, and `species` into the store from those metadata columns at ingestion time (empty/missing orientation maps to `'unknown'`). Because T02 is upstream of embed/cluster/eval, the GT is ALWAYS present in the store before you run. You read `gt_identity`/`orientation` from the store; you do NOT reload `WildlifeReID10k` yourself.
- **LeopardID2022 left/right flanks are DIFFERENT spot patterns.** Clustering is flank-aware in T05 (it may cluster left and right flanks of the same animal into separate clusters on purpose). Your evaluation MUST support a **flank-aware GT-identity convention** that uses the **SAME `{left, right, other}` flank-bucket convention as T05** (NOT raw orientation): the spot-bearing flanks `left` and `right` are kept as separate buckets, while `{front, back, down, unknown, ''}` are POOLED into a single `other` bucket. When the run is flank-aware, the effective ground-truth label for a crop is `f"{gt_identity}|{flank_bucket}"` (e.g. a single physical leopard photographed from both sides counts as **two** "sides" — `LEO|left` and `LEO|right` — the system is expected to discover). Pooling non-flank orientations into `other` avoids inflating the true-individual count by splitting one animal across up to 5 buckets. You report under whichever convention the run used and state it explicitly. (See Implementation notes.)
- **sklearn is already installed** in the repo venv and provides every standard metric you need: `sklearn.metrics.homogeneity_completeness_v_measure`, `adjusted_rand_score`, `adjusted_mutual_info_score`. (Verified available in `venv`.)
- Existing evaluation output convention: closed-set eval writes JSON via `evaluate.save_evaluation_results(results, dataset_name, tag, output_dir=EVALUATION_DIR)` and `constants.EVALUATION_DIR = './evaluations/full_evals'`. Follow that *style* but write to your own clustering subdir so you never collide with closed-set evals.

## Objective

Deliver a single self-contained module `reid_demo/eval.py` (plus tests) that:

1. Loads, from the T01 store, the set of crops for a given `dataset` that have **both** a cluster assignment (`cluster_id IS NOT NULL`) **and** a ground-truth label (`gt_identity IS NOT NULL`).
2. Builds aligned `predicted` (cluster) and `true` (GT) label arrays under the chosen flank convention.
3. Computes a **`ClusteringReport`** containing:
   - **Plain-language / biologist metrics**: true individual count, discovered cluster count, photos-correctly-grouped %, number of **merge errors** (two different true individuals lumped into one cluster) and **split errors** (one true individual scattered across multiple clusters), and singleton / candidate-new stats.
   - **Standard ML metrics**: homogeneity, completeness, V-measure, ARI, AMI, plus a precision/recall/F1 over **pairs** of photos (BCubed-style pairwise).
4. Produces **per-individual** and **per-cluster** breakdown tables so T06 (catalogue) / T09 (report) can show which specific animals were merged or split.
5. Writes a JSON report to `evaluations/clustering/<dataset>_<tag>.json`, optionally a CSV/HTML summary, and prints a human-readable summary to stdout.
6. Exposes a CLI: `python -m reid_demo.eval --dataset LeopardID2022 [--db ...] [--flank-aware] [--tag ...]`.

You do NOT cluster, embed, render catalogues, or modify any existing repo file (except an additive line in `STATUS_BOARD.md`).

## Scope

### In
- `reid_demo/eval.py`: the full metric/report engine + CLI described under Interface contract.
- Reading cluster + GT labels exclusively through the **T01 public API** (`reid_demo.store`): `connect`, `query_records`, `to_dataframe`, `count_by`.
- A flank-aware GT-label convention toggle (`--flank-aware` / `flank_aware=True`), defaulting OFF (label = `gt_identity` only); when ON, label = `f"{gt_identity}|{flank_bucket}"` where `flank_bucket` is the T05-matching `{left, right, other}` bucketing of `orientation` (`left`→`left`, `right`→`right`, everything else `{front, back, down, unknown, ''}`→`other`).
- Plain-language metrics: individuals found vs true, % photos correctly grouped, merge count, split count, candidate-new precision/recall (a "candidate new" that is truly a singleton individual vs one that should have matched an existing cluster).
- Standard metrics: homogeneity, completeness, V-measure, ARI, AMI, and pairwise (BCubed-style) precision/recall/F1.
- Per-individual table (one row per true individual: how many photos, how many clusters it was split across, dominant cluster, was it split) and per-cluster table (one row per discovered cluster: size, dominant true individual, purity, was it a merge of >1 true individual).
- JSON report writer + optional `--html`/`--csv` summary; stdout pretty-print.
- Graceful handling of: noise label `-1`, crops with no GT (excluded, but counted/reported as "unlabeled, skipped"), empty intersection (clear error, no crash).
- Unit tests `tests/test_eval.py` with a synthetic in-memory store (no model, no images).

### Out
- Producing cluster labels (T05), embeddings (T04), detection (T02), species labels (T03).
- The visual catalogue (T06) — you emit *tables/JSON* it can consume, not HTML montages of crops.
- Closed-set retrieval metrics (top-1/top-5/mAP) — those already exist in `tools/evaluate_reid_embeddings.py`; do NOT reimplement or call them.
- The Medvednica field report (T09) and end-to-end runner (T10).
- Any change to `main.py`, `evaluate.py`, `tools/evaluate_reid_embeddings.py`, `constants.py`, or `reid_demo/store.py`.

## Inputs

- A populated T01 SQLite store (default `data/reid_demo/reid_demo.sqlite`) where, for a labeled dataset, rows have non-null `cluster_id` AND non-null `gt_identity`. For local development you may seed your own synthetic store via the T01 API (see tests) — you do NOT need T05 to have actually run.
- `--dataset` name to evaluate (e.g. `LeopardID2022`, `ATRW`). All metrics are computed on `query_records(dataset=<name>)` rows.
- Optional `--flank-aware` flag and `--tag` label for the output filename.

## Outputs

- `reid_demo/eval.py`, `tests/test_eval.py`.
- JSON report at `evaluations/clustering/<dataset>_<tag>.json` (parent dir auto-created) with the structure under Interface contract.
- Optional `evaluations/clustering/<dataset>_<tag>.csv` (per-individual table) and `<dataset>_<tag>.html` (one-page summary) when `--csv` / `--html` passed.
- An additive line in `STATUS_BOARD.md` marking T07 deliverables.

## Interface contract

Downstream tickets (T06 catalogue, T09 report, T10 runner) import these. Do **not** rename.

### Dataclass

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

@dataclass
class ClusteringReport:
    dataset: str
    tag: str
    flank_aware: bool

    # ---- counts (plain language) ----
    n_photos_total: int            # rows for this dataset in the store
    n_photos_labeled: int          # rows with non-null gt_identity (the eval set numerator base)
    n_photos_clustered: int        # rows with non-null cluster_id
    n_photos_evaluated: int        # rows with BOTH gt_identity AND cluster_id (>=0 or noise per policy)
    n_photos_noise: int            # evaluated rows whose cluster_id == -1
    n_true_individuals: int        # distinct GT labels in evaluated set (under chosen flank convention)
    n_found_clusters: int          # distinct predicted cluster_id (>=0) in evaluated set

    # ---- plain-language quality ----
    pct_photos_correctly_grouped: float   # % of evaluated photos in the "majority-correct" sense (defined in notes), 0..100
    n_merge_errors: int            # clusters containing >1 true individual (count of such clusters)
    n_split_errors: int            # true individuals spread across >1 cluster (count of such individuals)
    merged_individual_groups: List[List[str]]   # the true-identity groups that got merged together
    split_individuals: List[str]   # true identities that were split

    # ---- candidate-new (singleton) quality ----
    n_candidate_new: int                   # rows with is_candidate_new == 1 in evaluated set
    candidate_new_precision: Optional[float]  # of flagged candidate-new, fraction that are truly singletons in GT
    candidate_new_recall: Optional[float]     # of true GT singletons, fraction flagged candidate-new

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

    def to_dict(self) -> Dict[str, Any]: ...
    def plain_language_summary(self) -> str: ...   # multi-line biologist-readable string
```

### Public functions (exact signatures)

```python
def load_eval_frame(conn, dataset: str, *, flank_aware: bool = False):
    """Read this dataset's rows from the T01 store and return a pandas DataFrame with at least
    columns: record_id, cluster_id, gt_identity, orientation, is_candidate_new, crop_path,
    species, review_status, plus a derived 'gt_label' column = gt_identity (flank_aware=False)
    or f'{gt_identity}|{flank_bucket}' (flank_aware=True), where flank_bucket maps orientation
    to the T05-matching {left, right, other} convention (left->left, right->right, all of
    {front, back, down, unknown, ''}->other). Uses reid_demo.store.query_records /
    to_dataframe only. Does NOT drop unlabeled rows here (caller decides)."""

def build_label_arrays(df, *, include_noise: bool = True):
    """From the eval frame, return (y_true: List[str], y_pred: List[int], record_ids: List[str])
    over rows where gt_label is not null AND cluster_id is not null. If include_noise=False,
    drop rows with cluster_id == -1. y_true are gt_label strings, y_pred are cluster ints."""

def standard_metrics(y_true, y_pred) -> Dict[str, float]:
    """Return {'homogeneity','completeness','v_measure','adjusted_rand_index',
    'adjusted_mutual_info','pairwise_precision','pairwise_recall','pairwise_f1'}
    using sklearn.metrics. Pairwise = BCubed-style over co-clustered photo pairs."""

def plain_language_metrics(y_true, y_pred, df) -> Dict[str, Any]:
    """Return the biologist-facing block: n_true_individuals, n_found_clusters,
    pct_photos_correctly_grouped, n_merge_errors, n_split_errors,
    merged_individual_groups, split_individuals, and candidate-new precision/recall."""

def evaluate_clustering(conn, dataset: str, *, flank_aware: bool = False,
                        tag: str = "default", include_noise: bool = True) -> ClusteringReport:
    """Top-level: load frame, build arrays, compute both metric families, assemble breakdown
    tables, return a fully populated ClusteringReport. Raises ValueError with a clear message
    if the evaluated set is empty (no rows with both gt_identity and cluster_id)."""

def save_report(report: ClusteringReport, out_dir: str = "evaluations/clustering",
                *, write_csv: bool = False, write_html: bool = False) -> str:
    """Write the SINGLE report file <dataset>_<tag>.json (always; this is the one file T10
    reads the headline pct_photos_correctly_grouped / n_true_individuals / n_found_clusters from)
    and optionally .csv (per_individual) / .html. Create out_dir. Return the JSON path."""
```

### CLI

```
python -m reid_demo.eval --dataset LeopardID2022 [--db data/reid_demo/reid_demo.sqlite] \
    [--flank-aware] [--tag mytag] [--no-noise] [--csv] [--html] [--out-dir evaluations/clustering]
```
- Prints `report.plain_language_summary()` to stdout, writes the JSON, prints the JSON path.
- Exit 0 on success; exit non-zero with a clear message if the evaluated set is empty or the dataset is absent from the store.

### Output JSON format (downstream-stable)

**You write exactly ONE file per evaluation: `evaluations/clustering/<dataset>_<tag>.json`.** There is no second/companion file — T10 reads the headline straight out of this single file. Top-level keys MUST be exactly the `ClusteringReport.to_dict()` field names above (snake_case), so T06/T09/T10 can load with `json.load` and read e.g. `report["pct_photos_correctly_grouped"]`, `report["n_found_clusters"]`, `report["n_true_individuals"]`, `report["per_individual"]`, `report["per_cluster"]`. The **headline contract T10 depends on** is the trio `pct_photos_correctly_grouped` (a percentage on the **0..100 scale**, e.g. `96.1`, NEVER a 0..1 fraction like `0.961`), `n_true_individuals` (int), and `n_found_clusters` (int) — all three live in this same file. `per_individual` rows MUST contain keys: `gt_label`, `n_photos`, `n_clusters`, `dominant_cluster`, `is_split`. `per_cluster` rows MUST contain keys: `cluster_id`, `n_photos`, `dominant_gt_label`, `purity`, `n_true_individuals`, `is_merge`.

## Existing code to reuse (real paths)

- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/reid_demo/store.py` — **your only data source.** Use `connect(db_path)`, `query_records(conn, dataset=...)` (returns `List[DetectionRecord]`), `to_dataframe(conn, dataset=...)` (returns a pandas DataFrame in `COLUMNS` order), and `count_by(conn, 'cluster_id', dataset=...)` / `count_by(conn, 'gt_identity', dataset=...)`. The record fields you need (`record_id`, `cluster_id`, `gt_identity`, `orientation`, `is_candidate_new`, `crop_path`, `species`, `review_status`, `dataset`) are all in the schema documented in `reid_demo/DATA_CONTRACT.md`. **Do not query SQLite directly; go through this module so the contract holds.**
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/evaluate.py` — `save_evaluation_results(results, dataset_name, tag, output_dir=EVALUATION_DIR)` shows the JSON-dump style/conventions to mirror (timestamped filename, `json.dump(..., indent=2)`). You write to `evaluations/clustering/` instead of `EVALUATION_DIR`; reuse the *pattern*, you may import this only if convenient (not required).
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/constants.py` — `EVALUATION_DIR = './evaluations/full_evals'` (style reference for where eval outputs go). Do not edit it; define your own `evaluations/clustering` default inside `eval.py`.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/tools/evaluate_reid_embeddings.py` — `_compute_retrieval_metrics()` / `_average_precision()` are the **closed-set** retrieval metrics. Read only to AVOID duplicating them; clustering metrics are a different family.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/utility_functions.py` — `load_dataset(subset)` documents where GT `identity`/`orientation` originate (raw `WildlifeReID10k`). **Reference only**; the GT is already in the store as `gt_identity`/`orientation` because **T02's `ingest_wildlife_dataset` populated it at ingestion time** (T02 is the sole owner). You never reload `WildlifeReID10k` yourself.
- `sklearn.metrics` (already installed in `venv`): `homogeneity_completeness_v_measure`, `adjusted_rand_score`, `adjusted_mutual_info_score`. Pairwise BCubed-style precision/recall you compute directly from `y_true`/`y_pred` (no extra dependency).

## Implementation notes

- **Use only the repo venv** (`./venv/bin/python`). Allowed deps: `sklearn` (already present), `numpy`, `pandas`, stdlib (`json`, `argparse`, `pathlib`, `collections`, `datetime`). No new pip installs.
- **Flank convention (critical for lynx/leopard).** Default `flank_aware=False`: `gt_label = gt_identity`. With `--flank-aware`: `gt_label = f"{gt_identity}|{flank_bucket}"`, using the SAME `{left, right, other}` bucketing as T05 (NOT raw `orientation`). The bucket function is: `left`→`left`, `right`→`right`, and everything else (`front`, `back`, `down`, `unknown`, `''`, NULL) → `other`. This keeps the two spot-bearing flanks as separate "individuals to discover" (left/right spot patterns differ and T05 may legitimately put them in different clusters) while POOLING the non-flank orientations into one `other` bucket so a single animal is not split across up to 5 buckets. Because the bucketing matches T05 exactly, the metric is directly comparable to the clusters T05 produced. **Always record `flank_aware` in the report and state it in the plain-language summary** so the number is never ambiguous. `unknown`/NULL/`''` orientation deterministically lands in the `other` bucket under flank-aware mode (do not crash, do not fall back to bare `gt_identity`).
- **`pct_photos_correctly_grouped` definition (write it in the docstring AND the contract doc-comment so it's reproducible):** for each predicted cluster, its "majority true label" is the most common `gt_label` among its photos; a photo is "correctly grouped" iff its `gt_label` equals its cluster's majority true label. The percentage is `(#correctly grouped photos) / (#evaluated photos) * 100`. This is the intuitive "what fraction of photos landed in the right pile" and is exactly homogeneity-flavored but in plain counts. (Noise photos `cluster_id == -1`: when `include_noise=True`, treat the entire noise bucket as its own pseudo-cluster for this calc; document this.)
- **Merge errors:** count predicted clusters whose photos span >1 distinct `gt_label`. For each such cluster, record the set of true labels merged → `merged_individual_groups`. **Split errors:** count distinct `gt_label`s whose photos appear in >1 distinct `cluster_id` (excluding noise from the "appears in" set unless `include_noise=True`); list them → `split_individuals`.
- **Candidate-new precision/recall:** a GT "singleton" = a `gt_label` with exactly 1 photo in the evaluated set. Precision = fraction of `is_candidate_new==1` rows whose `gt_label` is a GT singleton. Recall = fraction of GT-singleton photos that were flagged `is_candidate_new==1`. If there are zero flagged or zero singletons, set the respective metric to `None` (not 0/NaN) and say so.
- **Pairwise (BCubed-style) P/R/F1:** over all unordered pairs of evaluated photos, a pair is "predicted same" if same `cluster_id` and "true same" if same `gt_label`. Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = harmonic mean. Compute efficiently from per-cluster and per-label group sizes (do NOT materialize all O(n²) pairs explicitly for large n; use the standard sum-over-group-sizes contingency-table formulation). Test it against a tiny brute-force O(n²) implementation in `tests/`.
- **Per-individual table row:** `{gt_label, n_photos, n_clusters (distinct cluster_id the photos fell into), dominant_cluster (the cluster holding the most of its photos), is_split (n_clusters>1)}`. **Per-cluster table row:** `{cluster_id, n_photos, dominant_gt_label, purity (largest gt_label share in cluster), n_true_individuals (distinct gt_labels in cluster), is_merge (n_true_individuals>1)}`.
- **Empty / degenerate sets:** if no rows have both `gt_identity` and `cluster_id`, raise `ValueError("No evaluated rows: need both gt_identity and cluster_id for dataset=<name>")`. If only one cluster or one true label exists, sklearn metrics still return finite values — pass them through; do not special-case beyond what sklearn does.
- **JSON serialization:** all numpy scalars must be cast to Python `float`/`int` before `json.dump` (use a small helper or `float(...)`/`int(...)`). Round floats to 4 decimals in the JSON for readability but keep full precision available in the dataclass if you like.
- **stdout summary** (`plain_language_summary`) must read like the pitch sentence, e.g.:
  ```
  Dataset: LeopardID2022 (flank-aware: yes)
  Photos evaluated: 1431  |  Known individuals: 430  |  Found individuals: 412
  Correctly grouped: 96.1% of photos
  Merge mistakes: 3 (different cats grouped together)
  Split mistakes: 5 (one cat spread across multiple groups)
  Candidate-new flags: 18 (precision 0.83, recall 0.71)
  Standard metrics: V-measure 0.94 | ARI 0.91 | AMI 0.93
  ```
- **No GUI, no images opened.** You operate purely on labels from the store. Crop paths are passed through into the breakdown tables only so T06 can later show examples.
- Add a one-line note to `STATUS_BOARD.md` marking T07 deliverables (create the file if it doesn't exist; do not author other tickets' status lines).

## Acceptance criteria

- [ ] `reid_demo/eval.py` and `tests/test_eval.py` exist; no existing repo file is modified except an additive line in `STATUS_BOARD.md`.
- [ ] `python -c "from reid_demo.eval import ClusteringReport, load_eval_frame, build_label_arrays, standard_metrics, plain_language_metrics, evaluate_clustering, save_report"` succeeds (every contracted name importable).
- [ ] On a **synthetic in-memory store** seeded so that predicted clusters EXACTLY equal GT identities, `evaluate_clustering(...)` returns `v_measure == 1.0`, `adjusted_rand_index == 1.0`, `pct_photos_correctly_grouped == 100.0`, `n_merge_errors == 0`, `n_split_errors == 0`, and `n_found_clusters == n_true_individuals`.
- [ ] On a synthetic store with one engineered **merge** (two distinct GT identities forced into one cluster) and one engineered **split** (one GT identity scattered across two clusters), `n_merge_errors >= 1`, `n_split_errors >= 1`, `merged_individual_groups` contains the merged pair, and `split_individuals` contains the split identity.
- [ ] `standard_metrics` keys are exactly `{homogeneity, completeness, v_measure, adjusted_rand_index, adjusted_mutual_info, pairwise_precision, pairwise_recall, pairwise_f1}` and values are in `[0,1]` (ARI/AMI may be slightly negative — that is allowed and must not crash).
- [ ] Pairwise P/R/F1 from the efficient formula matches a brute-force O(n²) reference on a random synthetic labeling to within 1e-9 (tested in `tests/test_eval.py`).
- [ ] `--flank-aware` changes `gt_label` to `gt_identity|flank_bucket` using the `{left, right, other}` convention: a synthetic case where the same `gt_identity` has `left` and `right` photos placed in two clusters scores as **0 splits** under `--flank-aware` but **1 split** without it. A second case where the same `gt_identity` has `front` and `down` photos (both pooled into `other`) placed in one cluster scores as **0 splits** under `--flank-aware` (they share the `other` bucket, not split into separate buckets).
- [ ] Candidate-new precision/recall computed correctly on a seeded case; returns `None` (not 0/NaN) when no candidate-new flags or no GT singletons exist.
- [ ] `evaluate_clustering` raises a clear `ValueError` when no rows have both `gt_identity` and `cluster_id`.
- [ ] `save_report(...)` writes `evaluations/clustering/<dataset>_<tag>.json` whose top-level keys equal the `ClusteringReport` field names; reloading with `json.load` exposes `pct_photos_correctly_grouped`, `n_found_clusters`, `per_individual`, `per_cluster`.
- [ ] `python -m reid_demo.eval --dataset <seeded_dataset> --db <temp.sqlite>` prints the plain-language summary, writes the JSON, exits 0; on an absent/empty dataset it exits non-zero with a clear message.
- [ ] `python -m pytest tests/test_eval.py -q` passes under the repo venv with no new pip installs.

## How to verify

```bash
cd /home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project
source venv/bin/activate   # or use ./venv/bin/python directly

# 1. Contract surface imports
python -c "from reid_demo.eval import ClusteringReport, load_eval_frame, build_label_arrays, standard_metrics, plain_language_metrics, evaluate_clustering, save_report; print('OK')"

# 2. Seed a synthetic store via the T01 API, then evaluate (perfect clustering => v_measure 1.0)
python - <<'PY'
import os, tempfile
from reid_demo.store import connect, DetectionRecord, upsert_records, make_record_id
from reid_demo.eval import evaluate_clustering, save_report

db = os.path.join(tempfile.mkdtemp(), "eval.sqlite")
conn = connect(db)
recs = []
# 3 individuals, 3 photos each; cluster_id == perfect mapping
for ind in range(3):
    for j in range(3):
        rid = make_record_id(f"img_{ind}_{j}", 1)
        recs.append(DetectionRecord(
            record_id=rid, source_image=f"x/{ind}_{j}.jpg", source_stem=f"img_{ind}_{j}",
            det_index=1, crop_path=f"crops/{rid}.jpg",
            bbox_x=0.1,bbox_y=0.1,bbox_w=0.2,bbox_h=0.2,
            dataset="SYN", cluster_id=ind, cluster_conf=0.9, is_candidate_new=0,
            gt_identity=f"leopard_{ind}", orientation="left"))
upsert_records(conn, recs)
rep = evaluate_clustering(conn, "SYN", tag="perfect")
print(rep.plain_language_summary())
assert rep.v_measure == 1.0 and rep.pct_photos_correctly_grouped == 100.0
assert rep.n_merge_errors == 0 and rep.n_split_errors == 0
assert rep.n_found_clusters == rep.n_true_individuals == 3
p = save_report(rep)
print("wrote", p)
import json; d = json.load(open(p))
assert d["pct_photos_correctly_grouped"] == 100.0
assert "per_individual" in d and "per_cluster" in d
print("PERFECT-CASE OK")
PY

# 3. Merge + split engineered case
python - <<'PY'
import os, tempfile
from reid_demo.store import connect, DetectionRecord, upsert_records, make_record_id
from reid_demo.eval import evaluate_clustering
db = os.path.join(tempfile.mkdtemp(), "ms.sqlite"); conn = connect(db)
recs=[]
def mk(stem, cid, gid, orient="left"):
    rid = make_record_id(stem,1)
    return DetectionRecord(record_id=rid, source_image="x", source_stem=stem, det_index=1,
        crop_path="c", bbox_x=0,bbox_y=0,bbox_w=0.1,bbox_h=0.1, dataset="MS",
        cluster_id=cid, cluster_conf=0.5, is_candidate_new=0, gt_identity=gid, orientation=orient)
# MERGE: A and B both in cluster 0
recs += [mk("a1",0,"A"), mk("a2",0,"A"), mk("b1",0,"B")]
# SPLIT: C scattered across clusters 1 and 2
recs += [mk("c1",1,"C"), mk("c2",2,"C")]
upsert_records(conn, recs)
rep = evaluate_clustering(conn, "MS", tag="ms")
print(rep.plain_language_summary())
assert rep.n_merge_errors >= 1 and rep.n_split_errors >= 1
assert any(set(g) == {"A","B"} for g in rep.merged_individual_groups), rep.merged_individual_groups
assert "C" in rep.split_individuals, rep.split_individuals
print("MERGE/SPLIT OK")
PY

# 4. Flank-aware toggle
python - <<'PY'
import os, tempfile
from reid_demo.store import connect, DetectionRecord, upsert_records, make_record_id
from reid_demo.eval import evaluate_clustering
db=os.path.join(tempfile.mkdtemp(),"fl.sqlite"); conn=connect(db)
def mk(stem,cid,orient):
    rid=make_record_id(stem,1)
    return DetectionRecord(record_id=rid,source_image="x",source_stem=stem,det_index=1,crop_path="c",
        bbox_x=0,bbox_y=0,bbox_w=0.1,bbox_h=0.1,dataset="FL",cluster_id=cid,cluster_conf=0.5,
        is_candidate_new=0,gt_identity="LEO",orientation=orient)
# same individual LEO, left flank in cluster 0, right flank in cluster 1
upsert_records(conn,[mk("l1",0,"left"),mk("l2",0,"left"),mk("r1",1,"right"),mk("r2",1,"right")])
naive = evaluate_clustering(conn,"FL",tag="naive",flank_aware=False)
flank = evaluate_clustering(conn,"FL",tag="flank",flank_aware=True)
assert naive.n_split_errors == 1, naive.n_split_errors
assert flank.n_split_errors == 0, flank.n_split_errors
print("FLANK-AWARE OK: naive splits=%d, flank splits=%d" % (naive.n_split_errors, flank.n_split_errors))
PY

# 5. CLI on the seeded DB
python -m reid_demo.eval --dataset SYN --db "$(python -c "import tempfile,os;print(os.path.join(tempfile.gettempdir(),'nope.sqlite'))")" ; echo "absent-exit=$?"   # expect non-zero (empty)

# 6. Tests
python -m pytest tests/test_eval.py -q
```

## Open questions

1. **GT provenance — RESOLVED: T02 owns it.** `gt_identity`/`orientation` are populated into the store by **T02's `ingest_wildlife_dataset` adapter at ingestion time** (T02 is the sole owner), and T02 is upstream of embed/cluster/eval, so the GT is guaranteed present before T07 runs. T07 reads it from the store only and ships **no** backfill/`--backfill-gt` path — `utility_functions.load_dataset` is a documentation reference, never a runtime dependency. (No action needed; the prior "T05/T10 populate it / optional fallback" design is dropped.)
2. **Noise handling in the headline number.** Default `include_noise=True` treats `cluster_id == -1` as a single pseudo-cluster for `pct_photos_correctly_grouped`. Some biologists may prefer to exclude noise entirely (report it separately as "X photos the system was unsure about"). We expose both via `--no-noise`; confirm which is the *default headline* for the pitch (recommend: report both, headline = include_noise).
3. **Flank-aware as default?** For the lynx pitch the flank-aware convention is more honest (left/right are genuinely different patterns). Should `--flank-aware` be the DEFAULT for spotted-cat datasets while OFF for tigers (ATRW)? Recommend leaving the flag explicit and letting T10 set it per dataset; flag if T06/T09 want a single fixed convention.
4. **Whether to also emit a confusion-style merge matrix** (true individual × cluster contingency table) for T06 to visualize. Cheap to add to the JSON as an extra block; out of the required contract for now — confirm if T06 wants it.
