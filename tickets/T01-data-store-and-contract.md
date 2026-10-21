# T01 — Data store & detection-record contract

> **Status:** 🔵 In review · **Milestone:** M1 (Demo-ready)
> **Depends on:** — · **Blocks:** T02, T03, T04, T05, T06, T07, T08, T09, T10, T11, T12
> **Owner:** Claude Code
>
> **Deliverables landed:** `reid_demo/store.py`, `reid_demo/__init__.py`, `reid_demo/DATA_CONTRACT.md`, `tests/test_store.py` (27 tests, all pass). All acceptance criteria verified via the *How to verify* commands.
>
> Self-contained ticket for the lynx re-identification demo/pilot. See [STATUS_BOARD.md](../STATUS_BOARD.md) for the full plan, the binding design decisions, and the agent working protocol.

---

## Context

We are building a DEMO + PILOT MVP of an open-set, individual-animal re-identification system for Eurasian lynx (closest public analog: spotted big cats — LeopardID2022 leopards, ATRW Amur tigers). The existing repo (`/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project`) does CLOSED-SET re-id (query vs known gallery). The demo pivots the *decision layer only* to OPEN-SET CLUSTERING: take an unlabeled pile of animal crops, discover how many DISTINCT individuals are present (unknown count), and flag singletons that match nothing as candidate NEW individuals. Everything upstream (detect, crop, embed) is reused.

The whole demo is a small constellation of independent modules (T02–T10) handed to separate AI agents. They must all read and write the **same** per-crop "detection record" through a **single shared store and a single Python access module**. **This ticket (T01) is the backbone: it defines that store, the record schema, and the access API. It contains NO detection, classification, embedding, or clustering logic** — it only stores and serves records, and every other ticket cites this contract verbatim.

Pipeline shape the schema must support (one crop = one row, flowing left to right):

```
raw image --(T02 MegaDetector)--> crop + bbox + camera + timestamp
          --(T03 SpeciesNet)----> species + species_conf  (+ keep/drop filter)
          --(T04 Embedding)------> embedding_ref (cache key into a .pkl)
          --(T05 Clustering)-----> cluster_id + cluster_conf
          --(T08 HITL review)----> review_status + review fields
          --(T06 catalogue / T07 eval / T09 report / T10 runner): READ-ONLY consumers
```

Relevant real repo facts this ticket must stay consistent with:
- The Medvednica artifacts already exist at `data/MedvednicaDS/` (`megadetector_results.json`, `animal_detections.json`, `animals_classified.json`, `detections_cleaned.json`, `animal_crops/`, `animal_images/`, `trail_cam_data.csv`).
- MegaDetector bboxes are **normalized** `[x, y, w, h]` in `[0,1]` (top-left origin, width/height as fractions). This schema stores bboxes in that exact normalized form.
- Crop filenames in `animal_crops/` use `{image_stem}_crop{idx}_conf{int_conf_percent}.jpg` (e.g. `02020401_crop1_conf92.jpg`).
- SpeciesNet classification entries live under `detections[i].classifications` with parallel `classes` (full taxonomy strings `uuid;...;common_name`) and `scores` arrays; the human-readable species is `classes[k].split(';')[-1]`.
- Timestamps come from `trail_cam_data.csv` (`filepath,camera,num_detections,datetime,temperature`) and/or the `timestamp` field already stitched into `animals_classified.json`.
- Embeddings are produced by `global_embedding.py` as a `Dict[str, np.ndarray]` pickle keyed by an image/crop id (see `load_or_build_global_embeddings`, `global_embedding_cache_label`). T01 does NOT compute embeddings; it only stores a *reference* (the dict key + the cache file path) so T04 can write and downstream tickets can look the vector up. **Per the T04 embedding contract (D2), stored vectors are MODEL-NATIVE dimension (1536 for base `megadescriptor-l-384`, 384 only for a `linear_l2` checkpoint) and are stored RAW — NOT L2-normalized.** This schema only references those vectors by key; it never assumes a dimension or unit norm.
- LeopardID2022 / ATRW ground-truth metadata (for T07 eval) carries `identity`, `orientation` (left/right/front/back/down — the flank), `species`, `split`, `dataset`; loaded via the raw `WildlifeReID10k` metadata (see `utility_functions.load_dataset`). This schema reserves optional columns to carry `orientation` and `gt_identity` so T05/T07 can be flank-aware and scored. **Per D1, T02 is the SOLE owner of `gt_identity`+`orientation` for labeled datasets: its `ingest_wildlife_dataset` adapter populates both directly from the metadata columns at ingest. T01 only defines/stores these columns; it does not populate them.**

## Objective

Deliver a single self-contained Python module (`reid_demo/store.py`) plus a tiny package init that provides:
1. A **canonical detection-record schema** (exact field names, types, allowed values) — the data contract.
2. A **SQLite-backed store** (stdlib `sqlite3`, no new dependencies) implementing that schema, with an **optional Parquet/CSV export-import fallback** for portability.
3. A **clean Python access API** (create/connect, upsert, bulk upsert, query, update specific stages, update_extra, export, import) that all other tickets call.
4. A **schema/version constant and a written contract doc** so downstream agents can rely on it without reading code.

Out of scope: producing any actual detections, species labels, embeddings, clusters, or catalogues. Those are T02–T10.

## Scope

### In
- Define the `DetectionRecord` schema (a `@dataclass` + the SQLite table DDL) with every field listed under Interface contract.
- Implement `reid_demo/store.py` with the full access API below.
- Choose SQLite (stdlib `sqlite3`) as the primary store; provide Parquet (preferred) / CSV export + import helpers as a portable fallback.
- A `record_id` primary-key convention that is deterministic and collision-free across source images.
- Migrations-light: a `SCHEMA_VERSION` constant stored in a `meta` table; fail loudly on mismatch.
- A short markdown contract doc (`reid_demo/DATA_CONTRACT.md`) downstream tickets read.
- A self-test / smoke CLI (`python -m reid_demo.store --selftest`) that creates a temp DB, round-trips records, and exits non-zero on failure.
- Unit tests under `tests/test_store.py`.

### Out
- MegaDetector ingestion (T02), SpeciesNet (T03), embedding extraction (T04), clustering (T05), catalogue (T06), eval (T07), HITL UI (T08), Medvednica report (T09), end-to-end runner (T10).
- Any modification to existing pipeline files (`main.py`, `global_embedding.py`, etc.). T01 only ADDS the new `reid_demo/` package.
- Network access, model loading, image decoding.

## Inputs

- None at runtime beyond a target DB path (created on demand). The store is the *destination* for T02+ and the *source* for T05+.
- For test fixtures only: the agent MAY read `data/MedvednicaDS/animals_classified.json` and `data/MedvednicaDS/trail_cam_data.csv` to construct realistic dummy records, but the module itself must not hard-depend on those files.

## Outputs

- New package `reid_demo/` containing at least `__init__.py`, `store.py`, `DATA_CONTRACT.md`.
- A SQLite database file (default path `data/reid_demo/reid_demo.sqlite`, parent dir auto-created) holding one row per crop in table `detections`, plus a `meta` table holding `schema_version`.
- Optional exported `detections.parquet` / `detections.csv` produced by the export helper.
- `tests/test_store.py`.

## Interface contract

Downstream tickets depend on EXACTLY the following. Do not rename fields or functions.

### Canonical detection record (table `detections`)

Column order, name, SQLite type, Python type, nullability, meaning:

| # | column | sqlite type | python type | nullable | written by | meaning / allowed values |
|---|--------|-------------|-------------|----------|------------|--------------------------|
| 1 | `record_id` | TEXT PRIMARY KEY | str | no | T02 | Deterministic unique id for this crop. Convention: `f"{source_stem}__crop{det_index}"` (e.g. `02020401__crop1`). MUST be stable across re-runs. For a whole-frame B-track record (D1, `ingest_wildlife_dataset`) `det_index` is `1`. |
| 2 | `source_image` | TEXT | str | no | T02 | Path to the original full frame, relative to repo root or absolute (e.g. `data/MedvednicaDS/animal_images/IMG_0066.JPG`). |
| 3 | `source_stem` | TEXT | str | no | T02 | Filename stem of the source image without extension (e.g. `IMG_0066`). |
| 4 | `det_index` | INTEGER | int | no | T02 | **BINDING (D3):** 1-based index of this detection within the source image, counted over **KEPT ANIMAL detections in MegaDetector source-file order** (i.e. enumerate the detections of category `animal` that survive T02's keep filter, in the order they appear in the MegaDetector results file, and number them `1, 2, 3, …`). Matches the existing `crop1`-based crop filenames. For a whole-frame B-track record (D1) there is exactly one detection, so `det_index == 1`. T03's A-track join does NOT rely on positional `det_index` — it matches by `(source_stem, bbox)` nearest-match (see Implementation notes / D3). |
| 5 | `crop_path` | TEXT | str | no | T02 | Path to the cropped image file written to disk (e.g. `data/reid_demo/crops/IMG_0066__crop1.jpg`). |
| 6 | `bbox_x` | REAL | float | no | T02 | Normalized left x in `[0,1]` (MegaDetector convention). |
| 7 | `bbox_y` | REAL | float | no | T02 | Normalized top y in `[0,1]`. |
| 8 | `bbox_w` | REAL | float | no | T02 | Normalized width in `[0,1]`. |
| 9 | `bbox_h` | REAL | float | no | T02 | Normalized height in `[0,1]`. |
| 10 | `detector_conf` | REAL | float | yes | T02 | MegaDetector detection confidence `[0,1]`. |
| 11 | `camera_id` | TEXT | str | yes | T02 | Camera/trap identifier (e.g. `unknown_camera` or `Camera 1`). |
| 12 | `timestamp` | TEXT | str | yes | T02 | ISO-8601 string `YYYY-MM-DD HH:MM:SS` (matches Medvednica `timestamp`). |
| 13 | `species` | TEXT | str | yes | T03 | Human-readable common name (`classes[k].split(';')[-1]`, e.g. `eurasian lynx`). NULL until T03 runs. |
| 14 | `species_conf` | REAL | float | yes | T03 | Top species score `[0,1]`. |
| 15 | `species_class` | TEXT | str | yes | T03 | Full taxonomy string for the top class (the raw `classes[k]`), kept for traceability. |
| 16 | `embedding_ref` | TEXT | str | yes | T04 | Key into the embeddings pickle dict (the id under which T04 stored this crop's vector). Usually equals `record_id`. NULL until embedded. The referenced vector is MODEL-NATIVE dim and NOT L2-normalized (D2); consumers obtain matrices via the T04 API `get_embedding_matrix(normalize=True)` and read the dim from the matrix. |
| 17 | `embedding_path` | TEXT | str | yes | T04 | Filesystem path of the `.pkl` produced by `global_embedding.load_or_build_global_embeddings` that contains `embedding_ref`. |
| 18 | `cluster_id` | INTEGER | int | yes | T05 | Discovered individual id. `>=0` = an individual cluster; `-1` = noise/unassigned (sklearn DBSCAN convention). NULL until T05 runs. |
| 19 | `cluster_conf` | REAL | float | yes | T05 | Confidence/strength of this crop's assignment to `cluster_id`, `[0,1]`. |
| 20 | `is_candidate_new` | INTEGER | int (0/1) | yes | T05 | 1 if this crop is a singleton flagged as a candidate NEW individual, else 0. |
| 21 | `orientation` | TEXT | str | yes | T02 | Flank/orientation if known: one of the canonical values `left`,`right`,`front`,`back`,`down`,`unknown` (see ORIENTATIONS and the orientation policy below). For labeled datasets (LeopardID2022/ATRW) populated by T02 (D1) from GT metadata; **empty-string `''` / missing maps to `'unknown'` at ingest.** For unlabeled field data NULL/`unknown` until a flank estimator exists. Drives flank-aware clustering in T05 via the 3-bucket `{left, right, other}` policy (D4). |
| 22 | `gt_identity` | TEXT | str | yes | T02 | Ground-truth individual id when known (LeopardID2022/ATRW). Populated by T02's `ingest_wildlife_dataset` (D1) from the metadata `identity` column. NULL for unlabeled field data. Read ONLY by T07 (eval). |
| 23 | `review_status` | TEXT | str | no | T08 | One of `unreviewed` (default), `confirmed`, `rejected`, `merged`, `split`. Default `unreviewed`. |
| 24 | `review_note` | TEXT | str | yes | T08 | Optional free-text note from the human reviewer. |
| 25 | `dataset` | TEXT | str | yes | T02/T10 | Logical source/run name (e.g. `MedvednicaDS`, `LeopardID2022`). Lets one DB hold multiple runs; most queries filter on it. |
| 26 | `extra_json` | TEXT | str | yes | any (via `update_extra`) | JSON object escape hatch for module-specific extras WITHOUT schema changes. Default `"{}"`. Modules write keys via `update_extra(conn, record_id, key, value)` (NOT raw SQL); e.g. T03 (D3) writes `species_kept` here. |
| 27 | `created_at` | TEXT | str | no | store | ISO timestamp set on insert. |
| 28 | `updated_at` | TEXT | str | no | store | ISO timestamp set on every upsert/update. |

Required indexes: on `dataset`, `cluster_id`, `species`, `review_status`, `(dataset, cluster_id)`.

### Orientation value set & flank-bucket policy (BINDING, D4)

The `orientation` column holds exactly one of the canonical values in `ORIENTATIONS = {"left","right","front","back","down","unknown"}`, or NULL (unlabeled field data not yet estimated). **Normalization at ingest:** any empty-string `''` or missing orientation maps to `'unknown'` (T02's `ingest_wildlife_dataset` performs this mapping; D1).

Downstream flank-aware clustering (T05) and flank-aware GT scoring (T07) MUST collapse these values into **three buckets** before clustering/scoring:

| bucket | orientation values |
|--------|--------------------|
| `left` | `left` |
| `right` | `right` |
| `other` | `front`, `back`, `down`, `unknown`, `''` (and NULL) |

Rationale: spot-bearing flanks `{left, right}` are individually re-identifiable from spot pattern and cluster in SEPARATE buckets; `{front, back, down, unknown, ''}` are NOT individually re-identifiable from spot pattern, so pooling them into a single `other` bucket avoids inflating the discovered-individual count by splitting one animal across up to five buckets. T07's flank-aware GT label MUST use this SAME `{left, right, other}` convention (NOT `f"{id}|{raw_orientation}"`) so its metric matches T05. T01 stores the raw canonical value (with `''`→`'unknown'` normalization); the 3-bucket mapping is applied by T05/T07, not by the store.

### Python dataclass

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class DetectionRecord:
    record_id: str
    source_image: str
    source_stem: str
    det_index: int
    crop_path: str
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    detector_conf: Optional[float] = None
    camera_id: Optional[str] = None
    timestamp: Optional[str] = None
    species: Optional[str] = None
    species_conf: Optional[float] = None
    species_class: Optional[str] = None
    embedding_ref: Optional[str] = None
    embedding_path: Optional[str] = None
    cluster_id: Optional[int] = None
    cluster_conf: Optional[float] = None
    is_candidate_new: Optional[int] = None
    orientation: Optional[str] = None
    gt_identity: Optional[str] = None
    review_status: str = "unreviewed"
    review_note: Optional[str] = None
    dataset: Optional[str] = None
    extra_json: str = "{}"
    created_at: Optional[str] = None   # set by store on insert
    updated_at: Optional[str] = None   # set by store on write
```

### Module-level constants (must exist, exact names)

```python
SCHEMA_VERSION: int = 1
DEFAULT_DB_PATH: str = "data/reid_demo/reid_demo.sqlite"
TABLE_NAME: str = "detections"
COLUMNS: list[str]          # the 28 column names above, in order
REVIEW_STATUSES: set[str]   # {"unreviewed","confirmed","rejected","merged","split"}
ORIENTATIONS: set[str]      # {"left","right","front","back","down","unknown"}
```

### Public functions (exact signatures — downstream tickets import these)

```python
def connect(db_path: str = DEFAULT_DB_PATH, *, create: bool = True) -> sqlite3.Connection:
    """Open (and if create=True, initialize schema for) the SQLite store.
    Creates parent dirs. Sets row_factory = sqlite3.Row. Verifies SCHEMA_VERSION
    in the meta table; raises RuntimeError on mismatch."""

def init_db(conn: sqlite3.Connection) -> None:
    """Create `detections` + `meta` tables and indexes if absent; stamp SCHEMA_VERSION. Idempotent."""

def upsert_record(conn: sqlite3.Connection, record: DetectionRecord) -> None:
    """Insert or replace one record by record_id. Sets created_at on first insert,
    always refreshes updated_at. Validates review_status in REVIEW_STATUSES and
    orientation in ORIENTATIONS (or None). Commits."""

def upsert_records(conn: sqlite3.Connection, records: Iterable[DetectionRecord]) -> int:
    """Bulk upsert; returns count written. Single transaction."""

def get_record(conn: sqlite3.Connection, record_id: str) -> Optional[DetectionRecord]:
    """Fetch one record or None."""

def query_records(
    conn: sqlite3.Connection,
    *,
    dataset: Optional[str] = None,
    species: Optional[str] = None,
    cluster_id: Optional[int] = None,
    review_status: Optional[str] = None,
    has_embedding: Optional[bool] = None,   # filter embedding_ref IS [NOT] NULL
    orientation: Optional[str] = None,
    where_sql: Optional[str] = None,        # advanced raw WHERE fragment, params via where_params
    where_params: tuple = (),
    order_by: str = "record_id",
    limit: Optional[int] = None,
) -> list[DetectionRecord]:
    """Filtered fetch. All filters AND-combined. Returns dataclasses."""

def update_species(conn, record_id: str, species: str, species_conf: float,
                   species_class: Optional[str] = None) -> None:
    """T03 stage write. Refreshes updated_at. Commits."""

def update_embedding(conn, record_id: str, embedding_ref: str, embedding_path: str) -> None:
    """T04 stage write. Commits."""

def update_cluster(conn, record_id: str, cluster_id: int, cluster_conf: float,
                   is_candidate_new: int = 0) -> None:
    """T05 stage write. Commits."""

def update_review(conn, record_id: str, review_status: str,
                  review_note: Optional[str] = None,
                  cluster_id: Optional[int] = None) -> None:
    """T08 stage write. Validates review_status. Optionally re-assigns cluster_id
    (when a human merges/splits). Commits."""

def update_extra(conn, record_id: str, key: str, value) -> None:
    """Set a single key in this record's `extra_json` blob WITHOUT a schema change.
    Reads the current extra_json (default '{}'), parses it as a dict, sets
    dict[key] = value, re-serialises, and writes it back. Refreshes updated_at.
    Commits. This is the SANCTIONED way for modules to stash module-specific
    extras (e.g. T03 writes `species_kept` via update_extra — no raw SQL).
    `value` must be JSON-serialisable. Raises KeyError/ValueError if record_id
    is absent or extra_json is not a JSON object."""

def count_by(conn, column: str, *, dataset: Optional[str] = None) -> dict:
    """GROUP BY helper: returns {value: count} for the given column
    (e.g. count_by(conn,'species'), count_by(conn,'cluster_id')). Used by T06/T07/T09."""

def make_record_id(source_stem: str, det_index: int) -> str:
    """Return the canonical record_id: f'{source_stem}__crop{det_index}'. Single source of truth."""

# Portable fallback
def export_records(conn, out_path: str, *, fmt: str = "parquet",
                   dataset: Optional[str] = None) -> str:
    """Dump rows to .parquet (if pandas/pyarrow available) or .csv. Returns out_path.
    fmt in {'parquet','csv'}; auto-fallback to csv if pyarrow missing (warn)."""

def import_records(conn, in_path: str) -> int:
    """Load .parquet/.csv produced by export_records back into the store (upsert). Returns count."""

def to_dataframe(conn, *, dataset: Optional[str] = None):
    """Return all (or dataset-filtered) records as a pandas DataFrame with COLUMNS order.
    Convenience for T06/T07/T09. Raises informative error if pandas missing."""
```

### CLI

```
python -m reid_demo.store --selftest [--db <path>]      # round-trip test; exit 0 ok, non-zero on failure
python -m reid_demo.store --info --db <path>            # print schema_version, row count, count_by species & cluster_id
python -m reid_demo.store --export <out.parquet|csv> --db <path> [--dataset NAME]
```

### File-format guarantees for downstream

- SQLite file is the source of truth; `sqlite3.Row` access by column name works for all 28 columns.
- `export_records` CSV/Parquet column set and order == `COLUMNS`.
- `record_id` is the join key everywhere; `embedding_ref` joins into the embeddings pickle dict; `(dataset, cluster_id)` groups crops into individuals.

## Existing code to reuse (real paths)

- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/global_embedding.py` — for the embedding cache convention this schema references: `load_or_build_global_embeddings(image_paths, cache_path, *, model_name="megadescriptor-l-384", checkpoint_path=None) -> dict` (a `Dict[str, np.ndarray]` pickle) and `global_embedding_cache_label(model_name, checkpoint_path)`. T01 does not call these; it stores `embedding_ref`/`embedding_path` that point at their output. Read it only to keep field semantics aligned.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/constants.py` — follow its style of module-level path constants (`ROOT_DIR = os.path.dirname(os.path.abspath(__file__))`, `WILD_DATASET_PATH`, `DB_PATH = './data/{}/db/'`). Place new constants in `reid_demo/store.py`; do NOT edit `constants.py`.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/data/MedvednicaDS/animals_classified.json` — the real SpeciesNet output shape that fields 13–15 must be able to represent (`detections[i].classifications.classes/scores`, `bbox` normalized, top-level `timestamp`). Use as a fixture reference only.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/data/MedvednicaDS/trail_cam_data.csv` — header `filepath,camera,num_detections,datetime,temperature`; source of `camera_id` (`camera`) and `timestamp` (`datetime`). Fixture reference only.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/data/MedvednicaDS/animal_crops/` — real crop filenames (`02020401_crop1_conf92.jpg`) confirming the `source_stem` + `det_index` decomposition.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/utility_functions.py` — `load_dataset()` shows where `orientation`/`identity` GT comes from (raw `WildlifeReID10k` metadata) for the `orientation`/`gt_identity` columns. Reference only.

## Implementation notes

- Use only the standard library for the core store: `sqlite3`, `dataclasses`, `json`, `datetime`, `pathlib`, `os`. `pandas`/`pyarrow` are OPTIONAL and only for `to_dataframe`/Parquet export — import them lazily inside the function and degrade gracefully (CSV) when absent.
- Place the package at repo root: `reid_demo/__init__.py` re-exports the public names (`from .store import DetectionRecord, connect, upsert_record, ...`). Downstream tickets will do `from reid_demo.store import ...` or `from reid_demo import ...`.
- `connect()` must `PRAGMA foreign_keys`/`journal_mode` as you see fit, set `conn.row_factory = sqlite3.Row`, create parent directories, call `init_db`, and check the stored `schema_version` against `SCHEMA_VERSION`, raising `RuntimeError` on mismatch (the demo never needs auto-migration; a clear error is enough).
- **UPSERT portability (D3):** upserts use `INSERT ... ON CONFLICT(record_id) DO UPDATE SET ...`, which requires SQLite **3.24+**. `connect()` MUST guard this: check `sqlite3.sqlite_version_info >= (3, 24, 0)` (equivalently `sqlite3.sqlite_version >= "3.24"`) and raise a clear `RuntimeError` naming the detected version if the runtime SQLite is too old — OR implement a `SELECT`-then-`INSERT`/`UPDATE` fallback path so the store still works. (Python 3.12 stdlib bundles ≥3.37, so the check passes in the target env; the guard exists to fail loudly elsewhere rather than emit a cryptic syntax error.) Preserve the original `created_at` on conflict; always set `updated_at = now`.
- Timestamps written by the store use `datetime.now().isoformat(timespec="seconds")`.
- Validation: reject `review_status` not in `REVIEW_STATUSES` and `orientation` not in `ORIENTATIONS ∪ {None}` with a `ValueError`. **Orientation normalization (D4):** before validation, normalize an empty-string `''` orientation to `'unknown'` (do not reject `''`); a `None` orientation stays `None`. The canonical value set is `ORIENTATIONS = {"left","right","front","back","down","unknown"}`. The store stores the raw canonical value; the `{left, right, other}` 3-bucket collapse is T05/T07's responsibility, not the store's. Do NOT validate bbox ranges hard (warn at most) — detectors occasionally emit slightly out-of-range values and we don't want ingestion to crash.
- `make_record_id` is the single id source so T02 and T08 agree; T01 just concatenates `source_stem` and `det_index`. **The `det_index` convention is BINDING-defined in this ticket (D3): 1-based over KEPT ANIMAL detections in MegaDetector source-file order** (whole-frame B-track records use `det_index == 1`). T02 produces indices following exactly this rule; T01 documents it in `DATA_CONTRACT.md` so all tickets agree. T03's A-track join keys on `(source_stem, bbox)` nearest-match, NOT on positional `det_index` (see D3).
- Keep `extra_json` as the forward-compat escape hatch so we never need a v2 schema mid-demo; modules mutate it ONLY through `update_extra` (e.g. T03 stores `species_kept`), never via raw SQL.
- Write `reid_demo/DATA_CONTRACT.md` containing: the 28-column table (with the BINDING `det_index` definition and the `orientation` canonical value set + `{left, right, other}` 3-bucket policy and `''`→`'unknown'` ingest normalization), the dataclass, the public function signatures (INCLUDING `update_extra`), the `record_id`/`embedding_ref`/`(dataset,cluster_id)` join rules, a note that stored embeddings are MODEL-NATIVE dim and NOT pre-normalized (D2; obtain matrices via T04 `get_embedding_matrix(normalize=True)`), and a 7-line "how each ticket touches the store" map (T02 inserts incl. gt_identity/orientation for labeled data + populates whole-frame B-track records; T03 update_species + update_extra('species_kept') and joins by (source_stem,bbox) nearest-match; T04 update_embedding; T05 update_cluster; T08 update_review; T06/T07/T09/T10 read-only). This doc is what other agents read instead of this ticket.
- Add a one-line note to `STATUS_BOARD.md` (create it if absent) marking T01 deliverables; do not author other tickets' status.

## Acceptance criteria

- [x] `reid_demo/store.py`, `reid_demo/__init__.py`, `reid_demo/DATA_CONTRACT.md`, and `tests/test_store.py` exist; no existing repo file is modified except an additive line in `STATUS_BOARD.md`.
- [x] `python -c "from reid_demo.store import DetectionRecord, connect, init_db, upsert_record, upsert_records, get_record, query_records, update_species, update_embedding, update_cluster, update_review, update_extra, count_by, make_record_id, export_records, import_records, to_dataframe, SCHEMA_VERSION, DEFAULT_DB_PATH, TABLE_NAME, COLUMNS, REVIEW_STATUSES, ORIENTATIONS"` succeeds (every contracted name importable).
- [x] `COLUMNS` is exactly the 28 names in the specified order; the created `detections` table has exactly those columns with the specified SQLite types (verifiable via `PRAGMA table_info`).
- [x] `make_record_id("IMG_0066", 1) == "IMG_0066__crop1"`.
- [x] Round-trip: upserting a `DetectionRecord` then `get_record` returns an equal record (modulo `created_at`/`updated_at` being populated).
- [x] `upsert_record` twice with the same `record_id` updates in place (row count stays 1) and preserves `created_at` while advancing `updated_at`.
- [x] Stage updates work independently: after `update_species`, `update_embedding`, `update_cluster`, `update_review` on one record, all fields persist and `review_status` is validated (an invalid status raises `ValueError`).
- [x] `update_extra(conn, record_id, "species_kept", True)` then `get_record` shows `species_kept` inside the parsed `extra_json`; a second `update_extra` with a different key preserves the first (merge, not overwrite) and advances `updated_at`.
- [x] `orientation=""` upserts as `"unknown"` (normalized, not rejected); `orientation` not in `ORIENTATIONS ∪ {None}` raises `ValueError`.
- [x] `connect()` enforces the SQLite-version guard: on `sqlite3.sqlite_version_info < (3,24,0)` it raises a clear `RuntimeError` naming the version (or, if the SELECT-then-INSERT/UPDATE fallback is implemented, upserts still succeed); on the target env (`sqlite3.sqlite_version_info >= (3,24,0)`) `connect()` succeeds.
- [x] `query_records(dataset=..., species=..., has_embedding=True, cluster_id=...)` filters correctly on a seeded multi-record DB; `count_by(conn,'species')` and `count_by(conn,'cluster_id')` return correct group counts.
- [x] `export_records` to CSV produces a file whose header equals `COLUMNS`; `import_records` of that file into a fresh DB reproduces the rows (Parquet path tested when pyarrow is available, else CSV fallback path exercised and warns).
- [x] Opening a DB whose stored `schema_version` differs from `SCHEMA_VERSION` raises `RuntimeError`.
- [x] `python -m reid_demo.store --selftest` exits 0; `--info` prints schema_version, row count, and species/cluster group counts.
- [x] `tests/test_store.py` passes under the repo venv.

## How to verify

```bash
cd /home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project
source venv/bin/activate    # or the repo's active env

# 1. Imports / contract surface
python -c "from reid_demo.store import *; print('SCHEMA_VERSION', SCHEMA_VERSION); print(len(COLUMNS), 'columns'); assert len(COLUMNS)==28; print('OK')"

# 1b. SQLite UPSERT-portability guard (D3): runtime SQLite must be >= 3.24 for ON CONFLICT upserts.
python -c "import sqlite3; print('sqlite', sqlite3.sqlite_version); assert sqlite3.sqlite_version_info >= (3,24,0), 'too old for ON CONFLICT upsert'; print('OK')"
# and connect() itself must enforce this (clear RuntimeError if too old, or use its SELECT-then-INSERT/UPDATE fallback)
python -c "from reid_demo.store import connect; connect('/tmp/reid_ver.sqlite'); print('connect version-guard OK')"

# 2. Self-test round-trip (temp DB)
python -m reid_demo.store --selftest --db /tmp/reid_selftest.sqlite ; echo "exit=$?"

# 3. Table shape matches contract
python - <<'PY'
import sqlite3
from reid_demo.store import connect, COLUMNS, TABLE_NAME
conn = connect("/tmp/reid_shape.sqlite")
cols = [r[1] for r in conn.execute(f"PRAGMA table_info({TABLE_NAME})").fetchall()]
assert cols == COLUMNS, (cols, COLUMNS)
print("schema columns match contract:", cols)
PY

# 4. Stage writes + queries + export round-trip
python - <<'PY'
from reid_demo.store import *
conn = connect("/tmp/reid_rt.sqlite")
r = DetectionRecord(record_id=make_record_id("IMG_0066",1),
    source_image="data/MedvednicaDS/animal_images/IMG_0066.JPG",
    source_stem="IMG_0066", det_index=1,
    crop_path="data/reid_demo/crops/IMG_0066__crop1.jpg",
    bbox_x=0.49,bbox_y=0.04,bbox_w=0.05,bbox_h=0.17,
    detector_conf=0.78, camera_id="unknown_camera",
    timestamp="2025-06-02 04:27:51", dataset="MedvednicaDS")
upsert_record(conn, r)
update_species(conn, r.record_id, "eurasian lynx", 0.91, "uuid;...;eurasian lynx")
update_embedding(conn, r.record_id, r.record_id, "data/reid_demo/emb.pkl")
update_cluster(conn, r.record_id, 3, 0.88, is_candidate_new=0)
update_review(conn, r.record_id, "confirmed", "looks like the same cat")
update_extra(conn, r.record_id, "species_kept", True)
update_extra(conn, r.record_id, "note", "x")   # second key must merge, not overwrite
got = get_record(conn, r.record_id)
assert got.species=="eurasian lynx" and got.cluster_id==3 and got.review_status=="confirmed"
import json as _json
_ex = _json.loads(got.extra_json)
assert _ex.get("species_kept") is True and _ex.get("note")=="x", _ex
print("stage writes + update_extra OK ->", got.species, got.cluster_id, got.review_status, _ex)
# orientation '' normalizes to 'unknown' at ingest (D4)
r2 = DetectionRecord(record_id=make_record_id("IMG_0066",2),
    source_image="x", source_stem="IMG_0066", det_index=2, crop_path="x",
    bbox_x=0.0,bbox_y=0.0,bbox_w=1.0,bbox_h=1.0, orientation="", dataset="MedvednicaDS")
upsert_record(conn, r2)
assert get_record(conn, r2.record_id).orientation == "unknown"
print("orientation '' -> 'unknown' OK")
print("count_by species:", count_by(conn,"species"))
p = export_records(conn, "/tmp/reid_rt.csv", fmt="csv")
import csv
with open(p) as f: header = next(csv.reader(f))
assert header == COLUMNS, header
print("export header matches COLUMNS")
conn2 = connect("/tmp/reid_rt2.sqlite")
n = import_records(conn2, p); assert n==2; print("import round-trip rows:", n)
PY

# 5. Tests
python -m pytest tests/test_store.py -q
```

## Open questions

1. RESOLVED (D3). `det_index` is BINDING-defined: 1-based over KEPT ANIMAL detections in MegaDetector source-file order (whole-frame B-track records use `det_index == 1`). This matches the `crop1`-based crop filenames. Documented in `DATA_CONTRACT.md`. T03's A-track join keys on `(source_stem, bbox)` nearest-match, not on positional `det_index`.
2. Multi-run isolation: one shared SQLite with a `dataset` column (chosen here) vs one DB file per run. We chose a single DB + `dataset` filter so T10 can hold Medvednica and LeopardID2022/ATRW side by side; flag if any ticket prefers separate files.
3. RESOLVED (D4). The flank/orientation policy is fixed: spot-bearing flanks `{left, right}` cluster in separate buckets; `{front, back, down, unknown, ''}` (and NULL) pool into a single `other` bucket. `''` normalizes to `'unknown'` at ingest. T05 and T07 both use this `{left, right, other}` convention; T01 stores the raw canonical value only.
4. Whether `gt_identity`/`orientation` should instead live in a side table joined by `record_id` rather than columns on `detections` — kept inline for demo simplicity (T02 populates them at ingest per D1); revisit only if eval (T07) needs many GT attributes.
