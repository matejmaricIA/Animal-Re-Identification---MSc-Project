# DATA CONTRACT — `reid_demo` detection-record store (T01)

> **This file is the shared interface.** Read it instead of the T01 ticket before
> touching any other ticket (T02–T12). It is produced by T01 and every downstream
> ticket relies on EXACTLY the field names, function signatures, and join rules below.
> Binding design decisions (D1–D8) live in [`../STATUS_BOARD.md`](../STATUS_BOARD.md)
> and override any contradicting prose.

One **crop = one row** in table `detections`. The pipeline flows left to right and
each stage writes only its own columns:

```
raw image --(T02 MegaDetector)--> crop + bbox + camera + timestamp + (gt_identity/orientation for labeled data)
          --(T03 SpeciesNet)----> species + species_conf  (+ keep/drop via update_extra('species_kept'))
          --(T04 Embedding)------> embedding_ref (key into a .pkl)
          --(T05 Clustering)-----> cluster_id + cluster_conf + is_candidate_new
          --(T08 HITL review)----> review_status + review_note (+ optional cluster_id reassignment)
          --(T06 catalogue / T07 eval / T09 report / T10 runner): READ-ONLY consumers
```

The SQLite file is the **source of truth**. `record_id` is the join key everywhere;
`embedding_ref` joins into the embeddings pickle dict; `(dataset, cluster_id)` groups
crops into individuals.

---

## Canonical detection record (table `detections`)

28 columns, in this exact order (== `COLUMNS`). `export_records`/`to_dataframe` use the
same order. NOT NULL / DEFAULT constraints exist but the SQLite `type` (PRAGMA
`table_info`) is a bare `TEXT`/`INTEGER`/`REAL` as listed.

| # | column | sqlite type | python type | nullable | written by | meaning / allowed values |
|---|--------|-------------|-------------|----------|------------|--------------------------|
| 1 | `record_id` | TEXT PRIMARY KEY | str | no | T02 | `make_record_id(stem, det_index)` → `f"{source_stem}__crop{det_index}"`. Stable across re-runs. Whole-frame B-track record → `det_index == 1`. |
| 2 | `source_image` | TEXT | str | no | T02 | Path to the original full frame (relative to repo root or absolute). |
| 3 | `source_stem` | TEXT | str | no | T02 | Filename stem of the source image, no extension (e.g. `IMG_0066`). |
| 4 | `det_index` | INTEGER | int | no | T02 | **BINDING (D3):** 1-based index over **KEPT ANIMAL detections in MegaDetector source-file order**. Matches the `crop1`-based crop filenames. Whole-frame B-track record → `1`. T03's join does NOT rely on positional `det_index`. |
| 5 | `crop_path` | TEXT | str | no | T02 | Path to the cropped image file on disk. |
| 6 | `bbox_x` | REAL | float | no | T02 | Normalized left x `[0,1]` (MegaDetector convention; top-left origin). |
| 7 | `bbox_y` | REAL | float | no | T02 | Normalized top y `[0,1]`. |
| 8 | `bbox_w` | REAL | float | no | T02 | Normalized width `[0,1]`. |
| 9 | `bbox_h` | REAL | float | no | T02 | Normalized height `[0,1]`. |
| 10 | `detector_conf` | REAL | float | yes | T02 | MegaDetector detection confidence `[0,1]`. |
| 11 | `camera_id` | TEXT | str | yes | T02 | Camera/trap id (e.g. `unknown_camera`). |
| 12 | `timestamp` | TEXT | str | yes | T02 | ISO-8601 `YYYY-MM-DD HH:MM:SS`. |
| 13 | `species` | TEXT | str | yes | T03 | Human-readable common name (`classes[k].split(';')[-1]`). NULL until T03. |
| 14 | `species_conf` | REAL | float | yes | T03 | Top species score `[0,1]`. |
| 15 | `species_class` | TEXT | str | yes | T03 | Full taxonomy string for the top class (raw `classes[k]`). |
| 16 | `embedding_ref` | TEXT | str | yes | T04 | Key into the embeddings pickle dict (usually `== record_id`). NULL until embedded. **Vector is MODEL-NATIVE dim and NOT L2-normalized (D2).** |
| 17 | `embedding_path` | TEXT | str | yes | T04 | Filesystem path of the `.pkl` containing `embedding_ref`. |
| 18 | `cluster_id` | INTEGER | int | yes | T05 | Discovered individual id. `>=0` = a cluster; `-1` = noise/unassigned (DBSCAN convention). NULL until T05. |
| 19 | `cluster_conf` | REAL | float | yes | T05 | Strength of this crop's assignment, `[0,1]`. |
| 20 | `is_candidate_new` | INTEGER | int (0/1) | yes | T05 | 1 if singleton flagged as candidate NEW individual, else 0. (D5: singletons & DBSCAN noise → `cluster_id=-1` AND `is_candidate_new=1`.) |
| 21 | `orientation` | TEXT | str | yes | T02 | One of `ORIENTATIONS` or NULL. Labeled data populated by T02 from GT (D1); `''`/missing → `'unknown'` at ingest. Drives flank-aware clustering via the `{left,right,other}` 3-bucket policy (D4). |
| 22 | `gt_identity` | TEXT | str | yes | T02 | Ground-truth individual id (LeopardID2022/ATRW). Populated by T02's `ingest_wildlife_dataset` (D1). NULL for field data. Read ONLY by T07. |
| 23 | `review_status` | TEXT | str | no | T08 | One of `REVIEW_STATUSES`; default `unreviewed`. |
| 24 | `review_note` | TEXT | str | yes | T08 | Optional free-text reviewer note. |
| 25 | `dataset` | TEXT | str | yes | T02/T10 | Logical run name (`MedvednicaDS`, `LeopardID2022`, …). Most queries filter on it. |
| 26 | `extra_json` | TEXT | str | yes | any (via `update_extra`) | JSON-object escape hatch. Default `"{}"`. Modules write keys via `update_extra` (NOT raw SQL); e.g. T03 writes `species_kept`. |
| 27 | `created_at` | TEXT | str | no | store | ISO timestamp set on insert (preserved on conflict). |
| 28 | `updated_at` | TEXT | str | no | store | ISO timestamp refreshed on every write. |

**Required indexes:** `dataset`, `cluster_id`, `species`, `review_status`, `(dataset, cluster_id)`.

### `det_index` (BINDING, D3)

1-based index over **KEPT ANIMAL detections in MegaDetector source-file order**:
enumerate the `animal`-category detections that survive T02's keep filter, in the
order they appear in the MegaDetector results file, and number them `1, 2, 3, …`.
This matches the existing `…_crop1_conf…jpg` filenames. Whole-frame B-track records
(D1, `ingest_wildlife_dataset`) have exactly one detection → `det_index == 1`.
**T03's A-track join keys on `(source_stem, bbox)` nearest-match, NOT on positional
`det_index`.**

### Orientation value set & flank-bucket policy (BINDING, D4)

`orientation` ∈ `ORIENTATIONS = {"left","right","front","back","down","unknown"}` or
NULL. **Ingest normalization:** empty-string `''` / missing → `'unknown'` (T02 does
this; the store also normalizes `''`→`'unknown'` and a `None` stays `None`). The store
keeps the **raw canonical value**; the 3-bucket collapse below is applied by **T05
(clustering) and T07 (eval)**, never by the store:

| bucket | orientation values |
|--------|--------------------|
| `left` | `left` |
| `right` | `right` |
| `other` | `front`, `back`, `down`, `unknown`, `''`, NULL |

Rationale: spot-bearing flanks `{left,right}` are individually re-identifiable and
cluster in separate buckets; the rest are not, so they pool into one `other` bucket to
avoid splitting one animal across up to five buckets and inflating the count. T07's
flank-aware GT label MUST use this SAME `{left,right,other}` convention (NOT
`f"{id}|{raw_orientation}"`).

---

## Python dataclass

```python
from dataclasses import dataclass
from typing import Optional

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

## Module-level constants

```python
SCHEMA_VERSION: int = 1
DEFAULT_DB_PATH: str = "data/reid_demo/reid_demo.sqlite"
TABLE_NAME: str = "detections"
COLUMNS: list[str]          # the 28 column names above, in order
REVIEW_STATUSES: set[str]   # {"unreviewed","confirmed","rejected","merged","split"}
ORIENTATIONS: set[str]      # {"left","right","front","back","down","unknown"}
```

## Public functions (exact signatures — import from `reid_demo.store` or `reid_demo`)

```python
def connect(db_path: str = DEFAULT_DB_PATH, *, create: bool = True) -> sqlite3.Connection: ...
def init_db(conn: sqlite3.Connection) -> None: ...
def upsert_record(conn: sqlite3.Connection, record: DetectionRecord) -> None: ...
def upsert_records(conn: sqlite3.Connection, records: Iterable[DetectionRecord]) -> int: ...
def get_record(conn: sqlite3.Connection, record_id: str) -> Optional[DetectionRecord]: ...
def query_records(conn, *, dataset=None, species=None, cluster_id=None,
                  review_status=None, has_embedding=None, orientation=None,
                  where_sql=None, where_params=(), order_by="record_id",
                  limit=None) -> list[DetectionRecord]: ...
def update_species(conn, record_id, species, species_conf, species_class=None) -> None: ...
def update_embedding(conn, record_id, embedding_ref, embedding_path) -> None: ...
def update_cluster(conn, record_id, cluster_id, cluster_conf, is_candidate_new=0) -> None: ...
def update_review(conn, record_id, review_status, review_note=None, cluster_id=None) -> None: ...
def update_extra(conn, record_id, key, value) -> None: ...   # merge one key into extra_json; never raw SQL
def count_by(conn, column, *, dataset=None) -> dict: ...
def make_record_id(source_stem: str, det_index: int) -> str: ...
def export_records(conn, out_path, *, fmt="parquet", dataset=None) -> str: ...
def import_records(conn, in_path: str) -> int: ...
def to_dataframe(conn, *, dataset=None): ...                 # pandas DataFrame in COLUMNS order
```

### Behaviour notes downstream relies on

- `connect()` creates parent dirs, sets `row_factory = sqlite3.Row`, calls `init_db`,
  and verifies both the runtime SQLite version (>= 3.24 for `ON CONFLICT` upserts; D3)
  and the stored `SCHEMA_VERSION` (RuntimeError on mismatch).
- `upsert_record`/`upsert_records` set `created_at` on first insert (preserved on
  conflict) and always refresh `updated_at`. Validate `review_status` ∈
  `REVIEW_STATUSES` and `orientation` ∈ `ORIENTATIONS ∪ {None}` (raise `ValueError`);
  `orientation=''` is normalized to `'unknown'`, not rejected. Bbox ranges are warned
  on, never hard-rejected.
- `update_extra` reads → parses → merges one key → writes back (preserves other keys);
  raises `KeyError` if the record is absent, `ValueError` if `extra_json` is not a JSON
  object. This is the ONLY sanctioned way to stash module extras.
- Stage writers (`update_species`/`_embedding`/`_cluster`/`_review`) raise `KeyError`
  on an unknown `record_id` and refresh `updated_at`.

## Join rules

- `record_id` is the join key everywhere; build it only via `make_record_id`.
- `embedding_ref` is the key into the embeddings pickle dict produced by the T04
  service `reid_demo.embed` (`embed_records` / `embed_crops`; path in `embedding_path`).
  `reid_demo.embed` is the canonical interface downstream tickets (T05, T12) import; it
  is a thin wrapper over the legacy `global_embedding.load_or_build_global_embeddings`.
- `(dataset, cluster_id)` groups crops into individuals.

## Embedding note (D2)

Stored vectors are **MODEL-NATIVE dimension** (1536 for base `megadescriptor-l-384`,
384 only for a `linear_l2` checkpoint) and are stored **RAW — NOT L2-normalized**.
Consumers obtain matrices via the T04 API `reid_demo.embed.get_embedding_matrix(normalize=True)`
and read the dim from the matrix — never hard-code 384, never assume unit norm. This store
only references vectors by `embedding_ref`; it never assumes a dim or norm.

## CLI

```
python -m reid_demo.store --selftest [--db <path>]   # round-trip test; exit 0 ok, non-zero on failure
python -m reid_demo.store --info --db <path>         # schema_version, row count, count_by species & cluster_id
python -m reid_demo.store --export <out.parquet|csv> --db <path> [--dataset NAME]
```

## How each ticket touches the store

1. **T02** — inserts records (`upsert_record(s)`); for labeled data also populates
   `gt_identity`+`orientation` (D1) and writes whole-frame B-track records (`det_index==1`).
2. **T03** — `update_species(...)` + `update_extra(record_id, "species_kept", bool)`;
   joins `animals_classified.json` → records by `(source_stem, bbox)` nearest-match (D3).
3. **T04** — `update_embedding(record_id, embedding_ref, embedding_path)`; vectors stay model-native, un-normalized (D2).
4. **T05** — `update_cluster(record_id, cluster_id, cluster_conf, is_candidate_new)`; runs before T08 (D5).
5. **T08** — `update_review(record_id, review_status, review_note, cluster_id?)`; preserves non-`unreviewed` rows on T05 re-runs unless `--force` (D5).
6. **T06 / T07 / T09 / T10** — READ-ONLY: `query_records`, `count_by`, `to_dataframe`, `get_record` (T07 reads `gt_identity`).
7. **Any module** — module-specific extras go through `update_extra` only (never raw SQL); the `extra_json` blob is the forward-compat escape hatch so we never need a v2 schema mid-demo.
