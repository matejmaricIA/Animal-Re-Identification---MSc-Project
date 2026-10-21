# T02 — Ingestion + MegaDetector adapter

> **Status:** 🔵 In review · **Milestone:** M1 (Demo-ready)
> **Depends on:** T01 · **Blocks:** T03, T04, T06, T08, T10, T11
> **Owner:** _unassigned_
>
> Self-contained ticket for the lynx re-identification demo/pilot. See [STATUS_BOARD.md](../STATUS_BOARD.md) for the full plan, the binding design decisions, and the agent working protocol.

---

## Context

We are building a DEMO + PILOT MVP of an **open-set individual-animal re-identification** system for Eurasian lynx (public analogs: spotted big cats — LeopardID2022 leopards, ATRW Amur tigers). The existing repo at `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project` already does closed-set re-id; the demo pivots only the *decision layer* to open-set clustering. The full demo is a constellation of independent modules (T01–T10), each handed to a separate agent. They all communicate through **one shared store and one Python access module defined by T01** (`reid_demo/store.py`). You must read and write that store; do not invent your own schema.

**This ticket (T02) is the front door of the pipeline.** It takes either (a) a folder of raw camera-trap images, or (b) an already-computed MegaDetector results JSON, runs/loads MegaDetector, **drops empty frames, persons, and vehicles**, crops each surviving *animal* bounding box to a JPG on disk, and writes **one T01 `DetectionRecord` per crop** into the store. Everything downstream (T03 species, T04 embeddings, T05 clustering) reads those rows. For the unlabeled camera-trap (A-track) path, T02 owns these fields: `record_id, source_image, source_stem, det_index, crop_path, bbox_x/y/w/h, detector_conf, camera_id, timestamp, dataset` (and sets `orientation="unknown"`). It must NOT touch `species*`, `embedding*`, `cluster*`, or `review_*`.

**T02 is also the SOLE owner of ground-truth (`gt_identity`) and `orientation` for labeled re-id datasets (B-track).** A fourth ingestion adapter, `ingest_wildlife_dataset(...)` (defined below), ingests labeled WildlifeReID-10k subsets (LeopardID2022, ATRW) and **populates `gt_identity`, `orientation`, and `species` directly from the dataset metadata** at ingest time. No other ticket (T05/T07/T10) populates `gt_identity` or `orientation`; because T02 is upstream of embed/cluster/eval, owning these here removes the prior dependency cycle. T02 still leaves `embedding*`, `cluster*`, and `review_*` untouched on every path.

The pipeline shape (this ticket is the first arrow):

```
raw image --(T02 MegaDetector)--> crop + bbox + camera + timestamp   <-- YOU ARE HERE
          --(T03 SpeciesNet)----> species + species_conf
          --(T04 Embedding)------> embedding_ref
          --(T05 Clustering)-----> cluster_id + cluster_conf
          --(T06/T07/T08/T09/T10): consumers
```

### Real repo facts you must stay consistent with

- **Medvednica artifacts already exist** at `data/MedvednicaDS/`:
  - `megadetector_results.json` — 8,208 image records. Top-level keys `{"images":[...], "detection_categories":{"1":"animal","2":"person","3":"vehicle"}, "info":{...}}`. Each image record is `{"file": "Camera 1/IMG_0001.JPG", "detections": [...]}`. **Note the `file` field carries a camera-name subfolder** (e.g. `Camera 1/IMG_0001.JPG`); the basename is the actual image filename. Detection record: `{"category":"1"|"2"|"3", "conf":<float 0..1>, "bbox":[x,y,w,h]}` where **bbox is NORMALIZED** `[x_left, y_top, width, height]`, all in `[0,1]`.
  - `animal_images/` — the source frames, named by **basename only** (e.g. `02020401.JPG`, `IMG_0066.JPG`). The camera subfolder from the `file` field is NOT present on disk; files are flat under `animal_images/`.
  - `animal_crops/` — 4,194 already-cropped JPGs named `{image_stem}_crop{idx}_conf{int_conf_percent}.jpg` (e.g. `02020401_crop1_conf92.jpg`, and multi-detection stems like `02050583_crop2_conf92.jpg`). **`idx` is 1-based** here. Some stems have up to 8 crops.
  - `trail_cam_data.csv` — header `filepath,camera,num_detections,datetime,temperature`. `camera` is typically `unknown_camera`; `datetime` is ISO `YYYY-MM-DD HH:MM:SS`; match rows to images by **basename** of `filepath`. Example row: `/kaggle/working/processed_data/animal_images/IMG_0066.JPG,unknown_camera,1,2025-06-02 04:27:51,Not available`.
  - `animal_detections.json` — a flat dict `{ "IMG_0065.JPG": [ {"bbox":[...],"confidence":0.928}, ... ], ... }` (3,333 images, **animal boxes only**, no category). This is an alternate, pre-filtered input you may also accept.
- **Crop math** (the project's existing convention, from `deprecated/seminar_classify_species.py:26-33`): open the source frame, `W,H = img.size`, crop pixel box `(x*W, y*H, (x+w)*W, (y+h)*H)`. Reuse this exactly.
- **Filtering rule** (from `utils/clean_detections.py`): keep a detection only if `category` maps to `"animal"` (id `"1"`) AND `conf >= CONF_THRESHOLD`. The existing cleaner used `conf > 0.5`; default the threshold to **0.5** but make it a CLI flag.
- **MegaDetector is importable in the repo venv only** (`venv/bin/python -c "from megadetector.detection import run_detector"` works; the system python3 does NOT have it). So raw-image detection requires the venv. The Medvednica `megadetector_results.json` already exists, so the **primary demo path is JSON-in, no model needed**.

You do not need any of this conversation beyond what is written here.

## Objective

Deliver a single self-contained module `reid_demo/ingest.py` (plus CLI `python -m reid_demo.ingest`) that:

1. Accepts **one of FOUR** inputs via four adapters: (a) an existing MegaDetector results JSON (the Medvednica format above), (b) the flat `animal_detections.json` format, (c) a folder of raw images on which it runs MegaDetector via the repo venv to produce that JSON, or (d) a **labeled WildlifeReID-10k subset** (LeopardID2022 / ATRW) ingested whole-frame with ground-truth identity/orientation/species via `ingest_wildlife_dataset(...)`.
2. **Filters** detections to animals only (`category=="1"`/`"animal"`) above a confidence threshold; empty frames, persons, and vehicles produce no records.
3. **Crops** each surviving animal bbox to a JPG under a crops output dir, using the project's normalized-bbox→pixel crop math, and **reuses existing crops in `data/MedvednicaDS/animal_crops/` when present** (don't re-crop if a matching file already exists).
4. **Resolves `camera_id` and `timestamp`** from `trail_cam_data.csv` (and/or a `timestamp` already present in the JSON), by image basename.
5. **Writes one T01 `DetectionRecord` per crop** into the store via `reid_demo.store.upsert_records`, populating only T02-owned fields, tagged with a `--dataset` label (default `MedvednicaDS`).
6. Is **idempotent**: re-running over the same input updates rows in place (same `record_id`) and does not duplicate crops or rows.
7. For labeled B-track subsets, creates **one whole-frame detection record per image** (`bbox=(0.0,0.0,1.0,1.0)`, `crop_path` = the original full image path, NO cropping / NO MegaDetector — the whole frame IS the crop, matching how the existing `main.py` pipeline embeds dataset images directly) and **populates `gt_identity`, `orientation`, and `species`** from the metadata columns.

## Scope

### In
- `reid_demo/ingest.py` with the public API and CLI below.
- **Four** input adapters: MegaDetector-results-JSON, flat `animal_detections.json`, raw-image-folder (runs MegaDetector in venv), and **labeled WildlifeReID-10k subset** (`ingest_wildlife_dataset` — whole-frame records with GT identity/orientation/species).
- Ownership of `gt_identity`, `orientation`, and `species` **for labeled B-track datasets only** (populated from dataset metadata at ingest). A-track Medvednica records still leave `species*` to T03 and set `orientation="unknown"`.
- Animal-only + confidence filtering with a tunable threshold (A-track only; the B-track adapter does no detection/filtering).
- Cropping to disk with reuse of pre-existing crops, and a `--no-crop` mode that records crop paths without writing files (for fast dry runs / when crops already exist).
- Camera/timestamp resolution from `trail_cam_data.csv` and/or JSON `timestamp`.
- Writing `DetectionRecord`s through the T01 store API (no direct SQL).
- A summary printout (frames seen, % empty frames removed, detections kept vs dropped by category, crops written/reused, records upserted).
- Unit tests under `tests/test_ingest.py` using a tiny synthetic JSON + a couple of generated dummy images (PIL), no network, no model download.

### Out
- **Species labels on the A-track** (`species`, `species_conf`, `species_class` for Medvednica camera-trap frames) — that's **T03**. Leave NULL on the A-track. (On the B-track, `ingest_wildlife_dataset` DOES set `species` from the metadata — see Field-population contract.)
- **Embeddings** (`embedding_ref`, `embedding_path`) — **T04**. Leave NULL.
- **Clustering** (`cluster_id`, `cluster_conf`, `is_candidate_new`) — **T05**. Leave NULL.
- **Review** fields — **T08**.
- Defining or modifying the schema / store — that is **T01** (`reid_demo/store.py`); you only call it.
- Training or downloading any re-id model. The only model T02 may touch is MegaDetector, and only in the raw-image path.
- Editing existing repo files (`main.py`, `clean_detections.py`, `m2dspeciesnet.py`, etc.). You may *import* or copy small helper logic, but add new code under `reid_demo/` only. (Appending one line to `STATUS_BOARD.md` is allowed.)

## Inputs

Runtime inputs (via CLI / function args):
- A MegaDetector results JSON, default `data/MedvednicaDS/megadetector_results.json`. **OR** a flat `animal_detections.json`. **OR** `--images-dir <folder>` of raw `.JPG/.jpg/.png` to run MegaDetector on.
- The source images directory, default `data/MedvednicaDS/animal_images` (basename resolution; ignore the camera subfolder in the JSON `file` field).
- Optional metadata CSV, default `data/MedvednicaDS/trail_cam_data.csv`.
- Optional pre-existing crops dir to reuse, default `data/MedvednicaDS/animal_crops`.
- Output crops dir for newly written crops, default `data/reid_demo/crops/`.
- Target store DB path, default `reid_demo.store.DEFAULT_DB_PATH` (`data/reid_demo/reid_demo.sqlite`).
- `--dataset` label, default `MedvednicaDS`.
- `--conf-threshold` float, default `0.5`.

For the **B-track `ingest_wildlife_dataset` adapter** (labeled WildlifeReID-10k subsets):
- `subset` — the subset name, e.g. `LeopardID2022` or `ATRW`. Loaded via the project's existing `utility_functions.load_dataset(subset)` and/or its `metadata.csv`, which yields per-image rows with at least the columns `image_id`/`path` (relative image path), `identity`, `orientation`, and `species`.
- `max_identities` (int or None) — caps the number of **distinct identities** ingested (used by T10 `--smoke`). When set, ingest images for the first `max_identities` distinct identity values (deterministic order) and skip the rest.
- `limit` (int or None) — caps the number of source images ingested (debug), applied after the `max_identities` filter.
- The subset's image root directory (resolved by `utility_functions.load_dataset` / the metadata `path` column) — used as the `crop_path` source for whole-frame records (no cropping).

From T01 (consumed, do not redefine):
- `reid_demo.store.DetectionRecord` dataclass.
- `reid_demo.store.connect(db_path, *, create=True) -> sqlite3.Connection`.
- `reid_demo.store.upsert_records(conn, records) -> int` and `upsert_record(conn, record)`.
- `reid_demo.store.make_record_id(source_stem, det_index) -> str` (returns `f"{source_stem}__crop{det_index}"`).
- `reid_demo.store.DEFAULT_DB_PATH`, `TABLE_NAME`, `COLUMNS`, `ORIENTATIONS`.

## Outputs

- New file `reid_demo/ingest.py`.
- New file `tests/test_ingest.py`.
- Crop JPGs written under the chosen crops output dir (default `data/reid_demo/crops/`), named **`{source_stem}__crop{det_index}.jpg`** (note: double underscore, matching `record_id`; the `_conf{pct}` suffix from the legacy `animal_crops/` is NOT used for new crops — but legacy crops are still *reused* by matching `{stem}_crop{idx}_*.jpg`).
- One `DetectionRecord` row per kept crop (A-track) or per labeled image (B-track) in the T01 SQLite store, with T02 fields populated and all later-stage fields left at their schema defaults (NULL / `unreviewed`). For B-track rows, `gt_identity`, `orientation`, and `species` are additionally populated from the dataset metadata.
- A console (and optional `--report-json <path>`) summary of ingestion stats.
- An additive line in `STATUS_BOARD.md` noting T02 deliverables.

## Interface contract

Downstream tickets and the runner (T10) depend on EXACTLY these. Do not rename.

### Module-level constants (exact names)

```python
DEFAULT_MD_JSON: str        = "data/MedvednicaDS/megadetector_results.json"
DEFAULT_IMAGES_DIR: str     = "data/MedvednicaDS/animal_images"
DEFAULT_METADATA_CSV: str   = "data/MedvednicaDS/trail_cam_data.csv"
DEFAULT_EXISTING_CROPS: str = "data/MedvednicaDS/animal_crops"
DEFAULT_CROPS_OUT: str      = "data/reid_demo/crops"
DEFAULT_DATASET: str        = "MedvednicaDS"
DEFAULT_CONF_THRESHOLD: float = 0.5
ANIMAL_CATEGORY_ID: str     = "1"   # MegaDetector animal category
```

### Public functions (exact signatures)

```python
from typing import Optional, Iterable, List, Dict, Any, Tuple

def ingest(
    *,
    md_json: Optional[str] = DEFAULT_MD_JSON,
    images_dir: str = DEFAULT_IMAGES_DIR,
    metadata_csv: Optional[str] = DEFAULT_METADATA_CSV,
    existing_crops_dir: Optional[str] = DEFAULT_EXISTING_CROPS,
    crops_out_dir: str = DEFAULT_CROPS_OUT,
    db_path: Optional[str] = None,          # None -> reid_demo.store.DEFAULT_DB_PATH
    dataset: str = DEFAULT_DATASET,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    write_crops: bool = True,               # False -> record paths only, do not write JPGs
    limit: Optional[int] = None,            # cap on number of SOURCE frames (debug)
) -> Dict[str, Any]:
    """Run the full ingestion: load/filter detections, crop, resolve camera/timestamp,
    upsert DetectionRecords. Returns a stats dict (see IngestStats keys below).
    Opens the store via reid_demo.store.connect()."""

def ingest_from_images(
    *,
    images_dir: str,
    out_md_json: Optional[str] = None,   # where to write the MegaDetector JSON; temp if None
    md_threshold: float = 0.1,           # detector output threshold (keep low, filter later)
    **ingest_kwargs,
) -> Dict[str, Any]:
    """Run MegaDetector (repo venv) over a raw image folder to produce a results JSON,
    then call ingest() on it. Requires the 'megadetector' package; raise a clear
    RuntimeError with install hint if unavailable."""

def load_detection_frames(
    md_json: str,
    *,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Parse EITHER the MegaDetector-results format (top-level 'images' or 'predictions')
    OR the flat animal_detections.json format into a normalized list of frame dicts:
        {"source_basename": "IMG_0066.JPG",
         "camera_hint": "Camera 1" | None,
         "timestamp": "2025-06-02 04:27:51" | None,   # if present in JSON
         "animal_dets": [ {"det_index": int (1-based), "bbox": [x,y,w,h],
                           "conf": float}, ... ]}      # ALREADY filtered to animals>=thr
    Persons/vehicles and empty frames are dropped here. det_index is assigned 1-based
    over the kept animal detections in original order, to match existing crop naming."""

def crop_for_detection(
    source_image_path: str,
    bbox: Tuple[float, float, float, float],
    crop_out_path: str,
    *,
    existing_crop_path: Optional[str] = None,
    write: bool = True,
) -> str:
    """Reuse existing_crop_path if given and on disk (return it). Else, if write=True,
    crop normalized bbox -> pixels via (x*W, y*H, (x+w)*W, (y+h)*H), save JPEG q=90 to
    crop_out_path (mkdir parents), return crop_out_path. If write=False, return the
    path it WOULD write without creating the file."""

def resolve_metadata(
    metadata_csv: Optional[str],
) -> Dict[str, Dict[str, Optional[str]]]:
    """Return {image_basename: {"camera_id": str|None, "timestamp": str|None}}
    parsed from trail_cam_data.csv (match by basename of 'filepath' column;
    'camera'->camera_id, 'datetime'->timestamp). Returns {} if csv missing/None."""

def ingest_wildlife_dataset(
    subset: str,
    *,
    max_identities: Optional[int] = None,
    limit: Optional[int] = None,
    db_path: Optional[str] = None,   # None -> reid_demo.store.DEFAULT_DB_PATH
    dataset: Optional[str] = None,   # store --dataset label; defaults to `subset`
) -> Dict[str, Any]:
    """FOURTH adapter. Ingest a labeled WildlifeReID-10k subset (e.g. 'LeopardID2022',
    'ATRW') as GROUND-TRUTH re-id data. Loads the subset via
    utility_functions.load_dataset(subset) / its metadata.csv (columns include
    identity, orientation, species and a relative image path).

    Creates ONE WHOLE-FRAME DetectionRecord PER IMAGE:
      * bbox = (0.0, 0.0, 1.0, 1.0)   # the whole frame IS the crop
      * crop_path = the ORIGINAL full image path on disk (NO cropping, NO MegaDetector,
        matching how the existing main.py pipeline embeds dataset images directly)
      * detector_conf = 1.0           # whole-frame, no detector ran
      * gt_identity   = metadata 'identity'
      * orientation   = metadata 'orientation', with ''/missing -> 'unknown'
      * species       = metadata 'species'
      * dataset       = `dataset` or `subset`
    det_index is 1 for every record (one detection per image); record_id is the usual
    make_record_id(source_stem, 1).

    max_identities caps the number of DISTINCT identities ingested (deterministic order;
    used by T10 --smoke). limit caps the number of images, applied AFTER the
    max_identities filter. Returns a stats dict (see B-track IngestStats keys below).
    Opens the store via reid_demo.store.connect(). Leaves embedding*/cluster*/review_*
    at their dataclass defaults."""
```

### `det_index` / `record_id` convention (binding for the whole pipeline)

- `det_index` is **1-based**, assigned over the kept (animal, above-threshold) detections in the order they appear in the source JSON. This matches the existing `animal_crops/` naming (`_crop1`, `_crop2`, ...). **Document this in code and in your STATUS_BOARD line; T01's `make_record_id` and T08 rely on it.**
- `record_id == reid_demo.store.make_record_id(source_stem, det_index)` == `f"{source_stem}__crop{det_index}"`. `source_stem` is the source image basename without extension (e.g. `IMG_0066`, `02020401`).
- New crop filename: `f"{source_stem}__crop{det_index}.jpg"` (double underscore). Legacy reuse matches glob `f"{source_stem}_crop{det_index}_*.jpg"` (single underscore + conf suffix) in `existing_crops_dir`.

### Field-population contract (which columns T02 sets)

**A-track (MegaDetector / `animal_detections.json` / raw-image adapters).** For each kept crop, set on the `DetectionRecord`:
`record_id, source_image, source_stem, det_index, crop_path, bbox_x, bbox_y, bbox_w, bbox_h, detector_conf, camera_id, timestamp, dataset`. Set `orientation="unknown"` (it's in `ORIENTATIONS`). **Leave everything else at the dataclass default** (`species*`=None, `embedding*`=None, `cluster*`=None, `is_candidate_new`=None, `gt_identity`=None, `review_status="unreviewed"`, `extra_json="{}"`). Do not set `created_at`/`updated_at` (the store sets them).

`bbox_x/y/w/h` are the **normalized** values straight from the detector (no pixel conversion in the record; pixels are only used to make the crop file).

**B-track (`ingest_wildlife_dataset`).** For each labeled image, set on the whole-frame `DetectionRecord`:
`record_id, source_image, source_stem, det_index (=1), crop_path (= original full image path), bbox_x=0.0, bbox_y=0.0, bbox_w=1.0, bbox_h=1.0, detector_conf=1.0, dataset` AND the three GT/label fields **`gt_identity` (from metadata `identity`), `orientation` (from metadata `orientation`, ''/missing → `'unknown'`, must be in `ORIENTATIONS`), and `species` (from metadata `species`)**. Leave `camera_id`/`timestamp` at default (None) unless the metadata supplies them. **Leave `embedding*`, `cluster*`, `is_candidate_new`, `review_status="unreviewed"`, `extra_json="{}"` at the dataclass default.** Do not set `created_at`/`updated_at`. T02 is the sole writer of `gt_identity`/`orientation` for labeled datasets; no downstream ticket re-derives them.

### Stats dict (`IngestStats`) keys returned by `ingest()`

```
frames_total, frames_empty, frames_with_animals,
dets_total, dets_animal, dets_person, dets_vehicle, dets_below_threshold,
crops_written, crops_reused, crops_missing_source,
records_upserted, dataset, db_path
```
`frames_empty` counts source frames that yielded zero kept animal crops (used by T09's "% empty removed" story; compute `pct_empty = frames_empty / frames_total`).

For the **B-track `ingest_wildlife_dataset`**, return a stats dict with at least:
```
images_total, images_ingested, identities_total, identities_ingested,
records_upserted, subset, dataset, db_path
```
where `identities_ingested` reflects the `max_identities` cap and `images_ingested` reflects the `limit` cap.

### CLI

```
python -m reid_demo.ingest \
    [--md-json PATH | --images-dir DIR | --wildlife-subset NAME] \
    [--images-dir DIR] [--metadata-csv PATH] \
    [--existing-crops DIR] [--crops-out DIR] \
    [--db PATH] [--dataset NAME] \
    [--conf-threshold FLOAT] [--no-crop] [--limit N] \
    [--max-identities N] \
    [--report-json PATH]
```
- Default (no args) ingests Medvednica from the JSON into the default store and prints stats.
- `--images-dir` without `--md-json` triggers the raw-image MegaDetector path (`ingest_from_images`).
- `--wildlife-subset NAME` (e.g. `LeopardID2022`, `ATRW`) triggers the B-track path (`ingest_wildlife_dataset(NAME, max_identities=..., limit=...)`); `--max-identities` and `--limit` are passed through, and `--dataset` defaults to the subset name. The A-track-only flags (`--md-json`, `--existing-crops`, `--crops-out`, `--conf-threshold`, `--no-crop`) are ignored on this path.
- Exit non-zero on fatal error (missing input file, store mismatch); exit 0 on success.

## Existing code to reuse (real paths)

- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/reid_demo/store.py` — **T01, the only store API.** Use `connect`, `upsert_records`, `make_record_id`, `DetectionRecord`, `DEFAULT_DB_PATH`, `ORIENTATIONS`. Read `reid_demo/DATA_CONTRACT.md` for the field meanings. (If T01 isn't merged yet, code against these documented signatures.)
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/deprecated/seminar_classify_species.py` — `save_crop(img_path, bbox, dest_dir, stem, idx)` at lines 26-33 is **REFERENCE-ONLY for the crop MATH** (`crop((x*W, y*H, (x+w)*W, (y+h)*H))`, JPEG quality 90). It is **NOT a reusable function**: do NOT import or call it. Reimplement the math inside `crop_for_detection`, and note two deliberate differences from the legacy `save_crop`: (1) new crop filenames use the **double-underscore** `{stem}__crop{idx}.jpg` form (matching `record_id`), not the legacy single-underscore `_crop{idx}_conf{pct}` form; (2) the source image is resolved to an **absolute path** before opening. Also note its timestamp-injection pattern (lines 88-95: match `trail_cam_data.csv` by basename, `datetime` column) for `resolve_metadata`.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/utils/clean_detections.py` — REFERENCE for the **animal-vs-other + confidence filter** logic (`detection_categories` map, `is_animal`, `conf` cutoff). Reimplement the rule in `ingest.py`. **T02's policy INTENTIONALLY differs from this legacy cleaner:** T02 applies a **per-detection** `conf >= 0.5` cutoff (each animal box kept on its own confidence), whereas the legacy cleaner applied a **whole-frame strict `conf > 0.5`** filter. Do NOT recompute "kept"/empty counts from `clean_detections.py`; where an on-disk cleaned/detections file already exists (e.g. `detections_cleaned.json`), **trust the on-disk file** rather than re-running the legacy cleaner.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/utils/m2dspeciesnet.py` — shows the MegaDetector JSON shape (`images` -> `file`, `detections`) and that the basename is taken via `Path(im['file']).name`. Mirror this when reading `file`.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/data/MedvednicaDS/megadetector_results.json` — the real primary input. `animal_detections.json` (flat dict), `trail_cam_data.csv`, `animal_images/`, `animal_crops/` — the real fixtures to test against.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/utility_functions.py` — `load_dataset(subset, root=WILD_DATASET_PATH)` at line 26 is the **B-track loader** to reuse in `ingest_wildlife_dataset`. It returns a pandas DataFrame with at least `image_id`, `identity`, `dataset`, and `path` columns; the WildlifeReID-10k metadata also carries `orientation` and `species`. The image root is `constants.WILD_DATASET_PATH`; resolve each record's absolute image path by joining the root with the row's `path`. This is exactly the source `main.py` embeds from, so whole-frame `crop_path` = that absolute image path (no cropping).
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/constants.py` — style reference for path constants (`ROOT_DIR = os.path.dirname(os.path.abspath(__file__))`) and the source of `WILD_DATASET_PATH` (B-track image root). Do NOT edit it; define T02 constants in `ingest.py`.
- MegaDetector run API (venv only): `from megadetector.detection import run_detector` / `megadetector.detection.run_detector_batch`. Importable via `venv/bin/python` (confirmed); NOT in system python. The raw-image path must degrade with a clear error if the package is missing.

## Implementation notes

- **Primary path is JSON-in, no model.** The whole Medvednica demo runs from `megadetector_results.json`; do not require downloading/running MegaDetector for the default flow. Keep the model import lazy (inside `ingest_from_images`) so importing `reid_demo.ingest` never pulls in torch/megadetector.
- **Basename, not the camera subfolder.** In `megadetector_results.json`, `file` is `"Camera 1/IMG_0001.JPG"` but the image on disk is `animal_images/IMG_0001.JPG`. Resolve the source path as `os.path.join(images_dir, Path(file).name)`. Capture the leading subfolder (if any) as `camera_hint` — use it for `camera_id` only when the CSV has no camera for that basename.
- **camera_id precedence:** CSV `camera` (by basename) → JSON `camera_hint` (the `file` subfolder, e.g. `Camera 1`) → `"unknown_camera"`. Medvednica's CSV gives `unknown_camera`; that's fine.
- **timestamp precedence:** CSV `datetime` (by basename) → a `timestamp` field already on the JSON frame record (some project JSONs stitch it) → `None`. Keep the exact ISO string `YYYY-MM-DD HH:MM:SS`; do not reformat.
- **Crop reuse:** before writing a new crop, glob `existing_crops_dir` for `f"{stem}_crop{det_index}_*.jpg"`. If exactly one match exists, set `crop_path` to that existing file and count it as `crops_reused` (don't re-encode). If multiple/zero, fall back to writing a fresh crop at `crops_out_dir/{stem}__crop{det_index}.jpg`. This lets the demo reuse the 4,194 crops already on disk.
- **Missing source image:** if the source frame isn't found under `images_dir` and there's no reusable existing crop, skip that detection, increment `crops_missing_source`, and continue (do not crash a 8k-frame run on one missing file). Still safe to write the record only if a crop path exists — if no crop can be produced, **skip the record entirely** (a record must always point at a real crop file for T04 to embed).
- **Idempotency:** `record_id` is deterministic, so `upsert_records` overwrites in place. Crop files use deterministic names, so re-running overwrites/reuses rather than duplicating. Re-running must not change `records_upserted` semantics beyond "rows present".
- **Batch the upserts** in a single transaction via `upsert_records` (T01 wraps a transaction). For 8k frames / ~4k crops this is fine in memory; if `limit` is set, stop after that many *source frames*.
- **Two input formats in `load_detection_frames`:** detect by structure — if top-level is a dict with `"images"` or `"predictions"` → MegaDetector-results format (has `detection_categories`, per-det `category`/`conf`); if top-level is a flat dict mapping basenames → lists with `confidence` keys → `animal_detections.json` format (already animal-only; treat every box as animal, key is `conf` under name `confidence`, no person/vehicle counts). Assign 1-based `det_index` over kept boxes in file order in both cases.
- **Confidence filter** is **per-detection** and uses `>=` against `conf_threshold` (document this). This **INTENTIONALLY differs** from the legacy `utils/clean_detections.py`, which applied a **whole-frame strict `conf > 0.5`** filter: T02 evaluates each animal box on its own confidence with `>=`, which is friendlier and avoids the default 0.5 boundary landing exactly. Count `dets_below_threshold` for animals that failed the cutoff (persons/vehicles are counted under their own categories regardless of conf). Do not recompute kept/empty counts by re-running the legacy cleaner; if an on-disk cleaned detections file (e.g. `detections_cleaned.json`) exists, trust it as-is.
- **PIL** is available in the venv (Pillow 11.x). Use `PIL.Image`; `convert("RGB")` before crop/save to avoid mode issues. Guard against zero-area or out-of-range bboxes by clamping pixel coords to `[0,W]`/`[0,H]` and skipping if the resulting box has <1px width/height (count as `crops_missing_source`? no — count as a separate skip; simplest: skip + log, do not crash).
- **B-track `ingest_wildlife_dataset` (labeled subsets).** Load `df = utility_functions.load_dataset(subset)`. For each row, resolve the absolute image path by joining `constants.WILD_DATASET_PATH` with the row's `path`; that absolute path IS the `crop_path` (no cropping, no MegaDetector — the whole frame is the crop, mirroring how `main.py` embeds dataset images directly). Build one whole-frame `DetectionRecord` per image with `bbox=(0.0,0.0,1.0,1.0)`, `det_index=1`, `detector_conf=1.0`, `source_stem = Path(path).stem`, and populate `gt_identity` (from `identity`), `species` (from `species`), and `orientation` (from `orientation`). **Normalize orientation: empty string / missing / NaN → `'unknown'`**, and assert the result is in `ORIENTATIONS`. Apply `max_identities` by selecting the first N distinct `identity` values in a deterministic order (e.g. sorted), then apply `limit` to cap the number of images. Upsert via `reid_demo.store.upsert_records` in one transaction. This adapter must NOT import torch/megadetector; it only needs `utility_functions`, pandas, and the store. Tag rows with `dataset` (defaults to `subset`).
- Keep the module importable under **system python3 for tests that don't run the model** (tests must not require torch). Only `ingest_from_images` may need the venv.
- Add one line to `STATUS_BOARD.md` under T02 noting: module path, the 1-based `det_index` convention, default dataset `MedvednicaDS`, and that species/embeddings/clusters are intentionally left NULL.

## Acceptance criteria

- [ ] `reid_demo/ingest.py` and `tests/test_ingest.py` exist; no existing repo file is modified except an additive line in `STATUS_BOARD.md`.
- [ ] `python -c "from reid_demo.ingest import ingest, ingest_from_images, ingest_wildlife_dataset, load_detection_frames, crop_for_detection, resolve_metadata, DEFAULT_MD_JSON, DEFAULT_IMAGES_DIR, DEFAULT_METADATA_CSV, DEFAULT_EXISTING_CROPS, DEFAULT_CROPS_OUT, DEFAULT_DATASET, DEFAULT_CONF_THRESHOLD, ANIMAL_CATEGORY_ID"` succeeds (every contracted name importable, including the fourth adapter `ingest_wildlife_dataset`) **without importing torch/megadetector**.
- [ ] `load_detection_frames("data/MedvednicaDS/megadetector_results.json", conf_threshold=0.5)` returns frames where every `animal_dets` entry has `conf >= 0.5`, `det_index` starts at 1 and increments per frame, and frames with no kept animal box are excluded from `animal_dets` (length-0 lists allowed but contribute to `frames_empty`).
- [ ] `load_detection_frames` also parses the flat `data/MedvednicaDS/animal_detections.json` format (treats all boxes as animals, reads `confidence`).
- [ ] On a kept detection, `record.record_id == make_record_id(record.source_stem, record.det_index)` and `record.crop_path` points at an existing file on disk (either a reused legacy crop or a freshly written one).
- [ ] Each written **A-track** `DetectionRecord` has `species is None`, `gt_identity is None`, `embedding_ref is None`, `cluster_id is None`, `review_status == "unreviewed"`, `dataset == "<--dataset>"`, `orientation == "unknown"` (in `ORIENTATIONS`), and normalized `bbox_x/y/w/h` equal to the detector values. (B-track rows are covered by the `ingest_wildlife_dataset` criteria below — they DO set `species`, `gt_identity`, and a metadata-derived `orientation`.)
- [ ] Running `ingest(..., existing_crops_dir="data/MedvednicaDS/animal_crops", write_crops=True)` on a small `--limit` reuses legacy crops where a `{stem}_crop{idx}_*.jpg` match exists (`crops_reused > 0`) and never duplicates a `record_id` (re-running keeps row count stable).
- [ ] Persons/vehicles and empty frames produce **zero** records; the stats dict reports `dets_person`, `dets_vehicle`, `frames_empty`, and a derivable `pct_empty`, and `records_upserted == sum(len animal_dets kept that produced a crop)`.
- [ ] `crop_for_detection` with `write=False` returns a path and creates no file; with a valid `existing_crop_path` returns that path unchanged.
- [ ] `python -m reid_demo.ingest --md-json data/MedvednicaDS/megadetector_results.json --limit 20 --db /tmp/t02.sqlite` exits 0 and prints stats; querying the store afterward returns the expected rows for those frames.
- [ ] Idempotency: running the same CLI command twice yields the same number of rows in the store (no duplicates), with `updated_at` advanced.
- [ ] `ingest_wildlife_dataset("LeopardID2022", max_identities=3, db_path="/tmp/t02b.sqlite")` writes whole-frame rows where every row has `bbox_x==0.0 and bbox_y==0.0 and bbox_w==1.0 and bbox_h==1.0`, `det_index==1`, `detector_conf==1.0`, `crop_path` pointing at an existing original image, `gt_identity` non-null, `species` non-null, `orientation in ORIENTATIONS`, and `embedding_ref is None and cluster_id is None and review_status == "unreviewed"`. At most 3 distinct `gt_identity` values appear (the `max_identities` cap).
- [ ] On the B-track, any image whose metadata `orientation` is empty/missing is stored as `orientation == "unknown"` (never `''`).
- [ ] `python -m reid_demo.ingest --wildlife-subset LeopardID2022 --max-identities 3 --db /tmp/t02b.sqlite` exits 0, prints B-track stats, and the store holds the GT-populated rows above; `--dataset` defaults to `LeopardID2022`.
- [ ] `tests/test_ingest.py` passes under the repo venv (synthetic JSON + PIL-generated dummy images; no network, no model). B-track coverage uses a tiny synthetic metadata DataFrame / `load_dataset` stub (no real WildlifeReID-10k download) to assert the whole-frame + GT-population + `max_identities` + orientation-normalization behavior.

## How to verify

```bash
cd /home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project
source venv/bin/activate    # repo venv (has PIL + megadetector)

# 0. T01 store must be importable
python -c "from reid_demo.store import connect, DetectionRecord, make_record_id, DEFAULT_DB_PATH, ORIENTATIONS; print('store OK')"

# 1. Import surface (must NOT pull torch)
python -c "from reid_demo.ingest import ingest, ingest_from_images, ingest_wildlife_dataset, load_detection_frames, crop_for_detection, resolve_metadata; print('ingest OK')"

# 2. Parse the real MegaDetector JSON and check filtering + det_index
python - <<'PY'
from reid_demo.ingest import load_detection_frames
frames = load_detection_frames("data/MedvednicaDS/megadetector_results.json", conf_threshold=0.5)
print("frames:", len(frames))
empties = sum(1 for f in frames if not f["animal_dets"])
print("empty frames:", empties)
f = next(fr for fr in frames if fr["animal_dets"])
idxs = [d["det_index"] for d in f["animal_dets"]]
assert idxs == list(range(1, len(idxs)+1)), idxs
assert all(d["conf"] >= 0.5 for d in f["animal_dets"])
print("first non-empty frame:", f["source_basename"], "det_indexes:", idxs)
PY

# 3. Parse the flat animal_detections.json format
python - <<'PY'
from reid_demo.ingest import load_detection_frames
frames = load_detection_frames("data/MedvednicaDS/animal_detections.json", conf_threshold=0.5)
print("flat-format frames:", len(frames), "example:", frames[0]["source_basename"], len(frames[0]["animal_dets"]))
PY

# 4. End-to-end ingest of a small slice into a temp store, reusing legacy crops
python -m reid_demo.ingest \
    --md-json data/MedvednicaDS/megadetector_results.json \
    --images-dir data/MedvednicaDS/animal_images \
    --metadata-csv data/MedvednicaDS/trail_cam_data.csv \
    --existing-crops data/MedvednicaDS/animal_crops \
    --crops-out /tmp/t02_crops \
    --db /tmp/t02.sqlite --dataset MedvednicaDS \
    --limit 50 --report-json /tmp/t02_stats.json
echo "exit=$?"
cat /tmp/t02_stats.json

# 5. Inspect the rows the store now holds (uses T01 API)
python - <<'PY'
from reid_demo.store import connect, query_records, make_record_id, ORIENTATIONS
conn = connect("/tmp/t02.sqlite")
rows = query_records(conn, dataset="MedvednicaDS")
print("records:", len(rows))
r = rows[0]
assert r.record_id == make_record_id(r.source_stem, r.det_index)
assert r.species is None and r.embedding_ref is None and r.cluster_id is None
assert r.review_status == "unreviewed" and r.orientation in ORIENTATIONS
import os; assert os.path.exists(r.crop_path), r.crop_path
print("sample:", r.record_id, r.crop_path, r.camera_id, r.timestamp, r.detector_conf)
PY

# 6. Idempotency: re-run, row count unchanged
python -m reid_demo.ingest --md-json data/MedvednicaDS/megadetector_results.json \
    --images-dir data/MedvednicaDS/animal_images --existing-crops data/MedvednicaDS/animal_crops \
    --crops-out /tmp/t02_crops --db /tmp/t02.sqlite --limit 50 >/dev/null
python - <<'PY'
from reid_demo.store import connect, query_records
print("rows after re-run:", len(query_records(connect("/tmp/t02.sqlite"), dataset="MedvednicaDS")))
PY

# 7. B-track: ingest a labeled subset whole-frame with GT identity/orientation/species
python -m reid_demo.ingest --wildlife-subset LeopardID2022 --max-identities 3 \
    --db /tmp/t02b.sqlite --report-json /tmp/t02b_stats.json
echo "exit=$?"
cat /tmp/t02b_stats.json
python - <<'PY'
import os
from reid_demo.store import connect, query_records, make_record_id, ORIENTATIONS
rows = query_records(connect("/tmp/t02b.sqlite"), dataset="LeopardID2022")
print("B-track records:", len(rows))
assert len({r.gt_identity for r in rows}) <= 3              # max_identities cap
for r in rows:
    assert (r.bbox_x, r.bbox_y, r.bbox_w, r.bbox_h) == (0.0, 0.0, 1.0, 1.0)
    assert r.det_index == 1 and r.detector_conf == 1.0
    assert r.gt_identity is not None and r.species is not None
    assert r.orientation in ORIENTATIONS and r.orientation != ""
    assert r.embedding_ref is None and r.cluster_id is None and r.review_status == "unreviewed"
    assert os.path.exists(r.crop_path), r.crop_path
print("B-track GT-population OK")
PY

# 8. Tests
python -m pytest tests/test_ingest.py -q
```

## Open questions

1. **`det_index` base.** Confirmed 1-based to match the existing `animal_crops/` naming (`_crop1`). T01's `make_record_id` just concatenates whatever integer T02 passes; T05/T08 read `record_id` as opaque. Flag if any consumer assumed 0-based.
2. **Raw-image path scope.** The demo's Medvednica flow never needs to run MegaDetector (the JSON exists). Is `ingest_from_images` needed for the *first* pitch, or only for future parks' raw footage? It's specified here for completeness but can be a thin/lazy wrapper if time-boxed — the JSON path is the must-have.
3. **Sub-threshold animal crops.** Default keeps `conf >= 0.5`. Risnjak/lynx footage may be sparse; should the demo keep a lower threshold (e.g. 0.3) to avoid dropping faint nocturnal lynx? Exposed as `--conf-threshold`; default 0.5 for now, revisit with the park's data.
4. **Crop source of truth.** We reuse legacy `animal_crops/` (which were cropped with the same math + a `_conf` suffix). If T04's embedding quality depends on exact crop padding/letterboxing, confirm reused crops and freshly written crops are pixel-equivalent (they should be — same `(x*W,y*H,(x+w)*W,(y+h)*H)` math, q=90 JPEG). If T04 wants padding/aspect-preservation, that belongs in T04's preprocessing, not here.
5. **One DB, many runs.** Records are tagged with `--dataset`; LeopardID2022/ATRW ingestion (for T07 eval) goes through the dedicated B-track adapter `ingest_wildlife_dataset(subset, ...)`, which sets `--dataset` to the subset name by default. **T02 is the sole owner of `gt_identity`, `orientation`, and `species` for these labeled datasets and populates them at ingest from the metadata** — RESOLVED, no longer an open question. T07/T10 read these GT fields from the store; they do not re-derive them. (T10 plumbs `--max-identities` straight through to `ingest_wildlife_dataset(max_identities=...)` for its `--smoke` runs.)
