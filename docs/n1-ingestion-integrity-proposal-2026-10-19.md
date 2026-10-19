# N1 proposal: collision-safe ingestion and review-preserving re-ingestion

Date: 19 October 2026. Work item (e) in the [decision log](lynx-demo-decision-log.md). Source: Now item N1 of the [pitch-readiness analysis](research/pitch-readiness-analysis-2026-10-19.md#41-now--next--later-with-gates).

**Status: approved 19 Oct 2026 and implemented as written.** One refinement: SpeciesNet matching uses the frame with the longest shared trailing path (at least one folder), not a plain suffix match, so differing archive roots still resolve. User-run tests passed: focused 87 in 16.08 s; full suite 723 in 96.12 s. Not committed yet.

## Problem

There are two defects. Both were reproduced in temporary fixtures on 10 September ([pilot readiness](pilot-readiness-2026-09-10.md) §Gaps 1–2), and both are still present in today's code. Either one corrupts a partner archive silently, so a multi-camera pilot is blocked until they are fixed.

1. **Filename collisions.**
   - `ingest.load_detection_frames` keeps only the basename of each MegaDetector `file` (`_frame_basename_and_camera`, `reid_demo/ingest.py:89`).
   - `ingest()` then builds three things from the stem alone: the source path (`images_dir/basename`, line 422), the crop path (`{stem}__crop{n}.jpg`, line 432) and the record ID (`make_record_id(stem, n)`, line 464).
   - `resolve_metadata` keys the CSV by basename (line 307).
   - `Camera 1/IMG_0001.JPG` and `Camera 2/IMG_0001.JPG` therefore become one record. The second **overwrites** the first's crop file, metadata and row.
   - `species_filter.ingest_speciesnet_json` matches predictions to rows by stem (line 260), so it inherits the ambiguity.
2. **Re-ingestion resets downstream state.**
   - `store.upsert_records` refreshes every non-key column on conflict (`_UPSERT_SQL`, `reid_demo/store.py:318–326`).
   - Re-ingesting a reviewed record writes the dataclass defaults back. In September this changed `review_status=confirmed`, `cluster_id=7` and an embedding reference to `unreviewed`, `None` and `None`.

## Design principles

- **New behaviour is additive.** The existing `upsert_record(s)` keeps its full-replace semantics, because about 80 test call sites and `import_records` (restore from export) rely on them. Ingestion moves to a new stage-owned upsert.
- **No schema change.** `SCHEMA_VERSION` stays 1. The only new persistent state is a key/value row in the existing `meta` table.
- **Fail loudly rather than overwrite.** A real collision raises an error with the colliding paths. It is never resolved silently.
- **Existing demo databases keep working**, but only with an explicit legacy flag, and even then collisions are detected.

## Planned changes (5 files edited, 0 new modules)

### 1. `reid_demo/store.py`

**Scoped record IDs.** `make_record_id` itself is unchanged.

```python
def make_source_key(source_path: str, root: Optional[str] = None) -> str:
    """Stable, OS-independent identity of a source frame: the POSIX path relative to
    `root` (the ingest images_dir / dataset root), normalised; absolute normpath if the
    file is outside `root`."""

def make_scoped_record_id(source_key: str, det_index: int) -> str:
    """Collision-safe record_id: f'{stem}__{sha1(source_key)[:10]}__crop{det_index}'.
    Readable (keeps the stem), filesystem-safe, deterministic across re-ingests."""
```

**Per-dataset ID scheme.** These read and write a meta row; no schema change.

```python
RECORD_ID_SCHEMES = {"path", "legacy"}
def get_record_id_scheme(conn, dataset) -> Optional[str]     # meta key f"record_id_scheme:{dataset}"
def set_record_id_scheme(conn, dataset, scheme) -> None      # refuses to change an existing value
```

**Stage-owned ingestion upsert.** This is new; `upsert_records` is untouched.

```python
INGEST_OWNED = ["source_image", "source_stem", "det_index", "crop_path", "bbox_x", "bbox_y",
                "bbox_w", "bbox_h", "detector_conf", "camera_id", "timestamp", "dataset",
                "gt_identity"]                       # always refreshed from the incoming row
FILL_IF_NULL = ["orientation", "species", "species_conf", "species_class"]   # B-track metadata
# Never touched on conflict: embedding_*, cluster_*, is_candidate_new, review_status,
# review_note, extra_json, created_at.
GEOMETRY_COLS = ["source_image", "crop_path", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]

class RecordCollisionError(ValueError): ...

def ingest_upsert_records(conn, records) -> Dict[str, int]:
    """One transaction. For each record:
      - new id                       -> insert                               (inserted)
      - existing id, different source_image -> raise RecordCollisionError (both paths)
      - existing id, same geometry   -> refresh INGEST_OWNED, fill FILL_IF_NULL    (refreshed)
      - existing id, geometry changed (bbox/crop moved) -> as above, PLUS clear values
        derived from the old crop (embedding_ref/path, species*, cluster_id/conf,
        is_candidate_new). If review_status != 'unreviewed', KEEP the review and set
        extra_json['review_needs_recheck'] = {'reason': 'geometry_changed',
        'previous': {...old bbox/crop...}, 'at': ts}       (geometry_changed, reviews_flagged)
    Returns the counts in parentheses."""
```

Clearing derived values on a geometry change is an explicit, logged invalidation. The current code silently resets everything on any re-ingest. A human review is never cleared; it is only flagged for a re-check.

### 2. `reid_demo/ingest.py`

- **Keep the relative path.** `load_detection_frames` adds `source_relpath` (the full MD `file` value) to each frame. The existing `source_basename` and `camera_hint` stay.
- **Source resolution:** use `images_dir/source_relpath` if it exists. Otherwise fall back to `images_dir/basename` **only if that basename is unique in the batch** (today's flat Medvednica layout). Anything else is counted under a new `sources_ambiguous` stat and skipped.
- **Metadata CSV:** index by normalised relative path and by basename. The basename lookup is used only when that basename is unique in the CSV.
- **Record IDs:** a new argument `record_id_scheme: Optional[str] = None`, with CLI `--record-id-scheme {path,legacy}`.
  - The scheme is resolved from the store's meta row for the dataset.
  - If the dataset has no rows yet, the default is `"path"`, which is then recorded.
  - If the dataset already has rows but no meta row (an existing demo DB), ingestion **raises** and asks for an explicit `--record-id-scheme legacy`.
  - `path` uses `make_scoped_record_id(make_source_key(src, images_dir), n)`.
  - `legacy` keeps `make_record_id(stem, n)` but raises if two distinct sources in the batch share a stem. `ingest_upsert_records` independently raises `RecordCollisionError` against stored rows.
- **Crop files:** written as `crops_out_dir/{record_id}.jpg`, so they are unique under `path`. Legacy-crop reuse (`_find_legacy_crop`, stem-based) applies only under `legacy`.
- **Both tracks** (A and B) call `store.ingest_upsert_records`. The B-track source key is the dataset-relative `path`, which is already unique.
- **New stats keys:** `records_inserted`, `records_refreshed`, `records_geometry_changed`, `reviews_flagged_for_recheck`, `sources_ambiguous`, `record_id_scheme`. `records_upserted` keeps its meaning (inserted + refreshed + geometry changed).

### 3. `reid_demo/species_filter.py`

In `ingest_speciesnet_json`, candidates for a stem are first restricted to rows whose `source_image` ends with the prediction's normalised `filepath`, compared as a relative-path suffix. If the stem maps to a single source, behaviour is unchanged. If several sources share the stem and none or several match the suffix, the detection is counted in a new `skipped_ambiguous` field rather than matched by bbox alone. The nearest-bbox rule within a frame is unchanged.

### 4. `reid_demo/DATA_CONTRACT.md`

Document:
- scoped IDs and the per-dataset scheme;
- stage ownership of columns;
- `ingest_upsert_records` vs `upsert_records` (replace; used for import/restore and tests);
- the re-check flag.

### 5. Tests: `tests/test_store.py` and `tests/test_ingest.py`

New tests:

| Test | Asserts |
| --- | --- |
| `test_two_cameras_same_basename_are_distinct` | MD JSON with `cam_a/IMG_0001.JPG` and `cam_b/IMG_0001.JPG` in subfolders → 2 records, 2 distinct crop files with the right pixels, correct camera IDs and timestamps from a CSV keyed by path |
| `test_reingest_preserves_review_embedding_cluster` | After `update_embedding`, `update_cluster` and `update_review(confirmed)`, a second `ingest()` leaves all of them unchanged and advances `updated_at` (the 10 Sep reproduction, inverted) |
| `test_reingest_geometry_change_invalidates_but_keeps_review` | Moved bbox → embedding and cluster cleared, review kept, `review_needs_recheck` set, stats count 1 |
| `test_ingest_upsert_fill_if_null` | B-track species fills NULL but does not overwrite a T03 species |
| `test_ingest_upsert_collision_raises` | Same ID, different `source_image` → `RecordCollisionError`, transaction rolled back |
| `test_legacy_scheme_batch_collision_raises` | `legacy` + two sources with one stem → `ValueError` listing both paths |
| `test_existing_dataset_requires_explicit_scheme` | Rows present without a meta row → error unless `record_id_scheme="legacy"` |
| `test_ambiguous_flat_fallback_skipped` | Flat layout with a non-unique basename → `sources_ambiguous` counted, no record |
| `test_speciesnet_disambiguates_by_path` | Two cameras with the same stem → each prediction lands on its own record; an ambiguous one is counted |
| `test_upsert_records_still_replaces` | The existing replace semantics are pinned explicitly |

**Edits to existing tests (disclosed):** `tests/test_ingest.py` has 10 lookups of the form `make_record_id("IMG_A", 1)` against ingested rows (in `test_ingest_end_to_end`, `test_ingest_idempotent` and a few others). Under the new default scheme those IDs change. I will replace them with a small helper that finds the record by `(source_stem, det_index)`. The assertion `r.record_id == make_record_id(r.source_stem, r.det_index)` becomes the scoped equivalent. No assertion about behaviour is weakened.

No other test file builds IDs from ingestion. They create records directly with `make_record_id`, which is unchanged.

## Out of scope (listed in the Risnjak assessment; separate proposals if wanted)

- `run_demo._store_stage_done` treats a stage as done if **any** row has output.
- A forced fused re-clustering can overwrite reviewed records, and fresh cluster labels can collide with reviewed IDs.
- `embed_records(only_missing=True)` skips rows by presence only.

These matter for a pilot too. They touch `run_demo.py`, `cluster.py` and `embed.py`, and I'd propose them as N1b after this lands.

## Validation (user-run)

```bash
venv/bin/python -m pytest tests/test_store.py tests/test_ingest.py tests/test_species_filter.py -q
```

```bash
venv/bin/python -m pytest tests -q
```

The full suite needs to be rerun because `ingest` is used by `run_demo`. The last full-suite receipt was 302 passed (10 Sep).

## Risks

- **Existing local demo DBs** (for example `data/reid_demo/reid_demo.sqlite`) need `--record-id-scheme legacy` to re-ingest. This is a deliberate, loud change.
- **Crop filenames change** for new datasets (`IMG_0001__a1b2c3d4e5__crop1.jpg`). Catalogue HTML uses `crop_path`, so it is unaffected.
- **Size:** about 250 lines of source and about 200 lines of tests.
