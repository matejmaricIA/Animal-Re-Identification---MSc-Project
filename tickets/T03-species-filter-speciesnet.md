# T03 — SpeciesNet species-filter adapter

> **Status:** 🔴 Not started · **Milestone:** M1 (Demo-ready)
> **Depends on:** T01, T02 · **Blocks:** T10
> **Owner:** _unassigned_
>
> Self-contained ticket for the lynx re-identification demo/pilot. See [STATUS_BOARD.md](../STATUS_BOARD.md) for the full plan, the binding design decisions, and the agent working protocol.

---

## Context

We are building a DEMO + PILOT MVP of an **open-set individual-animal re-identification** system for Eurasian lynx (public analogs: LeopardID2022 leopards, ATRW Amur tigers). The repo at `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project` already has a closed-set pipeline; the demo adds a small constellation of independent modules (T01–T10) that all read/write **one shared per-crop "detection record"** through a single store module defined in **T01** (`reid_demo/store.py`). Read `reid_demo/DATA_CONTRACT.md` (produced by T01) before writing any code — it is the authoritative schema.

This ticket, **T03**, is the **species-classification + target-species filter** stage. Its job: for every crop already ingested by **T02** (one T01 `detections` row per crop, with `crop_path`, `source_stem`, `det_index`, `bbox`, etc. populated), attach a **species label + confidence + full taxonomy string**, and mark which crops are the **target species** (lynx for field data; leopard/tiger for the public eval datasets). The `species_kept` flag T03 writes is **REPORT-ONLY**: T09's Medvednica report and any human triage read it, but it is NOT a clustering-input selector. T05 selects its clustering inputs by querying the `species` column (e.g. `species` in the target alias set), not by `species_kept`. Everything downstream (T06 catalogue, T07 eval, T09 Medvednica report) also reads the `species*` columns T03 writes.

Pipeline position (one crop = one row, T03 fills the species columns):
```
T02 (crop + bbox + camera + timestamp)
  --> T03 (species + species_conf + species_class + keep/drop flag)   <-- THIS TICKET
  --> T04 (embedding) --> T05 (cluster) --> T06/T07/T08/T09 read-only
```

**Two honest data paths this ticket must serve:**

1. **Medvednica real Croatian footage (`dataset="MedvednicaDS"`)** — SpeciesNet has ALREADY been run; the result lives at `data/MedvednicaDS/animals_classified.json`. T03's primary job here is to **re-use that JSON** and write its per-detection classifications onto the matching T01 records — no re-cropping, no model download. The JSON's detections are joined to the T02 records by **`(source_stem, bbox)` nearest-match** (see Implementation notes / D3), NOT by positional enumerate. (T03 must ALSO be able to call SpeciesNet live on a fresh crop folder, but that path is secondary and may require a GPU/model the demo machine might not have.) For "kept"/empty counts T03 trusts the on-disk `data/MedvednicaDS/detections_cleaned.json` produced by T02; it does NOT recompute kept/empty from `clean_detections.py`. Note T02's per-detection `conf>=0.5` ingestion policy INTENTIONALLY differs from the legacy whole-frame strict `conf>0.5` cleaner, so counts may differ — that is expected.
2. **LeopardID2022 / ATRW (`dataset="LeopardID2022"` etc.)** — the species is **already known** (every image is a `leopard`, every ATRW image is a `tiger`); there is no need to run a species classifier. T03 provides `set_known_species()` to stamp the fixed species onto all rows so the keep-filter and downstream tickets behave uniformly. (Confirmed: in WildlifeReID-10k metadata, the `species` column for LeopardID2022 is the single value `leopard`.)

**Real data shapes T03 must consume (verified against the repo):**

- `data/MedvednicaDS/animals_classified.json` top-level keys: `['predictions']`. Each prediction record: `{'filepath': 'animal_images/IMG_0066.JPG', 'detections': [...], 'timestamp': '2025-06-02 04:27:51'}`. Each detection: `{'category': '1', 'conf': 0.784, 'bbox': [x,y,w,h] normalized, 'classifications': {'classes': [...], 'scores': [...]}}`.
- `classifications.classes` and `classifications.scores` are **parallel arrays sorted best-first**; index 0 is the top class. A class string is a full taxonomy: `uuid;class;order;family;genus;species;common_name`, e.g. `d372cda5-...;mammalia;cetartiodactyla;suidae;sus;scrofa;wild boar`. The human-readable common name is `class_string.split(';')[-1]`. The genus is field index 4, species epithet index 5 (e.g. `...;felidae;lynx;lynx;eurasian lynx`). `blank` (empty taxonomy `uuid;;;;;;blank`) means SpeciesNet thinks the crop is empty.
- Real Medvednica species breakdown (top class per crop) includes: `wild boar` (638), `blank` (445), `european roe deer` (349), `red fox` (101), `eurasian badger` (41), `brown bear` (21), `bobcat` (25), `domestic cat` (44), and a few `canada lynx` / `wild cat` / `cat family` hits. There is **no** wild Eurasian lynx in this particular Medvednica sample — that is fine and expected; T03 still labels everything and the filter simply yields few/zero lynx. The DEMO story is "we can label + filter to a target species on real Croatian footage", not "this sample contains lynx".

The **target-species matching must be tolerant**: SpeciesNet emits several cat-family names. For `target="lynx"` we want to keep any felid/lynx-genus class (`eurasian lynx`, `canada lynx`, `iberian lynx`, `lynx`, `bobcat`?, `wild cat`?, `cat family`?). This ticket defines an explicit alias/inclusion policy (see Implementation notes) rather than a single exact-string match.

## Objective

Deliver **one self-contained module** `reid_demo/species_filter.py` (+ tests) that:

1. Reads an existing SpeciesNet predictions JSON (e.g. `data/MedvednicaDS/animals_classified.json`) and writes `species` / `species_conf` / `species_class` onto the matching **T01 records** via `store.update_species`, joining each JSON detection to a T01 record by **`(source_stem, bbox)` nearest-match** (see D3 / Implementation notes) — **no re-cropping**, no positional enumerate, no index/index+1 heuristic.
2. Optionally runs SpeciesNet **live** on a folder of crops (reusing the proven crop+CLI+stitch pattern in `deprecated/seminar_classify_species.py`) when no precomputed JSON exists.
3. Provides `set_known_species()` to stamp a fixed species on datasets whose species is already known (LeopardID2022 → `leopard`, ATRW → `tiger`) **without** any model.
4. Applies a **target-species keep filter** with a tolerant alias policy, persisting a per-record `species_kept` flag in `extra_json` via `store.update_extra` and returning a `SpeciesFilterResult` summary (full species breakdown + kept/dropped counts + kept record ids). `species_kept` is **report-only**: T09 (report) consumes it; T05 does NOT — T05 selects clustering inputs by the `species` column.

**Out of scope:** detection/cropping (T02), embedding (T04), clustering (T05), catalogue (T06), eval (T07), HITL (T08), Medvednica report assembly (T09), runner (T10). T03 never deletes rows and never modifies existing repo files except an additive line in `STATUS_BOARD.md`.

## Scope

### In
- `reid_demo/species_filter.py` with the public API in **Interface contract**.
- Tolerant target-species matching (`is_target_species` + `TARGET_SPECIES_ALIASES`) for at least `lynx`, `leopard`, `tiger`.
- Ingest-from-JSON path (primary; no model needed) and live-SpeciesNet path (secondary; reuses `deprecated/seminar_classify_species.py` pattern).
- `set_known_species()` for datasets with a known fixed species.
- Per-record keep flag in `extra_json["species_kept"]` (1/0); `species`/`species_conf`/`species_class` always written.
- `SpeciesFilterResult` dataclass + a `--info`-style CLI that prints the breakdown.
- `tests/test_species_filter.py` using a tiny in-repo fixture (and the real `animals_classified.json` for a smoke check).
- One additive line in `STATUS_BOARD.md` for T03.

### Out
- Re-running MegaDetector / re-cropping (that is T02's job; T03 reads existing crop rows).
- Hard-deleting non-target rows (T06/T09 need the full species breakdown — only flag, never delete).
- Changing the T01 schema. Use the existing `species`/`species_conf`/`species_class` columns and the `extra_json` escape hatch for the keep flag.
- Any embedding/clustering logic.

## Inputs

- A T01 SQLite store (default `data/reid_demo/reid_demo.sqlite`) already populated by T02 with one row per crop for the given `dataset` (rows have `record_id`, `source_image`, `source_stem`, `det_index`, `crop_path`, `bbox_*`, `timestamp`, `camera_id`; `species*` are NULL).
- For Medvednica: the precomputed SpeciesNet JSON `data/MedvednicaDS/animals_classified.json` (real fixture; do not regenerate).
- For LeopardID2022/ATRW: nothing beyond the known species name passed to `set_known_species`.
- Optional (live path only): an installed SpeciesNet CLI (`python -m speciesnet.scripts.run_model`, model `kaggle:google/speciesnet/pyTorch/v4.0.1a`) and a crop folder. T03 must degrade gracefully (clear error) when this is absent.

## Outputs

- The T01 `detections` rows for the dataset have `species`, `species_conf`, `species_class` populated, and `extra_json` updated with `"species_kept": 0|1` (existing `extra_json` keys preserved).
- A returned `SpeciesFilterResult` (see contract) summarizing counts + breakdown + kept record ids.
- Optional console summary via the CLI.
- No new files in `data/` are required; the live path may write a temp crop-predictions JSON which it cleans up (mirror `seminar_classify_species.py`).

## Interface contract

Downstream tickets (T05 picks clustering inputs; T09 reports the breakdown; T10 runs the stage) depend on EXACTLY these names. Import the store from T01 — do not duplicate its logic.

```python
# reid_demo/species_filter.py
from dataclasses import dataclass, field
from typing import Optional, Dict, List

# --- target-species policy -------------------------------------------------
TARGET_SPECIES_ALIASES: Dict[str, set]   # e.g. {"lynx": {"lynx","eurasian lynx","canada lynx",
                                         #                "iberian lynx","wild cat","cat family","bobcat"},
                                         #       "leopard": {"leopard","african leopard","amur leopard",
                                         #                   "snow leopard","panthera pardus"},
                                         #       "tiger": {"tiger","amur tiger","panthera tigris"}}

def is_target_species(species_name: str, target_species: str) -> bool:
    """True if the SpeciesNet common name (or a full taxonomy string) matches the target.
    `species_name` may be a bare common name ('eurasian lynx') OR a full
    'uuid;...;genus;species;common_name' string. Matching is case-insensitive and uses
    TARGET_SPECIES_ALIASES[target_species]; it also matches when the genus field of a
    full taxonomy string equals the target (e.g. genus 'lynx' for target 'lynx',
    genus/species 'panthera;pardus' for 'leopard'). `target_species` is a key of
    TARGET_SPECIES_ALIASES; raise KeyError with the list of valid targets otherwise."""

# --- result summary --------------------------------------------------------
@dataclass
class SpeciesFilterResult:
    dataset: str
    target_species: str
    n_classified: int                 # records with a non-null species after this run
    n_kept: int                       # species_kept == 1
    n_dropped: int                    # classified but species_kept == 0
    n_unclassified: int               # T01 rows in dataset left with species NULL
    skipped_unmatched: int            # JSON detections with no matching T01 record
    species_breakdown: Dict[str, int] # {common_name: count} over all classified rows in dataset
    kept_record_ids: List[str]        # record_ids with species_kept == 1

# --- ingest a precomputed SpeciesNet JSON (PRIMARY path, no model needed) ---
def ingest_speciesnet_json(
    conn,                             # sqlite3.Connection from store.connect()
    json_path: str,                   # e.g. "data/MedvednicaDS/animals_classified.json"
    *,
    dataset: str,                     # which T01 rows to update (e.g. "MedvednicaDS")
    target_species: str,              # key into TARGET_SPECIES_ALIASES (e.g. "lynx")
    keep_threshold: float = 0.0,      # min species_conf to keep a target-species crop
    drop_nontarget: bool = False,     # if True, exclude non-kept rows from kept_record_ids (never deletes rows)
    species_index: int = 0,           # which class to treat as 'top' (default best = 0)
) -> SpeciesFilterResult:
    """For each detection in the JSON, JOIN to a T01 record by (source_stem, bbox)
    NEAREST-MATCH: stem = Path(filepath).stem; among the dataset's T01 rows with the same
    source_stem, pick the row whose stored bbox is closest to the JSON detection's bbox
    (e.g. min IoU-complement or min L2 over (x,y,w,h); require the match to be within a
    small tolerance, else count as skipped_unmatched). Do NOT enumerate positionally and do
    NOT try index/index+1. Write species=classes[0].split(';')[-1], species_conf=scores[0],
    species_class=classes[0] via store.update_species, then set extra_json['species_kept']
    via store.update_extra(conn, record_id, 'species_kept', 0|1). Idempotent."""

# --- run SpeciesNet live on crops (SECONDARY path) -------------------------
def classify_and_filter(
    conn,
    *,
    dataset: str,
    target_species: str,
    keep_threshold: float = 0.0,
    country: str = "HRV",
    batch_size: int = 16,
    model: str = "kaggle:google/speciesnet/pyTorch/v4.0.1a",
    reuse_existing_json: Optional[str] = None,   # if given, delegates to ingest_speciesnet_json
    drop_nontarget: bool = False,
) -> SpeciesFilterResult:
    """End-to-end. If reuse_existing_json is set -> ingest_speciesnet_json(...).
    Else: gather crop_path for every T01 row in `dataset`, run the SpeciesNet CLI once on
    that crop folder (same invocation as deprecated/seminar_classify_species.py), stitch
    per-crop predictions back by crop filename -> record_id, write species fields + keep flag.
    Raise a clear RuntimeError naming SpeciesNet if the CLI/model is unavailable."""

# --- known-species shortcut (LeopardID2022/ATRW) ---------------------------
def set_known_species(
    conn,
    *,
    dataset: str,
    species: str,                     # e.g. "leopard" or "tiger"
    species_conf: float = 1.0,
) -> int:
    """Stamp a fixed species on EVERY T01 row of `dataset` (no model). This is the
    B-track species stage and is still run for LeopardID2022 ('leopard') / ATRW ('tiger')
    even though the SpeciesNet model branch is skipped (D7d). Sets species_class to a
    synthetic '<species>' string (or None) and species_kept=1 for all rows via
    store.update_species + store.update_extra. Does NOT touch gt_identity or orientation —
    T02 is the SOLE owner of those for labeled datasets (D1). Returns rows written."""
```

### Persistence rules (must hold exactly)

- `species` <- `classes[0].split(';')[-1]` (lower-cased as emitted by SpeciesNet); `species_conf` <- `scores[0]` (float); `species_class` <- raw `classes[0]` (full taxonomy string). Use `store.update_species(conn, record_id, species, species_conf, species_class)`.
- Keep flag: set key `"species_kept"` to `1` iff `is_target_species(species, target_species) and species_conf >= keep_threshold`, else `0`, via `store.update_extra(conn, record_id, "species_kept", 1|0)`. **Use `store.update_extra` — never raw SQL.** T01's `update_extra` merges the single key into the existing `extra_json` (preserving other keys) and refreshes `updated_at`; T03 does not parse/rewrite the blob itself.
- `blank` classifications: write `species="blank"`, `species_kept=0` (never a target). A detection that has no `classifications` block at all -> leave species NULL, count in `n_unclassified`.
- Never `DELETE` rows. `drop_nontarget` only affects what goes into `kept_record_ids`.

### CLI

```
python -m reid_demo.species_filter \
    --dataset MedvednicaDS --target lynx \
    --reuse-json data/MedvednicaDS/animals_classified.json \
    [--keep-threshold 0.0] [--drop-nontarget] [--db data/reid_demo/reid_demo.sqlite]
# prints: n_classified, n_kept, n_dropped, n_unclassified, skipped_unmatched,
#         and the top ~20 rows of species_breakdown sorted desc.

python -m reid_demo.species_filter --dataset LeopardID2022 --set-known leopard --db <path>
# stamps every LeopardID2022 row with species=leopard, species_kept=1.
```

## Existing code to reuse (real paths)

- `reid_demo/store.py` + `reid_demo/DATA_CONTRACT.md` (**T01**, the data contract) — use `connect`, `init_db`, `make_record_id`, `get_record`, `query_records`, `update_species`, `update_extra`, `count_by`, `DetectionRecord`, `COLUMNS`, `TABLE_NAME`. `species`/`species_conf`/`species_class` are columns 13–15; `extra_json` is the escape hatch for `species_kept`, written via `store.update_extra(conn, record_id, key, value)` (added to the T01 public store API per D3). **Read this doc; do not re-implement DB access; do not write `extra_json` with raw SQL.**
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/deprecated/seminar_classify_species.py` — the proven SpeciesNet glue: `save_crop(img_path, bbox, dest_dir, stem, idx)` crops a normalized bbox; the CLI invocation `python -m speciesnet.scripts.run_model --folders <crops> --predictions_json <out> --batch_size N --model kaggle:google/speciesnet/pyTorch/v4.0.1a --country HRV`; and the stitch loop that maps `crop_filename -> (frame_record, det_idx)` and copies `classifications.classes/scores` back. **Reuse the CLI command and the parallel-array stitch logic; do NOT reuse its file-JSON output structure — T03 writes to the T01 store instead.**
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/utils/m2dspeciesnet.py` — confirms the SpeciesNet predictions format (`predictions[i].detections[i].classifications.classes/scores`) and that `filepath` is `animal_images/<name>`.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/utils/clean_detections.py` — pattern for confidence/category filtering (reference only). T03's filter is species-based, not detector-conf, and T03 does **not** call or recompute from this script: for kept/empty counts T03 trusts the on-disk `data/MedvednicaDS/detections_cleaned.json` (D3). T02's per-detection `conf>=0.5` policy intentionally differs from this legacy whole-frame strict `conf>0.5` cleaner.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/data/MedvednicaDS/animals_classified.json` — the real precomputed SpeciesNet output to ingest in the primary path and in tests.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/constants.py` — follow its module-level path-constant style (`ROOT_DIR = os.path.dirname(os.path.abspath(__file__))`). Do not edit it.

## Implementation notes

- **Matching JSON detections to T01 rows (D3 — nearest-match, NOT positional).** Each JSON prediction has `filepath` like `animal_images/IMG_0066.JPG`; `source_stem = Path(filepath).stem` (`IMG_0066`). For each JSON detection, gather the dataset's T01 rows sharing that `source_stem` (e.g. `store.query_records(conn, dataset=dataset, source_stem=stem)`), then pick the row whose stored bbox (`bbox_x, bbox_y, bbox_w, bbox_h`) is NEAREST to the JSON detection's normalized `bbox=[x,y,w,h]` — e.g. minimum L2 distance over the four normalized coords (IoU-based is acceptable). Require the best match to be within a small tolerance (e.g. L2 < ~0.05, or positive IoU); if no row is within tolerance, count it in `skipped_unmatched` and skip. **Do NOT enumerate positionally, do NOT compute `det_index` from the JSON, and do NOT use any "try index, then index+1" heuristic** — `det_index` is T01's binding 1-based KEPT-ANIMAL ordering and is not reconstructable from the SpeciesNet JSON. Greedily consume matched rows (so two detections in one frame don't both claim the same row). Note: the per-detection conf>=0.5 set of rows T02 ingested may not be 1:1 with the raw JSON detections (T02's policy intentionally differs from the legacy cleaner — D3), so genuine unmatched detections in `skipped_unmatched` are expected and not an error.
- **Tolerant target matching.** Define `TARGET_SPECIES_ALIASES` as a module-level dict of lowercase common names per target. `is_target_species` should: (a) lowercase + strip the input; (b) if the input contains `;` treat it as a full taxonomy string — split, take `common_name=parts[-1]`, `genus=parts[4]` if present, `species_epithet=parts[5]` if present — and match if `common_name` in the alias set OR `genus` equals the target key (e.g. genus `lynx` for `lynx`, genus `panthera` + epithet `pardus` for `leopard`); (c) else match the bare common name against the alias set. Keep the lynx set deliberately a bit inclusive for the demo (lynx genus + `wild cat`/`cat family` borderline felids), but document that `bobcat` (genus `lynx`, species `rufus`) is a true lynx-genus felid and is INCLUDED by genus match — note this in a comment so a reviewer understands why bobcats are kept under `target="lynx"`. Make the alias sets easy to edit.
- **Always label, only flag.** Write `species`/`species_conf`/`species_class` for every classified crop regardless of target. The keep filter is a flag, not a delete — T09's Medvednica report needs the full "% empty removed + species breakdown", and T06's catalogue may show non-target groups. `species_kept` is **report-only** (D7d): T05 selects clustering inputs by the `species` column (e.g. `species` in the target alias set), NOT by `species_kept`. So the authoritative contract for downstream input selection is the `species` value; `species_kept` is a convenience flag for reporting/triage only.
- **species_breakdown** must be computed from the store after writing (use `store.count_by(conn, "species", dataset=dataset)` if it returns `{species: count}`; otherwise group via `query_records`). It should include `blank` and all non-target species so the report is honest.
- **extra_json safety.** Always write `species_kept` through `store.update_extra(conn, record_id, "species_kept", 1|0)` (D3). Do NOT parse/`json.dumps` the `extra_json` blob yourself and do NOT issue a raw `conn.execute` UPDATE — `update_extra` is responsible for merging the single key while preserving other keys and refreshing `updated_at`.
- **Live path robustness.** Build the crop folder from existing `crop_path`s (symlink or copy into a temp dir keyed by `record_id` so the stitch back is unambiguous — name temp crops `<record_id>.jpg` to avoid the legacy stem+idx ambiguity). Run the CLI exactly as in `seminar_classify_species.py`. Wrap the `subprocess.run` and any `import speciesnet` in try/except and raise `RuntimeError("SpeciesNet CLI/model not available: <detail>. Use --reuse-json with a precomputed predictions file, or install speciesnet.")`. The demo machine may have no GPU; the **primary** path (reuse JSON) and `set_known_species` must work with zero SpeciesNet dependency.
- **set_known_species** is a pure-store operation: `query_records(dataset=...)`, then `update_species(conn, rid, species, species_conf, species_class=species)` and `store.update_extra(conn, rid, "species_kept", 1)` for each. This is the B-track species stage (D7d) — still run for LeopardID2022 (`leopard`) and ATRW (`tiger`) even though the SpeciesNet model branch is skipped — and gives those datasets a uniform `species` column for T05/T06/T07 without a classifier. Confirmed that LeopardID2022's WildlifeReID-10k `species` value is the single string `leopard`. It does NOT set/overwrite `gt_identity` or `orientation`: per D1 those are populated by T02's `ingest_wildlife_dataset` from the metadata and T02 is their SOLE owner.
- **Idempotency.** Re-running any path must yield identical species values and not duplicate rows (the store upserts by `record_id`; T03 only updates existing rows). Tests must assert a second run is a no-op on counts.
- **No new deps.** Stdlib only (`json`, `pathlib`, `subprocess`, `tempfile`, `shutil`, `argparse`, `dataclasses`). `PIL`/`pandas` only if you touch the live crop path (already in the venv per `seminar_classify_species.py`); guard imports so the primary path doesn't require them.
- Add one additive line to `STATUS_BOARD.md` marking T03 deliverables; do not edit other tickets' status.

## Acceptance criteria

- [ ] `reid_demo/species_filter.py` and `tests/test_species_filter.py` exist; no existing repo file is modified except an additive line in `STATUS_BOARD.md`.
- [ ] `python -c "from reid_demo.species_filter import classify_and_filter, ingest_speciesnet_json, set_known_species, is_target_species, SpeciesFilterResult, TARGET_SPECIES_ALIASES"` succeeds.
- [ ] `is_target_species` matches by common name AND by full-taxonomy genus: `is_target_species("eurasian lynx","lynx") is True`; `is_target_species("canada lynx","lynx") is True`; `is_target_species("d372cda5-...;mammalia;carnivora;felidae;lynx;rufus;bobcat","lynx") is True` (genus `lynx`); `is_target_species("wild boar","lynx") is False`; `is_target_species("leopard","leopard") is True`; `is_target_species("...;panthera;pardus;leopard","leopard") is True`; passing an unknown target raises `KeyError`.
- [ ] `ingest_speciesnet_json` on a tiny fixture (T01 rows + matching SpeciesNet JSON) joins each detection to the correct record by `(source_stem, bbox)` nearest-match (NOT positional index), writes `species=classes[0].split(';')[-1]`, `species_conf=scores[0]`, `species_class=classes[0]`, and sets `extra_json["species_kept"]` per the keep rule via `store.update_extra` (preserving any pre-existing `extra_json` keys; no raw SQL).
- [ ] The join is bbox-driven: a fixture where two detections in one frame are stored in non-positional bbox order is still matched correctly to the nearest-bbox record (a positional/enumerate or index+1 join would mis-assign them).
- [ ] Detections in the JSON with no matching T01 record (no row within bbox tolerance for that stem) are counted in `skipped_unmatched` and do not crash; T01 rows with no JSON classification stay `species=NULL` and are counted in `n_unclassified`.
- [ ] `n_kept == |{records with species_kept==1}|`, `n_kept + n_dropped == n_classified`, and `species_breakdown` is `{common_name: count}` over all classified rows in the dataset (includes `blank` and non-target species).
- [ ] `keep_threshold` is honored: a target-species crop with `species_conf < keep_threshold` is dropped (`species_kept=0`) while still being labeled.
- [ ] No rows are ever deleted; `drop_nontarget=True` only removes non-kept ids from `kept_record_ids`.
- [ ] `set_known_species(conn, dataset="LeopardID2022", species="leopard")` stamps every LeopardID2022 row with `species="leopard"`, `species_kept=1`, returns the row count, and requires no SpeciesNet.
- [ ] Re-running `ingest_speciesnet_json` is idempotent: identical species values, unchanged row count.
- [ ] `classify_and_filter(reuse_existing_json="data/MedvednicaDS/animals_classified.json", dataset="MedvednicaDS", target_species="lynx")` on a DB seeded from that JSON's detections runs with NO GPU/model and yields a `species_breakdown` containing at least `"wild boar"` and `"european roe deer"`.
- [ ] The live path (no `reuse_existing_json`) raises a clear `RuntimeError` naming SpeciesNet when the CLI/model is unavailable, instead of a raw traceback.
- [ ] CLI `python -m reid_demo.species_filter --dataset MedvednicaDS --target lynx --reuse-json data/MedvednicaDS/animals_classified.json --db <tmp>` exits 0 and prints the counts + top species_breakdown rows.
- [ ] `tests/test_species_filter.py` passes under `venv/bin/python -m pytest`.

## How to verify

```bash
cd /home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project
source venv/bin/activate    # or use venv/bin/python directly

# 0. Contract surface
python -c "from reid_demo.species_filter import classify_and_filter, ingest_speciesnet_json, set_known_species, is_target_species, SpeciesFilterResult, TARGET_SPECIES_ALIASES; print('imports OK')"

# 1. Target matching policy
python - <<'PY'
from reid_demo.species_filter import is_target_species
assert is_target_species("eurasian lynx","lynx")
assert is_target_species("canada lynx","lynx")
assert is_target_species("ba76d46e-...;mammalia;carnivora;felidae;lynx;rufus;bobcat","lynx")  # genus lynx
assert not is_target_species("wild boar","lynx")
assert is_target_species("leopard","leopard")
assert is_target_species("uuid;mammalia;carnivora;felidae;panthera;pardus;leopard","leopard")
print("target matching OK")
PY

# 2. End-to-end on the REAL Medvednica JSON (no GPU/model needed).
#    NOTE: this assumes T02 has seeded the store for MedvednicaDS. If running T03 in
#    isolation, the test suite builds a small seeded DB from the JSON's detections itself.
python - <<'PY'
from reid_demo.store import connect, count_by, query_records
from reid_demo.species_filter import classify_and_filter
conn = connect("/tmp/reid_t03.sqlite")
# (test fixture / T02 must have populated MedvednicaDS rows; see tests/test_species_filter.py)
res = classify_and_filter(conn, dataset="MedvednicaDS", target_species="lynx",
                          reuse_existing_json="data/MedvednicaDS/animals_classified.json")
print("classified:", res.n_classified, "kept:", res.n_kept, "dropped:", res.n_dropped,
      "unmatched:", res.skipped_unmatched)
top = sorted(res.species_breakdown.items(), key=lambda kv: -kv[1])[:10]
print("top species:", top)
assert any(k=="wild boar" for k,_ in top)
PY

# 3. Known-species shortcut (no model)
python - <<'PY'
from reid_demo.store import connect, query_records
from reid_demo.species_filter import set_known_species
conn = connect("/tmp/reid_t03_leopard.sqlite")
# (tests seed a couple of LeopardID2022 rows first)
n = set_known_species(conn, dataset="LeopardID2022", species="leopard")
print("stamped rows:", n)
PY

# 4. CLI
python -m reid_demo.species_filter --dataset MedvednicaDS --target lynx \
    --reuse-json data/MedvednicaDS/animals_classified.json --db /tmp/reid_t03_cli.sqlite ; echo "exit=$?"

# 5. Tests
python -m pytest tests/test_species_filter.py -q
```

## Open questions

1. **`det_index` base / JSON join — RESOLVED (D3).** T03 does NOT reconstruct `det_index` from the SpeciesNet JSON and does NOT use any index/index+1 heuristic. It joins each JSON detection to a T01 record by `(source_stem, bbox)` nearest-match (see Implementation notes). `det_index` is T01's binding 1-based ordering over KEPT ANIMAL detections in MegaDetector source-file order and is owned by T02/T01; T03 only reads bbox + source_stem. Genuine unmatched detections go to `skipped_unmatched`.
2. **Lynx alias breadth.** For the demo, should `target="lynx"` include borderline felids (`wild cat`, `cat family`, `bobcat` which is lynx-genus)? Proposed: include all lynx-genus felids by taxonomy and treat `wild cat`/`cat family` as configurable. The biologist reviews low-confidence keeps in T08, so erring inclusive is acceptable. Flag if T05/T07 prefer a stricter set.
3. **Where the keep flag lives — RESOLVED.** `extra_json["species_kept"]`, written via `store.update_extra` (no schema change, per the T01 escape-hatch design + D3). It is **report-only** (D7d): consumed by T09 for the Medvednica report. T05 does NOT read it — T05 selects clustering inputs by the `species` column. No dedicated column / schema bump is needed.
4. **ATRW species string.** `set_known_species` for ATRW uses `species="tiger"`; confirm T07/T06 expect `"tiger"` (vs `"amur tiger"`) so the alias set and catalogue labels line up.
