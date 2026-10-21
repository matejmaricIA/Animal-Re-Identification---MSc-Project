# T09 — Medvednica filtering report

> **Status:** 🔵 In review · **Milestone:** M1 (Demo-ready)
> **Depends on:** T01 · **Blocks:** T10
> **Owner:** _unassigned_
>
> Self-contained ticket for the lynx re-identification demo/pilot. See [STATUS_BOARD.md](../STATUS_BOARD.md) for the full plan, the binding design decisions, and the agent working protocol.

---

## Context

We are building a DEMO + PILOT MVP of an automated individual-animal re-identification system for Eurasian lynx, to cold-pitch Croatian national parks. The demo has **two honest parts**:

- **(A) REAL LOCAL DATA** — the park's *own* footage already processed: prove we can take a raw pile of camera-trap frames, throw away the empties/people/vehicles, and tell them what species are in what's left.
- **(B) INDIVIDUAL-ID CAPABILITY** — proven on public spotted-cat data (LeopardID2022, ATRW tigers) via the clustering pipeline (tickets T02–T08, T10).

**This ticket (T09) owns Part (A).** It is the "look, this works on YOUR cameras" slide of the pitch. It turns the **already-computed** Medvednica detection + species-classification JSONs into a **presentable, non-technical summary**: total frames in, % *empty* frames automatically removed (frames with ZERO detections of any kind), persons/vehicles removed (reported as a *separate* category, NOT lumped into "empty"), a species breakdown of what's left, and a strip of example animal crops. The audience is a **park biologist, not an ML engineer** — every number must be expressed in plain language (frames, animals, species), not ML jargon.

> **Empty vs people/vehicle (binding, per design decision D7b).** "Empty frames" means **frames with ZERO MegaDetector detections of ANY category** (animal/person/vehicle) — i.e. `total_frames − frames_with_any_detection = 8208 − 6177 = 2031`. Frames whose ONLY detections are person/vehicle are **NOT** "empty"; they are a **separate** removed category. The headline prose and every asserted number MUST keep these two buckets distinct: the report says "N empty frames" using `empty_frames=2031`, and separately "M photos of people/vehicles" using the distinct frame count `person_or_vehicle_frames` — never conflate them.

This is deliberately a **low-risk, "mostly assembling data that already exists"** ticket. Almost all the hard work (MegaDetector + SpeciesNet) was run previously; the artifacts sit on disk under `data/MedvednicaDS/`. Your job is to read those artifacts, compute the funnel/counts, and render a clean report (a Markdown page + a couple of PNG figures + a montage of example crops + a machine-readable JSON of the numbers).

### Where this sits in the larger demo

The full demo is a constellation of independent tickets (T01–T10) that share one data contract defined by **T01** (`reid_demo/store.py`, schema doc `reid_demo/DATA_CONTRACT.md`). T09 **depends only on T01**. T09 is a **READ-ONLY consumer** of the shared store — it never writes detection records. However, because the Medvednica artifacts were produced by an *earlier* run (not necessarily by T02/T03 into the store), the **authoritative source for the funnel numbers is the raw `data/MedvednicaDS/*.json` files**, which always exist. The store is an *optional secondary* enrichment: if a `MedvednicaDS` run has been ingested into the T01 SQLite store, you may additionally pull per-crop rows from it for cross-checking, but the report MUST work end-to-end from the JSONs alone with the store absent.

The final assembler ticket (T10) will call your CLI/function and drop your output page + figures into the pitch bundle alongside the catalogue (T06) and accuracy numbers (T07). So your outputs must be self-contained, path-stable, and re-runnable.

### Ground-truth shape of the real artifacts (verified on disk)

Under `data/MedvednicaDS/`:

- `megadetector_results.json` — the **raw, unfiltered** MegaDetector output. Top-level keys `{"images": [...], "detection_categories": {"1":"animal","2":"person","3":"vehicle"}, "info": {...}}`. Each image record: `{"file": "Camera 1/IMG_0001.JPG", "detections": [ {"category":"1", "conf":0.78, "bbox":[x,y,w,h]}, ... ]}`. `bbox` is **normalized** `[x,y,w,h]` in `[0,1]`. **8208 images total**; 5801 have ≥1 animal detection; detection-category tally is `animal: 11795, person: 876, vehicle: 4`. This file is the ground truth for "total frames in" and "what got removed".
- `detections_cleaned.json` — same `predictions` list format, **filtered to `conf >= 0.5` AND `category == "1"` (animal)**. Top-level keys `{"detection_categories", "info", "predictions"}`; record `{"filepath": "animal_images/IMG_0066.JPG", "detections": [ {"category":"1","conf":0.784,"bbox":[...]} ]}`. **1866 records have ≥1 kept detection; 2049 kept detections total.** Per design decision **D3, this on-disk file is the authoritative source for the "kept" animal counts — TRUST IT; do NOT recompute the kept set by re-running `utils/clean_detections.py`.** Note that the per-detection `conf >= 0.5` policy that produced this file (and that T02 applies per-detection) **intentionally differs** from any legacy whole-frame strict `conf > 0.5` cleaner — so the kept count is whatever this file says, not what a re-derivation might yield.
- `animals_classified.json` — full SpeciesNet output. Top-level `{"predictions": [...]}` (8208 records, most with empty `detections`). A classified detection adds `"classifications": {"classes": ["uuid;mammalia;...;wild boar", ...], "scores": [0.87, ...]}`. The human-readable species is `classes[0].split(";")[-1]`. Records carrying animals also carry a top-level `"timestamp": "2025-06-02 04:27:51"`. **2049 classified detections.** Verified top species (by detection): `wild boar 638, blank 445, european roe deer 349, red fox 101, white-tailed deer 64, bird 44, eurasian badger 41, european hare 40, ...`. NOTE: `"blank"` is a legitimate SpeciesNet class meaning "no identifiable animal" — treat it as a non-species and report it separately (it is part of the story: the classifier itself rejects some crops).
- `animal_crops/` — **4194** pre-cut crop JPGs. Naming: `{image_stem}_crop{idx}_conf{int_conf_percent}.jpg`, e.g. `02020401_crop1_conf92.jpg` and `IMG_0066_crop1_conf78.jpg`. Two stem styles coexist (`02020401` and `IMG_0066`); both map 1:1 to `animal_images/{stem}.JPG`.
- `animal_images/` — **3333** full frames that survived to the animal stage (the kept subset's source frames).
- `trail_cam_data.csv` — header `filepath,camera,num_detections,datetime,temperature`. `camera` is `unknown_camera` throughout this dump; `datetime` like `2025-06-02 04:27:49` is the timestamp source. Use it to report the date span of the footage and (if cameras were ever populated) a per-camera tally.
- `visualizations/` — 3333 `vis_{stem}.JPG` frames with boxes drawn (optional, nice-to-have source of example imagery).

Because the two stem conventions and the `conf` suffix exist on disk, your crop-name parser must handle `{stem}_crop{idx}_conf{NN}.jpg` robustly (strip the trailing `_conf\d+` to recover idx, and everything before `_crop\d+` to recover the stem).

## Objective

Produce a **self-contained, re-runnable report generator** that reads the existing `data/MedvednicaDS/` artifacts and emits, into a single output directory:

1. A **non-technical Markdown summary page** (`medvednica_report.md`) telling the funnel story in plain language.
2. **PNG figures**: a detection-funnel bar chart and a species-breakdown bar chart.
3. A **montage / contact sheet** of N example animal crops (with species + confidence captions where available).
4. A **machine-readable `medvednica_summary.json`** holding every number the page cites (so T10 can re-use the figures verbatim).

The deliverable is a CLI `python -m reid_demo.medvednica_report ...` plus an importable function, both writing the same outputs deterministically.

## Scope

### In
- A new module `reid_demo/medvednica_report.py` (lives inside the T01 package so it can import the store contract).
- Reading `megadetector_results.json`, `detections_cleaned.json`, `animals_classified.json`, `trail_cam_data.csv` and computing the funnel + species + temporal numbers.
- Rendering: Markdown page, 2 matplotlib PNG figures, 1 crop montage (matplotlib or PIL), 1 summary JSON.
- A small, well-named set of pure helper functions (counting/parsing) that are unit-testable without rendering.
- Optional read-only enrichment from the T01 store **if** a `MedvednicaDS` run is present (guarded; never required).
- A `--selftest`/smoke path and unit tests under `tests/test_medvednica_report.py`.
- One additive line in `STATUS_BOARD.md` marking T09 deliverables.

### Out
- Re-running MegaDetector, SpeciesNet, cropping, or any model inference (everything is already on disk — do NOT call any detector/classifier).
- Writing detection records into the T01 store (T09 is read-only on the store).
- Individual re-identification / clustering / catalogue (that is T05/T06).
- The LeopardID2022/ATRW accuracy numbers (that is T07).
- A web dashboard, interactivity, geolocation maps, or population density (Phase 2).
- Editing any existing repo file except adding one line to `STATUS_BOARD.md` and adding new files under `reid_demo/`, `tests/`.

## Inputs

All paths relative to repo root `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project`:

- `data/MedvednicaDS/megadetector_results.json` (REQUIRED — funnel "total in" + removed counts).
- `data/MedvednicaDS/detections_cleaned.json` (REQUIRED — kept animal detections after conf≥0.5 + category filter).
- `data/MedvednicaDS/animals_classified.json` (REQUIRED — species labels + scores + timestamps).
- `data/MedvednicaDS/animal_crops/` (REQUIRED — example crops for the montage).
- `data/MedvednicaDS/trail_cam_data.csv` (OPTIONAL — date span + per-camera tally; degrade gracefully if missing).
- `data/MedvednicaDS/animal_images/`, `data/MedvednicaDS/visualizations/` (OPTIONAL — alternate example imagery).
- T01 store at `data/reid_demo/reid_demo.sqlite` (OPTIONAL — only consulted if `--use-store` is passed AND it contains `dataset="MedvednicaDS"` rows).

Make the input directory configurable via `--data-dir` (default `data/MedvednicaDS`) so the same code can run on a future second park's dump.

## Outputs

Written to an output directory (`--out-dir`, default `Output/medvednica_report/`, created on demand):

- `medvednica_report.md` — the human-readable pitch page (see Interface contract for required sections).
- `figures/detection_funnel.png` — bar chart of the funnel stages.
- `figures/species_breakdown.png` — bar chart of species counts among kept animal detections (top-K + "other").
- `figures/example_crops.png` — montage of example animal crops with species/conf captions.
- `medvednica_summary.json` — the machine-readable numbers (schema below). This is the contract T10 consumes.

All figures referenced from the Markdown by **relative** paths (`figures/...`) so the folder is portable.

## Interface contract

Downstream (T10) depends on the function signature, the CLI, and the `medvednica_summary.json` schema. Do not rename these.

### Importable function

```python
def generate_medvednica_report(
    data_dir: str = "data/MedvednicaDS",
    out_dir: str = "Output/medvednica_report",
    *,
    top_k_species: int = 12,
    n_example_crops: int = 12,
    species_filter: list[str] | None = None,   # e.g. ["eurasian lynx"]; None = all real species
    use_store: bool = False,
    db_path: str | None = None,                # defaults to reid_demo.store.DEFAULT_DB_PATH when use_store
    seed: int = 0,                             # deterministic crop sampling
) -> dict:
    """Read the Medvednica artifacts under data_dir, compute the funnel + species
    + temporal summary, render medvednica_report.md, the three PNG figures, and
    medvednica_summary.json into out_dir. Returns the summary dict (== contents of
    medvednica_summary.json). Pure-read on inputs; never mutates data_dir or the store."""
```

### Pure helpers (must exist, unit-tested; exact names)

```python
def parse_crop_filename(name: str) -> tuple[str, int, int | None]:
    """'02020401_crop1_conf92.jpg' -> ('02020401', 1, 92).
    'IMG_0066_crop1_conf78.jpg'  -> ('IMG_0066', 1, 78).
    Returns (source_stem, crop_index, conf_percent_or_None). Tolerates a missing _conf suffix."""

def species_from_classes(classes: list[str]) -> str:
    """Return classes[0].split(';')[-1] (human-readable common name), '' if empty.
    Single source of truth for the taxonomy-string -> common-name rule."""

def compute_funnel(md_results: dict, cleaned: dict) -> dict:
    """Compute the detection funnel from the RAW megadetector_results.json (md_results)
    and detections_cleaned.json (cleaned). Returns the 'funnel' sub-dict of the summary
    schema below (total_frames, frames_with_any_detection, frames_with_animal,
    empty_frames, pct_empty_removed, person_detections, vehicle_detections,
    person_or_vehicle_frames, animal_detections_raw, animal_detections_kept,
    kept_frames, ...).

    Per D7b: empty_frames = total_frames - frames_with_any_detection (frames with ZERO
    detections of ANY category). person_or_vehicle_frames is the SEPARATE count of frames
    whose detections include a person/vehicle (category 2/3) but that are NOT empty; it is
    NOT folded into empty_frames. animal_detections_kept/kept_frames are read straight from
    `cleaned` (the on-disk detections_cleaned.json) per D3 — NOT recomputed from raw."""

def compute_species_counts(classified: dict, *, include_blank: bool = False) -> dict:
    """From animals_classified.json, return {common_name: count} over classified
    detections. 'blank' is excluded from the real-species count unless include_blank."""
```

(You MAY add more private helpers; the four above are the contracted, tested surface.)

### CLI

```
python -m reid_demo.medvednica_report \
    [--data-dir data/MedvednicaDS] [--out-dir Output/medvednica_report] \
    [--top-k-species 12] [--n-example-crops 12] [--species-filter "eurasian lynx"] \
    [--use-store] [--db <path>] [--seed 0]

python -m reid_demo.medvednica_report --selftest   # runs on data/MedvednicaDS, asserts core numbers > 0, exits non-zero on failure
```

`--species-filter` accepts a comma-separated list. When set, the montage and a "target-species" callout in the page focus on those species (lets the same tool produce a "lynx-only" slide if any lynx are present); the full funnel + full species chart are still produced.

### `medvednica_summary.json` schema (the T10-facing contract)

```jsonc
{
  "schema_version": 1,
  "dataset": "MedvednicaDS",
  "generated_at": "2026-06-09T12:00:00",          // ISO, store-style
  "data_dir": "data/MedvednicaDS",
  "funnel": {
    "total_frames": 8208,
    "frames_with_any_detection": 6177,
    "empty_frames": 2031,                          // ZERO detections of ANY category = total_frames - frames_with_any_detection (D7b)
    "pct_empty_removed": 24.7,                      // empty_frames / total_frames * 100, 1 dp
    "frames_with_animal": 5801,
    "person_detections": 876,                       // category==2 DETECTIONS in raw MD
    "vehicle_detections": 4,                        // category==3 DETECTIONS in raw MD
    "person_or_vehicle_frames": 0,                  // SEPARATE (D7b): non-empty frames whose detections are person/vehicle (computed from data; reported apart from empty_frames)
    "animal_detections_raw": 11795,                // all category==1 in raw MD
    "animal_detections_kept": 2049,                // from on-disk detections_cleaned.json (D3 — trusted, not recomputed)
    "kept_frames": 1866                            // records in detections_cleaned.json with >=1 kept animal detection (D3)
  },
  "species": {
    "total_classified_detections": 2049,
    "blank_detections": 445,                        // SpeciesNet 'blank' (reported separately)
    "real_species_detections": 1604,               // classified minus blank
    "n_distinct_species": 18,
    "counts": { "wild boar": 638, "european roe deer": 349, "red fox": 101, "...": 0 },
    "top_k": [ {"species": "wild boar", "count": 638, "pct": 39.8}, ... ]   // pct of real_species_detections
  },
  "temporal": {
    "date_min": "2025-06-02",
    "date_max": "2025-06-30",
    "n_dated_records": 3333,
    "cameras": { "unknown_camera": 3333 }          // {} if CSV missing
  },
  "target_species": {                              // present only when --species-filter given
    "filter": ["eurasian lynx"],
    "detections": 0,
    "frames": 0
  },
  "examples": [
    {"crop_path": "data/MedvednicaDS/animal_crops/02020401_crop1_conf92.jpg",
     "source_stem": "02020401", "species": "european roe deer", "species_conf": 0.91, "detector_conf": 0.92}
  ],
  "figures": {
    "funnel": "figures/detection_funnel.png",
    "species": "figures/species_breakdown.png",
    "examples": "figures/example_crops.png"
  },
  "report_md": "medvednica_report.md",
  "notes": [ "All numbers computed from data/MedvednicaDS artifacts; no models re-run." ]
}
```

Numbers in the example are the **verified true values** on the current data — your code must reproduce them (within the exact-match criteria below). Keys are the contract; values come from the data.

### `medvednica_report.md` required sections (in order)

1. **Title + one-sentence framing** ("Automated triage of Medvednica camera-trap footage").
2. **The headline** — a plain-language paragraph that keeps **empty** and **people/vehicle** distinct (D7b): "Of **8,208** photos, the system automatically discarded **N empty frames (X%)** — frames with nothing in them at all — and separately set aside **M photos of people/vehicles**, leaving **K photos containing animals**, which it sorted into **S species**." Here `N` = `funnel.empty_frames` (2031), `X` = `funnel.pct_empty_removed`, `M` = `funnel.person_or_vehicle_frames` (the SEPARATE frame count, NOT `person_detections+vehicle_detections` and NOT folded into the empty count), `K` = `funnel.frames_with_animal`, `S` = `species.n_distinct_species`. Every number comes from the summary dict, never hard-coded, and the empty vs people/vehicle wording must match the asserted fields so prose and numbers agree.
3. **Detection funnel** — embeds `figures/detection_funnel.png` + a small table.
4. **What species were found** — embeds `figures/species_breakdown.png` + a top-K table with counts and %; one line on `blank` ("the classifier itself rejected B crops as having no identifiable animal").
5. **Example detections** — embeds `figures/example_crops.png`.
6. **When the footage was taken** — date span + camera note.
7. **(If `--species-filter`)** a short "Target species: lynx" callout with its count (honest even when zero: "No lynx in this particular sample — the same pipeline detects and IDs them once they appear / on park footage that includes them").
8. **Method footnote** — one line: MegaDetector for detection, SpeciesNet for species, thresholds used (conf ≥ 0.5, animal category only). No ML metrics.

## Existing code to reuse (real paths)

- `data/MedvednicaDS/megadetector_results.json`, `detections_cleaned.json`, `animals_classified.json`, `trail_cam_data.csv`, `animal_crops/` — the inputs (described above).
- `reid_demo/store.py` (from T01) — import `from reid_demo.store import DEFAULT_DB_PATH, connect, count_by, query_records` for the **optional** `--use-store` enrichment path only. Read `reid_demo/DATA_CONTRACT.md` (T01) for the record schema. **Do not write to the store.** The species/funnel rule `classes[0].split(';')[-1]` mirrors how the store's `species` column is populated by T03, so numbers stay consistent.
- `utils/clean_detections.py` — the canonical filter that produced `detections_cleaned.json` (removes `conf < 0.5` OR `category != "1"`). **Reference only** — per D3 do NOT re-run it and do NOT recompute the kept count from it; your funnel's "kept" numbers come **straight from the on-disk `detections_cleaned.json`**. (Read it only if you need to understand provenance; the on-disk file is authoritative and its per-detection conf≥0.5 policy intentionally differs from any legacy whole-frame strict conf>0.5 cleaner.)
- `utils/m2dspeciesnet.py` / `deprecated/seminar_classify_species.py` — reference for how `animals_classified.json`'s `classifications.classes/scores` and the per-image `timestamp` were produced (the `classes[k].split(';')[-1]` common-name rule lives here). Reference only.
- `utils/analyze_classification_results.py` and `utils/generate_visualizations.py` — existing matplotlib/seaborn/pandas reporting code in this repo; follow their plotting/style idioms (figure sizing, `tight_layout`, saving PNGs) so the new figures look consistent. Reference only; do not import their hard-coded paths.
- `utils/make_classification_pipeline_assets.py` — example of assembling image montages (uses `cv2`); a montage reference. You may use PIL or matplotlib instead — pick the lightest path.

Confirmed available in the repo venv (`source venv/bin/activate`): `matplotlib 3.10.3`, `pandas 2.3.0`, `Pillow`, `numpy`. Prefer these; do not add new dependencies.

## Implementation notes

- **Funnel arithmetic, exactly (D7b empty-vs-people/vehicle + D3 trust-on-disk):**
  - `total_frames` = `len(md_results["images"])`.
  - `frames_with_any_detection` = images with **non-empty `detections`** (any category counts).
  - `empty_frames` = `total_frames - frames_with_any_detection` (= frames with **ZERO** detections of **ANY** category, 2031); `pct_empty_removed` = `round(empty_frames/total_frames*100, 1)`.
  - `frames_with_animal` = images with ≥1 detection whose `category == "1"`.
  - `person_detections` / `vehicle_detections` = count of **detections** with `category == "2"` / `"3"` (detection-level, can exceed the frame count).
  - `person_or_vehicle_frames` (D7b, the SEPARATE bucket) = count of images that are **NOT empty** (have ≥1 detection) AND have **no** `category == "1"` detection AND have ≥1 `category in {"2","3"}` detection. This is a **frame** count, kept strictly distinct from `empty_frames`; do NOT add it into `empty_frames` and do NOT confuse it with the detection-level `person_detections+vehicle_detections`. (Frames holding both an animal and a person/vehicle count under `frames_with_animal`, not here.)
  - `animal_detections_raw` = count of `category == "1"` detections in the raw MD file.
  - `animal_detections_kept` = **total detections in the on-disk `detections_cleaned.json`** (already filtered to conf≥0.5 + animal); `kept_frames` = cleaned records with ≥1 detection. Per **D3, TRUST these on-disk values — do NOT recompute the kept set from `utils/clean_detections.py` or re-filter the raw MD yourself.** The per-detection `conf >= 0.5` policy that produced this file intentionally differs from any legacy whole-frame strict `conf > 0.5` cleaner, so a re-derivation could disagree; the on-disk file wins.
- **Species counting:** iterate `animals_classified.json["predictions"][*]["detections"][*]["classifications"]`; take `species_from_classes(classes)` and the parallel `scores[0]` as `species_conf`. Tally into a Counter. `"blank"` → `blank_detections`, everything else → real species. `n_distinct_species` excludes `blank`. `top_k` percentages are over `real_species_detections`, capped at `top_k_species` with the remainder folded into `"other"` in the chart only (not in `counts`).
- **Temporal:** prefer the per-record `timestamp` in `animals_classified.json`; fall back to `trail_cam_data.csv`'s `datetime`. Parse dates leniently; `date_min`/`date_max` are `YYYY-MM-DD`. If a CSV `camera` column exists, tally it into `cameras` (here it is all `unknown_camera`); if the CSV is missing, `cameras = {}` and `n_dated_records` comes from JSON timestamps.
- **Example crops:** deterministically sample `n_example_crops` files from `animal_crops/` (seed the RNG with `seed`). For each, `parse_crop_filename` to get `source_stem` + index, then look up its species/conf by joining `source_stem` back to `animals_classified.json` (match on `filepath` basename stem). Caption each crop `"{species} ({conf:.0%})"` when known, else the stem. When `--species-filter` is set, prefer crops whose species is in the filter; if none match, fall back to any crops and note it. Skip unreadable/missing image files gracefully (don't crash the montage).
- **Figures:** keep them clean and legible for a non-technical viewer — readable axis labels in plain words ("photos", "animals"), value labels on bars, no chart-junk. Save at ~150 dpi, reasonable size (e.g. 8×5 in). Use a non-interactive backend (`matplotlib.use("Agg")`) so it runs headless.
- **Markdown:** generate purely from the summary dict (single render function) so the page and JSON can never disagree. All embedded image links use `figures/...` relative paths.
- **Determinism:** same inputs + same seed ⇒ byte-identical JSON (sort dict keys, round floats consistently) and the same sampled crops. This lets T10 cache/diff outputs.
- **Robustness / honesty:** if an input JSON is missing, raise a clear `FileNotFoundError` naming the file (don't silently emit a hollow report). If `animal_crops/` is empty, still produce the funnel + species sections and a placeholder note instead of crashing the montage.
- **No store dependency at runtime by default:** the module must import and run with `use_store=False` even if `data/reid_demo/reid_demo.sqlite` does not exist. Only touch `reid_demo.store` inside the `if use_store:` branch, importing lazily.
- Add one line to `STATUS_BOARD.md` (create it if absent) under a "T09" entry listing the deliverables; do not author other tickets' status.

## Acceptance criteria

- [ ] New files exist and no existing repo file is modified except an additive line in `STATUS_BOARD.md`: `reid_demo/medvednica_report.py`, `tests/test_medvednica_report.py`.
- [ ] `python -c "from reid_demo.medvednica_report import generate_medvednica_report, parse_crop_filename, species_from_classes, compute_funnel, compute_species_counts"` succeeds (all contracted names importable).
- [ ] `parse_crop_filename("02020401_crop1_conf92.jpg") == ("02020401", 1, 92)` and `parse_crop_filename("IMG_0066_crop1_conf78.jpg") == ("IMG_0066", 1, 78)` and a name without `_conf` yields a `None` third element without raising.
- [ ] `species_from_classes(["uuid;mammalia;cetartiodactyla;suidae;sus;scrofa;wild boar"]) == "wild boar"` and `species_from_classes([]) == ""`.
- [ ] Running `generate_medvednica_report("data/MedvednicaDS", "<tmp>")` on the real data returns a dict and writes `medvednica_report.md`, `figures/detection_funnel.png`, `figures/species_breakdown.png`, `figures/example_crops.png`, `medvednica_summary.json` into `<tmp>` (all five exist and are non-empty).
- [ ] The returned/written summary reproduces the verified funnel numbers from the current data: `funnel.total_frames == 8208`, `funnel.frames_with_any_detection == 6177`, `funnel.frames_with_animal == 5801`, `funnel.person_detections == 876`, `funnel.vehicle_detections == 4`, `funnel.animal_detections_raw == 11795`, `funnel.animal_detections_kept == 2049`, `funnel.kept_frames == 1866`, `funnel.empty_frames == 2031`, and `funnel.pct_empty_removed == round(2031/8208*100,1)`.
- [ ] **D7b empty-vs-people/vehicle separation:** `funnel.empty_frames == total_frames - frames_with_any_detection` (frames with ZERO detections of any category), and `funnel` ALSO carries an integer `person_or_vehicle_frames` (non-empty frames whose detections are person/vehicle only, no animal) computed from the data and kept SEPARATE — `person_or_vehicle_frames` is NOT added into `empty_frames`. The headline prose uses `empty_frames` for "empty" and `person_or_vehicle_frames` for "people/vehicles", so the asserted numbers and the prose agree (the page never describes person/vehicle frames as "empty").
- [ ] **D3 trust-on-disk:** `funnel.animal_detections_kept` and `funnel.kept_frames` are taken directly from `detections_cleaned.json` on disk (not recomputed by re-running `utils/clean_detections.py` or re-filtering the raw MD); they equal 2049 and 1866 respectively as that file states.
- [ ] `species.total_classified_detections == 2049`, `species.blank_detections == 445`, `species.real_species_detections == 1604`, and `species.counts["wild boar"] == 638`, `species.counts["european roe deer"] == 349`, `species.counts["red fox"] == 101`. `"blank"` is NOT in `species.counts`. `top_k` has ≤ `top_k_species` entries, each with `species/count/pct`, sorted by count desc.
- [ ] `medvednica_report.md` contains the literal headline numbers (8208, the empty-% , the kept-frame count, the species count) rendered from the summary (grep finds `8208` and the species count); it embeds all three figures via `figures/...` relative links; it contains NO raw ML metric terms like "mAP"/"top-1"/"V-measure".
- [ ] `medvednica_summary.json` parses as JSON, has `schema_version == 1`, `dataset == "MedvednicaDS"`, and contains the `funnel`, `species`, `temporal`, `examples`, `figures`, `report_md` keys. `examples` is a non-empty list; each example has a `crop_path` that exists on disk.
- [ ] Re-running with the same `seed` produces an identical `medvednica_summary.json` (deterministic; figures need not be byte-identical but the sampled `examples` list must be).
- [ ] `--species-filter "eurasian lynx"` runs without error, adds a `target_species` block with `filter == ["eurasian lynx"]` and integer `detections`/`frames` (honest 0 if none present), and the page includes the target-species callout.
- [ ] `generate_medvednica_report(..., use_store=False)` runs with no `reid_demo.sqlite` present (store never required); `use_store=True` against an absent/empty store degrades gracefully (warns, falls back to JSON-only numbers) rather than crashing.
- [ ] `python -m reid_demo.medvednica_report --selftest` exits 0 on the real data; a missing required JSON makes `generate_medvednica_report` raise `FileNotFoundError` naming the file.
- [ ] `tests/test_medvednica_report.py` passes under the repo venv (covers the four pure helpers on tiny in-memory fixtures + one end-to-end run on `data/MedvednicaDS` asserting the headline funnel numbers).

## How to verify

```bash
cd /home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project
source venv/bin/activate

# 1. Importable contract surface
python -c "from reid_demo.medvednica_report import generate_medvednica_report, parse_crop_filename, species_from_classes, compute_funnel, compute_species_counts; print('OK')"

# 2. Pure-helper sanity
python - <<'PY'
from reid_demo.medvednica_report import parse_crop_filename as p, species_from_classes as s
assert p("02020401_crop1_conf92.jpg") == ("02020401", 1, 92)
assert p("IMG_0066_crop1_conf78.jpg") == ("IMG_0066", 1, 78)
assert p("02020401_crop2.jpg")[2] is None
assert s(["uuid;mammalia;cetartiodactyla;suidae;sus;scrofa;wild boar"]) == "wild boar"
assert s([]) == ""
print("helpers OK")
PY

# 3. End-to-end on the real data + check verified numbers
python - <<'PY'
import json, os
from reid_demo.medvednica_report import generate_medvednica_report
out = "/tmp/medv_report"
summ = generate_medvednica_report("data/MedvednicaDS", out, top_k_species=12, n_example_crops=12, seed=0)
f = summ["funnel"]
assert f["total_frames"] == 8208, f
assert f["frames_with_any_detection"] == 6177, f
assert f["frames_with_animal"] == 5801, f
assert f["person_detections"] == 876 and f["vehicle_detections"] == 4, f
assert f["animal_detections_raw"] == 11795, f
assert f["animal_detections_kept"] == 2049 and f["kept_frames"] == 1866, f
# D7b: empty = ZERO detections of ANY category; people/vehicle frames reported SEPARATELY
assert f["empty_frames"] == 2031 == f["total_frames"] - f["frames_with_any_detection"], f
assert "person_or_vehicle_frames" in f and isinstance(f["person_or_vehicle_frames"], int), f
assert f["person_or_vehicle_frames"] != f["empty_frames"] or f["person_or_vehicle_frames"] == 0, f  # never folded into empty
sp = summ["species"]
assert sp["total_classified_detections"] == 2049, sp
assert sp["blank_detections"] == 445 and sp["real_species_detections"] == 1604, sp
assert sp["counts"]["wild boar"] == 638 and sp["counts"]["european roe deer"] == 349 and sp["counts"]["red fox"] == 101
assert "blank" not in sp["counts"]
for fn in ["medvednica_report.md","figures/detection_funnel.png","figures/species_breakdown.png","figures/example_crops.png","medvednica_summary.json"]:
    p = os.path.join(out, fn); assert os.path.exists(p) and os.path.getsize(p) > 0, p
assert summ["examples"] and all(os.path.exists(e["crop_path"]) for e in summ["examples"])
print("end-to-end OK")
PY

# 4. Markdown is plain-language and embeds figures
grep -q "8208" /tmp/medv_report/medvednica_report.md && echo "headline number present"
grep -q "figures/detection_funnel.png" /tmp/medv_report/medvednica_report.md && echo "funnel figure embedded"
! grep -Eiq "mAP|top-1|v-measure|ARI" /tmp/medv_report/medvednica_report.md && echo "no ML jargon"
# D7b: headline keeps "empty" and "people/vehicle" as distinct phrases (not conflated)
grep -Eiq "empty" /tmp/medv_report/medvednica_report.md && grep -Eiq "people|person|vehicle" /tmp/medv_report/medvednica_report.md && echo "empty vs people/vehicle distinguished"

# 5. Determinism of sampled examples
python - <<'PY'
import json
from reid_demo.medvednica_report import generate_medvednica_report
a = generate_medvednica_report("data/MedvednicaDS", "/tmp/medv_a", seed=0)
b = generate_medvednica_report("data/MedvednicaDS", "/tmp/medv_b", seed=0)
assert [e["crop_path"] for e in a["examples"]] == [e["crop_path"] for e in b["examples"]]
print("deterministic examples OK")
PY

# 6. Species filter + store-off / selftest
python -c "from reid_demo.medvednica_report import generate_medvednica_report as g; s=g('data/MedvednicaDS','/tmp/medv_lynx', species_filter=['eurasian lynx'], use_store=False); print('target_species', s['target_species'])"
python -m reid_demo.medvednica_report --selftest ; echo "selftest exit=$?"

# 7. Tests
python -m pytest tests/test_medvednica_report.py -q
```

## Open questions

1. **`blank` framing.** `blank` is the single largest SpeciesNet class (445). Reported separately as "crops the classifier rejected as having no identifiable animal." Confirm with the pitch narrative whether to also subtract `blank` frames from the headline "animals found" count, or keep them as detected-but-unclassifiable. (Default: keep funnel = MegaDetector animals; species chart = real species only; `blank` called out in prose.)
2. **Detection-vs-frame counting unit.** Numbers are reported as **detections** for species (an image can hold multiple animals) and **frames** for the funnel. Confirm the biologist-facing page should lead with frames ("photos") and mention detections only where multiple animals share a frame. (Default: lead with photos.)
3. **Camera dimension is degenerate here** (`unknown_camera` only). The per-camera tally is wired but flat. When real park footage with multiple cameras arrives, the same `cameras` block populates automatically — confirm no extra per-camera figure is needed for the demo. (Default: no per-camera chart in v1.)
4. **Store enrichment value.** Because the Medvednica artifacts pre-date the T01 store and the funnel is fully derivable from JSONs, `--use-store` is currently only a cross-check. Confirm whether T10 wants the report to *prefer* store numbers once a `MedvednicaDS` run has been ingested (so the page reflects any HITL/cleaning done in-store), or always report the raw-JSON funnel for honesty. (Default: raw-JSON funnel is authoritative; store is a non-authoritative cross-check.)
5. **Date span sanity.** `trail_cam_data.csv` and the JSON timestamps both start `2025-06-02`; confirm the full footage span to print (the report computes min/max from the data, but flag if the dump is a single-month slice produced by `utils/filter_dataset_by_month.py`).
