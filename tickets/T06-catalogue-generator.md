# T06 — Visual individual catalogue generator

> **Status:** 🔴 Not started · **Milestone:** M1 (Demo-ready)
> **Depends on:** T01, T02, T05 · **Blocks:** T10
> **Owner:** _unassigned_
>
> Self-contained ticket for the lynx re-identification demo/pilot. See [STATUS_BOARD.md](../STATUS_BOARD.md) for the full plan, the binding design decisions, and the agent working protocol.

---

## Context

We are building a DEMO + PILOT MVP of an **open-set, individual-animal re-identification** system for Eurasian lynx (closest public analog: spotted big cats — LeopardID2022 leopards, ATRW Amur tigers). Earlier tickets turn an unlabeled pile of camera-trap animal crops into **discovered individuals**: T02 detects + crops (and is the SOLE owner of `gt_identity`/`orientation`/`species` for labeled datasets), T03 labels species (field data), T04 embeds, and **T05 clusters** the embeddings into individuals (unknown count), writing a `cluster_id` per crop and flagging singletons and DBSCAN noise as candidate NEW individuals.

**Singleton / candidate-new convention (T05, binding):** singletons AND DBSCAN noise BOTH receive `cluster_id == -1` AND `is_candidate_new == 1` — there is a single authoritative rule, and `is_candidate_new` is what T06 keys on. There is no "assign cluster_id 0 to the 1-crop case" path; do not assume singletons get a fresh `>=0` id. A `cluster_id >= 0` always means a multi-crop discovered individual produced by clustering.

**This ticket (T06) is the showpiece a non-technical park biologist actually looks at.** It reads the clustered detection records out of the shared store and renders a **static, self-contained visual catalogue**: one entry (contact sheet) per discovered individual, plus an overall count summary expressed in *animals* ("Found 23 individuals across 412 photos"), not ML jargon. No model loading, no clustering, no network — pure rendering of already-computed results. It must open in any browser by double-clicking a single `index.html`, so it can be zipped and emailed to Risnjak NP.

All inter-module data flows through the **T01 shared store** (`reid_demo/store.py`, SQLite-backed, one row per crop). T06 is a **READ-ONLY consumer** of that store. You do NOT invent your own data format; you import T01's access API and read `DetectionRecord`s.

### The T01 data contract (verbatim — this is your only input schema)

The store lives at `reid_demo/store.py` and exposes a `detections` table, one row per crop, with these 28 columns (the ones T06 uses are marked **[USED]**):

| column | type | meaning (T06-relevant) |
|--------|------|------------------------|
| `record_id` **[USED]** | str | unique crop id, e.g. `02020401__crop1`. Join key. |
| `source_image` **[USED]** | str | path to original full frame. |
| `source_stem` | str | source filename stem. |
| `det_index` | int | detection index within source image. |
| `crop_path` **[USED]** | str | path to the cropped image on disk (the thumbnail you display). |
| `bbox_x/bbox_y/bbox_w/bbox_h` | float | normalized bbox (not needed by T06, but available). |
| `detector_conf` | float | MegaDetector confidence. |
| `camera_id` **[USED]** | str | camera/trap id, e.g. `unknown_camera`, `Camera 1`. |
| `timestamp` **[USED]** | str | ISO-8601 `YYYY-MM-DD HH:MM:SS`. |
| `species` **[USED]** | str | human-readable common name, e.g. `eurasian lynx`. NULL until T03. |
| `species_conf` | float | top species score. |
| `species_class` | str | full taxonomy string. |
| `embedding_ref` | str | key into embeddings pickle. |
| `embedding_path` | str | path of embeddings `.pkl`. |
| `cluster_id` **[USED]** | int | discovered individual id; `>=0` a multi-crop individual, `-1` = noise/singleton (candidate-new). NULL until T05. |
| `cluster_conf` **[USED]** | float | assignment confidence `[0,1]`. |
| `is_candidate_new` **[USED]** | int (0/1) | 1 = singleton OR DBSCAN noise flagged as a candidate NEW individual. Always paired with `cluster_id == -1` (D5). This is the authoritative flag downstream keys on. |
| `orientation` **[USED]** | str | flank, one of the canonical set `{left, right, front, back, down, unknown}`. Empty `''`/missing normalizes to `unknown` at ingest (T02). Lynx flanks differ left vs right; spot-bearing flanks `{left, right}` are individually re-identifiable, the rest are not. |
| `gt_identity` | str | ground-truth id (LeopardID2022/ATRW), populated by T02 at ingest, NULL for field data. Used by T07, not T06. |
| `review_status` **[USED]** | str | `unreviewed` (default), `confirmed`, `rejected`, `merged`, `split`. |
| `review_note` **[USED]** | str | free-text reviewer note. |
| `dataset` **[USED]** | str | run name, e.g. `MedvednicaDS`, `LeopardID2022`. Most queries filter on this. |
| `extra_json` | str | JSON escape hatch. |
| `created_at` / `updated_at` | str | store timestamps. |

T01's public API you will import (exact signatures from the T01 contract):

```python
from reid_demo.store import (
    connect, query_records, count_by, get_record, to_dataframe,
    DetectionRecord, DEFAULT_DB_PATH, COLUMNS,
)
# connect(db_path=DEFAULT_DB_PATH, *, create=True) -> sqlite3.Connection
# query_records(conn, *, dataset=None, species=None, cluster_id=None,
#               review_status=None, has_embedding=None, orientation=None,
#               where_sql=None, where_params=(), order_by="record_id", limit=None) -> list[DetectionRecord]
# count_by(conn, column, *, dataset=None) -> dict          # {value: count}
# to_dataframe(conn, *, dataset=None) -> pandas.DataFrame   # COLUMNS order
# get_record(conn, record_id) -> Optional[DetectionRecord]
```

**Join/grouping rules from T01 you must honor:** `(dataset, cluster_id)` with `cluster_id >= 0` groups crops into one multi-crop individual. `cluster_id == -1` is noise/singleton (NOT an individual). `cluster_id IS NULL` means clustering has not run for that row — skip it. `is_candidate_new == 1` is the authoritative flag (always paired with `cluster_id == -1`) that marks singletons and noise to highlight as "possible new animal." Route on `is_candidate_new`, not on the raw `cluster_id` value.

**Orientation / flank value set (DATA_CONTRACT, binding):** the canonical orientation values are `{left, right, front, back, down, unknown}` (empty `''`/missing already normalized to `unknown` by T02 at ingest). For T06's `by_flank` count, any value outside the canonical set — including NULL, empty, or an unexpected string — maps to `unknown` so the per-flank counts always sum to `crops_clustered` (see D7c below). Only `{left, right}` are spot-bearing/re-identifiable flanks; the `mixed_flank` warning fires only when a single cluster holds both `left` and `right`.

## Objective

Deliver a single self-contained module `reid_demo/catalogue.py` (plus a small HTML template asset) that, given a populated T01 store and a `dataset` name, produces a **static HTML catalogue directory** a non-technical biologist can browse offline:

1. An **overview page** (`index.html`) with the headline counts in plain language and a grid of individual cards (each card = best crop thumbnail + id + photo count + flank badge).
2. One **per-individual contact sheet** (an HTML section/page) showing all crops for that `cluster_id`, with per-crop metadata (camera, timestamp, species, confidence, review status) and an assignment-confidence indicator.
3. A dedicated **"candidate new individuals / unassigned"** section keyed on `is_candidate_new == 1` (which, per D5, is exactly the set of singletons + DBSCAN noise, all carrying `cluster_id == -1`) for human attention.
4. A machine-readable `catalogue_summary.json` (so T09 report and T10 runner can embed the numbers without re-deriving them).
5. Optional static **montage PNGs** (one contact-sheet image per individual) built with the existing `visualization_suite.collage.make_grid`, for slide decks / PDF.

The deliverable must be **flank-aware in presentation**: lynx left and right flanks are different patterns, so each individual card shows which flank(s) the cluster contains, and (when both flanks exist under one `cluster_id`) it is surfaced as a note, since most demos cluster per-flank.

## Scope

### In
- `reid_demo/catalogue.py`: pure rendering from the store to a static HTML directory + summary JSON + optional montage PNGs.
- A `build_catalogue(...)` Python entry point AND a `python -m reid_demo.catalogue ...` CLI.
- Plain-language count summary ("Found N individuals across M photos; K candidate new; J unassigned").
- Per-individual contact sheets with thumbnails sized down for portability (thumbnails embedded or copied into the output dir, so the bundle is movable).
- Sorting individuals by photo count (descending) so the most-photographed animals lead.
- Flank badges and a "mixed flank" warning per card.
- Highlighting low-confidence assignments (configurable `low_conf_threshold`) and review status (`confirmed`/`rejected`/`merged`) with a small visual marker.
- Graceful handling of missing crop files (placeholder tile, still counts the record).
- A self-contained HTML/CSS (no external CDN, no JS framework required; minimal inline JS at most for collapse/expand is OK but the page must be usable with JS disabled).
- Unit tests under `tests/test_catalogue.py` using a tiny seeded in-memory/temp store.

### Out
- Anything that writes back to the store (T06 is read-only). Human decisions are T08; do not implement review actions, only DISPLAY current `review_status`.
- Detection/cropping (T02), species labels (T03), embeddings (T04), clustering (T05), the eval metrics page (T07), the HITL UI (T08), the Medvednica filtering report (T09), the end-to-end runner (T10).
- Any web server / live dashboard / database edits / geolocation maps. The output is static files only.
- Re-deriving clusters or recomputing confidence — consume the columns as written by T05.

## Inputs

- A populated T01 SQLite store (default `data/reid_demo/reid_demo.sqlite`) where T05 has already written `cluster_id`/`cluster_conf`/`is_candidate_new`. Selected by `--dataset` (e.g. `MedvednicaDS` or `LeopardID2022`).
- The crop image files referenced by each record's `crop_path` (written by T02; for Medvednica these resemble `data/MedvednicaDS/animal_crops/02020401_crop1_conf92.jpg`, though T02 may rewrite them under `data/reid_demo/crops/`). T06 reads these images read-only; if a file is missing it renders a placeholder and logs a warning.
- Optionally `source_image` full frames (used only if `--show-full-frame` is set; default off — crops are the primary tiles).

T06 must NOT depend on any in-conversation knowledge; everything it needs is in the store columns above and the crop files on disk.

## Outputs

A static catalogue directory (default `data/reid_demo/catalogue/<dataset>/`, parent dirs auto-created), containing:

- `index.html` — overview: headline counts + clickable grid of individual cards.
- `individuals/individual_<cluster_id>.html` — one contact sheet per discovered individual (or a single-page variant with anchors; either layout is acceptable as long as `index.html` links resolve).
- `unassigned.html` — candidate-new singletons + noise (`cluster_id == -1`).
- `thumbs/` — downsized crop thumbnails copied/generated here so the directory is portable when zipped/moved (do NOT rely on absolute paths to the original crops in the HTML).
- `montages/individual_<cluster_id>.png` — optional contact-sheet PNGs (only when `--montages` is passed).
- `catalogue_summary.json` — machine-readable summary (schema below) for T09/T10.
- `assets/style.css` — the stylesheet (no external CDN).

## Interface contract

Downstream tickets (T09 report, T10 runner) depend on EXACTLY the following. Do not rename.

### Python API

```python
def build_catalogue(
    db_path: str = DEFAULT_DB_PATH,
    *,
    dataset: str | None = None,           # filter to one run; None = all rows in store
    out_dir: str | None = None,           # default: data/reid_demo/catalogue/<dataset or "all">
    species: str | None = None,           # optional extra species filter (e.g. "eurasian lynx")
    low_conf_threshold: float = 0.5,      # cluster_conf below this is flagged "review"
    thumb_size: int = 256,                # longest-edge px for thumbnails in thumbs/
    max_crops_per_individual: int | None = None,  # cap tiles per contact sheet (None = all)
    make_montages: bool = False,          # also render montages/*.png via visualization_suite
    title: str = "Individual Catalogue",  # page H1 / report title
) -> "CatalogueResult":
    """Read clustered records from the T01 store and render a static HTML catalogue
    directory + catalogue_summary.json. Read-only w.r.t. the store. Returns a
    CatalogueResult with output paths and the summary dict. Never raises on a
    missing crop file (renders a placeholder); raises only on an unreadable store
    or an empty result set after filtering."""

@dataclass
class CatalogueResult:
    out_dir: str
    index_html: str            # absolute path to index.html
    summary_json: str          # absolute path to catalogue_summary.json
    summary: dict              # the same content as catalogue_summary.json (see schema)
    individual_pages: dict     # {cluster_id(int): absolute_html_path(str)}
    montage_pngs: dict         # {cluster_id(int): absolute_png_path(str)}  (empty if make_montages=False)
```

### `catalogue_summary.json` schema (consumed by T09/T10 — stable keys)

```json
{
  "dataset": "MedvednicaDS",
  "species_filter": null,
  "generated_at": "2026-06-09T12:00:00",
  "low_conf_threshold": 0.5,
  "counts": {
    "total_crops": 412,
    "crops_clustered": 405,
    "individuals": 23,
    "candidate_new": 4,
    "unassigned_noise": 7,
    "low_confidence_crops": 11,
    "reviewed_confirmed": 0,
    "reviewed_rejected": 0
  },
  "headline": "Found 23 individuals across 405 photos (4 possible new, 7 unassigned).",
  "by_flank": {"left": 180, "right": 200, "front": 5, "back": 4, "down": 1, "unknown": 15},
  "individuals": [
    {
      "cluster_id": 0,
      "n_crops": 41,
      "flanks": ["left"],
      "mixed_flank": false,
      "species": ["eurasian lynx"],
      "cameras": ["unknown_camera"],
      "first_seen": "2025-06-02 04:27:51",
      "last_seen": "2025-06-04 19:03:10",
      "mean_cluster_conf": 0.87,
      "n_low_conf": 1,
      "n_confirmed": 0,
      "representative_crop": "thumbs/02020401__crop1.jpg",
      "page": "individuals/individual_0.html"
    }
  ]
}
```

Rules the schema must satisfy:
- `counts.individuals` = number of DISTINCT `cluster_id >= 0` after filtering (noise/singleton `-1` and NULL excluded).
- `counts.crops_clustered` = rows with `cluster_id >= 0`.
- `counts.candidate_new` = rows with `is_candidate_new == 1` (the authoritative candidate-new flag; per D5 these all also have `cluster_id == -1`).
- `counts.unassigned_noise` = rows with `cluster_id == -1`.
- `counts.low_confidence_crops` = rows with `cluster_conf < low_conf_threshold` (NULL conf counts as low).
- **`by_flank` (D7c — binding):** computed over `cluster_id >= 0` rows ONLY (the same population as `crops_clustered`). Each such crop is counted exactly once under its own `orientation`. Any NULL, empty, or non-canonical orientation value maps to the `"unknown"` bucket. The dict always contains all six canonical keys `{left, right, front, back, down, unknown}` (zero-filled), and **`sum(by_flank.values()) == counts.crops_clustered`** is an invariant that must hold. Noise/candidate-new rows (`cluster_id == -1`) are NOT included in `by_flank`.
- `headline` is the plain-language one-liner T09 can drop straight into the pitch.
- `individuals` list is sorted by `n_crops` desc, then `cluster_id` asc.
- `representative_crop` and `page` paths in the JSON are RELATIVE to `out_dir` (so the bundle is portable).

### CLI

```
python -m reid_demo.catalogue --db <path> --dataset MedvednicaDS [--out <dir>] \
    [--species "eurasian lynx"] [--low-conf 0.5] [--thumb-size 256] \
    [--max-crops 50] [--montages] [--title "Risnjak NP — Lynx Catalogue"]
# Writes the catalogue dir, prints the absolute index.html path and the headline line. Exit 0 on success.

python -m reid_demo.catalogue --selftest [--db <tmp>]
# Seeds a tiny throwaway store (3 fake individuals with cluster_id 0/1/2 + 1 candidate-new
# singleton with cluster_id == -1 / is_candidate_new == 1 (per D5), with placeholder crops
# generated on the fly), builds the catalogue into a temp dir, asserts index.html and
# catalogue_summary.json exist, counts are correct, and sum(by_flank.values()) == crops_clustered.
# Exit 0 on success, non-zero on failure.
```

### File-format guarantees for downstream

- `index.html` opens standalone (double-click) with NO network access; all CSS is local under `assets/`, all images under `thumbs/` (relative paths only).
- `catalogue_summary.json` keys above are stable; T09/T10 read it instead of re-querying the store.
- The whole `out_dir` is relocatable: zip it, move it, open `index.html` — links and images still resolve.

## Existing code to reuse (real paths)

- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/reid_demo/store.py` — **the store API** (from T01). Import `connect`, `query_records`, `count_by`, `to_dataframe`, `get_record`, `DetectionRecord`, `DEFAULT_DB_PATH`, `COLUMNS`. This is your sole data source. (If T01 is not yet merged when you start, the T01 contract above is authoritative; code against those signatures.)
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/visualization_suite/collage.py` — `make_grid(images, titles=None, cols=2, figsize=None) -> (grid_img_ndarray, meta)`. Use for the optional `montages/*.png` contact sheets. Note it expects BGR ndarrays (it calls `io.bgr_to_rgb`); load crops with `cv2.imread` (BGR) or convert RGB→BGR before passing.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/visualization_suite/style.py` — `set_style()` and `save_high_res(fig, path, dpi=300)` for consistent figure styling on the montages.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/visualization_suite/io.py` — `bgr_to_rgb`, `fig_to_image` helpers used by `collage`.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/constants.py` — follow its `ROOT_DIR = os.path.dirname(os.path.abspath(__file__))` style for any path constants; do NOT edit it. Output under `data/reid_demo/...` to match the demo's existing convention (T01 uses `data/reid_demo/`).
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/data/MedvednicaDS/animal_crops/` — real crop files (e.g. `02020401_crop1_conf92.jpg`) you can use for a realistic manual smoke test once T02 has populated the store; confirms thumbnail loading works on real footage.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/data/MedvednicaDS/visualizations/` — examples of existing per-image visualizations (`vis_02020401.JPG`) for visual style reference only.

### Libraries available in the repo venv (confirmed present)
`Pillow 11.2.1` (use `PIL.Image` for thumbnail resizing — preferred over matplotlib for portability), `pandas 2.3.0`, `matplotlib 3.10.3`, `jinja2 3.1.6` (you MAY use Jinja2 for templating, but inline f-strings are equally acceptable — keep it dependency-light and stdlib-first), plus `numpy`, `opencv-python`. Do NOT add new pip dependencies. Use the repo venv: `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/venv`.

## Implementation notes

- **Read once, render many.** Pull all needed rows with one `query_records(conn, dataset=..., species=...)` (or `to_dataframe`), then group in Python by `cluster_id`. Don't issue per-individual SQL in a loop. Use `count_by(conn, "cluster_id", dataset=...)` and `count_by(conn, "species", dataset=...)` for the headline numbers where convenient, but the authoritative grouping for cards should come from the rows you already loaded so flank/camera/timestamp aggregation is consistent.
- **What counts as an "individual":** distinct `cluster_id >= 0` (always a multi-crop cluster). Exclude `cluster_id == -1` (noise/singleton) and `cluster_id IS NULL` (not yet clustered) from the individual grid; route everything with `is_candidate_new == 1` (equivalently, `cluster_id == -1` per D5) to `unassigned.html`. Because D5 makes candidate-new and noise the SAME rows, key the unassigned section on `is_candidate_new == 1`; de-duplicate by `record_id`. Do NOT show a candidate-new/noise crop as its own individual card.
- **Representative crop selection:** pick the crop with the highest `cluster_conf` (ties broken by highest `detector_conf`, then `record_id`) as the card thumbnail. Fall back to first crop if confidences are NULL.
- **Flanks (D4 + D7c):** aggregate the set of `orientation` values per cluster. `mixed_flank = True` if the cluster contains both `left` and `right` (a real flag for a biologist, since lynx flanks shouldn't merge across sides — only `{left, right}` are spot-bearing/re-identifiable). Render `unknown`/NULL/non-canonical orientation as an "unknown flank" badge, not an error. Populate `by_flank` from `cluster_id >= 0` crops ONLY (count each crop once by its own orientation; map NULL/empty/any non-canonical value → `unknown`), zero-filling all six canonical keys, so that `sum(by_flank.values()) == counts.crops_clustered` holds exactly (D7c sum invariant). Do not count noise/candidate-new (`cluster_id == -1`) crops in `by_flank`.
- **Plain-language first.** The top of `index.html` and the `headline` string must read like "Found 23 individuals across 405 photos (4 possible new, 7 unassigned)." Put ML terms (cluster_conf, noise) in small print / tooltips only.
- **Low-confidence + review markers:** any crop with `cluster_conf < low_conf_threshold` (or NULL conf) gets a small "needs review" marker on its tile; tiles also show their `review_status` when not `unreviewed` (e.g. a green check for `confirmed`, a red mark for `rejected`). This is display only — T08 owns the actual decisions.
- **Portability is a hard requirement.** Generate thumbnails into `thumbs/` (downscale longest edge to `thumb_size` with `PIL.Image.thumbnail`) and reference them by RELATIVE path in the HTML. Never emit an absolute filesystem path into the HTML. Same for `assets/style.css`. The directory must work after being zipped and opened elsewhere.
- **Missing/unreadable crop files:** catch the load error, emit a gray placeholder tile (generate a small placeholder PNG once and reuse), increment a `missing_crops` counter you can log, but keep the record in counts. Never abort the whole build for one bad image.
- **Single-page vs multi-page:** either (a) one `index.html` + per-individual files under `individuals/`, or (b) one big `index.html` with anchor links and the per-individual sections inline. Pick one; `CatalogueResult.individual_pages` must still map each `cluster_id` to a resolvable href (a `#anchor` URL into index.html is acceptable for layout (b)).
- **Montages (optional):** when `make_montages=True`, for each individual build a grid via `visualization_suite.collage.make_grid` (cap tiles via `max_crops_per_individual`, default a sane cap like 25 for the PNG even if HTML shows all) and save to `montages/individual_<id>.png`. Wrap the import in try/except so a headless/matplotlib hiccup downgrades to "HTML only" with a warning instead of crashing.
- **No store writes.** Open the connection; only SELECT. Do not call any `update_*`/`upsert_*` function.
- **Keep it one module.** `reid_demo/catalogue.py` plus, if you use Jinja2, templates under `reid_demo/templates/`. Re-export `build_catalogue` and `CatalogueResult` from `reid_demo/__init__.py` (additive edit only; do not remove T01's exports).
- Add a one-line entry to `STATUS_BOARD.md` (create it if absent) marking T06 deliverables; do not edit other tickets' status lines.

## Acceptance criteria

- [ ] `reid_demo/catalogue.py` exists and adds no new pip dependency (only stdlib + already-present `PIL`/`pandas`/`matplotlib`/`numpy`/`cv2`/optional `jinja2`).
- [ ] `python -c "from reid_demo.catalogue import build_catalogue, CatalogueResult"` succeeds.
- [ ] `build_catalogue` is read-only w.r.t. the store: a test asserting the `detections` table contents are byte-identical before and after a build passes (e.g. compare `count_by` results / a full-row hash before and after).
- [ ] `python -m reid_demo.catalogue --selftest` exits 0: it seeds a throwaway store with a known layout (3 individuals of sizes 3/2/2 with `cluster_id` 0/1/2; one candidate-new singleton with `is_candidate_new == 1` AND `cluster_id == -1`, per D5 — there is no separate "noise vs singleton" row class), builds the catalogue, and asserts `summary["counts"]["individuals"] == 3`, `candidate_new == 1`, `unassigned_noise == 1`, `sum(by_flank.values()) == crops_clustered`, and that `index.html` + `catalogue_summary.json` exist.
- [ ] After a build on a seeded store, `out_dir/index.html`, `out_dir/catalogue_summary.json`, `out_dir/assets/style.css`, and one `out_dir/individuals/individual_<id>.html` (or resolvable anchor) per individual all exist.
- [ ] `catalogue_summary.json` validates against the schema above: required keys present; `counts.individuals` equals distinct `cluster_id >= 0`; `individuals` sorted by `n_crops` desc then `cluster_id` asc; `headline` is a non-empty plain-language sentence containing the individual count.
- [ ] All image and CSS `src`/`href` attributes in `index.html` are RELATIVE (no leading `/`, no `file://`, no absolute paths); a test greps the HTML to confirm.
- [ ] Portability: moving/copying `out_dir` to a different path leaves `index.html` openable with all `thumbs/*` and `assets/style.css` references still resolving (test by `shutil.copytree` to a new dir and asserting referenced relative files exist there).
- [ ] A record whose `crop_path` points to a non-existent file does NOT crash the build; the build completes and the record is still counted in `total_crops` (placeholder tile rendered).
- [ ] `mixed_flank` is `True` for a cluster containing both a `left` and a `right` crop and `False` otherwise.
- [ ] `by_flank` is computed over `cluster_id >= 0` rows only, contains all six canonical keys `{left, right, front, back, down, unknown}`, maps NULL/empty/non-canonical orientation to `unknown`, and `sum(by_flank.values()) == counts.crops_clustered` (D7c sum invariant — a test asserts this equality, including for a seeded row with NULL/empty orientation).
- [ ] Low-confidence crops (`cluster_conf < low_conf_threshold` or NULL) are counted in `counts.low_confidence_crops` and visibly marked in the per-individual page (test checks the count and that the marker class/string appears in the HTML).
- [ ] `--montages` produces one `montages/individual_<id>.png` per individual (when matplotlib is functional); without `--montages` the `montages/` dir is empty/absent and `CatalogueResult.montage_pngs == {}`.
- [ ] `tests/test_catalogue.py` passes under the repo venv.
- [ ] No existing repo file is modified except additive edits to `reid_demo/__init__.py` (re-exports) and `STATUS_BOARD.md` (one line).

## How to verify

```bash
cd /home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project
source venv/bin/activate

# 1. Import surface
python -c "from reid_demo.catalogue import build_catalogue, CatalogueResult; print('OK')"

# 2. Self-test (seeds throwaway store, builds catalogue, checks counts)
python -m reid_demo.catalogue --selftest --db /tmp/reid_cat_selftest.sqlite ; echo "exit=$?"

# 3. Programmatic build on a seeded store + assertions
python - <<'PY'
import os, json, sqlite3, hashlib
from reid_demo.store import connect, upsert_records, DetectionRecord, make_record_id, count_by
from reid_demo.catalogue import build_catalogue

db = "/tmp/reid_cat_demo.sqlite"
if os.path.exists(db): os.remove(db)
conn = connect(db)

# seed: 3 individuals (sizes 3/2/2 incl. one empty-orientation crop), 1 candidate-new/noise singleton (cluster_id == -1 / is_candidate_new == 1 per D5)
recs = []
def mk(stem, idx, cid, conf, flank, cand=0):
    return DetectionRecord(
        record_id=make_record_id(stem, idx),
        source_image=f"data/MedvednicaDS/animal_images/{stem}.JPG",
        source_stem=stem, det_index=idx,
        crop_path=f"/tmp/does_not_exist/{stem}__crop{idx}.jpg",  # missing on purpose
        bbox_x=0.1,bbox_y=0.1,bbox_w=0.2,bbox_h=0.2,
        detector_conf=0.9, camera_id="unknown_camera",
        timestamp="2025-06-02 04:27:51", species="eurasian lynx",
        species_conf=0.95, cluster_id=cid, cluster_conf=conf,
        is_candidate_new=cand, orientation=flank, dataset="MedvednicaDS")
for i in range(3): recs.append(mk(f"A{i}",1,0,0.9,"left"))
for i in range(2): recs.append(mk(f"B{i}",1,1,0.4,"right"))   # low-conf cluster
recs.append(mk("Cmix",1,2,0.8,"left")); recs.append(mk("Cmix2",1,2,0.8,"right"))  # mixed flank (left + right)
recs.append(mk("Cnull",1,2,0.8,""))   # empty orientation in a clustered cluster -> by_flank "unknown"
recs.append(mk("S",1,-1,0.3,"unknown",cand=1))   # candidate-new singleton (noise id, cluster_id == -1 per D5)
upsert_records(conn, recs)

before = count_by(conn, "cluster_id", dataset="MedvednicaDS")
res = build_catalogue(db, dataset="MedvednicaDS", out_dir="/tmp/reid_cat_out", low_conf_threshold=0.5)
after = count_by(conn, "cluster_id", dataset="MedvednicaDS")
assert before == after, "build must not mutate the store"

s = res.summary
assert s["counts"]["individuals"] == 3, s["counts"]
assert s["counts"]["candidate_new"] == 1, s["counts"]
assert s["counts"]["unassigned_noise"] == 1, s["counts"]
# cluster 2 has left+right -> mixed flank
ind2 = next(i for i in s["individuals"] if i["cluster_id"] == 2)
assert ind2["mixed_flank"] is True, ind2
# D7c: by_flank over cluster_id>=0 only, NULL/empty/non-canonical -> "unknown", sum invariant holds
bf = s["by_flank"]
assert set(bf) == {"left","right","front","back","down","unknown"}, bf
assert sum(bf.values()) == s["counts"]["crops_clustered"], (bf, s["counts"])
assert bf["unknown"] == 1, bf   # the empty-orientation clustered crop (Cnull); noise/candidate-new excluded
assert os.path.exists(res.index_html) and os.path.exists(res.summary_json)
# relative paths only in HTML
html = open(res.index_html).read()
assert "file://" not in html and 'src="/' not in html and 'href="/' not in html
print("HEADLINE:", s["headline"])
print("all assertions passed")
PY

# 4. Portability: copy the output dir and confirm referenced files resolve
python - <<'PY'
import shutil, os, re
src="/tmp/reid_cat_out"; dst="/tmp/reid_cat_moved"
shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst)
html=open(os.path.join(dst,"index.html")).read()
for m in re.findall(r'(?:src|href)="([^"]+)"', html):
    if m.startswith("http") or m.startswith("#"): continue
    p=os.path.normpath(os.path.join(dst, m))
    assert os.path.exists(p), f"missing after move: {m}"
print("portable OK")
PY

# 5. Tests
python -m pytest tests/test_catalogue.py -q
```

## Open questions

1. **Layout (multi-file vs single-page):** ticket allows either an `individuals/individual_<id>.html` per animal or one big `index.html` with anchors. Default recommendation: per-file pages for large runs (LeopardID2022 has hundreds of identities) to keep each page light; confirm if T10's demo bundle prefers a single scrollable page for emailing.
2. **Crop vs full-frame tiles:** default tiles are the `crop_path` thumbnails (tight on the animal, best for flank patterns). `--show-full-frame` (drawing the bbox on `source_image`) is out of scope for v1 but reserved; confirm the pitch doesn't need full frames in the catalogue (T09's Medvednica report may already cover full-frame examples).
3. **`is_candidate_new` vs `cluster_id == -1` overlap — RESOLVED (D5).** T05's policy is fixed: every singleton AND every DBSCAN-noise crop gets `cluster_id == -1` AND `is_candidate_new == 1`; there is no fresh `>=0` id and no "assign 0" path. T06 keys the "candidate new / unassigned" section on `is_candidate_new == 1`, and a candidate-new/noise crop NEVER appears as its own individual card (it's only in the unassigned section). No further confirmation needed.
4. **Thumbnail embedding:** copy-to-`thumbs/` (default, smaller HTML, more files) vs base64-inline data URIs (single self-contained HTML, larger file). Default is `thumbs/` for performance on hundreds of crops; flag if T10 wants a single-file HTML for emailing and we should add a `--inline-images` option.
5. **Multi-flank individuals:** for the demo we surface `mixed_flank` as a warning but still show one card. Per D4, T05 already clusters spot-bearing flanks `{left, right}` in SEPARATE buckets while pooling `{front, back, down, unknown, ''}` into one `other` bucket, so a `mixed_flank` individual (both `left` and `right` under one `cluster_id`) is an anomaly worth flagging rather than the norm. Treating left/right as fully separate catalogue entries is already the T05 clustering behavior; T06 only displays the warning. No T06 display change pending.
