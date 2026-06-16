# T10 — Demo assembly & end-to-end runner

> **Status:** 🔴 Not started · **Milestone:** M1 (Demo-ready)
> **Depends on:** T01, T02, T03, T04, T05, T06, T07, T09, T11, T12 · **Blocks:** —
> **Owner:** _unassigned_
>
> Self-contained ticket for the lynx re-identification demo/pilot. See [STATUS_BOARD.md](../STATUS_BOARD.md) for the full plan, the binding design decisions, and the agent working protocol.

---

## Signal layering (D8 — `--signals` and the accuracy "lift")

> **Amendment per binding decision D8 (see STATUS_BOARD.md).** Add a `--signals {global|global+fisher|full-funnel}` flag (default `global` for the `--smoke` path; `full-funnel` for the headline demo).
>
> - `global`: T04 embeddings → T05 (backbone only; T11/T12 not run).
> - `global+fisher`: also run the **T11** Fisher stage, then **T12** `build_fused_affinity` (global+Fisher, calibrated), and pass that affinity into T05.
> - `full-funnel`: as above, plus **T12** `gv_rerank` on borderline pairs to refine boundaries and feed T08's queue.
> - Register **T11** (Fisher) and **T12** (fusion/GV) as pipeline stages that run only when the chosen signal set needs them.
> - The demo bundle/manifest MUST report **both** global-only and full-funnel accuracy from T07 — the "lift" beat (*"global finds N individuals; with geometric verification, M"*). B-track still stamps species via `set_known_species` (D7d); read the headline from T07's single eval JSON on the 0–100 scale (D6).
> - **Acceptance (added):** `--signals global` runs with neither T11 nor T12; `--signals full-funnel` runs both and the manifest contains two accuracy numbers (global vs full-funnel).

## Context

We are building a DEMO + PILOT MVP of an **open-set, individual-animal re-identification** system for Eurasian lynx, to cold-pitch Croatian national parks (first target: Risnjak NP). The closest public analog to lynx is spotted big cats — **LeopardID2022** (leopards) and **ATRW** (Amur tigers). The existing repo (`/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project`) does CLOSED-SET re-id (query vs known gallery). The demo pivots the decision layer to OPEN-SET CLUSTERING: take an unlabeled pile of animal crops, discover how many DISTINCT individuals are present (unknown count), and flag singletons that match nothing as candidate NEW individuals. Everything upstream (detect, crop, embed) is reused.

The whole demo is a constellation of independent modules (T01–T09) built by separate agents. They all read/write the **same per-crop "detection record"** through a single shared store + access module defined in **T01** (`reid_demo/store.py`, `reid_demo/DATA_CONTRACT.md`). **This ticket, T10, is the conductor.** It contains NO new detection / classification / embedding / clustering / catalogue / eval / report logic — it only *calls* the T02–T09 modules in the right order, on the right datasets, and then *gathers their outputs into one shareable demo bundle* (a folder + an index page + a printable accuracy/summary sheet).

The pitch has **two honest parts**, and T10 must produce a bundle that tells both:

- **(A) REAL LOCAL DATA — capability to filter Croatian footage.** `data/MedvednicaDS/` already has MegaDetector + species classification run (`megadetector_results.json`, `animal_detections.json`, `animals_classified.json`, `detections_cleaned.json`, `animal_crops/`, `animal_images/`, `trail_cam_data.csv`). The framing to the park: "we can strip empty frames and sort your footage by species." This is the **T09 Medvednica filtering report**.
- **(B) INDIVIDUAL-ID CAPABILITY — proven on spotted cats.** Run the FULL open-set pipeline (T02 ingest → T04 embed → T05 cluster → T06 catalogue → T07 eval) on **LeopardID2022** (and optionally ATRW), and show "found N individuals vs M known; X% of photos correctly grouped" plus a visual catalogue. Framing: "lynx is the same spotted-cat individual-ID problem; we just need your photos."

T10 is what the user actually runs the night before the pitch: **one command produces the whole bundle.**

### Pipeline shape T10 orchestrates (each arrow is one of T02–T12; all share the T01 store)

```
                                  ┌─ (B) LeopardID2022 / ATRW (labelled; T02 ingest populates gt_identity + orientation + species) ─┐
raw crops / metadata ──► T02 ingest ──► T03 species* ──► T04 embed ─┬─► T05 cluster ──► T06 catalogue
                                  │                                 │   ▲                └─► T07 eval (vs gt_identity)
                                  │           (optional accuracy layer, --signals)        │
                                  │     T11 fisher ──► T12 fusion/GV ──► fused/GV affinity ┘
                                  └─ (A) MedvednicaDS (real field data) ─────────────────► T09 filtering report
                                                                                                      │
   ALL OUTPUTS ────────────────────────────────────────────────────────────────► T10 assembles demo bundle
```

\*\*Multi-signal clustering (binding decision D8).** The clustering pipeline runs on a SELECTABLE signal set chosen by a new `--signals` flag: `global` (default for `--smoke`; T04 embeddings only — the scalable M1-core backbone), `global+fisher` (fuse T04 global + T11 Fisher-vector cosine into the pairwise affinity, calibrated), or `full-funnel` (default for the headline demo; the `global+fisher` affinity PLUS T12 geometric-verification reranking on a SHORTLIST of borderline/candidate-merge pairs only). T04 (global embed) + T05 (clustering on global affinity) REMAIN the standalone backbone and have NO hard dependency on T11/T12, so the demo is never blocked on geometric verification (GV); the Fisher/GV path is a fast-follow accuracy layer, still M1. When the chosen signal set needs them, T10 runs the NEW **T11 Fisher stage** (`reid_demo/fisher.py`, per-crop Fisher vector cached in the T01 store) and the NEW **T12 fusion/GV stage** (`reid_demo/fusion.py`, fused calibrated pairwise affinity + GV reranker over borderline pairs) and passes the resulting fused/GV-refined affinity into T05's pluggable affinity interface. The demo bundle REPORTS BOTH the global-only and the full-funnel accuracy (the **"lift" beat**: "global finds N individuals; add geometric verification → M"). On spotted cats this lift is large (the repo's closed-set tables show global-only ELPephants 13.66% → 52% with global+Fisher+GV).

\*T03 (species stage) for the B-track is NOT skipped: T10 still runs the cheap, model-free `T03.set_known_species(dataset, species=...)` (`leopard` for LeopardID2022, `tiger` for ATRW) so every B-track row carries a uniform `species` column for T05/T06/T07. Only the SpeciesNet **model** branch of T03 (the GPU classifier) is skipped for the B-track. T10 makes the species *model* path optional per dataset via config; the known-species stamp always runs.

### Real repo facts T10 must stay consistent with

- **Venv:** the repo's active environment is `venv/` (`venv/bin/python`, Python 3.12). All subcommands run under it.
- **T01 store (the backbone — consume verbatim, do not re-define):** `reid_demo/store.py` exposes `connect(db_path, *, create=True)`, `query_records(...)`, `count_by(conn, column, *, dataset=None)`, `to_dataframe(conn, *, dataset=None)`, `export_records(...)`, and `make_record_id(source_stem, det_index)`. The store holds one row per crop in table `detections` with a `dataset` column so **one DB can hold MedvednicaDS, LeopardID2022, and ATRW side by side** (filter by `dataset`). Default DB path `data/reid_demo/reid_demo.sqlite`. Read `reid_demo/DATA_CONTRACT.md` for the 28-column schema and the "how each ticket touches the store" map.
- **LeopardID2022 / ATRW loading (T02 owns it):** `utility_functions.load_dataset(subset)` (e.g. `load_dataset("LeopardID2022")`) returns a DataFrame. Ground-truth `identity`, `orientation` (left/right/front/back/down — the flank), and `species` come from the **raw** `WildlifeReID10k` metadata at `data/wildlifedatasets/wildlifereid-10k/versions/7/metadata.csv` (columns: `identity,path,date,orientation,species,split,dataset,cluster_id`; 6806 LeopardID2022 rows). For the B-track, T10 ingests via **`T02.ingest_wildlife_dataset(subset, *, max_identities=None, limit=None, db_path=..., dataset=...)`** — T02's FOURTH adapter, which loads the subset, creates ONE whole-frame detection record per image (`bbox=(0,0,1,1)`, `crop_path` = the original full image path, NO cropping / NO MegaDetector), and **populates `gt_identity` (from `identity`), `orientation` (from `orientation`, empty/missing → `'unknown'`), and `species` (from `species`)**. **T02 is the SOLE owner of `gt_identity`/`orientation` for labeled datasets; T10 never populates or re-derives them.** T10 just tells T02 which subset to ingest and (for `--smoke`) caps distinct identities by plumbing `--max-identities` straight through to `ingest_wildlife_dataset(max_identities=...)`. **T10 must not re-implement metadata loading** — it invokes T02's CLI/function.
- **Flank-awareness:** lynx/leopard left and right flanks are DIFFERENT patterns. T05 is flank-aware via the `orientation` field; T10 passes whatever flank policy flag T05 exposes straight through (default: match same-side).
- **Multi-signal accuracy layer (T11/T12, owned by those tickets — consume, do not re-define):**
  - **T11 Fisher (`reid_demo/fisher.py`):** mirrors T04's service shape (batch over crops, cache, keyed by `record_id`; reuses the T01 store + T02 crops). Wraps `feature_extraction.py` (local descriptors; **DISK default**, also SuperPoint/ALIKED/SIFT) + `feature_aggregation.py` (`load_or_train_fisher_vectors(...)` → PCA+GMM → per-crop Fisher vector, power+L2 normalized, dim `2·K·D`). Produces a per-crop Fisher vector cached and referenced from the T01 store. Hard deps: T01, T02. Supports both A-track (Medvednica crops) and B-track (LeopardID2022 whole-frame crops). T10 invokes T11's CLI/function; it does NOT re-implement descriptor/Fisher logic.
  - **T12 fusion/GV (`reid_demo/fusion.py`):** provides (a) a FUSED PAIRWISE-AFFINITY provider combining T04 global + T11 Fisher cosine similarities, calibrated to probabilities via `calibration.py` (`isotonic_pchip` default — the WildFusion trick), which T05 consumes through its pluggable affinity interface; and (b) a GV RERANKER that runs `geometric_verification.py` (LightGlue + RANSAC/MAGSAC) on a SHORTLIST of borderline / candidate-merge pairs ONLY (never all N², O(borderline pairs); the pair budget is capped and logged if capped), returning a per-pair geometric score used to (i) refine cluster boundaries in T05 and (ii) prioritize the T08 review queue. Hard deps: T01, T04, T05, T11. T10 invokes T12's CLI/function; it does NOT re-implement fusion/GV logic.
  - **T05 affinity is PLUGGABLE and unchanged in hard deps:** `cluster_embeddings(...)` accepts an OPTIONAL precomputed pairwise affinity (or provider); default builds global cosine internally (current backbone behavior). T05 must NOT import T11/T12. **T10 is the wiring point**: when `--signals` requires it, T10 computes the T12 affinity (after running T11) and passes it into T05.
- **Audience:** outputs must be readable by a NON-TECHNICAL park biologist — the bundle's top-level summary is expressed in **animals** ("found 23 individuals vs 24 known; 96% of photos correctly grouped"), with ML metrics relegated to a details section.

### Out of scope (Phase 2, do NOT build)

Full web dashboard, geolocation maps, spatial capture-recapture / population density, deployment infra. T10 produces a **static, self-contained folder** (HTML + images + JSON/CSV/MD), nothing served or hosted.

## Objective

Deliver a single **end-to-end runner** that, with **one command**, executes the open-set pipeline on a configured dataset (default LeopardID2022) and assembles a **shareable demo bundle** combining:

1. the **visual individual catalogue** (from T06),
2. the **plain-language + standard accuracy numbers** (from T07), reported for BOTH the global-only backbone and the full-funnel signal set (the "lift" beat) when both were run,
3. the **Medvednica filtering report** (from T09),
4. a one-page **bundle index** (`index.html` + `SUMMARY.md`) that ties parts (A) and (B) together for a biologist,
5. a **`manifest.json`** describing exactly what ran (datasets, signal set, commands, store path, per-stage row counts, durations, output paths) for reproducibility.

T10 is **orchestration + assembly only**. It must call the real T02–T09 entry points (CLI or importable functions, whichever those tickets expose) and the T01 store; it must NOT duplicate their logic. If a stage's outputs already exist and are fresh, T10 may skip re-running it (idempotent / resumable), controlled by a `--force` flag.

## Scope

### In

- A new module `reid_demo/run_demo.py` with a CLI `python -m reid_demo.run_demo ...` (and a thin importable `run_demo(config) -> BundleResult`).
- A small declarative **config** (dataclass + optional YAML/JSON file) listing which datasets to process, which **signal set** to cluster on (`global` | `global+fisher` | `full-funnel`), and which stages each dataset goes through (B-track: full pipeline; A-track: report only). Sensible defaults so `python -m reid_demo.run_demo` "just works" on LeopardID2022 + MedvednicaDS.
- **A `--signals {global|global+fisher|full-funnel}` flag (binding decision D8).** Selects the clustering signal set: `global` (default for `--smoke`; T04 only — the standalone backbone, no T11/T12), `global+fisher` (T04 + T11 Fisher fused into the calibrated pairwise affinity), or `full-funnel` (default for the headline demo; the fused affinity PLUS T12 GV reranking on borderline pairs). When the chosen signal set needs them, T10 runs the **T11 Fisher stage** and **T12 fusion/GV stage** and passes the fused/GV-refined affinity into T05; with `global` it does neither and T05 uses its internal global cosine.
- **Stage orchestration:** for each configured dataset, run the appropriate subset of {T02 ingest, T03 species, T04 embed, **T11 fisher**, **T12 fusion/GV**, T05 cluster, T06 catalogue, T07 eval, T09 report} in dependency order, against the shared T01 store, capturing per-stage status, duration, stdout/stderr tail, and any structured result paths. **B-track order is `ingest → species → embed → [fisher → fusion] → cluster → {catalogue, eval}`** — the `fisher` (T11) and `fusion` (T12) stages run ONLY when `--signals` is `global+fisher` or `full-funnel` (`fisher` after `embed`, `fusion` after `fisher`, both BEFORE `cluster`; T12 GV reranking is only added under `full-funnel`). `cluster` (T05) always runs BEFORE any review and BEFORE catalogue/eval; T08 review is not part of the automated runner (see Out).
- **Lift reporting:** for the headline demo, T10 evaluates BOTH the global-only backbone and the full-funnel signal set (re-clustering on the global affinity for the baseline) so the bundle can report the lift ("global finds N individuals; add geometric verification → M"). The manifest's `headline` carries the chosen-signal numbers plus an optional `lift` block with the global-only baseline.
- **Bundle assembly:** copy/symlink each stage's primary artifacts into one output folder `demo_bundle/<run_name>/` with a stable internal layout; generate `index.html`, `SUMMARY.md`, and `manifest.json`.
- **Plain-language headline:** read T07's numbers and the store's `count_by` results to render the biologist-facing one-liner(s).
- **Resumability:** `--force` re-runs everything; without it, skip stages whose declared outputs already exist.
- **Dry-run:** `--dry-run` prints the exact ordered command/call plan without executing.
- A smoke path (`--smoke`) that runs the whole thing on a tiny subset (e.g. a capped number of LeopardID2022 identities) so the runner is testable in minutes, not hours.
- `tests/test_run_demo.py` (unit-level: config parsing, plan ordering, manifest shape, bundle assembly from mocked/stubbed stage outputs — NO heavy model runs).
- One additive line in `STATUS_BOARD.md` marking T10 deliverables.

### Out

- Implementing detection (T02), species filtering (T03), embedding (T04), clustering (T05), catalogue rendering (T06), eval metrics (T07), HITL UI (T08), or the Medvednica report internals (T09). T10 **calls** them.
- T08 (human-in-the-loop) is **not** part of the automated runner; T05 clustering runs BEFORE any T08 review, so the demo runner produces clusters that a human *could later* review — but T10 itself never invokes or requires T08. If T08 has separately produced reviewed clusters in the store, T10's downstream consumers (T06/T07) naturally reflect them (re-running T05 preserves human-reviewed rows per T05's policy), but T10 neither triggers nor depends on that.
- Any change to existing pipeline files (`main.py`, `global_embedding.py`, `utility_functions.py`, `constants.py`, etc.) or to T01–T09 modules. T10 only ADDS `reid_demo/run_demo.py`, its config, the bundle templates, and its test.
- Training new models, downloading datasets, network access. T10 assumes datasets already on disk (LeopardID2022 inside WildlifeReID-10k; MedvednicaDS present; ATRW present).
- Hosting/serving; the bundle is a static folder.

## Inputs

- **The T01 store** (`reid_demo/store.py`, default DB `data/reid_demo/reid_demo.sqlite`) — both the destination T02+ write to and the source T05+/T10 read from. T10 owns choosing/passing the DB path to every stage so they all share it.
- **Datasets already on disk** (T10 does not download):
  - LeopardID2022 — inside `data/wildlifedatasets/wildlifereid-10k/versions/7/` (filter metadata `dataset == LeopardID2022`; ingested by `T02.ingest_wildlife_dataset("LeopardID2022", ...)`, which loads it via `utility_functions.load_dataset("LeopardID2022")`). 6806 images; **T02 populates `gt_identity`/`orientation`/`species` at ingest** (T10 reads them, never writes them). **The default B-track dataset.**
  - ATRW — `data/atrw/` (optional second B-track example).
  - MedvednicaDS — `data/MedvednicaDS/` (the A-track real-data report; artifacts already computed).
- **The T02–T09 entry points** (CLIs or functions). T10 must discover these by reading each module's ticket/DATA_CONTRACT-style docstring at integration time. The runner declares each stage's invocation in ONE place (a stage registry, see Implementation notes) so wiring is centralized and easy to fix if a downstream signature differs slightly.
- **Optional config file** (`--config path.yaml|.json`) overriding defaults.

## Outputs

- `reid_demo/run_demo.py` (+ any small templates it needs, e.g. `reid_demo/templates/bundle_index.html`).
- `tests/test_run_demo.py`.
- A **demo bundle** directory, default `demo_bundle/<run_name>/` (run_name defaults to `<primary_dataset>_<YYYYMMDD_HHMMSS>`), containing at least:

```
demo_bundle/<run_name>/
├── index.html              # one-page entry: headline numbers + links to catalogue & report
├── SUMMARY.md              # same content as markdown (for pasting into emails/slides)
├── manifest.json           # machine-readable record of the whole run (see Interface contract)
├── catalogue/              # T06 output for the B-track dataset (copied/linked, self-contained)
│   └── index.html ...
├── accuracy/               # T07 output: the single <dataset>_<tag>.json report (ClusteringReport fields) + any plots
├── medvednica_report/      # T09 output for the A-track (figures + counts page)
└── logs/
    └── <stage>.log         # captured stdout/stderr per stage
```

- `manifest.json` is the **contract** other people/scripts read to know what the bundle contains and whether the run succeeded.

## Interface contract

Downstream/automation depend on EXACTLY the following from T10.

### CLI

```
python -m reid_demo.run_demo \
    [--config PATH]                 # YAML or JSON config; overrides defaults
    [--datasets NAME ...]           # which datasets to process; default: LeopardID2022 MedvednicaDS
    [--primary-dataset NAME]        # the B-track dataset driving the headline; default: LeopardID2022
    [--db PATH]                     # shared T01 store; default: data/reid_demo/reid_demo.sqlite
    [--out-dir PATH]                # bundle root; default: demo_bundle
    [--run-name NAME]               # default: <primary-dataset>_<timestamp>
    [--stages STAGE ...]            # subset to run; default: all applicable per dataset
    [--smoke]                       # tiny subset (cap identities/images) for a fast test run
    [--max-identities N]            # cap for --smoke / quick runs (default when --smoke: 8)
    [--force]                       # re-run stages even if outputs exist
    [--dry-run]                     # print the ordered plan, do not execute
    [--continue-on-error]           # keep going if a non-critical stage fails (record in manifest)
    [-v|--verbose]
```

- Exit code **0** iff all *critical* stages for the primary dataset succeeded AND the bundle (index.html + manifest.json) was written. Non-critical stage failure under `--continue-on-error` still allows exit 0 but is flagged `"status":"partial"` in the manifest. Any unhandled/critical failure → non-zero exit.
- `--dry-run` always exits 0 after printing the plan and writes nothing except (optionally) the plan to stdout.

### Importable API (exact names downstream/tests use)

```python
from dataclasses import dataclass, field
from typing import Optional

# A single stage's invocation spec, resolved from the stage registry.
@dataclass
class StageSpec:
    name: str                      # one of: "ingest","species","embed","cluster","catalogue","eval","report"
    ticket: str                    # "T02".."T09"
    critical: bool                 # if True, failure aborts the run (unless --continue-on-error)
    output_paths: list[str]        # declared artifacts; used for skip-if-exists
    # how to run it (registry fills exactly one of these):
    cli: Optional[list[str]] = None        # argv (without the python exe), run as subprocess
    func: Optional[str] = None             # "module:function" dotted path, called in-process
    kwargs: dict = field(default_factory=dict)

@dataclass
class DemoConfig:
    datasets: list[str]
    primary_dataset: str
    db_path: str
    out_dir: str
    run_name: str
    stages: Optional[list[str]] = None     # None => all applicable
    smoke: bool = False
    max_identities: Optional[int] = None
    force: bool = False
    continue_on_error: bool = False

@dataclass
class StageResult:
    name: str
    ticket: str
    status: str                    # "ok" | "skipped" | "failed"
    seconds: float
    output_paths: list[str]        # resolved, existing artifacts after the stage
    log_path: Optional[str]
    error: Optional[str] = None

@dataclass
class BundleResult:
    run_name: str
    out_dir: str                   # absolute path to demo_bundle/<run_name>/
    status: str                    # "ok" | "partial" | "failed"
    stages: list[StageResult]
    headline: dict                 # the biologist-facing numbers (see manifest schema)
    manifest_path: str
    index_html_path: str

def build_config(args) -> DemoConfig: ...
def plan_stages(config: DemoConfig) -> list[StageSpec]:
    """Resolve, per dataset, the ordered list of StageSpecs in dependency order
    (ingest -> species? -> embed -> cluster -> catalogue/eval for B-track;
     report for A-track). Pure; no side effects. Used by --dry-run and tests."""

def run_demo(config: DemoConfig) -> BundleResult:
    """Execute plan_stages(config), assemble the bundle, write manifest/index/summary,
    return BundleResult. Idempotent w.r.t. existing outputs unless config.force."""

def assemble_bundle(config: DemoConfig, stage_results: list[StageResult]) -> BundleResult:
    """Pure-ish: gather stage artifacts into demo_bundle/<run_name>/, render
    index.html + SUMMARY.md + manifest.json from stage outputs + store counts.
    Separated so tests can call it with stubbed StageResults (no model runs)."""

def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry. Returns process exit code."""
```

### `manifest.json` schema (the bundle contract)

```json
{
  "schema_version": 1,
  "run_name": "LeopardID2022_20260609_1431",
  "created_at": "2026-06-09T14:31:07",
  "status": "ok",                       // "ok" | "partial" | "failed"
  "db_path": "data/reid_demo/reid_demo.sqlite",
  "primary_dataset": "LeopardID2022",
  "datasets": ["LeopardID2022", "MedvednicaDS"],
  "command": "python -m reid_demo.run_demo --datasets ...",
  "headline": {                          // biologist-facing; nulls if eval not run
    "dataset": "LeopardID2022",
    "individuals_found": 412,            // == T07 n_found_clusters
    "individuals_true": 430,            // == T07 n_true_individuals (from gt_identity); null for unlabeled field data
    "pct_photos_correctly_grouped": 94.0,   // read verbatim from T07 on the 0..100 scale (e.g. 94.0, NEVER a 0..1 fraction)
    "candidate_new_individuals": 17,    // count is_candidate_new=1
    "crops_total": 6806
  },
  "stages": [
    {"name":"ingest","ticket":"T02","dataset":"LeopardID2022","status":"ok",
     "seconds":12.4,"rows_after":6806,"output_paths":[...],"log":"logs/ingest_LeopardID2022.log"},
    {"name":"species","ticket":"T03","dataset":"LeopardID2022","status":"ok","seconds":0.3,
     "rows_after":6806,"output_paths":[],"log":"logs/species_LeopardID2022.log"},
    {"name":"embed","ticket":"T04","dataset":"LeopardID2022","status":"ok","seconds":903.1, ...},
    {"name":"cluster","ticket":"T05", ...},
    {"name":"catalogue","ticket":"T06","output_paths":["catalogue/index.html"], ...},
    {"name":"eval","ticket":"T07","output_paths":["accuracy/LeopardID2022_demo.json"], ...},
    {"name":"report","ticket":"T09","dataset":"MedvednicaDS","output_paths":["medvednica_report/index.html"], ...}
  ],
  "store_counts": {                      // from T01 count_by(), per primary dataset
    "by_species": {"leopard": 6806},
    "by_cluster_size_histogram": {"1": 30, "2": 55, "...": 0},
    "review_status": {"unreviewed": 6806}
  },
  "bundle": {
    "index_html": "index.html",
    "summary_md": "SUMMARY.md",
    "catalogue_dir": "catalogue",
    "accuracy_dir": "accuracy",
    "medvednica_report_dir": "medvednica_report"
  }
}
```

- `headline.pct_photos_correctly_grouped`, `individuals_true`, and `individuals_found` are read from **T07's SINGLE structured output file** `evaluations/clustering/<dataset>_<tag>.json` (copied into the bundle under `accuracy/`). T10 reads the trio from that ONE file — **not two files** — mapping `pct_photos_correctly_grouped` (verbatim, **0..100 scale**, e.g. `94.0`), `n_true_individuals → individuals_true`, and `n_found_clusters → individuals_found`. Do NOT recompute clustering metrics in T10 — only re-express T07's numbers as "X% of photos correctly grouped" / "N vs M individuals". If T07 did not run (e.g. A-track-only), these are `null` and the headline degrades gracefully.
- All `output_paths` in the manifest are **relative to the bundle dir** so the folder is portable when zipped/copied.

### Stage registry (single source of wiring truth — required artifact)

T10 must centralize how each downstream stage is invoked in one dict/table inside `run_demo.py` (call it `STAGE_REGISTRY`), keyed by stage name, declaring per dataset-track: the `cli`/`func` to call, which flags carry the shared `--db`/`--dataset`/`--out`/`--max-identities`, the `output_paths` to check for skip-if-exists, and `critical`. This is the ONE place to edit when a downstream signature differs from the placeholder. The registry MUST be readable/inspectable by `--dry-run`.

## Existing code to reuse (real paths)

- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/reid_demo/store.py` — **T01 store**, the spine. Use `connect(db_path)`, `count_by(conn, column, *, dataset=...)`, `query_records(...)`, `to_dataframe(conn, dataset=...)`, `export_records(...)` for headline numbers and the `store_counts` block. Read `reid_demo/DATA_CONTRACT.md` for the schema and join keys (`record_id`, `(dataset, cluster_id)`, `is_candidate_new`, `gt_identity`).
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/utility_functions.py` — `load_dataset(subset)` (lines 26-91) is how datasets resolve (LeopardID2022 → raw `WildlifeReID10k` metadata; `data/wildlifedatasets/wildlifereid-10k/versions/7/metadata.csv` has `identity,path,date,orientation,species,split,dataset,cluster_id`). T10 only needs this to **validate** a requested dataset exists before invoking T02; T02 owns the actual ingest.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/constants.py` — follow its style for any path constants (`ROOT_DIR = os.path.dirname(os.path.abspath(__file__))`); reuse `WILD_DATASET_PATH` (line 62) and `EVALUATION_DIR` (line 57) to validate/locate inputs. Do NOT edit it.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/main.py` — reference only, for **how the repo wires a multi-stage run via argparse** (`--ds NAME ...`, `--count`, `--train`, lines 36-210) and how stages are sequenced; mirror its CLI ergonomics (`--datasets` ≈ `--ds`). Do NOT call `main.py`; the open-set demo path is T02–T09, not the closed-set `main.py`.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/run_final_comparisons.sh` — reference only, as a precedent for "one script kicks off a batch of runs and collects results."
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/data/MedvednicaDS/` — the A-track inputs T09 consumes (already-computed `megadetector_results.json`, `animals_classified.json`, `detections_cleaned.json`, `animal_crops/`, `trail_cam_data.csv`); T10 only points T09 at this folder.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/tools/evaluate_reid_embeddings.py` — reference only, the existing convention for writing a JSON results file with `model_info`/`metrics`/`per_dataset` (lines 350-493); T07 produces the open-set analog and T10 reads it.
- The repo venv: `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/venv/bin/python` — all subprocess stages launch with this interpreter (`sys.executable` when run under the venv).

## Implementation notes

- **Do not reinvent stages.** Every heavy operation is a T02–T09 call. T10's job is *ordering, plumbing the shared `--db`/dataset/out args, capturing results, and assembling a folder*. Keep `run_demo.py` thin and mostly declarative (the `STAGE_REGISTRY` + a generic `_run_stage(spec)` executor that handles subprocess vs in-process, timing, log capture, skip-if-exists, and `StageResult` construction).
- **Two tracks per the config:**
  - *B-track* (LeopardID2022 default, ATRW optional): `ingest → species → embed → cluster → {catalogue, eval}`. The `species` stage is **NOT skipped**: it runs the cheap, model-free `T03.set_known_species(conn, dataset=..., species=...)` (`species="leopard"` for LeopardID2022, `species="tiger"` for ATRW) to stamp a uniform `species` column on every B-track row — only the SpeciesNet **model** branch of T03 is skipped (no GPU classifier on already-known-species data). Ordering is binding: `cluster` (T05) runs BEFORE `catalogue`/`eval` (and, conceptually, before any T08 review). `eval` is critical-for-headline but `--continue-on-error`-eligible; `catalogue` is the showpiece (critical).
  - *A-track* (MedvednicaDS): `report` only (T09). Species filtering already done; do NOT re-run detection/embedding for the demo.
  Encode this as `DATASET_TRACKS = {"LeopardID2022":"B", "ATRW":"B", "MedvednicaDS":"A"}` (extensible), and the per-track known-species map `B_TRACK_SPECIES = {"LeopardID2022":"leopard", "ATRW":"tiger"}`, and have `plan_stages` pick the stage list per track (B-track always includes the `species` stage wired to `set_known_species` with the mapped species).
- **Shared store, multi-run:** pass the same `--db` to every stage; rely on the T01 `dataset` column so LeopardID2022 and MedvednicaDS rows coexist. Headline/`store_counts` always filter by the relevant dataset via `count_by(conn, ..., dataset=...)`.
- **Resolving downstream entry points robustly:** downstream tickets may expose a CLI (`python -m reid_demo.<mod> ...`) or a function. Put the *expected* invocation in `STAGE_REGISTRY` with a clearly-commented placeholder where the exact flag names are TBD, and make `_run_stage` fail with a precise, actionable error ("stage 'cluster' (T05): expected CLI `python -m reid_demo.cluster --db ... --dataset ...`; got non-zero exit / missing module — update STAGE_REGISTRY['cluster']") rather than a stack trace. List these assumptions in Open questions so the integrating run can fix wiring in one place. Prefer subprocess execution (isolation, clean log capture, no GPU-state bleed between stages) over in-process import for the heavy stages (embed/cluster/catalogue).
- **Skip-if-exists / `--force`:** before running a stage, if all its declared `output_paths` exist (and `--force` not set), mark `status="skipped"` and reuse them. For store-mutating stages (ingest/embed/cluster), additionally treat "the store already has rows for this dataset with the relevant column populated" as a completion signal (e.g. cluster is done if `query_records(dataset=..., where_sql="cluster_id IS NOT NULL")` is non-empty). Use the T01 API for these checks; never hand-write SQL against `detections` outside the documented `where_sql` hook.
- **`--smoke` / `--max-identities`:** plumb the cap straight down to **`T02.ingest_wildlife_dataset(subset, max_identities=N, ...)`** (T02's documented `max_identities` parameter caps the number of DISTINCT identities ingested), so fewer crops flow through species/embed/cluster. `--smoke` defaults `--max-identities` to 8 if not given. The smoke run must complete the full B-track (`ingest → species → embed → cluster → catalogue → eval`) on a handful of identities so CI/dev can exercise wiring without GPU-hours.
- **Headline derivation (no metric recomputation):** read T07's **SINGLE** JSON file `evaluations/clustering/<dataset>_<tag>.json` (path declared in `STAGE_REGISTRY['eval'].output_paths`; copied into the bundle under `accuracy/`). Its top-level keys are the `ClusteringReport` fields. Read the headline trio straight from that one file: `pct_photos_correctly_grouped` (verbatim, **already on the 0..100 scale** — do not multiply/divide by 100), `n_true_individuals → individuals_true`, and `n_found_clusters → individuals_found`. `candidate_new_individuals` comes from `count_by(conn,'is_candidate_new',dataset=...)` on the store. If a number is unavailable (e.g. eval didn't run), set `null` and omit that sentence from the headline — never fabricate, never recompute clustering metrics.
- **Bundle assembly is pure-ish and separately testable:** `assemble_bundle(config, stage_results)` takes already-produced `StageResult`s, copies their artifacts under `demo_bundle/<run_name>/`, and renders `index.html`/`SUMMARY.md`/`manifest.json`. Tests call it with hand-built `StageResult`s pointing at tiny fixture files (a 1-page fake catalogue, a fake T07 `<dataset>_<tag>.json` with `pct_photos_correctly_grouped` on the 0..100 scale) so the assembly + manifest + headline logic is validated with **no model execution**. Copy (don't symlink) so the bundle is portable when zipped.
- **index.html / SUMMARY.md content (biologist-first):** top: the headline sentence(s) in plain English ("Found 412 individuals vs 430 known; 94% of photos correctly grouped; 17 possible new individuals to review."). Then two cards: "(A) Your footage, filtered" → link into `medvednica_report/` with the empty-frame % and species counts; "(B) Proof we can ID individuals" → link into `catalogue/` and a short accuracy table. ML metrics (ARI/V-measure/homogeneity from T07) go in a collapsible "Technical details" section. No JS frameworks; plain HTML + inline CSS so it opens offline by double-click.
- **Logging:** each stage's stdout+stderr captured to `logs/<stage>_<dataset>.log` (tail included in `StageResult.error` on failure). Use stdlib `logging`/`subprocess`; no new dependencies beyond what T01–T09 already pull (pandas optional, via T01's `to_dataframe`).
- **Determinism / reproducibility:** record the exact resolved command line, the git commit (`git rev-parse HEAD` best-effort), Python version, and the config in `manifest.json` so the bundle is auditable.
- Add ONE additive line to `STATUS_BOARD.md` under T10; do not touch other tickets' rows.

## Acceptance criteria

- [ ] New files exist and no existing repo file is modified except an additive line in `STATUS_BOARD.md`: `reid_demo/run_demo.py`, `tests/test_run_demo.py`, and any `reid_demo/templates/*` used by the bundle. (`reid_demo/store.py` etc. from T01 are NOT modified.)
- [ ] `python -c "from reid_demo.run_demo import build_config, plan_stages, run_demo, assemble_bundle, main, DemoConfig, StageSpec, StageResult, BundleResult"` succeeds (every contracted name importable).
- [ ] `python -m reid_demo.run_demo --help` prints usage including `--config`, `--datasets`, `--primary-dataset`, `--db`, `--out-dir`, `--run-name`, `--stages`, `--smoke`, `--max-identities`, `--force`, `--dry-run`, `--continue-on-error`.
- [ ] `python -m reid_demo.run_demo --dry-run` exits 0, writes nothing, and prints an ordered plan that, for the default config, lists B-track stages `ingest → species → embed → cluster → catalogue → eval` for `LeopardID2022` (the `species` stage resolved to `T03.set_known_species(..., species="leopard")`) and A-track stage `report` for `MedvednicaDS`, each with the resolved invocation and `--db data/reid_demo/reid_demo.sqlite`. The plan must show `cluster` BEFORE `catalogue`/`eval`.
- [ ] `plan_stages(DemoConfig(...))` is pure (no filesystem/subprocess side effects) and returns `StageSpec`s in correct dependency order per dataset track (`ingest` before `species` before `embed`; `embed` before `cluster`; `cluster` before `catalogue`/`eval`; `report` independent).
- [ ] `assemble_bundle(config, stubbed_stage_results)` (fixtures, no model runs) produces `demo_bundle/<run_name>/` containing `index.html`, `SUMMARY.md`, `manifest.json`, and copied stage artifacts under `catalogue/`, `accuracy/`, `medvednica_report/`; all `output_paths` in the manifest are relative to the bundle dir.
- [ ] `manifest.json` validates against the documented schema: has `schema_version`, `run_name`, `status ∈ {ok,partial,failed}`, `headline` (with `individuals_found`/`pct_photos_correctly_grouped` populated from the fixture T07 output or `null`), a `stages` list with per-stage `status`/`seconds`/`output_paths`, and a `store_counts` block.
- [ ] The headline is derived, not invented: given a fixture T07 single JSON (`accuracy/<dataset>_<tag>.json` with keys `pct_photos_correctly_grouped` on the 0..100 scale, `n_true_individuals`, `n_found_clusters`) and a seeded store, `individuals_found` (=`n_found_clusters`)/`individuals_true` (=`n_true_individuals`)/`pct_photos_correctly_grouped` (verbatim, e.g. `94.0`)/`candidate_new_individuals` match the fixture/store values; if eval output is absent, those fields are `null` and `index.html` omits the corresponding sentence without error.
- [ ] Skip-if-exists works: running `assemble_bundle`/`run_demo` twice without `--force` marks already-present stages `"skipped"` and does not re-run them; with `--force` it re-runs (status `"ok"`).
- [ ] `--continue-on-error` with a deliberately failing non-critical stage yields `manifest.status == "partial"` and process exit 0; a failing **critical** stage (or no `--continue-on-error`) yields a non-zero exit and `manifest.status == "failed"` (manifest still written).
- [ ] `index.html` opens offline (no external network/JS-CDN dependencies) and contains the plain-language headline plus links to `catalogue/` and `medvednica_report/`; `SUMMARY.md` contains the same headline text.
- [ ] `python -m reid_demo.run_demo --smoke --max-identities 4` plumbs the cap through to `T02.ingest_wildlife_dataset(subset, max_identities=4, ...)`, completes the full B-track (`ingest → species → embed → cluster → catalogue → eval`) on the capped subset and produces a valid bundle, OR — if downstream stage CLIs are not yet finalized — fails with the precise "update STAGE_REGISTRY['<stage>']" message (not an opaque traceback) AND `tests/test_run_demo.py` still passes using stubbed stages. (Wiring assumptions must be listed in Open questions / a `STAGE_REGISTRY` comment block.)
- [ ] `python -m pytest tests/test_run_demo.py -q` passes under the repo venv with no GPU/model execution (stages stubbed/mocked).
- [ ] `STATUS_BOARD.md` has exactly one added line for T10; no other ticket's rows changed.

## How to verify

```bash
cd /home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project
source venv/bin/activate

# 1. Import surface
python -c "from reid_demo.run_demo import build_config, plan_stages, run_demo, assemble_bundle, main, DemoConfig, StageSpec, StageResult, BundleResult; print('OK')"

# 2. Help + dry-run plan (no execution, exit 0)
python -m reid_demo.run_demo --help
python -m reid_demo.run_demo --dry-run ; echo "dry-run exit=$?"
python -m reid_demo.run_demo --dry-run --datasets LeopardID2022 MedvednicaDS --primary-dataset LeopardID2022

# 3. plan_stages purity + ordering
python - <<'PY'
from reid_demo.run_demo import build_config, plan_stages, DemoConfig
cfg = DemoConfig(datasets=["LeopardID2022","MedvednicaDS"], primary_dataset="LeopardID2022",
                 db_path="data/reid_demo/reid_demo.sqlite", out_dir="demo_bundle",
                 run_name="test_plan")
specs = plan_stages(cfg)
names = [(s.ticket, s.name) for s in specs]
print(names)
order = {n:i for i,(_,n) in enumerate(names) if _ }  # last wins is fine for this check
assert any(s.name=="ingest" for s in specs)
ing = next(i for i,s in enumerate(specs) if s.name=="ingest")
spe = next(i for i,s in enumerate(specs) if s.name=="species")   # B-track set_known_species NOT skipped
emb = next(i for i,s in enumerate(specs) if s.name=="embed")
clu = next(i for i,s in enumerate(specs) if s.name=="cluster")
cat = next(i for i,s in enumerate(specs) if s.name=="catalogue")
assert ing < spe < emb < clu < cat, names   # cluster (T05) BEFORE catalogue
assert any(s.name=="report" for s in specs)
print("ordering OK")
PY

# 4. Bundle assembly from STUBBED stage outputs (no models) -> manifest + index
python - <<'PY'
import json, os, tempfile, pathlib
from reid_demo.run_demo import assemble_bundle, DemoConfig, StageResult
tmp = tempfile.mkdtemp()
# fake stage artifacts
cat = pathlib.Path(tmp,"fake_cat"); cat.mkdir(); (cat/"index.html").write_text("<h1>catalogue</h1>")
acc = pathlib.Path(tmp,"fake_acc"); acc.mkdir()
# T07 writes ONE file (keys = ClusteringReport fields); pct is on the 0..100 scale.
(acc/"LeopardID2022_demo.json").write_text(json.dumps(
    {"v_measure":0.93,"adjusted_rand_index":0.91,
     "n_found_clusters":412,"n_true_individuals":430,
     "pct_photos_correctly_grouped":94.0}))
rep = pathlib.Path(tmp,"fake_rep"); rep.mkdir(); (rep/"index.html").write_text("<h1>medvednica</h1>")
cfg = DemoConfig(datasets=["LeopardID2022","MedvednicaDS"], primary_dataset="LeopardID2022",
                 db_path="data/reid_demo/reid_demo.sqlite", out_dir=os.path.join(tmp,"demo_bundle"),
                 run_name="stub_run")
results = [
  StageResult(name="catalogue",ticket="T06",status="ok",seconds=1.0,
              output_paths=[str(cat/"index.html")],log_path=None),
  StageResult(name="eval",ticket="T07",status="ok",seconds=1.0,
              output_paths=[str(acc/"LeopardID2022_demo.json")],log_path=None),
  StageResult(name="report",ticket="T09",status="ok",seconds=1.0,
              output_paths=[str(rep/"index.html")],log_path=None),
]
br = assemble_bundle(cfg, results)
print("bundle:", br.out_dir, "status:", br.status)
mani = json.load(open(br.manifest_path))
assert mani["schema_version"]==1
assert mani["status"] in ("ok","partial","failed")
assert mani["headline"]["individuals_found"]==412   # T07 n_found_clusters
assert mani["headline"]["individuals_true"]==430    # T07 n_true_individuals
assert abs(mani["headline"]["pct_photos_correctly_grouped"]-94.0) < 1e-9   # 0..100 scale, read verbatim
assert os.path.exists(br.index_html_path)
assert os.path.exists(os.path.join(br.out_dir,"catalogue","index.html"))
assert os.path.exists(os.path.join(br.out_dir,"medvednica_report","index.html"))
html = open(br.index_html_path).read()
assert "412" in html and ("94" in html)  # headline rendered
print("assemble_bundle OK")
PY

# 5. Unit tests (stubbed stages; no GPU)
python -m pytest tests/test_run_demo.py -q

# 6. (Optional, heavy) real smoke run if downstream stages are wired:
python -m reid_demo.run_demo --smoke --max-identities 4 --run-name smoke_test ; echo "smoke exit=$?"
#   On success: demo_bundle/smoke_test/index.html + manifest.json exist and headline is populated.
#   If a downstream CLI isn't finalized, expect a precise STAGE_REGISTRY[...] error, not a traceback.

# 7. No unintended edits
git status --porcelain   # only new files under reid_demo/ & tests/ + 1 line in STATUS_BOARD.md
```

## Open questions

1. **Exact downstream entry points.** T02–T09 expose CLIs and/or functions whose final flag names this ticket cannot pin down in advance. T10 centralizes all wiring in `STAGE_REGISTRY` with documented placeholders and fails with an actionable "update STAGE_REGISTRY['<stage>']" message. Confirm the residual flag names with each ticket: does it offer `python -m reid_demo.<mod> --db <path> --dataset <NAME> --out <dir>`? Specifically — does **T05** read embeddings from the store/cache and write `cluster_id`/`cluster_conf`/`is_candidate_new` back to the store (so T10 needs no intermediate file)? (RESOLVED by the binding decisions: **T02** owns B-track ingest via `ingest_wildlife_dataset` and populates `gt_identity`/`orientation`/`species`; **T03** exposes `set_known_species(conn, dataset=..., species=...)`; **T07** writes the SINGLE file `evaluations/clustering/<dataset>_<tag>.json` with the headline trio — these are no longer open.)
2. **Smoke capping mechanism — RESOLVED.** **T02 accepts `max_identities` on `ingest_wildlife_dataset(subset, *, max_identities=None, limit=None, ...)`** (caps DISTINCT identities ingested). `--smoke` plumbs `--max-identities` (default 8) straight through to it, so the full B-track runs on a tiny capped subset without GPU-hours. No fallback / no-op needed.
3. **Which number is "% photos correctly grouped" — RESOLVED.** T10 surfaces **T07's `pct_photos_correctly_grouped` verbatim**, read from the single `evaluations/clustering/<dataset>_<tag>.json` file on the **0..100 scale** (e.g. `94.0`, never a 0..1 fraction). T07 owns the definition; T10 never recomputes. `individuals_found`/`individuals_true` likewise come from that file's `n_found_clusters`/`n_true_individuals`.
4. **A-track headline.** For MedvednicaDS (no `gt_identity`), the headline has no "found vs known" sentence; T10 surfaces only the empty-frame % and species counts from T09. Confirm T09 writes those two numbers in a machine-readable form (JSON) T10 can lift, rather than only baking them into HTML.
5. **ATRW inclusion.** ATRW is optional second proof. Default config ships LeopardID2022 + MedvednicaDS only (fast, sufficient for the pitch); confirm whether the pitch wants ATRW in the default bundle or kept as an opt-in `--datasets LeopardID2022 ATRW MedvednicaDS`.
6. **Bundle delivery format.** Static folder is delivered as-is; confirm whether a one-shot `.zip` of `demo_bundle/<run_name>/` should also be produced (trivial to add; left optional to avoid scope creep).
