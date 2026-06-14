# STATUS BOARD — Lynx Re-ID Demo & Pilot MVP

**Last updated:** 2026-06-12
**North star:** Turn the existing closed-set animal re-ID research code into an **open-set clustering pipeline** that ingests raw camera-trap photos, removes empties, isolates a target species, groups detections by **individual animal** with human-in-the-loop review, and produces a **visual catalogue + a defensible count** — packaged as a credible cold-pitch demo for Croatian national parks (first target: **Risnjak NP**, Eurasian lynx) and the basis of a no-cost pilot on a park's existing footage.

This board is the single source of truth for implementation progress. Every work item is a self-contained ticket under [tickets/](tickets/) that an independent AI agent (or human) can pick up and implement from the ticket alone.

---

## The demo this builds toward (context)

Two honest, complementary halves, both producible by `tickets/T10`:

- **(A) Real local data** — Medvednica camera-trap footage that already has MegaDetector + species classification run on it. Show: *% empty frames removed* and the *species breakdown* on genuine Croatian data. (Ticket **T09**.)
- **(B) Individual-ID capability** — **LeopardID2022** (a spotted big cat, the closest available analog to lynx; ATRW Amur tigers as a second example). Show: the pipeline clustering an unlabeled pile into individuals, validated against the dataset's known identities — a *visual catalogue* + *plain-language accuracy* ("found N individuals vs M known; X% of photos correctly grouped"). (Tickets **T04–T07**, enriched by **T11–T12**.) Framing to the park: *"lynx is the same spotted-cat individual-ID problem; I just need your photos."*

**Signal story (the differentiator):** the demo runs global-embedding clustering as a scalable backbone, then layers in the project's **Fisher vectors + geometric verification** (the 3-tier funnel) as an accuracy upgrade — and reports the **lift** ("global finds N; turn on geometric verification → M"). That fusion is the moat versus generic embedding-clustering tools.

**Explicitly out of scope for M1 (Phase 2 — see Backlog):** full web dashboard, geolocation/maps, spatial capture-recapture / population density, deployment infrastructure.

---

## Milestones

| Milestone | Goal | Definition of done |
|---|---|---|
| **M1 — Demo-ready** | A shareable cold-pitch demo bundle | `python -m reid_demo.demo` (T10) runs end-to-end on LeopardID2022 → produces the individual catalogue (T06) + accuracy numbers (T07, reported for **both** global-only and full-funnel) + the Medvednica filtering report (T09), assembled into one bundle. All M1 tickets ✅. |
| **M2 — Pilot-ready** *(future)* | Run a park's real footage safely at scale | Incremental ingestion of a real lynx footage set; camera-deployment registry (locations + timestamps); NDA-safe data handling; review workflow usable by a non-coder. *(Tickets not yet written — see Backlog.)* |

---

## Status legend

🔴 Not started · 🟡 In progress · 🔵 In review · ✅ Done · ⛔ Blocked

---

## Ticket board (M1)

| ID | Ticket | Status | Depends on | Blocks |
|----|--------|--------|------------|--------|
| [T01](tickets/T01-data-store-and-contract.md) | Data store & detection-record contract | 🔵 | — | T02 T03 T04 T05 T06 T07 T08 T09 T10 T11 T12 |
| [T02](tickets/T02-ingestion-megadetector.md) | Ingestion + MegaDetector adapter (+ labeled-dataset adapter) | 🔵 | T01 | T03 T04 T06 T08 T10 T11 |
| [T03](tickets/T03-species-filter-speciesnet.md) | SpeciesNet species-filter adapter | 🔵 | T01 T02 | T10 |
| [T04](tickets/T04-embedding-service.md) | Embedding service (global / MegaDescriptor) | 🔵 | T01 T02 | T05 T10 T12 |
| [T05](tickets/T05-open-set-clustering.md) | **Open-set clustering engine** (core; pluggable affinity) | 🔵 | T01 T04 | T06 T07 T08 T10 T12 |
| [T06](tickets/T06-catalogue-generator.md) | Visual individual catalogue generator | 🔴 | T01 T02 T05 | T10 |
| [T07](tickets/T07-clustering-evaluation.md) | Clustering evaluation harness | 🔴 | T01 T05 | T10 |
| [T08](tickets/T08-human-review-tool.md) | Human-in-the-loop review tool | 🔴 | T01 T02 T05 | — |
| [T09](tickets/T09-medvednica-report.md) | Medvednica filtering report | 🔵 | T01 | T10 |
| [T10](tickets/T10-demo-assembly.md) | Demo assembly & end-to-end runner (`--signals`) | 🔴 | T01 T02 T03 T04 T05 T06 T07 T09 T11 T12 | — |
| [T11](tickets/T11-local-fisher-service.md) | Local-feature + Fisher-vector service | 🔵 | T01 T02 | T10 T12 |
| [T12](tickets/T12-fusion-gv-reranking.md) | Multi-signal fusion + GV reranking | 🔴 | T01 T04 T05 T11 | T10 |

> Update the **Status** cell here whenever a ticket changes state, and mirror it in the ticket's own header.

> **T01 deliverables (🔵 in review):** `reid_demo/{__init__.py, store.py, DATA_CONTRACT.md}` + `tests/test_store.py` — the shared SQLite detection-record store, the 28-column contract, and the access API every other ticket imports. Read `reid_demo/DATA_CONTRACT.md` before starting any downstream ticket.

> **T02 deliverables (🔵 in review):** `reid_demo/ingest.py` + `tests/test_ingest.py` — ingestion + MegaDetector adapter. Four adapters (`ingest`/`load_detection_frames` for MegaDetector-results & flat `animal_detections.json`, `ingest_from_images` lazy-MegaDetector raw-image path, `ingest_wildlife_dataset` B-track) + `python -m reid_demo.ingest` CLI. **`det_index` is 1-based over kept animal detections in source-file order**; `record_id == make_record_id(stem, det_index)`; new crops `{stem}__crop{idx}.jpg`, legacy `{stem}_crop{idx}_*.jpg` reused. Default dataset `MedvednicaDS`; per-detection `conf >= 0.5` filter (drops empties/persons/vehicles). A-track sets only T02 fields + `orientation="unknown"`; B-track whole-frame records (`bbox=(0,0,1,1)`, `det_index=1`, `detector_conf=1.0`) populate `gt_identity`/`orientation`/`species` from metadata (D1). **species/embeddings/clusters/review intentionally left NULL** for T03/T04/T05/T08. Module imports without torch/megadetector (lazy).

> **T09 deliverables:** `reid_demo/medvednica_report.py` + `tests/test_medvednica_report.py` — read-only Medvednica filtering report (`generate_medvednica_report()` + helpers `parse_crop_filename`/`species_from_classes`/`compute_funnel`/`compute_species_counts` + `python -m reid_demo.medvednica_report` CLI/`--selftest`). Emits `medvednica_report.md`, `figures/{detection_funnel,species_breakdown,example_crops}.png`, and the T10-facing `medvednica_summary.json` from the existing `data/MedvednicaDS/*.json` (no models re-run; D7b empty≠people/vehicle, D3 kept counts trusted on-disk).

> **T03 deliverables (🔵 in review):** `reid_demo/species_filter.py` + `tests/test_species_filter.py` (17 passed). `is_target_species` (matches common name AND full-taxonomy genus: genus `lynx`→lynx; `panthera`+`pardus`/`tigris` disambiguates leopard vs tiger) + `TARGET_SPECIES_ALIASES`; `SpeciesFilterResult`; `ingest_speciesnet_json` (PRIMARY path; **(source_stem, bbox) nearest-match join**, greedy, L2<0.05, NOT positional det_index — D3); `classify_and_filter` (live SpeciesNet CLI fallback, raises clear `RuntimeError` if model/PIL unavailable); `set_known_species` (stamps fixed species + `species_kept=1`, no model, leaves `gt_identity`/`orientation` untouched — D1); `python -m reid_demo.species_filter` CLI. Writes `species`/`species_conf`/`species_class` via `update_species` + `species_kept` via `update_extra` (never raw SQL); **never deletes rows** (`--drop-nontarget` only filters returned `kept_record_ids`). Module imports without torch/speciesnet/PIL (lazy).

> **T04 deliverables (🔵 in review):** `reid_demo/embed.py` + `tests/test_embed.py` (14 passed). `embed_records`/`embed_crops` (MegaDescriptor global embeddings), `get_embedding_matrix(normalize=True)`, `load_embeddings`, `embedding_cache_path`, `EmbedResult`, constants `DEFAULT_EMB_DIR`/`DEFAULT_MODEL_NAME`; `python -m reid_demo.embed` CLI. Writes via `update_embedding(record_id, embedding_ref, embedding_path)`; **vectors stored MODEL-NATIVE dim, RAW — NOT L2-normalized (D2)**; dim read from the matrix (no hard-coded 384). Heavy deps (torch/timm/wildlife-tools) lazily imported; deterministic stub embedding path for tests. *(NOTE: the T04 service module is `reid_demo.embed` (`embed_records`/`embed_crops`), a thin wrapper over the legacy `global_embedding.load_or_build_global_embeddings`; T11 mirrored this `embed`/`fisher` service shape. Downstream T05/T12 import from `reid_demo.embed` — DATA_CONTRACT join-rule & D2 prose reconciled to say so.)*

> **T05 deliverables (🔵 in review):** `reid_demo/cluster.py` + `tests/test_cluster.py` (36 passed). Open-set, flank-aware clustering: pure core `cluster_embeddings`/`cluster_by_flank`/`assignment_confidence` (no DB/IO) + store driver `run_clustering` + `python -m reid_demo.cluster` CLI; `CropClustering`/`ClusterRunSummary`, constants `DEFAULT_BACKEND`/`CLUSTER_BACKENDS`/`DEFAULT_EPS`/`DEFAULT_MIN_SAMPLES`/`DEFAULT_DISTANCE_THRESHOLD`/`NOISE_LABEL`. Two cosine backends (DBSCAN default + threshold agglomerative) over a **precomputed cosine-distance matrix** so the **pluggable affinity (D8)** — optional `(N,N)` similarity matrix or provider callable, default internal global cosine — flows through one code path; **never imports T11/T12**. 3-bucket flank policy `{left, right, other}` in deterministic sorted order with globally-unique ids (D4); singletons/noise → `cluster_id=-1` AND `is_candidate_new=1`, conf `0.0` (D5); confidence = mean intra-cluster cosine sim in `[0,1]` (optional `ScoreCalibrator`). Reads embeddings via T04 `get_embedding_matrix(normalize=True)` (or `--embeddings` pkl override), dim read from the matrix (no hard-coded 384, D2); filters by `species` not `species_kept` (D7). Re-run safety: rows with `review_status != 'unreviewed'` preserved unless `--force`; `--dry-run` computes but writes nothing; deterministic/idempotent. Writes only `cluster_id`/`cluster_conf`/`is_candidate_new` via `update_cluster`.

> **T11 deliverables (🔵 in review):** `reid_demo/fisher.py` + `tests/test_fisher.py` (12 passed; 4 heavy e2e `skipif`-gated on torch+lightglue+crops). `build_fisher_records`/`build_fisher_vectors`, `get_fisher_matrix`, `load_fisher_vectors`, `fisher_cache_path`/`fisher_cache_label`, `FisherResult`, constants `DEFAULT_FISHER_DIR`/`DEFAULT_METHOD`/`DEFAULT_PCA_DIM`; `python -m reid_demo.fisher` CLI. **Reuses** existing research code (`feature_aggregation.ensure_local_descriptors`/`load_or_train_fisher_vectors`/`load_descriptors`, DISK via `feature_extraction.extract_features`) — not reimplemented; FV dim derived (`2·gmm.n_components·pca.n_components_`), no literal. Refs written only via `update_extra` (`fisher_ref`/`fisher_path`/`fisher_label`) — no new columns. Added a descriptor-cache coverage guard so a changed record-set rebuilds rather than silently returning empty. Module imports without heavy deps (lazy).

---

## Dependency graph & build order

```
T01  data contract                       (foundation — build first)
  │
  ├── T02  ingestion ──┬── T03  species filter
  │                    ├── T04  global embeddings ─┐
  │                    └── T11  local → Fisher ─────┤
  │                                                 ▼
  │                    T05  clustering  (global backbone; pluggable affinity)
  │                                                 │
  │                    T12  fusion + GV reranking ──┤   global+Fisher affinity +
  │                         (needs T04, T11, T05)   │   GV on borderline pairs only
  │                                                 ▼
  ├── T09  medvednica report     T06 catalogue · T07 eval · T08 review (← T12 optional)
  │
  └──────────────────────────────────────────────── T10  demo assembly (--signals)
```

Suggested parallel **waves** (each can be worked concurrently once the previous lands):

- **Wave 0:** T01
- **Wave 1:** T02, **T09** (T09 only needs T01 + existing Medvednica JSONs)
- **Wave 2:** T03, T04, **T11**
- **Wave 3:** T05 *(the core backbone)*
- **Wave 4:** T06, T07, T08, **T12**
- **Wave 5:** T10

> The **global backbone path** (T01→T02→T04→T05→T06/T07→T10 with `--signals global`) is a complete demo on its own. T11/T12 are a **fast-follow accuracy layer** — the demo is never blocked on geometric verification.

---

## Binding design decisions (resolved from adversarial review)

These were settled during ticket authoring to keep the set coherent. **They override any contradicting prose inside a ticket — if you find a conflict, these win, and fix the ticket.**

- **D1 — Labeled-data ingestion & ground-truth ownership.** **T02** owns a 4th adapter `ingest_wildlife_dataset(subset, *, max_identities=None, limit=None)` that ingests LeopardID2022/ATRW: one **whole-frame** record per image (`bbox=(0,0,1,1)`, crop = the original image, **no MegaDetector**), populating `gt_identity`, `orientation`, `species` from the WildlifeReID-10k metadata. T02 is the **sole** owner of `gt_identity`/`orientation` for labeled datasets (this removes a would-be dependency cycle with T07).
- **D2 — Embedding contract.** Embeddings are **model-native dim (1536 base / 384 linear_l2 checkpoint), NOT pre-normalized**. Consumers must use `get_embedding_matrix(normalize=True)` and read the dim from the matrix — never hard-code 384, never assume unit norm.
- **D3 — `det_index`, A-track join, store API.** `det_index` = 1-based over kept **animal** detections in MegaDetector source order. T03 joins `animals_classified.json` → records by **(stem, bbox) nearest-match** (no positional guessing). T01 exposes `update_extra()`; trust the on-disk `detections_cleaned.json` for "kept"/empty counts.
- **D4 — Flank/orientation policy.** Cluster `{left, right}` in **separate** buckets; pool `{front, back, down, unknown, ''}` into a single **`other`** bucket (not spot-re-identifiable). T07's flank-aware GT label uses the same `{left, right, other}` convention. Empty `''` → `unknown` at ingest.
- **D5 — Singletons / candidate-new / re-run safety.** Singletons **and** DBSCAN noise → `cluster_id = -1` **and** `is_candidate_new = 1` (the flag downstream keys on). Deterministic bucket ordering. **T05 runs before T08**, and re-running T05 must not silently wipe human review (`review_status != 'unreviewed'` is preserved unless `--force`).
- **D6 — Eval output contract.** T07 writes a **single** `evaluations/clustering/<dataset>_<tag>.json` with `pct_photos_correctly_grouped` (**0–100**), `n_true_individuals`, `n_found_clusters`; T10 reads the headline from it.
- **D7 — Scope/semantics.** T08's two interactive UIs are **optional**; T09 separates "empty" (zero detections) from "people/vehicle" frames; T10's B-track still stamps species via `set_known_species` (only the SpeciesNet model is skipped) and plumbs `--max-identities` into T02.
- **D8 — Multi-signal clustering (signal layer).** Global embedding (**T04**) + clustering (**T05**) are the **scalable M1 backbone** and run standalone. **T11** adds a per-crop **Fisher vector** (local features); **T12** fuses global+Fisher into the clustering affinity (calibrated via `calibration.py`) and runs **geometric verification on *borderline pairs only*** (never N², budget-capped) to sharpen boundaries and prioritize review. T05's affinity is **pluggable** (default global, no hard dep on T11/T12); T08 *optionally* consumes GV scores; T10's `--signals {global|global+fisher|full-funnel}` selects the layer and the demo reports the **global→full-funnel accuracy lift**.

All new code lives under a new package: **`reid_demo/`** (`store.py`, `embed.py`, `cluster.py`, `fisher.py`, `fusion.py`, `demo.py`, …). The store schema is documented by T01 in **`DATA_CONTRACT.md`** — read it before touching any ticket.

---

## Agent working protocol

When you (an AI agent or human) pick up a ticket:

1. **Read** this board, the ticket file, and **`DATA_CONTRACT.md`** (the shared interface — produced by T01). Confirm the tickets you depend on are ✅.
2. **Claim it:** set the ticket header `Status: 🟡 In progress` and `Owner:`, and update this board's table row.
3. **Implement to the ticket** — satisfy every Acceptance criterion, obey the Binding design decisions above, and **do not expand scope** (push extra ideas to the Backlog).
4. **Verify:** run the ticket's *How to verify* commands. Every acceptance criterion must pass.
5. **Hand off:** set `Status: 🔵 In review` (→ ✅ Done after review), update this board, and record the branch/PR. One ticket = one branch = one PR.
6. **Definition of Done:** code merged under `reid_demo/`; acceptance criteria pass; the ticket's interface contract is honoured so downstream tickets can rely on it; no contradiction with the Binding design decisions.

---

## Backlog — Phase 2 (M2, not yet ticketed)

- **Camera-deployment registry** (per-station location + active dates) → unlocks space-time priors, maps, and density math. *(Geolocation is per-station, not per-photo.)*
- **Real footage ingestion at scale** — incremental/idempotent ingest, large transfers, NDA-safe storage of sensitive carnivore-location data.
- **Population density** — Spatial Capture-Recapture (SCR) for marked species (lynx); **Random Encounter Model (REM/REST)** for unmarked species (boar/deer).
- **Web dashboard** — multi-user review + browsing on top of the store.
- **Multi-modal biometric fusion** — space-time priors, sequence aggregation, permanent horns (chamois/mouflon), face (bears) layered into the matching, extending the T12 fusion machinery.
- **Deployment** — packaging, auth, hosting for a park to self-serve.
