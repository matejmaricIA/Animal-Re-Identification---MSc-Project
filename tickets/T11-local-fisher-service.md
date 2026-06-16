# T11 — Local-feature + Fisher-vector service

> **Status:** 🔴 Not started · **Milestone:** M1 (Demo-ready)
> **Depends on:** T01, T02 · **Blocks:** T10, T12
> **Owner:** _unassigned_
>
> Self-contained ticket for the lynx re-identification demo/pilot. See [STATUS_BOARD.md](../STATUS_BOARD.md) for the full plan, the binding design decisions, and the agent working protocol.

---

## Context

We are building a DEMO + PILOT MVP of an **open-set, individual-animal re-identification** system for Eurasian lynx (closest public analog: spotted big cats — LeopardID2022 leopards, ATRW Amur tigers). The repo at `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project` already does CLOSED-SET re-id; the demo pivots the *decision layer only* to open-set clustering. Every demo module reads/writes the **same per-crop "detection record"** through the shared SQLite store defined in **T01** (`reid_demo/store.py`, `reid_demo/DATA_CONTRACT.md`).

The clustering backbone (T04 global MegaDescriptor embedding → T05 cosine clustering) works standalone today. **CURRENT GAP (closed by D8):** that backbone uses ONLY the global embedding as its similarity signal. The repo's existing strength — **local features (DISK/SuperPoint/ALIKED) → Fisher vectors**, and geometric verification — is unused. On spotted cats this matters a lot (the repo's closed-set ELPephants tables show global-only 13.66% → 52% with global+Fisher+GV). D8 adds these as a **SELECTABLE accuracy layer ON TOP of the global backbone, NOT a replacement.**

**This ticket (T11) is the Fisher-vector sibling of T04.** Where T04 (`reid_demo/embed.py`) wraps `global_embedding.py` to produce one cached **global** vector per crop keyed by `record_id`, T11 (`reid_demo/fisher.py`) wraps `feature_extraction.py` (local descriptors; **DISK default**) + `feature_aggregation.py` (PCA + GMM → Fisher vector) to produce one cached **Fisher** vector per crop keyed by `record_id`. T11 is a thin, clean wrapper: you do NOT reimplement descriptor extraction, PCA/GMM fitting, or the Fisher-vector math — you call the existing, proven functions and add store integration + a service shape that **mirrors T04** (a `load_or_build_*` cache wrapper, a `get_fisher_matrix(normalize=True)` read accessor).

Pipeline position (T11 produces a SECOND per-crop vector type alongside T04's global one):

```
raw image --(T02)--> crop + bbox + record  [INSERTS records, both A-track crops & B-track whole-frames]
          --(T03)--> species (optional filter)
          --(T04)--> global embedding   (reid_demo/embed.py)   ─┐
          --(T11 THIS TICKET)--> Fisher vector (reid_demo/fisher.py) ─┤── two signals per crop
          --(T12)--> fused affinity (global+Fisher) + GV rerank ─┘   (consumes T04 + T11)
          --(T05)--> clusters (global backbone; T12 supplies optional fused affinity)
```

### Where T11 sits in the D8 signal layering (binding — do not cross these boundaries)

- T11 **produces and caches** a per-crop Fisher vector and a read-side matrix accessor. **It does NOT fuse, calibrate, cluster, run geometric verification, or touch T05.** Fusion of global+Fisher into a pairwise affinity, calibration, and GV reranking are **T12** (`reid_demo/fusion.py`); T12 imports T11's read API. T11 must NOT import `reid_demo.cluster` (T05), `reid_demo.fusion` (T12), `calibration.py`, or `geometric_verification.py`.
- T11 has **NO hard dependency** on T04/T05/T12 and must build standalone. Its only hard deps are T01 (store) and T02 (which writes the records/crops T11 reads). Like T04, it works off the `crop_path` field in the store, never a hard-coded folder.
- **A-track** (Medvednica unlabeled crops under `data/MedvednicaDS/animal_crops/`, e.g. `02020401_crop1_conf92.jpg`) and **B-track** (LeopardID2022/ATRW whole-frame records where `crop_path` = the original full image, `bbox=(0,0,1,1)`, per D1) are BOTH supported — both are just `{record_id → crop_path}` to this service.

### Critical Fisher-vector facts (verified in repo — do not get these wrong)

1. **DISK is the default local-feature method** for this ticket. In `feature_aggregation.ensure_local_descriptors(image_items, method_name, out_dir)`, `method_name="disk"` routes to `feature_extraction.extract_features(image_items, MODEL_PATH, out_dir)`, which uses `lightglue.DISK(max_num_keypoints=...)`. DISK descriptors are **128-dim** per keypoint. (`"aliked"`/`"superpoint"` route to `extract_features_lightglue`; accept them as alternates but DEFAULT to `"disk"`.)
2. **Fisher-vector dimension is DERIVED, never hard-coded:** `fv_len = 2 * gmm.n_components * pca.n_components_` (feature_aggregation.py:227). With repo defaults `N_COMPONENTS_PCA=128`, `N_COMPONENTS_GMM=256` → `2*256*128 = 65536`. Read the dim from `gmm.n_components` / `pca.n_components_` or from `next(iter(fv_dict.values())).shape[0]` — **never assume 65536** (PCA dim is configurable via `pca_dim`).
3. **Fisher vectors are ALREADY L2-normalized at compute time** (feature_aggregation.py:217–220: power-normalize `sign(fv)*sqrt(|fv|)` then L2). This is the OPPOSITE of T04 (whose global vectors are stored RAW/un-normalized). So for T11 the on-disk cache is already unit-norm; `get_fisher_matrix(normalize=True)` re-normalizes defensively (idempotent, epsilon-guarded) so downstream cosine == dot regardless. A crop with **zero descriptors** gets a **zero Fisher vector** (`np.zeros(fv_len)`, feature_aggregation.py:232) — handle this: a zero vector cannot be L2-normalized (guard with epsilon) and is a legitimate "no local features" outcome, not a failure.
4. **PCA/GMM are FIT once per (dataset, method, config)** on a sample of all descriptors, then every crop's Fisher vector is computed against that shared model. This is unlike T04 (which is per-image independent): the Fisher pipeline has a **fit phase** (needs the whole descriptor pool) and a **transform phase** (per crop). Cache the fitted PCA/GMM so re-runs reuse them (the existing `load_or_train_fisher_vectors` already does this).

## Objective

Deliver one self-contained module `reid_demo/fisher.py` providing a clean **"Fisher-vectorize a store of crops"** batch API that mirrors T04's service shape:

1. Pull candidate detection records from the T01 store (optionally species-filtered, scoped by `dataset`), build `{record_id → crop_path}`.
2. Extract local descriptors (DISK default) for those crops by **calling the existing `feature_aggregation.ensure_local_descriptors` / `feature_extraction.extract_features`** (reuse, do not reimplement). Descriptors cache as HDF5 keyed by `record_id`.
3. Fit-or-load PCA+GMM and compute per-crop Fisher vectors by **calling the existing `feature_aggregation.load_or_train_fisher_vectors`** (reuse). Cache as a `Dict[str, np.ndarray]` pickle keyed by `record_id`.
4. Write a *reference* (`fisher_ref = record_id`, `fisher_path = <cache pickle>`) back into the T01 store via `update_extra` (the sanctioned no-schema-change escape hatch — T11 adds NO new columns).
5. Provide read-side helpers `load_fisher_vectors` and `get_fisher_matrix(conn, ..., normalize=True)` that **T12** calls to get an `(N, D)` matrix + ordered `record_id` list — exactly mirroring T04's `get_embedding_matrix`.

Out of scope: global embeddings (T04), fusion/calibration/GV/affinity (T12), clustering (T05), review (T08), catalogue/eval/report, the `--signals` flag (T10). You only turn crops into cached Fisher vectors and record references.

## Scope

### In
- New module `reid_demo/fisher.py` implementing the Interface contract below.
- Reuse of `feature_aggregation.ensure_local_descriptors`, `feature_aggregation.load_or_train_fisher_vectors`, `feature_aggregation.load_descriptors`, `feature_extraction.extract_features` / `extract_features_lightglue` — **call them, never copy them.**
- Descriptor HDF5 cache dir + Fisher-vector pickle cache under `data/reid_demo/fisher/...` (parent dirs auto-created), keyed by `record_id`.
- Store writes through `reid_demo.store.update_extra(conn, record_id, "fisher_ref", record_id)` and `update_extra(conn, record_id, "fisher_path", <pkl>)` only (NO raw SQL, NO new columns — T11 owns no `detections` column; D1 forbids touching T04/T05/T08 columns).
- Idempotency / additive caching: `only_missing=True` skips records that already have a `fisher_ref` in `extra_json` AND a present key in the cache; reuse on-disk HDF5 + PCA/GMM + FV pickle when present.
- Optional species filter (`only_species`) so we Fisher-vectorize only the target spotted-cat species when T03 has run (mirror T04's `only_species` semantics: exact lowercase equality).
- Graceful per-record failure (missing/unreadable crop file): pre-filter on existence, skip + count + continue.
- A read helper `get_fisher_matrix` + `load_fisher_vectors`.
- CLI `python -m reid_demo.fisher`.
- Tests `tests/test_fisher.py`.
- One additive line in `STATUS_BOARD.md` marking T11 deliverables; additive re-export lines in `reid_demo/__init__.py`.

### Out
- Editing `feature_extraction.py`, `feature_aggregation.py`, `constants.py`, `main.py`, `reid_demo/store.py`, `reid_demo/embed.py`, `reid_demo/cluster.py`, or any existing file (except the two additive touches above).
- Defining the record schema / DB primitives (T01 — import them). Adding any `detections` column (use `extra_json` via `update_extra`).
- Global embeddings (T04). Fusion / calibrated pairwise affinity / GV reranking / candidate-pair shortlisting (T12). Clustering, T05's pluggable affinity arg (T12 wires that). The `--signals` flag (T10).
- Training new local-feature models or downloading checkpoints. Use the existing DISK extractor and the existing PCA/GMM fit-on-this-dataset flow.

## Inputs

- **A T01 store** (SQLite) already containing detection records (from T02), each with a valid `crop_path` pointing at a crop image on disk (A-track tight crops, or B-track whole-frame paths), and optionally `species`/`species_conf` (from T03). Default DB path `data/reid_demo/reid_demo.sqlite` (`reid_demo.store.DEFAULT_DB_PATH`).
- **A `dataset` selector** (e.g. `MedvednicaDS`, `LeopardID2022`) to scope which records to process. One DB can hold several runs; PCA/GMM are fit per `(dataset, method, config)`.
- **Crop image files** on disk (e.g. `data/MedvednicaDS/animal_crops/02020401_crop1_conf92.jpg`). You read whatever path is in `crop_path`.
- Optional `method` (default `"disk"`), `pca_dim` (default from `constants.N_COMPONENTS_PCA` = 128), `only_species`, `limit`, `device` (pass-through; CPU must work).

Runtime inputs you do NOT need: raw frames, MegaDetector JSON, SpeciesNet output, global embeddings. You operate purely on `record_id → crop_path` from the store.

## Outputs

- `reid_demo/fisher.py` (new).
- A **local-descriptor HDF5 cache** (`descriptors.h5` + `keypoints.h5`) under a per-`(dataset, method)` directory, keyed by `record_id` (the HDF5 dataset key == `record_id`). Produced by reusing `ensure_local_descriptors` / `extract_features`. Recommended dir: `data/reid_demo/fisher/{dataset}_{method}_descriptors/`.
- A **Fisher-vector cache pickle** per `(dataset, method, pca_dim)` at `data/reid_demo/fisher/{dataset}_{fisher_label}.pkl` where `fisher_label = fisher_cache_label(method, pca_dim)` (see contract). Format: `Dict[str, np.ndarray]`, **keys are `record_id`**, values 1-D `float32` of length `2*gmm.n_components*pca.n_components_` (default 65536). Already L2-normalized (zero vector for no-descriptor crops). This is the SAME on-disk format the existing repo Fisher caches use, except keys are `record_id`s.
- A **fitted PCA + GMM pickle pair** cached alongside (produced by `load_or_train_fisher_vectors` via `PCA_PATH`/`GMM_PATH`/`FISHER_VECTORS` templates) so re-runs reuse the fitted models.
- Store side effects: for each successfully Fisher-vectorized record, `extra_json` gains `"fisher_ref" = record_id` and `"fisher_path" = <that pickle path>` (and optionally `"fisher_label"`) via `update_extra`. No `detections` column added.
- `tests/test_fisher.py`.
- Additive lines in `reid_demo/__init__.py` and `STATUS_BOARD.md`.

## Interface contract

Downstream (T12 especially) imports EXACTLY these. Do not rename. Names deliberately mirror T04 (`embed.py`) with `embedding`→`fisher`.

### Module-level constants

```python
DEFAULT_FISHER_DIR: str = "data/reid_demo/fisher"
DEFAULT_METHOD: str = "disk"            # DISK is the binding default local-feature method
DEFAULT_PCA_DIM: int  = N_COMPONENTS_PCA   # imported from constants.py (=128); do NOT hard-code 128 in logic
```

### Result dataclass (mirror T04's `EmbedResult`)

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class FisherResult:
    dataset: Optional[str]
    cache_path: str               # the .pkl holding the Fisher vectors
    method: str                   # local-feature method, e.g. "disk"
    fisher_label: str             # fisher_cache_label(method, pca_dim)
    pca_dim: int                  # PCA components actually used
    gmm_components: Optional[int] # GMM components (from the fitted/loaded GMM); None if nothing computed
    fisher_dim: Optional[int]     # observed vector length = 2*gmm_components*pca_dim (None if nothing computed)
    n_total: int                  # candidate records considered
    n_fishered: int               # records assigned a Fisher vector this run
    n_skipped: int                # already had fisher_ref (only_missing)
    n_failed: int                 # crop missing/unreadable (excluded before extraction)
    n_zero_vectors: int           # crops that produced an all-zero FV (no local descriptors)
    failed_ids: list = field(default_factory=list)   # record_ids excluded for missing/unreadable crops
```

### Public functions (exact signatures)

```python
def fisher_cache_label(method: str = DEFAULT_METHOD, pca_dim: int = DEFAULT_PCA_DIM) -> str:
    """Deterministic label distinguishing caches by method + pca_dim, e.g. 'disk_pca128'.
    Used in the Fisher pickle filename so different methods/dims get distinct caches.
    Mirrors the role of global_embedding.global_embedding_cache_label in T04."""

def fisher_cache_path(dataset: Optional[str], method: str = DEFAULT_METHOD,
                      pca_dim: int = DEFAULT_PCA_DIM,
                      cache_dir: str = DEFAULT_FISHER_DIR) -> str:
    """Deterministic path: f'{cache_dir}/{dataset or "all"}_{fisher_cache_label(method, pca_dim)}.pkl'.
    Creates no files; just returns the path (parent dir created by writers). Mirrors T04 embedding_cache_path."""

def build_fisher_vectors(crop_paths: Dict[str, str], cache_path: str, *,
                         dataset: Optional[str] = None,
                         method: str = DEFAULT_METHOD,
                         pca_dim: int = DEFAULT_PCA_DIM,
                         desc_dir: Optional[str] = None,
                         device=None) -> Dict[str, np.ndarray]:
    """Low-level (mirror of embed.embed_crops). Given {record_id -> crop_image_path}:
      (1) ensure local descriptors exist for these crops in an HDF5 cache keyed by record_id,
          by calling feature_aggregation.ensure_local_descriptors([(rid, path), ...], method, desc_dir)
          (which routes 'disk' -> extract_features, 'aliked'/'superpoint' -> extract_features_lightglue);
      (2) fit-or-load PCA+GMM and compute per-crop Fisher vectors by calling
          feature_aggregation.load_or_train_fisher_vectors(ds_tag=..., method_name=method,
              cache_suffix=..., pca_dim=pca_dim, descriptors_loader=<load this HDF5>);
      (3) persist the {record_id -> FV} dict to cache_path (Dict[str,np.ndarray] pickle) and return it.
    Keys with missing/unreadable crop files are pre-dropped (not in result). Crops with zero descriptors
    yield an all-zero FV (kept in result; counted as zero-vector by the caller). Does NOT touch the store.
    NOTE: ensure_local_descriptors short-circuits if desc_dir already exists; if you need to ADD crops to
    an existing descriptor cache, extract the new ones into a fresh temp dir and merge HDF5 keys, OR
    document that descriptor extraction is whole-set per (dataset, method). See Implementation notes."""

def build_fisher_records(conn, *,
                         dataset: Optional[str] = None,
                         method: str = DEFAULT_METHOD,
                         pca_dim: int = DEFAULT_PCA_DIM,
                         cache_dir: str = DEFAULT_FISHER_DIR,
                         only_missing: bool = True,
                         only_species: Optional[str] = None,
                         limit: Optional[int] = None,
                         device=None) -> FisherResult:
    """Main entry (mirror of embed.embed_records). (1) query_records(conn, dataset=dataset,
    [species filter]) to get candidate records (when only_missing, exclude records that already have a
    'fisher_ref' in extra_json AND a present key in the cache); (2) build {record_id -> crop_path},
    dropping missing-crop records into failed_ids; (3) call build_fisher_vectors into
    fisher_cache_path(...); (4) for each record now present in the cache, set extra_json keys via
    update_extra(conn, record_id, 'fisher_ref', record_id) and update_extra(conn, record_id,
    'fisher_path', cache_path); (5) return FisherResult. only_species matches store 'species' by exact
    lowercase equality (same convention as T04)."""

def load_fisher_vectors(cache_path: str) -> Dict[str, np.ndarray]:
    """Load a Fisher cache pickle. Thin wrapper over pickle for downstream symmetry (mirror T04
    load_embeddings). Raises FileNotFoundError with a clear message if absent."""

def get_fisher_matrix(conn, *,
                      dataset: Optional[str] = None,
                      normalize: bool = True,
                      only_clustered: bool = False) -> Tuple[np.ndarray, list]:
    """READ side for T12 (mirror of T04 get_embedding_matrix). Query records whose extra_json has a
    non-null 'fisher_ref', load each vector from its 'fisher_path'/'fisher_ref', stack into an (N, D)
    float32 matrix in stable sorted-record_id order, and return (matrix, record_ids). Fisher vectors are
    already L2-normalized on disk; normalize=True re-normalizes defensively with an epsilon guard
    (zero vectors stay zero, NOT NaN). Groups by fisher_path so each pickle loads once. Raises a clear
    RuntimeError naming record_id + path if 'fisher_ref' is set but the key is missing from the pickle
    (stale cache). 'only_clustered' is a convenience pass-through (cluster_id NOT NULL), default off."""
```

### CLI

```
python -m reid_demo.fisher --db <path> [--dataset NAME] [--method disk]
        [--pca-dim 128] [--cache-dir data/reid_demo/fisher]
        [--only-species "leopard"] [--all] [--limit N] [--device cuda|cpu]
```
- Default `--db` = `reid_demo.store.DEFAULT_DB_PATH`.
- `--all` sets `only_missing=False` (recompute the store-side refs for all matching records; PCA/GMM/FV pickle is still reused if present — true recompute requires deleting the cache by hand, like T04's `--all`).
- On exit print one summary line, e.g.:
  `T11 fisher: dataset=MedvednicaDS method=disk pca=128 dim=65536 total=120 fishered=120 skipped=0 failed=0 zero=0 cache=data/reid_demo/fisher/MedvednicaDS_disk_pca128.pkl`
- Exit 0 on success; non-zero if 0 records were found to process AND none already had `fisher_ref` (nothing to do is worth surfacing), or on unhandled exception.

### File-format guarantees for downstream (T12)

- Fisher cache pickle: `Dict[str, np.ndarray]`, keys are `record_id`, values 1-D `float32` at the **derived dim `2*gmm.n_components*pca.n_components_`** (default 65536; NEVER hard-coded). Already L2-normalized (zero vector for no-descriptor crops). Same on-disk format as existing repo `fisher_vectors_*.pkl`, so T12 MAY `pickle.load` directly but SHOULD use `load_fisher_vectors` / `get_fisher_matrix`.
- `fisher_ref == record_id` is guaranteed for records this ticket processed; `(fisher_path, fisher_ref)` (both inside `extra_json`) is the join into the pickle — symmetric with T04's `(embedding_path, embedding_ref)` columns.
- T12 obtains the Fisher matrix via `get_fisher_matrix(conn, normalize=True)` and the global matrix via T04's `get_embedding_matrix(conn, normalize=True)`; it aligns the two by `record_id` (the shared join key) to build the fused pairwise affinity. T11 guarantees the Fisher matrix's id order is stable sorted-`record_id` so alignment is a dict lookup.

## Existing code to reuse (real paths)

- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/feature_aggregation.py`
  - `ensure_local_descriptors(image_items, method_name, out_dir)` (lines 42–56): builds the HDF5 descriptor/keypoint cache if `out_dir` doesn't exist; routes `"disk"→extract_features`, `"aliked"/"lightglue"→extract_features_lightglue(...,"aliked")`, `"superpoint"→...("superpoint")`. **`image_items` accepts `(id, path)` tuples** (see `_parse_image_item` in feature_extraction.py:44 — pass `[(record_id, crop_path), ...]` so HDF5 keys are `record_id`).
  - `load_or_train_fisher_vectors(*, ds_tag, method_name, cache_suffix, pca_dim=N_COMPONENTS_PCA, descriptors=None, descriptors_loader=None) -> (PCA, GaussianMixture, dict)` (lines 59–85): loads cached `(pca, gmm, fv_dict)` if all three pickles exist (paths `PCA_PATH/GMM_PATH/FISHER_VECTORS` `.format(ds_tag, method_name, cache_suffix)`), else fits PCA+GMM on `stack_all_descriptors(...)` and computes FVs. **The returned `fv_dict` is `{image_id -> FV}` — pass descriptors keyed by `record_id` so FV keys are `record_id`.** This is your primary fit+transform call.
  - `load_descriptors(descriptors_file)` (lines 88–95): `{key -> np.ndarray}` from an HDF5 `descriptors.h5`. Use as the `descriptors_loader` payload.
  - `compute_fisher_vectors(image_descriptors, pca, gmm)` (lines 224–242) and `compute_fisher_vector` (172–222): the FV math (power-norm + L2; `fv_len = 2*gmm.n_components*pca.n_components_`; zero vector for empty descriptors). Read to understand the contract; you call them indirectly via `load_or_train_fisher_vectors`.
  - `feature_descriptor_dir(base_dir, method_name, split_name, seg_tag)` (lines 33–39): the repo's descriptor-dir naming convention; you may follow its style but your dir lives under `DEFAULT_FISHER_DIR`.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/feature_extraction.py`
  - `extract_features(image_paths, model_path, output_dir, max_keypoints=MAX_KEYPOINTS)` (lines 95–134): **DISK** extractor via `lightglue.DISK`; writes `descriptors.h5`+`keypoints.h5` keyed by `img_id`; 128-dim DISK descriptors; gpu cache cleared per image. The DEFAULT path.
  - `extract_features_lightglue(image_paths, output_dir, feature_type="aliked", max_keypoints=MAX_KEYPOINTS)` (lines 157–213): SuperPoint/ALIKED/DoGHardNet/SIFT alternates.
  - `_parse_image_item(item)` (lines 44–50): accepts a path string OR `(id, path)` tuple — the mechanism by which you key HDF5 by `record_id`.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/constants.py` — import `N_COMPONENTS_PCA` (=128, line 43), `N_COMPONENTS_GMM` (=256, line 42), `MAX_KEYPOINTS` (=2500, line 102), and the cache templates `PCA_PATH`/`GMM_PATH`/`FISHER_VECTORS` (lines 46–48, `'./data/{}/..._{}_{}.pkl'`) if you need to predict/inspect the fitted-model cache paths. **Do NOT edit constants.py.** Do NOT hard-code 128/256/65536 in your own logic — derive.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/utility_functions.py` — `load_stuff`/`save_stuff` (lines 286–306) are the pickle (de)serializers `load_or_train_fisher_vectors` uses for PCA/GMM/FV. Reference only.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/reid_demo/store.py` (T01) — import `connect`, `query_records`, `update_extra`, `get_record`, `DetectionRecord`, `DEFAULT_DB_PATH`, `make_record_id`. Find work via `query_records(conn, dataset=..., [species=...])`; record results via `update_extra(conn, record_id, "fisher_ref"/"fisher_path", value)`. **Do NOT write raw SQL against `detections`; do NOT add columns.**
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/reid_demo/embed.py` (T04) — **the structural template to mirror.** Same function shapes (`*_cache_label`, `*_cache_path`, `build_*` low-level, `build_*_records` store entry, `load_*`, `get_*_matrix`), same `only_missing`/`only_species`/`limit`/`device` params, same per-record-failure pattern, same `Result` dataclass style. Read it before writing `fisher.py`; T11 should feel like its sibling. **Difference:** T04 stores RAW vectors + L2-normalizes at read; T11 vectors are ALREADY L2-normalized on disk (read-time normalize is defensive/idempotent). T04 writes dedicated columns (`embedding_ref`/`embedding_path`); T11 writes into `extra_json` (no columns to spare).
- Repo venv: `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/venv/bin/python` (has torch/lightglue/h5py/sklearn/numpy). `python` is NOT on PATH; use `venv/bin/python`.

## Implementation notes

- **Reuse over reinvention.** Hard dependencies are `feature_aggregation.ensure_local_descriptors` (descriptor cache) and `feature_aggregation.load_or_train_fisher_vectors` (PCA/GMM fit + per-crop FV). Your value-add is: store integration, `record_id`-keyed descriptor+FV caching, species filtering, robust per-record skip, the read-side matrix helper, and the T04-mirrored service shape. Do NOT re-open the DISK extractor, re-fit PCA/GMM by hand, or re-implement the FV math.
- **Key everything by `record_id`.** Pass `[(record_id, crop_path), ...]` (tuple list) to `ensure_local_descriptors` so the HDF5 dataset keys are `record_id`s (via `_parse_image_item`). When you `load_descriptors(...)` that HDF5 and feed it to `load_or_train_fisher_vectors`, the returned FV dict is keyed by `record_id`, which is the demo's universal join key (T01 contract).
- **Cache-suffix discipline.** Call `load_or_train_fisher_vectors(ds_tag=<dataset or "all">, method_name=method, cache_suffix=f"reiddemo_pca{int(pca_dim)}", pca_dim=int(pca_dim), descriptors_loader=...)`. The `cache_suffix` keeps the PCA/GMM/FV pickles (written under `PCA_PATH/GMM_PATH/FISHER_VECTORS` = `./data/{ds_tag}/..._{method}_{suffix}.pkl`) distinct from the legacy closed-set caches. THEN copy/save the returned `fv_dict` to your contracted `fisher_cache_path(...)` under `DEFAULT_FISHER_DIR` (the file T12 reads). (You may instead point the loader at your own path layout, but the simplest correct approach is: let `load_or_train_fisher_vectors` manage the fitted-model cache, and you own the `{record_id->FV}` pickle at `fisher_cache_path`.)
- **Derive the dim, never hard-code.** After `load_or_train_fisher_vectors` returns `(pca, gmm, fv_dict)`, set `fisher_dim = 2 * gmm.n_components * pca.n_components_` and `gmm_components = gmm.n_components` for `FisherResult`. Equivalently read `next(iter(fv_dict.values())).shape[0]`. Never assume 65536.
- **Zero-descriptor crops.** `compute_fisher_vectors` returns a zero vector for a crop with no descriptors (and prints a skip line). These are VALID outputs (a crop where DISK found nothing), not failures: keep them in the cache, count them in `FisherResult.n_zero_vectors`, still write `fisher_ref`. In `get_fisher_matrix(normalize=True)` guard the L2 division with an epsilon (`v / max(||v||, 1e-12)`) so a zero vector stays zero, never NaN. Document that T12 may down-weight or ignore zero-FV crops.
- **Missing-file safety (mirror T04).** Before extraction, drop any `record_id` whose `crop_path` is not `os.path.exists` into `failed_ids` (the DISK/lightglue extractors `cv2.imread`/`load_image` would otherwise raise). Optionally wrap extraction so one corrupt JPEG doesn't kill the batch; the simplest correct approach is pre-filter on existence.
- **Descriptor-cache additivity caveat (read carefully).** `ensure_local_descriptors` **short-circuits if `out_dir` already exists** and writes the HDF5 in `"w"` (overwrite) mode otherwise — it is whole-set, not additive. For the demo's batch runs this is fine (extract all of a dataset's crops at once). For `only_missing` re-runs you do NOT need to re-extract: if the descriptor dir and the FV pickle already exist, `load_or_train_fisher_vectors` returns the cached `(pca, gmm, fv_dict)` and you only write store refs for records missing them. If you must add NEW crops to an existing descriptor dir, extract the new `(rid, path)` set into a temp dir, then merge its HDF5 keys into the existing `descriptors.h5`/`keypoints.h5` (h5py copy), and re-run the FV transform. **Document whichever behavior you ship**; the minimum bar is: first run extracts+fits+computes for the dataset; `only_missing` re-run recomputes nothing and is near-instant.
- **`only_missing` semantics.** When `True`, a record is a "skip" if its `extra_json` already contains a non-null `fisher_ref` AND that `record_id` is a key in the on-disk FV pickle (`fisher_cache_path`). Use `get_record` / `query_records` + a JSON parse of `extra_json` to detect this (there is no `has_fisher` filter in T01 — `query_records` only knows `has_embedding`; filter in Python on `extra_json`).
- **Store update loop.** After `build_fisher_vectors` returns the merged FV dict, iterate candidate records; for each whose `record_id` is a key in the dict, call `update_extra(conn, record_id, "fisher_ref", record_id)` then `update_extra(conn, record_id, "fisher_path", cache_path)`. (`update_extra` merges into `extra_json`, preserving other keys like T03's `species_kept`.) Records absent from the dict (failed/zero) are not ref'd unless zero (zero IS ref'd — it has a vector).
- **`get_fisher_matrix`.** Query records, parse `extra_json` for `fisher_ref`/`fisher_path`, group by `fisher_path`, `load_fisher_vectors` each pickle once, look up by `fisher_ref`, stack in sorted-`record_id` order, `float32`. If `normalize=True`, defensively L2-normalize rows with epsilon guard. Raise a clear `RuntimeError(f"stale fisher cache: record {rid} ref'd but missing from {path}")` on a missing key.
- **Device / CPU.** Accept `device` as `None|str|torch.device`; the DISK/lightglue extractors default to cuda-if-available internally. Do not hard-require CUDA; CPU must work (slow is fine for the demo's small piles). PCA/GMM are sklearn (CPU).
- **No new heavy deps.** numpy, torch, lightglue, h5py, sklearn, tqdm are already in the venv. Add nothing.
- **`reid_demo/__init__.py`:** append re-exports `from .fisher import build_fisher_records, build_fisher_vectors, get_fisher_matrix, load_fisher_vectors, FisherResult, fisher_cache_path, fisher_cache_label, DEFAULT_FISHER_DIR, DEFAULT_METHOD, DEFAULT_PCA_DIM` (additive; do not disturb T01/T04 exports).
- **`STATUS_BOARD.md`:** append one line, e.g. `- [x] T11 Local-feature + Fisher-vector service: reid_demo/fisher.py (build_fisher_records, get_fisher_matrix) + tests/test_fisher.py`. Do not edit other tickets' status.
- **Determinism.** Descriptor sampling for PCA/GMM uses a fixed RNG seed (42) inside `stack_all_descriptors`; GMM fit is otherwise deterministic enough for the demo. Keep all returned lists/matrices in sorted-`record_id` order.

## Acceptance criteria

- [ ] `reid_demo/fisher.py` exists; only `reid_demo/__init__.py` (additive) and one line in `STATUS_BOARD.md` are touched beyond new files; no existing pipeline file (`feature_extraction.py`, `feature_aggregation.py`, `constants.py`, `reid_demo/store.py`, `reid_demo/embed.py`, etc.) is modified.
- [ ] All contracted names import: `from reid_demo.fisher import build_fisher_records, build_fisher_vectors, load_fisher_vectors, get_fisher_matrix, FisherResult, fisher_cache_path, fisher_cache_label, DEFAULT_FISHER_DIR, DEFAULT_METHOD, DEFAULT_PCA_DIM`.
- [ ] `fisher_cache_label("disk", 128) == "disk_pca128"` (or your documented deterministic equivalent) and `fisher_cache_path("MedvednicaDS", "disk", 128)` ends in `MedvednicaDS_disk_pca128.pkl` under `DEFAULT_FISHER_DIR`.
- [ ] `build_fisher_vectors({rid: '<real crop>.jpg', ...}, '/tmp/fv.pkl', dataset='t')` on a handful of real Medvednica crops returns `{rid: np.ndarray}` with 1-D `float32` vectors all of length `2*gmm.n_components*pca.n_components_` (with defaults = 65536, but assert via the derived formula, NOT the literal), writes `/tmp/fv.pkl`, and a second call reusing the caches recomputes no descriptors/PCA/GMM.
- [ ] **Dim is derived, not hard-coded:** grep shows no literal `65536` (and no bare `2*256*128`) in `reid_demo/fisher.py` logic; `FisherResult.fisher_dim == 2 * gmm_components * pca_dim`.
- [ ] On a temp DB seeded with ≥3 T01 records pointing at real crops, `build_fisher_records(conn, dataset='MedvednicaDS')` writes ONE Fisher pickle under `data/reid_demo/fisher/`, and for each processed record `get_record(...).extra_json` parses to a dict with `fisher_ref == record_id` and `fisher_path == <pickle path>`. No `detections` column is added (PRAGMA table_info unchanged vs T01's 28 columns).
- [ ] Re-running with `only_missing=True` yields `FisherResult.n_fishered == 0` and `n_skipped == <count>` (no recompute).
- [ ] `build_fisher_records(..., only_species='leopard')` processes only records whose `species` matches case-insensitively; others get no `fisher_ref`.
- [ ] Every key in the produced pickle is a `record_id` present in the store; `load_fisher_vectors(path)[record_id]` equals the stored vector; `get_fisher_matrix(conn, dataset=...)` returns `(matrix, record_ids)` with `matrix.shape[0] == len(record_ids) == N`, and rows are unit-norm when `normalize=True` (per-row norm `1.0 ± 1e-5`) EXCEPT all-zero (no-descriptor) rows which stay exactly zero (no NaN).
- [ ] A record with a non-existent `crop_path` is excluded, counted in `FisherResult.n_failed`, its id in `failed_ids`, and the run still Fisher-vectorizes the others without crashing.
- [ ] `get_fisher_matrix` raises `RuntimeError` (naming the `record_id` + path) when `fisher_ref` is set in `extra_json` but the vector is missing from the referenced pickle (stale cache), instead of a bare `KeyError`.
- [ ] `python -m reid_demo.fisher --db <path> --dataset MedvednicaDS` exits 0 and prints a summary line with method/pca/dim/total/fishered/skipped/failed/zero/cache; `--limit N` caps processed records to N.
- [ ] T11 imports neither `reid_demo.cluster`, `reid_demo.fusion`, `reid_demo.embed`, `calibration`, nor `geometric_verification` (verify by grep on the import lines) — boundary respected; standalone build works.
- [ ] `tests/test_fisher.py` passes under `venv/bin/python -m pytest tests/test_fisher.py -q`.

## How to verify

```bash
cd /home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project
PY=venv/bin/python   # 'python' is not on PATH

# 0. Sanity: existing Fisher pipeline produces a derived-dim, L2-normalized FV from a real crop (slow on CPU)
$PY - <<'PY'
import glob, numpy as np, tempfile, os
from feature_aggregation import ensure_local_descriptors, load_or_train_fisher_vectors, load_descriptors
crops = sorted(glob.glob('data/MedvednicaDS/animal_crops/*.jpg'))[:8]
items = [("r%d"%i, c) for i,c in enumerate(crops)]
d = tempfile.mkdtemp()
ddir = os.path.join(d, "desc")
ensure_local_descriptors(items, "disk", ddir)
pca, gmm, fv = load_or_train_fisher_vectors(
    ds_tag="tmpfish", method_name="disk", cache_suffix="probe_pca128", pca_dim=128,
    descriptors_loader=lambda: load_descriptors(os.path.join(ddir, "descriptors.h5")))
v = next(iter(fv.values()))
exp = 2 * gmm.n_components * pca.n_components_
print("fv_dim", v.shape, "expected", exp, "norm", round(float(np.linalg.norm(v)),4))
assert v.ndim == 1 and v.shape[0] == exp
PY

# 1. Import surface
$PY -c "from reid_demo.fisher import build_fisher_records, build_fisher_vectors, load_fisher_vectors, get_fisher_matrix, FisherResult, fisher_cache_path, fisher_cache_label, DEFAULT_FISHER_DIR, DEFAULT_METHOD, DEFAULT_PCA_DIM; print('import OK')"

# 1b. Boundary: T11 must not import T05/T12/T04/calibration/GV
$PY - <<'PY'
import ast
src = open("reid_demo/fisher.py").read()
banned = {"reid_demo.cluster","reid_demo.fusion","reid_demo.embed","calibration","geometric_verification"}
for n in ast.walk(ast.parse(src)):
    if isinstance(n, ast.ImportFrom) and (n.module or "") in banned: raise SystemExit("banned import: "+n.module)
    if isinstance(n, ast.Import):
        for a in n.names:
            if a.name in banned: raise SystemExit("banned import: "+a.name)
print("boundary OK")
PY

# 2. End-to-end on a tiny seeded DB built from REAL Medvednica crops
$PY - <<'PY'
import glob, json, numpy as np
from reid_demo.store import connect, upsert_record, query_records, get_record, DetectionRecord, make_record_id
from reid_demo.fisher import build_fisher_records, get_fisher_matrix, load_fisher_vectors

conn = connect('/tmp/reid_fisher.sqlite')
crops = sorted(glob.glob('data/MedvednicaDS/animal_crops/*.jpg'))[:6]
rids = []
for i, c in enumerate(crops, start=1):
    stem = 'probe%d' % i; rid = make_record_id(stem, 1); rids.append(rid)
    upsert_record(conn, DetectionRecord(
        record_id=rid, source_image='x', source_stem=stem, det_index=1, crop_path=c,
        bbox_x=0.0,bbox_y=0.0,bbox_w=1.0,bbox_h=1.0,
        dataset='MedvednicaDS', species='eurasian lynx'))

res = build_fisher_records(conn, dataset='MedvednicaDS')
print('fishered', res.n_fishered, 'dim', res.fisher_dim, 'gmm', res.gmm_components, 'cache', res.cache_path)
assert res.n_fishered == len(crops) and res.n_failed == 0
assert res.fisher_dim == 2 * res.gmm_components * res.pca_dim

# store side effects live in extra_json (no new column)
for rid in rids:
    ex = json.loads(get_record(conn, rid).extra_json)
    assert ex.get('fisher_ref') == rid and ex.get('fisher_path') == res.cache_path

cache = load_fisher_vectors(res.cache_path)
assert set(cache.keys()) >= set(rids)

M, ids = get_fisher_matrix(conn, dataset='MedvednicaDS', normalize=True)
assert M.shape[0] == len(ids) == len(crops)
norms = np.linalg.norm(M, axis=1)
# unit norm except all-zero (no-descriptor) rows
assert np.all((np.abs(norms - 1.0) < 1e-5) | (norms < 1e-9)), norms
assert not np.isnan(M).any()

res2 = build_fisher_records(conn, dataset='MedvednicaDS')
assert res2.n_fishered == 0 and res2.n_skipped == len(crops)
print('idempotent + extra_json refs OK; all assertions passed')
PY

# 3. Missing-crop robustness
$PY - <<'PY'
from reid_demo.store import connect, upsert_record, DetectionRecord, make_record_id
from reid_demo.fisher import build_fisher_records
conn = connect('/tmp/reid_fisher_fail.sqlite')
rid = make_record_id('ghost', 1)
upsert_record(conn, DetectionRecord(record_id=rid, source_image='x', source_stem='ghost',
    det_index=1, crop_path='data/MedvednicaDS/animal_crops/NOPE.jpg',
    bbox_x=0.0,bbox_y=0.0,bbox_w=1.0,bbox_h=1.0, dataset='MedvednicaDS', species='eurasian lynx'))
res = build_fisher_records(conn, dataset='MedvednicaDS')
assert res.n_failed == 1 and rid in res.failed_ids
print('missing-crop handled:', res.failed_ids)
PY

# 4. No schema drift (T01's 28 columns unchanged)
$PY - <<'PY'
from reid_demo.store import connect, COLUMNS, TABLE_NAME
conn = connect('/tmp/reid_fisher.sqlite')
cols = [r[1] for r in conn.execute(f'PRAGMA table_info({TABLE_NAME})').fetchall()]
assert cols == COLUMNS, cols
print('no schema drift; columns ==', len(cols))
PY

# 5. CLI
$PY -m reid_demo.fisher --db /tmp/reid_fisher.sqlite --dataset MedvednicaDS ; echo "exit=$?"

# 6. Tests
$PY -m pytest tests/test_fisher.py -q
```

## Open questions

1. **Where to persist the `{record_id -> FV}` pickle vs the fitted-model cache.** `load_or_train_fisher_vectors` already writes PCA/GMM/FV pickles under `constants.PCA_PATH/GMM_PATH/FISHER_VECTORS` (`./data/{ds_tag}/...`). The contract additionally requires the FV dict at `fisher_cache_path(...)` under `DEFAULT_FISHER_DIR` (the file T12 reads). Recommended: let the existing function own the fitted-model cache, and T11 copies/saves the returned `fv_dict` to `fisher_cache_path`. Flag if T12 would rather read the legacy `FISHER_VECTORS` path directly (then `fisher_cache_path` becomes a thin alias).
2. **Descriptor-cache additivity.** `ensure_local_descriptors` is whole-set (short-circuits on existing dir, else overwrites). For the demo this is acceptable (extract a dataset's crops once). Confirm with T10 that batch-per-dataset extraction is the intended usage; if incremental crop addition is needed, the HDF5-merge path in Implementation notes is the fallback.
3. **`fisher_ref`/`fisher_path` in `extra_json` vs dedicated columns.** T11 owns no `detections` column (T04 took `embedding_ref`/`embedding_path`; D1 forbids T11 editing the schema/other tickets' columns), so it stores its refs in `extra_json` via `update_extra`. T12 reads them from `extra_json`. Confirm T12 expects this (it parses `extra_json["fisher_ref"]`, NOT a column). If a future schema bump (SCHEMA_VERSION 2) adds real `fisher_ref`/`fisher_path` columns, `get_fisher_matrix` would switch its lookup — keep the read path encapsulated so that swap is local.
4. **DISK vs ALIKED default.** D8/STATUS_BOARD name DISK as the default local feature; the closed-set tables that motivate the lift used DISK+Fisher+GV. ALIKED/SuperPoint are accepted alternates via `method`. Confirm T10's headline demo passes `method="disk"` (the default) unless an ablation says otherwise.
5. **`only_species` matching.** Mirrors T04: exact lowercase equality on the `species` column. Confirm the canonical species string T03/T02 write for the target (e.g. `leopard` vs `panthera pardus` vs `eurasian lynx`); T10 can call once per target species.
