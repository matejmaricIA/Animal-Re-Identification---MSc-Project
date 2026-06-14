# T04 — Embedding service

> **Status:** 🔴 Not started · **Milestone:** M1 (Demo-ready)
> **Depends on:** T01, T02 · **Blocks:** T05, T10, T12
> **Owner:** _unassigned_
>
> Self-contained ticket for the lynx re-identification demo/pilot. See [STATUS_BOARD.md](../STATUS_BOARD.md) for the full plan, the binding design decisions, and the agent working protocol.

---

## Context

We are building a DEMO + PILOT MVP of an **open-set, individual-animal re-identification** system for Eurasian lynx (closest public analog: spotted big cats — LeopardID2022 leopards, ATRW Amur tigers). The repo at `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project` already does CLOSED-SET re-id; the demo pivots the *decision layer only* to open-set clustering. Every demo module reads/writes the **same per-crop "detection record"** through the shared SQLite store defined in **T01** (`reid_demo/store.py`, see DATA CONTRACT below).

The pipeline flows left to right, one crop = one row:

```
raw image --(T02 MegaDetector)--> crop + bbox + camera + timestamp   [INSERTS records]
          --(T03 SpeciesNet)----> species + species_conf (+ keep/drop)
          --(T04 THIS TICKET)---> embedding_ref + embedding_path      [you populate these]
          --(T05 Clustering)-----> cluster_id + cluster_conf          [reads your embeddings]
          --(T06/T07/T08/T09/T10): downstream consumers
```

**This ticket (T04) is the Embedding service, and it is the AUTHORITATIVE definition of the embedding contract** that every consumer — T05 (clustering) especially, but also T07 — must match. It takes the crop images that T02 wrote to disk (and that T03 optionally species-filtered) and produces **cached MegaDescriptor-L-384 global embeddings**, one vector per crop, keyed by `record_id`. You are a thin, clean wrapper around the existing, proven `global_embedding.py` code — you do NOT reimplement model loading, preprocessing, or the pickle cache format. You then write a *reference* to each vector back into the T01 store (`embedding_ref`, `embedding_path`) so T05 can stack the vectors into a matrix and discover individuals.

### THE EMBEDDING CONTRACT (binding — T05/T07 MUST conform to this, not the reverse)

- Embeddings are stored at their **MODEL-NATIVE dimension**: **1536** for the base model `megadescriptor-l-384`, and **384 ONLY** for a `linear_l2` checkpoint. **Do NOT hard-code the dimension anywhere** — never assume 384.
- Stored vectors are **RAW / NOT L2-normalized** (the base model emits ~18.2-norm 1536-dim vectors; a `linear_l2` checkpoint emits already-unit 384-dim vectors). We store exactly what the model returns, un-normalized.
- **Every consumer obtains its matrix via the T04 read API `get_embedding_matrix(conn, ..., normalize=True)`** and reads the dimension **from the returned matrix** (`matrix.shape[1]`), never from a constant. `normalize=True` L2-normalizes rows at read time so cosine == dot product regardless of which model/checkpoint produced the vectors.
- The actual observed dim is surfaced via `EmbedResult.embedding_dim`; nothing downstream may assume "384-dim L2-normalized".

Crops on disk already exist for the real Medvednica run at `data/MedvednicaDS/animal_crops/` (4,194 crops named like `02020401_crop1_conf92.jpg`); for LeopardID2022 / ATRW, T02/T10 will populate crops and records. You must work off the `crop_path` field in the store, not off any specific folder name.

### Critical model fact (verified in repo — do not get this wrong)

`global_embedding.extract_global_embeddings(image_paths, model_name="megadescriptor-l-384")` (the **no-checkpoint** path) loads `megadescriptor.load_megadescriptor_l_384` = `timm` model `hf-hub:BVRA/wildlife-mega-L-384`. Its raw output is a **1536-dim, NOT-L2-normalized** vector (verified: existing cache `data/cowdataset/global_embeddings_test_megadescriptor-l-384_unsegmented.pkl` has shape `(1536,)`, dtype `float32`, norm ~18.2). Only the *checkpoint* path (`_CheckpointMegaDescriptor` with `projection_head="linear_l2"`) yields a **384-dim L2-normalized** vector. **Therefore the dimension is MODEL-NATIVE (1536 base / 384 linear_l2 checkpoint) and is NEVER hard-coded; embeddings are stored RAW (un-normalized).** Store them exactly as `global_embedding.py` returns them, and L2-normalize at READ time via `get_embedding_matrix(normalize=True)` so T05 gets unit vectors for cosine clustering regardless of which model/checkpoint produced them. Consumers read the dim from the returned matrix and surface it via `EmbedResult.embedding_dim` — they must not assume "384-dim L2-normalized".

## Objective

Deliver one self-contained module `reid_demo/embed.py` providing a clean **"embed a list/store of crops"** batch API that:
1. Pulls un-embedded detection records from the T01 store (optionally species-filtered).
2. Computes their MegaDescriptor-L-384 embeddings by **calling the existing `global_embedding.extract_global_embeddings`** (reuse, do not reimplement).
3. Caches them as a `Dict[str, np.ndarray]` pickle keyed by `record_id`, in the exact format `global_embedding.py` uses.
4. Writes `embedding_ref` (= `record_id`) and `embedding_path` back into the store via `reid_demo.store.update_embedding`.
5. Provides a read-side helper `get_embedding_matrix` that T05 calls to get an `(N, D)` L2-normalized matrix + ordered `record_id` list.

Out of scope: detection/cropping (T02), species labels (T03), clustering/decision logic (T05), any catalogue/eval/report. You only turn crops into cached vectors and record references.

## Scope

### In
- New module `reid_demo/embed.py` implementing the Interface contract below.
- Reuse of `global_embedding.extract_global_embeddings` / `global_embedding.global_embedding_cache_label` and `megadescriptor.load_megadescriptor_l_384` — call them, never copy them.
- Cache pickle written to `data/reid_demo/embeddings/{dataset}_{model_label}.pkl` (parent dirs auto-created), format identical to existing caches (`Dict[str, np.ndarray]`).
- Store writes through `reid_demo.store.update_embedding` only (never raw SQL on `detections`).
- Idempotency: `only_missing=True` skips records that already have `embedding_ref` AND a present key in the cache; reuse the existing on-disk pickle when present (T05 may re-run; embedding is expensive on GPU).
- Optional species filter (`only_species`) so we embed only the target spotted-cat species when T03 has run.
- Graceful per-record failure (missing crop file, unreadable image): skip, count, continue.
- A read helper `get_embedding_matrix` + `load_embeddings`.
- CLI `python -m reid_demo.embed`.
- Tests `tests/test_embed.py`.
- One additive line in `STATUS_BOARD.md` marking T04 deliverables; additive re-export lines in `reid_demo/__init__.py`.

### Out
- Editing `global_embedding.py`, `megadescriptor.py`, `main.py`, `reid_demo/store.py`, `constants.py`, or any other existing file (except the two additive touches above).
- Defining the record schema or DB access primitives (those are T01 — import them).
- Clustering, similarity thresholds, flank logic, candidate-new flags (T05).
- Local features / Fisher vectors / geometric verification (different feature track entirely; not needed for the demo decision layer).
- Fine-tuning, training, or downloading new checkpoints. Default to the public base model; merely *accept* a `checkpoint_path` and pass it through unchanged.

## Inputs

- **A T01 store** (SQLite) already containing detection records (from T02), each with a valid `crop_path` pointing at a crop image on disk, and optionally `species`/`species_conf` (from T03). Default path `data/reid_demo/reid_demo.sqlite`.
- **A `dataset` selector** (e.g. `MedvednicaDS`, `LeopardID2022`) to scope which records to embed. One DB can hold several runs.
- **Crop image files** on disk (e.g. `data/MedvednicaDS/animal_crops/02020401_crop1_conf92.jpg`, or crops T02 wrote under `data/reid_demo/crops/`). You read whatever path is in `crop_path`.
- Optional `checkpoint_path` (passed through to `global_embedding`); optional `device`; optional `only_species`.

Runtime inputs you do NOT need: raw frames, MegaDetector JSON, SpeciesNet output. You operate purely on `record_id -> crop_path` from the store.

## Outputs

- `reid_demo/embed.py` (new).
- An embedding cache pickle per (dataset, model) at `data/reid_demo/embeddings/{dataset}_{model_label}.pkl` where `model_label = global_embedding.global_embedding_cache_label(model_name, checkpoint_path)`. Format: `Dict[str, np.ndarray]`, **keys are `record_id`**, values are 1-D `float32` arrays as returned by `global_embedding.extract_global_embeddings` (typically `(1536,)` for the base model). This is byte-compatible with the existing repo caches except the keys are `record_id`s.
- Store side effects: for each successfully embedded record, `embedding_ref = record_id` and `embedding_path = <that pickle path>` set via `reid_demo.store.update_embedding` (which refreshes `updated_at`).
- `tests/test_embed.py`.
- Additive lines in `reid_demo/__init__.py` and `STATUS_BOARD.md`.

## Interface contract

Downstream tickets (T05 especially) import EXACTLY these. Do not rename.

### Module-level constants

```python
DEFAULT_EMB_DIR: str = "data/reid_demo/embeddings"
DEFAULT_MODEL_NAME: str = "megadescriptor-l-384"
```

### Result dataclass

```python
from dataclasses import dataclass, field

@dataclass
class EmbedResult:
    dataset: Optional[str]
    cache_path: str               # the .pkl that holds the vectors
    model_name: str
    model_label: str              # global_embedding_cache_label(model_name, checkpoint_path)
    embedding_dim: Optional[int]  # observed vector length, e.g. 1536 (None if nothing embedded)
    n_total: int                  # candidate records considered
    n_embedded: int               # newly computed this run
    n_skipped: int                # already had embedding (only_missing)
    n_failed: int                 # crop missing/unreadable
    failed_ids: list = field(default_factory=list)   # record_ids that failed
```

### Public functions (exact signatures)

```python
def embedding_cache_path(dataset: Optional[str], model_name: str = DEFAULT_MODEL_NAME,
                         checkpoint_path: Optional[str] = None,
                         cache_dir: str = DEFAULT_EMB_DIR) -> str:
    """Deterministic cache path: f'{cache_dir}/{dataset or "all"}_{model_label}.pkl'
    where model_label = global_embedding.global_embedding_cache_label(model_name, checkpoint_path).
    Creates no files; just returns the path (parent dir created by writers)."""

def embed_crops(crop_paths: Dict[str, str], cache_path: str, *,
                model_name: str = DEFAULT_MODEL_NAME,
                checkpoint_path: Optional[str] = None,
                device=None) -> Dict[str, np.ndarray]:
    """Low-level: given {key -> crop_image_path}, return {key -> np.ndarray} and persist to cache_path
    as a Dict[str,np.ndarray] pickle. If cache_path exists, load it and only compute MISSING keys,
    then merge + re-save (so re-runs are cheap and additive). Keys with missing/unreadable files are
    omitted from the result (not fatal). Internally calls global_embedding.extract_global_embeddings
    once on the subset that needs computing (load model once, infer many). 'key' is caller-defined;
    embed_records passes record_id. Does NOT touch the store."""

def embed_records(conn, *,
                  dataset: Optional[str] = None,
                  model_name: str = DEFAULT_MODEL_NAME,
                  checkpoint_path: Optional[str] = None,
                  cache_dir: str = DEFAULT_EMB_DIR,
                  only_missing: bool = True,
                  only_species: Optional[str] = None,
                  limit: Optional[int] = None,
                  device=None) -> EmbedResult:
    """Main entry. (1) query_records(conn, dataset=dataset, [species filter], has_embedding=False if
    only_missing) to get candidate records; (2) build {record_id -> crop_path}; (3) call embed_crops
    into embedding_cache_path(...); (4) for each record that now has a vector in the cache, call
    reid_demo.store.update_embedding(conn, record_id, embedding_ref=record_id, embedding_path=cache_path);
    (5) return EmbedResult. only_species matches store 'species' case-insensitively (substring or exact —
    document which; recommend exact-lowercase). Records with no crop on disk go to failed_ids."""

def load_embeddings(cache_path: str) -> Dict[str, np.ndarray]:
    """Load a cache pickle. Thin wrapper over pickle for downstream symmetry. Raises FileNotFoundError
    with a clear message if absent."""

def get_embedding_matrix(conn, *,
                         dataset: Optional[str] = None,
                         normalize: bool = True,
                         only_clustered: bool = False) -> Tuple[np.ndarray, list]:
    """READ side for T05. Query records with embedding_ref NOT NULL (has_embedding=True), load each
    vector from its embedding_path/embedding_ref, stack into an (N, D) float32 matrix in a stable
    record_id order, and return (matrix, record_ids). If normalize=True, L2-normalize each row
    (so cosine == dot product). Groups by embedding_path so each pickle is loaded once. Raises a clear
    RuntimeError naming the record_id + path if embedding_ref is set but the key is missing from the
    pickle (stale cache). 'only_clustered' is a convenience pass-through filter (cluster_id NOT NULL),
    default off; T05 typically calls with defaults BEFORE clustering exists."""
```

### CLI

```
python -m reid_demo.embed --db <path> [--dataset NAME] [--model megadescriptor-l-384]
        [--checkpoint <path>] [--cache-dir data/reid_demo/embeddings]
        [--only-species "leopard"] [--all] [--limit N] [--device cuda|cpu]
```
- Default `--db` = `reid_demo.store.DEFAULT_DB_PATH`.
- `--all` sets `only_missing=False` (recompute even already-embedded records).
- On exit print one summary line, e.g.:
  `T04 embed: dataset=MedvednicaDS dim=1536 total=120 embedded=120 skipped=0 failed=0 cache=data/reid_demo/embeddings/MedvednicaDS_megadescriptor-l-384.pkl`
- Exit 0 on success; non-zero if 0 records were found to embed AND none were already embedded (nothing to do is an error worth surfacing), or on unhandled exception.

### File-format guarantees for downstream

- Cache pickle: `Dict[str, np.ndarray]`, keys are `record_id`, values 1-D `float32` at the **model-native dim (1536 base / 384 linear_l2 checkpoint), NOT pre-normalized**. Same on-disk format as existing `global_embeddings_*.pkl`, so T05/T07 can `pickle.load` directly if they prefer, but SHOULD use `load_embeddings` / `get_embedding_matrix`. If a consumer reads the raw pickle directly it MUST normalize itself and MUST read the dim from the array — never hard-code 384, never assume unit norm.
- `embedding_ref == record_id` is guaranteed for records this ticket processed; `(embedding_path, embedding_ref)` is the join into the pickle.
- Vectors are stored RAW / UN-normalized (exactly as the model emits); normalization is a read-time choice via `get_embedding_matrix(normalize=True)`, which is the path every consumer SHOULD use. Read the dimension from `matrix.shape[1]`, not from a constant.

## Existing code to reuse (real paths)

- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/global_embedding.py`
  - `extract_global_embeddings(image_paths: Dict[str,str], model_name="resnet50", device=None, checkpoint_path=None) -> Dict[str, np.ndarray]` (lines 150-188). **Call this for the actual embedding work**; pass `model_name="megadescriptor-l-384"` and the caller's `device`/`checkpoint_path`. It already: loads the model once, iterates images, opens via PIL `.convert("RGB")`, preprocesses (384x384 + ImageNet norm), runs `torch.inference_mode`, squeezes to 1-D numpy, returns `{img_id: vector}`. You pass `{record_id: crop_path}` as `image_paths`.
  - `global_embedding_cache_label(model_name, checkpoint_path=None) -> str` (lines 27-36). Use for the cache filename label so different checkpoints get distinct caches.
  - `load_or_build_global_embeddings(image_paths, cache_path, *, model_name=..., checkpoint_path=None)` (lines 191-210). NOTE: its caching is all-or-nothing (returns the whole pickle if the file exists, else computes everything). For the demo you want **additive/partial** caching keyed by `record_id`, so implement `embed_crops`'s merge-missing logic yourself (load existing pickle, compute only absent keys via `extract_global_embeddings`, merge, re-dump). You MAY still use `load_or_build_global_embeddings` for the simple "embed exactly this set, no prior cache" case if convenient, but partial/additive is required for `only_missing` to be efficient.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/megadescriptor.py`
  - `load_megadescriptor_l_384(device=None) -> (model, preprocess)` (lines 10-28). You do not call this directly (it is reached via `extract_global_embeddings`), but read it to understand the 1536-dim base output and preprocessing.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/main.py` lines 447-460 — reference usage of `global_embedding_cache_label` + `load_or_build_global_embeddings` in the existing count pipeline; mirror its cache-path convention style.
- `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/reid_demo/store.py` (T01) — import `connect`, `query_records`, `update_embedding`, `DetectionRecord`, `DEFAULT_DB_PATH`, `COLUMNS`. Use `query_records(conn, dataset=..., has_embedding=False)` to find work; `update_embedding(conn, record_id, embedding_ref, embedding_path)` to record results. Do NOT write raw SQL against `detections`.
- Existing caches as format reference: `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/data/cowdataset/global_embeddings_test_megadescriptor-l-384_unsegmented.pkl` (verified `Dict[str, np.ndarray]`, `(1536,)` float32).
- Repo venv: `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/venv/bin/python` (has timm/torch/PIL). `python` is NOT on PATH; use `venv/bin/python`.

## Implementation notes

- **Reuse over reinvention.** The single hard dependency is `global_embedding.extract_global_embeddings`. Your value-add is: store integration, partial/additive `record_id`-keyed caching, species filtering, robust per-record skip, and the read-side matrix helper. Do not re-open the model yourself or re-preprocess images.
- **Load the model once per run.** `extract_global_embeddings` already loads the model once for the whole `image_paths` dict it receives. So batch *all* missing `record_id -> crop_path` pairs into ONE call to `extract_global_embeddings`, not one call per record. (`embed_crops` should compute the set of missing keys first, then make a single `extract_global_embeddings` call on just those.)
- **Missing-file safety.** Before building the dict you pass to `extract_global_embeddings`, drop any `record_id` whose `crop_path` does not `os.path.exists` (or is unreadable) into `failed_ids`. `extract_global_embeddings` will `Image.open(path)` and would otherwise raise on a bad path. If you want extra safety against corrupt JPEGs, you may wrap per-image, but the simplest correct approach is to pre-filter on existence and let `extract_global_embeddings` handle the rest, catching exceptions around the whole call only as a last resort. Document whichever you choose.
- **Cache key = `record_id`**, never the crop filename. This is the join key for the whole demo (T01 contract). `embed_records` therefore passes `{record_id: crop_path}` to `embed_crops`.
- **Partial cache merge:** `embed_crops` should: if `cache_path` exists, `pickle.load` it; compute `to_compute = {k:v for k,v in crop_paths.items() if k not in cached}`; if `to_compute`, run `extract_global_embeddings(to_compute, model_name=..., checkpoint_path=..., device=...)`; merge into the cached dict; `pickle.dump` back. Return the full merged dict (or at least all requested keys present). This makes `only_missing` reruns near-instant.
- **Store update loop:** after `embed_crops` returns, iterate the candidate records; for each whose `record_id` is a key in the merged cache, call `update_embedding(conn, record_id, embedding_ref=record_id, embedding_path=cache_path)`. Records still absent (failed) are not updated and go to `failed_ids`.
- **`only_missing` semantics:** when `True`, candidate set = `query_records(..., has_embedding=False)` (i.e., `embedding_ref IS NULL`). When `False`, candidate set = all matching records and you recompute (still additive merge into cache, but you may want `--all` to force a full recompute by deleting/overwriting cache — recommend: `--all` recomputes the store-update for all records but does NOT delete the pickle; if you need true recompute add a `force` path that drops keys before recomputing — keep this minimal, the demo just needs idempotent + a way to refill missing).
- **`get_embedding_matrix`:** group requested records by `embedding_path`, `load_embeddings` each pickle once, look up by `embedding_ref`. Stack in sorted-`record_id` order for determinism. L2-normalize rows with an epsilon guard (`v / max(||v||, 1e-12)`) — mirror the normalization style in `nested_importance_sampling.py:_l2_normalize_rows` / `_stack_vectors` so T05's downstream cosine math is consistent. Return `float32`.
- **Device:** accept `device` as `None|str|torch.device`; if a string, convert to `torch.device`. Pass through to `extract_global_embeddings` which itself defaults to cuda-if-available. Do not hard-require CUDA — CPU must work (slow is fine for the demo's small piles).
- **Determinism:** embeddings are deterministic in eval mode; no seeding needed. Keep ordering stable (sort by `record_id`) wherever you return lists/matrices.
- **No new heavy deps.** numpy, torch, PIL, timm, tqdm already in the venv. Do not add anything.
- **`reid_demo/__init__.py`:** append re-exports `from .embed import embed_records, embed_crops, get_embedding_matrix, load_embeddings, EmbedResult, embedding_cache_path` (additive; do not disturb T01's exports).
- **`STATUS_BOARD.md`:** append one line, e.g. `- [x] T04 Embedding service: reid_demo/embed.py (embed_records, get_embedding_matrix) + tests/test_embed.py`. Do not edit other tickets' status.

## Acceptance criteria

- [ ] `reid_demo/embed.py` exists; only `reid_demo/__init__.py` (additive) and one line in `STATUS_BOARD.md` are touched beyond new files; no existing pipeline file is modified.
- [ ] All contracted names import: `from reid_demo.embed import embed_records, embed_crops, load_embeddings, get_embedding_matrix, EmbedResult, DEFAULT_EMB_DIR, DEFAULT_MODEL_NAME, embedding_cache_path`.
- [ ] `embed_crops({'r1': '<real crop>.jpg'}, '/tmp/e.pkl')` returns `{'r1': np.ndarray}` with a 1-D `float32` vector of length **1536** (base model) and writes `/tmp/e.pkl`; a second call with the same `cache_path` reuses the cache and recomputes nothing.
- [ ] On a temp DB seeded with >=3 T01 records pointing at real crops, `embed_records(conn, dataset='MedvednicaDS')` writes ONE pickle under `data/reid_demo/embeddings/`, and for each processed record `update_embedding` makes `embedding_ref == record_id` and `embedding_path == <pickle path>`; afterwards `query_records(conn, dataset='MedvednicaDS', has_embedding=True)` returns all of them.
- [ ] Re-running with `only_missing=True` yields `EmbedResult.n_embedded == 0` and `n_skipped == <count>` (no recompute).
- [ ] `embed_records(..., only_species='leopard')` embeds only records whose `species` matches case-insensitively; others keep NULL `embedding_ref`.
- [ ] Every key in the produced pickle is a `record_id` present in the store; `load_embeddings(path)[record_id]` equals the stored vector; `get_embedding_matrix(conn, dataset=...)` returns `(matrix, record_ids)` with `matrix.shape[0] == len(record_ids) == N`, rows L2-normalized when `normalize=True` (per-row norm `1.0 +/- 1e-5`).
- [ ] A record with a non-existent `crop_path` is skipped, counted in `EmbedResult.n_failed`, its id in `failed_ids`, and the run still embeds the others without crashing.
- [ ] `get_embedding_matrix` raises `RuntimeError` (naming the `record_id` + path) when `embedding_ref` is set but the vector is missing from the referenced pickle (stale cache), instead of a bare `KeyError`.
- [ ] `python -m reid_demo.embed --db <path> --dataset MedvednicaDS` exits 0 and prints a summary line with dim/total/embedded/skipped/failed/cache; `--limit N` caps processed records to N.
- [ ] `tests/test_embed.py` passes under `venv/bin/python -m pytest tests/test_embed.py -q`.

## How to verify

```bash
cd /home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project
PY=venv/bin/python   # 'python' is not on PATH

# 0. Confirm the base model produces 1536-dim, un-normalized vectors (sanity, ~10s GPU / longer CPU)
$PY - <<'PY'
import numpy as np
from global_embedding import extract_global_embeddings
import glob
crop = sorted(glob.glob('data/MedvednicaDS/animal_crops/*.jpg'))[0]
e = extract_global_embeddings({'probe': crop}, model_name='megadescriptor-l-384')
v = np.asarray(e['probe']); print('dim', v.shape, 'dtype', v.dtype, 'norm', round(float(np.linalg.norm(v)),3))
assert v.ndim == 1
PY

# 1. Import surface
$PY -c "from reid_demo.embed import embed_records, embed_crops, load_embeddings, get_embedding_matrix, EmbedResult, DEFAULT_EMB_DIR, DEFAULT_MODEL_NAME, embedding_cache_path; print('import OK')"

# 2. End-to-end on a tiny seeded DB built from REAL Medvednica crops
$PY - <<'PY'
import glob, numpy as np
from reid_demo.store import connect, upsert_record, query_records, DetectionRecord, make_record_id
from reid_demo.embed import embed_records, get_embedding_matrix, load_embeddings

conn = connect('/tmp/reid_embed.sqlite')
crops = sorted(glob.glob('data/MedvednicaDS/animal_crops/*.jpg'))[:3]
for i, c in enumerate(crops, start=1):
    stem = 'probe%d' % i
    rid = make_record_id(stem, 1)
    upsert_record(conn, DetectionRecord(
        record_id=rid, source_image='data/MedvednicaDS/animal_images/%s.JPG' % stem,
        source_stem=stem, det_index=1, crop_path=c,
        bbox_x=0.1, bbox_y=0.1, bbox_w=0.2, bbox_h=0.2,
        dataset='MedvednicaDS', species='eurasian lynx'))

res = embed_records(conn, dataset='MedvednicaDS')
print('embedded', res.n_embedded, 'dim', res.embedding_dim, 'cache', res.cache_path)
assert res.n_embedded == 3 and res.n_failed == 0

# store side effects
rows = query_records(conn, dataset='MedvednicaDS', has_embedding=True)
assert len(rows) == 3
assert all(r.embedding_ref == r.record_id for r in rows)
assert all(r.embedding_path == res.cache_path for r in rows)

# pickle keyed by record_id
cache = load_embeddings(res.cache_path)
assert set(cache.keys()) >= {r.record_id for r in rows}

# read-side matrix, L2-normalized
M, ids = get_embedding_matrix(conn, dataset='MedvednicaDS', normalize=True)
assert M.shape[0] == len(ids) == 3
norms = np.linalg.norm(M, axis=1)
assert np.allclose(norms, 1.0, atol=1e-5), norms

# idempotent re-run
res2 = embed_records(conn, dataset='MedvednicaDS')
assert res2.n_embedded == 0 and res2.n_skipped == 3
print('idempotent OK; all assertions passed')
PY

# 3. Missing-crop robustness
$PY - <<'PY'
from reid_demo.store import connect, upsert_record, DetectionRecord, make_record_id
from reid_demo.embed import embed_records
conn = connect('/tmp/reid_embed_fail.sqlite')
rid = make_record_id('ghost', 1)
upsert_record(conn, DetectionRecord(record_id=rid, source_image='x', source_stem='ghost',
    det_index=1, crop_path='data/MedvednicaDS/animal_crops/NOPE_does_not_exist.jpg',
    bbox_x=0.0,bbox_y=0.0,bbox_w=0.1,bbox_h=0.1, dataset='MedvednicaDS'))
res = embed_records(conn, dataset='MedvednicaDS')
assert res.n_failed == 1 and rid in res.failed_ids
print('missing-crop handled:', res.failed_ids)
PY

# 4. CLI
$PY -m reid_demo.embed --db /tmp/reid_embed.sqlite --dataset MedvednicaDS ; echo "exit=$?"

# 5. Tests
$PY -m pytest tests/test_embed.py -q
```

## Open questions

1. **`only_species` matching:** exact-lowercase equality vs substring? SpeciesNet writes human-readable names like `eurasian lynx`, and LeopardID2022 records may carry `leopard`/`panthera pardus`. Recommend **exact lowercase equality** with `only_species` accepting a single value; T10 can call once per target. Confirm with T03/T10 what canonical species string ends up in the `species` column. If finer control is needed, T05/T10 can pre-filter via `query_records` and pass a record set — but the contract above keeps it inside `embed_records` for simplicity.
2. **One cache pickle per dataset vs per (dataset, run/segmentation):** chose per (dataset, model_label). If T10 runs the same dataset under multiple model checkpoints they get distinct files via `global_embedding_cache_label`. Flag if T05/T10 want segmentation tags (`segmented`/`unsegmented`) in the filename like the legacy caches — not needed for the demo (crops are already tight boxes).
3. **`--all` recompute semantics:** does any downstream ticket need a hard "delete and recompute" to fix a poisoned cache, or is additive-merge + missing-fill enough? Current plan: additive only; a stale/wrong vector would require deleting the pickle by hand. If T05 reports drift, add an explicit `force=True` that drops the requested keys before recompute.
4. **Vector storage normalization:** RESOLVED by D2. Vectors are stored RAW / UN-normalized (model-native); normalization happens at read time via `get_embedding_matrix(normalize=True)`. This keeps base-model (1536, ~18.2-norm) and `linear_l2` checkpoint (384, already-unit) outputs consistent for T05. T05/T07 MUST always call `get_embedding_matrix(normalize=True)`; if they read the raw pickle directly they MUST normalize themselves. Not optional — see file-format guarantees above.
5. **Embedding dim drift:** RESOLVED by D2. The dim is model-native (1536 base / 384 `linear_l2` checkpoint) and is NEVER hard-coded. `EmbedResult.embedding_dim` surfaces the actual dim and consumers read it from `matrix.shape[1]`. T05/T07 must not assume "384-dim L2-normalized" anywhere.
