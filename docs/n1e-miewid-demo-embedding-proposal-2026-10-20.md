# N1e proposal: MiewID in the demo embedding stage

Date: 20 October 2026. Work item N1e in the [decision log](lynx-demo-decision-log.md) ("Active work (live)"). Source: the Risnjak engineering row "MiewID is confined to the benchmark adapter" ([assessment](research/project-risnjak-assessment-2026-10-14.md)) and the [N1b handoff](n1b-pipeline-integrity-handoff-2026-10-19.md) optional item.

**Status: approved as written on 20 Oct 2026 and implemented. Focused tests (user-run, 21 Oct) first failed 2 of 9 on a test-stub device bug, fixed in the test file; the rerun gave 8 passed, 1 skipped (real model, by design). The full suite passed on 21 Oct, run by the assistant at the user's request: 787 passed, 1 skipped. Committed at the user's request on 21 Oct (see git log; not pushed).**

Implementation notes:
- **Tests:** 9 new tests, one more than listed (`test_unreadable_images_are_omitted`). The real-model test also needs `REID_DEMO_REAL_MIEWID=1`, so it never downloads or loads weights in a normal run.
- **Warning:** the DBSCAN-eps warning prints in `--dry-run` plans and, on stderr, before real runs.

## Problem

Every lynx result in this repository uses MiewID-msv3: public, or adapted on CzechLynx (F1, `models/czechlynx/miewid_facing_classes_v1`, calibration top-1 71.3%). The field pipeline cannot use it:

- `reid_demo/embed.py` sends every model to `global_embedding.extract_global_embeddings` (`embed.py:168`). That function knows MegaDescriptor and ResNet only (`global_embedding.py:150`).
- MiewID loading exists only in benchmark code: `reid_demo/benchmark_embeddings.load_miewid` (pinned revision `4f1d7f2b…`) and the adapted-checkpoint loader `reid_demo/benchmark_facing_adaptation.load_adapted_model`.
- The `embed` CLI already has `--model` and `--checkpoint`, and the runner (`run_demo.py`) passes neither. So a partner archive would be embedded with MegaDescriptor, which is not the model the lynx evidence is about.

## Proposed behaviour

- `python -m reid_demo.embed --model miewid-msv3` embeds crops with the public MiewID-msv3 at the pinned revision.
- `--model miewid-msv3 --checkpoint models/czechlynx/miewid_facing_classes_v1/best.pt` uses an adapted MiewID checkpoint. Its `manifest.json` must sit beside it; it is validated by the existing loader (selected best epoch, same publisher architecture, strict state keys).
- `python -m reid_demo.run_demo --embedding-model miewid-msv3 [--embedding-checkpoint PATH]` passes both through to the embed stage. The default stays MegaDescriptor, so existing bundles and caches are unchanged.

Preprocessing and output match the benchmark exactly, so demo vectors are comparable with the CzechLynx evidence:

- RGB input, resized to 440 × 440 bilinear, ImageNet normalisation;
- FP16 autocast on CUDA, FP32 on CPU;
- the native 2,152-D output, stored **raw** (D2). Consumers normalise at read time as now and read the dimension from the matrix.

**Cache binding (stronger than N1b's path-hash rule):**
- The cache label is `miewid-msv3-<revision[:12]>-pp1`, where `pp1` is the preprocessing version.
- With a checkpoint, `_ckpt-<sha256(best.pt)[:12]>` is added. The label therefore binds the checkpoint's **bytes**, so a retrained file at the same path gets a new cache.
- Through N1b's `_embedding_is_current`, switching models recomputes every row automatically.

## Planned changes

### 1. New module `reid_demo/embed_miewid.py`

A thin adapter, with heavy imports inside functions. It **imports** the benchmark loaders read-only; the benchmark files are not edited, and their hashes, which bind completed runs, stay the same.

```python
MIEWID_DEMO_MODELS = {"miewid-msv3"}
MIEWID_PREPROCESSING_VERSION = 1

def is_miewid(model_name: str) -> bool: ...
def miewid_cache_label(model_name: str, checkpoint_path: Optional[str]) -> str: ...
def load_demo_miewid(device, checkpoint_path: Optional[str] = None):
    """Public pinned MiewID, or an adapted checkpoint via its manifest (strict)."""
def extract_miewid_embeddings(image_paths: Dict[str, str], *, checkpoint_path=None,
                              device=None, batch_size: int = 16) -> Dict[str, np.ndarray]:
    """{key: raw float32 (2152,)}; unreadable files are omitted (same contract as
    global_embedding.extract_global_embeddings)."""
```

The checkpoint loader is chosen from the manifest: settings with an `arm` key (facing trials such as F1) → `benchmark_facing_adaptation.load_adapted_model`; otherwise (for example v2) → `benchmark_miewid_adaptation.load_adapted_model`. A file without a manifest is refused with a clear message.

### 2. `reid_demo/embed.py`

- `_model_label`: for MiewID names, delegate to `miewid_cache_label`. MegaDescriptor and ResNet labels are unchanged, so no existing cache moves.
- `embed_crops`: for MiewID names, call `extract_miewid_embeddings` instead of `global_embedding.extract_global_embeddings`. Everything else (additive cache, failures, store writes, N1b staleness) is shared.
- The CLI `--model` help lists `miewid-msv3`.

### 3. `reid_demo/run_demo.py`

- `DemoConfig.embedding_model` and `embedding_checkpoint` (default `None`, meaning MegaDescriptor), with the CLI flags `--embedding-model` and `--embedding-checkpoint`.
- `_spec_embed` appends `--model`/`--checkpoint` when they are set. The manifest command line records them.
- If MiewID is chosen with the default DBSCAN backend and no `--eps`, the plan prints one warning: DBSCAN's 0.30 cosine `eps` was tuned for MegaDescriptor; consider `--backend leiden`/`finch` or an explicit `--eps`. No default changes.

### 4. Docs

- `reid_demo/DATA_CONTRACT.md`: the D2 note lists MiewID (2,152-D raw) and the byte-bound checkpoint label.
- `docs/czechlynx-benchmark.md` is **not** edited (the N3 session has uncommitted edits there).

## Tests

**Edits to existing tests:** none expected. MegaDescriptor paths and labels are unchanged.

**New file `tests/test_embed_miewid.py`.** It uses a stub model: no download, no GPU.

| Test | Asserts |
| --- | --- |
| `test_cache_label_binds_revision_and_preprocessing` | The label contains the revision prefix and `pp1`; MegaDescriptor labels are unchanged |
| `test_checkpoint_label_binds_file_bytes` | Same path, different bytes → different label |
| `test_checkpoint_without_manifest_refused` | A clear error naming the expected `manifest.json` |
| `test_embed_records_dispatches_to_miewid` | With a stubbed loader returning 2,152-D vectors, `embed_records(model_name="miewid-msv3")` writes raw vectors and the cache path carries the MiewID label; `get_embedding_matrix` returns D = 2,152 |
| `test_switch_model_recomputes_all_rows` | MegaDescriptor → MiewID marks every row stale (N1b) and recomputes it |
| `test_preprocessing_matches_benchmark` | The stub sees 440 × 440 tensors normalised with ImageNet mean/std |
| `test_run_demo_passes_embedding_model` | `--embedding-model`/`--embedding-checkpoint` reach the embed argv and the manifest command; the default plan is unchanged |
| `test_real_miewid_dim` (skipped by default) | Real pinned model on one crop → 2,152-D; runs only if the weights are already in the local Hugging Face cache |

## What this does not establish

- **Accuracy on field crops.** Every lynx number in this repository used annotation-assisted **masked** crops. The demo embeds **MegaDetector bounding-box crops** with background. F1 was adapted on masked crops. Its behaviour on unmasked raw-trap crops, and on Croatian cameras, is unmeasured (pitch roadmap item X2, raw-frame pipeline).
- **Count reliability.** Swapping the embedding does not make clustering or counts reliable. Every grouping control failed on calibration (decision log, 14–17 Oct).
- **Licence.** MiewID-msv3's licence is disputed on Hugging Face ([pitch analysis](research/pitch-readiness-analysis-2026-10-19.md), "Licence"). Research use in the demo is what this proposal enables. Commercial use or delivery to a partner needs clarification with Conservation X Labs first.
- **Remote code.** Loading uses `trust_remote_code=True` at the pinned revision, as the benchmark already does. The first use downloads the weights over the network.

## Validation

After implementation I run `py_compile` and Ruff F. Tests are user-run unless you authorize me:

```bash
venv/bin/python -m pytest tests/test_embed_miewid.py tests/test_embed.py tests/test_run_demo.py -q
```

```bash
venv/bin/python -m pytest tests -q
```

A real-model check (GPU, downloads weights on first use) only if you want it:

```bash
venv/bin/python -m reid_demo.embed --db <db> --dataset <ds> --model miewid-msv3 --limit 8
```

## Order with N1c

N1c and N1e both edit `reid_demo/run_demo.py`, so they should be implemented one after the other: N1c first, because it changes the stage-skip logic that N1e's new flags pass through. N1d touches only `review.py` and can go in parallel with either.

## Risks

- **Re-embedding cost:** switching a dataset to MiewID recomputes every row once, at about 440 px per crop on GPU.
- **Coupling:** the demo now imports two benchmark modules. They are frozen by convention (their hashes bind runs), so the import is stable. If they ever change, the demo's MiewID path changes with them; the adapter's own tests would catch a signature change.
- **Size:** about 120 lines of new adapter, 25 lines across `embed.py`/`run_demo.py`, and 170 lines of tests.
