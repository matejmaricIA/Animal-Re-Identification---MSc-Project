# N1b–N1e real-run validation (21 Oct 2026)

Assistant-run at the user's request on `feat/reid-demo-pipeline` at `ad55b1ea`, which contains N1b `99e9a2da` and N1c–N1e. Work item: "N1b–N1e real-run validation" in the [decision log](../lynx-demo-decision-log.md). The pytest suites, which use stubs, had already passed: 787 passed, 1 skipped.

## 1. Real MiewID model test

```bash
REID_DEMO_REAL_MIEWID=1 venv/bin/python -m pytest tests/test_embed_miewid.py::test_real_miewid_dim -q
```

- **Result:** 1 passed in 4.80 s. The pinned MiewID-msv3 was loaded from the local Hugging Face cache, embedded one synthetic crop, and returned a finite 2,152-D vector.
- **Warnings:** 695, of the form "copying from a non-meta parameter in the checkpoint to a meta parameter … is a no-op". Because they could mean the weights were not actually loaded, I checked directly with `load_demo_miewid` on CUDA:
  - all 700 parameters and 510 buffers are on `cuda:0`, none on the `meta` device;
  - there are 51,109,277 parameters, and the sampled weight standard deviations are 0.29–0.51, i.e. not empty tensors;
  - `info.kind = pretrained_miewid_msv3`, dimension 2,152, input size 440.
- **Conclusion:** the weights are real. The warnings come from `transformers` building the model with placeholder tensors during initialisation. The benchmark loader, which `load_miewid` reuses, already refuses missing, unexpected or mismatched keys.
- **Not checked:** numerical equality with the benchmark's cached MiewID vectors for the same image.

## 2. `run_demo` smoke run, twice

### Setup

- **Isolation:** the runner writes some outputs to paths relative to the working directory (`data/reid_demo/catalogue/<ds>`, the embeddings cache, `evaluations/clustering/`), and the repository already holds June LeopardID2022 demo outputs there. The runs therefore used a fresh working directory, `/tmp/n1_smoke/work`, with two read-only symlinks:
  - `data/all_datasets.csv` → the repository file;
  - `data/wildlifedatasets/wildlifereid-10k/versions/7` → `~/.cache/kagglehub/datasets/wildlifedatasets/wildlifereid-10k/versions/7`. The images are not at the repository's `WILD_DATASET_PATH`.
- **Database and bundle:** `/tmp/n1_smoke/store.sqlite` (fresh) and `/tmp/n1_smoke/bundle/`.

```bash
cd /tmp/n1_smoke/work && PYTHONPATH=<repo> <repo>/venv/bin/python -m reid_demo.run_demo --smoke --datasets LeopardID2022 --db /tmp/n1_smoke/store.sqlite --out-dir /tmp/n1_smoke/bundle --run-name smoke1   # then smoke2
```

### Results

`--smoke` caps the run at 8 identities: 47 whole-frame records, global signals only, MegaDescriptor-L-384 on GPU.

| Stage | Run 1 (14 s) | Run 2 (4 s) |
| --- | --- | --- |
| ingest | ok (1.6 s); marker written; scheme `path` recorded | **skipped** (marker matches) |
| species (set_known=leopard) | skipped: ingest had already filled `species` = leopard on all 47 rows | skipped |
| embed | ok (9.6 s): 47 embedded, 1,536-D | ok (3.4 s): **0 embedded, 47 skipped**, 0 stale. It runs every time by design |
| cluster | ok; marker written | **skipped** (marker matches, nothing pending) |
| catalogue | ok; marker written | **skipped** (input digest unchanged); its 60 files still copied into the bundle |
| eval (global) | ok; marker written | **skipped** (input digest unchanged); report copied into the bundle |

- **Bundle status:** `ok` both times; `degraded` is empty.
- **Headline, identical in both runs:** 8 individuals found vs 18 flank-aware ground-truth labels; 68.0851% of photos correctly grouped; ARI 0.3672; 17 candidate-new singletons. This is a smoke check of the plumbing, not an accuracy result.

### Isolation check

- The repository's `data/reid_demo/catalogue/LeopardID2022`, the LeopardID2022 embeddings pickle and the fusion sidecar all keep their 16 June timestamps.
- `evaluations/clustering/` has no new file.
- `git status` shows only the decision-log edit.

## What this validates, and what it does not

**Validated:**
- N1b on a real run: store-stage completion and markers, model-bound embedding reuse, and no re-clustering when nothing is pending.
- N1c freshness of file outputs: catalogue and eval are reused only while their inputs are unchanged.
- N1e: the real MiewID weights load and produce 2,152-D vectors.

**Not exercised:**
- the full-funnel path: fisher, fusion, fused clustering and degradation on failure (`--signals full-funnel`);
- a run after a review or after new photos, where the catalogue and eval should rebuild and proposal labels should sit above confirmed animals;
- the N1d identity questions in the browser;
- an existing pre-N1 database (`--record-id-scheme legacy`);
- MiewID inside `run_demo` (`--embedding-model miewid-msv3`);
- accuracy on field crops.

All outputs are in `/tmp/n1_smoke/` and can be deleted.
