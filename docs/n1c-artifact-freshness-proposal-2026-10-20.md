# N1c proposal: fresh fusion, catalogue and evaluation outputs in the demo runner

Date: 20 October 2026. Work item N1c in the [decision log](lynx-demo-decision-log.md) ("Active work (live)"). Follows [N1b](n1b-pipeline-integrity-proposal-2026-10-20.md) (commit `99e9a2da`), which fixed completion checks for the store-writing stages and listed this as its first follow-up.

**Status: approved as written on 20 Oct 2026 and implemented. Focused tests passed (user-run, 21 Oct). The full suite passed on 21 Oct, run by the assistant at the user's request: 787 passed, 1 skipped. Committed at the user's request on 21 Oct (see git log; not pushed).**

Implementation notes:
- **Fisher dependency:** `fusion` also `depends_on` `fisher`, so a failed fisher degrades the run to global-only, like a failed fusion. Test `test_failed_fisher_also_skips_fusion` covers it.
- **Tests:** 8 new tests (the 7 listed plus that one). No existing test edited.
- **Skip decision:** in the new `_stage_done`, a file stage needs its output plus a fresh marker when it is marker-gated. A store stage keeps N1b's rule.

## Problem

N1b made the runner (`reid_demo/run_demo.py`) skip a store stage only when every eligible row is done. The three stages that write **files** still skip as soon as their file exists (`_outputs_exist`, `run_demo.py:559`):

| Stage | Output | Reads | What goes wrong today |
| --- | --- | --- | --- |
| fusion (T12) | `data/reid_demo/fusion/<ds>_<signals>.npz` | every embedded row's vector and Fisher vector (`fusion._aligned_signal_matrices`) | After new photos or a re-embedding, the old sidecar is kept. Missing rows make the fused cluster pass fail ("re-run the fisher and fusion stages", N1b). A re-embedding with the same rows is used **silently**, with affinities from the old model |
| catalogue (T06) | `data/reid_demo/catalogue/<ds>/index.html` + pages | every row's `cluster_id`, confidence, review status, orientation, crop | After re-clustering or a review, the bundle ships the old catalogue |
| eval (T07) | `evaluations/clustering/<ds>_<tag>.json` | every row's `cluster_id`, `gt_identity`, orientation | The bundle headline (individuals found, percentage correctly grouped, lift) can describe an earlier clustering |

Two further defects found while checking the code:

1. **A failed fusion leaves the fused pass on a stale sidecar.** `fusion` is non-critical ("degrades to global clustering on failure", `run_demo.py:341`), but the fused cluster pass and the fused evaluation still run afterwards. With no sidecar, the fused pass fails, and it is critical, so the run aborts instead of degrading. With an old sidecar that happens to cover the rows, it clusters on stale affinities without any warning, and the fused evaluation then reports those labels as the full-funnel result.
2. **The catalogue never deletes old pages.** `build_catalogue` writes `individuals/individual_<cid>.html` and `thumbs/<record>.jpg` into the same directory every time, and never removes pages for clusters that no longer exist (`catalogue.py:853`). The runner copies the whole directory into the bundle (`_copy_artifact`, mode `dir`), so stale, unlinked individual pages travel with it.

## Design principles

- **Same mechanism as N1b.** A file stage is done only when its output exists **and** a `meta` marker records the same invocation and the same input rows. No schema change.
- **Fingerprint what the stage reads, nothing else.** A review changes the catalogue and the evaluation, but not the fusion sidecar.
- **Degrade honestly.** If an upstream stage fails, everything that depends on its output is skipped, its old artifacts are left out of the bundle, and the manifest says why.
- **Additive.** Existing signatures keep working. The new `StageSpec` fields have defaults.

## Planned changes

### 1. `reid_demo/run_demo.py`: input fingerprints for file stages

New `StageSpec` field:

```python
    input_columns: List[str] = field(default_factory=list)  # store columns this stage reads (digest)
    input_where: Optional[str] = None                        # SQL restricting the rows read
```

Marker names: store stages keep `stage_done:<name>:<dataset>`. A file stage uses `stage_done:<name>@<first output path>:<dataset>`, so the baseline and fused evaluations (two files) have separate markers.

```python
def _input_digest(spec: StageSpec, conn) -> Optional[str]:
    """sha1 over (record_id, *input_columns) of the rows this stage reads, in record_id
    order; extra_json keys are addressed as 'extra.<key>'. None if the stage declares
    no input columns."""
```

`_stage_config_key(spec)` gains the digest when `input_columns` is set. `_run_stage` skips a stage with outputs only if:

```python
_outputs_exist(spec) and (not spec.input_columns or _file_stage_marker_matches(spec, db_path))
```

The marker is written after a successful run, exactly as for store stages. A stage without `input_columns` (the A-track report) keeps today's behaviour.

Declared inputs:

| Stage | `input_where` | `input_columns` |
| --- | --- | --- |
| fusion | `embedding_ref IS NOT NULL` | `embedding_ref`, `embedding_path`, `orientation`, `extra.fisher_ref`, `extra.fisher_path` |
| catalogue | — (all rows) | `cluster_id`, `cluster_conf`, `is_candidate_new`, `review_status`, `orientation`, `crop_path`, `species` |
| eval | — (all rows) | `cluster_id`, `is_candidate_new`, `gt_identity`, `orientation`, `review_status` |

`embedding_path` binds the model (after N1b, a vector is only ever reused from its own model's cache). `fisher_path` binds the Fisher method and PCA label (`fisher.fisher_cache_label`). Vector values are not hashed: within one cache file, an existing key is never overwritten.

Funnel mode behaves correctly without special cases. Both cluster passes rerun every time (N1b). Their labels are deterministic, so each evaluation's digest matches its previous run and it is skipped. A review changes `cluster_id`/`review_status`, so the catalogue and both evaluations rerun.

### 2. `reid_demo/run_demo.py`: dependency-aware degradation

New `StageSpec` field: `depends_on: List[str] = field(default_factory=list)`, holding labels of earlier specs for the same dataset.

- The fused cluster spec depends on `fusion(<signals>)`; the fused evaluation depends on `cluster(<signals>)`.
- In `run_demo`: if a dependency ended `failed`, or was itself skipped for this reason, the dependent spec is not run. It gets `status="skipped"`, `output_paths=[]` and `error="skipped: upstream stage <label> failed"` (the same shape as today's after-abort skips). With no output paths, its old artifact is neither copied into the bundle nor read by the headline.
- The catalogue does not depend on the fused pass, so it renders whatever labels the store holds: the baseline labels, if fusion failed in this run.
- `_build_manifest` adds `"degraded": ["cluster(full-funnel)", ...]`, and the bundle summary adds one plain sentence ("Fusion failed; results are global-only"). `_overall_status` is unchanged: a non-critical failure already gives `partial`.

### 3. `reid_demo/catalogue.py`: remove stale pages and thumbnails

At the end of `build_catalogue`, delete only files this module owns and did not write in this run:

- `individuals/individual_<cid>.html` for clusters not in this run;
- `thumbs/<token>.jpg` for records not in this run (the placeholder is kept);
- `montages/*.png` for clusters not in this run, when montages are enabled.

Nothing else in the output directory is touched. `CatalogueResult` gains `n_stale_removed: int = 0`.

### 4. Tests

**Edits to existing tests:** none expected. `test_skip_if_outputs_exist` and `test_force_reruns_even_if_outputs_exist` use specs without `input_columns`, so the old behaviour holds for them.

**New tests:**

| File | Test | Asserts |
| --- | --- | --- |
| test_run_demo | `test_eval_reruns_after_relabel` *repro* | Eval output exists and marker matches → skipped; a `cluster_id` change → reruns |
| test_run_demo | `test_catalogue_reruns_after_review` *repro* | A `review_status`/`cluster_id` change → reruns; an unrelated column (e.g. `detector_conf`) → skipped |
| test_run_demo | `test_fusion_reruns_on_new_or_reembedded_rows` *repro* | New embedded row → rerun; `embedding_path` change with the same rows → rerun; a review → skipped |
| test_run_demo | `test_baseline_and_fused_eval_markers_are_separate` | Two eval outputs keep two markers |
| test_run_demo | `test_failed_fusion_skips_fused_pass_and_eval` *repro* | Stubbed fusion failure → fused cluster and eval `skipped` with the upstream error, empty `output_paths`, not copied; bundle `partial`; manifest lists them as degraded |
| test_run_demo | `test_file_stage_without_inputs_keeps_presence_skip` | The A-track report still skips on presence |
| test_catalogue | `test_rebuild_removes_stale_individual_pages` *repro* | After relabelling, pages and thumbnails of vanished clusters and records are gone; unrelated files stay |

## Not in scope

- **A-track report:** it reads `data/<ds>/` files, not the store. Fingerprinting its input folder is possible but not needed for the lynx pilot.
- **Reviewed rows in the evaluation:** evaluation scores every row, including human-corrected ones, so after reviews the headline mixes machine and human work. Reporting how many evaluated photos were reviewed would be a one-field addition to `eval.py`; a separate decision.
- **Calibrator files** for fusion: the runner passes none today.

## Validation

After implementation I run `py_compile` and Ruff F. Tests are user-run unless you authorize me:

```bash
venv/bin/python -m pytest tests/test_run_demo.py tests/test_catalogue.py tests/test_fusion.py tests/test_eval.py -q
```

```bash
venv/bin/python -m pytest tests -q
```

## Risks

- **One extra rerun after upgrade:** the first run after this change rebuilds fusion, the catalogue and both evaluations once, because no markers exist yet.
- **Digest cost:** one query and one hash over the dataset's rows per file stage. Negligible next to the stages themselves.
- **Funnel bundles that degrade:** they no longer show a lift. That is the honest outcome.
- **Size:** about 120 lines of source (run_demo about 100, catalogue about 20) and 200 lines of tests.
