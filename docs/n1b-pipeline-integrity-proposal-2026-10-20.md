# N1b proposal: pipeline integrity (stage completion, review-safe re-clustering, label namespace, model-bound embeddings)

Date: 20 October 2026. Work item N1b in the [decision log](lynx-demo-decision-log.md) ("Active work (live)"). Source: the [handoff prompt](n1b-pipeline-integrity-handoff-2026-10-19.md) and rows 3–6 of the "Engineering issues" table in the [Risnjak assessment](research/project-risnjak-assessment-2026-10-14.md). Follows [N1](n1-ingestion-integrity-proposal-2026-10-19.md) (commit `fc852933`).

**Status: approved as written on 20 Oct 2026, with reserved proposal labels (option 3a). Implemented 20 Oct; py_compile passes and Ruff F reports no new findings (the 7 remaining in the changed files are pre-existing at HEAD). Tests passed on 20 Oct, run by the assistant at the user's request: focused suites 170 passed in 24.5 s; full suite 762 passed in 91 s, with the same 960 upstream warnings as the baseline. Committed at the user's request on 20 Oct (see git log; not pushed).**

Implementation notes (differences from the text below):
- **Tests:** 21 new tests instead of 19. Added: `test_store_stage_empty_dataset_not_done` and `test_run_stage_writes_marker_only_on_success` in `tests/test_run_demo.py`. All other names match the table below.
- **Helper:** `_write_stage_marker` is a separate helper in `run_demo.py`. Clustering clears the marker before reading records, so a failed run also leaves the stage not done.

## Summary

| # | Defect | Where | Reproduced 20 Oct | Fix in one line |
| --- | --- | --- | --- | --- |
| 1 | A stage counts as done if **one** row has output | `reid_demo/run_demo.py:520` `_store_stage_done` | Yes: species set on 1 of 9 rows → "done" | Done = every eligible row has output, plus a completion marker for the exact invocation where the store cannot tell (ingest, cluster) |
| 2 | The fused clustering pass is forced and overwrites reviewed rows | `run_demo.py:465` (`force=True`), `reid_demo/cluster.py:847` | Yes: reviewed row's cluster 999 → 0, status still `confirmed` | Fused pass runs unforced; the fused affinity is sliced to the rows being clustered; explicit `--force` is logged and resets the review |
| 3 | New cluster labels reuse the IDs of reviewed animals | `cluster.py:851` + `cluster_by_flank` labels from 0 | Yes: reviewed identity 1; on rerun another animal's rows got label 1 | Proposal labels start above every cluster ID the run does not rewrite |
| 4 | Embeddings are reused by presence, whatever the model | `reid_demo/embed.py:255` | Yes: checkpoint B after model A → 0 embedded, 3 skipped, rows still on A's cache | Reuse only if the vector lives in this call's cache (model + checkpoint label) and the key exists; otherwise recompute |

Reproduction: a throwaway script on temporary SQLite stores with synthetic vectors and a stubbed model call. It is not a test run, and it touched no repository data. Observed output:

```text
D1 species: 1/9 rows set, 8 NULL -> _store_stage_done = True
D3 first-run ids=[0, 1, 2]; reviewed identity id=1; rerun unreviewed ids=[0, 1]; collision with reviewed id: [1]
D2 force=False (reviews protected): raises ValueError: affinity matrix must be a square (8, 8) matrix aligned to the sorted ids; got shape (9, 9)
D2 force=True (as planned by plan_stages): reviewed row cluster 999 -> 0, status still 'confirmed'
D4 model A: embedded 3; then checkpoint B: embedded 0, skipped 3; rows still point at ['DS_megadescriptor-l-384.pkl'] (B's cache would be DS_megadescriptor-l-384_ckpt-other-b0c9878dc0.pkl)
```

The D2 lines explain why `force=True` was there. The fusion sidecar covers every embedded row, so excluding reviewed rows breaks the matrix shape. And with the old any-row check, an unforced fused pass would have been skipped as soon as the baseline pass wrote `cluster_id`. The fix has to address both, not just delete `force=True`.

## Design principles

- **Additive.** Existing public signatures keep working; new behaviour comes through new keyword arguments and dataclass fields with defaults.
- **No schema change.** `SCHEMA_VERSION` stays 1. The only new persistent state is one `meta` row per (stage, dataset), `stage_done:<stage>:<dataset>`, the same pattern N1 used for the record-id scheme.
- **Reviews are never overwritten implicitly.** The only path that re-clusters a reviewed row is an explicit `--force`. It is logged, and it resets the row to `unreviewed`, so a machine label never sits under a human status.
- **Fail loudly rather than guess.** A fused affinity that does not cover the rows to be clustered raises a clear error. It is not padded or partly used.
- **Proposals are temporary; confirmed identities are not.** The `cluster_id` of a reviewed row is a confirmed identity. The `cluster_id` of an unreviewed row is a proposal that a rerun may change. After this change, a proposal never reuses an identity's number.

## Defect 1: stage completion

### Current behaviour

`_store_stage_done` runs `query_records(..., where_sql="<col> IS NOT NULL", limit=1)` and returns true if it finds one row. Consequences:

- A partial or crashed batch is never finished on rerun (species, embed, cluster).
- A changed invocation is never rerun. Example: a `--smoke` run ingests 8 identities. A later full run skips ingest because rows exist, and silently stays at 8 identities.
- Funnel mode: the fused clustering pass would be skipped because the baseline pass already wrote `cluster_id`. This is masked today only by `force=True` (defect 2).
- `fisher`'s `skip_store_column="fisher_ref"` names a column that does not exist (`fisher_ref` lives in `extra_json`). The query raises, the exception is swallowed, and fisher always runs. The behaviour is accidental but harmless, because `fisher` checks its own cache.

### Fix

Per stage:

| Stage | Done when (new) | Before |
| --- | --- | --- |
| ingest | The dataset has rows **and** the last successful ingest recorded this exact invocation (argv, including `--max-identities`) | any row |
| species | The dataset has rows and **no** row has `species` NULL | any row has `species` |
| embed | **Never skipped by the conductor.** `embed` itself processes only pending or stale rows (defect 4) and does not load the model when nothing is pending | any row has `embedding_ref` |
| fisher | unchanged: always runs, self-incremental. `skip_store_column=None` is made explicit | nonexistent column, always runs |
| cluster (baseline and fused) | No unreviewed, embedded row has `cluster_id` NULL **and** the last clustering of this dataset was this exact invocation (for the fused pass this includes the affinity file's size and mtime) | any row has `cluster_id` (baseline); never (fused, forced) |

Both cluster passes share one marker, `stage_done:cluster:<dataset>`, so the last writer wins. In funnel mode the two passes therefore rerun on every conductor run. That is correct, because they overwrite the same column, and it is cheap. Any non-dry `run_clustering` call, including a manual CLI run, clears that marker first, so the conductor never trusts labels it did not write.

`run_demo.py`, planned code:

```python
@dataclass
class StageSpec:
    ...
    skip_store_column: Optional[str] = None  # column every eligible row must have non-NULL
    # additive (N1b):
    skip_store_where: Optional[str] = None   # SQL restricting the eligible rows (default: all rows)
    marker: bool = False                     # also require a completion marker for this invocation
    input_paths: List[str] = field(default_factory=list)  # files whose size/mtime enter the marker key


def _stage_config_key(spec: StageSpec) -> str:
    """Fingerprint of one stage invocation: name, argv or func+kwargs, and input-file identity."""
    inputs = []
    for path in spec.input_paths:
        try:
            st = os.stat(path)
            inputs.append([path, st.st_size, st.st_mtime_ns])
        except OSError:
            inputs.append([path, None, None])
    payload = {"name": spec.name, "cli": spec.cli, "func": spec.func,
               "kwargs": spec.kwargs, "inputs": inputs}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _store_stage_done(spec: StageSpec, db_path: str) -> bool:
    """For store-mutating stages: is every eligible row of this dataset done?

    Done means the dataset has rows; no eligible row (all rows, or those matching
    ``skip_store_where``) still has ``skip_store_column`` NULL; and, for ``marker``
    stages, the last successful run of this stage recorded this exact invocation.
    """
    if not spec.dataset or not (spec.skip_store_column or spec.marker):
        return False
    if not os.path.exists(db_path):
        return False
    try:
        conn = store.connect(db_path, create=False)
    except Exception:
        return False
    try:
        if not store.query_records(conn, dataset=spec.dataset, limit=1):
            return False
        if spec.marker:
            marker = store.get_stage_marker(conn, spec.name, spec.dataset)
            if not marker or marker.get("config") != _stage_config_key(spec):
                return False
        if spec.skip_store_column:
            pending = f"{spec.skip_store_column} IS NULL"
            if spec.skip_store_where:
                pending = f"({spec.skip_store_where}) AND {pending}"
            if store.query_records(conn, dataset=spec.dataset, where_sql=pending, limit=1):
                return False
        return True
    except Exception:
        return False
    finally:
        conn.close()
```

In `_run_stage`, after a stage ends `ok` and `spec.marker` is set, the conductor writes `{"config": _stage_config_key(spec), "label": spec.display_label, "at": <ISO time>}`. A failed marker write is logged to the stage log and leaves the stage `ok`; the next run simply repeats the stage.

Stage builders:

```python
_CLUSTER_ELIGIBLE = "review_status = 'unreviewed' AND embedding_ref IS NOT NULL"

# _spec_ingest:  marker=True; appends --record-id-scheme <x> when cfg.record_id_scheme is set
# _spec_embed:   skip_store_column=None   (embed is incremental and model-aware itself)
# _spec_fisher:  skip_store_column=None   (was "fisher_ref", which is not a column)
# _spec_cluster: when not force -> skip_store_column="cluster_id",
#                skip_store_where=_CLUSTER_ELIGIBLE, marker=True;
#                the fused (func) variant also sets input_paths=[affinity_path]
```

**New conductor option** `--record-id-scheme {path,legacy}` (a `DemoConfig.record_id_scheme` field defaulting to None, passed through to ingest). It is needed because ingest now reruns once on an existing database that has no marker. A dataset ingested before N1 needs `legacy` on that run, exactly as N1 requires for direct ingest. After that, the scheme is read from `meta`.

`store.py`, planned additions (no other store behaviour changes):

```python
def get_stage_marker(conn, stage: str, dataset: str) -> Optional[dict]: ...   # meta "stage_done:<stage>:<dataset>"
def set_stage_marker(conn, stage: str, dataset: str, value: dict) -> None: ... # insert or replace, JSON
def clear_stage_marker(conn, stage: str, dataset: Optional[str] = None) -> None: ...  # None = every dataset
```

### Backward compatibility

- An existing database has no markers. On the first conductor run after this change, ingest reruns once (cheap for B-track whole-frame records; N1 upserts preserve all downstream state), embed runs and finds nothing pending, and cluster reruns once.
- The conductor-level `--force` still reruns every stage. It never re-clusters reviewed rows, because the conductor no longer passes `force` to clustering.
- `test_skip_via_store_column` keeps passing: one row with `species` set is a complete one-row dataset.

## Defect 2: forced fused re-clustering overwrites reviews

### Current behaviour

- `plan_stages` adds the fused pass with `force=True` (`run_demo.py:465`).
- `run_clustering(force=True)` re-clusters every row, including reviewed ones (`cluster.py:847–849`), and writes `cluster_id` with `update_cluster`.
- The row keeps `review_status='confirmed'` (or merged, split) but now carries a machine label. On the next unforced run that machine label is protected as if a human had set it.

### Fix

1. `plan_stages` builds the fused cluster spec with `force=False`.
2. `_cluster_with_affinity` stops discarding the sidecar's `record_ids`. It passes clustering a provider that slices the sidecar to whatever rows are being clustered. `cluster_by_flank` already accepts provider callables and calls them once per flank bucket with that bucket's sorted IDs.

```python
    import numpy as np

    matrix, ids = fusion.load_affinity(affinity_path)
    pos = {rid: i for i, rid in enumerate(ids)}

    def _sidecar_affinity(sorted_ids, _normalized_embeddings):
        missing = [rid for rid in sorted_ids if rid not in pos]
        if missing:
            raise ValueError(
                f"fused affinity {affinity_path} lacks {len(missing)} record(s) to be "
                f"clustered (e.g. {missing[:3]}); re-run the fisher and fusion stages."
            )
        idx = [pos[rid] for rid in sorted_ids]
        return matrix[np.ix_(idx, idx)]

    return cluster.run_clustering(
        db_path, dataset=dataset, affinity=_sidecar_affinity, force=force,
        flank_policy=flank_policy, **tune,
    )
```

When the sidecar covers every row, the result is identical to today's matrix path, because both cut the same sub-matrices per bucket. A regression test pins this.

3. **Explicit corrections are logged.** `run_clustering(force=True)` and `python -m reid_demo.cluster --force` remain the explicit way to re-cluster reviewed rows. For each reviewed row that is actually rewritten, before the write:
   - append `{"at", "reason": "cluster --force", "previous_cluster_id", "previous_review_status", "previous_review_note"}` to `extra_json['review_overrides']` (a list, via `store.update_extra`);
   - reset `review_status` to `unreviewed` via `store.update_review`.

   `ClusterRunSummary` gains `n_review_overridden: int = 0` (an additive field), and the summary sentence mentions it. The `--force` help text says what happens.

### Backward compatibility

`test_rerun_safety_preserves_review` asserts that `force=True` changes the pinned cluster 999. That still holds. No existing test asserts the review status after a forced run.

## Defect 3: cluster-label namespace

### Current behaviour

Unforced clustering excludes reviewed rows, and `cluster_by_flank` labels the remaining rows `0..m-1` in the same `(dataset, cluster_id)` namespace. A new proposal can take the number of a confirmed animal, so the catalogue, evaluation and counts silently merge two animals. Rows outside the run (another species under `--species`, or unembedded rows with stale labels) can collide the same way.

### Fix (recommended: reserved labels)

In `run_clustering`, after `cluster_by_flank` and before any write:

```python
        written = {rid for rid in result.image_ids}
        scope = records if species_filter is None else query_records(conn, dataset=dataset)
        reserved = {
            r.cluster_id for r in scope
            if r.record_id not in written and r.cluster_id is not None and r.cluster_id >= 0
        }
        offset = max(reserved) + 1 if reserved else 0
        if offset:
            labels = result.labels.copy()
            labels[labels >= 0] += offset
            result = dataclasses.replace(result, labels=labels)
        params["label_offset"] = offset
        params["n_reserved_ids"] = len(reserved)
```

Properties:

- **No collisions.** Every proposal label is greater than every ID that the run does not rewrite: reviewed identities, rows outside the species filter, and unembedded rows.
- **Unchanged when nothing is reserved.** With no reviews and no out-of-scope labels, the offset is 0 and labels are exactly as today.
- **Idempotent.** The reserved set does not change between two identical runs, so neither do the labels.
- **Identity numbers stay stable under review merges.** The review tool's merge keeps the smaller of two IDs (`review.py:605`). Proposals are always above identities, so attaching a proposal to an identity keeps the identity's number.
- **Explicit attachment** to an existing identity stays a human action through the existing review paths: a singleton `same` with `target_cluster_id`, or a pair `same` that merges two clusters. Clustering never attaches a new record to a confirmed identity on its own.

**Visible consequence.** After a rerun, unreviewed rows that used to share a cluster with reviewed rows get a new proposal label. They show as a separate proposal until a reviewer attaches them. Today the same rows are relabelled too; whether they land back on the identity's number, on another animal's number, or on a fresh one is a coincidence. The fix makes this deterministic and visible instead of occasionally hidden or silently wrong. The review queue cannot yet ask "does this proposal belong to identity X?" That is the separate "review queue is narrow" item; see Follow-ups.

**Alternative, not recommended for N1b:** anchor-aware proposals. Include confirmed and merged rows as fixed context when clustering. If a proposal cluster contains reviewed rows of exactly one identity, write that identity's number on its unreviewed rows as a proposal. Fewer review items after a rerun, but it changes what the clustering sees, needs rules for split and rejected rows and for clusters with two identities, and comes close to the automatic attachment the handoff rules out.

## Defect 4: model-blind embedding reuse

### Current behaviour

`embed_records(only_missing=True)` skips any row with `embedding_ref` set (`embed.py:255`). After a model or checkpoint change, the rows keep the old vectors and the old cache path. `get_embedding_matrix` catches a dimension change, but a same-dimension change goes through silently (for example base MegaDescriptor → a fine-tuned 1,536-D checkpoint). N1 already clears `embedding_ref` when the crop geometry changes, so the remaining gap is the model.

### Fix

The cache file name already encodes the model: `{dataset}_{global_embedding_cache_label(model, checkpoint)}.pkl`, and the label includes a hash of the checkpoint path. So the binding needs no new state:

```python
def _embedding_is_current(record, cache_path: str, cache_keys: set) -> bool:
    """True iff the record's vector lives in THIS call's cache (same model label,
    checkpoint, dataset prefix and cache dir) and its key is actually present."""
    return (
        record.embedding_ref is not None
        and record.embedding_path is not None
        and os.path.abspath(record.embedding_path) == os.path.abspath(cache_path)
        and record.embedding_ref in cache_keys
    )
```

In `embed_records`:

- Load the intended cache's keys once.
- Skip a row only if `only_missing` and `_embedding_is_current(...)`.
- A row that has an embedding but is not current is **stale**. It is recomputed into the intended cache and counted in a new `EmbedResult.n_stale: int = 0` field (additive; the CLI line prints it).
- If a stale row cannot be recomputed (crop missing or unreadable), its `embedding_ref`/`embedding_path` are cleared rather than left pointing at another model. `extra_json['embedding_cleared']` records the reason and the previous path. A row whose current embedding is fine is never cleared.
- `store.update_embedding`'s type hints are widened to `Optional[str]` so that the clear is a typed call. There is no behaviour change in the store.

### Backward compatibility

- Rows embedded by the conductor already carry exactly this cache path, so they are reused and nothing is recomputed after the upgrade.
- Rows embedded through a different cache directory or dataset prefix are recomputed once. The vectors are identical; only the reference moves.
- `test_embed_records_idempotent` (3 skipped on rerun) keeps passing.

### Limits (documented, not fixed here)

- The label hashes the checkpoint **path**, not its bytes. A checkpoint overwritten in place is not detected. The repository already writes each training run to a fresh directory; DATA_CONTRACT will say so.
- A preprocessing change inside `global_embedding` must change the cache label to invalidate old vectors.
- `get_embedding_matrix` still does not reject two same-dimension models in one matrix. After this fix, that can only arise from a manual `--limit` run part-way through a model change.

## Files (to edit only after approval)

| File | Change |
| --- | --- |
| `reid_demo/run_demo.py` | `StageSpec` fields, `_stage_config_key`, new `_store_stage_done`, marker write in `_run_stage`, stage-builder settings, unforced fused pass, sidecar slicing in `_cluster_with_affinity`, `--record-id-scheme` passthrough |
| `reid_demo/cluster.py` | clear the cluster marker on non-dry runs; reserved-label offset; logged `--force` override; `n_review_overridden`; help text and docstring (D5 note) |
| `reid_demo/embed.py` | `_embedding_is_current`, stale handling, `n_stale`, clear-on-failed-stale |
| `reid_demo/store.py` | stage-marker get/set/clear; `update_embedding` type hints |
| `reid_demo/DATA_CONTRACT.md` | stage completion, label namespace, force logging, embedding currency rule |
| `tests/test_run_demo.py`, `tests/test_cluster.py`, `tests/test_embed.py`, `tests/test_store.py` | new tests below; one disclosed assertion edit |
| `AGENTS.md`, decision log, this proposal | status updates |

Not touched: `reid_demo/review.py`, `ingest.py`, `fusion.py`, `fisher.py`, any `reid_demo/benchmark_*.py` or `tools/*czechlynx*` file, any output directory, the `disk` submodule.

## Tests

**Edit to an existing test (disclosed):** `tests/test_run_demo.py::test_plan_full_funnel_includes_fisher_fusion_and_baseline` asserts `fused_cluster.kwargs["force"] is True`. It becomes `is False`, plus an assertion that the fused spec is marker-gated and lists the affinity file as an input. No other existing assertion changes.

**New tests.** Tests marked *repro* fail on today's code.

| File | Test | Asserts |
| --- | --- | --- |
| test_run_demo | `test_partial_species_stage_is_not_done` *repro* | 1 of 3 rows with species → not done; all 3 → done |
| test_run_demo | `test_cluster_done_ignores_reviewed_and_unembedded` | Only unreviewed embedded rows with NULL `cluster_id` count as pending |
| test_run_demo | `test_ingest_marker_binds_invocation` *repro* | After a successful stubbed run → done; a different `--max-identities` → not done |
| test_run_demo | `test_cluster_marker_last_pass_wins` | Baseline marker → fused pass not done, and vice versa |
| test_run_demo | `test_manual_cluster_run_clears_marker` | A direct `run_clustering` voids the conductor's marker |
| test_run_demo | `test_embed_stage_is_never_store_skipped` | The embed spec has no store skip condition |
| test_run_demo | `test_fused_pass_preserves_reviewed_rows` *repro* | Plan's fused spec, run through `_run_stage`, leaves a reviewed row's `cluster_id` and status untouched and labels the others |
| test_run_demo | `test_sidecar_slicing_matches_matrix_path` | Full-coverage sidecar: labels equal `run_clustering(affinity=matrix)` |
| test_run_demo | `test_sidecar_missing_records_raises_clearly` | Error names the missing records and says to rerun fusion |
| test_run_demo | `test_record_id_scheme_passthrough` | `--record-id-scheme legacy` reaches the ingest argv |
| test_cluster | `test_rerun_never_reuses_reviewed_ids` *repro* | After review, no unreviewed row shares a reviewed row's `cluster_id` |
| test_cluster | `test_rerun_avoids_out_of_scope_labels` | With `species_filter`, other-species labels are not reused |
| test_cluster | `test_label_offset_zero_without_reviews_and_idempotent` | Offset 0 with no reserved IDs; identical labels on rerun with reviews |
| test_cluster | `test_force_logs_and_resets_review` | `review_overrides` holds the previous cluster, status and note; status `unreviewed`; `n_review_overridden == 1` |
| test_cluster | `test_review_merge_keeps_identity_number` | Pair `same` between a proposal member and an identity member keeps the identity's ID (uses `review.apply_decisions`) |
| test_embed | `test_model_change_recomputes` *repro* | Model A then checkpoint B → 3 embedded, `n_stale == 3`, rows point at B's cache |
| test_embed | `test_missing_cache_key_recomputes` | A key deleted from the pickle is recomputed, not skipped |
| test_embed | `test_failed_stale_embedding_is_cleared` | Stale row with a missing crop → reference cleared, `embedding_cleared` logged; current rows untouched |
| test_store | `test_stage_marker_round_trip_and_clear` | get/set/replace/clear, including clear-all-datasets |

## Validation (user-run)

After implementation, I run `py_compile` and Ruff F on the changed files. Then the user runs:

```bash
venv/bin/python -m pytest tests/test_run_demo.py tests/test_cluster.py tests/test_embed.py tests/test_store.py tests/test_review.py -q
```

```bash
venv/bin/python -m pytest tests -q
```

Baseline: 723 passed (19 Oct).

## Follow-ups (not in N1b; separate proposals if wanted)

- **N1c, artifact freshness.** File-output stages (`fusion` sidecar, `catalogue`, `eval`) still skip when their file exists. After new photos, the bundle can show a stale catalogue or headline. The fused pass then fails loudly ("re-run the fisher and fusion stages"), which is today's failure with a clearer message. The fix needs input fingerprints for those stages.
- **Proposal-to-identity review questions.** Let the queue ask whether a proposal cluster or singleton belongs to a confirmed identity. This is part of the Risnjak "review queue is narrow" item, and it is what turns the visible fragmentation from defect 3 into a quick confirmation.
- **MiewID in the demo embed path.** It is wired only into the benchmark adapter. A separate item, needing the user's go-ahead.
- **Stale proposal labels** on unreviewed rows that are no longer clustered (for example after losing an embedding) are reserved, not cleared.

## Risks

- **First run on an existing database:** ingest reruns once (needs `--record-id-scheme legacy` for datasets ingested before N1), embed checks every row, and cluster reruns once. Proposal labels then move above the reviewed IDs.
- **Label churn:** proposal numbers shift whenever the set of reserved IDs grows. A saved review session that uses `target_cluster_id` for a singleton can then point at a different proposal. Pair decisions re-read live IDs and are unaffected. Labels already change between runs today, so this hazard is not new, but it should be documented.
- **Bundle status:** embed now shows `ok` instead of `skipped` on reruns, at the cost of a few seconds of imports. In funnel mode both cluster passes rerun every time.
- **Size:** about 200 lines of source and 350 lines of tests.
