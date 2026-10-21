# Handoff: N1b pipeline integrity — fresh-session prompt

Written 19 October 2026. N1 (collision-safe, review-preserving ingestion) is done and committed (`fc852933`). N1b covers the four remaining data-integrity defects that the [Risnjak assessment](research/project-risnjak-assessment-2026-10-14.md) lists in its engineering table. Any agent picking this up claims it in the "Active work (live)" table of [the decision log](lynx-demo-decision-log.md) first.

## State at handoff

- **Branch:** `feat/reid-demo-pipeline`, head `7ade3225`. Nothing after `606082f2` is pushed.
- **Never commit** the `disk` submodule's local edit.
- **Full test suite:** 723 passed on 19 Oct (user-run, 96 s). That is the baseline.

## The prompt

```text
Continue work in /home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project (branch feat/reid-demo-pipeline). You are starting work item N1b: the remaining data-integrity fixes in the reid_demo field pipeline, needed before a partner's camera archive can be ingested safely.

FIRST:
1. Read AGENTS.md: the coordination rules, "Always first display to me each code change you plan to do", and "the user runs tests". Then read docs/lynx-demo-decision-log.md, starting with "Active work (live)" and the 19 October entries.
2. Claim N1b in the Active work table (owner, state, checklist, files you will touch). Update it at every milestone: proposal, approval, implemented, tests, committed. Document everything in the repo, never only in chat.
3. Read:
   - docs/n1-ingestion-integrity-proposal-2026-10-19.md (what N1 already fixed, and its design principles);
   - reid_demo/DATA_CONTRACT.md;
   - the "Engineering issues" table in docs/research/project-risnjak-assessment-2026-10-14.md;
   - the N1 entry in docs/research/pitch-readiness-analysis-2026-10-19.md §4.1.

ALREADY DONE (N1, commit fc852933):
- Path-scoped record ids, with the scheme recorded per dataset in the meta table (legacy datasets need --record-id-scheme legacy).
- store.ingest_upsert_records: re-ingestion never resets reviews, embeddings or clusters. A crop-geometry change clears derived values and sets extra_json['review_needs_recheck'].
- upsert_record(s) keep replace semantics.

N1b SCOPE (verify each defect in code before proposing):
1. Stage completion is "any row" (reid_demo/run_demo.py, _store_stage_done around line 520). A stage counts as done if ONE row in the dataset has the column non-NULL, so a partial or failed batch is never finished on rerun. Wanted: done means every eligible record has a valid output for the intended configuration; otherwise run on the pending records only.
2. Forced fused re-clustering overwrites reviews. plan_stages adds a second, fused clustering pass with force=True (run_demo.py around line 465). reid_demo/cluster.py then includes reviewed rows (the force branch around line 846), overwriting human merges and splits. Wanted: recompute proposals separately from confirmed identities; never overwrite a reviewed assignment without an explicit, logged correction.
3. Cluster-label namespace collision. Non-forced clustering excludes reviewed rows, but new labels start at 0 in the same dataset namespace, so a new cluster can take the id of an existing reviewed animal. Wanted: persistent identity ids distinct from temporary cluster-proposal labels (or reserved/remapped labels), and explicit attachment of new records to existing identities.
4. Presence-only embedding cache. reid_demo/embed.py embed_records(only_missing=True) (around lines 193–255) skips any row whose embedding_ref is set, without checking the model, checkpoint or preprocessing version. A model change (for example MegaDescriptor → MiewID) silently keeps stale vectors. N1 already clears embedding_ref when the crop changes, so the remaining gap is model and version. Wanted: bind each embedding to a model/version key and recompute when it differs.
Optional, ask first, probably a separate item: MiewID is only wired into the benchmark adapter (reid_demo/benchmark_embeddings.py), not the demo embed path.

HOW:
- Draft docs/n1b-pipeline-integrity-proposal-<date>.md first. For each defect, give: a reproduction (a failing regression test you will add), the minimal fix, any schema or meta change (avoid bumping SCHEMA_VERSION if possible; N1 used meta rows), backward compatibility for existing demo DBs, and the tests to add or edit (disclose every edit to an existing test).
- Show the planned code changes to the user and wait for approval before implementing.
- Prefer additive changes and keep existing public signatures working. tests/test_run_demo.py, tests/test_cluster.py, tests/test_embed.py and tests/test_review.py are the relevant suites.
- After implementing: run ruff F and py_compile yourself. Give the user the focused pytest command and then `venv/bin/python -m pytest tests -q` (baseline 723 passed). Wait for their results unless they authorize you to run them.
- Update reid_demo/DATA_CONTRACT.md, AGENTS.md (Pilot Readiness section), the proposal status and a dated decision-log entry. Commit only when the user asks; never push; leave the disk submodule out.

Out of scope: CzechLynx benchmark code (reid_demo/benchmark_*.py and tools/*czechlynx*), whose hashes bind completed runs; model training; any selection or test evaluation.
```
