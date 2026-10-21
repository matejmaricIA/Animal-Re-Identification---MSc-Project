# Handoff: N3 timed reviewer trial — fresh-session prompt

Written 19 October 2026. N3 is a Now item in the [pitch-readiness analysis](research/pitch-readiness-analysis-2026-10-19.md#41-now--next--later-with-gates). It is the single most important missing measurement: every expert-time figure so far uses invented per-click costs. Any agent picking this up claims it in the "Active work (live)" table of [the decision log](lynx-demo-decision-log.md) first.

## State at handoff

- **Branch:** `feat/reid-demo-pipeline`, head `7ade3225`. Nothing after `606082f2` is pushed.
- **Never commit** the `disk` submodule's local edit.
- **The review app** lives outside this repository at `/home/mm-workstation/re-id/czechlynx-review-app`. Its existing input-audit database and exports must stay intact.

## The prompt

```text
Continue the CzechLynx lynx re-identification work in /home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project (branch feat/reid-demo-pipeline). You are starting work item N3: a timed trial with real reviewers that measures how long ranked-matching decisions actually take, and how often reviewers get them wrong.

FIRST:
1. Read AGENTS.md: the coordination rules, "Always first display to me each code change you plan to do", and "the user runs tests". Then read docs/lynx-demo-decision-log.md, starting with "Active work (live)" and the 18–19 October entries.
2. Claim N3 in the Active work table (owner, state, checklist, files and outputs you will touch). Update it at every milestone. Document everything in the repo, never only in chat.
3. Read:
   - docs/research/pitch-readiness-analysis-2026-10-19.md: the executive summary, §2.6 (criteria P2 and P3), §3F and §4.1;
   - docs/research/czechlynx-n2-pruning-safeguards-results-2026-10-19.md;
   - docs/research/czechlynx-frozen-selection-results-2026-10-19.md;
   - docs/research/czechlynx-review-effort-results-2026-10-18.md;
   - /home/mm-workstation/re-id/czechlynx-review-app/README.md and its server.py and static/ files.

WHY:
- Every "expert-hours" number so far comes from simulated reviewers with placeholder costs per decision:
  - placeholder: 8 s comparison, 10 s pair question, 5 s glance;
  - "fast-click": 2–6 s.
- On calibration, the N2 candidate (P(match) ordering, ε 0.001, W4 auto-accept τ 0.99, duplicate-merge second check) needs 2,350 decisions vs 5,325 for full manual ranked matching (W1). That is −56% in decisions; the saving in TIME is unmeasured.
- Pitch criterion P2 needs ≥ 40% less measured wall-clock time than manual matching.
- P3 needs MEASURED reviewer error. The simulations assumed 0/3/5%; Croatian experts measured 92.7% manual identification accuracy (Blašković et al. 2025), with a different denominator.

DATA AND CONSTRAINTS:
- CzechLynx calibration only: 6,352 photos, 55 lynx, 1,791 camera-day bundles. Selection is spent and the official test is locked; do not touch either.
- F1 vectors already exist: calibration dir evaluations/czechlynx/facing_adaptation_calibration_classes_v1/clustering; cache data/czechlynx/cache/miewid_facing_classes_v1/. Prepared masked crops are under data/czechlynx/prepared_masked/masked/. No new model inference should be needed.
- evaluations/czechlynx/pruning_safeguards_f1_v1/ and the committed benchmark modules are final and hash-bound: read and reuse them, never edit them. Put new logic in new files.
- Reviewers must never see identity labels or identity-bearing file names.

YOUR TASK: draft a PROPOSAL first (docs/czechlynx-n3-timed-review-proposal-<date>.md). Do not implement until the user approves. Decide and justify:
1. Where the UI lives. Either a matching mode in the sibling app, or a separate copy of the app with its own empty data/ directory; the README already recommends one copy per audit. Do not disturb the existing input-audit database or exports.
2. Task design. Suggested starting point:
   - Precompute a fixed task list from the existing simulation code (reid_demo/benchmark_review_simulation.py, benchmark_nonmatch_pruning.py, benchmark_pruning_safeguards.py).
   - Each task is a query bundle plus candidate catalogue animals (exemplar photos) in the order the workflow would show them. The reviewer answers same / different / unsure, or picks a match / "new animal".
   - Log a timestamp per click, the answer and the time on screen.
   - Map decisions onto the ledger kinds (rank1_comparison, further_comparison, second_comparison, pair_question, merge_confirmation, audit_glance, extra_animal), so measured seconds can replace the placeholder costs.
   - State the limitation: a precomputed catalogue state means reviewer mistakes do not propagate. Measured error rates are then fed back into the existing sequential simulation.
   - Argue for or against a live sequential catalogue instead.
3. What reviewers see: masked crops vs original frames (experts normally use originals), zoom, and how many exemplars per candidate animal.
4. Protocol: at least 2 reviewers (the user plus one more if available), about 300 bundles each, W1 vs the assisted workflow counterbalanced or randomized, an excluded practice block, breaks. Record reviewer experience. Note that the user's own time is the scarce resource; propose a realistic schedule.
5. Pre-registered decision rule, fixed before any session:
   - SUCCESS if measured per-kind times, applied to the N2 decision counts (simulations.csv carries counts per kind), give ≥ 40% wall-clock saving vs W1, AND reviewer error rates per decision kind are recorded with confidence intervals.
   - Below 25% saving: reframe the pitch to error reduction only.
   - Also report the "unsure" rate (unidentifiable photos) and whether second-check disagreements are themselves error-prone.
6. Outputs:
   - a timestamped export (CSV/JSON);
   - a small re-costing script (a new file) that applies the measured costs and errors to the saved N2 and W1 simulation rows;
   - docs/research/czechlynx-n3-timed-review-results-<date>.md;
   - decision-log, proposal-status and AGENTS.md updates.
7. Scope estimate (files, lines, tests) and the app-test command for the user: python3 -m unittest discover -s <app>/tests -v

After approval: implement, run static checks yourself, and give the user the test command. Wait for their results unless they authorize you. Then hand them the session instructions. Reviewer sessions are theirs to run. Commit only when asked; never push; leave the disk submodule out.
```
