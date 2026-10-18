# Handoff: (a) non-match pruning + safeguards — fresh-session prompt

Written 18 October 2026. The user decided the plan: **(a) now, then (b) facing-aware re-adaptation of the embedding**. This file holds the prompt that starts (a) in a fresh session. Any agent picking up (a) should claim it in the "Active work (live)" table of [the decision log](lynx-demo-decision-log.md) first.

## State at handoff

- **Branch** `feat/reid-demo-pipeline`. Commits `9065277d`, `3f5a0f11`, `64b70e6e` and `0a25c2c8` are **not pushed** (`606082f2` and earlier are on the remote).
- **Uncommitted in the working tree:** the minimum-review round (`reid_demo/benchmark_gated_matching.py`, `tools/simulate_czechlynx_gated_matching.py`, `tests/test_benchmark_gated_matching.py`, `docs/research/czechlynx-gated-matching-results-2026-10-18.md`, and updates to the proposal and decision log), plus this handoff. A fresh session in the same checkout sees all of it. Commit only when the user asks.
- **Do not include** the `disk` submodule's local `detect.py` edit in any commit.
- **Outputs** under `evaluations/czechlynx/` are gitignored. They stay on disk and are referenced from the docs.

## The prompt

Copy everything inside the block into the new session:

```text
Continue the CzechLynx lynx re-identification work in /home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project (branch feat/reid-demo-pipeline). You are starting work item (a): calibrated non-match pruning plus safeguards, to cut expert review cost.

FIRST, before anything else:
1. Read AGENTS.md (note the "Multi-agent coordination" section), then docs/lynx-demo-decision-log.md, starting with "Active work (live)", "Requirement update — 18 October 2026", the current snapshot and the 18 October dated entries.
2. Claim item (a) in the Active work table (owner = this session, state, progress checklist, files/outputs you will touch). Keep that row current at every milestone: proposal drafted, approved, implemented, tests passed, run started/finished, results reviewed, committed. Several agents may work in this repo, and everything must be documented in the repo as it happens, never only in chat.
3. Read these results, newest first:
   - docs/research/czechlynx-gated-matching-results-2026-10-18.md (the direct predecessor)
   - docs/research/czechlynx-review-effort-results-2026-10-18.md
   - docs/research/czechlynx-flank-grouping-results-2026-10-18.md
   - docs/research/czechlynx-edge-grouping-results-2026-10-18.md
   - docs/research/czechlynx-coverage-diagnostics-results-2026-10-18.md

WHERE WE ARE (calibration pool: 6,352 photos, 55 lynx, 1,791 camera-day bundles, frozen FP32 v2 MiewID vectors):
- Automatic grouping peaks at pairwise F1 ≈ 0.60. The user therefore accepted expert-confirmed, embedding-ranked catalogue matching as the route to reliable counts, but ONLY if the bulk is automated. The product is an all-in-one system (camera-trap ingestion → empty/species classification → re-ID/counting) that must reliably save researchers a lot of time with minimal expert review.
- Ranked catalogue matching (W1), with a simulated expert confirming every bundle, gives a verified 55/55 count with 5,283 decisions (≈1.85 h per 1,000 photos at placeholder costs: 5 s glance, 8 s catalogue comparison, 10 s pair question). The correct animal ranks first for 83% of known-animal bundles. The 55 new-animal confirmations alone cost 1,485 comparisons, because each needs a full-catalogue scan.
- Confidence-gated automation (W4, reid_demo/benchmark_gated_matching.py): an identity-cross-fitted confidence model with appearance + space-time features (seen at same camera/cell, km to nearest previous sighting, days since seen) reaches AUC 0.979. About 50% of bundles are auto-decided at ≈0.4% misassigned photos. In a post-hoc variant where the expert scans the full catalogue before declaring a new animal, the count is exact, but expert time only falls to ≈1.5 h per 1,000 photos (−14% to −20%).
- The remaining expert cost is new-animal confirmation (full scans) and hard matches, not routine matches.
- Other findings: a top-K-capped expert path creates duplicate animals (count +28 to +70); 3% reviewer error gives 8–11% misassigned photos; appearance-only automation can collapse everything into one animal at cold start; audit 95% intervals covered the truth in all perfect-reviewer runs but only 73% of all automated runs.

YOUR TASK (a): draft a concrete PROPOSAL first. Do not implement until the user approves (AGENTS.md: present code changes for approval first). Save it as docs/czechlynx-nonmatch-pruning-proposal-<date>.md, linked from the decision log. It should cover:
- A calibrated "not this animal" model. For a bundle against each catalogue animal, estimate P(match) from label-free features (appearance scores, rank and margin, space-time history, facing agreement, identifiability risk). Train it identity-cross-fitted, as the confidence model was.
- A pruned expert path. The expert checks candidates in ranked order, but the system stops the scan when every remaining candidate's P(match) is below ε, so declaring a new animal means checking a short list instead of the whole catalogue. Measure the extra duplicates this causes against the time saved, over an ε grid. Keep the full-scan behaviour as the reference.
- Combination with the existing W4 auto-accept (τ grid) and the space-time features.
- Safeguards:
  - A cold-start guard: no automation until the catalogue has ≥ N animals and/or enough bundles.
  - Double confirmation of new-animal decisions.
  - Periodic duplicate checks between catalogue animals.
  - Audits by a second, independent simulated reviewer.
  - Report reviewer-error sensitivity (0%, 3%).
- Cost-model sensitivity: one-click rank-1 confirmations may take far less than 8 s, so report raw decision counts and at least two cost settings.
- Declared targets, before running, carried over from the minimum-review proposal unless the user changes them:
  - ≥ 70% of bundles automatic;
  - ≤ 1% misassigned photos;
  - |count error| ≤ 1;
  - ≤ 0.5 expert-h per 1,000 photos at placeholder costs;
  - the audit 95% interval covers the true error.
  Also report the saving vs W1 (5,283 decisions) and vs the full-scan W4 variant.
- Code plan: new files only (suggested reid_demo/benchmark_nonmatch_pruning.py, tools/simulate_czechlynx_nonmatch_pruning.py, tests/test_benchmark_nonmatch_pruning.py). Reuse Catalogue, Effort, Reviewer, bundle_keys, fast_quality, plain (benchmark_review_simulation), and Context, match_features, replay_records, fit_fold_models, audit, wilson_interval (benchmark_gated_matching). Keep the usual contracts: calibration only; refuse selection; empty output directory; input/cache SHA-256 before and after; no neural imports at tool import; synthetic tests including identity-free features, no own-animal training, error propagation, no automatic new animals, and pruning-induced duplicates.

CONSTRAINTS:
- Calibration pool only. The selection pool and official test assignments stay reserved.
- No count configuration export, no park outreach, no edits to existing source or earlier outputs without approval.
- Preserve failed runs.
- Label any analysis outside an approved grid as post-hoc.
- The user usually authorizes CPU runs by the assistant after approving a proposal; GPU work needs explicit authorization.
- Commit only when asked, and never include the disk submodule edit. Unpushed commits exist; do not push unless asked.

After (a) is finished and reviewed, the agreed next item is (b): facing-aware re-adaptation of the MiewID embedding with (animal, facing) classes on the fitting pool. It needs its own proposal, approval and GPU authorization. Do not start (b) within (a).
```

## Why (a) before (b)

(a) attacks the measured time sink directly: new-animal confirmations and hard matches. It is CPU-only, reuses the existing simulation framework, and its safeguards (cold start, reviewer error, audits) are needed in any product whatever the embedding. (b) could improve both automation and ranking, but it needs GPU training, and its benefit is easier to quantify once (a) defines the review workflow it plugs into.
