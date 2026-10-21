# Proposal: freeze one review operating point and evaluate it once on the reserved selection pool

Date: 19 October 2026. Work item (c) in the [decision log](lynx-demo-decision-log.md) "Active work" table. Status: **approved by the user on 19 October 2026 as written, pre-registered (`97571909`) and run once on selection. S1, S2 and S4 not met; S3 met.** See [results](research/czechlynx-frozen-selection-results-2026-10-19.md).

## Why

Every number since 14 October is **calibration development evidence**: the same 6,352 photos of 55 lynx were used to design, compare and tune W1, W4, the non-match pruning (a) and the facing-aware embedding (b). Development numbers are optimistic by construction.

The reserved **selection pool** (4,725 photos, 55 *different* lynx) exists to answer one question honestly: does a frozen operating point hold up on unseen animals? It was used once before, on 16 September, for public-MiewID recognition only (63.79% top-1). Nothing in the review or counting work has touched it.

This needs a change to the standing rule. AGENTS.md and the decision log say *"do not export count config or run selection"*. **Approving this proposal lifts that rule for exactly one frozen configuration and one selection run.**

## The frozen operating point

Everything below is fixed in a configuration file, and its fingerprint is committed to the repository **before** any selection photo is read.

| Component | Frozen choice | Calibration evidence |
| --- | --- | --- |
| Embedding | F1 facing-aware MiewID: `models/czechlynx/miewid_facing_classes_v1/best.pt`, epoch 7, state `533fa325…` | Protocol top-1 71.3% (v2 68.6%) |
| Bundling | Same camera + same day, processed in date order | — |
| W4 auto-accept | **τ = 0.99 (primary)**; τ = 0.98 is evaluated as a pre-declared secondary and is not used to choose anything | F1, τ 0.99: 48% automatic, 0.24% misassigned, count +1, 2,116 decisions. τ 0.98: 53.5%, 0.33%, +1, 2,015; with 3% reviewer error the count is −2 vs +1 at τ 0.99 |
| Candidate order + pruning | P(match) order, ε = 0.001, at least one candidate always shown | (a): new-animal checks −58%, no extra duplicates on v2 |
| Safeguards | Cold-start guard (N = 10 animals, B = 100 expert bundles); double confirmation of new animals (c = 3); disagreement check (δ = 0.05); duplicate checks every 100 bundles (r = 3) plus one at the end; automation audit (n = 100); count audit (n = 10) | (a) ablations |
| Reviewer models | Expert and second reviewer at 0% (primary) and 3% error; separate seeded generators | — |
| Costs | Placeholder and fast-click settings, with raw decision counts | — |

**Models are trained on all of calibration and then frozen.** Selection animals are disjoint from calibration animals, so no cross-fitting is needed at test time. Each is serialized as plain JSON (scaler means and scales, coefficients, intercept):

- the **W4 confidence model**, trained on the replay records of the calibration catalogue;
- the **P(match) model**, trained on the per-candidate replay records (both trained with the same label-free features as (a) and (b));
- the **facing classifier**, trained on the 1,115 pose-labelled calibration photos using F1 vectors.

**Identifiability risk is frozen too.** Today's risk score ranks each photo against the other photos of the same collection. For a true deployment simulation, selection photos are instead ranked against the **frozen calibration distributions** of the four risk signals.

## The one-shot selection run

1. **Verify everything against the freeze record:** configuration fingerprint, F1 checkpoint and state hashes, code hashes, metadata, protocol plan and raw-metadata hashes. Any mismatch stops the run before a selection photo is read.
2. **Extract F1 vectors** for the 4,725 selection photos into a new cache, `data/czechlynx/cache/miewid_facing_classes_v1_selection/` (GPU, ≈ 2–3 min). Extraction never sees identities.
3. **Build the label-free context:** camera, cell, day, coordinates from raw metadata (90% coverage), frozen-classifier facing and frozen-reference risk.
4. **Simulate:**
   - the frozen operating point (τ 0.99 primary, τ 0.98 secondary) at 0% and 3% reviewer error, with both audits;
   - **W1** (the expert confirms every bundle in cosine order, full scan) on the same F1 vectors, as the saving reference.
5. **Protocol recognition** on the selection gallery/query split (top-1, top-5, mAP@50, macro top-1), for comparison with the 16 September public-MiewID selection result.
6. **Breakdowns,** pre-declared:
   - per animal;
   - with and without **lynx_287**, which holds 1,062 photos (22% of the pool);
   - by source (snpa, foe_carpaths, foe_bohemia).

**Truth is used only** by the simulated reviewers and for evaluation, as in all earlier simulations. Nothing is tuned on selection: no τ, ε or safeguard changes, and no reruns with different settings.

**One shot, enforced in code:**
- The runner refuses an existing output directory, and refuses to run if any completed selection run with the same freeze fingerprint exists.
- A failed run is preserved. A retry is allowed only if the failure happened **before any simulation result was computed** (for example an I/O or GPU error during extraction), in a new `_retry` directory, documented in the log.

## Declared criteria (fixed before the run)

Primary setting: τ 0.99, perfect reviewer.

| Question | Rule | Calibration value (F1, τ 0.99) |
| --- | --- | --- |
| **S1 accuracy holds** | ≤ 1% misassigned photos **and** \|count error\| ≤ 1 | 0.24%, +1 |
| **S2 the time saving holds** | Expert decisions per 1,000 photos ≤ 1.25 × calibration (≤ 416), **and** ≥ 50% fewer decisions than W1 on selection | 333 per 1,000; −60% vs W1 |
| **S3 honest self-assessment** | The automation audit's 95% interval contains the true bundle error; the count audit's interval contains the true count | Met (perfect reviewer) |
| **S4 reviewer-error robustness** | With 3% reviewer error, \|count error\| ≤ 2 and ≤ 3% misassigned | +1, 1.9% |

The five carried-over targets (≥ 70% automatic, ≤ 1% misassigned, \|count error\| ≤ 1, ≤ 0.5 h per 1,000 photos, audit coverage) are reported as well. Automation and hours are expected to fail, as on calibration.

**How results will be worded:**
- **S1–S3 met:** "the frozen operating point generalizes to unseen animals of the same dataset." That is still not park validation: same cameras, regions and annotation-assisted crops; upstream exposure unresolved.
- **Any S fails:** a documented generalization gap, with the breakdowns used to locate it. **No retuning on selection.** Any revised method would then need a new held-out evaluation, because selection is spent after this run. The official time_open test split stays locked.

## Proposed code changes (new files only; no edits to existing source)

1. **`reid_demo/benchmark_frozen_review.py`** (~350 lines):
   - JSON-serialized logistic models, with a `predict_proba`-compatible wrapper so the existing W4 and (a) simulation code runs unchanged;
   - frozen facing classifier and frozen risk reference;
   - configuration build, fingerprint and verification;
   - selection frame loader (protocol-validated pool membership, byte verification);
   - label-free context builder;
   - simulation and metrics, reusing `PrunedSimulation`, `summarize`, `count_audit`, `audit`, `simulate_catalogue`, `match_features`, `Context` and `fast_quality` unchanged. Every identity maps to the single frozen model.
2. **`tools/freeze_czechlynx_review_config.py`** (CPU, calibration only):
   - fits and writes the frozen models and configuration to `evaluations/czechlynx/frozen_review_config_v1/`;
   - runs an **in-sample calibration sanity run** of the frozen configuration, labelled in-sample. It only checks that frozen-model plumbing reproduces numbers close to the cross-fitted ones, and is not evidence.
3. **`tools/evaluate_czechlynx_frozen_review.py`** (GPU extraction, then CPU; selection only; one-shot locks):
   - writes `evaluations/czechlynx/frozen_review_selection_v1/` (manifest, summary, simulations, per-animal breakdowns, report).
4. **`tests/test_benchmark_frozen_review.py`** (synthetic):
   - JSON models reproduce sklearn probabilities;
   - frozen risk percentiles use the reference, not the test collection;
   - the facing classifier is frozen;
   - the configuration fingerprint changes on any setting or model change, and verification rejects tampering;
   - the selection runner refuses calibration input, existing outputs, a completed same-fingerprint run, and changed checkpoint, code or metadata;
   - an allowed retry happens only before simulation results;
   - single-model mapping, no labels in features, no neural imports at tool import, failed manifests preserved.

Untouched: existing source, all earlier outputs and caches, the F1/v2 trials and the official test split. Output paths are confirmed absent.

## Sequence and who runs what

1. User approval of this proposal. **This includes lifting the "no count configuration / no selection" rule for this one run, and permission for the assistant to commit the freeze record** (docs only) before the selection run.
2. Implement; run the CPU tests (assistant, if authorized).
3. Freeze (CPU, ≈ 2 min) and review the in-sample sanity run. Record the configuration fingerprint in the decision log and **commit it**: the pre-registration.
4. **One-shot selection run:** GPU extraction ≈ 3 min, then CPU ≈ 3 min. GPU work by the assistant needs explicit authorization; otherwise the assistant hands over the command.
5. Results doc and decision-log entry, with criteria S1–S4 reported mechanically.

## Limits

- Selection is the same dataset as calibration: same parks, cameras, annotation-assisted masked crops and day-level dates. It tests generalization to **new animals**, not to new sites, raw camera frames or real reviewers.
- Simulated reviewers and placeholder costs, as before.
- MiewID's upstream exposure to CzechLynx remains unresolved.
- No park outreach follows from this; readiness still needs the reviewer-timing trial, imperfect-auditor audits and raw-frame end-to-end evidence.

## Questions for the user

1. **Approve the scope?** That covers the frozen choices above (τ 0.99 primary, τ 0.98 secondary), criteria S1–S4, the one-shot locks, four new files, and lifting the selection rule once.
2. **Authorization:** may the assistant run the CPU tests, the freeze, the commit of the freeze record and the GPU selection run? Or should it hand over commands for you to run?
