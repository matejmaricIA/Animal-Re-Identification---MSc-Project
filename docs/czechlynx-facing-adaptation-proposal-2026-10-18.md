# Proposal: facing-aware re-adaptation of the MiewID embedding

Date: 18 October 2026. Work item (b) in the [decision log](lynx-demo-decision-log.md) "Active work" table. Status: **approved by the user 18 October 2026 as written, then implemented and run. This is a documented negative: R1 and R2 were not met by either arm. F1 is a modest improvement (protocol top-1 71.3% vs 68.6%; safe automation 54% vs 45%).** See [results](research/czechlynx-facing-adaptation-results-2026-10-18.md).

## Why

Two things are now established:
- **Item (a) found that review cost now depends mostly on the embedding** ([results](research/czechlynx-nonmatch-pruning-results-2026-10-18.md)). The best exact-count setting needs 2,211 expert decisions; 990 of them are one-click confirmations of bundles that could not be automated safely, and safe automation stops at 45%. Pruning and gate changes are exhausted; a sharper embedding is the remaining lever.
- **The flank results showed the binding constraint:** the embedding does not separate **same-side photos of different lynx** well ([flank results](research/czechlynx-flank-grouping-results-2026-10-18.md)). Lynx spot patterns differ between the left and right flanks.

The v2 adaptation trained ArcFace with one class per animal. That forces the two flanks of an animal, which share no spot pattern, into one class. The network can only do that by leaning on side-independent cues (size, coat tone, context) rather than spot patterns.

Read-only checks today support this:

| Check (calibration, saved vectors, no inference) | Public MiewID | Adapted v2 |
| --- | ---: | ---: |
| Facing predictable from the embedding (pose AUC, identity-cross-fitted) | **0.9995** | 0.9946 |
| Share of photos with confident facing (p < 0.2 or > 0.8) | 83% | 74% |
| Accuracy on confident pose-labelled photos | 99.8% | 99.5% |
| Cross-encounter nearest neighbour is the same animal: any side (micro / macro) | 77.7% / 62.4% | 81.4% / 67.1% |
| Same side | 76.6% / 62.1% | 80.2% / 66.7% |
| Opposite side | 30.2% / 14.9% | 32.8% / 18.2% |

The v2 adaptation reduced facing information in the embedding. It improved same-side retrieval only modestly (macro 62.1% → 66.7%). Opposite-side retrieval is near useless in both models, so pulling flanks together buys almost nothing. **Hypothesis:** training with (animal, facing) classes lets the network spend its capacity on side-specific spot patterns, raising same-side discrimination and therefore rank-1 share and safe automation. This is a hypothesis, not an established cause.

## Facing labels for the fitting pool (training data only)

Pose keypoints exist for only **2,415 of 16,504 fitting photos (14.6%)**, covering 114 of 165 animals, 98 of them with both facings.

Facing for the remaining photos comes from a **facing classifier on public MiewID embeddings**:
- trained on the fitting pose labels;
- identity-cross-fitted within fitting (5 folds), the same method as the flank work (`benchmark_flank_grouping.facing_probabilities`);
- on calibration, this classifier reaches AUC 0.9995.

Labels: pose where available; otherwise faces right if p > 0.8, faces left if p < 0.2, and **"other"** otherwise (frontal, rear or ambiguous; about 17% expected). Public MiewID vectors for the fitting photos do not exist yet, so this needs a one-off GPU extraction of about 16.5k photos with the pinned public model (minutes).

## Design (two training arms; everything else identical to v2)

Shared with v2 (`models/czechlynx/miewid_adaptation_v2_fp32`), so that only the class definition changes:
- **Initialization:** public MiewID-msv3 at the pinned revision.
- **Data:** the identical fitting inner split (plan fingerprint `ef8d6405…`, train 14,044 photos / 131 animals; holdout 2,460 / 34).
- **Training recipe:** FP32 (no autocast, as in the v2 correction); ArcFace scale 30 and margin 0.3; LR 2e-5 for the embedding and 1e-3 for the head; one head-only warm-up epoch; at most 10 epochs with patience 3; no resume.
- **Augmentation and batches:** same photometric augmentation, and **no horizontal flip**, since flipping would mislabel facing. Same balanced batches of 4 classes × 2 photos, preferring different encounters and cameras.

| Arm | Classes | What it tests |
| --- | --- | --- |
| **F1 explicit facing classes** | One ArcFace class per (animal, facing ∈ {left, right, other}): up to 393 classes for 131 animals | Whether separating flanks during training sharpens same-side discrimination |
| **F2 sub-center ArcFace** | One class per animal with K = 3 sub-centers (Deng et al. 2020); the network discovers view modes without facing labels | Whether the same gain comes without facing labels, which would be simpler and more robust in deployment |

**Checkpoint selection on the inner holdout** (development of the fitting pool only), fixed before training:
- primary: **macro same-facing cross-encounter top-1**, where the query and candidates have the same facing label and come from different encounters;
- tie-breaker: macro identity recall@50, then the earliest epoch.

The v2 rule is also reported per epoch, for comparison. An epoch-0 best means an unsuccessful adaptation, as before.

## Evaluation (calibration only; selection stays reserved)

For each completed arm:

1. **Calibration comparison,** composed exactly like the v2 comparison:
   - extract calibration vectors with the selected checkpoint into a new cache;
   - write the global reference;
   - run the same 25 clustering controls and the pair/connectivity diagnostic.

   This yields a calibration report that `load_calibration` accepts. It also reports any-side / same-side / opposite-side nearest-neighbour top-1 (as in the table above), protocol recognition (top-1, mAP@50, recall@50), facing predictability, and comparisons with **public MiewID and v2** on identical photos.
2. **Review cost,** the outcome that matters: the unchanged committed `tools/simulate_czechlynx_nonmatch_pruning.py` (all 148 simulations, CPU) on the new calibration report, with `--no-expect-saved-references`. The saved-number check is v2-specific; the live-W1 reference check still applies. The result is compared with v2's (a) results, especially the best setting: P(match) order, ε 0.001, τ 0.99, safeguards on.

## Declared rules (fixed before training)

Each arm is judged against v2 on the same calibration photos:

- **R1 same-side discrimination improves:** same-side cross-encounter nearest-neighbour top-1 rises by **≥ 3 points macro** (66.7% → ≥ 69.7%).
- **R2 review cost improves:** in the (a) best setting with a perfect reviewer, **≥ 15% fewer expert decisions than 2,211**, with \|count error\| ≤ 1 and ≤ 1% misassigned photos. Alternatively, max automation with ≤ 1% misassigned rises from 45% to **≥ 55%**.
- **R3 no hidden cost:** with 3% reviewer error, the best-setting count error stays within ±2 and misassignment no worse than v2's 2.3% + 1 point.
- **The five carried-over targets** (≥ 70% automatic, ≤ 1% misassigned, \|count error\| ≤ 1, ≤ 0.5 h per 1,000 photos at placeholder costs, audit coverage) are checked again. A configuration meeting all five would be a candidate operating point, pending the reserved selection evaluation, real reviewer timing and park data.

**If neither arm meets R1 and R2,** the result is recorded as a documented negative for facing-aware training on this data. No further training variants follow without a new proposal. If both arms pass, the one with fewer decisions in R2 is preferred; both are reported.

Known risk, measured rather than assumed: facing-aware training may make opposite-side photos of an animal even less similar. A bundle showing only the unseen flank of a known animal then depends on space-time evidence and the expert's scan. Pruning could turn such cases into duplicates. The (a) simulation counts these explicitly.

## Proposed code changes (new files only; no edits to existing source)

1. **`reid_demo/benchmark_facing_adaptation.py`** (~300 lines):
   - fitting facing-label construction (pose first, then the cross-fitted public-MiewID classifier, then "other"), bound by SHA-256;
   - (animal, facing) class maps;
   - the sub-center ArcFace head;
   - same-facing inner-holdout metrics and the new selection key;
   - trial preparation reusing `build_inner_plan`, which must reproduce v2's inner plan exactly;
   - trial loading and checkpoint provenance checks mirroring the native trial.

   It reuses `EncounterBatchSampler` and `ImageDataset` by passing a training frame whose class key replaces the identity column. The holdout frame keeps true identities.
2. **`tools/label_czechlynx_fitting_facing.py`** (~150 lines), GPU:
   - extracts public MiewID vectors for the fitting pool into a new cache `data/czechlynx/cache/miewid_public_fitting_v1/` (pinned model, same preprocessing and contracts);
   - writes `evaluations/czechlynx/fitting_facing_labels_v1/` (labels, cross-fitted pose AUC on fitting, label counts, manifest).
3. **`utils/finetune_czechlynx_miewid_facing.py`** (~330 lines), GPU:
   - the v2 training loop, copied and not modified in place, including FP32, failure diagnostics, gradient and update proofs, warm-up state check, and no resume;
   - `--arm classes|subcenter`;
   - `--prepare-only` first, with no neural imports.
   - Trials go to `models/czechlynx/miewid_facing_classes_v1/` and `models/czechlynx/miewid_facing_subcenter_v1/`.
4. **`tools/evaluate_czechlynx_facing_adaptation.py`** (~250 lines), GPU then CPU:
   - the calibration comparison composed from the existing stages (`write_global_reference` pattern, `run_experiment`, `run_diagnostic`);
   - facing-stratified retrieval comparison with public MiewID and v2;
   - output to `evaluations/czechlynx/facing_adaptation_calibration_{classes,subcenter}_v1/`, with caches `data/czechlynx/cache/miewid_facing_{classes,subcenter}_v1/`.
5. **`tests/test_benchmark_facing_adaptation.py`** (synthetic, CPU, no model downloads):
   - label precedence (pose over prediction over "other");
   - identity-cross-fitted facing labels;
   - class maps;
   - the sub-center head (shape, max over sub-centers, margin on the target only);
   - same-facing metrics and selection;
   - the inner plan equals v2's construction;
   - no horizontal flip in the transform;
   - trial and checkpoint provenance rejection;
   - prepare-only without neural imports;
   - selection refused, empty or new output directories, and failed manifests preserved.

The review-cost step uses the committed `simulate_czechlynx_nonmatch_pruning.py` unchanged. Outputs: `evaluations/czechlynx/nonmatch_pruning_facing_{classes,subcenter}_v1/`. All listed output paths are confirmed absent.

Untouched: existing source, the v1/v2 trials, all earlier outputs and caches, data, masks, splits and audits, and the reserved selection/test pools.

## Runtime, GPU and who runs what

| Step | Device | Estimate |
| --- | --- | --- |
| Public MiewID extraction for fitting facing labels | GPU (RTX 4070 Ti SUPER) | ≈ 5–10 min |
| F1 training (≤ 10 epochs × ≈ 280 s, as v2) | GPU | ≤ ≈ 50 min |
| F2 training | GPU | ≤ ≈ 50 min |
| Calibration extraction + 25 controls + diagnostic, per arm | GPU, then CPU | ≈ 5 min each |
| (a) review-cost simulation, per arm | CPU | ≈ 8 min each |

In total: **about 2 GPU-hours at most**, plus about 30 min of CPU.

**Sequence:**
1. Implement, then the user approves running the tests.
2. Prepare-only for both arms, with the inner plan checked against v2.
3. Facing labels, then review the label counts and fitting AUC.
4. F1 training, review, then F2 training, review.
5. Calibration and review-cost evaluation.
6. Results doc.

Each milestone is recorded in the decision log. Per AGENTS.md, model training and benchmarks are normally user-run. **GPU work by the assistant needs explicit authorization;** otherwise the assistant hands over exact commands.

## Limits

- **Upstream exposure:** MiewID's possible exposure to CzechLynx remains unresolved, as before.
- **Facing labels:** predicted facing labels contain some errors (about 0.2% on confident calibration photos), and "other" mixes frontal, rear and ambiguous views.
- **Evidence status:** calibration is development evidence. A gain here still needs the reserved selection evaluation, real reviewer timing and independent park data.
- **Night/IR** behaviour is not separately validated.
- **No count configuration, selection evaluation or outreach follows from this work.**

## Questions for the user

1. **Approve the scope as written?** That covers two arms (explicit facing classes, sub-center ArcFace), the fitting facing labels, the declared rules R1–R3, and five new files.
2. **GPU authorization:** may the assistant run the GPU steps (fitting extraction, two trainings, calibration extraction), about 2 GPU-hours? Or should it hand over the commands for you to run?
3. **Optional reduction:** run F1 only first, and F2 only if F1 fails R1? This saves about 50 GPU-minutes.
