# CzechLynx input audit and first MiewID adaptation trial

**Latest status, 17 October:** the approved audit/adaptation experiment and FP32 correction are **complete through training and full calibration review**. [Calibration review/receipt](research/czechlynx-miewid-fp32-calibration-results-2026-10-17.md): top-1 65.6437% → 68.6377%, macro top-1 56.0883% → 60.3884%, mAP@50 31.0078% → 40.1974%; none of the 25 grouping controls gives reliable counts, exact identity-group recovery zero throughout. All source/input/trial/cache/artifact checks pass and original evidence is preserved. Keep the improved adapted recognition baseline; next present a concrete local/global candidate/pair-calibration/robust-grouping scope. No count config or frozen selection evaluation; local integration, night/flank/independent validation and readiness remain open. Earlier pending statuses below are historical.

Date: 14 October 2026; status updated 16 October. **Status: approved and implemented; user reports 26 focused tests passed, 93 warnings in 9.35s. All 321 human input reviews, the summarized receipt and training preparation are now complete.** The user agrees with the [pair/connectivity review's recommendations](research/czechlynx-pair-connectivity-results-2026-10-14.md), approved this concrete scope and reaffirmed continuation. Static syntax/lint review is complete. Actual training and calibration validation remain user-run/pending. [Handoff commands](czechlynx-benchmark.md#input-audit-and-native-miewid-adaptation).

## Purpose and sequence

**Real audit and human review complete:** User-run preparation contains 321 unique crops, with 966 artifact hashes/current sampled source/prepared bytes verified and exact saved source-annotation alignment throughout. The user completed all 321 records through the sibling browser app; the final export matches live payloads/revisions and was imported by the existing model-free summarizer into `evaluations/czechlynx/input_audit_review_v1/receipt.json`. [Completed findings](research/czechlynx-input-audit-review-results-2026-10-16.md). Uniform fitting/calibration include 55/128 and 27/128 unusable-marking judgments and 24/128 and 20/128 crop/mask issues. Preserve the approved unchanged-input first trial; any corrections or quality exclusions require a separate scope. Some verified illumination notes are appearance-based and do not establish independently verified capture conditions. Training preparation and inventory review are also complete, as recorded below; real training/calibration remain pending.

**User-run test receipts:** First attempt: 21 passed, five failed, 93 upstream pycocotools NumPy deprecation warnings in 6.46s. Corrected its shared inner-plan pandas index/column sorting ambiguity and extended the existing invariance regression for already-indexed input. Pre-FP32 rerun: 26 passed, 93 warnings in 9.35s. Latest result after the approved FP32 correction: **37 passed, 121 warnings in 11.61s** (65 audit / 56 adaptation warnings), covering precision/provenance, numerical-failure/no-update and synthetic native-shaped training/composed calibration paths. This does not establish real-model or field accuracy. The assistant has not executed tests or models.

**User-run training preparation complete and reviewed (16 October):** train 14,044 images / 131 identities; validation 2,460 / 34. Exact reconstruction/current bindings match, all fitting images remain, cross-role identity/encounter/source/prepared-hash overlap is zero, and all holdout images have a legal metadata-defined positive. Next is the same command without `--prepare-only`, then review the inner-holdout result before calibration. [Saved-plan evidence](research/czechlynx-miewid-training-preparation-results-2026-10-16.md).

Determine whether adapting the strongest available embedding model improves identity evidence across lynx photographs. The current unthresholded k50 graph connects only 10/55 identities even after truth removes every incorrect edge. Better recognition alone will not establish photographed-individual counts.

1. Produce and review a bounded image/annotation audit.
2. Run one declared MiewID adaptation recipe on the existing fitting pool, with an inner identity-disjoint holdout.
3. Compare calibration recognition, pair evidence, candidate connectivity and actual grouping/count errors with the completed frozen reference.
4. Use the results to specify broader candidates and calibrated global/local robust merging.

Tests, audit commands, training and experiments remain user-run. Audit review precedes training. The new evaluator supports calibration only; clustering selection stays reserved for a subsequently chosen, frozen method.

## Read-only findings

### Suspect crop faithfully reproduces the supplied annotation

For `clx_e981b521482a803cb7855493` / `lynx_280`, direct original-metadata/RLE decoding and pixel comparison found one matching original row, an annotation bbox of x/y/width/height **120/568/1,919/1,022**, and **exact equality** between the prepared mask and cropped annotation and between prepared RGB and the declared masked source crop. The foreground fraction is 99.6864%.

No preparation mismatch was found in this case. Source annotation/identifiability still needs review; this does not establish an empty frame, wrong identity or general annotation correctness. [Evidence receipt](research/czechlynx-adaptation-feasibility-2026-10-14.json).

### Preserve linked fitting groups and native architecture

The fitting pool has **16,504 images / 165 identities in 156 atomic identity groups**. Five groups contain multiple identities (largest six), bound by encounters/duplicate constraints. They are split units, not same-animal assertions. Images per identity range from 1 to 1,441, median 31; `lynx_048` has one encounter. Split whole linked groups and balance training by identity.

The inspected cached publisher code uses `efficientnetv2_rw_m`, learned GeM pooling and a native 2,152-D batch-normalization output. Its forward returns embeddings. Preserve that full path, attaching a separate training head. The existing MegaDescriptor trainer uses a different architecture.

## Exact seven additions

| File | Planned change and reason |
| --- | --- |
| `reid_demo/benchmark_input_audit.py` | Deterministic cohorts, source/prepared/mask checks, review schema and cohort summaries; establish usable input evidence. |
| `tools/audit_czechlynx_inputs.py` | Model-free report/review-CSV preparation and completed-review import; make manual inspection practical. |
| `tests/test_benchmark_input_audit.py` | Verify sampling, alignment, split restrictions, unknown review labels and protected outputs. |
| `reid_demo/benchmark_miewid_adaptation.py` | Fitting-only inner split, encounter-aware identity sampler, transforms/head and strict native checkpoint contracts; keep adaptation auditable. Neural imports remain lazy. |
| `utils/finetune_czechlynx_miewid.py` | Prepare-only preflight, declared training recipe, inner-holdout evaluation, timings and best/last checkpoints; GPU execution stays user-controlled. |
| `tools/evaluate_czechlynx_miewid_adaptation.py` | Standalone full calibration comparison, adapted cache extraction and composition of existing clustering/pair-diagnostic interfaces; measure the count target. |
| `tests/test_benchmark_miewid_adaptation.py` | Verify leakage barriers, sampler fallbacks, effective gradients, native checkpoint round trips, cache isolation and complete-run handling using synthetic models. |

Existing benchmark, adapter, clustering and diagnostic source files are used through their current interfaces. Any necessary change to them would need an amended presentation before implementation. Documentation will update AGENTS.md, the decision log, implementation plan, benchmark commands and this proposal. No new dependency installation is proposed.

## A. Input audit

- Reconstruct the frozen protocol and validate prepared metadata. Sample only fitting and eligible calibration records from official time_open train.
- Seed 42: **128 uniform fitting images + 128 uniform calibration images**, one seeded image per calibration identity, and the 11 reviewed diagnostic crops. Deduplicate display records but retain all cohort memberships; maximum 322 records.
- Summarize uniform, identity-coverage and diagnostic-extreme cohorts separately. Selected extremes cannot estimate prevalence.
- Show original photograph, crop and mask overlay. Check bytes, dimensions, source annotation bbox and prepared pixel alignment; failures remain explicit.
- Review animal visibility, usable markings, crop/mask correctness, blur/occlusion, flank and illumination evidence, with reviewer/time/notes and unknown values. Monochrome alone cannot become verified IR/night. Visible markings do not by themselves prove successful identity determination.
- Import reviews into a checksummed receipt recording completeness, failures and unresolved questions. Protect original metadata, identity labels, images, annotations, caches and earlier reports.
- Reviews do not automatically delete images, relabel identities, impose same-animal encounter links or establish a learned quality-rejection policy.

Review this evidence before training. Any confirmed annotation/preparation correction gets a specific scope. Otherwise preserve fitting membership and every eligible calibration photograph. Audit labels do not enter training loss or calibration inference; unresolved quality questions remain visible.

## B. Training contract

### Roles and checkpoint selection

- Reconstruct the existing `identity_pools_v1` plan; restrict all training/inner-validation inputs to fitting.
- Deterministically split approximately 80/20 by **whole fitting identity groups**, balancing source representation; record exact IDs and deviations from indivisible groups. Reconstruct membership on load.
- Require zero inner cross-split identity/encounter/source-hash/prepared-hash overlap. Outer calibration, selection and official time_open test never enter training or inner validation.
- Inner validation ranks images of its own unseen identities against other inner-validation images, excluding self, shared encounters and duplicate source/prepared photographs. Report queries/identities with no eligible positive separately without dropping them from the inventory.
- Select by **macro-identity cross-encounter recall@50**, then macro top-1, then earlier epoch. Frozen epoch zero is a reference and eligible best checkpoint. Camera/appearance/connectivity results are diagnostics, not additional selection knobs.
- This first trial trains the inner-training portion only. It does not automatically refit all fitting identities or select checkpoints using outer calibration.

### One declared recipe

| Setting | Proposed default |
| --- | --- |
| Initialization / output | Current pinned complete MiewID-msv3; native 2,152-D embeddings, L2-normalized for cosine/ArcFace |
| Loss | Separate ArcFace head, scale 30 and margin 0.3, on inner-training identities; excluded from inference |
| Trainable embedding path | Backbone, learned GeM and embedding batch-normalization affine parameters; publisher final layer excluded |
| Optimizer | AdamW; embedding LR `2e-5`, head LR `1e-3`, weight decay `1e-4` |
| Batch | Four identities x two observations; prefer different encounters, then cameras, within labelled identity |
| Small-identity fallback | Log repeated/augmented observations when distinct examples are unavailable |
| Transform | RGB, 440 x 440 bilinear resize, ImageNet normalization |
| Augmentation | Mild brightness/contrast/saturation jitter (0.1), grayscale probability 0.2; preserve crop geometry |
| Batch normalization | Fixed running statistics; permitted embedding affine parameters train; native state/dimensions preserved |
| Schedule | Maximum ten epochs; first head warmup, then embedding adaptation with cosine LR; patience three |
| Numerical controls | Fixed FP32 training model/head/inputs; training autocast and scaling disabled; strict finite checks and gradient norm clipped at 1.0; structured failure-only diagnostics distinguish invalid entries from finite-entry norm overflow |
| Evaluation precision | Unchanged CUDA FP16 autocast (CPU FP32), FP32 normalization/stored matrices; runtime metadata separate from training precision |
| Seed | 42; record loader/sampler seeds, device and environment |

This compares one combined treatment with frozen MiewID. It cannot attribute effects separately to augmentation, sampling or fine-tuning. Grayscale augmentation does not establish actual night/IR or opposite-flank robustness.

Keep the publisher's unused final layer unchanged. Demonstrate finite, effective gradients in intended embedding parameters after warmup. Preserve native state keys; distinguish complete native embedding weights from the separate training head.

### Preflight and artifacts

- `--prepare-only` validates metadata, byte bindings, completed audit receipt and settings and writes the inner plan/inventory without neural loading or GPU allocation.
- Actual training requires that matching plan/receipt and prints membership/workload before loading the model.
- Write `inner_plan.json`, `manifest.json`, epoch metrics/timings, `best.pt`, `last.pt` and inner-validation summaries. Bind image/mask bytes, train/holdout IDs, native model/source/config/environment, trainable groups and selected epoch.
- Strict versioned, tensor-safe checkpoint loading rejects state-key/dimension/architecture/revision/preprocessing/provenance mismatch.
- No training resume is included. A matching prepared directory can advance once; started/failed/completed runs cannot be overwritten.
- If epoch zero stays best, report failed adaptation on inner holdout and review before commissioning calibration. Saving weights is not an improvement.

## C. Full calibration comparison

- Validate completed checkpoint/training receipt, original completed calibration reference and output protections before loading the adapted model.
- Evaluate **all 6,352 eligible images / 55 identities**, with original gallery/query membership and explicit exclusions. No quality-selected dropping or sampling caps.
- Read baseline predictions/vectors from saved artifacts, verifying their recorded bytes/contracts. Do not rerun the frozen model or replace historical source hashes with current ones.
- Extract adapted vectors into a distinct cache bound to native state, checkpoint/training provenance, transforms, input bytes, environment and implementation. Keep baseline cache read-only.
- Produce a standard completed calibration global-reference directory with existing manifest/gallery/query/vector interfaces; invoke current clustering and pair diagnostics on it. No frozen source changes are needed by design.
- Keep the existing **25 clustering profiles, k15/k50 budgets and six cutoffs** for a controlled representation comparison. Cutoffs are sensitivity controls, not probability thresholds. Any fresh recognition rejection threshold is reported as in-sample and separate from clustering.
- Compare micro/macro recognition, camera/appearance proxies, same/different pair evidence, oracle connected identities/fragments/isolated images, actual exact group recovery, component pair precision/recall, merges, fragmentation, count errors, sparse identities and unresolved observations. Truth enters after candidate/cluster inference.
- Combine global, clustering and pair reports with a baseline comparison page. Complete status requires every stage and required artifact/observation; partial failures remain failed.
- No automatic cluster-config export; no selection mode in this first evaluator.

## Verification and proposed user workflow

Focused synthetic checks must cover edited plans, linked mixed-identity encounters, duplicate hashes, singleton/single-encounter animals, unknown/stale audit reviews, absent gradients, incompatible native checkpoints, training-head exclusion, checkpoint cache separation, label-free inference, baseline preservation and partial-run failure.

The seven additions are implemented; the two regression suites are written but not run by the assistant. The [command handoff](czechlynx-benchmark.md#input-audit-and-native-miewid-adaptation) provides:

1. The two new focused pytest suites, including integration with current clustering/diagnostic interfaces.
2. Audit preparation, manual CSV review and review-summary import.
3. Trainer `--prepare-only` and inspection of exact membership/settings.
4. User-run training; then review its inner-holdout result.
5. User-run calibration comparison if that result warrants it.

Commands will name the current metadata/protocol, source receipts and separate output/cache paths. Training logs report steps and epoch timing; runtime estimates require the first measured epoch. No training has been executed to estimate it.

## Decision after this trial

An improved inner holdout followed by better calibration connectivity and grouping would support further candidate/pair/graph development. Better top-1 alone does not achieve counting. If adaptation fails, retain the frozen reference and reconsider the treatment from its per-identity/view evidence.

Broader/encounter-diverse candidates, calibrated DISK/LightGlue evidence and robust merging remain next. Added candidates alone can introduce false bridges; previous RoMa evidence does not justify making it the default.

Upstream exposure is unresolved; local holdouts cannot undo it. Quality rejection, actual night/IR/flank evidence, appropriately independent final evaluation, count uncertainty, acceptable review burden and safe park ingestion remain open. This trial does not complete the count milestone or qualify Risnjak abundance estimation.
