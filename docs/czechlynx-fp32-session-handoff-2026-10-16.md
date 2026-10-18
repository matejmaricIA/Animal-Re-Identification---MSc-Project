# Fresh-session handoff: approved CzechLynx FP32 correction

Updated: 17 October 2026, Europe/Zagreb.

## Current status and next step

**Latest status, 17 October:** the approved FP32 correction, focused tests, preparation, completed training and **full calibration comparison/review are complete**. [Calibration review/receipt](research/czechlynx-miewid-fp32-calibration-results-2026-10-17.md): top-1 65.6437% → 68.6377%, macro top-1 56.0883% → 60.3884%, mAP@50 31.0078% → 40.1974%; all 25 grouping controls still fail reliable counts with zero exact identity-group recovery. Legacy yields 26 mixed groups for 55 identities; oracle k50 remains 10/55 connected, fragments 196 → 187. All 647 artifacts, both caches and saved assignment/count/pair arithmetic pass; original evidence/training/audit/failed v1 is preserved. **Next: present a concrete local/global candidate/pair-calibration/robust-grouping scope** based on these failures, with new-code approval required. Do not rerun training/calibration, export a count config or evaluate frozen selection. The [working diary](lynx-demo-decision-log.md) is authoritative; earlier pending instructions below are historical.

The requested fresh session implemented the approved three-file correction on 16 October 2026. Static AST/compile, Ruff F and whitespace checks pass. Static comparison confirms the existing split/sampler/augmentation, evaluation extraction, selection and checkpoint helpers are unchanged; the fixed FP32 setting flows through their existing provenance. The user completed fresh v2 preparation and its saved split/bindings have been reviewed, with final preservation checks on 17 October. All 22 protected file hashes match, including every failed-v1 file, audit receipt/reviews, metadata/protocol, baseline implementation and unrelated tracked modifications. No tests, toy/model runs, preparation, training, benchmarks or outreach were performed by the assistant.

**Focused suite passed, user-reported: 37 passed, 121 warnings in 11.61s.** All warnings are upstream pycocotools NumPy deprecations (65 audit / 56 adaptation). **V2 preparation and saved-plan review are complete; next is user-run training.** The [review/receipt](research/czechlynx-miewid-fp32-preparation-results-2026-10-16.md) confirms exact v1/reconstructed membership, fixed FP32 settings and current source/environment/protocol/audit bindings. Run fingerprint: `675d43e66317e459cf7d0672789ed19d4a3db1d91f1e606dc7b8233b3e8cd22d`. The prepared directory contains only `manifest.json` and `inner_plan.json`; final checks confirm neither changed. Use the same prepared v2 command without `--prepare-only`, then review completed/failed evidence before calibration. Preserve the existing human audit and failed v1. Real FP32 training and reliable clustering/count gains remain open.

## Authorization and original task

The user explicitly said: **“I approve this correction, but I would like to execute it in a fresh session”**, then requested documentation and a continuation prompt. The fresh session subsequently reaffirmed and implemented the three-file source/test correction without repeating approval. Material expansion beyond this scope still needs approval.

The original handoff session only recorded the approval and prepared documentation. At that handoff, recorded implementation hashes still matched failed v1 and the FP32 setting was absent. Source hashes/settings subsequently changed as approved; the failed trial, audit and data remain preserved. The user has now prepared `models/czechlynx/miewid_adaptation_v2_fp32`; the saved plan is reviewed. No tests or models were run by the assistant in either session.

Repository: `/home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project`.

Read `AGENTS.md`, `docs/lynx-demo-decision-log.md`, this handoff, and `docs/research/czechlynx-miewid-training-failure-2026-10-16.md` before implementing. Use the latest approval/status entries; earlier approval-pending and ready-to-start-v1 notes describe historical states.

## Intended outcome

The user wants stronger technical evidence before offering collaboration to Risnjak National Park, initially for lynx monitoring/counting from individual markings. Reliable clustering and defensible counts of photographed individuals are required; retrieval suggestions alone do not satisfy the goal. Photographed-individual counts are not total park abundance. No outreach is authorized.

The completed frozen calibration controls did not produce reliable clusters/counts. On 6,352 calibration images / 55 identities, all 25 profiles had zero exact single-pure-group identity recovery. A truth-filtered, unthresholded k50 diagnostic connects only 10/55 identities, in 196 fragments; this is an oracle diagnostic, not deployed counting. The approved next experiment adapts the native MiewID embedding on fitting data, then assesses full calibration recognition, pair evidence, connectivity and grouping/count errors if the inner holdout improves. Clustering selection remains unrun, with no configuration exported. Upstream MiewID exposure to CzechLynx photographs/individuals remains unresolved.

## Work already complete

### Input audit and review receipt

- User-run input preparation and all **321 human reviews** are complete, following 318 AI drafts. Saved export matched every live completed payload/revision.
- The assistant ran the existing model-free summarizer successfully: `evaluations/czechlynx/input_audit_review_v1/receipt.json` and `reviews.json` exist.
- Receipt SHA256: `ffa4000e8cccc219dc3e53da82df2714263b12f3e878d448416f2031c63c93f3`.
- Preserve the bound source CSV at `/home/mm-workstation/re-id/czechlynx-review-app/exports/reviews_complete_20261016T130044652360Z_bccd027d.csv`, SHA256 `4e5285142551e66a7f1b93301032bcdf2e1b0e54da05515eac9636812af2244e`.
- The standalone review app lives outside the repository at `/home/mm-workstation/re-id/czechlynx-review-app`. No app changes are needed.
- Uniform fitting/calibration samples: unusable markings 55/128 and 27/128; crop/mask issues 24/128 and 20/128. These are reviewed sample judgments; combined purposive cohorts cannot estimate a whole-dataset error rate.
- Some verified IR/night evidence relies on appearance. Preserve submitted labels; they do not establish independently verified capture conditions. Audit labels do not enter training loss or inference.
- All saved preparation alignment checks reproduced the supplied annotations exactly. Annotation correctness and preparation fidelity are separate questions. No images, masks, identities or membership were corrected or excluded.

Details: `docs/research/czechlynx-input-audit-review-results-2026-10-16.md`. **Do not repeat the human audit.** Its implementation binding excludes the three files in the approved correction.

### Prepared inner split

The user ran and the assistant reviewed `--prepare-only` for `models/czechlynx/miewid_adaptation_v1`:

| Role | Images | Identities | Atomic groups | Encounters |
| --- | ---: | ---: | ---: | ---: |
| Training | 14,044 | 131 | 123 | 6,348 |
| Validation | 2,460 | 34 | 33 | 1,297 |

All 16,504 fitting images remain. Whole linked groups stay together. Cross-role identity, encounter, source-photo and prepared-photo hash overlap is zero. No outer calibration/selection or official-test input entered the split. Every validation image has a legal same-identity cross-encounter nonduplicate positive; this establishes eligibility, not model accuracy. A 20% identity holdout is not necessarily 20% of images.

Outer protocol fingerprint: `e3401b266c0e9af162ecd4d16530c0e9f6d5d7f623920c2e24a6af18c04a5fa5`.

Inner-plan fingerprint: `ef8d64056a89ef7bce2e37db9d089bd86ce92ff12b922f9f4acbd951db0c88e7`.

Details: `docs/research/czechlynx-miewid-training-preparation-results-2026-10-16.md`.

### First real training run failed

The user ran the prepared trial. Saved `models/czechlynx/miewid_adaptation_v1/manifest.json` records `status: failed`, elapsed 173.218 seconds, NVIDIA GeForce RTX 4070 Ti SUPER, CUDA 12.6, FP16 AMP and initial gradient scale 1,024.

- Frozen inner holdout: macro recall@50 **0.9263512632**, macro top-1 **0.6542339389**.
- Head-only epoch 1 finished in **89.800 seconds**, mean loss 8.11597, with full holdout metrics exactly unchanged from epoch zero. This is expected during warmup.
- Epoch 2 batch 1 proved a native embedding parameter update: `backbone.global_pool.p`, finite gradient norm 9.63144.
- After progress batch 200, a later batch raised `RuntimeError: The total norm of order 2.0 for gradients from parameters is non-finite, so it cannot be clipped`.
- The exact failing batch, parameter gradients and effective scale at failure were not recorded. FP16 instability is plausible, not established. The failure could involve non-finite gradient entries or overflow when reducing otherwise finite entries.
- The strict check stopped that batch before its optimizer step. Do not disable it or silently skip difficult observations.
- `epochs.json` contains only epochs 0 and 1; latest validation is epoch 1. There is no completed adaptation epoch/checkpoint eligible for calibration.
- Preserve the failed v1 directory. No training resume is implemented. Do not edit or re-sign its manifest to make it reusable.

Failed manifest SHA256: `46d2fc8d3faebc95ad87ab09ca6b18b8b600d7b4ac3784c86cf952b9b552b58b`.

Failure details and artifact hashes: `docs/research/czechlynx-miewid-training-failure-2026-10-16.md`. The current loss-scaling/unscale/clip order matches the relevant PyTorch guidance; the strict exception occurs before GradScaler's recovery step. Do not assert a wrong clipping order or a proven GeM defect.

## Approved implementation scope — implemented

Exactly three source/test files, plus documentation/status updates:

1. **`reid_demo/benchmark_miewid_adaptation.py`**
   - Add and validate a fixed `training_precision: float32` recipe setting.
   - Bind it through existing preparation/checkpoint provenance; new source hashes/settings require a fresh prepared trial.
   - Keep validation/extraction precision and ranking controls as currently implemented. FP32 refers to training, not a blanket change to evaluation.
2. **`utils/finetune_czechlynx_miewid.py`**
   - Enforce FP32 training model/inputs and forward/backward computation; disable training autocast and gradient loss scaling.
   - Record effective training and evaluation precision separately, without misleading `amp` metadata.
   - Retain strict finite embedding/loss/gradient checks, clipping, fixed BN running statistics, head-only warmup preservation and actual embedding-update proof.
   - On numerical failure, save stage, epoch, batch, batch image IDs, offending parameter names and non-finite gradient counts. Distinguish non-finite entries from norm-reduction overflow with finite entries. Inspect detailed gradient contents on failure, avoiding routine per-batch scans.
   - Persist useful diagnostics without masking the original failure or applying that batch's optimizer update. Avoid nonstandard NaN/Infinity JSON values or raw tensor dumps.
3. **`tests/test_benchmark_miewid_adaptation.py`**
   - Extend prepare-only/native-shaped CPU training coverage for the fixed precision contract and runtime precision metadata.
   - Add meaningful controlled cases for invalid gradients and overflowing total norms from finite entries; verify structured diagnostics and no invalid optimizer update.
   - Tests stay user-run. Do not execute CPU toy training yourself under the guise of verification.

Keep the original audit/baseline implementation, data, masks, identities, split, sampler, augmentation, learning rates, validation extraction and checkpoint-selection rule intact. The approved first trial preserves original image inclusion despite the quality concerns. Any cleanup/exclusion, batch-size change, BF16 option, skip/retry strategy, model/GeM modification or broader scope needs its own concrete approval.

The current native path is EfficientNetV2-M + learned GeM + BN with 2,152-D embeddings, a separate ArcFace training head, fixed BN statistics, batch four identities x two observations, maximum ten epochs including one head warmup, patience three. Checkpoint selection is macro cross-encounter recall@50, then macro top-1, then earlier epoch; epoch zero remains eligible. Keep this recipe except the approved precision correction/diagnostics.

## Execution sequence for the fresh session

Steps 1–7 and benchmark calibration step 5 are complete, including saved-evidence reviews. The FP32 correction experiment is finished; it improves recognition but does not achieve reliable counts. Next is a separately presented local/global/candidate/grouping scope. Preserve completed evidence and reserved selection; do not repeat completed commands without a new reason.

1. Read instructions/evidence; inspect the three current files and related existing tests. Check current git status and failed evidence without rerunning old experiments.
2. State that the correction is already approved and summarize the planned changes; implement the approved scope without another approval request.
3. Update AGENTS.md, decision log, failure/proposal status and benchmark instructions. Static syntax/lint/whitespace checks are allowed. Do not claim old tests validate the correction.
4. Give the user this focused command and await their output:

   ```bash
   venv/bin/python -m pytest \
     tests/test_benchmark_input_audit.py \
     tests/test_benchmark_miewid_adaptation.py -q
   ```

   Completed by the user after the correction: 37 passed / 121 upstream pycocotools NumPy warnings in 11.61s. The earlier 26-test result predates the patch.
5. After passing results, hand the user the new model-free preparation command below. Confirm the current directory is fresh; if already occupied, inspect it and choose a new name instead of overwriting.

   ```bash
   venv/bin/python utils/finetune_czechlynx_miewid.py \
     --metadata data/czechlynx/prepared_masked/masked/metadata.csv \
     --protocol-plan data/czechlynx/protocols/identity_pools_v1.json \
     --audit-receipt evaluations/czechlynx/input_audit_review_v1/receipt.json \
     --out-dir models/czechlynx/miewid_adaptation_v2_fp32 \
     --prepare-only --threads 4 --workers 4
   ```

6. Review the saved v2 plan/current bindings and confirm the intended split matches v1. Then give the user the same command without `--prepare-only`. FP32 may take more time/memory; do not promise a runtime from the FP16 warmup. The original workload is 1,756 batches/epoch, at most ten epochs, with early stopping. Epoch 2 is the first embedding-training epoch.
7. Review completed/failed manifest and epoch history. If epoch zero stays best, record unsuccessful adaptation before considering calibration. If adaptation improves the inner holdout, review it before handing over the existing full calibration comparison with new v2 trial/cache/output paths. Do not reuse old adapted caches or launch calibration automatically.

Tests, model runs, training and performance benchmarks remain user-run unless explicitly authorized otherwise. Do not start them in the fresh session. Do not contact parks or authors. No commits, reset, clean, staging or unrelated edits are requested. The working tree contains substantial pre-existing changes and untracked implementation files; preserve them. Files are shared in place, so a fresh conversation sees the same uncommitted workspace.

## Continuation prompt after implementation

```text
Continue CzechLynx MiewID FP32 validation in /home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project.

Read AGENTS.md, docs/lynx-demo-decision-log.md, and docs/czechlynx-fp32-session-handoff-2026-10-16.md, including its linked failure evidence.

The approved three-file correction is implemented: fixed FP32 training recipe/provenance in reid_demo/benchmark_miewid_adaptation.py; FP32 training with autocast/scaling disabled, separate evaluation precision and precise numerical diagnostics in utils/finetune_czechlynx_miewid.py; and focused precision/invalid-gradient/norm-overflow regressions in tests/test_benchmark_miewid_adaptation.py. AST/compile, Ruff F and whitespace checks pass. Do not reimplement or ask for the same approval. The user reports 37 tests passed, 121 upstream warnings in 11.61s. Fresh v2 preparation and exact saved-plan/binding review are complete; see docs/research/czechlynx-miewid-fp32-preparation-results-2026-10-16.md. Real FP32 training remains pending.

Preserve the completed audit/receipt, original data and splits, evaluation precision, baseline evidence, and failed models/czechlynx/miewid_adaptation_v1 directory. Do not disable finite checks, silently skip batches, or resume v1. Use a fresh miewid_adaptation_v2_fp32 trial after validation.

I run tests, model runs, training and benchmarks. FP32 v2 training and full calibration comparison are complete and reviewed; see docs/research/czechlynx-miewid-fp32-calibration-results-2026-10-17.md and its receipt. Recognition improved, but all 25 controls still fail reliable grouping/counting. Do not rerun completed experiments or use frozen selection. Preserve the improved adapted recognition baseline, original controls/data/audit/failed-v1 evidence and unrelated work. Present a concrete source-change scope for a controlled thesis local/global candidate/pair-calibration/robust-grouping experiment; diagnose prior integration rather than assuming a fault or benefit. Keep docs/lynx-demo-decision-log.md current. Reliable photographed-individual counts remain required before approaching Risnjak; no outreach is authorized.
```
