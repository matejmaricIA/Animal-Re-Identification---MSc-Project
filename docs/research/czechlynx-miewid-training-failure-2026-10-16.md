# CzechLynx first adaptation failure and approved correction

Date: 16 October 2026; status updated 17 October. Real user-run v1 training failed. **The approved FP32 correction is implemented, the user reports 37 focused tests passed / 121 warnings in 11.61s, and v2 preparation/split review is complete.** [V2 review/receipt](czechlynx-miewid-fp32-preparation-results-2026-10-16.md) confirms exact v1 membership, current FP32/source/environment/protocol/audit bindings and preserved evidence. Next is user-run v2 training; real FP32 results remain pending. No retry, tests, toy/model inference or training was performed by the assistant. [Current handoff](../czechlynx-fp32-session-handoff-2026-10-16.md). Earlier pending statuses below are historical.

**Subsequent result, 17 October:** user-run FP32 v2 completed and [passed saved-evidence/CPU checkpoint review](czechlynx-miewid-fp32-training-results-2026-10-17.md), elapsed 39.97 min, selected epoch 6. Internal macro top-1/recall@50 improve by 5.9477/2.6806 percentage points. No numerical failure recurred in this trial; the exact cause of v1's failure remains unproven. Original failed-v1 artifacts are unchanged. Next is user-run full calibration comparison; reliable counts remain unvalidated. This supersedes the v2-training-pending handoff above without changing the original failure evidence below.

**Subsequent calibration review, 17 October:** the full adapted comparison is complete and [reviewed](czechlynx-miewid-fp32-calibration-results-2026-10-17.md). Calibration recognition improves, but none of the 25 controls yields reliable counts. Preserve completed evidence and the improved adapted baseline; next is a separately presented local/global candidate/pair/grouping scope. No configuration export or selection evaluation. The calibration-pending handoff above is historical; the original v1 failure findings below remain unchanged.

## Observed result

Trial: `models/czechlynx/miewid_adaptation_v1`. Its manifest records status `failed`, elapsed time 173.218 seconds, CUDA 12.6, NVIDIA GeForce RTX 4070 Ti SUPER, FP16 automatic mixed precision and initial gradient scale 1,024.

Frozen inner-holdout metrics are macro recall@50 **0.9263512632** and macro top-1 **0.6542339389**, on 2,460 images / 34 identities. Epoch 1 finished in 89.800 seconds with mean training loss 8.11597. Its full holdout metrics exactly match epoch zero, as expected for the head-only warmup. Lower training loss during warmup is not an embedding improvement.

The manifest proves an effective native embedding update at epoch 2, batch 1: parameter `backbone.global_pool.p`, finite gradient norm 9.63144. The later user log reaches batch 200, then reports a non-finite total gradient norm before the next progress line. The exact failing batch and offending gradients were not recorded. A completed epoch-2 evaluation or checkpoint does not exist: `epochs.json` contains only epochs 0 and 1, and the latest saved evaluation is epoch 1. Do not use this failed trial for calibration.

Read-only inspection confirmed the epoch history, latest evaluation, failed status, baseline equality and current source hashes against the manifest. The failed trial remains untouched.

| Saved artifact | SHA256 |
| --- | --- |
| `manifest.json` | `46d2fc8d3faebc95ad87ab09ca6b18b8b600d7b4ac3784c86cf952b9b552b58b` |
| `epochs.json` | `30b1a878f5cf6407d18eccf9c7a8446369d9aa0fcd03fd1bc2e998d8b35774a9` |
| `inner_validation_latest.json` | `4272bacfbcd4026b39373e16153d165070cbc0db293399e4908059d922543d42` |

## Diagnosis and limits

The code scales the loss, calls backward, unscales gradients, then calls `clip_grad_norm_(..., error_if_nonfinite=True)` before the optimizer/scaler step. Thus this exception stopped that batch before its optimizer update. The exception establishes a non-finite calculated total norm; the saved evidence cannot distinguish non-finite gradient entries from overflow while reducing otherwise finite gradients.

FP16 numerical instability is a plausible cause, not a demonstrated root cause. The native publisher code includes learned GeM power/root operations, and its pooling parameter was updated successfully at the start of epoch 2; neither fact identifies that layer as the failure source. The successful first update also means this was not an immediate inability to backpropagate through the native model.

[PyTorch 2.7 AMP guidance](https://docs.pytorch.org/docs/2.7/notes/amp_examples.html#gradient-clipping) specifies unscaling before clipping and describes GradScaler's step-skipping behavior for invalid gradients. The current strict norm check raises before that recovery path. This is a deliberately strict numerical policy, not evidence that the unscale/clip order is wrong. Disabling the finite-norm check alone is not a suitable correction. Full-precision regions are supported by the [AMP documentation](https://docs.pytorch.org/docs/2.7/amp.html); a full-precision training retry can test the mixed-precision hypothesis, without guaranteeing that it resolves the failure.

The initial commentary described a non-finite gradient; the precise established finding is a non-finite *total gradient norm*. Existing logs lack the extra information needed to resolve that distinction. No image, quality label or learning rate has been identified as the cause.

## Approved source-change scope — implemented, validation pending

The user approved and reaffirmed changes to these three source/test files; the fresh session implemented this scope:

| File | Approved change |
| --- | --- |
| `reid_demo/benchmark_miewid_adaptation.py` | Add a fixed, validated `training_precision: float32` setting to the trial/checkpoint recipe. The new source hashes and setting require a fresh prepared trial. Keep evaluation extraction precision and ranking controls as currently implemented. |
| `utils/finetune_czechlynx_miewid.py` | Run the training model/inputs and forward/backward computation in FP32, disable training autocast and loss scaling, and record the effective training/evaluation precision separately. Retain strict finite embeddings/loss/gradient checks, clipping, warmup-state preservation and actual-update proof. Record failure stage, epoch, batch, image IDs, gradient parameter names with non-finite counts, and whether finite entries produced an overflowing total norm. Inspect gradients only on failure to avoid a routine diagnostic cost each batch. Record the details before leaving the trial failed; apply no optimizer update on that failure. |
| `tests/test_benchmark_miewid_adaptation.py` | Extend the existing prepare-only/CPU native-shaped training regressions for the fixed precision contract and recorded runtime precision. Add controlled invalid-gradient and finite-gradient norm-overflow cases that verify structured failure details and no invalid optimizer update. Tests remain user-run. |

AGENTS.md, the decision log, experiment proposal and benchmark handoff track implementation separately from test status. Original baseline/audit source, inputs, review CSV/receipt, identity split, sampler, augmentation, learning rates, validation extraction and selection rule are preserved. The source change does not require repeating the human audit because its implementation binding excludes these adaptation files.

After implementing the approved correction and receiving user-run focused test results, hand the user the preparation command for a **fresh** `models/czechlynx/miewid_adaptation_v2_fp32` trial with the same metadata, outer protocol, audit receipt and seed; review that exact reconstructed split, then let the user start training. Do not resume or overwrite the failed v1 directory. FP32 may increase runtime and GPU memory consumption; actual fit/timing must be measured. Any required batch-size change would be explicit and recorded as a separate recipe change.

The correction fixes the recipe to `training_precision: float32` and records effective training and evaluation precision separately. Training autocast and gradient scaling are disabled. Failure manifests include `failure_context` and `numerical_failure`: exact stage/epoch/batch/image IDs and whether that batch reached its optimizer update. On backward/clipping failures, detailed inspection records gradient parameter names, NaN/positive-infinity/negative-infinity counts, original-dtype norm finiteness and FP64 norms. `nonfinite_gradient_entries` and `finite_gradient_norm_overflow` distinguish entry corruption from reduction overflow; aggregate overflow can have no individually overflowing parameter, with all contributing summaries retained. Non-finite numeric JSON values are represented by null plus explicit status/counts. Inspection runs only after failure; diagnostic or persistence errors do not replace the original exception.

Focused regressions cover the precision contract, preparation/checkpoint provenance, actual CPU training precision under inherited autocast, saved runtime metadata, invalid gradients and finite-entry individual/aggregate norm overflow. Failed training cases assert no optimizer update; diagnostic/save failure cases preserve the original strict exception. The user supplied the passing two-suite result after this correction: **37 passed, 121 warnings in 11.61s** (65 audit / 56 adaptation warnings). The previous 26-test result predates the patch. A successful retry still requires inner-holdout review before calibration, and does not by itself establish reliable grouping/counts, independently verified night/IR robustness, upstream pretraining independence or park readiness.

Implementation static checks: all three files parse/compile; Ruff F passes. AST comparison confirms all existing adaptation helpers except settings validation are unchanged, including exact split construction, sampler, transforms, evaluation extraction and checkpoint selection/provenance helpers. All 22 protected file hashes match the before-edit inventory, including every failed-v1 artifact and the saved audit receipt/reviews. After the user supplied the passing tests, read-only checks again confirm the failed-v1 manifest/audit receipt retain their recorded hashes and no v2 directory exists. No tests or model execution occurred by the assistant. User-run fresh preparation and saved-split review are next.
