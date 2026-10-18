# CzechLynx FP32 training preparation review

Prepared and reviewed: 16 October 2026. Final preservation check and training handoff: 17 October 2026, Europe/Zagreb.

The user completed model-free `--prepare-only` for `models/czechlynx/miewid_adaptation_v2_fp32`. The saved directory contains exactly `manifest.json` and `inner_plan.json`, with status `prepared`. The assistant independently reconstructed the split and checked its source, environment, protocol and audit bindings without neural imports, tests, model inference, training or performance benchmarks. The final check confirms the reviewed files remain unchanged.

## Saved contract

| Item | Value |
| --- | --- |
| Run fingerprint | `675d43e66317e459cf7d0672789ed19d4a3db1d91f1e606dc7b8233b3e8cd22d` |
| Inner-plan fingerprint | `ef8d64056a89ef7bce2e37db9d089bd86ce92ff12b922f9f4acbd951db0c88e7` |
| Prepared manifest SHA256 | `da39c1bfb9c16d3771635a8fcae794109d9fefcbbad0be8fdba910bd1234b55f` |
| Inner-plan SHA256 | `9351779bc068cdec598f1383d65a7876e5731b900a6a09496603463b1481d316` |
| Audit receipt SHA256 | `ffa4000e8cccc219dc3e53da82df2714263b12f3e878d448416f2031c63c93f3` |
| Training precision | Fixed `float32`; training autocast and loss scaling disabled |

Current settings exactly equal the declared defaults. Relative to failed v1, the only recipe change is `training_precision: float32`; the changed bound implementation files are exactly `reid_demo/benchmark_miewid_adaptation.py` and `utils/finetune_czechlynx_miewid.py`. Evaluation extraction, sampler, augmentation, learning rates and selection rule are preserved. Manifest/run fingerprints, current package environment and all implementation hashes match.

## Split and inclusion

| Role | Images | Identities | Atomic groups | Encounters | Bohemia / Carpathians / SNPA images |
| --- | ---: | ---: | ---: | ---: | --- |
| Training | 14,044 | 131 | 123 | 6,348 | 3,023 / 6,578 / 4,443 |
| Validation | 2,460 | 34 | 33 | 1,297 | 515 / 707 / 1,238 |

The saved inner plan is exactly equal to failed v1's original plan and independent reconstruction with seed 42 and holdout fraction 0.2. All 16,504 fitting images remain; whole atomic groups are preserved. Cross-role identity, encounter, source-photo hash and prepared-photo hash overlap is zero. No outer calibration, selection or official `time_open` test image enters either role. All 2,460 validation images have a legal same-identity positive under the existing encounter/source/prepared-hash exclusions. This is metadata eligibility, not evidence of recognition accuracy or upstream pretraining independence.

The fitting metadata digest matches v1. The user's prepare-only path verified all fitting image/mask bytes. The assistant's read-only receipt validation independently rechecked the audit's artifacts, bound source CSV/imported reviews and sampled image/mask bytes. No human review was repeated or altered. All 22 previously protected file hashes match, including every failed-v1 artifact, audit receipt/reviews, baseline implementation and unrelated tracked modifications. Failed-v1 manifest SHA256 remains `46d2fc8d3faebc95ad87ab09ca6b18b8b600d7b4ac3784c86cf952b9b552b58b`.

Machine-readable evidence: [review receipt](czechlynx-miewid-fp32-preparation-results-2026-10-16.json), SHA256 `9dc80f08360a3c568fda85d04bb0f9a83cff07f760c25068e5b3ed8b5370eb89`.

## Training handoff

The user-reported post-correction focused suite passed: **37 passed, 121 upstream pycocotools NumPy warnings in 11.61s**. Preparation and saved-split review are now complete. Next is the [user-run training command](../czechlynx-benchmark.md#4-user-run-gpu-training), removing only `--prepare-only` from the same v2 command.

Workload: maximum ten epochs of 1,756 balanced batches, four identities x two observations, one head-only warmup epoch, embedding adaptation from epoch 2, patience three. Epoch zero remains eligible under macro cross-encounter recall@50, macro top-1, then earliest epoch. FP32 runtime/memory remains unmeasured; the failed FP16 warmup timing does not predict it. No resume or automatic batch-size changes are authorized.

Share completed/failed output and saved `manifest.json`/`epochs.json` for review before calibration. An epoch-zero best is unsuccessful adaptation. The input-quality findings, unresolved upstream exposure, night/IR/flank handling, calibration, reliable photographed-individual counts and park-readiness requirements remain open. No training or outreach was performed by the assistant.
