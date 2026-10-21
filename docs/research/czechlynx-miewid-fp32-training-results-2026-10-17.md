# CzechLynx completed FP32 adaptation review

Date: 17 October 2026. User-run trial: `models/czechlynx/miewid_adaptation_v2_fp32`. Saved status is **complete**, elapsed **2,397.940 seconds / 39.97 minutes**, with epochs 0–9 recorded. The assistant performed read-only artifact, metadata and CPU tensor-safe checkpoint inspection. No model was instantiated, no inference/training/test/benchmark was executed, and source/data/trial artifacts were not changed.

## Result and selection

The declared selector correctly chooses **epoch 6** by macro cross-encounter recall@50, then macro top-1, then earliest epoch. Epochs 7–9 did not improve that selector; the run stopped at the declared patience of three. Epoch 9 has marginally higher top-1 but lower recall@50, so it does not replace epoch 6.

| Inner-holdout metric | Frozen epoch 0 | Selected epoch 6 | Change |
| --- | ---: | ---: | ---: |
| Macro recall@50 | 92.6351% | 95.3157% | +2.6806 percentage points |
| Macro top-1 | 65.4234% | 71.3711% | +5.9477 percentage points |
| Unresolved images | 0 | 0 | 0 |

These are identity-averaged results on the 2,460-image / 34-identity fitting-pool inner holdout used to select the checkpoint. They establish a successful adaptation under this local experiment, not independent final-test/park accuracy or reliable photographed-individual counts. All validation images remain eligible. Epoch 1's full metrics exactly equal epoch zero, consistent with head-only warmup.

| Epoch | Macro recall@50 | Macro top-1 | Epoch seconds |
| --- | ---: | ---: | ---: |
| 0 | 92.6351% | 65.4234% | 17.697 |
| 1 | 92.6351% | 65.4234% | 92.079 |
| 2 | 94.1746% | 70.6520% | 282.662 |
| 3 | 94.5947% | 71.4168% | 275.749 |
| 4 | 95.1810% | 71.1937% | 279.186 |
| 5 | 94.8786% | 70.7882% | 282.093 |
| 6 | 95.3157% | 71.3711% | 289.069 |
| 7 | 95.2914% | 70.9468% | 281.956 |
| 8 | 95.2040% | 71.0512% | 276.678 |
| 9 | 95.2143% | 71.4622% | 269.112 |

Every training epoch contains 14,048 balanced sampled observations (1,756 batches), with finite mean losses. Sampling is not guaranteed once-per-image coverage. CUDA device: NVIDIA GeForce RTX 4070 Ti SUPER, CUDA 12.6. Recorded training parameters/inputs are FP32 with autocast and gradient scaling disabled. Evaluation retains FP16 CUDA autocast and FP32 normalization/matrices. No numerical failure is recorded. This successful FP32 run does not prove the exact cause of the earlier FP16 failure.

## Artifact and checkpoint review

| Binding/artifact | Value |
| --- | --- |
| Run fingerprint | `675d43e66317e459cf7d0672789ed19d4a3db1d91f1e606dc7b8233b3e8cd22d` |
| Completed manifest SHA256 | `efd9d8f3ccfc0a8db2379c6d092df45b91322c19063025fb7d0493dc68e15cf1` |
| Selected `best.pt` SHA256 | `c1e7792f0aeee7cbf0b9fa6a0de7f10a1f36770e6db45acb90397e803cbba32f` |
| `last.pt` SHA256 | `576de6a20e6cc7038b99a575e73b76bc0704cfa91661ad70f9d62fba4a0cf3cb` |
| Selected native-state SHA256 | `f0bf17cc7214b86ce253e4f5d1fc61e10082ccd1aefaf45c8cc0ef4f8b904c12` |

The existing completed-trial validator passes: reconstructed split, current implementation/package environment, protocol/metadata/review bindings, all fitting image/mask bytes, review receipt and all six saved artifact hashes match. The original inner plan is retained. Recomputed checkpoint selection and saved metrics agree; the latest epoch-9 query rows reproduce its macro metrics and exact holdout membership.

CPU `weights_only=True` inspection validates both checkpoints' provenance, native-state digests and exact metric/epoch binding. Native keys, shapes and dtypes match the preserved epoch-zero checkpoint from failed v1, whose state digest equals this run's initial state and the frozen calibration model's weight digest. Each checkpoint contains 1,210 native tensors, 698 changed from initialization; all floating native weights and the separate 131 × 2,152 head are finite FP32. All 510 batch-normalization running-stat/count buffers and two unused final-layer tensors remain exactly unchanged. The actual native update proof records `backbone.global_pool.p` at epoch 2 / batch 1, finite total gradient norm 9.6340417862.

All 22 protected hashes still match, including every failed-v1 artifact, original audit receipt/reviews, baseline implementation and unrelated tracked modifications. Failed v1 is preserved. The original baseline calibration/diagnostic artifact contracts and saved frozen-cache SHA256 pass read-only checks; the initial native model, 6,352 calibration images / 55 identities, 25 profiles and original threshold/budget controls match. Saved baseline ranking reconstruction is checked by the user-run evaluator before extraction, not rerun during this review.

Machine-readable [review receipt](czechlynx-miewid-fp32-training-results-2026-10-17.json), SHA256 `cf1a1d9cdcb25de466a8a1d7de810ccf5c7557e73b61e4222f437156bad82033`.

## Diagnostics and limits

Inner-holdout truth-filtered unthresholded k50 connectivity improves from 12/34 to 19/34 fully connected identities, with fragments 70 → 58 and images without a correct neighbour 20 → 14. At k15, fully connected identities improve 1 → 6 and fragments 159 → 128. These oracle graphs remove wrong links using reference truth; they are not deployable groupings/counts, and residual disconnection remains.

The other-camera-only-positive subgroup's macro top-1 improves 58.5722% → 68.7550% across 156 queries / 28 identities. This is reference-availability diagnostics, not a camera-disjoint validation protocol. The opposite-colour-proxy-only subgroup remains seven queries / two identities, top-1 50% and recall@50 100%; it does not establish day/night/IR or opposite-flank robustness. Upstream pretrained-model exposure, original input-quality limitations and count/review tolerances remain unresolved.

## Next action

The completed/improved trial is eligible for the already implemented [full calibration comparison](../czechlynx-benchmark.md#5-full-calibration-comparison). The adapted cache and report paths below were confirmed absent:

- `data/czechlynx/cache/miewid_adaptation_v2_fp32`
- `evaluations/czechlynx/miewid_adaptation_calibration_v2_fp32`

The user runs that command. It reuses immutable frozen baseline vectors and compares recognition, pair evidence, connectivity and all 25 grouping/count controls on all 6,352 eligible calibration photographs. No selection evaluation/configuration export or local-feature fusion is included. Review its actual false merges, fragmentation, exact recovery and count errors before interpreting the count milestone or specifying the next local/global/grouping scope. The user's combined thesis-method direction remains in the working diary; no park outreach or readiness claim follows from this training result alone.
