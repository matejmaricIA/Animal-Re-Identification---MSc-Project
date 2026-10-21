# CzechLynx MiewID training preparation review

Date: 16 October 2026. The user ran the existing trainer with `--prepare-only` after completing the human input audit. The command succeeded without neural imports or GPU allocation. The assistant inspected the saved artifacts and reconstructed membership using metadata only; no tests, model inference, training or performance benchmarks were run during this review.

## Saved trial and bindings

Trial directory: `models/czechlynx/miewid_adaptation_v1`. At review it contains only `manifest.json` and `inner_plan.json`, with status `prepared`.

- Run fingerprint: `8abf59ee36939497c5c4996c1bebd61bd0356064161f9b524c4f28694e9ff1af`.
- Inner-plan fingerprint: `ef8d64056a89ef7bce2e37db9d089bd86ce92ff12b922f9f4acbd951db0c88e7`.
- Prepared manifest SHA256: `756964488c38ff666bb088de0b387c9e96576ff542637bfc960dbe264bc05907`. Training will change this manifest's status and contents.
- Inner-plan SHA256: `9351779bc068cdec598f1383d65a7876e5731b900a6a09496603463b1481d316`.
- Review receipt SHA256: `ffa4000e8cccc219dc3e53da82df2714263b12f3e878d448416f2031c63c93f3`.
- Outer protocol fingerprint: `e3401b266c0e9af162ecd4d16530c0e9f6d5d7f623920c2e24a6af18c04a5fa5`.

The saved plan exactly equals reconstruction with seed 42 and holdout fraction 0.2. Manifest/run fingerprints, current implementation and package environment, metadata/protocol bindings, audit receipt and imported reviews, and the fitting metadata input digest match. The user's prepare-only command also verified all fitting image/mask bytes. No new neural imports occurred in the assistant's inspection.

## Inner split

| Role | Images | Identities | Atomic groups | Encounters | Bohemia / Carpathians / SNPA images |
| --- | ---: | ---: | ---: | ---: | --- |
| Training | 14,044 | 131 | 123 | 6,348 | 3,023 / 6,578 / 4,443 |
| Validation | 2,460 | 34 | 33 | 1,297 | 515 / 707 / 1,238 |
| Total | 16,504 | 165 | 156 | 7,645 | 3,538 / 7,285 / 5,681 |

All original fitting images are retained. Whole linked groups remain together: four multi-identity groups are in training (sizes 2, 2, 2 and 6), one is in validation (size 2). Linked groups constrain splitting; they do not assert that different identity labels denote the same animal.

Cross-role overlap is zero for identities, encounters, source photograph hashes and prepared image hashes. All rows remain official `time_open` train. No outer calibration, outer selection or official-test image is included. This establishes separation within the local experiment; upstream MiewID pretraining exposure remains unresolved.

The holdout is 34/165 identities (20.6%) and 2,460/16,504 images (14.9%). The split targets identity/source representation with whole groups; it is not an exact 20% image split. All three source regions are present in both roles. Source image proportions differ, so aggregate holdout scores should not be interpreted as uniform performance across regions.

A metadata-only eligibility check using the evaluator's encounter/source/prepared-hash exclusions found a legal same-identity positive for **all 2,460 validation images across all 34 identities**. This establishes available evaluation comparisons, not recognizability or correct model retrieval. No metadata-ineligible validation images were found.

## Training handoff

Proceed with the [existing user-run GPU training command](../czechlynx-benchmark.md#4-user-run-gpu-training), using the same directory, inputs and settings and removing only `--prepare-only`.

Declared workload: at most ten epochs, each with 1,756 balanced batches of four identities x two observations. That is at most 17,560 training batches and 14,048 sampled observations per epoch, with repeated sampling rather than guaranteed once-per-image coverage. The first epoch trains the separate classification head only; embedding adaptation starts at epoch 2. Early stopping uses the existing patience of three after warmup. Frozen epoch zero is eligible for checkpoint selection.

The run first evaluates the frozen model on the holdout. Select the best checkpoint by macro cross-encounter recall@50, then macro top-1, then earlier epoch. Epoch-2 timing is more representative of embedding training than the head-only warmup; no measured real-training runtime is available yet. A started run cannot resume under this implementation, and failed/started/completed output cannot be reused as a fresh trial.

Share the training result, `manifest.json` and `epochs.json` before requesting the full calibration comparison. If epoch zero remains best, record unsuccessful adaptation and reconsider before calibration. Better holdout recognition alone does not establish improved clustering/counts.

The [completed input audit](czechlynx-input-audit-review-results-2026-10-16.md) remains a documented limitation: original images, quality issues and all membership are retained for the approved first representation comparison. No correction, exclusion, new recipe or selection evaluation was introduced. This handoff does not close the actual-training, calibration, reliable-count or park-readiness milestones.
