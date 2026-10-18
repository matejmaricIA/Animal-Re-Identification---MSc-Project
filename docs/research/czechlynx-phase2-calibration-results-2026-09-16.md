# Phase 2 segmented MiewID calibration review — 16 September 2026

User-run report: `evaluations/czechlynx/phase2_calibration_miewid_global/index.html`.
The assistant inspected saved artifacts and predictions only; no tests or inference were rerun.

## Artifact review

The manifest is complete and identifies development/calibration. Manifest and threshold protocol fingerprints match the frozen plan:
`e3401b266c0e9af162ecd4d16530c0e9f6d5d7f623920c2e24a6af18c04a5fa5`.
Ordered gallery/query IDs match the plan and thresholds; predictions cover all 1,830 queries exactly once. Current ranking implementation hashes match the calibration request.

Frozen MiewID-msv3 uses the pinned revision, segmented inputs and 2,152-D embeddings. Gallery: 4,522 images. Queries: 1,336 known across 44 identities and 494 unknown across 11 identities. No Fisher, geometric matching or clustering ran.

## Retrieval and rejection

| Measure | Result |
| --- | --- |
| Top-1, known queries | 877/1,336 = 65.64% |
| Top-5, known queries | 1,102/1,336 = 82.49% |
| mAP@50 | 0.3101 |
| Recall@50 | 94.84% |
| Mean per-identity top-1 | 56.09% |
| Mean per-encounter top-1 | 72.00% |
| Rejection threshold | 0.5372944474220277 |
| Accepted precision | 480/534 = 89.89% |
| Correct known-query coverage | 480/1,336 = 35.93% |
| Unknown false accepts | 24/494 = 4.86% |
| Incorrect accepted known matches | 30 |

The threshold was chosen on these same calibration queries with a 5% target unknown false-accept rate. This is not a 5% total matching-error guarantee: 54/534 accepted assignments are wrong, including 30 incorrect matches between known animals and 24 accepted unknown animals. Selection must load this threshold unchanged.

The new pool has different identities, gallery support and query membership from the historical 51.49% experiment. The difference is not a controlled model improvement. Image-weighted and identity-weighted accuracy also differ substantially.

## Appearance and camera diagnostics

- Colour queries: 67.82% top-1, 665 known queries.
- Near-monochrome queries: 63.49%, 671 known queries.
- Correct gallery references only in the opposite appearance category: 0/8 top-1, from three identities and four encounters; recall@50 is 4/8 and none are accepted. This tiny subset signals an unresolved failure case, not a precise estimate of day/night accuracy.
- Correct gallery references only at other cameras: 159/295 = 53.90%.
- Same-camera correct reference available: 718/1,041 = 68.97%.

Near-monochrome is a foreground colour proxy, not verified night/infrared. Camera categories describe reference availability and are not controlled causal measurements of camera bias.

Unknown-image counts are uneven. Mean per-unknown-identity false-accept rate is 7.68%, versus the image-weighted 4.86%. For example, lynx_036 has 5/12 false accepts; lynx_086 contributes 286 unknown queries with 11 false accepts. Keep denominators and identity-level errors visible when comparing to selection, whose unknown images are heavily concentrated on lynx_287.

## Next action

Run the documented selection command with the calibration `thresholds.json`, same model and settings, and a separate output directory. Do not change ranking code or tune thresholds from these results before that run. Compare recognition, accepted precision/coverage, unknown errors by identity, and camera/appearance failure groups. Selection remains development validation; upstream MiewID exposure to CzechLynx identities is unresolved. Clustering is still unvalidated.
