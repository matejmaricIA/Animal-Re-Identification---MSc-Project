# Phase 2 segmented MiewID selection review — 16 September 2026

User-run report: `evaluations/czechlynx/phase2_selection_miewid_global/index.html`.
The assistant reviewed saved artifacts and predictions only. No tests, extraction or benchmark were rerun.

## Frozen validation and scope

Manifest status is complete, with development/selection role. The protocol fingerprint is
`e3401b266c0e9af162ecd4d16530c0e9f6d5d7f623920c2e24a6af18c04a5fa5`.
Gallery and query IDs exactly match the plan; all 2,336 query predictions appear once. The global model specification and ranking request match calibration. The threshold artifact hash matches both selection provenance and the earlier calibration review receipt; no new selection threshold file exists. Current ranking implementation hashes still match calibration.

Selection uses 2,389 gallery images, 881 known queries across 42 identities and 1,455 unknown queries across 11 identities. Two other known identities are gallery-only. All selection identities are distinct from fitting and calibration within this protocol. Upstream MiewID exposure remains unresolved; this is development selection validation, not final untouched or new-park evaluation.

## Calibration versus selection

| Measure | Calibration | Selection |
| --- | --- | --- |
| Known-query top-1 | 877/1,336 = 65.64% | 562/881 = 63.79% |
| Known-query top-5 | 82.49% | 699/881 = 79.34% |
| mAP@50 | 0.3101 | 0.3155 |
| Correct identity in top 50 | 94.84% | 822/881 = 93.30% |
| Mean per-identity top-1 | 56.09% | 47.28% |
| Mean per-encounter top-1 | 72.00% | 70.36% |
| Accepted precision | 480/534 = 89.89% | 325/383 = 84.86% |
| Correct-known coverage | 35.93% | 325/881 = 36.89% |
| Unknown false accepts | 24/494 = 4.86% | 46/1,455 = 3.16% |

The unchanged confidence threshold is 0.5372944474220277. Selection accepts 337 known queries, of which 325 are correct and 12 are incorrect. It additionally accepts 46 unknown queries incorrectly, producing 58 wrong assignments among 383 accepted queries. A low unknown false-accept rate does not imply high precision among all accepted assignments. Acceptance precision also depends on how many unknowns appear: selection has a much higher unknown-query share than calibration. The two pools contain different animals and galleries, so these differences do not isolate a causal model change.

The image-weighted 50% recognition target is exceeded on this selection pool. The equal-weight-per-animal average is below 50%; five of 42 known query identities have no correct top-1 match, several with very few queries. More photographed animals contribute disproportionately to the headline result.

## Unknown-identity imbalance

Mean per-unknown-identity false-accept rate is **17.77%**, versus the image-weighted 3.16%. The identity-level denominators are important:

| Identity | Unknown queries | False accepts |
| --- | --- | --- |
| lynx_007 | 44 | 0 |
| lynx_020 | 4 | 2 |
| lynx_023 | 74 | 2 |
| lynx_028 | 5 | 3 |
| lynx_049 | 27 | 1 |
| lynx_102 | 7 | 4 |
| lynx_273 | 127 | 1 |
| lynx_281 | 11 | 2 |
| lynx_287 | 1,062 | 31 |
| lynx_290 | 85 | 0 |
| lynx_299 | 9 | 0 |

Removing lynx_287 for a descriptive sensitivity calculation leaves 15/393 false accepts (3.82%); it does not eliminate the discrepancy with the per-identity average. Several rarely photographed unknown animals have high observed error rates, but their small and correlated samples are imprecise. Do not change pool membership, discard those animals or tune the threshold on selection.

## Appearance, camera and source

- Colour known queries: 378/560 = 67.50% top-1.
- Near-monochrome known queries: 184/321 = 57.32%.
- Only opposite-appearance correct references: 1/5 top-1, all five from one identity and two encounters. None pass the acceptance threshold. This is too little independent evidence to establish lighting robustness.
- Only other-camera correct references: 76/155 = 49.03%; recall@50 is 84.52%.
- Same-camera correct reference available: 486/726 = 66.94%.

These are descriptive groups, not controlled experiments. Near-monochrome is a foreground colour proxy and is not a verified night/infrared label.

Source-level acceptance is uneven: FoE Carpathians has 47/83 accepted assignments correct (56.63%), on a query mixture of 185 known and 1,368 unknown images. All 36 errors there are accepted unknowns. SNPA has 245/265 accepted assignments correct (92.45%), but accepts 10/43 unknown images (23.26%). Source rates mix identity composition, reference coverage and unknown prevalence, so they do not establish a geographic cause.

## Decision and next phase

The phase 2 global-only reference is complete: user-reported focused tests passed (106 tests), protocol creation is audited, and both calibration and frozen selection reports have been reviewed. This completes the evaluation foundation, not pilot readiness or every future pipeline comparison.

Keep frozen segmented MiewID as the baseline. High recall@50 (93.30%) suggests room to improve ranking and rejection within the shortlist; it does not prove local matching will supply reliable evidence. Next is the phase 3 concrete implementation proposal for calibrated fusion/local matching, learning on permitted development data and evaluating frozen decisions on this same selection pool. An existing Fisher/GV comparison on the new pools remains useful as a control; historical gallery-fitted Fisher models cannot substitute.

Track image- and identity-weighted accuracy, accepted precision at useful coverage, unknown errors per identity, and appearance/camera groups together. Reliable clustering remains unvalidated and was disabled in this run. Present the next code changes for approval before implementation. Preserve the official test split and the current results.
