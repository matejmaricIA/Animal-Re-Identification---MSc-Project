# Segmented MiewID development results

Reviewed 13 September 2026. **MiewID is the recommended frozen global baseline for the next phase.** It reaches the 50% known-query top-1 development milestone, with 156/303 correct. It has not yet established held-out pilot accuracy or useful clustering.

**Upstream training overlap remains unverified.** Earlier MiewID training explicitly included Eurasian lynx, but the public evidence does not establish whether this v3 checkpoint saw CzechLynx photographs or individuals. The inspected local weight file matches the publisher's October 2024 artifact; none of the dated images in this development run is newer than December 2021. The later publication of CzechLynx cannot establish independence because related private Czech source data existed earlier. See the [training-provenance audit](miewid-czechlynx-training-provenance-2026-09-13.md) for evidence and a proposed later-photograph evaluation. The reported 51.49% is a development benchmark result, not verified transfer to animals absent from upstream training.

## Verified comparison

The completed run uses the same 750 gallery and 560 query images as the segmented MegaDescriptor reference, including 303 known and 257 unknown queries. Gallery/query ordering, prepared metadata hash, input mode, stage, split and saved reference hashes agree. The model is frozen MiewID-msv3 at revision `4f1d7f2b521149e5fe34bb85f377248ce9971a7d`, using masked pixels and 2152-D embeddings.

The new run contains no Fisher fitting, geometric matching or clustering. All analysis below reads existing predictions and metadata; it does not rerun a model.

| Metric | Segmented MegaDescriptor global | Previous segmented full pipeline | Segmented MiewID global |
| --- | ---: | ---: | ---: |
| Top-1 | 20.13% (61/303) | 29.37% (89/303) | **51.49% (156/303)** |
| Top-5 | 35.31% | 39.60% | **62.05%** |
| mAP@50 | 0.0623 | 0.0760 | **0.1722** |

Against MegaDescriptor alone, MiewID fixes 106 known queries and worsens 11: a net gain of 95, or **31.35 percentage points**. Against the previous full segmented pipeline, it fixes 84 and worsens 17. Identity-averaged top-1 is 50.70%, so the improvement is also present when identities receive equal weight; this does not remove encounter correlation or replace a held-out test.

## Candidate availability

| Correct identity occurs within | MegaDescriptor global | MiewID global |
| --- | ---: | ---: |
| Top 10 | 135/303 (44.55%) | **202/303 (66.67%)** |
| Top 20 | 177/303 (58.42%) | **223/303 (73.60%)** |
| Top 50 | 235/303 (77.56%) | **253/303 (83.50%)** |

Of MiewID's 147 incorrect top-1 queries, 46 have the correct identity at ranks 2–10, 51 at ranks 11–50, and 50 have no correct identity in the top 50. Thus 97 errors have a correct candidate within reach of a 50-candidate reranker. An 83.5% perfect-reranking ceiling is available at that depth; it is not a forecast for geometric matching.

## Appearance and camera diagnostics

The colour categories reuse the existing foreground chroma diagnostic: the 90th percentile of `max(R,G,B) − min(R,G,B)` is at most five after resizing images/masks to 128px. **Near-monochrome is an appearance proxy, not a confirmed night/infrared label.**

| Known-query subgroup | Queries | MegaDescriptor global correct | Previous full pipeline correct | MiewID global correct |
| --- | ---: | ---: | ---: | ---: |
| Near-monochrome query | 97 | 27 (27.84%) | 26 (26.80%) | **49 (50.52%)** |
| Colour query | 206 | 34 (16.50%) | 63 (30.58%) | **107 (51.94%)** |
| Correct references only in the opposite colour-proxy category | 27 | 0 | 0 | **5 (18.52%)** |
| At least one correct reference in the same colour-proxy category | 276 | 61 (22.10%) | 89 (32.25%) | **151 (54.71%)** |
| Correct references only at other recorded cameras | 103 | 10 (9.71%) | 15 (14.56%) | **47 (45.63%)** |
| At least one correct reference at the same recorded camera | 200 | 51 (25.50%) | 74 (37.00%) | **109 (54.50%)** |

Monochrome and colour aggregate accuracy is now similar, but cross-appearance matching remains weak. All five correct cross-appearance queries belong to **lynx_203**, across three recorded encounters. Those successes do not establish broad visible/infrared robustness across animals.

The camera key is `(source, trap_id)`. The other-camera subgroup means that the correct identity has no gallery reference at the query's recorded camera. This is not a camera-disjoint training/evaluation experiment; it does not establish that the model never saw those cameras or identities in pretraining.

## Visual examples

[Contact sheet: five successes/errors](../../evaluations/czechlynx/dev_masked_miewid_global/error_analysis/cases.jpg)

Columns show the query, MegaDescriptor's first candidate, MiewID's first candidate, and a correct reference. The correct reference is the first true match in MiewID's saved ranking where available; otherwise it is a representative gallery image. Selection is illustrative, not a random error sample.

- **Cross-appearance success:** MiewID matches a monochrome lynx_203 image to a colour reference, despite differing crop and pose.
- **Correct candidate at rank two:** a low-contrast partial lynx_155 crop still favours a wrong monochrome animal over its colour reference. The true animal is already in the shortlist.
- **Correct identity absent from the top 50:** another lynx_155 crop has very little visible detail and a tight, irregular animal mask; MiewID returns another low-detail crop. This case suggests image quality and available foreground evidence matter alongside illumination.
- **Regression and confident error:** two lynx_083 queries select lynx_123 with a similar pose and colour, while correct references differ in pose. This is consistent with broad appearance outweighing the distinguishing coat pattern, but does not prove which feature caused the score.

These examples motivate detailed local matching and a quality/uncertainty audit. They do not justify inventing missing coat detail or assigning every difficult image an identity.

## Acceptance and clustering limits

At its development-calibrated threshold of approximately 0.5232, MiewID accepts 118 queries: **97 correct known matches, nine wrong known matches and 12 unknown animals incorrectly attached**. Accepted-match precision is **82.20%**, compared with 69.23% for the previous full segmented pipeline. Correctly accepted known-query coverage is 97/303, or 32.01%.

The unknown false-accept rate remains 12/257 (4.67%) by the in-sample calibration procedure. These numbers remain short of the proposed 95% accepted-link precision target and are not independently calibrated test results.

Clustering was disabled (`cluster_limit=0`, empty `clustering_metrics.json`). This run gives no evidence that MiewID clusters are reliable. The earlier graph problems still require explicit evaluation and correction.

## Next steps

1. **Retain this frozen MiewID run as the new baseline.** Preserve its inputs, checkpoint, predictions and caches.
2. **Implement phase 2 of the tracked plan:** protect fitting, calibration and selection partitions; formalize appearance/camera/quality diagnostics. The historical development sample remains a comparison anchor.
3. **Add calibrated local matching over a wider shortlist.** Compare independent DISK/ALIKED LightGlue scores with the MiewID-only baseline, initially using up to 50 candidates. The 97 recoverable shortlist errors provide a concrete reason for this experiment. Adding the old Fisher/power-scoring combination is not assumed to improve the result.
4. **Build conservative clustering from validated pair evidence.** Permit weakly supported images to remain unresolved, and evaluate precision, recall and coverage together.
5. **Fine-tune after that reference is established**, emphasizing cross-camera and cross-appearance pairs if those failures persist.

The user-run benchmark confirms model loading and the real-data path. Focused test results have not been reported and must not be marked passed. No implementation beyond phase 1 is inferred from this result.

## Evidence

- [MiewID benchmark](../../evaluations/czechlynx/dev_masked_miewid_global/index.html) and [saved global comparison](../../evaluations/czechlynx/dev_masked_miewid_global/global_model_comparison.json).
- [MegaDescriptor reference](../../evaluations/czechlynx/dev_masked_ensamble/index.html).
- [Analysis counts, source hashes and cross-appearance case IDs](czechlynx-miewid-results-2026-09-13.json).
- [Per-image appearance diagnostics](czechlynx-lighting-diagnostics-2026-09-12.csv).
- [Illustrated error review](../../evaluations/czechlynx/dev_masked_miewid_global/error_analysis/index.html).
- [Tracked implementation plan](../czechlynx-implementation-plan.md).
