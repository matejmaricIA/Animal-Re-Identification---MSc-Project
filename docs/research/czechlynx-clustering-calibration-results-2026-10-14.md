# CzechLynx full-collection clustering calibration review

Date: 14 October 2026. Scope: review of the user-run calibration report in `evaluations/czechlynx/cluster_calibration_global_v1/`. [Machine-readable evidence receipt](czechlynx-clustering-calibration-results-2026-10-14.json).

## Decision

**None of the 25 tested global-only configurations supports reliable photographed-individual counts on this calibration collection.** Keep the selection pool untouched for clustering while diagnosing the measured failures. No configuration was exported and no selection command was started by the assistant. This does not close the required clustering/count milestone.

The next proposed step is a [calibration-only pair and connectivity diagnostic](../czechlynx-clustering-diagnostics-proposal-2026-10-14.md), before deciding whether to improve pair scoring, graph rules, candidate coverage or the embedding model. A new implementation requires approval under AGENTS.md.

## What completed and what was checked

- The user ran the approved standalone benchmark: all **6,352 eligible photographs from 55 reference identities**, including unknown-to-gallery individuals, and all 25 predefined profiles.
- Saved manifest status is `complete`; run fingerprint is `42525241c50df2815c3c0b22d80a2a25260d0ae8b15141a15f4104c18604098b`.
- The report and current implementation/source contracts match. All **595 recorded artifact hashes** checked successfully. Each profile's assignment membership, reference identities, component counts, mixed-group counts, fragmented-identity counts, singleton counts and predicted/correct pair arithmetic match its saved metrics.
- All 6,352 cached 2152-dimensional vectors were loaded through read-only SQLite. The user-run loader validated input bytes, vector checksums, shape and finite values; no IDs were missing. All profiles report zero unresolved/zero-vector observations.
- All appearance measurements were reused from the existing cache. There were zero new pixel measurements.
- The saved runtime is **39.497 seconds**: vector validation 3.832s, appearance evidence 0.617s, candidate retrieval 3.680s. Profile timings include clustering and metric calculation; total timing additionally includes report generation and other run work. Initial Python imports precede the recorded timer.
- The assistant's earlier 10–30 minute runtime estimate was too high. The observed run takes about 40 seconds under these conditions; future method changes and workloads may differ.
- Earlier focused tests were user-reported **62 passed, one upstream LightGlue warning, in 8.50s**. No tests, new clustering runs, model inference or training were executed by the assistant for this review.

The repeated scikit-learn warnings occur when predicted labels become numerous relative to the number of samples. They do not indicate a failed run or invalid component IDs. In this experiment they accompany severe fragmentation. No warning-suppression or source changes were made.

## Representative results

All rows refer to the same calibration collection of **55 photographed individuals**. Pair precision is the fraction of within-component image pairs belonging to the same reference identity. Pair recall is the fraction of true same-identity image pairs placed together. These are image-pair metrics and remain sensitive to unequal identity sizes and correlated photographs.

| Configuration | Provisional groups | Signed count error | Pair precision | Pair recall | Mixed groups | Fragmented identities |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing neighbour graph, k=15, NetworkX Louvain | 18 | -37 | 14.70% | 23.54% | 17 | 55/55 |
| Threshold graph, k=15, cosine >= .35 | 86 | +31 | 6.60% | 97.17% | 3 | 26/55 |
| Threshold graph, k=15, cosine >= .65 | 4,623 | +4,568 | 98.87% | 1.32% | 3 | 55/55 |
| Consistent merging, k=15, cosine >= .35 | 1,607 | +1,552 | 90.26% | 2.28% | 498 | 55/55 |
| Consistent merging, k=50, cosine >= .35 | 1,568 | +1,513 | 90.07% | 2.29% | 521 | 55/55 |

**Macro exact single-pure-group identity recovery is zero for every profile.** No reference identity has its entire eligible photograph set recovered as one pure component under any tested configuration. This strict metric does not mean that no correct image pairs or pure partial groups were recovered.

Among these controls the existing baseline has the highest ARI (0.1089) and pair F1 (0.1810). Its 18 groups still merge animals extensively. None of the new controls gives a joint clustering/count result that meets the intended outcome. Higher precision or a closer total alone cannot qualify a method.

### Why the 86-group result is misleading

At threshold .35, **one component contains 6,250 of the 6,352 photographs and all 55 reference identities**. There are 73 singleton proposals among the 86 groups. The apparently smaller count error (+31, 56.36%) conceals almost complete merging of the collection plus isolated fragments. Only three mixed components is not evidence of low merge severity: one mixed component can contain every individual.

### Why strict thresholds and consistent merging fail counting

At threshold .65, pair precision reaches 98.87%, but recall falls to 1.32%; 4,003 of the 4,623 groups are singletons. At threshold .75, precision is 100%, but recall is only 0.154% and the provisional count is 5,608. High purity comes with extreme fragmentation.

The all-cross-component consistency rule improves pair precision over the baseline at .35, but yields 1,607 groups at k=15, including 498 mixed groups and 442 singletons. Every identity is fragmented. Multiple small mixed groups can coexist with much higher pair precision than a few very large mixed groups; mixed-group totals cannot be compared without their sizes and pair errors.

Consistency at k=15/.35 yields cross-encounter precision/recall **89.58%/2.15%**, cross-camera **84.27%/1.52%**, and cross-appearance-proxy **90.34%/1.44%**. These low recalls expose fragmentation beyond same-encounter bursts. Appearance proxies are not actual day/night/IR annotations.

### Candidate-budget evidence

For every tested threshold-graph setting, k=15 and k=50 have **identical assignment-file hashes**, not merely equal headline metrics. Increasing the budget to 50 does not change these threshold components. Consistent merging changes only modestly at low thresholds; it does not recover usable counts.

This result does not establish that candidate recall is sufficient. Added candidates can be below the threshold, redundant within an already connected group, or unsuitable for the consistency rule. Per-identity candidate connectivity still needs direct diagnosis.

## Bounded visual review

Inspected eight saved thumbnails from four deterministic preselected pairs:

1. First false-merge example at threshold k=15/.35: `lynx_111` and `lynx_208`. Different poses and illumination are visible; occluding vegetation remains visible in one crop. The example represents a shared component; its endpoints need not be directly connected by an admitted edge.
2. First false-merge example at consistent k=15/.35: `lynx_117` and `lynx_264`. One crop is monochrome and one colour, with different poses.
3. First fragmentation example at consistent k=15/.35: two `lynx_002` photographs from different cameras. The views differ, including a rear/oblique view.
4. First cross-appearance-proxy example at consistent k=15/.35: two `lynx_013` photographs, one bright colour and one dark monochrome, with partial/occluded visible animal regions.

These examples make view, illumination, visibility and segmentation/occlusion quality relevant hypotheses for the next diagnostic. They do not isolate the cause of a model error, validate flank labels or establish night/IR robustness. Original frames and the complete per-identity collection were not visually audited.

## What to do next

1. **Retain this completed calibration report as the reference.** Do not select a profile merely because its group total is closest to 55. No tested profile is recommended for a count demonstration.
2. **Diagnose direct pair evidence and true candidate connectivity using calibration only.** The saved edge CSVs contain at most 5,000 highest-score decisions per profile, so they cannot alone explain all weak bridges. The proposed diagnostic reconstructs the full permitted candidate graph from immutable cached vectors and distinguishes correct edges, misleading links and missing true paths.
3. Use the results to choose the next method change. If direct similarities separate identities reasonably but weak edges propagate errors, prioritise calibrated pair evidence and graph merging rules. If views of the same animal lack usable candidate links, prioritise candidate diversity, view-aware evidence or model adaptation. A scalar calibration cannot create information absent from the embeddings.
4. After an improved method is defined and chosen using calibration, freeze its configuration and evaluate selection once. Further independent evaluation, count uncertainty, night/IR and flank evidence, and reviewer-effort measurement remain necessary for park readiness.

## Scientific limits

This is development calibration, not independent field validation. Historical eight-group/60-individual results use a different collection/protocol and must not be compared numerically as a like-for-like baseline. The phase 2 retrieval result remains a separate task. Upstream MiewID photograph/identity/inherited-training exposure is unresolved. All prepared flank labels remain unknown; encounters can contain multiple animals. No acceptable count-error or review-effort tolerance has been agreed. Counts in this experiment describe labelled individuals represented in a specified photographic collection, not total Risnjak abundance.

