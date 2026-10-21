# Segmented MiewID full-pipeline results

Reviewed 13 September 2026. **Keep frozen MiewID global retrieval as the top-1 baseline.** The current Fisher and geometric scoring rules do not improve top-1 on this development sample. Fisher does add useful candidates, while the clustering diagnostic remains unsuitable for identifying individuals.

## Controlled comparison

The completed run has the same masked metadata, 750 gallery images, 560 queries, 303 known queries, 257 unknown queries and global-model specification as the earlier MiewID run. Saved global prediction records are identical, including rankings, scores and acceptance decisions. The ensemble loads the original segmented PCA128/GMM256 models for DISK, SuperPoint and ALIKED. This is a development comparison with unresolved upstream training overlap.

| Stage | Correct top-1 / 303 | Top-1 | Top-5 | mAP@50 |
| --- | ---: | ---: | ---: | ---: |
| MiewID global | 156 | 51.49% | 62.05% | 0.1722 |
| MiewID + Fisher | 155 | 51.16% | 64.36% | 0.1710 |
| Full funnel | 153 | 50.50% | 63.37% | 0.1639 |

The full pipeline is three correct queries below the global baseline. This small net difference does not establish a general ranking between methods across future datasets; individual changes reveal both useful evidence and harmful overrides.

| Transition | Incorrect to correct | Correct to incorrect | Net correct |
| --- | ---: | ---: | ---: |
| Global → global+Fisher | 13 | 14 | −1 |
| Global+Fisher → full funnel | 16 | 18 | −2 |
| Global → full funnel | 21 | 24 | −3 |

These are paired top-1 correctness changes. Additional queries change to a different wrong identity, so the totals do not represent every altered ranking. Query IDs and identity counts are retained in the [analysis receipt](czechlynx-miewid-full-pipeline-results-2026-09-13.json).

## Candidate generation is useful

| Correct identity appears within | Global | Global+Fisher | Full funnel |
| --- | ---: | ---: | ---: |
| First 10 returned candidates | 202/303 | 209/303 | 209/303 |
| First 20 | 223/303 | 228/303 | 228/303 |
| First 50 | 253/303 | 260/303 | 260/303 |
| Entire candidate shortlist | 253/303 | 272/303 | 272/303 |

Global uses 50 shortlist candidates. The fused shortlist is the union of each model's top 50 and can contain up to 100 images, so 272/303 is **89.77% recall at a potentially larger candidate budget**, not recall@50. At an equal returned depth of 50, Fisher improves candidate availability from 83.50% to 85.81%.

Fisher adds a correct candidate for 19 queries whose global top-50 lacked one. Only two of those newly covered queries place a correct candidate in the fused top 10, and only one becomes a correct full-funnel top-1. The current scorer therefore leaves much of the additional coverage unused.

Across all known queries, 51 have a true match at fused ranks 11–50, and another 12 have a match in the larger union but outside the returned top 50. Geometric verification only reorders the leading 10 candidates; it cannot promote these 63 queries to correct top-1 under this configuration.

This supports retaining Fisher as an experimental candidate source. It does not establish that equal weighting is the right scoring rule or that a wider geometric pass alone will improve accuracy.

## Geometric evidence and overrides

All **5,600 requested retrieval pairs were attempted**, with zero skipped pairs. The slight regression is not explained by geometric verification silently failing to execute.

The [current retrieval scorer](../../reid_demo/benchmark.py) averages clipped global and Fisher cosine scores. It then reorders the leading candidates using **−inliers × log(1 − fused score)**. These are fixed scoring rules; the development rejection threshold does not calibrate the relative reliability of the component signals.

Among the attempted pairs, same-identity pairs have median seven inliers (822 pairs); different-identity pairs have median five (4,778 pairs). Different animals can therefore receive nonzero geometric support. These are selected shortlist pairs, not a random population sample or a calibrated probability estimate.

Of the 18 correct matches broken by GV, 12 are replaced by a wrong candidate with at most ten inliers. Four previously correct top candidates have zero inliers. The evidence supports investigating when weak geometry should be allowed to override global similarity; it does not establish a safe inlier cutoff.

Two examples from the saved records:

- Query **clx_31cd0681fbc50486b3d0c228**, lynx_311: GV promotes the correct match from rank two, with 68 inliers versus six for the previously leading wrong animal.
- Query **clx_1ea61b05178362a88b9e283e**, lynx_232: the correct candidate has 11 inliers, but a wrong lynx_031 candidate has 28 and becomes top-1.

Thus strong inlier counts can help, while raw count alone is insufficient to guarantee identity. No causal claim about the exact image texture or lighting is made from these records alone.

## Appearance and camera checks

The table reuses the previously saved foreground chroma categories. Near-monochrome is an appearance proxy, not verified night/infrared metadata.

| Known-query subgroup | Queries | Global correct | Global+Fisher correct | Full funnel correct |
| --- | ---: | ---: | ---: | ---: |
| Near-monochrome query | 97 | 49 | 51 | 47 |
| Colour query | 206 | 107 | 104 | 106 |
| Correct gallery references only in opposite colour-proxy category | 27 | 5 | 5 | 5 |
| Correct references only at other recorded cameras | 103 | 47 | 43 | 45 |

Cross-appearance correctness remains **5/27 (18.52%)**, with every success belonging to lynx_203 in each stage. There is no demonstrated improvement in the targeted illumination-transfer weakness. Camera comparisons use the existing (source, trap_id) key and do not establish independence from upstream training cameras.

## Acceptance precision and coverage

All thresholds were calibrated and assessed on this development run.

| Stage | Accepted | Correct accepts | Wrong known accepts | Unknown accepts | Accepted precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global | 118 | 97 | 9 | 12 | 82.20% |
| Global+Fisher | 111 | 90 | 9 | 12 | 81.08% |
| Full funnel | 106 | 90 | 4 | 12 | 84.91% |

The full funnel increases in-sample accepted precision while correctly accepting seven fewer queries than global. Correctly accepted known-query coverage falls from 32.01% to 29.70%. None of these figures establishes the proposed 95% accepted-link precision on an independent evaluation.

## Clustering remains the major blocker

The actual backend is **NetworkX Louvain**, the available fallback for requested Leiden. All 560 query images are assigned to clusters; none remains unresolved.

| Stage | Clusters found / 60 true individuals | Within-cluster pair precision | Pair recall | Adjusted Rand index |
| --- | ---: | ---: | ---: | ---: |
| Global | 6 / 60 | 5.10% | 42.48% | 0.0486 |
| Global+Fisher | 8 / 60 | 7.43% | 41.79% | 0.0873 |
| Full funnel | 8 / 60 | 7.42% | 39.14% | 0.0863 |

Pair precision means the fraction of all image pairs placed together that belong to the same animal. It is different from retrieval accuracy. Every full-funnel cluster mixes identities: each contains 11–29 animals. The largest has 101 images from 29 individuals.

The [clustering implementation](../../reid_demo/cluster.py) constructs a symmetric graph from each image's 15 nearest neighbours, without a minimum identity-evidence threshold, and partitions it into communities. The benchmark supplies uncalibrated similarities and no flank gating. This structure can connect weakly related images and provides no explicit unresolved-image decision in this run. Changing the community-detection library or requesting more clusters is not established as a solution.

Only 210 clustering pairs were geometrically refined, with no skipped or budget-capped pairs. The 2,000-pair budget was not exhausted; raising that budget alone would not alter this run's selected pairs.

These clusters should not be presented as individual catalogues or population counts.

## Recommended next implementation

1. **Complete the phase 2 split and diagnostic foundation.** Protect fitting, calibration and model-selection data within official train; preserve the current run as a historical diagnostic and keep official test reserved.
2. **Measure calibrated score fusion.** Retain MiewID-only as the control, preserve per-signal evidence, and compare a calibrated rule against the current equal-weight/inlier formula. Evaluate wider candidate verification as a separate controlled choice, rather than assuming that more pairs help.
3. **Implement conservative clustering after calibration.** Require sufficient pair evidence before adding identity links, permit unresolved images, and evaluate merging errors alongside coverage. A model should not need to place every photograph into a confident identity group.
4. **Keep illumination experiments and supervised fine-tuning in the plan.** The cross-appearance subgroup remains weak, and the present result does not resolve it.

These are proposed implementation steps, not completed code changes. Concrete changes and tests must be presented for confirmation before implementation. Test and benchmark execution remains with the user.

## Evidence and status

- [Completed benchmark report](../../evaluations/czechlynx/dev_masked_miewid_ensamble/index.html).
- [Saved retrieval metrics](../../evaluations/czechlynx/dev_masked_miewid_ensamble/metrics.json) and [clustering metrics](../../evaluations/czechlynx/dev_masked_miewid_ensamble/clustering_metrics.json).
- [Analysis receipt: input hashes, transitions, subgroups and cluster composition](czechlynx-miewid-full-pipeline-results-2026-09-13.json).
- [Tracked implementation plan](../czechlynx-implementation-plan.md) and [upstream training-provenance audit](miewid-czechlynx-training-provenance-2026-09-13.md).

This review reads saved predictions, CSV metadata and source code. No tests, model inference, training or benchmark runs were executed. Focused regression-test results remain unreported.
