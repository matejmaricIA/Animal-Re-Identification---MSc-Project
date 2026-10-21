# CzechLynx calibrated edge grouping (Stage A) and local-evidence pilot (Stage B): results

Date: 18 October 2026. Scope: [approved proposal](../czechlynx-edge-grouping-proposal-2026-10-18.md). Calibration pool only (6,352 images, 55 animals), frozen FP32 v2 vectors. Reports:

- Stage A: `evaluations/czechlynx/edge_grouping_v2_fp32/`
- Pilot local evidence: `evaluations/czechlynx/edge_local_pilot_v2_fp32_r2/`
- Gate: `evaluations/czechlynx/edge_local_gate_v2_fp32/`

## Conclusion

**Grouping improved substantially over every saved control, but it is still far from reliable counting, and local features do not help.**

- **Best grouping:** pairwise F1 0.585 (precision 0.58, recall 0.59), vs 0.264 for the best of the 25 saved controls. Only **1/55** animals is recovered as one clean group. The best profile needs 87 corrections (38 merges + 49 splits).
- **Edge model:** cross-fitted AUC 0.87–0.89 on the full candidate graphs, just under the 0.90 rule. Raw cosine alone is near chance within these graphs (0.48–0.64). Ranks, shared neighbours and metadata links carry most of the signal.
- **Local evidence (DISK+LightGlue):** adds about **0.000 AUC** (gate requires +0.02). Its weak pooled signal (AUC about 0.60) depends on which animal is involved, and it inverts on unseen animals (identity-cross-fitted local-only AUC 0.35). Per the proposal, this is a documented negative for the thesis local/global combination on this data. The full Stage B run is **not** recommended.
- **Day-level encounter links are deployable on CzechLynx.** Same trap + same day joins different animals in 0.92% of linked pairs (vs 0.81% for annotated encounters). It reaches **46/55** oracle connectivity at k=150, vs 44/55 for annotated encounters.

None of these results is a count. The selection pool stays reserved and no configuration is exported.

## Runs and checks

- Implementation: three new modules/CLIs plus tests (20 Stage A, 9 Stage B). 144 tests pass across the related suites; Ruff F is clean.
- **Stage A:** `complete` in 253 s (CPU). No models loaded; cache hashes unchanged.
- **Stage B pilot:** 20,000 label-free stratified edges over 6,298 images, 607 s on an RTX 4070 Ti SUPER (about 33 edges/s). 19,429 verified, 507 with too few matches, 64 geometry failures, 0 matcher errors.
- Run history, for the record:
  - A first Stage A run was stopped by the assistant after a code edit (its manifest held only `running`) and removed. It was rerun on the final code.
  - The first pilot run (`edge_local_pilot_v2_fp32`) failed at about 14k edges because of a mask-cache eviction bug (`KeyError`). It is preserved; the fix and a regression test were added, and `_r2` was run from scratch.

## Metadata link rules (oracle, truth used only to evaluate)

| Rule | Linked pairs | Different-animal share | Oracle connected (k=100 → with links) | (k=150 → with links) |
| --- | ---: | ---: | --- | --- |
| Annotated encounter | 22,752 | 0.81% | 23 → 37 | 33 → 44 |
| Same trap + day | 68,448 | 0.92% | 23 → **43** | 33 → **46** |
| Same location + day | 42,042 | 1.54% | 23 → 40 | 33 → 44 |

## Edge models (identity-cross-fitted, full candidate graphs)

| Graph | Edges | Same-animal share | Cosine-only AUC | Stage A AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| annotated, k=100 | 435,797 | 43% | 0.643 | 0.876 | 0.135 |
| annotated, k=150 | 623,999 | 37% | 0.639 | 0.855 | 0.139 |
| derived, k=100 | 466,501 | 47% | 0.478 | **0.888** | 0.128 |
| derived, k=150 | 652,481 | 40% | 0.515 | 0.868 | 0.134 |

The "ambiguous band" (cosine-only p between 0.2 and 0.8) turned out to cover most edges, because cosine alone barely separates candidates. Band AUCs are therefore close to the overall AUCs.

## Grouping (48 profiles; best by pairwise F1)

| Arm, k | Profile | Groups (multi-image) | Precision | Recall | Exact /55 | Merges | Splits |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| derived, 100 | G2 average-linkage τ=0.5, s=1 | 235 (134) | 0.584 | 0.586 | 1 | 38 | 49 |
| derived, 150 | G3 Louvain, res 1.0 | 99 (58) | 0.621 | 0.535 | 0 | 29 | 52 |
| derived, 100 | G3 Louvain, res 2.0 | 125 (80) | 0.749 | 0.412 | 0 | 35 | 53 |
| annotated, 150 | G3 Louvain, res 1.0 | 111 (52) | 0.580 | 0.484 | 0 | 27 | 51 |
| best saved control | legacy k=15 Louvain | 26 | 0.266 | 0.263 | 0 | 25 | 54 |

- **G1 threshold** chains almost everything together at every τ (precision about 0.07), as the old controls did. With 450k–650k edges, even a small share of confidently wrong edges bridges animals.
- **Matching counts can hide cancelling errors.** Louvain on derived k=150 gives 58 multi-image groups for 55 animals, a near-correct-looking count produced by 29 merges and 52 splits. This is exactly why count error alone is not an acceptance metric.

## Local evidence pilot (gate)

| Graph | Pilot edges | Stage A AUC | + local AUC | Δ | Local-only AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| annotated, k=100 | 11,947 | 0.899 | 0.899 | +0.001 | 0.419 |
| annotated, k=150 | 16,885 | 0.881 | 0.882 | +0.000 | 0.410 |
| derived, k=100 | 15,066 | 0.925 | 0.925 | +0.000 | 0.351 |
| derived, k=150 | 19,953 | 0.912 | 0.912 | +0.000 | 0.354 |

Pooled raw signal is weak: match count AUC 0.59, homography inliers 0.60 (median 5 vs 5), affine inliers 0.60 (median 5 vs 4); on kNN-only edges, affine inliers reach 0.63. A follow-up check on the pilot edges confirmed the cause. A local-only logistic model reaches **0.60 AUC with random cross-validation but 0.35 with identity cross-fitting**. The local signal tracks which animal (and which images) are involved, not whether two photos show the same animal, so it does not transfer to unseen animals. The looser partial-affine model did not change this.

## Addendum (18 Oct, after the flank run): local evidence on same-flank pairs

About half of the pilot's same-animal pairs face opposite ways, and local matching cannot confirm those by design, which could have hidden a real effect. A read-only re-analysis of the saved pilot evidence, using the flank run's predicted facing, rules that out:

| Pilot subset | Edges | Affine-inlier AUC | Median inliers (same / different) | Cosine AUC |
| --- | ---: | ---: | --- | ---: |
| All | 20,000 | 0.597 | 5 / 4 | 0.477 |
| Opposite-facing | 7,509 | 0.585 | 5 / 4 | 0.347 |
| Same-facing | 12,491 | 0.608 | 5 / 4 | 0.559 |
| Same-facing, both confident | 7,744 | 0.601 | 6 / 5 | 0.636 |
| Same-facing, confident, kNN only | 6,692 | 0.625 | 6 / 5 | 0.812 |

Local evidence is about as "predictive" on opposite-facing pairs, where spot correspondence is impossible, as on same-facing pairs. Its weak signal therefore reflects shared image conditions (same camera, day or scale), not spot-pattern matching. The matcher finds only 5–6 geometric inliers even for same-side photos of the same lynx.

Identity-cross-fitted models on the 7,744-edge subset were unstable (cosine-only 0.500, local-only 0.397, combined 0.517) and are not relied on.

**Scope of this negative:** it covers generic DISK+LightGlue keypoints on the prepared masked crops, verified with a single homography or partial affine. It does not cover spot-specific matching on pose-rectified flank patches, SIFT/HotSpotter-style descriptors, non-rigid verification, or native-resolution crops.

## Deviations from the proposal

- **G3 uses networkx Louvain**, not Leiden. `leidenalg` is not installed, and installing it would change which backend the frozen legacy control resolves to. The proposal wrongly called Leiden an existing dependency.
- **G2 support** is `min(s, |A|·|B|)` strong edges. As literally written, s=3 would have prevented two single images from ever merging.
- **The derived arm links on trap-day or location-day**, because 769 calibration images lack a location.
- **The partial-affine threshold** was fixed at 0.03 normalized units before the pilot, looser than the legacy homography's 0.005.
- **The local evidence store** is a compact SQLite file of scalar features, not the per-pair JSON cache with point lists.

## What this means and possible next steps (not approved)

The coverage problem is largely solved by wider candidate lists plus day-level links (oracle 46/55). The binding constraint is now **pairwise evidence quality**: AUC about 0.87–0.89 on edges whose global cosine is near chance. Neither better grouping rules nor DISK/LightGlue fixed that here. Options worth weighing:

1. **Flank-aware comparison.** Opposite-flank pairs almost never link (2.6% vs 29%), so they mostly add noise. The raw metadata has left/right pose keypoints for 5,206 images, enough to train or validate a flank classifier. Comparing only same-flank photos, and joining flanks through encounters, could cut wrong edges sharply.
2. **A learned pair verifier trained on the fitting pool**, e.g. a cross-attention re-ranker over MiewID feature maps. This replaces handcrafted local matching, and it must be evaluated with identity cross-fitting as here.
3. **A measured human-review baseline.** The best profile needs 87 corrections for 6,352 photos / 55 animals. That review-effort figure is an alternative way to state "reliable after bounded review" and should be weighed against the count-error tolerance that has not been agreed yet.

Reliable photographed-individual counts are still unachieved.
