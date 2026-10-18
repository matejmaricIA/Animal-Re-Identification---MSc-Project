# CzechLynx flank-aware grouping: results

Date: 18 October 2026. Scope: [approved proposal](../czechlynx-flank-grouping-proposal-2026-10-18.md). Calibration pool only (6,352 images, 55 animals), frozen FP32 v2 vectors. Report: `evaluations/czechlynx/flank_grouping_v2_fp32/` (`index.html`, `summary.json`, `profiles.csv`, `facing.csv`, `side_edge_models.csv`).

## Conclusion

**Splitting by flank does not fix grouping. The binding constraint is within-flank individual discrimination by the embedding.**

- **Facing works:** predicted facing reaches pose AUC 0.995 and human-audit AUC 0.989 (n=145, including 103 images without pose), and is confident on 74% of images.
- **Grouping barely moves:** best animal-level pairwise F1 is **0.602** (edge grouping without flanks: 0.585; best saved control: 0.264), with **0/55** exact recoveries.
- **Flank clusters are as impure as animal clusters:** flank-level precision 0.6–0.7, at most 8/110 flank units recovered exactly.
- **Removing opposite-flank pairs doesn't help the edge model:** per-side edge AUC is 0.85–0.90, the same as without the split. Different animals photographed from the same side remain confusable at this candidate depth.
- **Cross-flank bridging behaves as expected but cannot repair impure clusters:** one-to-one matching pairs 20–76 clusters, and the co-occurrence minimum m barely matters.
- **Count intervals are not usable:** 33/144 profiles contain the true 55, but every interval is 19 or more wide and rests on about 30 merge errors. As before, a plausible count can hide cancelling merge and split errors.

Declared rules: **facing classifier usable: met.** Flank-level progress, animal-level progress and count candidate: **not met**. No count configuration; selection stays reserved.

## Run and checks

- Run by the assistant with the user's authorization (CPU), `complete` in 215 s. No models loaded; cache hash unchanged.
- 13 new tests; 132 pass across the related suites; Ruff F clean.
- 1,633 calibration images (26%) have ambiguous facing (p in [0.2, 0.8]). The "unresolved" policy leaves them out of grouping; it raises precision slightly and lowers recall sharply.

## Facing classifier

| Measure | Value |
| --- | --- |
| Pose-labelled images (cross-fitted) | 1,115: AUC 0.995, accuracy 96.5% |
| Human audit (audit "right" = faces right) | 145: AUC 0.989, accuracy 94.5%; on 103 without pose: AUC 0.986 |
| Confident (p < 0.2 or > 0.8) | 74% of all images, 72% of images without pose |
| Predicted faces-right share | 49% |

## Grouping (144 profiles = 2 arms × 3 k × 2 ambiguity policies × 4 within-flank methods × 3 m)

| Profile | Precision | Recall | F1 | Exact /55 | Merges | Splits | Flank exact /110 | Count interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| derived / k=100 / assign / Louvain 1.0 / m=1 | 0.649 | 0.561 | **0.602** | 0 | 31 | 54 | 1 | [58, 81] |
| derived / k=150 / assign / Louvain 1.0 | 0.587 | 0.576 | 0.581 | 0 | 28 | 53 | 1 | [53, 77] |
| derived / k=50 / assign / Louvain 1.0 | 0.689 | 0.494 | 0.575 | 0 | 32 | 54 | 1 | [68, 93] |
| derived / k=100 / assign / G2 τ=0.7 | 0.676 | 0.327 | 0.441 | 0 | 51 | 55 | **8** | [186, 273] |
| Edge grouping best (no flanks) | 0.584 | 0.586 | 0.585 | 1 | 38 | 49 | — | — |

Profiles with higher flank-level precision fragment heavily (G2 τ=0.7: 186–273 count bounds). Profiles with fewer fragments merge animals.

## Interpretation

Three rounds now point the same way:

1. Candidate coverage is solvable: oracle 46/55 with day-level links.
2. Graph features and grouping rules lift F1 from 0.26 to about 0.60.
3. Neither DISK/LightGlue nor flank separation adds discrimination.

What remains is that the **frozen adapted MiewID embedding does not separate same-side photos of different lynx reliably enough**, consistent with its 68.6% calibration top-1. Further graph or grouping engineering on these vectors is unlikely to reach reliable counts.

## Possible next steps (not approved)

1. **Representation:** adapt MiewID with **(animal, facing)** as the training class on the fitting pool. Facing labels come from pose plus the facing classifier, which needs adapted vectors for fitting images (GPU inference). This stops training from pulling opposite flanks together, which appearance cannot support, and may sharpen same-side discrimination. Evaluate with the same identity-cross-fitted edge pipeline.
2. **Human-in-the-loop baseline:** measure the review effort to reach correct groups from the best profile (about 85 merge/split corrections for 6,352 photos). Then agree with the user whether "reliable after bounded review" can satisfy the requirement.
3. **Data quality:** restrict to confident-facing, low-risk images and see whether a high-precision core count is achievable, reporting the rest as unresolved.

## Implementation notes

- Count bounds count only flank clusters with ≥ 2 images; singletons are reported, not counted. This was an implementation choice not stated in the proposal.
- `faces_left` / `faces_right` describe the image frame. Whether the audit's left/right meant facing direction still needs user confirmation; the analysis depends only on consistency.
