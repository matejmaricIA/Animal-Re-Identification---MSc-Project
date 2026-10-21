# Proposal: flank-aware grouping with pose-derived and embedding-predicted facing

Date: 18 October 2026. Status: **approved and run 18 October 2026: facing works, but flank-aware grouping does not reach the progress rules**. See [results](research/czechlynx-flank-grouping-results-2026-10-18.md).

## Why

The [edge-grouping results](research/czechlynx-edge-grouping-results-2026-10-18.md) left pairwise evidence as the bottleneck: edge-model AUC 0.87–0.89, best pairwise F1 0.585, 1/55 exact recoveries. The checks below show a structural reason. **About half of all same-animal photo pairs show opposite flanks**, and a lynx's left and right coat patterns are different, so those pairs cannot be matched by appearance. Today they sit in the same candidate graph as the solvable same-flank pairs, adding wrong edges and missing links. Metadata, on the other hand, bridges flanks very reliably.

## Read-only evidence (calibration pool unless stated)

| Check | Result |
| --- | --- |
| Pose keypoints in raw metadata | 5,206 images (20 named keypoints: nose, eyes, ear bases, tail base, left/right limbs). Positions are given even for the hidden side, so facing is taken from head vs tail-base geometry |
| Pose-derived facing (head–tail x-offset ≥ 0.5 × body length) | 2,390 / 2,346 side views, 102 frontal/rear/oblique, 368 missing head or tail. Calibration: 1,115 side views over 38/55 animals; fitting: 2,415 over 114 |
| Pose facing vs human audit | **56/56 agree, but with the names swapped**: the audit's "right" equals "faces right in frame". The audit labels apparently record facing direction, not the anatomical flank shown. This proposal uses neutral `faces_left` / `faces_right` |
| Facing from adapted MiewID vectors (logistic, inputs ×10, identity-cross-fitted) | AUC **0.995** vs pose labels, **0.989** vs audit (n=145, of which 103 have no pose), accuracy 94.5%. Confident (p<0.2 or >0.8) on 72% of unlabelled images. An unscaled first attempt was confident on only 20% because regularization acted on unit-norm inputs |
| Public MegaDescriptor (bbox) for comparison | Facing AUC 0.60, i.e. nearly flip-invariant |
| Same-animal pairs among pose-labelled images | 37,443 same-facing vs **36,283 opposite-facing** |
| kNN k=50 candidate edges by predicted facing | 74% same-facing. Same-animal rate 55% for same-facing edges vs 35% for opposite-facing; 18% of positive edges are opposite-facing |
| Flank-unit oracle (unit = animal × predicted facing; same-facing kNN edges only) | k=50: 20/110 units connected; k=150: 43/110 |
| Opposite-facing pairs at the same trap on the same day | 31,645 pairs, **98.95% the same animal**. 50/55 animals have at least one such pair, and all 55 are seen facing both ways |

## Design (calibration only, CPU)

**1. Facing for every image.**
- Pose-derived labels where keypoints exist.
- An identity-cross-fitted logistic facing classifier on the frozen adapted vectors gives p(faces_right) everywhere else. Each image is predicted by a model that never saw its animal.
- Images with p in [0.2, 0.8] are flagged **ambiguous**. They are assigned to their more likely side, and a sensitivity run leaves them unresolved.
- Selection-pool pose labels are not used.

**2. Within-flank grouping.**
- Separate candidate graphs per facing: same-facing kNN k ∈ {50, 100, 150} plus same-facing trap-day links.
- The Stage A edge features plus facing-confidence features, and the identity-cross-fitted edge model, are trained separately per facing.
- G2 average-linkage and G3 Louvain from the edge-grouping module produce **flank clusters**.

**3. Cross-flank bridging.**
- For each left-facing cluster × right-facing cluster, the evidence is the number of opposite-facing image pairs sharing a trap-day (derived arm) or an annotated encounter (upper-bound arm).
- Clusters are joined by **one-to-one matching** (greedy on counts, requiring ≥ m co-occurrences, m ∈ {1, 2, 3}). One-to-one matching stops two animals passing the same trap on one day from chaining clusters together.

**4. Count with bilateral bounds**, following the standard capture-recapture treatment of partial identities.
- Matched pairs count as one animal each. Unmatched clusters are what remains.
- Report a **lower bound** of `pairs + max(unmatched_left, unmatched_right)` and an **upper bound** of `pairs + unmatched_left + unmatched_right`, plus the cluster breakdown. A single number is not reported without its interval.

**5. Evaluation (truth used only here).**
- Flank-unit metrics: unit = animal × facing, with known facing restricted to pose/audit-labelled images and predicted facing reported separately.
- Animal-level metrics via `evaluate_clustering`: pairwise precision/recall/F1, exact recoveries, merges/splits.
- Count-interval coverage of the true 55 and the interval width.
- Side-by-side comparison with the Stage A best (F1 0.585, 1/55) and the 25 saved controls.

## Proposed code changes (new files only; no edits to existing source)

1. **`reid_demo/benchmark_flank_grouping.py`** (~400 lines): pose facing parser; identity-cross-fitted facing classifier; per-facing graphs/edge models/grouping (reusing `benchmark_edge_grouping` and `benchmark_coverage_diagnostics`); one-to-one cross-flank matching; bilateral count bounds; flank-unit and animal-level evaluation; report.
2. **`tools/evaluate_czechlynx_flank_grouping.py`** (~150 lines): CLI with the same contracts as the earlier tools (calibration only, selection refused, empty output directory, cache/manifest hashes before and after). The raw metadata file is bound by SHA-256; audit labels are used for validation only.
3. **`tests/test_benchmark_flank_grouping.py`** — synthetic checks:
   - Pose facing geometry, including frontal/missing cases.
   - The facing classifier never predicts an image with a model trained on its animal.
   - Opposite-facing edges never enter within-flank graphs.
   - One-to-one matching refuses a chain through a shared trap-day.
   - Bounds arithmetic.
   - Selection is refused; no neural imports.

Untouched: existing source and data, the adapted model, caches, audit, and the reserved selection/test pools.

## Declared outcome rules (calibration development evidence)

These are defaults; adjust them before approving if needed.

| Outcome | Rule |
| --- | --- |
| Facing classifier usable | Audit agreement AUC ≥ 0.97 and accuracy ≥ 0.93 (cross-fitted), ≥ 70% confident |
| Flank-level progress | ≥ 30/110 flank units recovered exactly with flank-level pairwise precision ≥ 0.90 |
| Animal-level progress over Stage A | Pairwise F1 ≥ 0.70 and ≥ 10/55 exact recoveries with ≤ 10 merge errors |
| Count candidate | True 55 inside [lower, upper] and width ≤ 11 (20%) for a profile that also meets animal-level progress. This would justify freezing one configuration and proposing the reserved selection evaluation |

## Risks and open questions

- **Naming convention.** Please confirm what the audit's "left/right" meant. The pipeline itself only needs consistency.
- **Facing training data.** The facing classifier is trained on calibration pose labels with cross-fitting, which is acceptable for development. For a frozen method it should be trained on fitting-pool labels, which needs adapted-model vectors for the fitting images (GPU inference, separate approval).
- **Bridging depends on capture design.** CzechLynx often captures both sides on one trap-day, possibly with paired cameras. Whether Risnjak cameras do the same is unknown and worth asking about later. Without such captures, unmatched flank clusters widen the bounds rather than producing wrong merges.
- **Frontal, rear and partial views** stay ambiguous. They are reported and never silently dropped.
- **Night/IR** performance of the facing classifier and grouping is not separately validated.

## Runtime and who runs what

CPU only, estimated 5–10 minutes (per-facing edge models plus about 3 × 2 × 9 grouping/bridging profiles). Tests are synthetic and fast. No GPU or model loading.
