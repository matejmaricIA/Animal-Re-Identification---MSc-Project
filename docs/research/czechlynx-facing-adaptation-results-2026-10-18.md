# CzechLynx facing-aware MiewID re-adaptation: results

Date: 18 October 2026. Scope: [approved proposal](../czechlynx-facing-adaptation-proposal-2026-10-18.md) (work item (b)). Training used the fitting pool only. Evaluation used calibration only (6,352 photos, 55 lynx); selection stays reserved.

Artifacts:

| What | Where |
| --- | --- |
| Fitting facing labels | `evaluations/czechlynx/fitting_facing_labels_v1/`; public-MiewID fitting cache `data/czechlynx/cache/miewid_public_fitting_v1/` |
| Trials | `models/czechlynx/miewid_facing_classes_v1/` (F1), `models/czechlynx/miewid_facing_subcenter_v1/` (F2) |
| Calibration comparisons | `evaluations/czechlynx/facing_adaptation_calibration_{classes,subcenter}_v1/` (`index.html`, `comparison.json`, `clustering/`, `pair_diagnostics/`) |
| Review cost | `evaluations/czechlynx/nonmatch_pruning_facing_{classes,subcenter}_v1/` |

## Conclusion

**Facing-aware training does not sharpen same-side discrimination. The embedding-level hypothesis behind (b) is not supported.** Under the pre-declared rule, (b) is a **documented negative**: neither arm meets R1 and R2.

The explicit-facing-classes arm (F1) is still a **modest, real improvement** over v2:
- Protocol top-1 rises from 68.6% to 71.3%.
- Safe automation (≤ 1% misassigned) rises from 45% to 54%.
- The best admissible review setting needs 9% fewer expert decisions.
- It is more robust to reviewer error.

**R1, same-side discrimination (≥ +3 points macro): not met by either arm.** Cross-encounter nearest-neighbour top-1, with common facing labels for all models (pose, else the public-MiewID prediction):

| Model | Any side (micro / macro) | **Same side** | Opposite side | Facing AUC (own vectors) |
| --- | --- | --- | --- | ---: |
| Public MiewID | 76.9% / 62.2% | 76.6% / 62.3% | 30.2% / 13.8% | 0.9995 |
| v2 (animal classes) | 80.9% / 66.8% | 80.8% / **67.6%** | 32.2% / 15.4% | 0.9946 |
| F1 (animal × facing classes) | 80.8% / 67.1% | 80.6% / **67.6%** (−0.04) | 29.3% / 12.4% | 0.9997 |
| F2 (sub-center ArcFace, K = 3) | 81.1% / 67.3% | 81.0% / **67.9%** (+0.3) | 31.9% / 16.6% | 0.9957 |

F1 restores facing information in the embedding and weakens opposite-side matching, as designed, but same-side retrieval does not move. Merging flanks into one class was therefore not what limited same-side discrimination.

**R2, review cost (≥ 15% fewer than 2,211 decisions, or ≥ 55% safe automation): not met.** At the declared best setting (P(match) order, ε 0.001, τ 0.99, safeguards on, perfect reviewer):

| | v2 | F1 | F2 |
| --- | ---: | ---: | ---: |
| Expert decisions | 2,211 | **2,116 (−4.3%)** | 2,173 (−1.7%) |
| Automated bundles | 45.2% | 48.4% | 46.9% |
| Misassigned photos | 0.19% | 0.24% | 0.19% |
| Count error | 0 | +1 | +1 |
| h / 1,000 photos (placeholder / fast-click) | 0.77 / 0.35 | 0.74 / 0.33 | 0.76 / 0.34 |
| Max automation with ≤ 1% misassigned (any G1 setting) | 45.2% | **53.6%** | 47.1% |
| Best admissible setting (\|count error\| ≤ 1, ≤ 1% misassigned) | 2,211 at τ 0.99 | **2,015 at τ 0.98 (−8.9%)**: 53.5% automated, 0.33% misassigned | 2,173 at τ 0.99 |
| Replay rank-1 share, cosine / P(match) order | 82.9% / 93.2% | 84.5% / 94.3% | 83.8% / 93.9% |

**R3, no hidden cost under 3% reviewer error** (count within ±2, misassignment ≤ v2 + 1 point; best setting):
- **F1 meets it:** count +1 and 1.9% misassigned (v2: +1, 2.3%). At τ 0.98, F1 gives 1.9% vs 5.7% for v2.
- **F2 fails it:** count +3 and 2.8% misassigned.

**Carried-over targets:** still no candidate operating point. Automation stays below 70%, and hours stay above 0.5 per 1,000 photos at placeholder costs.

## Runs and checks

- **Implementation:** five new files; existing source unedited. 16 new tests pass, 164 across eight related suites; Ruff F clean.
- **Fitting facing labels** (GPU, 94 s): of 16,504 photos, 2,415 are labelled from pose, 11,431 predicted and 2,658 "other" (16.1%). The identity-cross-fitted pose AUC on fitting is 0.9996.
- **Trial preparation:** both arms reproduce v2's inner split exactly (`ef8d6405…`).
- **Training** (GPU, RTX 4070 Ti SUPER, FP32, at most 10 epochs each; v2 for comparison: best epoch 6, top-1 0.714, recall@50 0.953):

  | | F1 classes | F2 sub-center |
  | --- | --- | --- |
  | Run time | 2,813 s | 2,843 s |
  | Classes | 358 | 131 animals |
  | Selected epoch | 7 | 9 |
  | Inner holdout, same-facing top-1 | 0.689 → 0.751 | 0.689 → 0.754 |
  | Inner holdout, identity top-1 | 0.654 → 0.721 | 0.654 → 0.724 |
  | Inner holdout, recall@50 | 0.926 → 0.934 | 0.926 → 0.955 |

  Both runs proved a native embedding update at epoch 2.
- **Weight check:** the calibration vectors were checked to carry the adapted weights. The loaded state hash equals each checkpoint's (F1 `533fa325…`, F2 `ff92f0ef…`), not the public initialization's. Repeated "copy to meta parameter" warnings come from the publisher model's own loading: they also appear when only the public model loads.
- **Calibration comparisons:** 182 s and 180 s; all 25 controls plus the pair diagnostic completed; frozen public and v2 caches unchanged.
- **Review-cost runs:** the committed (a) simulation, unchanged; 436 s and 429 s. The live-W1 reference check passed on both.
- **Inner-holdout caveat:** the same-facing gains are not comparable with v2, because v2 never recorded that metric. Calibration, measured on identical photos, is the fair comparison.

## What this means

1. **The flank-merging objective was not what limited the embedding.** With flanks kept apart (F1) or allowed to form their own view modes (F2), same-side retrieval stays at about 67–68% macro, the same as v2. The limit lies elsewhere. Plausible causes, not established:
   - training data quantity and diversity (131 training animals);
   - image conditions (IR/night, blur, partial bodies);
   - the 440 px global descriptor's ability to resolve spot patterns.
2. **F1 is still the better embedding for the review workflow.** It improves protocol recognition, safe automation and robustness to reviewer error at no accuracy cost. It costs some opposite-side similarity and gives three pruning-induced duplicate events at ε 0.001; the count stays within ±1. It is reasonable to carry F1 forward as the working embedding, but that is a user decision: it did not pass the declared rules.
3. **Further gains will not come from more class-definition tweaks on this data.** Per the declared rule, no further training variants follow without a new proposal. The remaining levers are:
   - real reviewer timing: the fast-click cost model already meets 0.5 h;
   - engineered audits;
   - more and different training data, such as other lynx datasets or park data;
   - end-to-end integration to measure the full workflow.

## Limits

- **Calibration is development evidence.** Selection stays reserved. MiewID's upstream exposure to CzechLynx remains unresolved.
- **Facing labels:** predicted labels contain some errors, and "other" mixes frontal, rear and ambiguous views.
- **Review costs** are simulated with placeholder costs, as in (a).
- **No count configuration, selection evaluation or outreach follows from this work.**
