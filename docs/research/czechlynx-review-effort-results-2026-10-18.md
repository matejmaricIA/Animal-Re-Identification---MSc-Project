# CzechLynx simulated review effort: results

Date: 18 October 2026. Scope: [approved proposal](../czechlynx-review-effort-proposal-2026-10-18.md). Calibration pool only (6,352 photos, 55 lynx), frozen FP32 v2 vectors. Report: `evaluations/czechlynx/review_effort_v2_fp32_r2/` (`index.html`, `summary.json`, `simulations.csv`, `curves.csv`, `curves.png`).

## Conclusion

**Grouping first does not save expert work. Embedding-ranked catalogue matching (W1) is by far the most efficient way to a verified grouping and count.**

- **W1 is the cheapest workflow that ends fully correct.** A perfect simulated reviewer, working through 1,791 trap-day bundles in date order and comparing each against the catalogue in embedding-similarity order, reaches the **exact verified grouping (F1 1.000, 55/55 animals, count error 0)** with **5,283 decisions**. At the declared placeholder costs that is about **11.7 expert-hours, or 1.8 h per 1,000 photos**.
- **Unassisted matching (W0) costs 5.7× more:** 29,930 decisions (66.5 h).
- **Group-then-correct (W2) costs more than W1 and still ends wrong.** From the best automatic grouping it needs 1.27× W1's decisions and is left with about 100 surplus fragment groups (F1 0.963, count error +107 at K=25). From the flank grouping it needs 1.91×; from the legacy grouping, 2.92×. Only the legacy start (26 coarse groups) ends exactly correct.
- **The ranking is what saves the work.** The correct animal is ranked **first for 82.9%** of known-animal bundles, top 3 for 90.6% and top 10 for 95.9% (median rank 1, maximum 47). Discovering the 55 new animals costs 1,485 comparisons, because a new animal is only confirmed after scanning the whole catalogue.
- **Trap-day bundling matters a lot:** without it, W1 needs 16,334 decisions (3.1×).
- **Reviewer errors dominate the outcome.** With 3% wrong yes/no answers, W1 ends at F1 0.916 with count error +5. Every workflow degrades similarly, and W2 more than W1 (F1 0.76–0.88).

Declared rules: **grouping saves work: not met** (all W2 ratios > 1). Strong saving: not met. Robust to reviewer error: not met. These rules tested whether *grouping-first* helps, and it does not.

## Run and checks

- Run by the assistant with the user's authorization (CPU).
- First run `review_effort_v2_fp32` completed all 88 simulations but failed while writing the report: a numpy int64 reached the JSON writer. It is preserved as failed. The serializer was fixed with a regression test and the run repeated as `_r2` (`complete`, 131 s). Its numbers are identical to the first run's log.
- **Starting groupings reproduce the saved runs exactly:**

  | Start | Groups | Pairwise precision |
  | --- | ---: | ---: |
  | Edge grouping | 235 | 0.58395 |
  | Flank grouping | 173 | 0.64933 |
  | Legacy control | 26 | 0.26603 |

- 15 new tests; 146 pass across the related suites (before the serializer test was added); Ruff F clean. No models loaded; cache and saved-input hashes unchanged.

## Main setting (perfect reviewer, trap-day bundles)

| Workflow | Glances | Catalogue comparisons | Pair questions | Decisions | Hours | h / 1,000 photos | F1 | Exact /55 | Groups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W0 unassisted | 0 | 29,916 | 0 | 29,930 | 66.5 | 10.5 | 1.000 | 55 | 55 |
| **W1 ranked** | 0 | 5,269 | 0 | **5,283** | **11.7** | **1.8** | **1.000** | **55** | **55** |
| W2 edge, K=25 | 1,842 | 4,285 | 581 | 6,721 | 13.7 | 2.2 | 0.963 | 29 | 162 |
| W2 flank, K=25 | 2,165 | 7,739 | 197 | 10,111 | 20.8 | 3.3 | 0.969 | 28 | 147 |
| W2 legacy, K=25 | 2,953 | 12,414 | 61 | 15,431 | 31.9 | 5.0 | 1.000 | 55 | 55 |

Decision counts also include 3–14 "extra animal in bundle" decisions per workflow.

**Why W2 loses.**
- The split check costs one glance per (group, bundle, animal) piece.
- Every moved-out piece is then placed by ranked comparison against a catalogue of hundreds of fragment groups, not 55 animals. A piece that matches none of them costs a full scan.
- The merge phase stops after K consecutive "no" answers while many true fragment pairs are still unasked, because the cross-edge ranking of fragment pairs is weak.

## Sensitivity (W1 and best W2, K=25)

| Setting | W1 decisions | W1 F1 / count error | W2 edge decisions | W2 edge F1 / count error |
| --- | ---: | --- | ---: | --- |
| Main | 5,283 | 1.000 / 0 | 6,721 | 0.963 / +107 |
| Unidentifiable photos set aside | 5,011 | 1.000 / 0 | 6,145 | 0.959 / +103 |
| 3% reviewer error | 7,313 | 0.916 / +5 | 5,364 | 0.793 / +161 |
| 3% error + set-aside | 6,750 | 0.936 / +8 | 5,212 | 0.794 / +151 |
| No bundling | 16,334 | 1.000 / 0 | 17,014 | 0.946 / +122 |

"Unidentifiable" means a bundle whose photos are all in the top 20% of the label-free identifiability risk. The set-aside variant counts the animals with any resolved photo.

## What this means

- **There is a practical route to a verified count today, but it is assisted catalogue matching with expert confirmation of every bundle.** On these data that is about 1.8 expert-hours per 1,000 photos, with one click usually enough because the right animal ranks first 83% of the time. Its output is a fully verified grouping and count, not just suggestions. But it is the workflow the 22 September requirement called insufficient on its own. **Whether expert-confirmed ranked matching can satisfy "reliable clustering and counts" is a user decision.**
- **Reviewer accuracy matters more than the workflow.** A 3% error rate turns an exact count into +5. A real tool needs safeguards, such as a second look before creating a new animal (≈55 × top-5 extra comparisons) and periodic duplicate checks between catalogue animals. These safeguards were not simulated.
- **Automatic grouping is not wasted.** A W3 variant was not part of this proposal and was not run: feed split-checked automatic groups into W1 as larger units instead of trap-day bundles. It could combine both strengths and would be a natural next measurement.
- **A better embedding pays off directly here.** Every rank position saved on 1,750 known-animal bundles is a decision saved, so facing-aware re-adaptation remains the representation lever.

## Limits

- **Simulated reviewer:** real decision times vary, and some "same animal?" questions are unanswerable from photos. The placeholder costs (5 s glance, 8 s comparison, 10 s pair question) are not measured times.
- **Bundles are day-level;** a park with exact timestamps would have smaller, purer bundles.
- **Calibration development evidence only;** no selection evaluation, count configuration or outreach.
