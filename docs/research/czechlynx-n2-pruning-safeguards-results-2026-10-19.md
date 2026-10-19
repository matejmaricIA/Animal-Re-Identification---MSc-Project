# N2: pruning floors and duplicate-merge second check on calibration — results

Date: 19 October 2026. Scope: [approved proposal](../czechlynx-n2-pruning-safeguards-proposal-2026-10-19.md) (work item (e)). Calibration only (6,352 photos, 55 lynx, 1,791 camera-day bundles), F1 vectors. Selection is spent and the official test is locked; neither was touched.

**Report:** `evaluations/czechlynx/pruning_safeguards_f1_v1/` (`index.html`, `summary.json`, `simulations.csv`, `settings.csv`, `manifest.json`).

## Conclusion

**The duplicate-merge second check works and should be adopted. The pruning floors did not earn their cost on calibration, and the region holdout did not reproduce the selection failure they were meant to fix.**

**1. Merge check: a large, robust effect.** Averaged over all 18 settings × 5 seeds, region folds:

| Reviewer error | Merge check | Misassigned photos | Wrong merges per run | Decisions |
| --- | --- | ---: | ---: | ---: |
| 3% | off | 2.79% | 5.2 | 3,041 |
| 3% | on | **1.59%** | **0.2** | 2,940 |
| 5% | off | 4.50% | 8.4 | 3,380 |
| 5% | on | **2.49%** | **0.3** | 3,222 |

- Misassignment roughly halves, and wrong merges almost disappear.
- Total decisions even fall, because wrong merges no longer cascade into later corrections.
- True duplicates blocked by a mistaken second "no" are rare: 0.03–0.14 per run.
- **Trade-off:**
  - Without the check, wrong merges and pruning duplicates partly cancel in the count (mean −0.3 at 3% error).
  - With it, residual errors are mostly surplus animals (mean +0.8 at 3%, range −1 to +4).
  - Misassignment is the better-behaved error; the count is no longer flattered by cancellation.

**2. Cosine floor: no benefit.**
- In every setting, under both fold schemes, the appearance top-5 or top-10 floor **never found a hidden true match**: 0 matches in 49–206 extra comparisons per run.
- It costs about 100–200 decisions (+4–9%).
- The true matches that P(match) prunes on calibration are not in the appearance top-10 either. The "a bad location prior hides an obvious appearance match" hypothesis gets no support here.

**3. Minimum shortlist (min_shown): weak evidence.**
- Under identity folds, showing at least 10 candidates removed 1 of the 2 Carpathian pruning duplicates (count +1 → 0) for about +70 decisions (+3%).
- Under region folds there were no pruning duplicates to remove at 0% error, so it only added cost (+2.5%).

**4. The region holdout did not reproduce the selection failure.**
- At 0% reviewer error, region folds produced **zero** pruning duplicates in every setting, including the unmodified baseline. Rule R1 therefore passed trivially and does not test the floors.
- Region-trained models are *less* sharp than identity-cross-fitted ones:
  - Brier 0.0126 vs 0.0039;
  - a lower miss rate at ε 0.001 (0.06% vs 0.17%).
- So they prune less (2,350 vs 2,116 decisions) and automate less (44.7% vs 48.4%).
- This is the honest cost of applying models to another region. The over-confident pruning seen on *selection* (10 duplicate events) came from a model trained on all calibration animals; this proxy does not reproduce that mode [inferred].
- **Pruning fragility on new data therefore remains open, and only new held-out data can test it.**

## Declared rules (primary: region folds)

W1 (live, F1 vectors) = 5,325 decisions, so the R2 limit is 2,662. 9 of the 36 settings meet R1–R4; all 9 have the merge check on. No setting without the merge check passes R3 or R4.

**Mechanical candidate** (cheapest passing): ε 0.001, min_shown 1, cosine_floor 0, **merge check on**. This is the frozen (c) operating point plus the merge check.

| | 0% reviewer error | 3% (median of 5) | 5% (median of 5) |
| --- | ---: | ---: | ---: |
| Decisions | 2,350 (**−55.9% vs W1**) | 2,545–2,594 | 2,727–2,849 |
| Automated bundles | 44.7% | — | — |
| Misassigned photos | 0.24% | 1.07% | 1.79% |
| \|count error\| | 0 | 1 (range −1 to +1) | 1 (range +1 to +2) |
| Hours per 1,000 photos (placeholder / fast-click) | 0.82 / 0.38 | ≈ 0.90 / 0.42 | ≈ 0.97 / 0.47 |

- Neither redirect fired: some settings meet R1 with R2, and some merge-check settings meet R4.
- Among merge-check settings, pass or fail on R3 is largely seed noise. The failures are median |count| of 2–4 at 3% or 5% error on otherwise similar rows. **The robust distinction is merge check on vs off, not the floor settings.**

**Audits** (not part of N2) still fail with a noisy auditor. With the candidate at 3% error, the count-audit-20 interval missed the truth in 5/5 seeds and the automation audit in 4/5. This is unchanged from (a); auditor consensus or bias correction remains a separate open item.

## Run and checks

- Run by the assistant with the user's authorization, on CPU. Status `complete` in 2,188 s: 794 simulations plus models for both fold schemes.
- **Reference check passed exactly.**
  - The default settings on identity folds reproduce the saved F1 rows (`nonmatch_pruning_facing_classes_v1`) field for field: decisions 2,116 / 2,485, count +1 / +1, merges, wrong merges, automation and misassigned photos.
  - The saved run bound the same vector cache and calibration manifest.
  - Live W1 = 5,325.
- Input hashes (cache, raw metadata, calibration manifest) were unchanged before and after.
- Tests: 18 new, plus 39 in the related suites (pruning, frozen review, gated matching): 57 passed. Ruff F is clean.
- No existing source was edited. `benchmark_nonmatch_pruning.py` and `benchmark_frozen_review.py` are untouched, so their hash bindings hold.

**Deviations, all disclosed:**
- R2 uses 0.5 × live F1 W1 (2,662). The proposal had quoted v2's W1 (2,641); this was corrected in the proposal before the run.
- `--threads 4` was used to match the saved run.
- R4's thresholds are applied to every setting. Merge-off settings fail them.

## What this means

1. **Adopt the merge check.** It is cheap, robust in simulation, and fixes the S4 failure mechanism seen on selection (6 of 10 merges wrong).
2. **Drop the cosine floor.** It never helped on calibration.
3. **Whether to keep min_shown 10 is a judgement call:** about +3% cost, with one observed benefit under identity folds.
4. **Pruning on unseen animals is still the open risk.** Calibration cannot reproduce the selection failure. The decisive test is the blind partner evaluation (X1).

**For that test, freeze** either:
- the mechanical candidate (ε 0.001 + merge check); or
- a more conservative variant (ε 0.0003 + merge check: 2,512 decisions, still −53% vs W1).

Pre-register the choice before any partner data is read.

## Limits

- Two regions only, with 17 and 38 animals. Day-level dates; supplied masks; simulated reviewers with independent errors; placeholder costs.
- Calibration development evidence only. No count configuration is approved for deployment, and there was no outreach.
