# CzechLynx calibrated non-match pruning with review safeguards: results

Date: 18 October 2026. Scope: [approved proposal](../czechlynx-nonmatch-pruning-proposal-2026-10-18.md) (work item (a)). Calibration pool only (6,352 photos, 55 lynx, 1,791 camera-day bundles), frozen FP32 v2 vectors.

Reports:
- Main run: `evaluations/czechlynx/nonmatch_pruning_v2_fp32/` (`index.html`, `summary.json`, `simulations.csv`, `pmatch_models.csv`, `placements_g1.csv.gz`, `pruning.png`, `tradeoff.png`).
- Post-hoc check: `evaluations/czechlynx/nonmatch_pruning_v2_fp32_posthoc_pmatch_auto/` (script and JSON).

## Conclusion

**Expert time for a verified, exact count falls by more than half, mainly because space-time-aware re-ranking puts the right animal first far more often. Pruning helps less than hoped, the targets are still not all met, and audits do not survive a noisy reviewer.**

- **Best exact-count setting** (perfect reviewer): P(match) candidate order, ε = 0.001, W4 auto-accept τ = 0.99, all safeguards on.
  - Count error 0, 0.19% misassigned photos (12 of 6,352), pairwise F1 0.999, 45% of bundles automatic.
  - **2,211 expert decisions: 0.77 h per 1,000 photos at placeholder costs, 0.35 h at fast-click costs.**
  - That is **−58% vs W1** (5,283 decisions, 1.85 h) and **−50% vs the post-hoc full-scan W4** at the same τ (4,449, 1.54 h). At fast-click costs, W1 would be 0.92 h, so the saving is −62%.
- **The calibrated P(match) model works.** Identity-cross-fitted, both sides excluded from training: AUC **0.998** with space-time features (appearance only: 0.988) and well calibrated (see below). In the replay, the true animal ranks first for **93.2%** of known parts in P(match) order vs 82.9% in cosine order.
- **Re-ranking is the main lever, not pruning:**
  - P(match) order cuts comparisons on hard matches from **2,379 to 417 (−82%)**.
  - Pruning at ε = 0.001 then cuts new-animal comparisons by 57–58% (1,644 → 684–699) with no extra duplicates.
  - Larger ε adds 2–16 duplicate animals. The declared rule needed a ≥ 75% cut with \|count error\| ≤ 1, so pruning alone falls short.
- **Space-time context is essential.** Appearance-only models leave count errors of +10 to +59 even with all safeguards on.
- **Reviewer error remains the largest risk, but the safeguards contain the count:**
  - With 3% wrong expert answers, the best setting ends with **count +1 and 2.3% misassigned**, vs 8.0% for the full-scan W4 and 9.5% for W1 at the same error rate.
  - Turning all safeguards off at 3% error gives count errors of +15 to +37.
- **Audits do not survive reviewer error.** With a perfect auditor they cover the truth (automation audit 100% of G1 runs; count audit 100%). With a 3% auditor:
  - the automation audit covers the truth in only **50%** of runs;
  - the count audit covers it in **0–3%**, because a noisy auditor scanning about 25 older animals often answers a false "same animal".

  A deployable audit needs consensus checks or a bias correction.
- **Post-hoc (outside the approved grid):** gating auto-accept on P(match) instead of the W4 confidence model is **worse**: 41% automatic at 0.82% misassigned vs 45% at 0.19%. 64% automation costs 3.4% misassigned. The 70% automation target is not reachable by switching the gate.

**Declared targets** (perfect reviewer, safeguards on, placeholder costs):

| Target | Result |
| --- | --- |
| ≥ 70% automatic | **Not met.** Maximum 45% with ≤ 1% misassigned (52% at 1.32%) |
| ≤ 1% misassigned photos | Met in 30 of 40 G1 rows (every τ ≥ 0.99 row and every no-automation row) |
| \|count error\| ≤ 1 | Met in 16 rows (ε ≤ 0.001 in both orders) |
| ≤ 0.5 h per 1,000 photos | **Not met** at placeholder costs (best with \|count error\| ≤ 1: 0.73). It would be met at fast-click costs (0.33–0.36), but the target was declared at placeholder costs |
| Audit interval covers the true error | Met in every perfect-reviewer automated G1 run |

**No candidate operating point:** automation and hours fail. Additional rules:
- **D1 pruning works: not met.** The largest cut with \|count error\| ≤ 1 is 58%.
- **D2 re-ranking helps hard matches: met** (−82%).
- **D3 safeguards earn their cost: met in 2/4 combinations.** Count error drops from +15–37 to 0–5 in all four, but in the P(match) arms misassignment is not halved and the extra time is 28–39%.
- **D4 cold-start guard: mechanically met, but uninformative.** At τ = 0.99 no collapse happened even without the guard.

## Run and checks

- Run by the assistant with the user's authorization (CPU). `complete` in **440 s**, 148 simulations. No models loaded; the cache, raw-metadata and calibration-manifest hashes were unchanged before and after.
- **Reference check passed.** The safeguard-free cosine full scan reproduces live W1 exactly (labels and 5,283 decisions). With τ = 0.98 / 0.99 / 0.995 it reproduces the saved post-hoc full-scan W4 rows: 4,279 / 4,449 / 4,578 decisions, automation 0.609 / 0.532 / 0.457, count error 0.
- 18 new tests; 167 pass across the eight related suites; Ruff F clean. No existing source was edited.
- The post-hoc check reused the committed module functions from a separate script. Its evidence is saved in its own directory; the main output is untouched.

## P(match) model (identity-cross-fitted on 56,987 replay records, 1,750 true matches)

| | Space-time + appearance | Appearance only |
| --- | ---: | ---: |
| AUC / Brier | **0.998** / 0.0044 | 0.988 / 0.0076 |
| True animal ranked first (replay) | **93.2%** (cosine: 82.9%) | 84.8% |
| Replay comparisons for known parts | 2,003 (cosine: 3,784) | 3,384 |
| Miss rate (true match below ε) at ε = 0.001 / 0.003 / 0.01 / 0.03 | 0 / 0.23% / 0.97% / 2.1% | 0.86% / 1.5% / 3.1% / 5.7% |
| Non-matches pruned at ε = 0.001 / 0.01 | 86% / 95% | 70% / 89% |
| New-animal short list, mean (max), at ε = 0.001 / 0.01 | 9.7 (25) / 3.9 (9) | 18.3 (43) / 9.4 (24) |

Reliability (space-time model): P ≥ 0.9 → 97.8% observed matches (n = 1,360); P < 0.1 → 0.13% (n = 54,685). Mid-range bins are mildly over-confident (for example, 0.3–0.4 → 19% observed).

## Main grid (G1, perfect reviewer, safeguards on)

| Order | ε | τ | Automated | Misassigned | Count error | Pruning duplicates | Decisions | New-animal comp. | Hard-match comp. | h / 1,000 (placeholder / fast) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cosine | 0 | none | 0% | 0% | 0 | 0 | 5,651 | 1,644 | 2,379 | 1.99 / 1.00 |
| cosine | 0 | 0.99 | 45% | 0.19% | 0 | 0 | 4,945 | 1,644 | 2,392 | 1.73 / 0.94 |
| cosine | 0.001 | 0.99 | 45% | 0.19% | 0 | 0 | 4,661 | 1,360 | 2,392 | 1.63 / 0.88 |
| P(match) | 0 | none | 0% | 0% | 0 | 0 | 3,870 | 1,644 | 417 | 1.37 / 0.62 |
| P(match) | 0 | 0.99 | 45% | 0.19% | 0 | 0 | 3,156 | 1,644 | 418 | 1.10 / 0.55 |
| P(match) | 0.001 | none | 0% | 0% | 0 | 0 | 2,910 | 684 | 417 | 1.03 / 0.41 |
| P(match) | 0.001 | 0.98 | 52% | 1.32% | 0 | 0 | 2,078 | 697 | 406 | 0.73 / 0.33 |
| **P(match)** | **0.001** | **0.99** | **45%** | **0.19%** | **0** | **0** | **2,211** | **699** | **418** | **0.77 / 0.35** |
| P(match) | 0.003 | 0.99 | 45% | 0.19% | +4 | 5 | 2,041 | 560 | 411 | 0.72 / 0.31 |
| P(match) | 0.01 | 0.99 | 45% | 0.19% | +10 | 13 | 1,942 | 489 | 405 | 0.68 / 0.29 |
| P(match) | 0.03 | 0.99 | 44% | 0.19% | +16 | 22 | 1,872 | 447 | 392 | 0.66 / 0.27 |

"Pruning duplicates" counts new animals declared while the true animal was in the catalogue but cut by ε. The periodic duplicate checks merge some of them; the count error shows what remains.

**Where the 2,211 decisions of the best setting go:**

| Decision type | Count |
| --- | ---: |
| Rank-1 one-click confirmations | 990 |
| Further comparisons | 743 |
| Safeguards (second-reviewer re-checks 159, disagreement checks 46, duplicate pair questions 163) | 368 |
| Automation audit glances | 100 |
| Extra-animal splits | 10 |

The one-click confirmations dominate at placeholder costs. That is why the fast-click setting, where they cost 2 s, nearly meets the hours target. Real reviewer timing decides which setting is realistic.

A 10-animal count audit adds about 280 comparisons (best setting: 0.88 / 0.42 h with it). It gives a wide interval: estimate 0% duplicates, count interval [39.7, 55] for the true 55.

## Reviewer error (3%) on the same grid

| Setting | Misassigned | Count error | Exact /55 | Decisions | h / 1,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| W1 full scan, no automation (post-hoc reference) | 9.5% | +11 | 0 | 7,726 | 2.70 |
| Full-scan W4 τ = 0.99, no safeguards (post-hoc reference) | 8.0% | −1 | 1 | 5,746 | 2.00 |
| Cosine, ε = 0, τ = 0.99, safeguards on | 2.0% | 0 | 26 | 6,496 | 2.27 |
| **P(match), ε = 0.001, τ = 0.99, safeguards on** | **2.3%** | **+1** | 23 | **2,606** | **0.91** |
| P(match), ε = 0.01, τ = 0.99, safeguards on | 2.6% | +2 | 15 | 2,089 | 0.73 |

Automation-audit coverage across 3%-error runs: 50%. Count-audit coverage: 0–3%.

## Safeguard ablations (G2, τ = 0.99, ε ∈ {0.003, 0.01}, both orders)

| Change from defaults | Perfect reviewer | 3% reviewer error |
| --- | --- | --- |
| All safeguards off | Count +2 to +16 (pruning duplicates are no longer merged); 11–30% fewer decisions | **Count +15 to +37**; misassigned 1.4–7.2% |
| No disagreement check | Same accuracy; only 1–3% fewer decisions (the check is cheap) | **Misassigned 2.1–12.1%**, cosine arms 7.0–7.3% with only 2 exact animals: the main guard against false "yes" answers |
| No double confirmation | Count +2 to +12 vs +2 to +10 | Count up to +17: the guard against reviewer-made duplicates |
| No duplicate checks | Count rises by 0–4 | Misassigned mostly lower (0.8–2.2% vs 1.5–3.0%) but count higher (+3 to +14 vs 0 to +5). Duplicate pair questions have no second check, so a false "yes" wrongly merges two animals |
| Guard off (vs N = 10, B = 100) | Automation 53% vs 45%, misassigned 0.41% vs 0.19%, same counts | Mixed (count +1 to +10 vs 0 to +5) |
| Guard N = 20, B = 200 | Automation 28%; 6–21% more decisions | Automation 26–27% |
| Appearance-only models | **Count +10 to +59**, with or without the guard; automation 30–37% | Count +11 to +46 |

## Post-hoc: auto-accept gated on P(match) (not in the approved grid)

Setting: P(match) order, ε = 0.001, safeguards on; whole bundles auto-accepted when the top candidate's P(match) ≥ τ_p.

| τ_p | Automated | Misassigned (0% / 3% reviewer error) | Count error | Decisions | h / 1,000 |
| --- | ---: | --- | ---: | ---: | ---: |
| 0.90 | 64% | 3.37% / 16.3% | 0 / −2 | 1,852 | 0.65 |
| 0.95 | 58% | 1.97% / 13.2% | 0 / −1 | 1,977 | 0.69 |
| 0.99 | 41% | 0.82% / 4.0% | 0 / −1 | 2,279 | 0.80 |

Compare the W4 gate at τ = 0.99: 45% automated, 0.19% misassigned. The P(match) model was trained on single-animal parts, but automation acts on whole bundles, some holding several animals, which W4's bundle-level features capture better.

## What this means

1. **A verified exact count now costs about 0.8 expert-hours per 1,000 photos at placeholder costs, and about 0.35 if a one-click rank-1 confirmation takes 2 s.** That is about 40% of confirm-everything matching. The product claim "saves researchers a lot of time" becomes plausible, but it rests on **real reviewer timing**, which is unmeasured. A small timed UI trial would show which cost setting holds.
2. **The remaining cost is mostly one-click confirmations of non-automated bundles** (990 of 2,211). Raising safe automation above 45% is the next lever. The post-hoc check shows that a different gate does not achieve it. A better embedding (item (b), facing-aware re-adaptation) should raise both the rank-1 share and safe automation.
3. **Space-time context does the heavy lifting**, with only day-level dates and cell/camera coordinates. Park data with exact timestamps and camera IDs should help further. The reliance is also a risk: a resident animal seen far from its usual cameras, such as a disperser, gets a low P(match) and could be pruned into a duplicate. Keep ε small (≤ 0.001) and keep the duplicate checks.
4. **Safeguards are worth it for the count:**
   - double confirmation stops reviewer-made duplicates;
   - the disagreement check stops false "yes" answers;
   - duplicate merges need their own second check, because without it they cause wrong merges under reviewer error.
5. **Audits must be engineered for imperfect auditors.** Single-auditor estimates were unbiased only with a perfect auditor. Options include consensus audits (two auditors must agree), bias-corrected estimates using a measured auditor error rate, or auditing against a short P(match) list instead of all older animals.

## Limits

- Simulated reviewers; both cost settings are placeholders. Raw decision counts by kind and per-placement logs are saved for re-costing.
- **P(match) and confidence models are cross-fitted on calibration** (development only). A deployable model needs training on data not used for the embedding's adaptation. The models are trained on a perfect-review replay and applied to an imperfect, evolving catalogue.
- Day-level dates and coordinates for about 88% of photos. Night/IR and flank effects are inherited from the embedding.
- **Calibration development evidence only.** No selection evaluation, count configuration or outreach follows from this.
