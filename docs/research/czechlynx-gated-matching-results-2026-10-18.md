# CzechLynx confidence-gated matching (minimum review): results

Date: 18 October 2026. Scope: [approved proposal](../czechlynx-minimum-review-proposal-2026-10-18.md). Calibration pool only (6,352 photos, 55 lynx, 1,791 camera-day bundles), frozen FP32 v2 vectors. Reports:

- Main run: `evaluations/czechlynx/gated_matching_v2_fp32/` (`index.html`, `summary.json`, `simulations.csv`, `confidence_models.csv`, `tradeoff.png`).
- Post-hoc check: `evaluations/czechlynx/gated_matching_v2_fp32_posthoc_fullscan/` (script and JSON).

## Conclusion

**Automation is accurate, but on its own it saves little expert time. The expensive expert work is confirming new animals and resolving hard matches, not the easy matches the system can take over.**

- **Confidence is predictable:** the identity-cross-fitted confidence model reaches AUC **0.979** with space-time features (camera, location cell and distance history; days since last seen), vs 0.924 from appearance alone.
- **Easy matches can be automated safely:** in the sequential simulation (automatic mistakes propagate), auto-accepting at τ = 0.99 decides **about half of all bundles with ~0.3–0.4% of photos misassigned**.
- **The count fails as designed.** The expert path checks only the top-K candidates before declaring a new animal. True matches ranked below K become duplicates, so count error is **+28 (K=20) to +70 (K=10) even with no automation**. The duplicate sweep (≤ 50 pairs) recovers few of them. This is a flaw in the proposal's design, not in automation.
- **Post-hoc fix, clearly outside the approved grid:** the expert scans the full ranked catalogue before declaring a new animal, as in W1. The **count becomes exact (error 0)** at τ = 0.98–0.995, with 46–61% of bundles automated and 0.22–1.57% of photos misassigned.
- **Expert time barely drops:** 1.48–1.59 h per 1,000 photos vs 1.85 with no automation (−14% to −20%). Automated bundles were mostly one-click decisions anyway. New-animal confirmations (each a full scan; 1,485 comparisons in W1) and hard matches dominate the cost.
- **Reviewer error is the largest risk:** with 3% wrong expert answers, misassigned photos rise to 8–11% in every configuration, even though counts stay within −1 to +4 in the post-hoc variant.
- **Cold-start hazard:** with appearance-only features at τ = 0.95, the system auto-merged *everything* into one animal (99.7% misassigned). With a one-animal catalogue the "margin" looks enormous, so no new animal can ever appear. A deployable system needs a cold-start guard.

**Declared targets:**

| Target | Result |
| --- | --- |
| ≥ 70% automatic | Not met (max 68% at τ=0.95, which has 3.4% misassigned) |
| ≤ 1% misassigned | Met at τ ≥ 0.99 |
| \|count error\| ≤ 1 | Not met in the approved design; met in the post-hoc full-scan variant |
| ≤ 0.5 h per 1,000 photos | Not met (best 0.86 h, with count +66) |
| Audit interval covers the true error | Met in all perfect-reviewer space-time runs; 73% across all automated runs |

**No configuration met all five targets**, so there is no candidate operating point.

## Run and checks

- Run by the assistant with the user's authorization (CPU). `complete` in 135 s. No models loaded; cache hash unchanged.
- 11 new tests; 158 pass across the related suites; Ruff F clean.
- The replay has 1,790 records; top-1 is correct for 80.3% of them (new animals count as incorrect).
- The post-hoc variant reused the committed module functions from a separate script, so the committed code and the main run's provenance are unchanged. Its evidence is saved next to the main output.

## Main grid (perfect reviewer, space-time features, sweep 25, audit 100)

| τ | K | Automated | Misassigned photos | Count error | Decisions | h / 1,000 photos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.95 | 10 | 65% | 2.14% | +66 | 2,507 | 0.86 |
| 0.98 | 10 | 57% | 1.04% | +67 | 2,679 | 0.92 |
| 0.99 | 10 | 49% | 0.30% | +70 | 2,824 | 0.98 |
| 0.995 | 10 | 42% | 0.11% | +69 | 2,952 | 1.02 |
| none | 10 | 0% | 0% | +69 | 3,610 | 1.26 |
| 0.99 | 20 | 51% | 0.41% | +31 | 3,690 | 1.28 |
| none | 20 | 0% | 0% | +28 | 4,502 | 1.58 |

The low hours in this table are partly an artefact: the expert stops after K candidates, which is exactly what creates the duplicates.

## Post-hoc: full-scan new-animal confirmation

| Reviewer error | τ | Automated | Misassigned | Count error | Exact /55 | Decisions | h / 1,000 | Audit estimate [95% CI] vs true bundle error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | none | 0% | 0% | 0 | 55 | 5,283 | 1.85 | — |
| 0 | 0.98 | 61% | 1.57% | 0 | 41 | 4,279 | 1.48 | 2.0% [0.6, 7.0] vs 2.9% |
| 0 | **0.99** | **53%** | **0.41%** | **0** | 45 | 4,449 | 1.54 | 2.0% [0.6, 7.0] vs 1.7% |
| 0 | 0.995 | 46% | 0.22% | 0 | 45 | 4,578 | 1.59 | 0.0% [0, 3.7] vs 1.0% |
| 3% | none | 0% | 9.5% | +11 | 0 | 7,726 | 2.70 | — |
| 3% | 0.99 | 45% | 8.0% | −1 | 1 | 5,746 | 2.00 | 4.0% [1.6, 9.8] vs 4.2% |

## What this means for a minimum-review product

1. **The time sink is confirming new animals and hard matches, not routine matches.** Automating easy matches is necessary but saves only ~15–20%. Big savings need the system to **rule out catalogue animals confidently**, so that confirming a new animal means checking a short list rather than the whole catalogue (1,485 of W1's 5,269 comparisons). This is the next lever: a calibrated "not this animal" model that stops the scan once remaining candidates are very unlikely.
2. **Space-time context is worth keeping and extending.** It lifted confidence AUC from 0.924 to 0.979 using only day-level dates and cell/camera coordinates. Exact timestamps and camera coordinates from a park should do more; they belong in any data request.
3. **Reviewer quality must be engineered:** double confirmation for new-animal decisions, periodic duplicate checks, and audits run by a second reviewer. A 3% expert error rate costs more accuracy than any automation setting.
4. **Safety rails are required:** a cold-start guard (no automation until the catalogue has enough animals) and calibrated audit reporting.
5. **A better embedding would raise automation and cut hard-match cost at the same time:** the correct animal ranks first for 83% of bundles now. Facing-aware re-adaptation remains the representation lever.

## Limits

- **Simulated expert:** placeholder costs. A one-click rank-1 confirmation may take far less than 8 s, which would make the remaining work even more dominated by new-animal confirmation.
- **Training data:** the confidence model is cross-fitted on calibration (development only); a deployable one needs fitting-pool training.
- **Data and scope:** day-level dates; calibration development evidence only; no selection evaluation, count configuration or outreach.
