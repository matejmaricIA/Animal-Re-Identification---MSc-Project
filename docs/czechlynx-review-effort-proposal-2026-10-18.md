# Proposal: measure the expert review effort needed for a verified lynx count

Date: 18 October 2026. Status: **approved and run 18 October 2026: grouping-first does not save work; ranked catalogue matching is the most efficient verified workflow.** See [results](research/czechlynx-review-effort-results-2026-10-18.md).

## Why

Three rounds of automatic grouping on the CzechLynx calibration pool reached pairwise F1 of about 0.60 at best, with 0–1 of 55 animals recovered exactly ([edge grouping](research/czechlynx-edge-grouping-results-2026-10-18.md), [flank grouping](research/czechlynx-flank-grouping-results-2026-10-18.md)). The limit is the embedding's same-side discrimination; neither graph engineering nor local features (as tested) removed it.

An expert-reviewed workflow produces a verified grouping and count by construction. The open question is **how much expert work** that takes, and whether the automatic grouping actually saves work compared with how lynx photos are identified today. This experiment measures that on calibration data, so the user can decide whether "reliable after bounded review" satisfies the demo requirement. It does not decide that question, and it does not relax the requirement for reliable clustering and counts.

## What is simulated

A **simulated expert** answers review questions using the reference identities. This is the standard way to cost a human-in-the-loop workflow before real users exist, and the thesis HITL-NIS experiments did the same. Truth is used only to answer questions the workflow asks; the workflow never sees labels otherwise.

Three workflows are compared on the same 6,352 calibration photos (55 animals):

**W0 — catalogue matching, unassisted (today's manual baseline).**
- Photos arrive in date order, bundled by trap-day (photos from one trap on one day form one unit, as a person would review them).
- For each bundle the expert compares it with the existing catalogue animals in arbitrary order until a match is found, or creates a new catalogue entry.
- Cost: number of bundle-vs-animal comparisons.

**W1 — catalogue matching, ranked (assisted matching).** As W0, but catalogue animals are ordered by embedding similarity (max cosine between the bundle and the animal's photos). Cost: the rank of the correct animal, or the full catalogue size for a new animal.

**W2 — group first, then correct (grouping-assisted review).** Start from a fixed automatic grouping, then:
1. **Split check.** For each proposed group, the expert reviews its trap-day bundles against the group's majority animal. Each bundle is one glance, and wrong bundles are moved out (a bundle containing two animals costs one extra decision). The moved-out bundles are then placed by the W1 ranked catalogue procedure.
2. **Merge check.** Corrected groups are paired in order of aggregated edge probability (mean cross-edge log-odds from the cross-fitted edge model). The expert answers "same animal?" for each pair until a stopping rule: K consecutive "no" answers, K ∈ {10, 25, 50}.
3. **Residual errors** left by the stopping rule are reported, not hidden.

W2 runs on three fixed starting groupings, rebuilt deterministically from the saved code and seeds:
- Edge grouping best (derived, k=100, G2 average-linkage τ=0.5).
- Flank grouping best (derived, k=100, assign, Louvain 1.0, m=1).
- The legacy saved control (k=15 Louvain), as a weak reference.

## Measurements

- **Decisions** by type: bundle glances, pairwise "same animal?" questions, catalogue comparisons.
- **Estimated expert time** from a declared cost model: 5 s per bundle glance, 10 s per pairwise question, 8 s per catalogue comparison. These are placeholders; results also report raw counts so the user or a park expert can apply their own costs.
- **Final quality** after review: pairwise precision/recall/F1, exact recoveries, count error, and residual merges and splits.
- **Effort–quality curve** for W2: quality as a function of decisions spent (W2 stopping at intermediate budgets).
- **Per 1,000 photos** normalisation, plus the split of effort between split checks and merge checks.

## Sensitivity checks

- **Reviewer error.** A perfect reviewer, and one answering 3% of pairwise questions wrongly at random (fixed seed), to show how the workflow degrades.
- **Unidentifiable photos.** Real experts cannot identify every photo: the human audit marked 55/128 uniform fitting-sample photos as lacking usable markings. Variant: bundles whose photos all fall in the top 20% identifiability risk cost one glance and are set aside as **unresolved** rather than matched. They are reported, not counted.
- **Bundling.** Trap-day bundles vs single photos (no bundling), to show how much the day-level metadata alone saves.

## Proposed code changes (new files only; no edits to existing source)

1. **`reid_demo/benchmark_review_simulation.py`** (~350 lines):
   - Bundle construction.
   - The simulated reviewer (perfect / noisy).
   - The W0/W1 sequential catalogue simulations.
   - The W2 split and merge phases.
   - Cost model, effort–quality curves and report.
   - Rebuilding the three starting groupings, reusing `benchmark_edge_grouping` / `benchmark_flank_grouping`.
2. **`tools/simulate_czechlynx_review.py`** (~150 lines): CLI with the same contracts as earlier tools (calibration only, selection refused, empty output directory, cache/manifest hashes checked before and after).
3. **`tests/test_benchmark_review_simulation.py`**, on synthetic collections:
   - W0/W1 cost arithmetic on hand-computed cases.
   - The ranked catalogue never uses labels to order candidates.
   - Split check moves exactly the wrong bundles.
   - The merge stopping rule.
   - A noisy reviewer is reproducible for a fixed seed.
   - A perfect reviewer with no stopping limit reaches the true grouping.
   - Selection is refused; no neural imports.

Untouched: existing source, data, models, caches, earlier outputs, and the reserved selection/test pools.

## Declared outcome rules

| Outcome | Rule |
| --- | --- |
| Grouping saves expert work | W2 (best starting grouping, perfect reviewer, K=25) reaches pairwise F1 ≥ 0.98 and \|count error\| ≤ 1 with **≤ 50%** of W1's decisions |
| Strong saving | Same, with ≤ 25% of W1's decisions |
| Robust to reviewer error | With 3% pairwise errors, W2 still reaches F1 ≥ 0.95 |

No rule decides what absolute effort is acceptable for a park. The report states expert-hours per 1,000 photos under the declared costs, and the acceptable tolerance remains the user's decision, along with whether bounded review satisfies "reliable counts".

## Limits

- A simulated reviewer is not a real expert: real decisions take variable time, and some "same animal?" questions are genuinely unanswerable from the photos. The unresolved-photo variant only approximates this.
- CzechLynx bundles use day-level dates; real parks may have exact timestamps (smaller, purer bundles) or different camera designs.
- Calibration-pool development evidence only; no selection evaluation, count configuration or outreach.

## Runtime and who runs what

CPU only, estimated 5–15 minutes: rebuilding three groupings plus the sequential simulations over about 1,800 trap-day bundles (1,791 counted earlier). Tests are synthetic and fast. No GPU or model loading.

## Related

`nested_importance_sampling.py` (thesis HITL-NIS) estimates a count with uncertainty from a *sample* of human judgments, without reviewing everything. If W2's full-review cost turns out too high, a sampled estimate with an interval is the natural follow-up and could reuse this simulator.
