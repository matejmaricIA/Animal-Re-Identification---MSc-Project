# Proposal: confidence-gated matching with minimum expert review

Date: 18 October 2026. Status: **approved and run 18 October 2026: no configuration met all targets. Automation is accurate but saves little expert time; the top-K expert-path design caused large count errors (fixed in a post-hoc variant).** See [results](research/czechlynx-gated-matching-results-2026-10-18.md).

## Why

The user decided (see the decision log, "Requirement update — 18 October 2026") that expert-confirmed ranked matching can deliver reliable counts **only if the bulk of the work is automated**. The envisioned product is an all-in-one system: ingestion → empty/species classification → re-identification and counting. Expert review must shrink to a minimum while reliability is measured and reported.

The [review-effort simulation](research/czechlynx-review-effort-results-2026-10-18.md) showed that embedding-ranked catalogue matching (W1) reaches a verified 55/55 count with 5,283 expert decisions (≈1.8 h per 1,000 photos at placeholder costs), because the right animal ranks first 83% of the time. This proposal measures how much of W1 can be **decided automatically** without losing reliability, and what the residual errors and expert workload are.

## Read-only feasibility check (in-sample, indicative only)

Replaying W1 with a perfect reviewer and recording each bundle's best-match score s1 and its margin over the second-best catalogue animal:

| Auto-accept rule (top-1 match) | Bundles auto-assigned | Wrong assignments among them |
| --- | ---: | ---: |
| margin ≥ 0.20 | 41.5% | 0.27% |
| margin ≥ 0.15 | 53.3% | 0.42% |
| margin ≥ 0.10 | 64.9% | 1.37% |
| margin ≥ 0.05 | 77.2% | 5.75% |

- Margin separates correct from incorrect top-1 matches with AUC 0.934; s1 alone gets 0.877.
- "Known animal vs new animal" from s1 reaches AUC 0.87, not safe enough to auto-create animals.
- These numbers use the same data to choose thresholds. The experiment below cross-fits by animal and simulates sequentially, so that wrong automatic decisions contaminate the catalogue as they would in reality.

## Design: W4, confidence-gated ranked matching (calibration only, CPU)

Photos arrive in date order, bundled by camera and day, exactly as in W1. For each bundle the system computes label-free match features against the current catalogue:
- **Appearance:** best and second-best animal scores (s1, s2), margin, top-3 mean, bundle size, the bundle's identifiability risk, and predicted facing agreement with the matched animal's photos.
- **Space and time history:** has the top candidate been seen at this camera before, or in this location cell? Distance in km to its nearest previous sighting, where coordinates exist (88% of calibration photos). Days since it was last seen.

Then the policy:

1. **Auto-accept** the top candidate if a calibrated confidence model gives P(top-1 correct) ≥ τ, with τ ∈ {0.95, 0.98, 0.99, 0.995}.
2. **Otherwise, send the bundle to the expert queue.** The expert sees the top-K ranked candidates (K ∈ {10, 20}) and either picks one or declares a new animal. **New animals are never created automatically**, because every spurious new animal inflates the count. A true match ranked below K becomes a duplicate animal, which is measured.
3. **Duplicate sweep** at the end: the expert reviews the M most similar pairs of catalogue animals, M ∈ {0, 25, 50}, and merges duplicates.
4. **Audit sample:** the expert re-checks a random sample of auto-accepted bundles (n ∈ {100, 200}). From it the system **estimates its own error rate with a 95% interval**, which is what a researcher would rely on in practice. The simulation checks whether that interval covers the true error.

**Confidence model.** Logistic regression on the features above. Its training records come from the W1 perfect-review replay, and it is identity-cross-fitted: a bundle's confidence comes from a model never trained on that animal. The simulation then runs sequentially, so automatic mistakes contaminate later matching. This is development-only training on calibration; a deployable model would be trained on fitting-pool data, which needs adapted vectors for fitting images (GPU inference, separate approval).

**Sensitivity.** Reviewer error 0% and 3%, and space-time features on vs off (to show their value).

## Measurements

- Share of bundles decided automatically.
- Expert decisions by type, and hours per 1,000 photos at the same placeholder costs as before. Savings vs W1 (expert confirms everything) and vs W0 (unassisted).
- Misassigned photos, duplicate animals, merged animals, count error, pairwise F1 and exact recoveries.
- The audit's estimated error and 95% interval vs the true error.
- A trade-off curve of automation vs error across τ.

## Declared targets (the "useful and reliable" bar; adjust before approving if needed)

| Target | Rule |
| --- | --- |
| Mostly automatic | ≥ 70% of bundles decided without the expert |
| Reliable grouping | ≤ 1% of photos assigned to the wrong animal |
| Reliable count | \|count error\| ≤ 1 (of 55) |
| Large time saving | Expert effort ≤ 0.5 h per 1,000 photos at placeholder costs (W1: 1.8 h; unassisted: 10.5 h) |
| Honest self-assessment | The audit's 95% interval contains the true error rate |

A configuration meeting all five would be a **candidate operating point**. It would then need the reserved selection evaluation, real reviewer timing, and independent park data before any claim.

## Proposed code changes (new files only; no edits to existing source)

1. **`reid_demo/benchmark_gated_matching.py`** (~400 lines): match-feature computation including space-time history; W1 replay to collect confidence-training records; identity-cross-fitted confidence model; the sequential W4 simulation with auto-accept, expert queue, top-K new-animal check, duplicate sweep and audit; metrics, curves and report. It reuses `Catalogue`, `Reviewer`, `Effort`, bundling and `fast_quality` from `benchmark_review_simulation`.
2. **`tools/simulate_czechlynx_gated_matching.py`** (~150 lines): CLI with the usual contracts (calibration only, selection refused, empty output directory, input/cache hashes before and after). Coordinates come from the raw metadata, bound by SHA-256.
3. **`tests/test_benchmark_gated_matching.py`** (synthetic):
   - Features never read labels.
   - The confidence model never scores a bundle with a model trained on its animal.
   - Auto-accepted errors propagate to later matching.
   - New animals are never auto-created.
   - Top-K misses become duplicates, and the duplicate sweep merges them.
   - The audit interval arithmetic.
   - Selection is refused; no neural imports.

Untouched: existing source, data, models, caches, earlier outputs, and the reserved selection/test pools.

## How this fits the all-in-one system

This experiment covers the re-identification and counting core. The upstream stages already exist in the repository's `reid_demo` pipeline (ingestion, MegaDetector-based empty removal, species filtering, crops, review decisions and catalogue output; tickets T01–T12), plus the Medvednica glue scripts. They have not been validated end-to-end on raw lynx camera-trap footage. Once a core operating point is found, the next step is to integrate W4 behind those stages and measure the full raw-photo → verified-count path. Real camera IDs, exact timestamps and coordinates would make bundles purer and the space-time features stronger, so they belong in any park data request.

## Limits

- The simulated reviewer and placeholder costs are not real expert behaviour or timing.
- Calibration development evidence only.
- Day-level dates on CzechLynx.
- Night/IR and flank effects are inherited from the embedding, not separately validated.

## Runtime and who runs what

CPU only, estimated 10–20 minutes: one W1 replay, then about 2 × 4 × 2 × 3 × 2 sequential simulations plus the audits. Tests are synthetic and fast. No GPU or model loading.
