# CzechLynx frozen review operating point on the reserved selection pool: results

Date: 19 October 2026. Scope: [approved proposal](../czechlynx-frozen-selection-proposal-2026-10-19.md) (work item (c)).

The frozen configuration was **pre-registered** in commit `97571909` before any selection photo was read:
- fingerprint `f24028291c0c02f9e901ee035ccc3ef23e1291695bdee15307ae893e5828951c`;
- [config copy](czechlynx-frozen-review-config-2026-10-19.json).

The one-shot run is in `evaluations/czechlynx/frozen_review_selection_v1/` (`index.html`, `summary.json`, `simulations.csv`, `per_animal.csv`). Selection: 4,725 photos of 55 lynx not used in any fitting or calibration step; 1,700 camera-day bundles.

## Conclusion

**The frozen operating point partly generalizes to unseen animals. Under the pre-declared criteria the evaluation fails S1, S2 and S4; only S3 is met.**

What held:
- **Automation is accurate:** 51.5% of bundles are decided automatically, with 0.80% of photos misassigned (calibration: 48.4%, 0.24%).
- **The relative time saving holds:** 59% fewer expert decisions than W1 on the same selection photos (calibration: 60%).
- **Self-assessment holds** with a perfect reviewer (S3): the automation-audit interval [0, 3.7%] contains the true bundle error (1.8%), and the count-audit interval [44, 61] contains 55.
- **Recognition improves on unseen animals:** protocol top-1 is 69.9% (public MiewID on the same selection, September: 63.8%); macro top-1 is 56.1% vs 47.3%.

What did not:
- **The count is off by +6** (61 groups for 55 animals). The surplus comes from **pruning**:
  - 10 declarations happened while the true animal was in the catalogue but below ε, against 3 on calibration;
  - the periodic duplicate checks merged 3 of them, leaving 7.
  - The surplus is concentrated in **snpa**: 34 groups for its 28 animals. foe_carpaths has a count error of 0 and foe_bohemia +1.

  The P(match) model, fitted on calibration animals, is too sure that some selection animals are "not this one". The likeliest suspects are its space-time features in that region: the risk of a miscalibrated location prior that the (a) results flagged. This is a hypothesis, not established.
- **The per-photo cost is higher** than on calibration: 453 vs 333 decisions per 1,000 photos, so S2 fails its absolute limit of 416. Selection is inherently costlier per photo: even W1 needs 1,110 decisions per 1,000 photos here vs 838 on calibration, because bundles are smaller (2.8 vs 3.5 photos per bundle).
- **Reviewer-error robustness fails** (S4): with 3% wrong answers, misassignment reaches 3.6% (count error 0). Six of the 10 duplicate merges were wrong. The duplicate pair question has no second check, a weakness already noted in (a).

| Criterion (declared before the run) | Selection | Calibration (cross-fitted) | Met |
| --- | --- | --- | --- |
| S1: ≤ 1% misassigned and \|count error\| ≤ 1 | 0.80%, **+6** | 0.24%, +1 | **No** |
| S2: ≤ 416 decisions per 1,000 photos and ≥ 50% saving vs W1 | **453**, 59.2% | 333, 60.3% | **No** |
| S3: both audit intervals contain the truth | Yes, both | Yes | **Yes** |
| S4: 3% reviewer error: \|count error\| ≤ 2 and ≤ 3% misassigned | 0, **3.6%** | +1, 1.9% | **No** |

Carried-over targets: misassignment ≤ 1% and audit coverage met; automation ≥ 70%, \|count error\| ≤ 1 and ≤ 0.5 h per 1,000 photos not met. The time is 1.01 h per 1,000 photos at placeholder costs and 0.47 h at fast-click costs.

## All four frozen simulations

| τ | Reviewer error | Automated | Misassigned | Count error | Exact /55 | Pruning duplicate events (unresolved) | Duplicate merges (wrong) | Decisions | h / 1,000 (placeholder / fast) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |
| **0.99 (primary)** | 0 | 51.5% | 0.80% | **+6** | 40 | 10 (7) | 3 (0) | 2,141 | 1.01 / 0.47 |
| 0.99 | 3% | 50.9% | 3.58% | 0 | 24 | 5 (3) | 10 (6) | 2,349 | 1.10 / 0.53 |
| 0.98 (secondary) | 0 | 56.5% | 1.31% | +6 | 37 | 10 (7) | 3 (0) | 2,052 | 0.97 / 0.46 |
| 0.98 | 3% | 55.7% | 5.25% | 0 | 17 | 7 (3) | 12 (8) | 2,348 | 1.10 / 0.54 |

**W1 reference** on the same photos, where the expert confirms every bundle with a full scan: 5,244 decisions, count 0, F1 1.000. With 3% reviewer error: 6,780 decisions, count +1, F1 0.881.

The count of 0 under 3% reviewer error is **not** a success: duplicates and wrong merges cancel (17 split errors, 14 merge errors). As before, a correct total can hide cancelling errors.

**Breakdowns** (primary, perfect reviewer):

| Subset | Photos | Count error | Misassigned |
| --- | ---: | ---: | ---: |
| snpa | 1,983 | +6 | 1.41% |
| foe_bohemia | 700 | +1 | 0.57% |
| foe_carpaths | 2,042 | 0 | 0.29% |
| lynx_287 only | 1,062 | 0 | 0% |

Without lynx_287 (3,663 photos), the errors are slightly worse (count +7, 1.04% misassigned), because all of lynx_287's photos are assigned correctly. The imbalanced animal does not cause the errors; it dilutes them. Nine animals are split across 2–3 catalogue entries; 40 of 55 are recovered exactly.

## Run and checks

- **Pre-registration:** the code, the proposal and a copy of the frozen configuration were committed (`97571909`) before the run. The runner verified the configuration fingerprint, code hashes, checkpoint, trial manifest, metadata, protocol and raw-metadata hashes before reading selection, and again at the end.
- **Freeze:** calibration only, 305 s. Models were fitted on all calibration animals: confidence on 1,790 replay records, P(match) on 56,987 records, facing on 1,115 pose-labelled photos. Frozen risk references were stored.
- **In-sample sanity on calibration:** 48.8% automated, 0.19% misassigned, count 0, 2,064 decisions. This only confirmed the plumbing.
- **Selection run:** `complete` in 173 s. F1 vectors for 4,725 photos were extracted into the new cache `data/czechlynx/cache/miewid_facing_classes_v1_selection/`, and the extraction was verified to use the frozen F1 weights. No retry; the one-shot lock now blocks any further run with this fingerprint.
- **Tests:** 10 new tests; 161 pass across eight related suites; Ruff F clean.
- **Nothing was retuned or rerun** on selection. No post-hoc selection analysis beyond the declared outputs.

## What this means

1. **Pruning is the fragile part.** On calibration, ε = 0.001 looked free: no extra duplicates for v2 and 3 for F1. On unseen animals it creates 10, and the duplicate checks recover only 3. A calibrated "not this animal" model trained on 55 animals does not transfer reliably, especially in one source. Any deployment needs either:
   - a more conservative pruning rule, such as a larger always-shown floor or ε chosen with a margin;
   - or a stronger, cheaper duplicate sweep.

   Choosing either needs **new held-out data**, because selection is now spent.
2. **The automation side generalizes.** The W4 gate and the embedding hold up on new animals: 51% automated at 0.8% misassigned, and the relative saving vs full manual review is stable at ≈ 60%.
3. **Duplicate merges need a second check.** Under reviewer error they cause most of the misassignment. This is a cheap safeguard fix.
4. **Absolute costs depend on the collection:** bundle sizes and animals per area. Per-photo targets should be stated relative to full manual review (W1) or measured per site, not as one number.
5. **Evidence status:** selection is spent. The official time_open test split remains locked. Any revised method needs a new evaluation set, for example park data with expert-verified identities, or a renewed protocol agreed with the user.

## Limits

- Selection is the same dataset as calibration: same parks, cameras, annotation-assisted masked crops and day-level dates. It tests new animals, not new sites, raw frames or real reviewers.
- Simulated reviewers and placeholder costs. MiewID's upstream exposure to CzechLynx is unresolved.
- No count configuration is approved for deployment, and there is no outreach.
