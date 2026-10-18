# CzechLynx candidate-coverage and identifiability diagnostic: results

Date: 18 October 2026. Scope: [approved proposal](../czechlynx-coverage-diagnostics-proposal-2026-10-18.md), run on calibration only (6,352 images, 55 identities), using the frozen FP32 v2 adapted vectors. Report: `evaluations/czechlynx/coverage_diagnostics_v2_fp32/index.html`.

## Conclusion

**Missing links can mostly be recovered, and the bottleneck moves to rejecting wrong links.** Adding within-encounter links connects 28/55 animals instead of 10/55. A post-hoc check (not in the declared plan) adding encounter links to a k=150 candidate list connects 44/55. The cost is that 64% of k=150 candidate edges join different animals. The next method change should therefore verify candidate edges (calibrated global + local evidence, false-bridge-resistant grouping) on a wider, encounter-linked graph. Retraining or abstaining on low-quality images should not be the next change.

Opposite flanks of the same animal almost never link by appearance. Unusable images are a minor cause of fragmentation, not the main one.

All numbers here use reference truth to remove wrong edges (oracle connectivity). They show what a perfect edge filter could reach on these graphs. They are not counts or achieved accuracy.

## Run and checks

- Run by the assistant with the user's explicit authorization (read-only, CPU-only). Status `complete` in 151.6 s. No models loaded, cache hash unchanged before and after.
- New tests: 21 passed. Combined with the existing clustering and cluster-diagnostic suites: 90 passed. Ruff F is clean.
- Self-check: the new candidate builder reproduces the calibration graphs exactly (k=15: 69,379 edges; k=50: 219,430 edges; max score difference 0). The oracle baseline matches the saved diagnostic: 10/55 connected, 187 fragments.

## Results against the declared rules

| Hypothesis | Declared rule | Result | Met |
| --- | --- | --- | --- |
| H1 unidentifiable images | ≤20% risk-abstention at least doubles connected animals vs random, and risk AUC ≥ .70 | AUC .73 vs human "usable markings: no". But at 20% abstention, risk-based dropping gives 10 connected vs 11.8 for random; at 30%, 14 vs 12.1 | No |
| H3 encounter links | ≥ +5 connected animals, ≤1% of encounter pairs join different animals | 10 → **28** connected, fragments 187 → 107; 0.81% of encounter pairs (184/22,752, from 2 encounters) join different animals | Yes |
| H4 wider candidates | ≥ +5 connected animals, precision drops by less than a third | k=100: 10 → **23** connected, precision .499 → .410. Query expansion (21 at k=100) and mutual kNN (8) do no better than plain kNN | Yes (plain k=100) |
| H2 flanks | Same-flank median cosine ≥ .15 above opposite-flank | Medians .149 vs .060 (gap .089) | No, but see below |

### Candidate variants (oracle, k-NN only)

| Variant | Connected /55 | Fragments | Direct-edge precision |
| --- | ---: | ---: | ---: |
| kNN k=15 | 3 | 342 | .637 |
| kNN k=50 (current) | 10 | 187 | .499 |
| kNN k=100 | 23 | 119 | .410 |
| mutual kNN k=100 | 8 | 184 | .515 |
| query expansion m=3, k=100 | 21 | 136 | .437 |

### Post-hoc: encounter links plus wider kNN

Not part of the declared plan; computed in a scratch script with the same module functions after the main run.

| k | Connected /55 | Fragments | Images without a correct neighbour | kNN precision |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 28 | 107 | 32 | .499 |
| 100 | 37 | 80 | 16 | .410 |
| 150 | 44 | 71 | 12 | .357 |
| 200 | 46 | 67 | 9 | .321 |

## Interpretation

**Flanks.** The declared cosine-gap rule was not met, but the link rate shows what matters for grouping. Among audited same-animal pairs, **29% of same-flank pairs** sit in each other's top-50, compared with **2.6% of opposite-flank pairs** (188 and 151 pairs). Opposite flanks essentially never link by appearance, as expected for lynx coat patterns. They can be joined through encounters that show both sides. That probably explains part of the encounter gain: 113 encounters span more than one camera, which is consistent with paired cameras.

**Image quality.** The label-free risk score agrees reasonably with the human audit (AUC .73 for unusable markings, .74 for crop/mask issues). Images outside each animal's largest fragment are only slightly riskier (AUC .57). Dropping the riskiest images does reduce fragments (138 vs 167.5 random at 30%), but it doesn't connect animals. The 16 score-extreme examples inspected before the run overstated this cause. The risk score is still useful later for an explicit *unresolved* bucket.

**Scale of the remaining gap.** Even at baseline, 6,100/6,352 images (96%) are in their animal's largest oracle fragment. The 0/55 exact-recovery result comes from a few hundred stray images spread across most animals.

## Caveats

- **Encounter IDs come from the dataset's `encounter` field**, not from timestamps. Timestamps here are day-only, and how the annotators defined encounters (and whether they used identity knowledge) is unverified. In a deployment, encounters would need to be derived from site + time windows, which needs full timestamps. Treat the encounter gain as an upper bound until derived encounters are tested.
- Two calibration encounters contain more than one animal (184 pairs). Treating encounter links as absolute would merge those animals, so grouping must allow them to be overruled.
- Audit cross-tabs use 193 purposive + uniform audited images. They are descriptive and cannot be read as collection rates.
- The post-hoc table was not pre-declared. Wider lists trade coverage for precision; no grouping or count was run.

## Next step (proposal needed)

Run grouping on a wider candidate graph with encounter links, and add a step that verifies candidate edges:

1. Candidates: kNN k=100–150 ∪ encounter links.
2. Edge evidence: calibrated global score plus local verification (DISK/LightGlue or RoMa) on candidate edges. This is the point where the thesis local/global combination should pay off. Encounter links count as strong evidence that can be overruled.
3. Grouping that resists false bridges, compared with the existing 25 controls on the same calibration pool.
4. Separately, check whether encounters can be derived from camera + time, without identities, on data with full timestamps.

Selection stays reserved and no count configuration is exported. Reliable counts are still unachieved.
