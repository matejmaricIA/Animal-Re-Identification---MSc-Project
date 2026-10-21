# Proposal: candidate-coverage and identifiability diagnostic

Date: 18 October 2026. Status: **approved and run, 18 October 2026**. See [results](research/czechlynx-coverage-diagnostics-results-2026-10-18.md).

## Why this is the next step

The adapted (FP32 v2) model improved recognition, but grouping still recovers zero animals as one clean group. The oracle diagnostic is the binding constraint: with every wrong link removed using reference truth, the k=50 candidate graph fully connects only **10/55** calibration identities (187 fragments). Score calibration, local verification or a different clustering rule can only accept or reject links that are already in that graph. So the next question is **why correct links are missing**, and which label-free change would add them.

Candidate explanations, each with a different fix:

| Hypothesis | If true, the fix is |
| --- | --- |
| H1. Many photographs cannot support identification (partial, frontal/rear, washed-out IR, broken masks) and become singleton fragments | An explicit *unresolved* bucket / quality gate; count from identifiable images, report unresolved ones separately |
| H2. Opposite flanks of the same animal never link | Flank-aware grouping: separate flank catalogues joined only through encounters showing both sides |
| H3. Photos from one encounter are split apart | Add label-free within-encounter links before grouping |
| H4. The top-k neighbour list is too narrow or noisy | Candidate expansion (larger k, reciprocal neighbours, query expansion) |
| H5. None of the above; the representation itself misses same-flank, good-quality pairs | Local/global evidence (DISK/LightGlue, RoMa) on fragment-boundary pairs, or further adaptation |

## What the saved evidence already suggests (not a conclusion)

I inspected the 16 saved review examples in `miewid_adaptation_calibration_v2_fp32/pair_diagnostics/review_pairs.json` (8 "disconnected same identity" and 8 "lowest-score same identity" pairs). These are **deterministic score extremes, not a representative sample**.

- None is a clear left-side vs right-side pair. Most involve a frontal or rear view, a zoomed partial crop, a washed-out or near-black IR image, a tiny crop, or a fragmented mask. One washed-out `lynx_086` IR image appears in five of the eight lowest-score pairs.
- One pair (`lynx_036`, same camera `snpa-52`, both clear full side views facing the same way) scores 0.04. That looks like a genuine model miss on a usable pair (H5).

This shifts the emphasis from flanks (H2) toward identifiability (H1), but it is not evidence. H2 is still tested directly below, using the human audit's flank labels.

## Existing data this can use without new labelling

- Frozen adapted calibration vectors: `data/czechlynx/cache/miewid_adaptation_v2_fp32/global_84e030d67653ff177f2e.sqlite`, reached through the completed report `evaluations/czechlynx/miewid_adaptation_calibration_v2_fp32/clustering`.
- Saved masks/crops and metadata (`bbox`, `width`, `height`, `encounter_id`, camera, date).
- The completed human audit (`evaluations/czechlynx/input_audit_review_v1/reviews.json`, receipt-bound): **193 of 321 reviewed images are calibration images**, with flank (88 right / 57 left / 48 unknown), visibility (98 partial / 87 clear / 8 no clear animal), usable markings (147 yes / 44 no), crop/mask (30 issues) and illumination. These are purposive plus uniform cohorts, so results from them are descriptive only and cannot be read as whole-collection rates.

## Proposed code changes (three new files, no edits to existing source)

### 1. `reid_demo/benchmark_coverage_diagnostics.py` (new, ~350 lines)

Reuses `load_calibration`, `check_diagnostic_output`, `prepare_evaluation`, `oracle_connectivity` from `benchmark_cluster_diagnostics.py`, and `candidate_neighbours` / `load_frozen_vectors` from `benchmark_clustering.py`. Calibration partition only; selection is refused by the existing loader. Every edge set is built without identities; truth enters only in the evaluators.

- `image_quality_features(frame)`: model-free, per image, from saved crop + mask: mask area fraction, number of mask connected components and the largest component's share, bbox touching the original image border (truncation), crop short side in pixels, luminance mean/std inside the mask (washed-out or near-black), plus the existing near-monochrome proxy.
- `identifiability_risk(features)`: a **fixed formula declared now**, not fitted to calibration truth: the mean of percentile ranks of (small crop, low in-mask contrast, fragmented mask, truncated bbox). Higher = less likely identifiable.
- `fragment_tables(inputs, labels, features)`: per-image rows (oracle fragment ID, in-largest-fragment flag, has-a-correct-candidate-neighbour flag, features, encounter, camera, date) and per-fragment rows (size, cameras, encounters, date span, mean risk). This is the membership detail the current reports don't save.
- `abstention_curve(inputs, edges, risk, fractions=(0, .05, .10, .20, .30))` (H1): drop the highest-risk images (a label-free ordering), then recompute oracle connectivity on the rest. Paired with a **random-drop control** at the same fractions (fixed seeds, 20 repeats) so a gain isn't just from having fewer images. Reports fully connected identities, fragments, images abstained, and identities left with no images.
- `encounter_edges(inputs)` (H3): within-encounter links built from metadata only. Reports connectivity of candidates ∪ encounter edges, plus how many encounter links join *different* reference identities (the false merges this rule would cause).
- `expanded_candidates(ids, matrix)` (H4): label-free variants: k=100; mutual/reciprocal kNN at k=50/100; database-side query expansion (each vector replaced by the renormalised weighted mean of itself and its top m=3/5 neighbours, then re-ranked). For each variant: oracle connectivity **and** the number and precision of direct candidate edges, because more edges also mean more potential false bridges.
- `audit_crosstabs(per_image, reviews)` (H1/H2, on the 193 audited images): has-correct-neighbour rate and largest-fragment membership by flank, usable markings, visibility, crop/mask and illumination; same-identity cosine for same-flank vs opposite-flank vs unknown pairs; and the risk score's AUC against `usable_markings = no` (checks the label-free proxy against human judgement).
- `write_coverage_report(...)`: CSVs, `summary.json` and `index.html`, with fragment thumbnails for the eight most-fragmented identities.

### 2. `tools/diagnose_czechlynx_coverage.py` (new, ~110 lines)

A CLI following `tools/diagnose_czechlynx_clusters.py`: same manifest/status/failure handling, cache before/after hash check, and an empty-output-directory requirement. Arguments: `--calibration-dir`, `--audit-reviews` (validated against the existing receipt hash), `--out-dir`, `--threads`, `--seed`.

### 3. `tests/test_benchmark_coverage_diagnostics.py` (new)

On small synthetic graphs and masks:

- Abstention removes the intended images, and the random control is deterministic for a given seed.
- Encounter edges are counted correctly and a mixed-identity encounter is reported as a false link.
- Mutual kNN and query expansion are label-free (permuting identities leaves the edges unchanged).
- The quality features detect fragmented masks, truncation and washed-out crops.
- Cross-tabs handle `unknown` labels.
- Selection-partition input and non-empty output directories are refused (existing guards).

Untouched: the existing source, data, masks, identities, splits, caches, the adapted model, the audit, the failed v1 evidence and the reserved selection/test pools. No model inference and no training.

## Decision rules, declared before running

These are proposed defaults; adjust them before approving if you want different bars.

| Result | Interpretation and next step |
| --- | --- |
| Abstaining on ≤20% highest-risk images at least doubles fully connected identities vs the random-drop mean, and the risk AUC vs `usable_markings = no` is ≥ 0.70 | H1 supported: build an explicit unresolved bucket and quality gate into grouping, and report counts as photographed individuals among identifiable images, with unresolved images listed |
| Encounter links add ≥ 5 fully connected identities with ≤ 1% of encounter links joining different identities | H3 supported: attach photos via encounters in the grouping step |
| A candidate variant raises fully connected identities by ≥ 5 while direct-edge precision drops by less than a third | H4 supported: adopt it in the candidate builder for the next clustering calibration |
| Opposite-flank same-identity median cosine is ≥ 0.15 lower than same-flank (descriptive, audited images only) | H2 supported: flank-aware grouping becomes the priority. Otherwise, deprioritise it |
| None of the above | H5: run local/global evidence (RoMa, DISK/LightGlue) on fragment-boundary pairs, starting with clear same-flank misses like `lynx_036` |

None of these outcomes produces a count configuration, touches the selection pool or justifies outreach. They choose the next method change, which goes back through clustering calibration.

## Expected cost

Reading 6,352 masks plus a few kNN passes over 6,352 × 2,152-D vectors: about one to three minutes on CPU, with no GPU and no model loading. Tests use synthetic data and run in seconds.

## Commands (after implementation)

```bash
venv/bin/python -m pytest tests/test_benchmark_coverage_diagnostics.py tests/test_benchmark_cluster_diagnostics.py -q
```

```bash
venv/bin/python tools/diagnose_czechlynx_coverage.py \
  --calibration-dir evaluations/czechlynx/miewid_adaptation_calibration_v2_fp32/clustering \
  --audit-reviews evaluations/czechlynx/input_audit_review_v1/reviews.json \
  --out-dir evaluations/czechlynx/coverage_diagnostics_v2_fp32 --threads 4
```
