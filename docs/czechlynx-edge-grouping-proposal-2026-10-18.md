# Proposal: grouping on a wider, encounter-linked graph with calibrated edge verification

Date: 18 October 2026. Status: **approved; Stage A and the Stage B pilot run 18 October 2026; the local-evidence gate failed, so the full Stage B run is not recommended**. See [results](research/czechlynx-edge-grouping-results-2026-10-18.md).

## Why

The [coverage diagnostic](research/czechlynx-coverage-diagnostics-results-2026-10-18.md) showed that most missing links can be recovered. Encounter links plus a kNN k=150 candidate list connect 44/55 calibration animals under the oracle, up from 10/55. But 64% of those candidate edges join different animals. The next method has to **decide which candidate edges to trust** and **group without letting a few wrong edges merge animals**. This is where the thesis local + global combination should prove itself, or be shown not to help.

## Evidence that constrains the design

- **Local evidence has been weak so far.** In the saved pairwise experiment (1,000 hard top-10 shortlist pairs, pre-adaptation ranking), DISK+LightGlue homography inlier counts separate same from different animals with AUC 0.70 (median 6 vs 5 inliers), and RoMa is at chance (AUC 0.45). A single homography is a poor model for a deforming animal, and very few inliers survive. Local evidence therefore has to show **added** value over cheaper signals before it is computed on every edge.
- **Cost.** DISK+LightGlue runs at about 0.026 s per pair on the GPU. k=100 ∪ encounters is about 420k edges (≈3 h); k=150 is about 608k (≈4.4 h). That is affordable, but only once a pilot shows a benefit.
- **Encounters can't yet be derived as a deployment would.** CzechLynx has no EXIF times (0/60 sampled originals) and only day-level dates. Its annotated encounters are finer than trap-day groups (18,782 encounters vs 10,540 trap-day groups), and 113 calibration encounters span cameras. Medvednica has second-level timestamps but no camera IDs or identities. So the deployable encounter test here is limited to **day-level** derivation; time-window encounters need park data with full timestamps.
- **Unused resource.** The raw CzechLynx metadata contains left/right pose keypoints for 5,206 images, which preparation dropped. They could give flank labels later. Not used in this scope.

## Design

Calibration pool only (6,352 images, 55 animals), frozen FP32 v2 vectors. Selection stays reserved. Two stages, with the expensive one gated.

### Stage A: label-free edge features, calibrated edge model, grouping (CPU)

1. **Candidate graph:** kNN k ∈ {100, 150} ∪ encounter links, built with the existing coverage-diagnostic builder.
2. **Encounter arms.** The same pipeline runs with two kinds of encounter evidence:
   - **E-annotated:** dataset `encounter` field (upper bound; not deployable as-is).
   - **E-derived:** label-free same trap + same date, and same location cell + same date (deployable at day level). For each, report the oracle gain and the share of within-group pairs joining different animals, alongside the annotated encounters.
3. **Edge features (no identities):** cosine score; rank of b in a's list and of a in b's; mutual-rank indicator; k-reciprocal shared-neighbour Jaccard (k=20, a standard re-ID re-ranking signal); same camera; same trap-day; same location-day; annotated-encounter flag (E-annotated arm only); appearance-proxy pair type (colour/near-monochrome); both endpoints' identifiability risk from the coverage diagnostic.
4. **Edge model:** logistic regression giving P(same animal) per edge, **cross-fitted by identity** (5 folds; each edge is scored by a model that never saw either endpoint's animal). Report out-of-fold AUC, calibration (reliability bins, Brier score), and AUC within the ambiguous global-score band.
5. **Grouping algorithms** on the out-of-fold edge probabilities:
   - **G1 threshold:** union-find on p ≥ τ (the current control style, but on calibrated p instead of raw cosine).
   - **G2 average-linkage:** merge two groups only if the **mean** log-odds over all their candidate cross-edges is ≥ τ, with at least s supporting edges. One wrong edge can't merge two animals.
   - **G3 Leiden (CPM)** on positive log-odds weights, using the existing Leiden dependency, over a small resolution grid.
   - Small grid: τ ∈ {0.5, 0.7, 0.9} (G1/G2), s ∈ {1, 3} (G2), and 3 resolutions (G3). About 15 profiles per arm.
6. **Evaluation** reuses `evaluate_clustering` (pairwise precision and recall, merge and split errors, exact single-group recovery). It adds the predicted group count, the count excluding singleton groups, and **corrections needed** (merge errors + split errors) as a review-effort measure. Every profile is compared with the existing 25 controls on the same calibration pool.

### Stage B: local evidence on candidate edges (GPU, gated)

1. **Pilot:** DISK extraction on calibration images (reusing the existing `LocalFeatures` cache), then LightGlue on a **stratified 20k-edge sample**: all score bands, both encounter arms, positives and negatives. About 10–15 min GPU.
2. **Local features per edge:** raw LightGlue match count and mean match confidence (no RANSAC); homography inliers and coverage (existing `geometric_evidence`); inliers under a looser **partial-affine** model. The looser models address the deformation problem behind the weak inlier counts.
3. **Gate:** local evidence goes to the full candidate graph only if adding it to the Stage A edge model raises out-of-fold AUC by ≥ 0.02 overall **or** ≥ 0.05 in the ambiguous band. If it fails, the result is a documented negative for the thesis local/global question on this data, and grouping proceeds with Stage A features.
4. **Full run (only if the gate passes):** all edges of the chosen graph, then Stage A's model and grouping are rerun with local features added.

## Proposed code changes (new files only; no edits to existing source)

1. **`reid_demo/benchmark_edge_grouping.py`** (~450 lines): candidate graph assembly with encounter arms (reusing `knn_edges` and `oracle_components` from the coverage module); label-free edge features; identity-cross-fitted logistic edge model (scikit-learn, already a dependency); G1/G2/G3 grouping; evaluation via `evaluate_clustering`; and the report (HTML, CSVs, reliability plot).
2. **`tools/evaluate_czechlynx_edge_grouping.py`** (~150 lines): Stage A CLI (calibration only, refuses selection, empty output directory, cache/manifest hashes before and after). Optional `--local-evidence <file>` to add Stage B features.
3. **`tools/extract_czechlynx_edge_local.py`** (~180 lines): Stage B CLI. It takes an edge list (pilot sample or full graph), reuses `LocalFeatures`, `DiskLightGlueMatcher`, `EvidenceCache` and `geometric_evidence`, adds the partial-affine count, and writes a per-edge local-evidence file with model/code fingerprints. It can resume from its evidence cache.
4. **`tests/test_benchmark_edge_grouping.py`** and **`tests/test_extract_czechlynx_edge_local.py`**. Synthetic-data checks:
   - Features are identity-free (permuting labels leaves them unchanged).
   - Cross-fitting never scores an edge with a model that saw its animals.
   - G2 refuses a single-bridge merge that G1 accepts.
   - Derived encounters group by trap/location-day correctly.
   - Selection is refused.
   - The Stage B CLI works with a mocked matcher and imports no neural code until it is constructed.

Untouched: existing source, data, masks, splits, adapted model, caches, audit, failed v1 evidence and the reserved selection/test pools.

## Declared outcome rules (development evidence on calibration, not final validation)

These are defaults; adjust them before approving if needed.

| Outcome | Rule | Next step |
| --- | --- | --- |
| Edge model works | Out-of-fold AUC on candidate edges ≥ 0.90 in the E-derived arm | Use it for grouping |
| Local evidence helps | Pilot ΔAUC ≥ 0.02 overall or ≥ 0.05 in the ambiguous band | Run Stage B on the full graph |
| Grouping progress | Some profile reaches pairwise precision ≥ 0.95, recall ≥ 0.70 and ≥ 15/55 exact recoveries (currently 0) | Refine that family |
| Count candidate | Additionally, non-singleton groups within ±10% of 55 and ≤ 2 merge errors | Freeze one configuration, then the reserved selection evaluation can be proposed |
| Day-level encounters are deployable | E-derived reaches ≥ 80% of E-annotated's exact recoveries for the best profile | Day-level linking is enough for a pilot; otherwise park data with real timestamps is a prerequisite |

None of these outcomes justifies outreach or a count claim. A count candidate on calibration still needs the frozen selection evaluation and independent park data.

## Runtime and who runs what

- **Stage A:** CPU, estimated 5–15 min including the cross-fitting and about 30 grouping profiles (two encounter arms × ~15).
- **Stage B pilot:** GPU, about 15–25 min including DISK extraction.
- **Stage B full:** GPU, 3–4.5 h, only if the gate passes.

Tests are fast and synthetic. Stage B loads neural models (DISK, LightGlue), which goes beyond the read-only authorization given for the coverage diagnostic. It needs its own go-ahead or a user run.
