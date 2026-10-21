# CzechLynx calibration pair and connectivity diagnostic

Date: 14 October 2026. **Status: approved, implemented and user-validated: 41 focused tests passed in 5.19s; real diagnostic completed and reviewed.** The user explicitly approved this diagnostic scope after the [completed clustering calibration review](research/czechlynx-clustering-calibration-results-2026-10-14.md). The three presented source/test additions have been implemented. The [diagnostic review/receipt](research/czechlynx-pair-connectivity-results-2026-10-14.md) records candidate coverage, pair-score overlap and graph-propagation limitations; reliable counts remain unachieved. AGENTS.md requires prior approval for further code changes. Tests and benchmarks remain user-run.

## Purpose and decision to inform

All 25 global-only clustering controls fail the photographed-individual count outcome on calibration. The threshold .35 graph contains a 6,250-image component spanning all 55 animals; stricter or complete-link style controls create thousands of fragments. We need to distinguish:

1. **Candidate coverage:** can the existing k=15/50 graph connect the photographs of each animal when misleading edges are removed?
2. **Pair evidence:** how much do raw cosine scores of the same and different animals overlap, especially across encounters, cameras and appearance proxies?
3. **Graph propagation:** how much worse are component-level errors than direct-edge errors at the existing settings?

These answers determine whether the next method change should prioritise calibrated joint pair evidence and graph rules, better candidate/view coverage, or embedding adaptation. No diagnosis is predetermined. Scalar score calibration cannot create missing embedding information.

## Approved code changes

These are three additive files. The current embedding, retrieval, fusion, clustering and benchmark implementations are not edited as part of this proposal.

| File | Implemented responsibility and reason |
| --- | --- |
| `reid_demo/benchmark_cluster_diagnostics.py` | Add a calibration-report loader using existing source/artifact validation and immutable frozen-vector reads. Reconstruct the exact deterministic k=15/50 candidate unions without identity labels. Add a separate truth-based evaluator for direct-pair similarity distributions, existing-grid precision/recall, subgroup evidence, and true-edge candidate connectivity. Write checksummed diagnostic data and portable visual examples. This separates representation/candidate limitations from graph propagation without changing inference. |
| `tools/diagnose_czechlynx_clusters.py` | Add a standalone CPU CLI requiring one completed **calibration** clustering report and a separate output directory. Reject selection reports, output overlap and changed source/model/implementation/input contracts. Record completion/failure, timings, settings and hashes. Read existing reports/cache only; never regenerate embeddings or export a clustering configuration. |
| `tests/test_benchmark_cluster_diagnostics.py` | Add meaningful synthetic regressions: a single false bridge between otherwise connected animals; one animal whose correct candidate links are disconnected; strict-threshold fragmentation despite adequate unthresholded candidates; zero/unresolved vectors; undefined subgroup denominators; input-order invariance; changed artifacts/model/input bytes; selection rejection; and preservation of source caches/reports. Include a model-free CLI fixture. |

All three source additions exist. After static checks, the user reported 41 focused tests passed in 5.19s and completed the real diagnostic on all 6,352 calibration images / 20,170,776 unordered pairs in 14.913s. Saved evidence and current contracts are reviewed; implementation/run completion does not establish achieved counting reliability. Existing source files bound to the completed calibration report remain unchanged.

## Data and evaluation contract

Use only the **6,352 eligible calibration photographs** from the completed report. Reconstruct its original calibration source contracts and exact membership, retaining the six existing undated exclusions. The selection and official test collections are not diagnostic inputs.

The candidate builder receives stable IDs and normalized cached vectors, with the existing cosine rounding and deterministic tie policy. Identity truth enters only after candidate construction, in explicitly labelled evaluation functions. No target animal count or encounter-based same-animal constraint is supplied to candidate inference.

Existing edge CSVs contain only the first 5,000 score-ordered decisions per profile; they cannot represent all weak graph links. Reconstruct the full k=15/50 edge unions from the same frozen vectors, verify their candidate counts against the saved audits, and report all diagnostic aggregates. Do not imply that the retained edge sample is exhaustive.

Compute all unordered same/different image-pair score histograms in bounded blocks, without storing a dense matrix or extracting new features. Fixed bins and the already tested thresholds are declared before the diagnostic run. Report image-pair and macro-identity positive recall separately, with null values for undefined denominators. These measurements remain correlated and do not support invented confidence intervals.

## Outputs and interpretation

### Direct pair and candidate evidence

For each existing threshold and candidate budget, report same/different admitted-edge counts and direct-edge precision. Compare these with the existing saved component metrics. Compute all-pair score distributions and threshold precision/recall to expose ambiguous score regions. Include per-identity positive-score/recall summaries and within/across encounter, camera and appearance-proxy groups.

Identity-size imbalance and repeated photographs remain explicit. Colour/near-monochrome groups are proxies, not actual day/night/IR labels. Missing camera/appearance information remains unknown. Encounter membership does not generate true links.

### Connectivity with perfect truth-based edge filtering

For diagnostic evaluation only, remove every different-identity candidate edge and measure connected components **within each reference identity**, both before score thresholding and under the original threshold grid. Report how many identities are connected, per-identity fragment counts and images with no correct candidate neighbour.

This uses ground truth and is an **oracle diagnostic of the fixed candidate graph**, not a deployable clustering method, achieved accuracy or a bound on all future methods. It tells us whether a graph restricted to these candidates can recover an identity even with perfect filtering. It does not solve opposite-flank linkage, establish independent generalisation or supply count uncertainty.

### Review examples

Select deterministic, bounded examples of high-score different-identity candidate edges, low-score same-identity pairs and disconnected same-identity candidate groups. Save the exact selection policy, full endpoint IDs/scores and portable thumbnails. Use descriptive metadata for context; retain unknown flank and actual night/IR fields. Bounded examples do not replace a complete visual audit.

### Artifacts

Separate `manifest.json`, `summary.json`, pair/subgroup tables, per-identity connectivity tables, bounded pair-review examples and `index.html`. Bind the diagnostic to the completed calibration report, cached model/input specification, source bytes, diagnostic implementation and dependencies. Do not modify the completed calibration artifacts.

## User-run handoff

Recorded focused regression command (user reports 41 passed in 5.19s):

```bash
venv/bin/python -m pytest tests/test_benchmark_cluster_diagnostics.py -q
```

Recorded diagnostic command (completed and reviewed; no repeat needed for this result):

```bash
venv/bin/python tools/diagnose_czechlynx_clusters.py \
  --calibration-dir evaluations/czechlynx/cluster_calibration_global_v1 \
  --out-dir evaluations/czechlynx/cluster_pair_diagnostics_global_v1 \
  --threads 4 --chunk-size 64
```

Both commands are available; execution remains with the user. No additional threshold sweep, learned calibrator, local matcher run, fine-tuning or selection evaluation is included in this scope.

## Implemented evidence and output safeguards

- The CLI takes its cache path, candidate budgets and score thresholds from the completed calibration report. No selection, sampling, budget or threshold-override flags exist. Current defaults therefore diagnose k=15/50 and the original six thresholds; synthetic/custom source reports use their own already tested grid.
- Reconstruct the completed report's exact collection/protocol/model contracts and verify all recorded source artifacts. Source appearance measurements are read from the completed report; no new pixel measurements or appearance-cache writes occur.
- Load vectors through SQLite `mode=ro`, verifying current prepared image/mask bytes and vector checksums. Diagnostic provenance binds the source manifest/protocol, diagnostic code/dependencies and full SQLite byte checksum, plus the loaded matrix digest. Detect cache or source-manifest edits during execution.
- The full candidate unions must match every source control's candidate/zero-vector counts and retained ordered score/ID prefix. Evaluate every reconstructed edge; the 5,000-row source prefix is not treated as exhaustive evidence.
- Process all unordered pairs in bounded blocks. Histograms use fixed .02-wide bins over [-1, 1], including +1 in the final bin, and score arithmetic follows the existing seven-decimal cosine policy. Per-identity positive-score sums use integer quantized units so block order does not change them.
- Zero vectors remain unresolved. Histograms exclude unscorable pairs, but their same-identity pairs remain in recall denominators, including macro positive recall. Zero denominators yield null; eligible identity counts are exposed. Oracle complete-fragment totals are null if any unresolved observation remains. True single-photo identities and connected multi-photo identities are reported separately.
- Output requires a new/empty directory, protects source reports, cache, prepared images/masks and the protocol file, and records failed runs after report creation. Retry failed runs in a new directory. Existing diagnostic reports cannot be overwritten.

Artifacts include `manifest.json`, `summary.json`, `pair_thresholds.csv`, `pair_histograms.csv`, `identity_pair_statistics.csv`, `identity_pair_histograms.csv`, `candidate_thresholds.csv`, `oracle_connectivity.csv`, `component_comparisons.csv`, `review_pairs.json`, `pair_score_histograms.png`, a portable `index.html` and checksummed thumbnails. Review examples select at most eight pairs per kind using fixed score/ID order; disconnected examples use unthresholded truth-filtered components at the largest saved candidate budget. These are diagnostic extremes, not a representative visual audit. Night/IR labels remain unverified.

## Subsequent decision

The [completed review](research/czechlynx-pair-connectivity-results-2026-10-14.md) finds both missing true paths and misleading edges that propagate through components. Unthresholded k50 oracle connectivity recovers complete groups for only 10/55 identities, and high raw-cosine precision entails very low positive recall. Prioritise a source/identifiability audit, improved representations across views, candidate diversity and calibrated pair/robust graph evidence. Further method code needs its own concrete presentation and approval.

Only after defining and choosing an improved method using calibration should its configuration be frozen for selection. Reliable counts, upstream provenance, night/IR and flank linkage, uncertainty, reviewer effort and appropriately independent validation remain open.

