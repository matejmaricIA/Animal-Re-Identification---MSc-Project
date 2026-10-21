# CzechLynx clustering and photographed-individual count milestone

**Status: approved and implemented, 14 October 2026; 62 focused tests user-reported passed, full calibration completed and reviewed, reliable counts not achieved.** User direction: prioritise stronger clustering/count evidence before contacting parks. This supersedes the outreach-first sequence suggested after the project assessment. Reliable clustering and a defensible photographed-individual count remain the required outcome. Selection remains unrun. The [subsequent diagnostic](czechlynx-clustering-diagnostics-proposal-2026-10-14.md) was approved and implemented; its user-run tests/results remain pending.

**Immediate recommendation**

Build a dedicated full-collection clustering/count experiment using the existing frozen MiewID embeddings. Establish whether conservative clustering can substantially improve identity grouping and counts, then use its measured failures to guide learned pair scoring, local matching, night/IR treatment and fine-tuning.

The first implementation is deliberately bounded: a proper count-evaluation protocol, three global-only clustering controls and auditable outputs. It is the first experiment within the larger milestone, not a claim that these controls will achieve field reliability. Subsequent code changes will be proposed from its results.

**Evidence checked for this proposal**

- Current `tools/evaluate_czechlynx.py::run_clustering` clusters a bounded sample of query images. That is a grouping diagnostic, not a count experiment on the complete defined collection.
- Phase 2 provides identity-disjoint development calibration and selection pools. Use their full eligible gallery/query unions: **6,352 calibration images** (six existing undated exclusions remain explicit) and **4,725 selection images**. Unknown-to-gallery animals are included in discovery; the gallery identities are not given to the clustering algorithm.
- The matching frozen embedding cache is `data/czechlynx/cache/masked/global_4e16d8880ef85c0842a4.sqlite`. Read-only inspection found 11,953 vector IDs and zero missing IDs for either proposed collection. Cache specification hash: `4e16d8880ef85c0842a4addb1df1ea807809a30502275b7daaeb8315ae3d516e`. This was an inventory check; individual vector bytes will still require checksum/shape validation when loaded by the proposed tool.
- All **39,760 prepared images have orientation `unknown`**. The inspected raw CSV has no validated viewpoint field. Flank linkage therefore needs additional evidence; the first experiment must not claim validated opposite-flank handling.
- The full prepared metadata has **22 encounter IDs associated with multiple identities**. Encounter membership cannot be used as an automatic same-animal constraint. Truth was used to discover this metadata limitation, not to supply algorithmic links.
- No tests, inference or clustering benchmark was executed while preparing this proposal.

**Exact proposed source additions**

| File | Proposed responsibilities |
| --- | --- |
| `reid_demo/benchmark_clustering.py` | Read-only frozen-vector loading; full-collection protocol validation; label-free candidate and clustering functions; separate clustering/count evaluator; versioned calibration/configuration contracts; report data and HTML generation. |
| `tools/evaluate_czechlynx_clusters.py` | Dedicated CLI for calibration profiles and frozen selection evaluation; output manifests and run status; no neural extraction or missing-vector regeneration. |
| `tests/test_benchmark_clustering.py` | Focused synthetic regressions for protocol/cache failures, count semantics, weak bridges, unknown discovery, unresolved cases and frozen selection contracts. |

Documentation additions/updates: this proposal, benchmark command documentation, implementation tracker, decision log and AGENTS.md. The experiment reads existing artifacts and writes a separate report directory. It does not mutate the detection store or existing benchmark reports.

**Protocol and cache contract**

The CLI accepts a completed phase 2 global reference, `identity_pools_v1.json`, the prepared metadata and cache directory. It reconstructs the eligible collection from the plan, checking the reference role, exact ordered membership, existing exclusions and model/preprocessing specification. Calibration and selection have separate output directories and identities.

Load the existing SQLite cache with `mode=ro`, checking its specification fingerprint, each image-input hash, vector checksum, finite values and native dimensions. Remove the saved reporting-only `dimension` field when reconstructing the original cache specification. Missing or incompatible evidence is a failure with explicit IDs; the tool does not silently extract features or drop images.

Algorithms receive stable image IDs, normalized embeddings and permitted descriptive metadata, without ground-truth identities, target individual counts or the reference gallery identity catalogue. Identity truth enters only the separate evaluator. The discovery task includes images of unknown-to-gallery animals.

**The three controls**

1. **Existing neighbour-graph baseline on the full collection.** Reuse the current global-only clustering method to establish a like-for-like reference for the new collection. Report its actual Leiden/Louvain implementation explicitly. The historical query-sample result remains a separate reference.

2. **Absolute-threshold graph.** Construct candidate neighbours in deterministic chunks and admit only pairs exceeding a cosine threshold. Disconnected images are retained as singleton proposals. Candidate rank alone is insufficient to create an edge. This measures the effect of rejecting weak compulsory neighbours, while exposing remaining transitive bridge errors.

3. **Cluster-consistent merging.** Process candidate merges deterministically, requiring all cross-component image similarities to clear the chosen consistency threshold before merging components. Compute cross-component similarities on demand in bounded blocks; do not treat missing comparisons as support. This complete-link style control can block a single strong bridge from joining mutually incompatible groups. Its likely fragmentation cost must be measured.

The new controls use direct global cosine evidence, labelled as similarity rather than an identity probability. Calibration can profile a fixed, documented grid of candidate budgets and cosine thresholds. The grid is specified before looking at new outcomes. No `n_clusters` target is passed to any algorithm. Full-collection image sampling caps are rejected.

The first version uses images as nodes. Encounter membership contributes diagnostics and review context, not hard same-animal links. Future track or paired-camera evidence must distinguish multiple animals before aggregation. This limitation remains visible in the report.

**Calibration, selection and parameter choice**

Calibration produces configuration profiles and visual error evidence. Review them jointly for pair precision, pair recall, fragmentation, count error and coverage. A configuration should not be chosen solely because its cluster total happens to match the calibration truth.

Export an explicitly chosen configuration as a versioned `cluster_config.json`, bound to the calibration run, protocol, algorithm implementation and frozen model/input specification. Selection requires that configuration and evaluates it unchanged. It cannot produce threshold profiles, choose a better configuration using selection truth, or substitute the retrieval rejection threshold for a clustering threshold.

Both pools are development data and the selection pool has already been inspected for recognition. Results should be described accordingly. Repeated method development can consume selection's independence; a final untouched evaluation remains necessary. Preserve the official test partition and reserved temporal records.

**Outputs that make the count assessable**

- `manifest.json`: configuration, collection IDs, exclusions, input/model/code fingerprints, cache coverage, runtime and completion/failure status.
- `assignments.csv`: one row per evaluated image, provisional component ID, component size and evidence status. Evaluation truth is clearly separate from inference outputs.
- `edges.csv`: admitted candidate similarities and evidence about rejected component merges, with bounded, documented evidence retention.
- `metrics.json`: true collection count; provisional component count including singleton proposals; signed/absolute/relative count error; pair precision/recall/F1; ARI; mixed components; per-identity fragmentation and distribution of component sizes.
- Separate numbers for multi-image groups, singleton proposals, observations lacking corroboration across encounters and any explicitly unresolved observations. An unresolved photograph is neither a confirmed new animal nor a zero contribution to the count.
- Encounter and camera diagnostics, including whether an identity is recovered across distinct encounters/cameras and whether groups are dominated by repeated frames.
- Threshold-sensitivity plots from calibration, explicitly labelled sensitivity analysis rather than statistical confidence intervals or guaranteed count bounds.
- A portable HTML report with preselected examples of false merges, fragmentation, isolated images and cross-camera/appearance difficulties. Colour-proxy groups retain their existing limitations.

The raw component total is a provisional algorithmic count. A set of high-confidence-looking groups is not automatically a certified minimum because different groups may represent the same animal. Sensitivity across thresholds is not a complete uncertainty model. The report should make those distinctions clear rather than manufacture a precise count interval.

**Regressions to add**

Meaningful synthetic cases should include: two animal groups linked by one misleading edge; a genuine single-photo animal; all observations unresolved; two animals sharing an encounter; a false split and false merge that cancel in total count; missing/corrupt vectors; metadata/config edits; wrong model/input mode; and attempts to fit or profile on selection. Also verify deterministic results and equivalent candidate scores across chunk sizes.

Cache-only execution must import and run without loading neural models, and source SQLite cache contents must be preserved. The user runs the focused suite; command:

```bash
venv/bin/python -m pytest tests/test_benchmark_clustering.py tests/test_benchmark.py tests/test_benchmark_protocol.py -q
```

All three source additions now exist. [Calibration, explicit profile export and frozen selection commands](czechlynx-benchmark.md#full-collection-clustering-and-count-controls) use the existing masked cache and separate report directories. No test or benchmark was started by the assistant.

Implementation details: default calibration profiles are the existing k=15/resolution=1 baseline plus the threshold and consistent controls with k=15/50 and thresholds .35/.45/.55/.65/.75/.85 (25 profiles). Cosines use float64 arithmetic rounded to seven decimals. Complete-link checks reuse an incompatibility witness while components grow, preserving the consistency rule. Edge CSV retention defaults to 5,000 score-ordered decisions, with counts for every candidate. Undefined pair precision/recall denominators are reported as null; macro exact identity-group recovery also exposes pure but fragmented groups. A zero vector remains unresolved, and complete count error is withheld when any unresolved observations exist. New pixel diagnostics and all reports are written in the new output directory; source caches are read only.

**Remaining work within the required milestone**

After the first control results, propose the next measured improvement: calibrated joint global/local pair scores, better candidate diversity across encounters, region-based local evidence or backbone fine-tuning, depending on the observed failure categories. The prior RoMa diagnostic does not justify replacing the current matcher indiscriminately.

Create an auditable annotation path for flank and actual day/IR labels, with unknown permitted, then evaluate within-night and across-day/night identity connections and their consequences for clusters/counts. Use paired-view or track evidence for opposite-flank linkage where available; count uncertain linkage explicitly. Automatic annotation must itself be validated.

Count uncertainty, realistic reviewer burden and independent evaluation remain part of the final milestone. No numeric success tolerance is approved yet. The first deliverable supplies evidence to choose a defensible target without confusing image purity, provisional component totals or assisted matching with achieved population monitoring.

**Approval and handoff**

The user approved the three source additions and documentation updates. They have been implemented and statically reviewed. On 14 October 2026 the user reported the focused clustering/benchmark/protocol suite passed: **62 passed, 1 warning in 8.50s**. The warning concerns upstream LightGlue's deprecated `torch.cuda.amp.custom_fwd` decorator. The user then completed all 25 calibration profiles on 6,352 images / 55 identities in 39.497s; the [saved results and evidence receipt](research/czechlynx-clustering-calibration-results-2026-10-14.md) are reviewed. None provides reliable counts: relaxed thresholds collapse identities into giant mixed groups, while strict/consistent rules create thousands of fragments. Exact single-pure-group recovery is zero throughout. No configuration was exported; selection remains unrun. The user subsequently approved the [pair/connectivity diagnostic](czechlynx-clustering-diagnostics-proposal-2026-10-14.md), which is implemented with user-run validation pending. Neural extraction, new training, park outreach and a declaration of readiness were not part of this implementation. Tests and benchmarks remain user-run.
