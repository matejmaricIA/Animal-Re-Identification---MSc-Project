# CzechLynx phase 2 implementation proposal

Status: approved and implemented on 16 September 2026; the user reports 106 focused tests passed, 1 upstream LightGlue warning, in 5.93s. The user-generated protocol, calibration and frozen selection benchmarks have been reviewed. The phase 2 global reference is complete; pilot reliability is not established. This phase establishes independent development evaluation and diagnostics. It does not promise an accuracy increase.

## Motivation

The controlled segmented benchmark gives global/global+Fisher/full-funnel top-1 of 156/155/153 out of 303. Geometric reranking fixes 16 and breaks 18 matches. Cross-appearance correctness remains 5/27, and the current full-funnel graph produces eight mixed clusters for 60 individuals with 7.42% pair precision.

The next experiments need to learn when component scores are reliable and when an image should remain unresolved. Fitting those decisions and reporting their accuracy on the same queries would overstate progress. The first deliverable is a frozen division of official training data for fitting, calibration and model selection.

A metadata-only feasibility inspection of time_open official train found 27,587 images, 275 identities and 262 indivisible identity groups after linking identities sharing encounters, source-image hashes or prepared-image hashes. The largest group contains six identities. There are 21 missing timestamps. Source-image totals are 9,978 SNPA, 12,122 FoE Carpathians and 5,487 FoE Bohemia. This is a feasibility inventory, not a completed split or a benchmark result.

## Data contract

Use three identity-disjoint development pools, approximately 60% fitting, 20% calibration and 20% selection by identity count. Group constraints mean exact percentages are not guaranteed. Assignment is deterministic under a seed, stable to row ordering, and aims to balance source coverage; actual counts and deviations are explicit. Source balancing does not establish geographic independence.

- **Fitting pool:** PCA/GMM fitting now and supervised backbone training in a later phase. Its animal identities cannot appear in either calibration or selection.
- **Calibration pool:** separate gallery/query encounters, with known queries and entire held-out identity groups absent from its gallery. This pool will support score-calibrator fitting in phase 3; phase 2 uses it to choose the existing rejection thresholds.
- **Selection pool:** its own gallery/query encounters and unknown identities. It uses frozen fitting artifacts and calibration thresholds. It never fits thresholds or calibrators. It is development validation for comparing methods, not a final untouched test after repeated model selection.
- **Official test:** retains the dataset's original assignments and is never used to construct or adjust the three development pools. The 1,614 images dated after the v3 upload remain reserved.

Connect identities sharing an encounter or nonmissing source/image hash before assigning pools. Within calibration and selection, connect related encounters before making gallery/query assignments so a duplicate crop or shared encounter cannot leak across the boundary. Mixed-animal groups remain atomic; no identity is partially removed to satisfy a nominal count. No hidden identity/image caps are applied.

Within each calibration/selection pool, target approximately 20% of eligible identity groups as unknown, with all their retained encounters on the query side. For known identities, choose later eligible encounter groups for queries, targeting 25%, while preserving earlier gallery support. Validate the resulting known/unknown inventory, group integrity and nonempty required subsets. If constraints cannot be satisfied, stop with an explicit explanation instead of silently weakening them.

Missing dates are not fabricated. Undated records may be used for fitting if their identity group belongs to the fitting pool; exclude and list them when constructing temporal calibration/selection gallery/query partitions. Group connections are computed before these exclusions so exclusions cannot conceal cross-pool relationships.

The protocol artifact contains metadata SHA-256, algorithm version, seed, requested fractions, all role-specific IDs, excluded IDs with reasons, group membership, source/identity/encounter counts, dates, known/unknown composition and a protocol fingerprint. This protects against leakage introduced by this repository; upstream MiewID training overlap remains unresolved.

## Approved files and behavior

| File | Implemented changes |
| --- | --- |
| New reid_demo/benchmark_protocol.py | Pure protocol construction/loading/validation; identity and encounter grouping; deterministic assignments; complete overlap and eligibility audit; role-specific fitting/gallery/query access. |
| New utils/prepare_czechlynx_protocol.py | Metadata-only CLI to create the versioned protocol JSON and summary. It will not load models, extract features or run retrieval. Start with time_open; reject unsupported official split choices for this new protocol explicitly. |
| tools/evaluate_czechlynx.py | Add optional --protocol-plan and --partition calibration/selection for development runs. Resolve PCA/GMM fitting IDs independently of gallery IDs; extract fitting local features only when Fisher is requested. Calibration may create thresholds; selection must load matching calibration thresholds and, for Fisher, the matching fitting bundle. Validate artifacts before neural work. |
| tools/evaluate_czechlynx.py, threshold validation | Bind new artifacts to protocol version/hash, data, fitting IDs, calibration IDs, model identity and ranking settings. Selection cannot recalibrate or overwrite a calibration artifact. Reject old development bundles or thresholds that violate the new pool contract. The existing fitting-subset-of-gallery check must be replaced with an explicit fitting-pool check for the new protocol. |
| New reid_demo/benchmark_diagnostics.py | Reproduce the existing foreground chroma proxy and camera grouping in a maintained module. Cache appearance measurements by image/mask hashes and diagnostic version. Calculate known-query subgroup counts, identity/encounter counts, ranking metrics, accepted precision and coverage; include missing-metadata categories. |
| reid_demo/benchmark_report.py | Label calibration versus selection clearly; show role inventory and artifact provenance, subgroup results, and clustering pair precision/recall/unresolved counts next to cluster counts. Expose source/camera breakdowns without implying camera-disjoint training. |
| New tests/test_benchmark_protocol.py and tests/test_benchmark_diagnostics.py; existing benchmark integration/report tests | Add meaningful boundary and integration coverage using small synthetic metadata and mocked embeddings/local features. No model downloads required. |
| docs/czechlynx-benchmark.md, docs/czechlynx-implementation-plan.md, AGENTS.md | Document protocol creation, calibration and selection commands, artifact compatibility, diagnostics, limitations and user-run status. |

Existing commands without --protocol-plan retain the historical protocol, output semantics and cache behavior. New protocol runs use distinct report directories and contract fields. Existing per-image global/local vectors can be reused when their existing content/version contracts match. The historical Fisher bundle is not automatically valid: PCA/GMM models must be fitted only on the new fitting pool. Do not invalidate expensive feature caches merely because the protocol changes.

The new role flags are development-only. For this phase, explicitly reject --stage test combined with the new protocol flags until the final-test adapter is separately specified; legacy frozen test commands retain their existing behavior. Do not imply that new protocol thresholds can be passed into the legacy test command.

No score-weight learning, new geometric matcher, clustering algorithm change or backbone fine-tuning is included in phase 2. Those remain subsequent, separately reviewable changes after this evaluation foundation exists.

## Diagnostics

For each signal, retain the overall top-1/top-5/mAP and add the following with denominators and unique identity/encounter counts:

1. Near-monochrome versus colour queries, and queries whose correct gallery references exist only in the opposite proxy category. This remains a colour proxy, not a verified day/night or infrared label.
2. Same-camera-reference availability versus correct references only at other recorded cameras, using (source, trap_id), with missing camera values reported separately.
3. Per-source results and continuous foreground intensity/contrast summaries. Do not invent a universal “bad image” threshold from these development failures.
4. Correct-identity recall at 10/20/50 and full shortlist-union recall, including union size so larger candidate budgets remain visible.
5. Accepted precision, correct-match coverage and unknown false accepts under the frozen rejection threshold.
6. Clustering pair precision/recall, cluster count, true-individual count, mixed clusters and unresolved images if clustering is enabled. Results remain diagnostics, not abundance estimates.

## Validation plan

The user runs tests and benchmarks. No test command below is to be executed by the assistant.

Focused tests must establish:

- Deterministic output under row reordering and a fixed seed.
- No image, encounter, duplicate-source or identity overlap between fitting/calibration/selection pools.
- Connected mixed-animal encounters and duplicate chains remain together.
- No official test record enters a development pool; original official split columns are unchanged.
- Known and unknown queries have the intended gallery membership, with explicit behavior for missing dates, insufficient encounters and infeasible group constraints.
- Tampered metadata, modified protocol IDs, incompatible Fisher fitting IDs and mismatched model/threshold provenance are rejected.
- Selection loads thresholds without fitting them; integration stubs fail immediately if a fitting or threshold-calibration function is called unexpectedly.
- A fitting pool that is correctly separate from the gallery is accepted in the new protocol, while a historical bundle trained on selection identities is rejected.
- Appearance categories use foreground pixels only; missing camera data cannot be silently classified as a known camera.
- Historical mode retains existing protocol outputs; reports distinguish in-sample calibration from selection validation.

Focused user-run command:

```bash
venv/bin/python -m pytest \
  tests/test_benchmark_protocol.py \
  tests/test_benchmark_diagnostics.py \
  tests/test_benchmark_embeddings.py \
  tests/test_benchmark.py \
  tests/test_benchmark_comparison.py \
  tests/test_benchmark_report.py \
  tests/test_benchmark_ensemble.py \
  tests/test_benchmark_masks.py \
  tests/test_benchmark_gv_masks.py -q
```

The first operator workflow after those checks is metadata-only protocol creation and inventory review, followed by frozen segmented MiewID global-only calibration and selection runs. This establishes the new reference before paying for Fisher refitting or wider geometric matching. Exact runnable commands are in [the phase 2 command reference](czechlynx-benchmark.md#phase-2-independent-calibration-and-selection).

Acceptance is correct separation and artifact enforcement, reproducible reports and a completed user-run reference. Higher recognition accuracy is not an acceptance criterion for this infrastructure phase. Later calibrated fusion and conservative clustering must demonstrate a precision/coverage benefit against that reference.

## Approval and tracking

The user approved this proposal. The protocol preparer, runner integration, diagnostics, reports and regression tests are implemented. Tests, protocol generation and real-data benchmarks have not been run by the assistant; the user has completed that sequence and the global reference has been reviewed. Static syntax and whitespace inspection do not establish runtime correctness or accuracy.
