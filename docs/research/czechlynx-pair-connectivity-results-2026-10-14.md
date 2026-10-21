# CzechLynx pair-score and candidate-connectivity diagnostic review

Date: 14 October 2026. Scope: user-run `evaluations/czechlynx/cluster_pair_diagnostics_global_v1/`, following the [clustering calibration review](czechlynx-clustering-calibration-results-2026-10-14.md). [Machine-readable receipt](czechlynx-pair-connectivity-results-2026-10-14.json).

## Conclusion and next decision

The current frozen embeddings and candidate graphs have **both missing true connections and misleading accepted connections**. The existing threshold/complete-link controls do not solve either problem well enough for counting. Changing a cosine cutoff alone is not a sufficient demonstrated solution. A scalar monotone score calibration preserves pair ordering and cannot supply links excluded from the candidate graph.

The next improvement should address candidate diversity/coverage and identity evidence across views, with explicit handling of photographs that cannot support identification. Before training or fitting new scores, audit questionable image/mask/visibility examples. A controlled supervised backbone adaptation on the existing **fitting pool only** is a candidate next experiment, alongside calibrated global/local pair evidence and less brittle component merging. These are recommendations, not implemented or newly approved code scopes or guaranteed improvements.

Keep the clustering selection pool unevaluated until a method is chosen using calibration. No configuration was exported.

## Completion and evidence checks

- The user reports **41 focused diagnostic tests passed in 5.19s**, with no warnings shown in the provided output.
- Diagnostic manifest is `complete`; all **6,352 vectors** are valid, all **20,170,776 unordered pairs** were scored, and zero observations/pairs are unresolved or unscorable.
- There are **55 reference individuals** and **1,326,307 same-identity image pairs**. The remaining 18,844,469 pairs have different reference identities.
- Candidate unions contain **71,450 edges at k=15** and **232,008 at k=50**, matching the source calibration counts and retained score/ID prefixes.
- Verified all **50 diagnostic artifact hashes**, current source/implementation bindings, source-cache before/after/current SHA-256 equality, pair-population/histogram totals, per-identity macro-recall arithmetic and oracle totals against the saved tables.
- Run fingerprint: `bdf7844b5f7b052e6ca8d15e15abd93d6359f808a9dac1755530a3cbee321ffc`.
- Saved total runtime is **14.913s**: all-pair evaluation 4.493s, candidate diagnostics 1.027s, candidate reconstruction 3.732s, vector validation 1.151s and report generation 1.009s, plus validation/other run work.
- No tests, diagnostic/model runs, clustering or training were executed by the assistant. The assistant reviewed saved reports, performed read-only arithmetic/hash checks, inspected images and updated documentation.

## 1. Candidate coverage remains insufficient

Oracle evaluation removes every different-identity edge using reference truth. It then measures connectivity inside each animal's fixed candidate graph. This is an evaluation diagnostic, not a deployable classifier, achieved identity recovery or a bound on all future methods.

| Budget and cutoff | Fully connected identities | Oracle fragments | Images with no correct candidate neighbour |
| --- | ---: | ---: | ---: |
| k=15, no cutoff | 2/55 | 382 | 219 |
| k=50, no cutoff | 10/55 | 196 | 100 |
| k=15, cutoff .35 | 2/55 | 552 | 358 |
| k=50, cutoff .35 | 3/55 | 439 | 292 |

All 55 identities have multiple photographs. None is entirely without correct candidate edges, but many have detached photographs or separate groups. Increasing the budget from 15 to 50 helps coverage; it still cannot connect 45 of 55 identities even with perfect filtering.

At k=50 without a cutoff, selecting each identity's largest truth-filtered component covers **6,090/6,352 = 95.88% of photographs**, leaving 262 outside these components. Mean coverage across identities is **81.22%**. There are 100 singleton oracle fragments. Large identity groups dominate the image-weighted figure; full-set recovery and rare/sparsely photographed animals remain harder.

These figures use both truth-filtered edges and truth-selected largest components. They are not model accuracy, predicted clean-core coverage or a certified animal catalogue.

Examples of unthresholded k=50 fragmentation: `lynx_201` has 52 photographs in 12 components (largest 30); `lynx_013` has 31 in seven (largest 15); `lynx_209` has 13 in seven (largest four). This makes macro-identity and sparse-identity evaluation necessary.

## 2. Raw cosine precision comes with low positive recall

These figures cover all unordered pairs, not query recognition or candidate-edge precision.

| Raw cutoff | Pair precision | Same-identity pair recall | Macro-identity positive recall |
| --- | ---: | ---: | ---: |
| .35 | 32.70% | 11.78% | 23.92% |
| .45 | 76.73% | 3.84% | 12.03% |
| .55 | 98.17% | 1.16% | 5.17% |
| .65 | 99.90% | 0.31% | 2.38% |

All 55 identities have positive pairs and contribute to the overall macro figure. The histogram plot shows extensive overlap between same/different score populations. Good query ranking can coexist with low absolute similarities between many photographs of the same animal. A useful retrieval result does not imply that one global threshold will recover complete identity groups.

At cutoff .55:

- Across cameras: **96.44% precision / 0.65% recall**, macro positive recall 1.87%, 49 eligible identities.
- Across colour/near-monochrome appearance proxies: **98.40% / 0.57%**, macro positive recall 0.80%, 45 eligible identities.

These are structural camera/colour groups, not a camera-disjoint model evaluation or verified night/IR/flank labels. Context inventory has zero missing encounter/camera/appearance proxies but **all 6,352 flank labels remain unknown**.

## 3. Incorrect direct edges propagate through components

Candidate selection changes the score population. At k=15 and cutoff .35, direct admitted-edge precision is **59.24%** (38,252 correct / 64,568 admitted), whereas threshold-component pair precision is **6.60%**. At k=50/.35, direct-edge precision is **46.82%**, and component precision is still 6.60%, with the known giant 6,250-image group spanning every animal.

At k=50/.55, direct admitted-edge precision is **98.16%**, with **285 wrong edges** among 15,470 admitted edges. The existing threshold graph's component-pair precision is **16.03%**. A small fraction of incorrect edges can still connect identity groups and generate many indirect false pairs.

Direct-edge and component-pair precision have different denominators. Their differences describe propagation in these saved graphs, not a causal effect estimate or confidence interval. Low positive pair recall alone also does not require failure: a sparse graph can connect an identity. The oracle connectivity results independently show where sparse true paths are actually absent.

Robust merging and better pair evidence are required together. The existing complete-link rule blocks bridges but demands every cross-component raw similarity exceed the threshold; the calibration results demonstrate severe fragmentation. Added candidates alone also admit more false links unless scoring/merging improves.

## 4. Bounded visual and input review

Inspected the histogram figure and six preselected pairs: 12 endpoint views covering 11 unique prepared crops, plus one original frame.

- The strongest wrong candidate pair (cosine **0.6884**, `lynx_096`/`lynx_086`) shows similar rear/side poses in monochrome.
- The second (0.6792, `lynx_250`/`lynx_276`) shows conspicuous similar collars. Possible reliance on pose/collar/coarse appearance is a hypothesis requiring an attribution or controlled ablation; the images do not prove what the model uses.
- Both inspected extreme low-score positive pairs share `clx_e981b521482a803cb7855493` (`lynx_280`). Its binary mask retains **99.6864% of a 1,919 x 1,022 crop**. The original `07972_lynx_280.jpg` has no clearly identifiable lynx silhouette/markings in the inspected view. Verify its source image and annotation rather than assuming the frame is empty or the preparation decoder is wrong. No record was relabelled, removed or changed.
- Disconnected `lynx_013` examples differ in appearance proxy and visible animal regions. Disconnected `lynx_019` examples have a partial rear view and a fuller side view.

The suspect image is a specific input-audit case, not evidence that all supplied masks are defective or the cause of the complete score overlap. These examples are deterministic extremes, not a representative full audit. Annotation correctness, usable marking visibility, original scene versus crop, blur/occlusion and actual camera illumination mode should be distinguished.

Uninformative inputs can still yield nonzero neural vectors. The current zero-vector unresolved policy therefore does not establish adequate image-quality handling. Future quality rejection must be validated, retain photographs in the defined collection with explicit unresolved status, and expose its count/coverage/review consequences. Silently dropping hard photographs is not an acceptable improvement claim.

## Recommended improvement sequence

**Subsequent decision, 14 October:** The user agrees with these recommendations. The [concrete input-audit/adaptation proposal](../czechlynx-input-audit-adaptation-proposal-2026-10-14.md) presents the first seven-file code scope; its approval is pending. Additional read-only comparison confirms the suspect lynx_280 crop/mask exactly reproduce the supplied source annotation and declared masking, with no preparation mismatch in that case. Source identifiability remains uncertain; no record was changed. [Supplementary receipt](czechlynx-adaptation-feasibility-2026-10-14.json).

1. **Audit questionable inputs and define an identifiability policy.** Start with these saved examples and a representative calibration sample. Record animal visibility, crop/mask correctness, usable markings, pose/partial-view uncertainty and actual night/IR labels where evidence permits. Preserve unknowns and original split/identity metadata.
2. **Improve identity representation across views as a controlled experiment.** Consider fine-tuning the strongest frozen backbone on the existing identity-disjoint fitting pool, with encounter/camera-diverse positives and declared illumination/partial-view augmentations. Compare with the unchanged frozen baseline on calibration. Augmentation is a candidate treatment, not validated night/IR or opposite-flank linkage. Model changes and checkpoint support need a concrete code proposal and approval.
3. **Improve candidate coverage and pair evidence together.** Compare broader and encounter-diverse candidates while preserving original neighbours as a reference. Encounter/camera metadata can diversify candidate selection without creating same-animal constraints. Evaluate calibrated global/local evidence, including DISK/LightGlue where supported, before deciding on graph merging rules. The previous RoMa diagnostic does not justify indiscriminate matcher replacement.
4. **Evaluate actual grouping/count outcomes.** Measure merges, fragments, unresolved observations, macro recovery, sparse identities and reviewer effort. Replace the brittle all-cross-pair raw-cosine criterion only through an explicit controlled comparison. Do not select by a numerically convenient count or retrieval accuracy alone.
5. Choose one method on calibration, freeze it, then evaluate clustering selection. An appropriately independent final evaluation, upstream provenance resolution, flank/night evidence and count uncertainty remain necessary for park readiness.

No additional code changes, training or benchmark runs are authorised merely by this recommendation. The approved diagnostic is implemented and user-validated. Reliable photographed-individual counts remain unachieved, and no total Risnjak abundance claim is supported.

