# CzechLynx adapted-model full calibration review

Date: 17 October 2026. User-run report: `evaluations/czechlynx/miewid_adaptation_calibration_v2_fp32`. Status: **complete**, elapsed **172.195 seconds**. The adapted model improves calibration retrieval, but **none of the 25 original grouping controls demonstrates reliable photographed-individual counts**. No selection evaluation or configuration export followed this result.

The assistant reviewed saved artifacts, read-only vector caches, assignment/count/pair arithmetic and a bounded set of existing figures/thumbnails. No model loading/inference, training, tests, fresh candidate construction, pair scoring or performance benchmark was run by the assistant. No source, input, trial or original evaluation artifact was changed.

## Recognition on identical inputs

The gallery has 4,522 photos; queries are 1,336 known-to-gallery and 494 unknown-to-gallery photos. All 6,352 eligible calibration photos / 55 reference identities remain. Recognition below uses the same known queries, gallery and depth 50 as the frozen baseline.

| Metric | Frozen | Adapted |
| --- | ---: | ---: |
| Top-1 | 877/1,336 = 65.6437% | 917/1,336 = 68.6377% |
| Macro identity top-1 | 56.0883% | 60.3884% |
| Top-5 | 82.4850% | 83.1587% |
| mAP@50 | 31.0078% | 40.1974% |
| Recall@50 | 94.8353% | 95.8832% |

There are 118 corrected and 78 worsened known queries, for a net 40 additional top-1 successes (+2.9940 percentage points). Macro identity top-1 improves by 4.3000 points. These are calibration development results; they do not establish independent field performance or upstream pretraining independence.

Recognition rejection was separately recalibrated in-sample: raw threshold 0.537294 → 0.388476. Correct-known acceptance coverage improves 480/1,336 = 35.9281% → 602/1,336 = 45.0599%; accepted precision improves 480/534 = 89.8876% → 602/655 = 91.9084%. Both thresholds admit 24/494 unknown queries (4.8583%). These operating points are fitted and evaluated on calibration, not an independent rejection qualification.

## Grouping and photographed-individual counts

All 25 profiles assign finite-vector observations; unresolved-vector count is zero. This does not mean all identity decisions are confident or correct. Singleton components are provisional proposals, not certified new animals. **Exact recovery of an identity as one pure component is zero for every profile**, as in the frozen control.

Representative controls, with 55 actual photographed individuals:

| Control | Groups frozen → adapted | Pair precision frozen → adapted | Pair recall frozen → adapted |
| --- | ---: | ---: | ---: |
| Legacy k15 Louvain | 18 → 26 | 14.702% → 26.603% | 23.542% → 26.292% |
| Threshold k15 / .35 | 86 → 1,545 | 6.600% → 7.983% | 97.171% → 51.060% |
| Threshold k15 / .45 | 1,120 → 3,028 | 6.884% → 69.259% | 62.430% → 10.177% |
| Threshold k15 / .55 | 3,079 → 4,236 | 16.034% → 99.802% | 10.611% → 2.471% |
| Threshold k15 / .65 | 4,623 → 5,258 | 98.869% → 99.983% | 1.318% → 0.450% |
| Consistent k15 / .35 | 1,607 → 3,120 | 90.261% → 98.864% | 2.280% → 0.886% |

Across all adapted controls, group counts range from 26 to 6,149. Legacy's 26 is numerically closer to 55 than its former 18, but 25/26 groups contain multiple identities and 54/55 identities span multiple groups. Its largest group contains 848 photos from 39 identities. It remains unsuitable as a count/catalogue.

At threshold .35, the largest group still contains 4,119 photos from 50 identities, alongside many fragments/singletons. At threshold .55, only three groups are mixed and pair precision is 99.802%, but all 55 identities are split and pair recall is 2.471%; 4,236 proposed groups cannot represent a defensible 55-animal count. Consistent k15/.35 has 79 mixed groups and splits all 55 identities despite high pair precision. A correct total, if encountered elsewhere, would still need merge/split validation.

All six threshold-profile assignment files are identical between k15 and k50 at the corresponding cutoff; their profile labels do not imply different outcomes. Full per-profile before/after scalar results are retained in the review receipt.

## Pair evidence and connectivity

All **20,170,776 unordered pairs** were scored in the user-run diagnostic, with zero unscorable pairs and 1,326,307 same-identity pairs in the reference labels. The adapted candidate unions contain 69,379 k15 pairs and 219,430 k50 pairs; the saved candidate audit reports exact reconstruction/prefix agreement.

| Unthresholded truth-filtered oracle | Frozen | Adapted |
| --- | ---: | ---: |
| k15 fully connected identities | 2/55 | 3/55 |
| k15 fragments | 382 | 342 |
| k50 fully connected identities | 10/55 | 10/55 |
| k50 fragments | 196 | 187 |
| k50 images with no correct neighbour | 100 | 88 |

The adapted k50 union contains more true links (95,056 → 109,504) despite a smaller union (232,008 → 219,430). Yet 45/55 identities remain disconnected even after truth removes every wrong edge. This diagnoses missing paths in the fixed candidate graph; it is not a deployed count, a guaranteed bound on future methods, or evidence that local methods can automatically supply the missing paths.

The six raw cosine cutoffs were deliberately retained for the representation comparison. Their operating points change materially: at .35, all-pair precision/positive recall are 32.701%/11.776% frozen versus 96.782%/3.358% adapted. At .55, they are 98.174%/1.155% versus 99.936%/0.469%. Histogram figures show both score distributions shifted toward lower cosine values while remaining overlapped. Thus more fragmentation at the same numerical cutoff is not, on its own, proof of worse representation; pair-score calibration and candidate coverage must be evaluated separately from ranking.

Sparse wrong links can still propagate. Adapted k50/.35 direct-edge precision is 96.544%, while the resulting threshold components have only 7.983% component-pair precision. The fixed controls expose both missing true paths and harmful connectivity; a higher top-1 score alone does not solve either problem.

## Integrity and bounded visual review

- Top-level status/bindings and **647 artifact hashes** pass. Current evaluator, training-trial/checkpoint bindings, metadata/protocol and both frozen/adapted source contracts match. The completed training artifacts, failed v1, original audit/receipt and 22 protected file hashes are unchanged.
- Both vector caches were opened in SQLite read-only mode. All 6,352 image/mask/vector bytes per model pass their contracts; loaded matrices are finite 6,352 × 2,152 FP32 arrays and their hashes match the saved diagnostic matrix hashes. Cache hashes match before/after and the original frozen cache remains unchanged.
- All **50 saved assignment profiles** (25 frozen + 25 adapted) independently reproduce component totals, count errors, same/predicted/correct pair totals, pair precision/recall, merge/split counts, exact identity-group recovery and component sizes from saved labels. No clustering was rerun. Per-identity threshold/histogram arithmetic and recognition correction/worsening counts agree with saved tables.
- Inspected both saved histogram figures and the first preselected pair from each diagnostic category: three pairs / six thumbnails out of 24 pairs. The high-score different-reference-identity pair has similar pale rear views; the low-score same-reference-identity pair differs in visible view/appearance, with faint markings in one thumbnail; the disconnected same-reference-identity example differs in crop support and visible view. These score extremes are not a representative visual audit, causal attribution, annotation correction, verified flank labels or evidence of actual night/IR conditions.

Machine-readable [review receipt](czechlynx-miewid-fp32-calibration-results-2026-10-17.json), SHA256 `37b2de811f4b9d2cc4e3f65b9ea38feff91098159484124b2965d762599fd0f0`. Original full [comparison report](../../evaluations/czechlynx/miewid_adaptation_calibration_v2_fp32/index.html).

## Next direction

Preserve this adapted model/report as a completed, improved recognition baseline. Do not export a count configuration or run frozen selection from these failing grouping controls. Further training alone is not yet justified as the next correction.

The next concrete code proposal should investigate the user's intended thesis local/global combination on identical candidate pairs: distinguish shortlist misses from erroneous pair evidence; audit local mask/coordinate/geometry handling; measure DISK/LightGlue and declared Fisher contributions alongside adapted global scores; calibrate pair evidence with explicit missing/failed-local handling; and test candidate diversity plus grouping rules resistant to false bridges. Local evidence can validate or reject considered links; it cannot recover an absent candidate unless candidate construction also changes. Raw cosine thresholds should not be treated as probabilities or reused without calibration after representation changes.

That is a proposed method direction, not an implemented or approved new source scope. Present exact files/changes before modifying code; all tests/models/benchmarks remain user-run. Input inclusion stays unchanged unless separately scoped. Preserve frozen selection for a subsequently chosen method. Verified night/flank handling, independent data, acceptable count error/review burden and park readiness remain open. Reliable photographed-individual counts are still unachieved.
