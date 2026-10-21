# CzechLynx completed input-audit review

Date: 16 October 2026 (Europe/Zagreb). The user reports completing the browser audit. Read-only comparison confirms **321/321 completed records**, one reviewer, no outstanding drafts, and exact agreement between the completed CSV and the live app's payloads and revisions. The assistant ran the existing model-free summarizer successfully; no tests, models, training or performance benchmarks were run.

## Saved evidence

- Source CSV, preserved outside this repository: `/home/mm-workstation/re-id/czechlynx-review-app/exports/reviews_complete_20261016T130044652360Z_bccd027d.csv`.
- CSV SHA256: `4e5285142551e66a7f1b93301032bcdf2e1b0e54da05515eac9636812af2244e`. The adjacent app export JSON matches its bytes, all 321 completed records and their revisions.
- Review receipt: `evaluations/czechlynx/input_audit_review_v1/receipt.json`; imported labels: sibling `reviews.json`.
- Receipt SHA256: `ffa4000e8cccc219dc3e53da82df2714263b12f3e878d448416f2031c63c93f3`.
- Receipt fingerprint: `ff7dd76ddcdf2bd7c641a55eb17427533a4d4571cd6b462a8b6dd70ac0aed71b`.
- Original audit manifest SHA256: `4c78d5f1a679feb80c233622985fb4db818997405c64690d46c26a0388321d5f`.

The summarizer validated exact sample/schema/choices, reviewer/timestamp completeness, nonempty evidence for verified illumination, deterministic protocol/source bindings, the saved audit artifacts and current sampled original/prepared/mask bytes. It does not independently verify the biological judgments or free-text illumination claims. The 321 saved preparation checks previously established exact reproduction of supplied annotations, rather than annotation correctness.

Keep the CSV at its exact path with unchanged bytes: training preparation validates that source against the receipt. Further review edits require a new export and a new summary directory.

## Cohort findings

These are completed user judgments assisted by the earlier AI drafts. Counts describe these samples; the identity-coverage and diagnostic cohorts are purposive. Do not use the combined sample as a dataset-wide quality rate. There are 322 cohort memberships across 321 unique images because one image belongs to both calibration uniform and identity coverage. Columns below overlap and must not be added together.

| Cohort | Images | Markings yes / no / unknown | Crop/mask issue / unknown | No clear animal | Severe blur/occlusion |
| --- | ---: | --- | --- | ---: | ---: |
| Uniform fitting | 128 | 71 / 55 / 2 | 24 / 7 | 12 | 41 |
| Uniform calibration | 128 | 100 / 27 / 1 | 20 / 2 | 5 | 31 |
| Calibration identity coverage | 55 | 39 / 15 / 1 | 10 / 0 | 2 | 14 |
| Calibration diagnostic extremes | 11 | 9 / 2 / 0 | 1 / 0 | 1 | 3 |

In the uniform fitting sample, 55/128 (43.0%) were marked as having unusable markings and 24/128 (18.8%) as crop/mask issues. Corresponding calibration figures are 27/128 (21.1%) and 20/128 (15.6%). This is meaningful evidence of a visual-input problem; it does not establish an identity-label error, a decoder defect, or the fraction of clustering failures caused by quality.

Across all 321 unique records: visibility clear 134 / partial 167 / no clear animal 20; markings yes 218 / no 99 / unknown 4; crop/mask correct 258 / issue 54 / unknown 9; blur/occlusion minor 54 / moderate 177 / severe 89 / unknown 1. Partial visibility can coexist with usable markings; a mask issue need not make all remaining visual evidence unusable.

Examples warranting a separately scoped source/target investigation include reviewed tree/root or rock regions (`clx_2b9653348ecae6a66f829a06`, `clx_79928077256eb1e182468f0b`, `clx_8c09c1e780e0cdc168a5f92f`) and apparent carcass regions (`clx_bdba7004a69e0b48620155a9`, `clx_be35156ed2730af75f0e495c`). These notes often retain the original `AI suggestion:` rationale; completion records human acceptance/editing, not an independent second annotation. The original suspect `clx_e981b521482a803cb7855493` is now reviewed as no clear animal, unusable markings and a crop/mask issue. Its previously demonstrated exact source-annotation alignment still stands. No identities, masks or images were corrected or removed.

## Flank and illumination limits

Flank counts are left 98 / right 132 / unknown 91. These are human-reviewed sample labels; the full dataset's prepared flank metadata remains unchanged and unknown. They do not establish opposite-flank matching performance or links between sides.

Illumination counts are colour 164 / monochrome 129 / unknown 1 / verified day 12 / verified IR 9 / verified night 6. Several evidence entries rely on appearance, for example “It really looks IR” or “glowing eyes”; some night entries simply say “clearly night.” Such statements do not establish independently verified IR or capture time. Four day entries mention capture times or camera information, but the underlying information was not independently checked in this handoff. Preserve all submitted labels and report them as reviewer assertions; do not claim a validated night/IR subset from these counts. The validator only requires nonempty evidence text. This limitation does not block the existing adaptation recipe because audit labels do not enter its loss or inference.

## Next action

Proceed with the existing **user-run `--prepare-only`** command in [the benchmark instructions](../czechlynx-benchmark.md#3-prepare-and-inspect-the-training-plan). This reads the completed receipt and writes the declared inner split without loading a model or allocating GPU memory. Review its exact inventories before GPU training. No trial directory existed at this handoff.

The approved first adaptation trial preserves the original fitting membership and every eligible calibration photograph, making it an unchanged-input representation comparison. The audit now documents a substantial limitation of that trial; success or failure cannot by itself isolate the effects of poor inputs. Any correction, quality exclusion, mask replacement or cleaned-data comparison needs a concrete separate scope, a new derived-data lineage and protected baseline evidence. Do not silently remove only reviewed difficult calibration photographs or reuse old caches for changed pixels.

Audit completion closes the manual review/receipt step. Training preparation, actual adaptation, calibration comparison, reliable clustering/counts, independent upstream exposure evidence, and park readiness remain open. No clustering selection or outreach occurred.
