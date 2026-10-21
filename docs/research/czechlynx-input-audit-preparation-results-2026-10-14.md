# CzechLynx input-audit preparation review

Date: 14 October 2026. User-run report: `evaluations/czechlynx/input_audit_v1/`. [Approved scope](../czechlynx-input-audit-adaptation-proposal-2026-10-14.md); [review instructions](../czechlynx-benchmark.md#2-prepare-the-input-audit).

## Completed preparation

The user reports **321 audited crops**. Read-only review confirms complete status, deterministic sample reconstruction, current protocol/input/implementation bindings, all **966 artifact hashes**, 321 original-photo hashes and current sampled prepared-image/mask bytes. All 321 saved alignment checks report exact reproduction of the supplied source annotation and declared masked pixels. No source annotation, identity or split was changed.

Manifest SHA-256: `4c78d5f1a679feb80c233622985fb4db818997405c64690d46c26a0388321d5f`.

Audit fingerprint: `e5614a57b5f68f89149c5b2941b9ec53add3d9c8425a2619ea46b8ab2a5e7c5a`.

| Cohort | Images | Identities | Encounters | Source images: Bohemia / Carpathians / SNPA |
| --- | ---: | ---: | ---: | --- |
| Uniform fitting | 128 | 62 | 126 | 33 / 62 / 33 |
| Uniform calibration | 128 | 32 | 122 | 28 / 51 / 49 |
| Calibration identity coverage | 55 | 55 | 55 | 14 / 17 / 24 |
| Calibration diagnostic extremes | 11 | 7 | 11 | 0 / 5 / 6 |

There are 322 cohort memberships but 321 unique images. `clx_d11d4b9d58a2b38e5afa0581` belongs to both calibration uniform and identity coverage. Preserve that overlap in cohort summaries. All samples are official time_open train; zero selection-pool images are sampled. Keep uniform and purposive summaries separate when assessing frequency of problems.

Foreground fraction inside the supplied crop ranges from 30.11% to 99.6864%, median 50.98%. The sole sampled fraction above 99% is the already-flagged `clx_e981b521482a803cb7855493`. Foreground fraction alone is not a validated identifiability or mask-quality rule.

## Bounded panel inspection

Inspected three saved panels, without assigning completed human review labels:

- Calibration uniform `clx_0182096998cf8da7febdf8fd`: partial rear/body view with visible coat spots. Partial animal visibility can coexist with visible markings.
- Fitting uniform `clx_01befd693d59e42e05bac697`: clear animal outline with a bright/washed-out appearance and weaker marking contrast. Visibility and useful marking evidence need separate judgments; check full-resolution source pixels for borderline cases.
- Diagnostic `clx_e981b521482a803cb7855493`: source-mask overlay retains almost the whole active scene. The source identifiability/annotation concern remains unresolved despite exact preparation alignment; no empty-frame or wrong-identity conclusion is made.

This is a three-panel inspection, not a completed visual audit of 321 photographs. Actual night/IR and anatomical flank labels remain unverified by this review.

## Next action

Created the separate working copy `evaluations/czechlynx/input_audit_reviews_v1.csv` using a no-overwrite copy. No reviewer fields or labels were filled by the assistant. The original hashed template/report is preserved.

Review every image against the original/crop/mask panels and fill the working CSV, retaining its columns and image IDs. Record visibility, usable markings, crop/mask correctness, blur/occlusion, flank and illumination using the exported choices. Unknown is valid. `minor` blur/occlusion covers negligible problems. Verified illumination needs source/camera evidence; monochrome alone is a proxy. Use the animal's anatomical flank and retain unknown if uncertain. Reviewer and a valid review date/time are required on every completed row; add notes for questionable evidence.

For borderline blur or spots, inspect the full-resolution photograph at the matching `checks[].source_path` in `manifest.json`; previews can hide fine detail. The audit's exact alignment checks validate preparation fidelity. Biological annotation correctness and identifiability require visual review.

After the CSV is complete, the user runs the documented `summarize` command and shares `input_audit_review_v1/receipt.json`. Review its cohort results before training preparation. No human-review receipt, real training or adaptation calibration is complete yet. The assistant performed saved-evidence/hash/panel review, copied a working CSV and updated documentation; it did not execute the audit, tests, models, training, benchmarks or outreach.
