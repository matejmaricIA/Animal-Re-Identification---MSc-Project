# CzechLynx benchmark

This evaluates the existing identification pipeline before an institutional pilot. It supports public MegaDescriptor or MiewID-msv3 global embeddings, a global + Fisher shortlist, and geometric reranking with LightGlue. Fisher features can use DISK alone or the existing DISK + SuperPoint + ALIKED ensemble. It also produces a bounded query-image clustering diagnostic and a portable HTML report. Work and validation are tracked in [the implementation plan](czechlynx-implementation-plan.md).

## Input audit and native MiewID adaptation

**Latest status (17 October 2026): training and full calibration comparison are complete and reviewed.** [Calibration review/receipt](research/czechlynx-miewid-fp32-calibration-results-2026-10-17.md): top-1 65.6437% → 68.6377%, macro top-1 56.0883% → 60.3884%, mAP@50 31.0078% → 40.1974%. None of the 25 grouping controls achieves reliable counts; exact identity-group recovery stays zero. Legacy gives 26 groups for 55 identities with substantial mixing; stricter controls split the collection into thousands. Oracle k50 still connects only 10/55 identities (truth-filtered, not deployable counts). All 647 artifacts, both vector caches, assignment/pair/count arithmetic and protected evidence pass review. Preserve the completed adapted baseline, original controls/audit/trials and reserved selection. Commands below are historical reference, not rerun instructions. Next is a concrete local/global candidate/pair-calibration/grouping proposal; the [diary](lynx-demo-decision-log.md) tracks remaining gates.

Historical focused result before the FP32 correction: **26 passed / 93 warnings in 9.35s**. The first attempt returned 21 passed / five failed in 6.46s; its shared inner-plan pandas index/column ambiguity was corrected by resetting the temporary index. Those results do not validate the new precision/diagnostic regressions. The warnings are upstream pycocotools NumPy deprecations.

The [approved seven-file scope](czechlynx-input-audit-adaptation-proposal-2026-10-14.md) and subsequent three-file correction are implemented and user-tested. **All 321 human audit reviews/receipt, v2 preparation, training and full calibration review are complete**. [Completed audit findings](research/czechlynx-input-audit-review-results-2026-10-16.md). The trial adapts native 2,152-D MiewID using only the fitting pool, with a linked-group inner holdout. Recognition improved on inner holdout and calibration; reliable grouping/counts and park readiness remain unachieved.

**Preparation history (16 October 2026; training failure above supersedes its execution handoff):** The user completed `--prepare-only`; [saved-plan review](research/czechlynx-miewid-training-preparation-results-2026-10-16.md) confirms 14,044 training images / 131 identities and 2,460 validation images / 34 identities, zero cross-role identity/encounter/photo-hash overlap and all validation images metadata-eligible. The final browser export was validated and summarized into `evaluations/czechlynx/input_audit_review_v1/receipt.json`; no audit rerun is needed. Preserve `/home/mm-workstation/re-id/czechlynx-review-app/exports/reviews_complete_20261016T130044652360Z_bccd027d.csv` unchanged. Uniform fitting/calibration have 55/128 and 27/128 unusable-marking judgments and 24/128 and 20/128 mask issues. These labels do not automatically filter training/evaluation. Some verified illumination notes rely on appearance; the receipt establishes structural completeness, not independent verification of capture conditions. Corrections or quality exclusions need their own scope.

Run from the repository root in this order. Stop on failures and share the output. Training follows review of the input audit and prepared inner plan.

### 1. Focused tests

Completed by the user after the FP32 correction: **37 passed, 121 warnings in 11.61s**. Steps 2–5 and their evidence reviews are complete; commands are retained for reference.

```bash
venv/bin/python -m pytest \
  tests/test_benchmark_input_audit.py \
  tests/test_benchmark_miewid_adaptation.py -q
```

Tests use temporary photographs, synthetic native-shaped CPU models and mocked publisher loading. The composed test exercises the real 25-profile clustering and full pair-report interfaces on a small fixture. Passing tests cannot establish field accuracy.

### 2. Prepare the input audit

**Completed by the user, including human review and receipt:** 321 unique crops, with all 966 artifacts/current sampled input bytes verified and 321 exact source-annotation alignment checks. One image overlaps calibration uniform/identity coverage. [Preparation review](research/czechlynx-input-audit-preparation-results-2026-10-14.md). No audit/review/summarize rerun is needed. Commands and browser instructions below are historical reference; retain the bound export/receipt unchanged. The app keeps its own reviews and exports outside the repository.

```bash
venv/bin/python tools/audit_czechlynx_inputs.py prepare \
  --metadata data/czechlynx/prepared_masked/masked/metadata.csv \
  --protocol-plan data/czechlynx/protocols/identity_pools_v1.json \
  --raw-metadata data/czechlynx/raw/CzechLynxDataset-Metadata-Real.csv \
  --images-root data/czechlynx/raw \
  --out-dir evaluations/czechlynx/input_audit_v1
```

Open `evaluations/czechlynx/input_audit_v1/index.html`. The report shows source/prepared/mask views for up to 322 distinct records: 128 uniformly sampled fitting and 128 calibration photographs, one per calibration identity and the 11 previously reviewed diagnostic crops, deduplicated with cohort memberships retained. It verifies source annotation/prepared pixel alignment. Uniform and purposive summaries remain separate.

#### Browser review on the local network

User-approved app: `/home/mm-workstation/re-id/czechlynx-review-app`, outside this project. Open **http://192.168.8.110:8766** on the same network, or `http://localhost:8766` on the workstation while its server runs. To start/restart:

```bash
bash /home/mm-workstation/re-id/czechlynx-review-app/run.sh
```

Enter your reviewer name, inspect the original/crop/mask overlay (click to zoom), choose the six audit fields and use **Save and next**. Unknowns remain valid; verified illumination needs evidence. Drafts autosave, but completion requires the explicit save action. SQLite state/history and dated exports persist in the sibling folder. No GPU/models/dependency installs. Identity labels are hidden. The user completed the browser workflow and exported all 321 reviews on 16 October. The 13 focused app tests still have no reported result; command retained for reference:

```bash
python3 -m unittest discover -s /home/mm-workstation/re-id/czechlynx-review-app/tests -v
```

**AI draft history (15 October 2026; human review completed 16 October):** User-authorized visual inspection covered all 321 preview sets plus 12 larger source/crop checks. Imported 318 uncompleted AI drafts; preserved two completed user reviews and one existing draft. Notes beginning `AI suggestion:` explain proposed labels. At the AI handoff there were 2 complete / 319 drafts; the user has since completed all 321, with a saved export and summarized receipt. Uncertain labels remain valid. Separate catalogue/handoff/evidence remain outside the repository under sibling `exports/ai_draft_catalogue_20261014T224307731683Z.json`, `exports/ai_draft_handoff_20261014T224307731683Z.md` and `data/ai_draft_evidence_20261014T224307731683Z`. The app server was restarted in the background on the same port; no app/benchmark source changes or assistant-run tests/training/benchmarks accompanied the drafts.

After all 321 reviews, use **Export completed audit**. The app shows the CSV's absolute workstation path and offers a download. Pass that exact saved path to the existing summarize command:

```bash
venv/bin/python tools/audit_czechlynx_inputs.py summarize \
  --audit-dir evaluations/czechlynx/input_audit_v1 \
  --reviews /home/mm-workstation/re-id/czechlynx-review-app/exports/reviews_complete_TIMESTAMP_ID.csv \
  --out-dir evaluations/czechlynx/input_audit_review_v1
```

Replace the example filename with the path shown by the app. Earlier export snapshots are never overwritten by the app; preserve the chosen CSV after summarizing. A progress export contains unfinished rows and cannot substitute for a completed review receipt. The browser app does not execute the summarizer, tests, training or benchmarks. Full launch/storage/backup details: sibling `README.md`. Port 8765 was occupied, so the app defaults to 8766.

#### Manual CSV review

Copy the review template, then fill its rows while inspecting the report:

```bash
cp evaluations/czechlynx/input_audit_v1/reviews_template.csv \
  evaluations/czechlynx/input_audit_reviews_v1.csv
```

Keep the columns/order. Choices are in `review_schema.json`; unknown is valid. Each row needs `reviewer` and `reviewed_at` (an ISO date/time). Verified illumination requires `illumination_evidence` from source/camera information; monochrome alone is insufficient. Source identity labels are absent from the inspection page. Reviews describe visible evidence, not verified identity recognition.

After completing the CSV:

```bash
venv/bin/python tools/audit_czechlynx_inputs.py summarize \
  --audit-dir evaluations/czechlynx/input_audit_v1 \
  --reviews evaluations/czechlynx/input_audit_reviews_v1.csv \
  --out-dir evaluations/czechlynx/input_audit_review_v1
```

Share the cohort summary in `input_audit_review_v1/receipt.json` for review before training. Confirmed annotation/preparation problems need their own correction scope. These review labels do not automatically remove photographs or change identities, and do not train a quality filter. Editing the source CSV after summarizing invalidates its receipt; create a new summary directory for revised reviews.

### 3. Prepare and inspect the training plan

**V2 completed by the user and reviewed:** training 14,044 images / 131 identities / 123 atomic groups / 6,348 encounters; validation 2,460 / 34 / 33 / 1,297. [Saved v2 review](research/czechlynx-miewid-fp32-preparation-results-2026-10-16.md) confirms the exact v1/reconstructed inner-plan fingerprint `ef8d64056a89ef7bce2e37db9d089bd86ce92ff12b922f9f4acbd951db0c88e7`, new run fingerprint `675d43e66317e459cf7d0672789ed19d4a3db1d91f1e606dc7b8233b3e8cd22d`, fixed `training_precision: float32`, current bindings and zero overlaps. Command retained for reference; continue to step 4. Original fitting membership is retained; reviewed quality issues do not cause automatic exclusions.

```bash
venv/bin/python utils/finetune_czechlynx_miewid.py \
  --metadata data/czechlynx/prepared_masked/masked/metadata.csv \
  --protocol-plan data/czechlynx/protocols/identity_pools_v1.json \
  --audit-receipt evaluations/czechlynx/input_audit_review_v1/receipt.json \
  --out-dir models/czechlynx/miewid_adaptation_v2_fp32 \
  --prepare-only --threads 4 --workers 4
```

This is model-free and validates the review receipt, all fitting image/mask bytes and exact role membership. Inspect `inner_plan.json` and the printed inventory. Whole linked identity groups stay together; training/inner validation share no identity, encounter or source/prepared photograph hashes. Neither uses outer calibration, selection or official time_open test. Inner-validation queries without a legal cross-encounter positive are reported separately.

### 4. User-run GPU training

**Completed by the user and reviewed:** v2 status `complete`, epochs 0–9, selected epoch 6, elapsed 39.97 min. Command retained for reference; do not rerun/resume this directory. Continue to step 5. Preserve failed v1.

```bash
venv/bin/python utils/finetune_czechlynx_miewid.py \
  --metadata data/czechlynx/prepared_masked/masked/metadata.csv \
  --protocol-plan data/czechlynx/protocols/identity_pools_v1.json \
  --audit-receipt evaluations/czechlynx/input_audit_review_v1/receipt.json \
  --out-dir models/czechlynx/miewid_adaptation_v2_fp32 \
  --threads 4 --workers 4
```

Default: four identities x two observations, ten maximum epochs including one head-only warmup, AdamW embedding/head LR 2e-5/1e-3, fixed BN running statistics, mild jitter and grayscale augmentation. Training model/head/inputs use FP32 with autocast and loss scaling disabled. Validation/extraction retains CUDA FP16 autocast (CPU FP32), FP32 normalization and FP32 stored matrices; runtime precision is recorded separately for training and evaluation. Strict finite checks, gradient clipping at 1.0, warmup state preservation and actual embedding-update proof remain. On numerical failure the manifest records exact stage/epoch/batch/image IDs and failure-only gradient details, separating non-finite entries from finite-entry norm overflow. The failed batch does not reach its optimizer update. No training resume is implemented. A started/failed/completed trial needs a fresh directory and matching prepare-only step. Any batch-size change requires a separately approved and recorded recipe change.

The declared workload is 1,756 balanced batches per epoch, at most 17,560 batches before early stopping. The completed run stopped at epoch 9: 15,804 training batches, 14,048 sampled observations per epoch. Frozen epoch zero took 17.697s, head warmup 92.079s; adaptation epochs took 269.112–289.069s each, total run 2,397.940s. No peak-memory evidence is recorded. These timings describe this workstation/trial, not a guarantee for future runs.

The trainer writes epoch metrics/timings, native `best.pt`/`last.pt` and a completed or failed manifest. It selects by macro cross-encounter recall@50, macro top-1 and earliest epoch, with frozen epoch zero eligible. Grayscale/camera diagnostics are not verified night/flank results. Share `manifest.json` and `epochs.json` for review. If epoch zero stays best, review that unsuccessful result before a calibration run; the evaluator rejects it. Initialisation still has unresolved upstream exposure.

### 5. Full calibration comparison

**Completed by the user and reviewed:** 172.195s, all 6,352 eligible calibration photos / 55 identities, same original 25 controls and all 20,170,776 pairs. [Review/receipt](research/czechlynx-miewid-fp32-calibration-results-2026-10-17.md). Recognition improves, but counts remain unreliable. Command retained for reference; its output and adapted cache now exist and must be preserved. No count config/selection run is justified by these controls:

```bash
venv/bin/python tools/evaluate_czechlynx_miewid_adaptation.py \
  --metadata data/czechlynx/prepared_masked/masked/metadata.csv \
  --protocol-plan data/czechlynx/protocols/identity_pools_v1.json \
  --trial-dir models/czechlynx/miewid_adaptation_v2_fp32 \
  --baseline-calibration-dir evaluations/czechlynx/cluster_calibration_global_v1 \
  --baseline-diagnostics-dir evaluations/czechlynx/cluster_pair_diagnostics_global_v1 \
  --cache-dir data/czechlynx/cache/miewid_adaptation_v2_fp32 \
  --out-dir evaluations/czechlynx/miewid_adaptation_calibration_v2_fp32 \
  --threads 4 --chunk-size 64 --batch-size 8 --workers 2
```

The evaluator checks that the baseline's native initialization weights, protocol and inputs match the trial, verifies saved rankings against immutable baseline vectors, and loads only the adapted model. All 6,352 eligible calibration photographs remain. New embeddings use a separate checksummed cache, with verified partial extraction entries reusable for a new comparison directory. Reports include `global/`, `clustering/`, `pair_diagnostics/` and a combined `index.html`/`comparison.json`. Complete top-level status requires every stage.

The original 25 profiles and k15/k50 candidate budgets remain the representation comparison controls. Inspect macro recognition, oracle connectivity, merges/fragments, exact group recovery, unresolved coverage and count errors. Retrieval improvement alone does not meet the count goal. No cluster configuration is exported and this evaluator has no selection/test mode. Photographed groups do not estimate total park abundance.

## Data and preparation

The separate [RoMa versus DISK/LightGlue pairwise experiment](czechlynx-roma-pairwise.md)
uses saved phase 2 MiewID calibration candidates. It reports correspondence evidence
and visualisations; it does not modify retrieval, Fisher aggregation or thresholds.

Use the real-image release in [Zenodo record 17592004](https://zenodo.org/records/17592004). The record title says v1.0; the real metadata uses the schema called CzechLynxv2 by WildlifeDatasets. The importer deliberately does not combine synthetic and real photographs. Downloaded files are checked against the publisher's size and MD5; extraction rejects unsafe paths and checks ZIP CRCs.

    venv/bin/python utils/prepare_czechlynx.py --download

The default output is data/czechlynx/prepared/{frame,bbox}/metadata.csv. Frame mode preserves original photographs. Bbox mode crops the rectangle enclosing the supplied instance mask: it is annotation-assisted identification, not a test of automatic detection or segmentation. Stable IDs use full source paths, so equal camera filenames cannot collide. The original files and official split columns remain unchanged.

Preparation checks image decoding, RLE mask dimensions and bounds, missing fields, duplicate source paths, split values, dates, and file hashes. It writes data_audit.json and manifests. It stops on invalid images by default. An explicitly reviewed exclusion run can use --on-error exclude; modes prepared together retain the same valid-image set. No validated flank field exists in this schema, so orientation stays unknown.

Optional `--modes masked` uses the full provided segmentation mask. It saves lossless RGB PNGs at the same crop coordinates as bbox mode: foreground pixels retain their decoded source-image values, and background pixels become RGB (128, 128, 128). Each image has an aligned grayscale PNG mask with values 0 and 255. Metadata includes `mask_path` and `mask_sha256`; image/mask pairs have checksummed preparation receipts. Stable image IDs, source hashes, encounter IDs and official split assignments are preserved. This evaluates identification with provided segmentation; it does not evaluate automatic segmentation quality.

## Historical protocol (without --protocol-plan)

Development uses only the official training portion of the selected split. A seeded sample reserves approximately 20% of identities as unseen queries; later encounters of other identities supply known queries. Shared encounters across identities are kept on one side. Missing dates are recorded, not invented. Image caps are approximate because complete encounters are sampled.

PCA and GMM are fitted only on development gallery images. A test run must load that frozen model. Rejection thresholds are selected from unseen development queries, with a requested empirical false-acceptance ceiling. Development rejection rates are calibration figures, not independent validation. Test thresholds must have matching data, weights, Fisher model and ranking settings, with all calibration images inside official train.

Final tests preserve the published temporal splits. Gallery/query image IDs, encounter IDs, source checksums, and input checksums must be disjoint. The real metadata contains 10 mixed-individual encounters crossing the published time_closed split (259 images involved). The default policy stops on overlap. --overlap-policy purge-query explicitly removes overlapping query rows and records their IDs; those results must be described as the purged official protocol. time_open has no shared encounter IDs in the downloaded metadata. The geographic split is not silently repurposed as known-identity retrieval.

Reference identities are passed only to protocol construction and scoring. Prediction receives image IDs and features. Clustering receives no target count; its groups are proposals, not park population estimates.

## Phase 2: independent calibration and selection

Implemented after approval on 16 September 2026; the user reports **106 focused tests passed, 1 warning in 5.93s**. The warning is LightGlue's deprecated PyTorch AMP decorator. The protocol inventory, user-run calibration and frozen selection results are reviewed. Calibration top-1 is 65.64%, with in-sample accepted precision 89.89% at 35.93% correct-known coverage. Selection top-1 is 63.79%, with accepted precision 84.86% at 36.89% correct-known coverage. See the [calibration review](research/czechlynx-phase2-calibration-results-2026-09-16.md) and [selection review](research/czechlynx-phase2-selection-results-2026-09-16.md). No baseline rerun is needed. This phase adds an evaluation protocol and diagnostics. It does not change MiewID weights, fusion scoring or clustering.

The metadata-only preparer divides **official time_open train** into approximately 60% fitting, 20% calibration and 20% selection by identity count. Shared encounters and duplicate source/prepared images bind identities into indivisible groups. Actual pool sizes, source coverage, dates and exclusions are saved in the protocol JSON. Each evaluation pool has its own gallery, later known queries and unknown identities absent from that gallery. Undated evaluation images are explicitly excluded; identities with insufficient temporal support may remain gallery-only and are listed. Constraints can prevent an exact target fraction.

Fisher PCA/GMM fitting uses only the fitting pool. Calibration chooses rejection thresholds. Selection loads those thresholds unchanged and cannot fit Fisher models or thresholds. The original official test assignments remain reserved. MiewID's upstream training exposure is still unresolved, so these partitions establish separation within this experiment, not proven independence from pretraining.

Run the following from the repository root, in order. Stop on an error and share it before proceeding.

**1. Focused tests (mocked models; no pretrained downloads)**

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

**2. Create and inspect the metadata-only split**

```bash
venv/bin/python utils/prepare_czechlynx_protocol.py \
  --metadata data/czechlynx/prepared_masked/masked/metadata.csv \
  --out data/czechlynx/protocols/identity_pools_v1.json \
  --seed 42
```

This prints pool sizes and known/unknown query counts; it performs no model inference. Review that inventory before the following runs. Repeating the command with identical inputs is allowed; overwriting a different plan is rejected. Do not change seeds to select a favourable accuracy result.

**3. Calibrate segmented MiewID global-only**

```bash
venv/bin/python tools/evaluate_czechlynx.py \
  --metadata data/czechlynx/prepared_masked/masked/metadata.csv \
  --protocol-plan data/czechlynx/protocols/identity_pools_v1.json \
  --partition calibration \
  --stage development --split time_open --seed 42 \
  --global-model miewid --signals global \
  --cache-dir data/czechlynx/cache/masked \
  --out-dir evaluations/czechlynx/phase2_calibration_miewid_global \
  --cluster-limit 0 --batch-size 8 --workers 2
```

**4. Evaluate selection identities with frozen thresholds**

```bash
venv/bin/python tools/evaluate_czechlynx.py \
  --metadata data/czechlynx/prepared_masked/masked/metadata.csv \
  --protocol-plan data/czechlynx/protocols/identity_pools_v1.json \
  --partition selection \
  --stage development --split time_open --seed 42 \
  --global-model miewid --signals global \
  --thresholds evaluations/czechlynx/phase2_calibration_miewid_global/thresholds.json \
  --cache-dir data/czechlynx/cache/masked \
  --out-dir evaluations/czechlynx/phase2_selection_miewid_global \
  --cluster-limit 0 --batch-size 8 --workers 2
```

Both report directories contain `index.html`, `metrics.json`, per-query predictions, `diagnostics.json`, `fitting.csv`, `gallery.csv` and `query.csv`. Only calibration writes `thresholds.json`. Global-only runs do not extract fitting-pool features; Fisher/GMM settings have no effect on these first runs.

Reports distinguish in-sample calibration from selection validation. Diagnostics include foreground colour-proxy and camera-reference groups, missing metadata, per-source accuracy, recall@10/20/50, shortlist-union recall and size, accepted precision, correct-known coverage and unknown false accepts. Continuous foreground intensity/contrast measurements are descriptive, not validated image-quality thresholds. Near-monochrome does not prove night/infrared. Camera groups describe availability of correct gallery references, not camera-disjoint training. Clustering pair precision/recall is shown when clustering is enabled; the initial commands keep it disabled.

These runs use different identities and gallery/query membership from the historical 60-identity benchmark. Do not attach its `--compare-to` or sampling caps. Accuracy is not directly comparable to the earlier 51.49%. More images can require additional extraction, but compatible cached per-image vectors are reused.

### Phase 2 artifact rules and later full-pipeline comparison

- New role flags require `--stage development --split time_open`. The plan seed must match. Sampling caps and legacy official-test mode are rejected with these flags.
- Selection must use the same signal list, model, extractor/ranking settings, implementation and compatible runtime/model fingerprint as calibration. Keep the same code and environment between paired runs; changed contracts fail before feature extraction. Changing output directory or batch size is allowed. Do not pass phase 2 thresholds to a legacy command.
- A saved Fisher model must identify exactly the new fitting pool, with its metadata/image hashes and protocol binding. Historical gallery-fitted models are rejected. Feature caches remain reusable under their original content/version contracts.
- For a later ensemble comparison, use new calibration and selection output directories; replace `--signals global` in **both** commands with `--signals global global+fisher full-funnel --method ensamble --gv_features disk --pca-dim 128 --gmm-components 256`. Calibration omits `--fisher-model` so PCA/GMM are fitted on the permitted fitting pool. Selection must receive that calibration run's `fisher_bundle.json` as `--fisher-model` and its `thresholds.json` as `--thresholds`. The three extractors each use 256 GMM components. This is the existing fusion pipeline, not calibrated WildFusion.
- Fitting local features can add substantial work. Mask-retention counts in a calibration Fisher report cover the fitting pool as well as that run's gallery/query; selection uses its gallery/query. Establish the global-only reference first.
- Repeated selection experiments make this development validation, not a final untouched test. Final-test adaptation, score-calibrated fusion, illumination experiments, conservative clustering and supervised fine-tuning remain later phases.

## Runs

### Frozen MiewID comparison on segmented images

This first comparison isolates the global embedding model. It uses the same segmented images and development split as the saved `dev_masked_ensamble` run and compares against that run's **global-only** predictions: 61/303 known queries correct (20.13%). The 29.37% result includes Fisher and geometric reranking and is a different pipeline.

The user runs these commands from the repository root. First run the focused regressions (mocked models; no pretrained downloads):

```bash
venv/bin/python -m pytest \
  tests/test_benchmark_embeddings.py \
  tests/test_benchmark_comparison.py \
  tests/test_benchmark.py \
  tests/test_benchmark_report.py \
  tests/test_benchmark_ensemble.py -q
```

Then run MiewID once:

```bash
venv/bin/python tools/evaluate_czechlynx.py \
  --metadata data/czechlynx/prepared_masked/masked/metadata.csv \
  --cache-dir data/czechlynx/cache/masked \
  --out-dir evaluations/czechlynx/dev_masked_miewid_global \
  --global-model miewid --signals global \
  --compare-to evaluations/czechlynx/dev_masked_ensamble \
  --stage development --split time_open --seed 42 \
  --max-identities 60 --max-images-per-identity 25 \
  --overlap-policy purge-query \
  --cluster-limit 0 --batch-size 8 --workers 2
```

Expected partitions: **750 gallery / 560 query images**, including **303 known / 257 unknown queries**. Before neural extraction, `--compare-to` verifies that the completed reference uses identical prepared metadata, input mode, stage, split, ordered gallery/query IDs and retrieval depth. It reads the reference's `predictions_global.json`; it does not rerun MegaDescriptor. A mismatched reference stops the run rather than producing an uncontrolled comparison. Keep the existing reference files unchanged while resuming a run.

`--signals global` skips all local extraction, Fisher fitting and geometric matching. `--cluster-limit 0` skips clustering for this first model comparison. Ensemble/GMM settings therefore do not affect this command. They remain available for subsequent experiments.

The output is `evaluations/czechlynx/dev_masked_miewid_global/index.html`. Its comparison table shows both models' top-1, top-5, mAP@50 and recall at 10/20/50, plus fixed/worsened known queries. `global_model_comparison.json` records those query IDs, counts and reference hashes. The usual per-query predictions, partition CSVs and manifest are also saved. Recall@K means at least one correct gallery identity occurs in the first K candidates; it is not top-1 accuracy. Rejection metrics remain development calibration figures and are not used for this model-ranking comparison.

MiewID is loaded lazily from [`conservationxlabs/miewid-msv3`](https://huggingface.co/conservationxlabs/miewid-msv3) at commit `4f1d7f2b521149e5fe34bb85f377248ce9971a7d`. The adapter uses the publisher's complete embedding model, including GeM pooling and batch normalization, with **440 × 440 RGB input, ImageNet normalization, 2,152-dimensional output and L2 normalization**. It runs in evaluation mode with gradients disabled. Source: [publisher model code](https://huggingface.co/conservationxlabs/miewid-msv3/blob/4f1d7f2b521149e5fe34bb85f377248ce9971a7d/modeling_miewid.py) and [model card](https://huggingface.co/conservationxlabs/miewid-msv3/blob/4f1d7f2b521149e5fe34bb85f377248ce9971a7d/README.md).

The first run downloads missing Hugging Face weights/custom model files. The publisher's constructor also requests its timm backbone initialization, which may download additional weights on first use. The existing environment contains the required libraries: `transformers<5`, timm, torch and huggingface-hub are declared in `requirements.txt`, with safetensors supplied through Transformers' dependencies. No training is performed. Model-loading/extraction failures are recorded as failed manifests, and rerunning the same command reuses completed vector transactions. If GPU memory is insufficient, reduce `--batch-size` and choose a new output directory; verified per-image vectors remain reusable from the same cache directory.

The default `--global-model megadescriptor` keeps the existing MegaDescriptor behavior and compatible cache/run fingerprints. MiewID vectors use a different namespace binding the model state, pinned model/code revision, downloaded Python source hashes, library versions, preprocessing and input mode. `--ckpt` remains a MegaDescriptor checkpoint option and is rejected with MiewID. An optional `--miewid-revision` must be a full commit hash of a compatible model; changing it creates a distinct cache and requires compatible thresholds. Never reuse MegaDescriptor rejection thresholds for MiewID.

Reviewed user-run result (13 September 2026): segmented MiewID global-only reaches **156/303 top-1 (51.49%)**, top-5 62.05% and mAP@50 0.1722; recall@50 is 83.50%. It fixes 106 and worsens 11 known queries versus the same-input MegaDescriptor global baseline. Cross-appearance accuracy remains 5/27, all successes on lynx_203; clustering was disabled. See the [results review](research/czechlynx-miewid-results-2026-09-13.md). The later phase 2 focused suite passed 106 tests (user-reported). Fine-tuning and changes to score fusion or clustering are later phases in the tracked plan.

## Segmented MiewID with the existing full pipeline

This development comparison uses the same 750 gallery / 560 query images as the completed MiewID global-only run. It evaluates global, global+Fisher and full-funnel retrieval, plus the existing query-image clustering diagnostic. The model remains frozen; calibrated WildFusion and clustering algorithm changes remain later work.

Run from the repository root:

```bash
venv/bin/python tools/evaluate_czechlynx.py \
  --metadata data/czechlynx/prepared_masked/masked/metadata.csv \
  --cache-dir data/czechlynx/cache/masked \
  --out-dir evaluations/czechlynx/dev_masked_miewid_ensamble \
  --stage development --split time_open --seed 42 \
  --global-model miewid \
  --signals global global+fisher full-funnel \
  --method ensamble --gv_features disk \
  --pca-dim 128 --gmm-components 256 \
  --fisher-model evaluations/czechlynx/dev_masked_ensamble/fisher_bundle.json \
  --compare-to evaluations/czechlynx/dev_masked_miewid_global \
  --max-identities 60 --max-images-per-identity 25 \
  --overlap-policy purge-query \
  --shortlist-size 50 --rerank-size 10 --depth 50 \
  --cluster-limit 2000 --cluster-gv-budget 2000 \
  --batch-size 8 --workers 2
```

- Every stage uses the prepared segmented inputs; local points outside the provided animal masks are filtered.
- Fisher uses DISK + SuperPoint + ALIKED, equally weighted, with PCA 128 and **256 GMM components per extractor**. The command loads the existing segmented development bundle; compatible local/Fisher and MiewID caches are reused.
- Retrieval takes the union of the top 50 global and Fisher candidates and geometrically reranks the leading 10 using DISK + LightGlue. These match the earlier full-pipeline ranking settings.
- The reference comparison checks the input protocol and compares global-only rankings. Its expected global result remains 156/303 (51.49%); fixed/worsened counts against that same MiewID baseline should be zero. Compare the three stage metrics to assess the added Fisher/GV contribution.
- Clustering uses the existing diagnostic for all 560 queries (below the 2,000-image cap), with a separate budget of up to 2,000 geometric pairs. Its output is provisional, not validated pilot clustering.
- Development calibrates new rejection thresholds for the MiewID configuration. Do not pass the earlier MegaDescriptor or global-only threshold file.
- The reused bundle stays at the original path passed above; the new report does not create a replacement bundle. Its thresholds are written to the new output directory.

Output: `evaluations/czechlynx/dev_masked_miewid_ensamble/index.html`. Review the three top-1/top-5/mAP@50 summaries and `clustering_metrics.json`. The current fusion may improve or reduce ranking accuracy; this run measures that question. Upstream training overlap remains unresolved as documented in the [provenance audit](research/miewid-czechlynx-training-provenance-2026-09-13.md).

Reviewed user-run result (13 September 2026): global/global+Fisher/full-funnel top-1 is 156/155/153 of 303 (51.49%/51.16%/50.50%). Global predictions exactly match the reference. Fisher adds 19 candidate-union hits, but GV fixes 16 and breaks 18 top-1 matches. Cross-appearance remains 5/27. Full-funnel clustering produces eight mixed groups for 60 individuals, with 7.42% pair precision and no unresolved images. Keep global MiewID as the top-1 baseline. See the [full results review](research/czechlynx-miewid-full-pipeline-results-2026-09-13.md). The user ran this benchmark; the later phase 2 focused suite passed 106 tests (user-reported).

Small smoke run (reduced Fisher dimensions, for plumbing and speed checks):

    venv/bin/python tools/evaluate_czechlynx.py \
      --metadata data/czechlynx/prepared/bbox/metadata.csv \
      --out-dir evaluations/czechlynx/smoke_bbox \
      --max-identities 10 --max-images-per-identity 15 \
      --pca-dim 32 --gmm-components 16 --max-descriptors 5000 \
      --cluster-limit 200 --cluster-gv-budget 100 \
      --overlap-policy purge-query

Development baseline (the encounter cap can exceed the nominal image budget):

    venv/bin/python tools/evaluate_czechlynx.py \
      --metadata data/czechlynx/prepared/bbox/metadata.csv \
      --out-dir evaluations/czechlynx/dev_bbox \
      --max-identities 60 --max-images-per-identity 25 \
      --overlap-policy purge-query

Repeat with frame metadata and another output directory for the input-mode comparison. Default Fisher settings are PCA 128 / 256 mixture components; changing these creates a distinct model/cache. Native MegaDescriptor features remain 1536 dimensional. DISK uses 1024 keypoints and its existing 1024-pixel resize. An optional --ckpt uses the existing scratch checkpoint loader and rejects unmatched weights.

### Ensemble development and frozen test

`--method ensamble` selects DISK, SuperPoint and ALIKED with weights of 1/3 each. Explicit subsets such as `--method disk aliked` also use equal weights. The default remains `--method disk`, preserving existing single-model commands and compatible caches.

Each extractor has its own PCA and GMM, fitted on exactly the same development gallery IDs. **The default is 256 GMM components per extractor and PCA dimension 128 per extractor.** The descriptor sampling limits also apply separately to each extractor. The runner matches the existing `combine_fisher_vectors` weighted sum and L2-normalises the result for cosine ranking and clustering. The combined vector remains 65,536 dimensional at these settings.

Run development first, using the same 60-identity sample as the DISK baseline:

```bash
venv/bin/python tools/evaluate_czechlynx.py \
  --metadata data/czechlynx/prepared/bbox/metadata.csv \
  --out-dir evaluations/czechlynx/dev_bbox_ensamble \
  --stage development --split time_open \
  --method ensamble --gv_features disk \
  --pca-dim 128 --gmm-components 256 \
  --max-identities 60 --max-images-per-identity 25 \
  --overlap-policy purge-query \
  --batch-size 2 --workers 0
```

Inspect `evaluations/czechlynx/dev_bbox_ensamble/index.html`. It identifies the extractors, weights, dimensions, GMM component counts and GV method. Compare its metrics with `evaluations/czechlynx/dev_bbox/index.html`. The fitted model bundle is `evaluations/czechlynx/dev_bbox_ensamble/fisher_bundle.json`; it references the three checksummed model files in the cache. Keep those files along with the bundle.

After reviewing development results, run the full held-out test:

```bash
venv/bin/python tools/evaluate_czechlynx.py \
  --metadata data/czechlynx/prepared/bbox/metadata.csv \
  --out-dir evaluations/czechlynx/test_open_bbox_ensamble \
  --stage test --split time_open \
  --method ensamble --gv_features disk \
  --pca-dim 128 --gmm-components 256 \
  --fisher-model evaluations/czechlynx/dev_bbox_ensamble/fisher_bundle.json \
  --thresholds evaluations/czechlynx/dev_bbox_ensamble/thresholds.json \
  --overlap-policy purge-query \
  --batch-size 2 --workers 0
```

Test runs load all three frozen models and never fit on test images. An ensemble cannot use the earlier DISK-only `.pkl` or rejection thresholds. Changing the extractor set, weights, combination, model bundle or GV configuration requires compatible development calibration. GV defaults to DISK + LightGlue for ensembles; `--gv_features superpoint` or `--gv_features aliked` selects another extractor. Use the same GV choice in development and test. For a single extractor, GV defaults to that extractor.

Compatible global and DISK caches are reused; only missing SuperPoint/ALIKED features and models are computed. Each extractor's local features and Fisher vectors have separate caches. Combination uses bounded blocks and a memory-mapped output, keeping the ranking matrix the same size as a single Fisher matrix. Disk usage grows because all three component matrices/caches are retained alongside the combined matrix. Missing extractor/matcher weights are fetched by the existing LightGlue library when the command runs.

For an optional ensemble smoke run, use another output directory, `--max-identities 10 --max-images-per-identity 15 --pca-dim 32 --gmm-components 16 --max-descriptors 5000 --cluster-limit 200 --cluster-gv-budget 100`. These reduced models and thresholds cannot be used for the 128/256 test command above.

### Single-extractor frozen test

For the DISK baseline, use fisher_model.path from the completed development manifest:

    venv/bin/python tools/evaluate_czechlynx.py \
      --metadata data/czechlynx/prepared/bbox/metadata.csv \
      --out-dir evaluations/czechlynx/test_open_bbox \
      --stage test --split time_open \
      --fisher-model <development-model.pkl> \
      --thresholds evaluations/czechlynx/dev_bbox/thresholds.json \
      --overlap-policy purge-query

Run a separate development/calibration pass with --split time_closed before its frozen test. Test runs do not accept identity/image sampling flags. A global-only run uses --signals global and needs a matching global-only development threshold artifact; no Fisher model is required.

Crops, local HDF5 shards, per-image vector caches, fitted models, and pair evidence are reused. Receipts bind image bytes, model weights, relevant implementation files, preprocessing and parameters. A changed or damaged cache fails explicitly. Local shards and vector transactions survive interrupted runs; a rerun completes missing extraction. Ordered query/gallery pair evidence is cached because geometric matching can be directional. Only load Fisher pickle artifacts generated locally by this runner.

Retrieval is query-chunked. Fisher matrices use a memory map, and ranking validates already-normalized vectors in bounded blocks to avoid another full normalized copy; gallery/query slices still need RAM. Clustering builds a dense affinity only for a seeded sample of at most --cluster-limit images (default 2000, hard limit 5000); it does not cluster all 39,760 images. The manifest reports whether the requested graph backend used Leiden or the available NetworkX Louvain fallback. Geometric clustering uses the existing borderline-pair budget, separate from retrieval reranking.

### Ensemble with provided segmentation masks

The masked comparison changes the images used by **MegaDescriptor, DISK, SuperPoint and ALIKED**. It also discards detected local keypoints outside the aligned animal mask before Fisher fitting/transformation and geometric verification. Keypoints remain in crop coordinates; membership uses the nearest pixel (floor(x + 0.5), floor(y + 0.5)), rejecting coordinates outside the image. Thus this is a full segmented-input comparison, not a GV-only ablation.

First prepare from the already downloaded real dataset into a separate directory:

```bash
venv/bin/python utils/prepare_czechlynx.py \
  --modes masked \
  --out-dir data/czechlynx/prepared_masked
```

This processes the full collection; the development sampling flags apply later in the evaluator. Existing bbox/frame metadata and outputs remain available. Run preparation again with the same command to reuse verified PNG pairs and finish missing outputs. Invalid/empty annotation masks stop preparation by default; masks are never silently replaced with all-foreground masks.

Then run the same development sampling and ensemble settings as the bbox comparison:

```bash
venv/bin/python tools/evaluate_czechlynx.py \
  --metadata data/czechlynx/prepared_masked/masked/metadata.csv \
  --cache-dir data/czechlynx/cache/masked \
  --out-dir evaluations/czechlynx/dev_masked_ensamble \
  --stage development --split time_open --seed 42 \
  --method ensamble --gv_features disk \
  --pca-dim 128 --gmm-components 256 \
  --max-identities 60 --max-images-per-identity 25 \
  --overlap-policy purge-query \
  --batch-size 16 --workers 2
```

The sampling settings select the same images before overlap checks. Compare the saved gallery/query IDs and exclusion audits against `dev_bbox_ensamble` when interpreting results; any changed exclusions must be reported. The report at `evaluations/czechlynx/dev_masked_ensamble/index.html` includes per-extractor counts of detected/retained keypoints and images with no retained points. Such images remain in the evaluation: their Fisher vector is zero, and GV with an empty feature set is skipped rather than treated as a failed geometric match.

Masked inputs get separate global/local/Fisher caches. Local caches and pair evidence bind mask provenance and the filtering policy; individual local/Fisher cache entries also bind mask bytes so a changed mask cannot reuse a vector just because RGB pixels are unchanged. New development PCA/GMM models and rejection thresholds are required. Existing bbox models/thresholds are incompatible.

After reviewing masked development results, its frozen held-out test is:

```bash
venv/bin/python tools/evaluate_czechlynx.py \
  --metadata data/czechlynx/prepared_masked/masked/metadata.csv \
  --cache-dir data/czechlynx/cache/masked \
  --out-dir evaluations/czechlynx/test_open_masked_ensamble \
  --stage test --split time_open --seed 42 \
  --method ensamble --gv_features disk \
  --pca-dim 128 --gmm-components 256 \
  --fisher-model evaluations/czechlynx/dev_masked_ensamble/fisher_bundle.json \
  --thresholds evaluations/czechlynx/dev_masked_ensamble/thresholds.json \
  --overlap-policy purge-query \
  --batch-size 16 --workers 2
```

Filtering is applied after detection on the segmented image. Mask boundaries can still influence descriptors; retained keypoints are not proof that every descriptor sees only foreground. This experiment also changes global embeddings and refits the Fisher models, so any gain cannot be attributed solely to GV filtering.

### Bbox retrieval with provided masks only for geometric verification

`--gv-mask-metadata` uses the already-prepared mask PNGs to filter local keypoint centres **only for GV**. Global embeddings, the Fisher ensemble, and the underlying local descriptors still use the original bbox images. No gray-background images are used in this mode. The filter reads the existing bbox local-feature cache through a separate bounded view and never rewrites it. Descriptor patches around retained points can still contain background; this is a test of foreground keypoint selection, not foreground-only descriptor support.

The mask and bbox metadata must contain the same full prepared image IDs, source hashes, crop boxes, identities, encounters, dates (including missing dates), and official splits. Masks are validated against bbox crop dimensions and checksums before extraction or cached pair evidence is used. Mask provenance and filtering implementation are included in the GV evidence namespace and threshold contract. This flag requires bbox inputs and the `full-funnel` signal.

Reuse the existing bbox ensemble bundle and cache for the controlled development comparison:

```bash
venv/bin/python tools/evaluate_czechlynx.py \
  --metadata data/czechlynx/prepared/bbox/metadata.csv \
  --gv-mask-metadata data/czechlynx/prepared_masked/masked/metadata.csv \
  --cache-dir data/czechlynx/cache \
  --out-dir evaluations/czechlynx/dev_bbox_gv_mask_ensamble \
  --stage development --split time_open --seed 42 \
  --method ensamble --gv_features disk \
  --pca-dim 128 --gmm-components 256 \
  --fisher-model evaluations/czechlynx/dev_bbox_ensamble/fisher_bundle.json \
  --max-identities 60 --max-images-per-identity 25 \
  --overlap-policy purge-query \
  --batch-size 16 --workers 2
```

No preparation rerun is needed if the provided-mask PNGs already exist. This reuses the three fitted bbox PCA/GMM models without refitting. Compatible global, local and Fisher caches are reused; GV pair evidence is recomputed under a separate namespace. The report includes raw/retained GV keypoint counts over all gallery/query images, including those with zero retained features. Empty feature sets skip GV and do not remove images or change Fisher vectors. Counts remain available when replaying cached evidence.

Use a new output directory. The run must preserve the bbox baseline's gallery/query IDs and global/global+Fisher predictions; only the full-funnel ranking and its GV-refined clustering can change. New development thresholds are calibrated for this mode, so do not pass the old bbox or fully-masked `thresholds.json`. The original bbox bundle remains the Fisher-model artifact, and the new output contains the new thresholds.

After reviewing development, the corresponding frozen test command is:

```bash
venv/bin/python tools/evaluate_czechlynx.py \
  --metadata data/czechlynx/prepared/bbox/metadata.csv \
  --gv-mask-metadata data/czechlynx/prepared_masked/masked/metadata.csv \
  --cache-dir data/czechlynx/cache \
  --out-dir evaluations/czechlynx/test_open_bbox_gv_mask_ensamble \
  --stage test --split time_open --seed 42 \
  --method ensamble --gv_features disk \
  --pca-dim 128 --gmm-components 256 \
  --fisher-model evaluations/czechlynx/dev_bbox_ensamble/fisher_bundle.json \
  --thresholds evaluations/czechlynx/dev_bbox_gv_mask_ensamble/thresholds.json \
  --overlap-policy purge-query \
  --batch-size 16 --workers 2
```

Regression coverage includes unchanged bbox predictions and model reuse, foreground filtering on both sides of a pair, empty features, metadata/mask integrity, distinct evidence caches, and compatible frozen test thresholds. Run locally (the assistant does not execute tests or benchmarks):

```bash
venv/bin/python -m pytest \
  tests/test_benchmark_gv_masks.py \
  tests/test_benchmark_masks.py \
  tests/test_benchmark_ensemble.py \
  tests/test_benchmark.py \
  tests/test_benchmark_report.py \
  tests/test_fusion.py -q
```

## Full-collection clustering and count controls

Approved and implemented on 14 October 2026. The standalone `tools/evaluate_czechlynx_clusters.py` reads completed phase 2 global references and cached vectors, then evaluates the full eligible gallery/query union: 6,352 calibration images and 4,725 selection images for the current plan. This is development collection discovery, including unknown-to-gallery animals. It is separate from the historical query-sample clustering diagnostic. The user reports the focused suite below passed: **62 passed, 1 warning in 8.50s**, on 14 October 2026. The warning is upstream LightGlue's AMP decorator deprecation. Full calibration is now complete and reviewed; frozen selection remains unrun.

**Current result/next decision:** All 25 profiles completed in 39.497s on 6,352 images / 55 identities, with zero unresolved observations. None supports reliable counts. The 86-group threshold .35 result contains a 6,250-image component spanning every identity; strict/consistent rules produce thousands of fragments. Exact single-pure-group recovery is zero for every configuration. [Full review and receipt](research/czechlynx-clustering-calibration-results-2026-10-14.md). No configuration has been exported. Keep selection untouched while the [approved calibration-only pair/connectivity diagnostic](czechlynx-clustering-diagnostics-proposal-2026-10-14.md) is user-validated; do not run selection merely because calibration completed.

Focused regressions (user-reported passed):

```bash
venv/bin/python -m pytest \
  tests/test_benchmark_clustering.py \
  tests/test_benchmark.py \
  tests/test_benchmark_protocol.py -q
```

Calibration command (completed for the report above):

```bash
venv/bin/python tools/evaluate_czechlynx_clusters.py \
  --metadata data/czechlynx/prepared_masked/masked/metadata.csv \
  --protocol-plan data/czechlynx/protocols/identity_pools_v1.json \
  --reference-dir evaluations/czechlynx/phase2_calibration_miewid_global \
  --partition calibration \
  --cache-dir data/czechlynx/cache/masked \
  --out-dir evaluations/czechlynx/cluster_calibration_global_v1 \
  --threads 4 --chunk-size 64
```

Default profiles: the existing neighbour-graph baseline (k=15, resolution=1, actual Leiden/Louvain backend recorded), plus `threshold` and `consistent` controls for k=15/50 and cosine thresholds .35/.45/.55/.65/.75/.85. The threshold graph retains only sufficiently similar candidate edges. The consistent control additionally checks all cross-component similarities before merging, using bounded blocks and cached incompatibility witnesses. These controls are candidate-limited, greedy experiments; they are not guaranteed optimal clustering. The baseline still constructs dense matrices and can require several GB of RAM. No target cluster count, identity labels or encounter must-links enter inference.

Existing vectors are opened with SQLite `mode=ro` and checked for model/input contracts, actual image/mask bytes, vector checksums, dimensions and finite values. Missing evidence fails rather than dropping images or triggering extraction. Existing colour-proxy diagnostics are read only; missing pixel measurements are stored solely in the new report. The script does not load neural models or modify the detection database.

Inspect `index.html`, `profiles.csv` and `metrics.json`, alongside each profile's images and detailed diagnostics. Review count error, mixed groups, fragmented identities, pair recall and coverage together. All component totals are provisional; singleton proposals are included, but not certified new animals. Zero-vector observations stay unresolved; complete count error is null if any remain. Pair precision/recall with zero denominators is null rather than vacuously perfect. Cross-encounter/camera pair metrics and macro exact identity-group recovery help expose burst-driven purity and fragmentation. Appearance groups remain colour proxies; flank labels remain unknown.

After choosing a profile from **calibration only**, replace the quoted placeholder below with its exact `profile_id` from `profiles.csv`:

```bash
venv/bin/python tools/evaluate_czechlynx_clusters.py \
  --calibration-dir evaluations/czechlynx/cluster_calibration_global_v1 \
  --export-profile 'REPLACE_WITH_CHOSEN_PROFILE_ID' \
  --config-out evaluations/czechlynx/cluster_calibration_global_v1/cluster_config.json
```

The export binds the chosen parameters to a completed, checksummed calibration report, protocol, metadata, model and implementation. Frozen selection uses that exact choice:

```bash
venv/bin/python tools/evaluate_czechlynx_clusters.py \
  --metadata data/czechlynx/prepared_masked/masked/metadata.csv \
  --protocol-plan data/czechlynx/protocols/identity_pools_v1.json \
  --reference-dir evaluations/czechlynx/phase2_selection_miewid_global \
  --partition selection \
  --cache-dir data/czechlynx/cache/masked \
  --config evaluations/czechlynx/cluster_calibration_global_v1/cluster_config.json \
  --out-dir evaluations/czechlynx/cluster_selection_global_v1 \
  --threads 4 --chunk-size 64
```

Selection evaluates one frozen profile and rejects grid/budget overrides. Retrieval `thresholds.json` files cannot substitute for this clustering configuration. Changing relevant code or dependencies invalidates the configuration and requires a fresh calibration report; preserve old reports and choose new output directories. Source/calibration directories are protected from output overlap.

Each run includes `manifest.json`, the exact `collection.csv`, `appearance.json`, `profiles.csv`, `metrics.json` and a portable HTML overview. Profile folders contain `parameters.json`, `assignments.csv` (evaluation truth clearly identified), bounded `edges.csv`, full edge-decision counts, identity/component diagnostics and deterministic review examples with local thumbnails. Calibration additionally contains `calibration_sensitivity.png`; its curves are sensitivity analyses, not confidence intervals or count bounds. Edge retention defaults to 5,000 score-ordered decisions per profile and does not change clustering. Failed runs cannot export configurations.

The [approved scope and remaining milestone](czechlynx-clustering-count-proposal-2026-10-14.md) still require proper night/IR and flank evidence, count uncertainty, review effort and appropriately independent validation. No achieved clustering/count improvement is claimed from implementation alone.

## Calibration pair and connectivity diagnostic

Approved and implemented on 14 October 2026; user reports **41 focused tests passed in 5.19s**, and the completed real diagnostic is [reviewed](research/czechlynx-pair-connectivity-results-2026-10-14.md). All 20,170,776 pairs / 6,352 vectors were scored in 14.913s, with zero unresolved observations. Unthresholded oracle k15 connects 2/55 identities in 382 fragments; k50 connects 10/55 in 196. Raw cutoff .55 yields 98.17% pair precision / 1.16% positive recall. These results expose missing candidate paths and score/graph ambiguity; they do not establish reliable counting. The diagnostic uses the completed calibration report's exact metadata, frozen cache, existing budgets and cutoffs. No new model, local matcher or learned calibrator was introduced. Selection and official test observations are not evaluated.

Focused regression command (user-reported passed):

```bash
venv/bin/python -m pytest tests/test_benchmark_cluster_diagnostics.py -q
```

Diagnostic command (completed for this report; no repeat needed for review):

```bash
venv/bin/python tools/diagnose_czechlynx_clusters.py \
  --calibration-dir evaluations/czechlynx/cluster_calibration_global_v1 \
  --out-dir evaluations/czechlynx/cluster_pair_diagnostics_global_v1 \
  --threads 4 --chunk-size 64
```

The CLI requires a new/empty output directory, including for retries after failure. It rejects selection reports and budget/threshold/sampling overrides. It validates source artifacts and current contracts, loads vectors with SQLite `mode=ro`, checks prepared image/mask bytes and vector checksums, and verifies full reconstructed candidate counts plus saved score/ID prefixes. Cache/protocol/source/code hashes bind the result; concurrent cache or source-manifest edits fail. Source appearance evidence is reused without pixel measurement or source-cache writes. The numerical computation is CPU-only and model-free.

Outputs include `summary.json`, `pair_thresholds.csv`, `pair_histograms.csv`, `identity_pair_statistics.csv`, `identity_pair_histograms.csv`, `candidate_thresholds.csv`, `oracle_connectivity.csv`, `component_comparisons.csv`, `review_pairs.json`, `pair_score_histograms.png`, portable `index.html`, thumbnails and a checksummed completion/failure manifest. All unordered image-pair scores are processed in bounded blocks, with fixed .02-wide bins across [-1, 1] and existing seven-decimal cosine rounding. Compare image-pair recall with macro-identity positive recall and subgroup support. Missing encounter/camera/appearance metadata remains explicit.

Oracle connectivity removes different-identity candidate edges using evaluation truth, then measures each identity's remaining fragments before thresholding and at existing cutoffs. This is a diagnostic of the fixed graph, not achieved identity recovery, a deployable animal count or a bound on all future methods. True single-photo and multi-photo identities are reported separately. Zero vectors remain unresolved; histograms exclude their unscorable pairs, positive recall denominators retain them, complete oracle totals are null when unresolved, and undefined metrics are null. Colour proxies do not establish actual day/night/IR labels or flank linkage.

Review examples select up to eight high-score wrong candidate pairs, eight low-score same-identity pairs and eight same-identity pairs in disconnected unthresholded oracle groups, using fixed score/ID ordering. They are diagnostic extremes with a recorded selection policy, not a representative full visual audit. Direct-edge and component-pair precision have different populations; differences are descriptive. Use the evidence to propose the next method change, then freeze an improved method before evaluating selection. [Approved scope and limitations](czechlynx-clustering-diagnostics-proposal-2026-10-14.md).

## Timed reviewer trial (N3)

Measures real seconds per decision and reviewer error rates for W1 (full ranked matching) vs the N2 candidate, on *calibration* only. [Approved proposal](czechlynx-n3-timed-review-proposal-2026-10-20.md); the reviewer instructions are in the sibling app's `README.md`.

**1. Tests (user-run):**

    venv/bin/python -m pytest tests/test_benchmark_timed_review.py tests/test_benchmark_pruning_safeguards.py tests/test_benchmark_nonmatch_pruning.py -q
    python3 -m unittest discover -s /home/mm-workstation/re-id/czechlynx-matching-review-app/tests -v

**2. Task build** (CPU, no inference; the user runs it or authorizes the assistant):

    venv/bin/python tools/build_czechlynx_timed_review_tasks.py --calibration-dir evaluations/czechlynx/facing_adaptation_calibration_classes_v1/clustering --raw-metadata data/czechlynx/raw/CzechLynxDataset-Metadata-Real.csv --out-dir evaluations/czechlynx/timed_review_tasks_v1_r2 --threads 4 --seed 42

- **Reference check, or nothing is written:** the build must reproduce the saved N2 candidate row (`evaluations/czechlynx/pruning_safeguards_f1_v1`, all ledger kinds) and W1 (`nonmatch_pruning_facing_classes_v1` split 1,804 / 3,507 / 14; live total 5,325). Per-stratum decision counts must sum back to both.
- **Leak checks:** it scans every served image file for identity strings, and the browser payloads for identity labels, image IDs and file names.
- **Outputs:** `tasks.json` and `images.json` (read by the app); `truth.json` (evaluation only; the app never opens it); `strata_counts.csv`; `pre_registration.json`; `contact_sheet.html` (practice images only); `manifest.json` with the task-set fingerprint.

**While the calibration binding differs:** N1b commit `99e9a2da` changed `reid_demo/cluster.py`, which the calibration clustering report binds by hash. The builder and the re-costing re-simulation must therefore run from an export of `7ade3225` plus the four N3 files, with absolute paths. The 20 Oct build did exactly this; see `docs/research/czechlynx-n3-task-build-provenance-2026-10-20.json`.

    git archive 7ade3225 | tar -x -C <export-dir>
    cp reid_demo/benchmark_timed_review.py <export-dir>/reid_demo/
    cp tools/build_czechlynx_timed_review_tasks.py tools/recost_czechlynx_timed_review.py <export-dir>/tools/
    venv/bin/python <export-dir>/tools/recost_czechlynx_timed_review.py --tasks-dir $PWD/evaluations/czechlynx/timed_review_tasks_v1_r2 --export <export.csv> --out-dir $PWD/evaluations/czechlynx/timed_review_recost_v1

The leak scan reads image metadata only: JPEG APPn/COM segments and trailing bytes, PNG text and Exif chunks. Compressed pixel data was dropped from the scan after a chance byte run in one calibration JPEG stopped attempt 2; every served file is still hash-checked.

**3. Sessions:** commit the proposal and the pre-registration before session 1, with the user's permission. Then run the app:

    bash /home/mm-workstation/re-id/czechlynx-matching-review-app/run.sh --tasks-dir /home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/evaluations/czechlynx/timed_review_tasks_v1_r2 --port 8767

**4. Re-costing** (after the sessions, using the app's export):

    venv/bin/python tools/recost_czechlynx_timed_review.py --tasks-dir evaluations/czechlynx/timed_review_tasks_v1_r2 --export /home/mm-workstation/re-id/czechlynx-matching-review-app/exports/matching_decisions_STAMP_ID.csv --out-dir evaluations/czechlynx/timed_review_recost_v1

**Primary estimate:** decision times are recomputed from the raw events; the client's values are a cross-check. The saving is S = 1 − T<sub>N2</sub>/T<sub>W1</sub>, where T<sub>arm</sub> = Σ N<sub>s,k</sub> · mean capped seconds over cells s,k:
- the means come from natural (non-injected) tasks, capped at 120 s;
- N<sub>s,k</sub> are the full-replay decisions per stratum and kind;
- the 95% CI comes from a stratified bootstrap over tasks.

**Verdict:** SUCCESS at S ≥ 40% with per-kind error CIs reported; PARTIAL at 25–40%; REFRAME below 25%.

**Secondary estimates and checks:**
- the pooled per-kind formula, a Horvitz–Thompson total and medians;
- caps of 60 s and none, a W1-favourable bound, and extra-animal at 25 s;
- error rates by kind and stratum, with unsure and unidentifiable rates;
- agreement and second-check reliability;
- a re-simulation of W1 and the N2 candidate at the measured error rates (5 seeds, region folds).

`--allow-incomplete` permits an interim look only; the verdict is then not evaluated.

## Results and interpretation

Each completed directory includes:

- index.html with method comparison, rejection rates, grouping metrics, and deterministic success/error/improvement examples.
- catalogue/index.html with proposed groups and reference identity counts for inspection.
- comparison.csv, predictions.csv/json, per-method predictions, clustering_metrics.json, and clusters.csv.
- manifest.json, data_audit.json, gallery.csv, query.csv, and development thresholds.json.

Copy the whole report directory to share it; the HTML uses local thumbnails and relative links. Source dataset paths remain in the machine-readable provenance files.

Top-1, top-5 and mAP@K are computed only for queries with at least one matching gallery image; unseen queries are counted separately. AP@K divides summed precision at relevant ranks by min(relevant gallery images, K). This is truncated mAP@K, not full-gallery mAP. The fused methods rank the union of the global and Fisher shortlists; shortlist recall measures whether the correct identity reached that union. Full-funnel reranks its leading candidates using the existing inlier/log-score rule.

Unseen false acceptance, known false rejection, and correctly accepted known queries use frozen development thresholds on test. Repeated frames within encounters remain correlated, so raw image-level accuracy should not be read as independent trials. Geometric coverage distinguishes attempted and skipped pairs. Missing verification never demotes a candidate or affinity; actual zero-inlier evidence may do so.

Clustering metrics include ARI/AMI, pairwise precision/recall/F1, merged and split reference identities, and unresolved images. Each unresolved image receives its own metric label rather than treating all noise as one animal. Reference counts describe photographed identities in that sample only. These are offline checks of identification/grouping; they do not establish detector quality, multi-camera ingestion safety, or population abundance estimation.

## Validation

    venv/bin/python -m pytest tests/test_prepare_czechlynx.py \
      tests/test_benchmark.py tests/test_benchmark_ensemble.py tests/test_benchmark_masks.py \
      tests/test_benchmark_report.py \
      tests/test_fisher.py tests/test_fusion.py -q
    venv/bin/python -m pytest tests -q

Tests cover preparation and annotation crops, leakage/overlap policies, unseen queries, exact/truncated metrics, deterministic ranking, fit-only data access, cache invalidation/corruption, missing geometric evidence, noise scoring, threshold provenance, portable reports, and a model-free development/test CLI integration. Real-data smoke, cached replay, development comparisons, and held-out runs are recorded separately; passing mocked/unit tests does not establish lynx accuracy.

The ensemble tests additionally cover equality with the legacy Fisher combination, empty/single-keypoint extractor output, method-specific local/GV caches, bundle validation, shared fitting partitions, changed GV calibration contracts, and development/cached/frozen CLI execution without neural inference. Per the user's workflow, the implementation agent leaves execution of tests and benchmarks to the user; no ensemble accuracy or passing-test claim is implied by these changes.

Masked-input regressions cover compressed/uncompressed RLE, foreground pixel preservation, aligned masks including holes, bbox/masked protocol parity, corrupt preparation receipts, coordinate/descriptor alignment, filtering for all three extractors, zero-descriptor Fisher/GV handling, mask-only cache invalidation, report counts, and masked development/cached/frozen CLI wiring. Masked preparation, test execution and real-data accuracy validation are left to the user.
