# Lynx demo requirements and decision log

Last updated: 20 October 2026. Read this before work on lynx demo readiness, park outreach or the CzechLynx plan. Preserve dated updates and distinguish requirements, evidence, hypotheses and proposed work.

## Active work (live) — check before starting anything

Every agent working here updates this section at start, at each milestone and at the end (user requirement, 18 October 2026). Several agents may be active; claim work here before touching shared files or outputs, and do not run into another agent's output directory.

| Work item | Owner | State | Progress / next step | Files and outputs in use |
| --- | --- | --- | --- | --- |
| Flank-aware grouping ([proposal](czechlynx-flank-grouping-proposal-2026-10-18.md), [results](research/czechlynx-flank-grouping-results-2026-10-18.md)) | Claude Code session, 18 Oct 2026 | **Done, awaiting the user's choice of next step** | 1 ☑ implement · 2 ☑ CLI + tests · 3 ☑ tests (13 new; 132 related) · 4 ☑ calibration run (215 s) · 5 ☑ results review + docs · 6 ☑ committed at the user's request (see git log; not pushed) | Output `evaluations/czechlynx/flank_grouping_v2_fp32/` is complete; do not modify it. No other files are claimed. | `reid_demo/benchmark_flank_grouping.py`, `tools/evaluate_czechlynx_flank_grouping.py`, `tests/test_benchmark_flank_grouping.py` (new); output `evaluations/czechlynx/flank_grouping_v2_fp32/` |
| Review-effort measurement ([proposal](czechlynx-review-effort-proposal-2026-10-18.md), [results](research/czechlynx-review-effort-results-2026-10-18.md)) | Claude Code session, 18 Oct 2026 | **Done; user decided (see Requirement update)** | all steps ☑; committed `0a25c2c8` (not pushed) | Outputs `evaluations/czechlynx/review_effort_v2_fp32/` (failed) and `..._r2/` (complete) are final; do not modify. No other files are claimed. |
| Minimum-review confidence-gated matching ([proposal](czechlynx-minimum-review-proposal-2026-10-18.md), [results](research/czechlynx-gated-matching-results-2026-10-18.md)) | Claude Code session, 18 Oct 2026 | **Done; user chose next steps (a) then (b), see rows below** | 1 ☑ proposal · 2 ☑ approval · 3 ☑ implement · 4 ☑ tests (11 new; 158 related) · 5 ☑ run (135 s) + labelled post-hoc full-scan variant · 6 ☑ results + docs · 7 ☑ committed at the user's request together with (a) (see git log; not pushed) | Outputs `evaluations/czechlynx/gated_matching_v2_fp32/` and `..._posthoc_fullscan/` are final; do not modify. No other files are claimed. |
| **(a) Non-match pruning + safeguards** (cut new-animal and hard-match review cost) — [handoff](czechlynx-nonmatch-pruning-handoff-2026-10-18.md), [proposal](czechlynx-nonmatch-pruning-proposal-2026-10-18.md), [results](research/czechlynx-nonmatch-pruning-results-2026-10-18.md) | Claude Code session (non-match pruning), 18 Oct 2026 | **Done; awaiting the user's next decision.** | Best exact count: 2,211 decisions, 0.77 h per 1,000 photos at placeholder costs (−58% vs W1); targets not all met. 1 ☑ claim · 2 ☑ proposal · 3 ☑ approval · 4 ☑ implement · 5 ☑ tests (18 new; 167 related) · 6 ☑ run (440 s) + labelled post-hoc check · 7 ☑ results + docs · 8 ☑ committed at the user's request together with the minimum-review round (see git log; not pushed) | Outputs `evaluations/czechlynx/nonmatch_pruning_v2_fp32/` and `..._posthoc_pmatch_auto/` are final; do not modify. No other files are claimed. |
| **(b) Facing-aware re-adaptation of the embedding** — [proposal](czechlynx-facing-adaptation-proposal-2026-10-18.md), [results](research/czechlynx-facing-adaptation-results-2026-10-18.md) | Claude Code session, 18 Oct 2026 | **Done; documented negative under the declared rules; awaiting the user's next decision** | R1 and R2 not met by either arm. F1 (animal × facing classes) is a modest real improvement: protocol top-1 68.6% → 71.3%, safe automation 45% → 54%, −9% decisions at its best admissible setting, R3 met. 1 ☑ claim · 2 ☑ proposal · 3 ☑ approval + GPU · 4 ☑ implement · 5 ☑ tests (16 new; 164 related) · 6 ☑ labels + training (F1 2,813 s, F2 2,843 s) · 7 ☑ calibration + review cost · 8 ☑ results + docs · 9 ☑ committed at the user's request on 19 Oct (see git log; not pushed) | Outputs are final; do not modify: `evaluations/czechlynx/fitting_facing_labels_v1/`, `data/czechlynx/cache/miewid_public_fitting_v1/`, `models/czechlynx/miewid_facing_{classes,subcenter}_v1/`, `data/czechlynx/cache/miewid_facing_{classes,subcenter}_v1/`, `evaluations/czechlynx/facing_adaptation_calibration_{classes,subcenter}_v1/`, `evaluations/czechlynx/nonmatch_pruning_facing_{classes,subcenter}_v1/`. No files claimed. |
| **(c) Frozen operating point + one-shot selection evaluation** — [proposal](czechlynx-frozen-selection-proposal-2026-10-19.md), [results](research/czechlynx-frozen-selection-results-2026-10-19.md) | Claude Code session, 19 Oct 2026 | **Done; S1, S2 and S4 not met, S3 met; awaiting the user's next decision** | Selection is spent: 51.5% automated, 0.80% misassigned, count **+6** (pruning duplicates, concentrated in snpa), 59% saving vs W1. 1 ☑ claim · 2 ☑ proposal · 3 ☑ approval · 4 ☑ implement · 5 ☑ tests (10 new; 161 related) · 6 ☑ freeze + pre-registration commit `97571909` · 7 ☑ one-shot selection run (173 s) · 8 ☑ results + docs · 9 ☑ results committed at the user's request (see git log; not pushed) | Outputs are final; do not modify: `evaluations/czechlynx/frozen_review_config_v1/`, `evaluations/czechlynx/frozen_review_selection_v1/`, `data/czechlynx/cache/miewid_facing_classes_v1_selection/`. The one-shot lock blocks any rerun with this fingerprint. No files claimed. |
| **(d) Pitch-readiness analysis and improvement research** (read-only audit, web research, gated roadmap; no code/tests/models) | Claude Code session, 19 Oct 2026 | **Done; awaiting the user's decision on the Now items (N1–N5)** | 1 ☑ claim · 2 ☑ audit receipts/results docs · 3 ☑ web research (monitoring practice, reporting, expert baselines, tools, buyers) · 4 ☑ improvement literature A–H · 5 ☑ write [analysis doc](research/pitch-readiness-analysis-2026-10-19.md) · 6 ☑ dated decision-log entry · 7 ☑ close row. No code, tests, models or outputs touched; committed `161ec5aa` (not pushed) | `docs/research/pitch-readiness-analysis-2026-10-19.md` is final; no files claimed. |
| **(e) N1 ingestion/store integrity + N2 pruning floor and duplicate-merge second check** ([N1](n1-ingestion-integrity-proposal-2026-10-19.md), [N2](czechlynx-n2-pruning-safeguards-proposal-2026-10-19.md), [N2 results](research/czechlynx-n2-pruning-safeguards-results-2026-10-19.md)) | Claude Code session, 19 Oct 2026 | **Done; committed at the user's request (not pushed)** | 1 ☑ claim · 2 ☑ read code · 3 ☑ proposals · 4 ☑ approval · 5 ☑ N1 implemented (static checks pass) · 6 ☑ N1 tests (user-run 19 Oct: focused 87 passed in 16.08 s; full suite 723 passed, 960 upstream warnings, 96.12 s) · 7 ☑ N2 implemented · 8 ☑ N2 tests (18 new; 57 with related suites) · 9 ☑ N2 calibration run (2,188 s; reference reproduced exactly) · 10 ☑ results + docs · 11 ☑ committed 19 Oct: pitch analysis `161ec5aa`, N1 `fc852933`, N2 + decision log in the next commit (see git log); not pushed; disk submodule excluded | Output `evaluations/czechlynx/pruning_safeguards_f1_v1/` is final; do not modify. No files claimed. |
| **N1b pipeline integrity** (stage-done any-row check, forced re-clustering of reviewed rows, cluster-label namespace, model-blind embedding skips) — [handoff prompt](n1b-pipeline-integrity-handoff-2026-10-19.md), [proposal](n1b-pipeline-integrity-proposal-2026-10-20.md) | Claude Code session (N1b), 20 Oct 2026 | **Done; committed at the user's request (see git log; not pushed)** | All four defects verified in code and reproduced on temporary stores (scratch script, no models, not a test run). 1 ☑ claim · 2 ☑ verify defects in code · 3 ☑ proposal · 4 ☑ approval (20 Oct, as written, reserved labels 3a) · 5 ☑ implement (4 source files, 21 new tests, 1 disclosed assertion edit) · 6 ☑ static checks (py_compile; Ruff F no new findings) · 7 ☑ tests (assistant-run at the user's request, 20 Oct: focused 170 passed; full suite 762 passed, 91 s) · 8 ☑ docs (DATA_CONTRACT, AGENTS.md, decision log) · 9 ☑ committed 20 Oct at the user's request (N1b files only; disk submodule and N3 files excluded; not pushed) | `docs/n1b-pipeline-integrity-proposal-2026-10-20.md` (new). Planned, after approval only: `reid_demo/run_demo.py`, `reid_demo/cluster.py`, `reid_demo/embed.py`, `reid_demo/store.py`, `reid_demo/DATA_CONTRACT.md`, `tests/test_run_demo.py`, `tests/test_cluster.py`, `tests/test_embed.py`, `tests/test_store.py`. No benchmark modules, outputs or `review.py`/`ingest.py`/`fusion.py`. |
| **N3 timed reviewer trial** (measured seconds and error per decision, W1 vs assisted workflow, calibration only) — [handoff prompt](czechlynx-n3-timed-review-handoff-2026-10-19.md), [proposal](czechlynx-n3-timed-review-proposal-2026-10-20.md) | Claude Code session (N3), 20 Oct 2026 | **Pre-registered: task set `timed_review_tasks_v1_r2` (fingerprint `580d922d…`) and rule committed 21 Oct at the user's request (see git log; not pushed). Next: the contact-sheet check and the reviewer sessions (user-run)** | 1 ☑ claim · 2 ☑ read code/docs · 3 ☑ proposal · 4 ☑ user approval · 5 ☑ implement · 6 ☑ static checks (py_compile, Ruff F, `node --check`) · 7 ☑ user-run tests (20 Oct: repo 54 passed in 5.02 s, 57 upstream SciPy deprecation warnings; app 12 OK in 1.89 s) · 8 ☑ task build (attempt 3, 20 Oct, 85 s; reference check passed) ☑ pre-registration commit (21 Oct, user's request; not pushed) · 9 ☐ user-run reviewer sessions · 10 ☐ re-costing + results + docs | Created (uncommitted): `reid_demo/benchmark_timed_review.py`, `tools/build_czechlynx_timed_review_tasks.py`, `tools/recost_czechlynx_timed_review.py`, `tests/test_benchmark_timed_review.py`, and the new sibling app `/home/mm-workstation/re-id/czechlynx-matching-review-app/` (no `data/` or `exports/` yet). Outputs: `evaluations/czechlynx/timed_review_tasks_v1/` (attempt 2, failed; manifest preserved); `evaluations/czechlynx/timed_review_tasks_v1_r2/` (attempt 3, the task set in use); planned `evaluations/czechlynx/timed_review_recost_v1/`. Code for builds and re-costing: a scratch export of `7ade3225` plus the N3 files (see the 20 Oct entries). Read-only: the existing input-audit app, its `data/` and `exports/`, `evaluations/czechlynx/pruning_safeguards_f1_v1/`, committed benchmark modules. |
| **N1c artifact freshness** (fusion, catalogue and eval outputs skipped on file presence) — [proposal](n1c-artifact-freshness-proposal-2026-10-20.md) | Claude Code session (N1b), 20 Oct 2026 | **Done; committed at the user's request (see git log; not pushed)** | Verified in code: fusion, catalogue and eval skip on file presence; a failed fusion leaves the fused pass on a stale sidecar; old catalogue pages are never removed. 1 ☑ claim · 2 ☑ verify in code · 3 ☑ proposal · 4 ☑ approval · 5 ☑ implement (run_demo, catalogue; 8 new tests) · 6 ☑ tests (focused user-run 21 Oct; full suite 787 passed + 1 skipped, assistant-run at the user's request) · 7 ☑ docs · 8 ☑ committed 21 Oct at the user's request (not pushed; disk submodule and other sessions' files excluded) | `docs/n1c-artifact-freshness-proposal-2026-10-20.md` (new). Code only after approval. |
| **N1d identity-attachment review questions** (ask whether a proposal belongs to a confirmed animal) — [proposal](n1d-identity-attachment-review-proposal-2026-10-20.md) | Claude Code session (N1b), 20 Oct 2026 | **Done; committed at the user's request (see git log; not pushed)** | Verified in code: confident proposal clusters get no review item; the browser page cannot attach a singleton to an animal. 1 ☑ claim · 2 ☑ verify in code · 3 ☑ proposal · 4 ☑ approval · 5 ☑ implement (review.py; 8 new tests) · 6 ☑ tests (focused user-run 21 Oct; full suite 787 passed + 1 skipped, assistant-run at the user's request) · 7 ☑ docs · 8 ☑ committed 21 Oct at the user's request (not pushed; disk submodule and other sessions' files excluded) | `docs/n1d-identity-attachment-review-proposal-2026-10-20.md` (new). Code only after approval. |
| **N1e MiewID in the demo embedding stage** — [proposal](n1e-miewid-demo-embedding-proposal-2026-10-20.md) | Claude Code session (N1b), 20 Oct 2026 | **Done; committed at the user's request (see git log; not pushed)** | Verified in code: MiewID loads only in benchmark adapters; the demo embed path and runner cannot select it. 1 ☑ claim · 2 ☑ verify in code · 3 ☑ proposal · 4 ☑ approval · 5 ☑ implement (new embed_miewid.py, embed, run_demo; 9 new tests, 1 skipped by default) · 6 ☑ tests (stub device bug fixed; rerun 8 passed + 1 skipped; full suite 787 passed + 1 skipped, assistant-run at the user's request) · 7 ☑ docs · 8 ☑ committed 21 Oct at the user's request (not pushed; disk submodule and other sessions' files excluded) | `docs/n1e-miewid-demo-embedding-proposal-2026-10-20.md` (new). Code only after approval; benchmark modules stay read-only. |
| **N1b–N1e real-run validation** (real MiewID test; `run_demo --smoke` LeopardID2022 run twice) — [results](research/n1-demo-real-run-validation-2026-10-21.md) | Claude Code session (N1b), 21 Oct 2026 | **Done; committed at the user's request (see git log; not pushed)** ([results](research/n1-demo-real-run-validation-2026-10-21.md)) | 1 ☑ claim · 2 ☑ real MiewID test (passed; weights verified real) · 3 ☑ smoke run 1 (ok, 14 s) · 4 ☑ smoke run 2 (ok, 4 s; every fresh stage skipped) · 5 ☑ results + docs | Everything is written under `/tmp/n1_smoke/` (work dir, store, bundle, staging), run from that directory with read-only symlinks to `data/all_datasets.csv` and the kagglehub WildlifeReID-10k cache. Nothing is written into the repo's `data/` or `evaluations/`. |

Open questions for the user: what the audit's left/right labels meant (pose geometry suggests facing direction, not the anatomical flank).

Last completed items: coverage diagnostic (`9065277d`), edge grouping + local pilot (`3f5a0f11`), flank-aware grouping (`64b70e6e`), review effort (`0a25c2c8`); none of these commits is pushed. Minimum-review gated matching and (a) non-match pruning are committed together (`9e589d27`); (b) facing-aware re-adaptation is committed separately (19 Oct). None of these is pushed.

**Agreed plan (user decision, 18 Oct): do (a), then (b). Both are done (18 Oct); the next step is the user's decision.** (a) Calibrated non-match pruning to shorten new-animal and hard-match checks, plus a cold-start guard and reviewer-error safeguards, in a fresh session started from the [handoff prompt](czechlynx-nonmatch-pruning-handoff-2026-10-18.md). (b) Facing-aware re-adaptation of the embedding afterwards. End-to-end integration behind the ingestion/classification stages follows once the review-cost picture is settled. Context: expert-confirmed ranked matching is acceptable only if the bulk is automated (see "Requirement update — 18 October 2026").

## Required outcome — user clarification, 22 September 2026

The demo must support **reliable clustering and a defensible assessment of the number of individual lynx represented in the photographs**. Ranked suggestions and human review can support this outcome, but an assisted-matching-only demonstration does not satisfy the user's goal. Do not quietly defer clustering/count assessment to an optional later product.

Specify the collection, time window, opposite-flank handling, unresolved images and count uncertainty. This does not by itself establish total park abundance or density. Acceptable count error, uncertainty and manual review burden still need agreement; do not invent approved tolerances or assume extensive manual identification satisfies the requirement.

The assistant's preceding summary overemphasised outreach and assisted matching. The user corrected that framing. Exploratory conversations do not close technical gates or justify promising reliable counts. This logging request does not authorise park outreach.

## Requirement update — user decision, 18 October 2026

After the [review-effort results](research/czechlynx-review-effort-results-2026-10-18.md), the user accepted expert-confirmed, embedding-ranked matching as a route to reliable counts, **on condition that the bulk of the work is automated**.

The envisioned product is an **all-in-one system**: camera-trap image ingestion → empty/species classification → re-identification and counting. Some expert review is acceptable, but it must be reduced to a minimum. The system is useful to wildlife researchers only if it **reliably saves them a lot of time**.

Implications:
- Full manual confirmation of every bundle (W1 as simulated, ≈1.8 h per 1,000 photos at placeholder costs) is a fallback, not the target.
- The target is automatic decisions where confidence is demonstrably high, expert review only for uncertain cases, and measured, reported error rates.
- The 22 September requirement for reliable clustering and counts still stands. This update defines how reliability may be achieved, through automation plus bounded review with measured error. It does not relax it.

## Current snapshot — 18 October 2026

This is the working diary and the entry point for current status. Update this snapshot and add a dated entry after each substantive experiment, evidence review or user decision. Record what changed, why, observed results and limitations, the next action and remaining gates. Preserve failed trials and historical entries. Distinguish user-reported execution, saved-artifact inspection, hypotheses and completed validation. Updates happen during active work; this does not imply continuous background monitoring.

| Part | Where we are | Next evidence needed |
| --- | --- | --- |
| Inputs and protocol | All 321 human reviews/receipt complete; fitting split and FP32 preparation reviewed; original image inclusion preserved | Keep quality limitations visible; no unapproved cleanup or exclusions |
| Adapted global model | Calibration top-1: public MiewID 65.6%, v2 68.6%, facing-aware F1 (animal × facing ArcFace classes, 18 Oct) **71.3%**. Same-side cross-encounter NN macro top-1 stays ≈ 67.6% for v2, F1 and F2, so facing-aware training did not sharpen same-side discrimination. F1 raises safe automation 45% → 54% in the review simulation | Decide whether to carry F1 forward. Further embedding gains likely need more or different training data, not class-definition changes |
| Clusters and photographed-individual counts | Calibration (development): exact count at 0.2% misassigned photos with about 2,100 expert decisions (F1, P(match) pruning, W4 τ 0.99, safeguards). **Frozen one-shot on selection (19 Oct, unseen animals):** 51.5% automated, 0.80% misassigned, **count +6** from pruning-induced duplicates (concentrated in snpa), 59% fewer decisions than full manual review, audits cover the truth. S1, S2 and S4 not met | Safer pruning or stronger duplicate sweeps; a second check on duplicate merges; real reviewer timing. Any re-test needs new held-out data (selection is spent; official test locked) |
| Thesis local + global evidence | 18 Oct pilot: DISK+LightGlue adds ≈0 AUC to calibrated edge evidence and does not generalize across animals (identity-cross-fitted local-only AUC .35). User explicitly wants a combined final approach. Earlier combination gave no consistent gain; a flawed prior integration is a hypothesis, not an established cause | Diagnose candidate coverage, patch/mask geometry and score fusion; propose a controlled comparison of adapted global alone against local/global combinations on identical observations |
| Independent and field validation | Upstream exposure unresolved; verified night/IR and flank handling, raw-camera crop quality and independent park transfer unqualified | Freeze an improved method before selection; independently check relevant camera/view/illumination cases and expert-reviewed park data |
| Park demonstration | Research prototype; reliable counting is still unvalidated. Count-error tolerance, uncertainty and acceptable review effort remain unagreed | Demonstrate correct grouping/counts with false merges, splits, unresolved images and measured review effort; protect media/reviews and qualify a bounded supervised pilot |

Training figures describe the inner holdout used for checkpoint selection; the full comparison above is calibration development evidence. See the [completed calibration review/receipt](research/czechlynx-miewid-fp32-calibration-results-2026-10-17.md) and [training review](research/czechlynx-miewid-fp32-training-results-2026-10-17.md). Recognition improved, but reliable grouping/counting was not achieved. No independent park accuracy, defensible delivery date or percentage-ready estimate follows from these results.

**Immediate sequence:** training/calibration reviews are complete; present a concrete local/global candidate/pair-evidence and grouping proposal addressing the observed missing true paths, changed score operating points and false-bridge propagation. Preserve the improved recognition baseline and unchanged inputs; selection stays reserved. The user preference establishes direction, not approval for an unspecified new code scope or assistant-run experiments. Night/flank treatment and reliable count validation remain required.

## Unresolved issues

### Clustering and individual counts

Historical segmented MiewID full-funnel clustering produced eight mixed groups for 60 individuals, with 7.42% pair precision. Phase 2 global selection disabled clustering. Neither supports reliable individual counts.

The 14 October full-collection calibration experiment is now complete and [reviewed](research/czechlynx-clustering-calibration-results-2026-10-14.md): 25 global-only controls on 6,352 images / 55 individuals. None gives reliable grouping/counting; exact single-pure-group identity recovery is zero throughout. The 86-group threshold result hides a 6,250-image group containing all 55 animals; strict and consistent controls fragment the collection into thousands of groups. Selection remains unrun. The subsequent pair/connectivity diagnostic is implemented, user-tested (41 passed) and [reviewed on real calibration data](research/czechlynx-pair-connectivity-results-2026-10-14.md). Both missing candidate paths and misleading pair/graph evidence remain; reliable counts are still unachieved.

Required evidence: cluster and count validation against known identities, including false merges, fragmentation, unresolved cases, pair precision/recall, coverage and per-identity behaviour. A numerically correct total can hide cancelling merge/split errors. Human review requirements must be measured.

### Possible upstream training leakage

Saved phase 2 selection gives 562/881 = 63.79% top-1. This is a verified calculation on saved predictions, **not verified leakage-free accuracy**. MiewID-msv3 exposure to CzechLynx photographs or individuals remains unresolved. Local fitting/calibration/selection separation does not establish independence from upstream pretraining. Neither leakage nor its absence has been established.

Distinguish exact/derived photographs, the same animals in other photographs, and inherited training from earlier checkpoints. The author email attempt was policy-blocked; no response is recorded. Mean per-identity top-1 is 47.28%, and accepted precision is 84.86%; retrieval scores alone do not qualify clustering/counting. Later capture dates alone do not prove unseen identities.

### Local features do not improve the current combined result

On the historical 303-known-query comparison, global/global+Fisher/full-funnel top-1 was 156/155/153. This is a different protocol from the phase 2 63.79% result. The user finds this lack of improvement suspicious and wants it investigated.

Proposed diagnostic scope: feature/mask alignment, coordinate handling, geometric assumptions, correct-match shortlist coverage, score calibration/fusion, subgroup errors and comparisons on identical pairs and independent data. Lack of gain is a concern to investigate, not proof of leakage or proof that local features cannot help. These are hypotheses/checks, not identified causes.

### Nighttime and infrared photographs

There is no validated night-image strategy. Near-monochrome diagnostics are colour proxies, not verified night/IR labels. Opposite-appearance-only selection contains five queries from one identity, with one correct top-1; that does not establish robustness.

Propose trustworthy day/night/IR annotations or audited groups; evaluate night-to-night and day-to-night links separately and their effects on clustering/counts. RGB/grayscale, illumination handling and training are candidate experiments, not established solutions. Do not silently discard night images or count separate day/night groups as different animals.

## Current experiment and status correction

On 22 September the assistant inspected saved `evaluations/czechlynx/roma_pairwise_calibration/manifest.json` and `summary.json`: status `complete`, zero matcher errors, 100 queries. This supersedes older notes saying real-model execution is pending. The manifest does not establish focused-test results. Full evidence/visual review remains open; initial aggregate inspection does not establish a RoMa improvement. Run completion alone demonstrates no fusion, rejection, clustering or count improvement.

## Next work to propose

1. Use the completed/reviewed FP32 calibration comparison to specify the next method change. Internal and calibration recognition improved, but none of the 25 controls achieves reliable counts; avoid another unmotivated training run or selection evaluation.
2. Retain the thesis local/global combination as the intended direction. Review RoMa/DISK-LightGlue evidence, diagnose the lack of local-feature gain, and present the concrete same-observation adapted-global/local/fused comparison. Keep upstream provenance investigation separate.
3. Present grouping/count improvements and a night/IR/flank evaluation, with explicit uncertainty and unresolved-case handling. Reliable counting remains required.
4. Freeze an improved method, then validate on appropriately independent/frozen data. Preserve official test assignments and do not tune on the locked test.
5. Qualify a bounded park demonstration against count error, review effort and safe ingestion/review preservation. Agree tolerances before calling it ready.

These priorities do not approve new implementation or benchmark runs. Present concrete code changes for approval first. Tests/training/benchmarks remain user-run unless explicitly requested otherwise.

## References

- [Implementation plan](czechlynx-implementation-plan.md)
- [Selection review](research/czechlynx-phase2-selection-results-2026-09-16.md)
- [Historical full-pipeline/clustering review](research/czechlynx-miewid-full-pipeline-results-2026-09-13.md)
- [Provenance audit](research/miewid-czechlynx-training-provenance-2026-09-13.md)
- [RoMa experiment](czechlynx-roma-pairwise.md)
- [Proposed pilot gates](research/czechlynx-segmentation-roadmap-2026-09-12.md#8-proposed-pilot-gates): discussion targets, not achieved results or agreed count-error tolerances.

## Dated updates

### 2026-10-21

**N1b–N1e real-run validation (assistant-run at the user's request): both checks passed.** [Results](research/n1-demo-real-run-validation-2026-10-21.md).
- **Real MiewID test** (`REID_DEMO_REAL_MIEWID=1`): 1 passed in 4.80 s; 2,152-D finite output.
  - It emitted 695 "meta parameter … no-op" warnings, so the weights were checked directly: all 700 parameters and 510 buffers are on CUDA with real values (51.1 M parameters).
  - The warnings come from `transformers` initialising the model with placeholder tensors.
- **`run_demo --smoke --datasets LeopardID2022`**, twice, into a fresh `/tmp/n1_smoke`:
  - **Setup:** run from a `/tmp` working directory with read-only symlinks to `data/all_datasets.csv` and the kagglehub WildlifeReID-10k cache, so the repository's June LeopardID2022 demo outputs were not overwritten.
  - **Run 1:** ok in 14 s; 47 crops embedded; markers written.
  - **Run 2:** ok in 4 s. Ingest, species, cluster, catalogue and eval were skipped as fresh; embed ran with 0 embedded and 47 skipped. The headline was identical (8 found vs 18 flank-aware labels, 68.09% grouped, ARI 0.367), and the bundle status was `ok` with nothing degraded.
- **Not exercised:**
  - full-funnel/fusion degradation;
  - reruns after a review or new photos;
  - N1d questions in the browser;
  - a legacy pre-N1 database;
  - MiewID inside `run_demo`;
  - field accuracy.

**N1c, N1d and N1e committed at the user's request (not pushed).** The commit contains only these files:
- the source files `reid_demo/run_demo.py`, `catalogue.py`, `embed.py`, `review.py` and the new `embed_miewid.py`;
- `reid_demo/DATA_CONTRACT.md`;
- the tests `tests/test_run_demo.py`, `test_catalogue.py`, `test_review.py` and the new `test_embed_miewid.py`;
- the three proposals;
- these log edits.

The disk submodule is excluded. The N1b commit is `99e9a2da` and the N3 pre-registration commit is `9d7f0c96`.

**N1c–N1e tests passed (assistant-run at the user's request).**
- **Rerun** `tests/test_embed_miewid.py` after the stub fix: **8 passed, 1 skipped in 3.42 s**. The skip is the real-model test, which needs `REID_DEMO_REAL_MIEWID=1`.
- **Full suite** `venv/bin/python -m pytest tests -q`: **787 passed, 1 skipped, 961 warnings in 94.45 s**; no failures.
  - 788 collected = 723 baseline + 21 (N1b) + 19 (N3, committed as `9d7f0c96`) + 25 (N1c 8, N1d 8, N1e 9).
  - Warnings are the 960 upstream/placeholder warnings of the baseline, plus one more expected placeholder-tile warning from the new catalogue rebuild test.
- **What this covers:** synthetic regressions only. It does not exercise a real conductor run on an existing demo database, a partner archive, the real MiewID weights, or field-crop accuracy.
- **Next:** a real conductor run on a copy of an existing demo database, and a MiewID real-model check (`REID_DEMO_REAL_MIEWID=1`), both only if the user wants them.

**N1c–N1e focused tests (user-run): 282 passed, 2 failed, 1 skipped in 34.57 s.**
- **Command:** `pytest tests/test_run_demo.py tests/test_catalogue.py tests/test_embed.py tests/test_embed_miewid.py tests/test_review.py tests/test_cluster.py tests/test_store.py tests/test_fusion.py tests/test_eval.py -q`.
- **Passed:** all new N1c (8) and N1d (8) tests and every existing test in those suites.
- **The two failures** are both in `tests/test_embed_miewid.py`: `test_embed_records_dispatches_to_miewid` and `test_switch_model_recomputes_all_rows`.
  - **Cause:** a test-stub bug, not an adapter bug. With `device=None` the adapter correctly chose CUDA; the stub model built `torch.arange(...)` on the CPU and added it to the CUDA batch.
  - **Fix:** one line in the stub (`device=x.device`), within the approved N1e test file. No source change. py_compile and Ruff F pass.
- **Skipped:** the real-model test (`REID_DEMO_REAL_MIEWID` not set), as designed.
- **Next:** rerun `tests/test_embed_miewid.py`, then the full suite.
- **Index:** the N3 session has staged its own files for a commit, so the git index is left alone.

**N3 leak-scan test passed (user-run).** `venv/bin/python -m pytest tests/test_benchmark_timed_review.py -q`: **19 passed in 0.84 s**. This includes the new metadata-only leak-scan test.
- The four N3 files and the `_r2` task-set manifest still match the digests in the [provenance receipt](research/czechlynx-n3-task-build-provenance-2026-10-20.json), so the tested code is the code that built the task set.
- **Next:** the pre-registration commit, with the user's permission, before session 1. Then the contact-sheet check and the reviewer sessions (user-run).

**N3 pre-registration committed at the user's request (not pushed; `disk` submodule excluded).** The commit holds:
- the four N3 code and test files;
- the proposal and its handoff note;
- the provenance receipt and the pre-registration copy (identical to the task set's `pre_registration.json`; task-set fingerprint `580d922d9ad498c77bd5c66ec3e7fab98e9b35eb31ba8cf970f584a8ac2f47ba`);
- the N3 section of `docs/czechlynx-benchmark.md`;
- only the N3 lines of this log.

The other session's uncommitted N1c/N1d/N1e rows and entries stay in the working tree. After this commit, no task, stratum, cap or estimator may change.

### 2026-10-20

**N3 task set built (attempt 3); reference check passed; ready for the pre-registration commit.** The user approved the metadata-only leak scan and the `_r2` rebuild. [Provenance receipt](research/czechlynx-n3-task-build-provenance-2026-10-20.json); [pre-registration copy](research/czechlynx-n3-timed-review-preregistration-2026-10-20.json), identical to the task set's own file.
- **Leak-scan change** (`tools/build_czechlynx_timed_review_tasks.py`, `readable_metadata`): scan JPEG APPn/COM segments and bytes after the end-of-image marker, PNG tEXt/zTXt/iTXt/eXIf chunks, and any other format whole. Compressed pixel data is no longer scanned; every served file is still hash-checked. One new test (19 in the file); static checks pass; the user-run test is pending.
- **Code:** the same `7ade3225` export plus the four N3 files (re-copied and re-verified: bindings equal, files identical).
- **Run:** complete in 85 s (N2 replay 74 s). Reference check passed: the N2 candidate row (2,350, all kinds) and W1 (5,325) are reproduced; the strata sum back to both.
- **Task set** `evaluations/czechlynx/timed_review_tasks_v1_r2/` (fingerprint `580d922d9ad498c77bd5c66ec3e7fab98e9b35eb31ba8cf970f584a8ac2f47ba`):
  - 300 main tasks (150 per arm), no stratum shortfall, plus 20 practice tasks;
  - 12 injected; 5,771 served images;
  - 12 blocks of 25 per slot: R1/R3 ABBA, R2/R4 BAAB; all slots get the same 300 tasks.
- **Expected load:** about 1,430 decisions per reviewer if every answer matches the reference (W1 1,054, N2 379).
- **Checks:** aggregate checks found no identity, path, truth or stratum text in the browser payloads; feedback appears only on practice tasks. The app's `TaskSet` loads the set as its default. The input-audit app's hashes are unchanged.
- **Remaining:**
  - user-run tests of the leak-scan change;
  - a visual check of `contact_sheet.html` (practice images only);
  - the pre-registration commit (user's permission), before session 1;
  - the reviewer sessions;
  - re-costing (its re-simulation needs the same export while the `cluster.py` binding differs).

**N3 task build attempt 2 (user-chosen remedy): reference check passed; stopped at the leak scan, a false positive.**
- **Provenance:** run from a scratch export of `7ade3225` (`git archive`) plus the four N3 files. Verified before the run: the calibration-report implementation binding and the saved N2 run's code binding are equal, and the four N3 files are byte-identical to the checkout.
- **Reference check passed:** the traced replays reproduce the saved N2 candidate row (all ledger kinds) and W1 (5,325 = 1,804 / 3,507 / 14); the strata sum back to both.
- **The stop:** the builder then refused calibration image 602 (row 602 of the calibration `collection.csv`). Its original JPEG contains the bytes `LYNx8` at byte 560,531 of 566,749. That is inside the compressed scan data (which starts at byte 609), among random binary, so it is a chance byte sequence, not metadata; the masked crop has no match.
- `evaluations/czechlynx/timed_review_tasks_v1/manifest.json` records status `failed` and is preserved. No tasks were written.
- **Proposed fix:** scan only image metadata (JPEG APPn/COM segments and trailing bytes; PNG text and Exif chunks).

**N3 task build blocked by a binding change from N1b; nothing written.** The user authorized the assistant to run the build. It exited at calibration loading: "Clustering report is incomplete, damaged or from a different implementation". No output directory was created.
- **Cause (verified):** `facing_adaptation_calibration_classes_v1/clustering/manifest.json` binds `reid_demo/cluster.py` by SHA-256. That file is unchanged at `7ade3225` and changed by N1b commit `99e9a2da` (+78/−4). The other seven bound files and all dependency versions still match.
- **Consequence:** every tool that loads this calibration report refuses to run while `cluster.py` differs. That covers the N3 builder and the N3 re-costing re-simulation, and also the existing N2, (a) and frozen-review calibration tools.
- Saved outputs are unaffected. Nothing was changed to work around the check; the remedy is the user's choice.

**N3 tests passed (user-run).** repo `pytest tests/test_benchmark_timed_review.py tests/test_benchmark_pruning_safeguards.py tests/test_benchmark_nonmatch_pruning.py -q`: 54 passed, 57 upstream SciPy L-BFGS-B deprecation warnings, 5.02 s; app `unittest`: 12 tests OK, 1.89 s. The repo run covers the 18 new tests plus the N2 and (a) suites they build on. The app tests ran on a synthetic task set and loopback server. Neither establishes behaviour on real data: the task build's exact reference reproduction, a browser walkthrough and the reviewer sessions remain. Next: the task build (user-run, or assistant-run with the user's authorization), then a pre-registration commit (with the user's permission) before session 1.

**N3 implemented; static checks pass; user-run tests pending.** The user approved the [proposal](czechlynx-n3-timed-review-proposal-2026-10-20.md) as written and asked the assistant to implement it and run static checks only.

New repo files (no existing source edited; committed benchmark modules and saved outputs untouched):
- `reid_demo/benchmark_timed_review.py`:
  - `TracedSafeguardSimulation`, which copies `place` / `duplicate_check` with trace records and per-kind reviewers;
  - `traced_w1`;
  - a catalogue that reconstructs earlier states across merges;
  - strata and decision counts, stratified task sampling with injected opposite-answer tasks, and payload building;
  - event-based timing (pauses, undo, a pause carried into a task), the stratified estimator, bootstrap, secondary estimates, error rates, agreement, `MeasuredReviewer` and the decision rule.
- `tools/build_czechlynx_timed_review_tasks.py`:
  - must reproduce the saved N2 candidate row (all ledger kinds) and W1 (5,325 = 1,804 / 3,507 / 14) exactly, and the strata must sum back to both;
  - scans every served image for identity strings and every browser payload for labels, image IDs and file names;
  - writes tasks, images, truth, strata counts, the pre-registration and a practice-only contact sheet.
- `tools/recost_czechlynx_timed_review.py`: verifies the task-set and export bindings and applies the rule. It re-simulates W1 and the N2 candidate at the measured error rates (5 seeds).
- `tests/test_benchmark_timed_review.py`: 18 synthetic tests.

New sibling app `/home/mm-workstation/re-id/czechlynx-matching-review-app/`:
- files: `server.py`, `static/` (3 files), `run.sh`, `README.md` and `tests/test_matching_app.py` (12 unittest tests);
- port 8767, standard library only;
- it never opens `truth.json` and refuses to store anything inside the input-audit app.

Static checks: py_compile and Ruff F are clean on all new Python; `node --check` passes on `app.js`. No tests, task build or sessions were run by the assistant.

Additions beyond the proposal, disclosed:
- a confirmation bar after the final answer (Enter continues; Backspace undoes); its time is not decision time;
- answers within 0.2 s of a screen appearing are ignored (double presses);
- the contact sheet shows practice images only, so the user (reviewer R1) does not see main tasks;
- test counts are 18 repo and 12 app tests, vs about 14 and 16 estimated.

The input-audit app's database and exports hashes were re-checked and are unchanged.

**N3 timed reviewer trial: claimed and proposed; awaiting approval; no code changed.** [Proposal](czechlynx-n3-timed-review-proposal-2026-10-20.md).

Why: all expert-time figures so far rest on placeholder costs. On calibration, the N2 candidate needs 2,350 decisions vs 5,325 for W1, but on different kinds of work:
- 1,485 of W1's comparisons (28%) are new-animal full scans, likely fast;
- N2 leaves mostly hard comparisons.

Read from the saved rows, the plausible time saving ranges from about 24% to 56% [inferred]. Only measurement can settle P2.

Proposed design:
- **App:** a new sibling app `/home/mm-workstation/re-id/czechlynx-matching-review-app/` on port 8767. The input-audit app (still serving on 8766) is not edited. A read-only hash baseline of its database and exports is recorded in the proposal and will be re-checked at every milestone.
- **Tasks:** precomputed from traced perfect-reviewer replays of W1 and the N2 candidate (region folds). The builder must reproduce the saved rows exactly.
- **Screens:** one candidate per screen, Same / Different / Unsure, each click mapped to one ledger kind. No scores, identities or file names are shown.
- **Sampling:** 300 stratified tasks per reviewer (150 per arm), identical for both reviewers, in ABBA blocks, plus 20 practice tasks. About 20% injected opposite-answer tasks in second checks, pair questions and audits; these count for error rates only.
- **Reviewer time:** about 3.5 h each, in 4 sessions.
- **Pre-registered rule:**
  - primary saving S = 1 − Σ N<sub>s,k</sub> t̄<sub>s,k</sub> (N2) / Σ N<sub>s,k</sub> t̄<sub>s,k</sub> (W1), using stratified mean capped decision times;
  - SUCCESS at ≥ 40% with per-kind error CIs reported; 25–40% partial; < 25% reframe the pitch to error reduction.
  - Unsure and unidentifiable rates and second-check reliability are always reported.
  - Measured error rates are fed back into the sequential simulations.
- **Scope:** 4 new repo files and a new app of about 6 files. Committed benchmark modules and saved outputs stay untouched.

**N1c, N1d and N1e approved as written and implemented; static checks pass; tests pending; not committed.** The user asked for N1c first, then N1e, plus N1d.
- **N1c** (`reid_demo/run_demo.py`, `reid_demo/catalogue.py`):
  - file stages carry `input_columns`, so their marker binds a digest of the rows they read;
  - `depends_on` drives degradation;
  - the manifest lists `degraded`, and SUMMARY.md says so;
  - `CatalogueResult.n_stale_removed`.
  - Disclosed addition: fusion also depends on `fisher`, so a failed fisher degrades to global-only too.
- **N1e** (new `reid_demo/embed_miewid.py`, `reid_demo/embed.py`, `reid_demo/run_demo.py`):
  - MiewID dispatch and cache label;
  - `--embedding-model`/`--embedding-checkpoint`;
  - a DBSCAN-eps warning when MiewID is used with default DBSCAN.
  - Benchmark modules are imported, not edited.
- **N1d** (`reid_demo/review.py`):
  - `identity` items with `proposal_size`/`identity_size`/`similarity`;
  - `_apply_identity`; summary and session counts;
  - the page shows both photos and an identity heading.
- **Tests:**
  - 25 new: N1c 8 (`test_run_demo` 7, `test_catalogue` 1), N1e 9 (`tests/test_embed_miewid.py`; the real-model test is skipped unless `REID_DEMO_REAL_MIEWID=1` and the weights are cached), N1d 8 (`test_review`).
  - No existing test was edited.
- **Static checks:** py_compile passes. Ruff F shows the same 10 findings as HEAD in the changed files, and none in the new files.
- **Docs:** DATA_CONTRACT, AGENTS.md and the proposal statuses are updated.

**N1c, N1d and N1e proposed at the user's request; awaiting approval; no code changed.** These are the three N1b follow-ups, each verified in code first.
- [**N1c artifact freshness**](n1c-artifact-freshness-proposal-2026-10-20.md):
  - fusion, catalogue and eval outputs are reused only if a `meta` marker records the same invocation and a digest of the store columns each one reads;
  - a failed fusion skips the fused cluster pass and fused evaluation, keeps their old artifacts out of the bundle and lists them as degraded (today it can cluster on a stale sidecar or abort);
  - the catalogue removes pages and thumbnails of clusters and records that no longer exist.
  - About 120 lines of source, 7 new tests.
- [**N1d identity-attachment review questions**](n1d-identity-attachment-review-proposal-2026-10-20.md):
  - a new `identity` review item asks whether a proposal (cluster or singleton) is the same animal as its nearest confirmed animal (single-linkage cosine, flank-aware, no tuned threshold);
  - answers: same → merge into that animal; different → logged on the records, then the next candidate; new → confirm as a new animal;
  - identity items go first, largest proposal first.
  - `review.py` only; 8 new tests; existing tests unaffected (their fixture has no embeddings).
- [**N1e MiewID in the demo embedding stage**](n1e-miewid-demo-embedding-proposal-2026-10-20.md):
  - a new adapter `reid_demo/embed_miewid.py` imports the benchmark loaders read-only, so public pinned MiewID-msv3 or an adapted checkpoint with its manifest (e.g. F1) works through `embed --model miewid-msv3` and `run_demo --embedding-model`;
  - same 440-px ImageNet preprocessing as the benchmark; raw 2,152-D vectors;
  - the cache label binds the revision, preprocessing version and checkpoint **bytes**.
  - Default stays MegaDescriptor.
  - Caveats stated: masked-crop evidence vs unmasked field crops, no count-reliability claim, disputed licence for commercial use.
- **Order:** N1c and N1e both edit `run_demo.py` and should be implemented one after the other, N1c first. N1d is independent.

**N1b committed at the user's request (not pushed).** Only N1b files: the 4 `reid_demo` source files, `DATA_CONTRACT.md`, the 4 test files, the proposal, the handoff and the N1b parts of this log. The disk submodule and the N3 session's files and log entries are excluded.

**N1b tests passed (assistant-run at the user's request).**
- **Focused run:** `tests/test_run_demo.py tests/test_cluster.py tests/test_embed.py tests/test_store.py tests/test_review.py`: **170 passed in 24.52 s**.
- **Full suite** `venv/bin/python -m pytest tests -q`: **762 passed, 960 warnings in 91.19 s**, no failures.
  - 762 = the 723 baseline + N1b's 21 new tests + 18 tests in the N3 session's uncommitted `tests/test_benchmark_timed_review.py`, which the full run also collects.
  - The 960 warnings match the baseline: upstream LightGlue AMP, scipy and pycocotools deprecations, plus the expected catalogue placeholder-tile warnings.
- **What this covers:** the synthetic regressions for all four defects. It does not exercise a real partner archive or a real conductor run on an existing demo database.
- **Next:** N1c (file-output freshness), proposal-to-identity review questions and MiewID in the demo embed path remain proposed follow-ups, each needing its own proposal.

**N1b implemented; user-run tests pending; not committed.** The user approved the proposal as written, with reserved proposal labels (option 3a).

Source changes:
- `reid_demo/run_demo.py`:
  - completion checks over every eligible row;
  - invocation markers for ingest and cluster;
  - embed and fisher never runner-skipped;
  - fused pass unforced, with the sidecar sliced to the clustered rows;
  - `--record-id-scheme` passthrough.
- `reid_demo/cluster.py`:
  - marker cleared on non-dry runs;
  - proposal-label offset above the reserved IDs;
  - logged `--force` overrides with a review reset;
  - `n_review_overridden`.
- `reid_demo/embed.py`: model/checkpoint-bound reuse, `n_stale`, and clear-on-failed-stale.
- `reid_demo/store.py`: stage-marker helpers and `update_embedding` type hints.

Tests:
- 21 new tests: 12 run_demo, 5 cluster, 3 embed, 1 store.
- One disclosed edit: the fused spec's `force` assertion, True → False.
- Static checks: py_compile passes. Ruff F shows no new findings; the 7 left in the changed files are pre-existing at HEAD.

Docs: `DATA_CONTRACT.md` and AGENTS.md are updated.

Commands for the user:
- `venv/bin/python -m pytest tests/test_run_demo.py tests/test_cluster.py tests/test_embed.py tests/test_store.py tests/test_review.py -q`
- then the full suite (baseline 723 passed).

**N1b pipeline integrity: claimed and proposed; awaiting approval; no code changed.** [Proposal](n1b-pipeline-integrity-proposal-2026-10-20.md).

All four remaining `reid_demo` integrity defects were verified in code and reproduced on temporary stores (scratch script, synthetic vectors, stubbed model; not a test run):
- **Stage completion:** with species set on 1 of 9 rows, `_store_stage_done` returns true.
- **Forced fused re-clustering:** a reviewed row's cluster 999 became 0 with status still `confirmed`. Without `force`, the fused pass fails, because the fusion sidecar (9 × 9) covers reviewed rows too. That is why `force=True` was there.
- **Label namespace:** after a reviewer confirmed identity 1, a rerun gave another animal's rows label 1.
- **Embedding reuse:** embedding with checkpoint B after model A embedded 0 rows and skipped 3, which stayed on A's cache.

Proposed fixes:
- **Completion:** a stage is done only when every eligible row has output. Ingest and cluster also need a `meta` completion marker for the exact invocation. Embed always runs and is itself incremental.
- **Fused pass:** runs unforced, with the sidecar sliced to the rows being clustered. An explicit `--force` is logged in `extra_json['review_overrides']` and resets the row to `unreviewed`.
- **Labels:** proposal labels start above every cluster ID that the run does not rewrite, so confirmed identity numbers are never reused and stay stable under review merges.
- **Embeddings:** reused only if they live in the cache for the requested model and checkpoint and the key exists. Otherwise they are recomputed; a stale one whose recompute fails is cleared.

Other points:
- No schema change. A new conductor option, `--record-id-scheme`, is needed because ingest reruns once on existing databases.
- **Tests:** 19 new tests and 1 disclosed assertion edit (fused spec `force` True → False).
- **Visible consequence:** after a rerun, unreviewed companions of reviewed rows become separate proposals until attached by review. Today the same relabelling happens and collisions are silent.
- **Follow-ups proposed separately:** artifact freshness for fusion, catalogue and eval (N1c); proposal-to-identity review questions; MiewID in the demo embed path.

### 2026-10-19

**(e) N1 tests passed (user-run).**
- Focused `tests/test_store.py tests/test_ingest.py tests/test_species_filter.py`: **87 passed in 16.08 s**.
- Full suite `venv/bin/python -m pytest tests -q`: **723 passed, 960 warnings in 96.12 s**. The warnings are upstream LightGlue AMP, scipy L-BFGS-B and pycocotools NumPy deprecations, plus expected catalogue placeholder-tile warnings.
- This validates the synthetic collision, re-ingestion, scheme and SpeciesNet regressions. It does not exercise a real multi-camera archive.
- N1b items remain open (stage-done any-row check, forced re-clustering of reviewed rows, cluster-label namespace, presence-only embedding skips).
- Committed at the user's request in three commits on `feat/reid-demo-pipeline` (not pushed; disk submodule excluded): pitch analysis `161ec5aa`; N1 `fc852933`; N2, its docs and these log updates in the following commit.

**(e) N2 run and reviewed: adopt the merge check; the floors did not help; pruning on new data stays open.** The user approved N1 and N2 as written and authorized the N2 CPU tests and run. [Results](research/czechlynx-n2-pruning-safeguards-results-2026-10-19.md).
- **Run:** 794 simulations on calibration (F1), identity and leave-one-region-out folds (Carpathians, 17 animals, vs Bohemian Forest, 38 animals); 2,188 s. The defaults reproduce the saved F1 rows exactly (2,116 / 2,485 decisions); live W1 is 5,325.
- **Merge check (second reviewer confirms each duplicate merge):**
  - halves misassignment under reviewer error (region folds, 3%: 2.8% → 1.6%; 5%: 4.5% → 2.5%);
  - wrong merges 5.2 → 0.2 per run; slightly fewer decisions overall.
  - Residual count errors become surplus animals instead of cancelling.
- **Cosine floor:** never found a hidden true match (0 in 49–206 extra comparisons per run); costs +4–9%.
- **min_shown 10:** removed 1 of 2 Carpathian pruning duplicates under identity folds (+3% cost); no effect under region folds.
- **Region holdout did not reproduce the selection failure:** 0 pruning duplicates at 0% error in every setting, so R1 passed trivially. Region-trained models are less sharp (Brier .0126 vs .0039), prune less (2,350 vs 2,116 decisions) and automate less (44.7% vs 48.4%).
- **Declared rules:** 9/36 settings pass R1–R4, all with the merge check. Mechanical candidate: ε 0.001, no floors, merge check on: 2,350 decisions (−55.9% vs W1), 0.24% misassigned, count 0; at 3% error 1.07% / |count| 1; at 5% 1.79% / |count| 1. R3 differences among merge-check settings are mostly seed noise.
- **Audits** still fail with a noisy auditor (unchanged, separate item).
- **Next:** adopt the merge check. Choose between ε 0.001 and the more conservative ε 0.0003 (2,512 decisions) when pre-registering for the blind partner test (X1). Pruning fragility on unseen animals needs new held-out data.

**(e) N1 implemented; user-run tests pending.** Changes:
- path-scoped record IDs with a per-dataset scheme in `meta`; a `legacy` flag required for pre-existing datasets, and collisions refused;
- a stage-owned `ingest_upsert_records` that never resets reviews, embeddings or clusters, flagging reviews on geometry change;
- path-aware CSV metadata and SpeciesNet matching.

About 10 existing `tests/test_ingest.py` lookups were updated as disclosed, and `test_ingest_reuses_legacy_crops` now passes `record_id_scheme="legacy"`. New tests: 8 store, 8 ingest, 1 SpeciesNet. Static checks: py_compile and Ruff F, no new findings. Command for the user: `venv/bin/python -m pytest tests/test_store.py tests/test_ingest.py tests/test_species_filter.py -q`, then the full suite.

**(e) N1 and N2 proposals drafted; awaiting approval; no code changed.**

[N1](n1-ingestion-integrity-proposal-2026-10-19.md) makes ingestion collision-safe and re-ingestion review-preserving:
- **Record IDs:** scoped to the source path (`{stem}__{sha1(relpath)[:10]}__crop{n}`). The scheme is stored per dataset in the `meta` table; there is no schema change. Existing DBs need an explicit `legacy` flag, and collisions raise.
- **New `store.ingest_upsert_records`** with stage-owned columns. Reviews, embeddings and clusters are never reset. If the geometry changes, derived values are cleared and the review is kept but flagged for a re-check.
- **Disambiguation:** CSV metadata and SpeciesNet matching are disambiguated by path.
- **Scope:** 5 files; about 10 test lookups in `tests/test_ingest.py` are updated (disclosed).

[N2](czechlynx-n2-pruning-safeguards-proposal-2026-10-19.md) adds pruning floors and a merge check:
- **Floors:** a minimum number of shown candidates, plus an appearance-only cosine top-k floor.
- **Merge check:** a second reviewer must confirm each duplicate merge.
- **Test design:** calibration only, with a leave-one-region-out fold: Carpathians, 17 animals vs Bohemian Forest, 38 animals; verified from metadata.
- **Rules N2-R1–R4** are declared in the proposal.
- **Scope:** 3 new files; committed modules are untouched because the frozen fingerprint binds their hashes. About 40 CPU-minutes.
- **Reference check:** must exactly reproduce the saved F1 rows (2,116 / 2,485 decisions).

**(d) Pitch-readiness analysis done (read-only; no code, tests, models or outreach).** [Analysis](research/pitch-readiness-analysis-2026-10-19.md).

Verdict:
- **An automated count is not pitchable now** (framing (b) unsupported).
- **Honestly pitchable:** a research collaboration on an expert-verified identification workflow with an audit trail (framing (a)). Simulated support on *selection*: 59% fewer decisions, 0.80% misassigned. It fails on the count (+6) and has no real-reviewer, Croatian or night/IR evidence.

New external facts (sources in the analysis):
- **Croatian volume is small:** 4,201 lynx photos over 14 seasons and 225 individuals (Bini et al., *J. Zool.*), so roughly 300 per season. The expert-hours saving on lynx matching is therefore a weak selling point.
- **The Croatian experts have tested AI:** experienced manual 92.7% vs MiewID operated by an expert 71.7% (Blašković et al. 2025, *J. Zool.*). Pitch alongside human expertise, not instead of it.
- **Expert misidentification is documented:** snow leopards 12.5% of occasions and +35% abundance; leopards −22% individuals after a computer-assisted re-check. A catalogue "ghost audit" is a concrete low-risk first pilot.
- **Competition:** Whiskerbook (MiewID) is already used by French and Croatian teams. MiewID v4 is announced (+12% for lynx) but its weights were not found public.
- **AnimalCLEF 2026:** lynx anchoring is carried by global descriptors; LightGlue alone is weak there too.

Proposed pitch criteria P1–P7 (revising S1–S4):
- ≤ 2% misassigned and |count error| ≤ max(1, 5% of N);
- ≥ 50% fewer decisions and ≥ 40% less **measured** wall-clock time;
- robust at **measured** reviewer error;
- night/IR-stratified results and raw frames;
- all on blind partner data photographed after 15 Oct 2024.

Proposed Now items, each needing approval:
- **N1:** ingestion/store integrity fixes (bugs still present at `reid_demo/ingest.py:307`);
- **N2:** conservative pruning floor + duplicate-merge second check, chosen by leave-one-source-out on calibration;
- **N3:** timed matching review mode in the sibling app;
- **N4:** same-observation local/global diagnostic with a pre-registered refutation rule;
- **N5:** research conversation with the Zagreb Veterinary Faculty lynx team (outreach is the user's decision).

Selection remains spent and the official test locked. Nothing was committed.

**(c) one-shot selection evaluation done: the frozen operating point partly generalizes; S1, S2 and S4 not met.** The run completed in 173 s with no retry. The pre-registered fingerprint (`f2402829…`, commit `97571909`) was verified before and after.

Primary result (τ 0.99, perfect reviewer) on 4,725 photos of 55 unseen lynx:
- **Automation:** 51.5% of bundles at 0.80% misassigned.
- **Count:** +6 (61 groups). There were 10 pruning-duplicate events vs 3 on calibration; duplicate checks merged 3. The surplus is concentrated in snpa (+6; foe_carpaths 0).
- **Cost:** 2,141 decisions, 59.2% fewer than W1 (5,244). This is 453 per 1,000 photos vs the 416 limit; selection is costlier per photo even for W1 (smaller bundles).
- **Time:** 1.01 h per 1,000 photos at placeholder costs, 0.47 at fast-click.
- **Audits** cover the truth (S3 met).
- **3% reviewer error:** 3.6% misassigned (S4 not met). 6 of 10 duplicate merges were wrong, because pair questions have no second check.
- **Recognition:** protocol top-1 on selection is 69.9% vs 63.8% for public MiewID (Sept).

Nothing was retuned or rerun on selection. **Selection is now spent; the official test split stays locked.**

[Results](research/czechlynx-frozen-selection-results-2026-10-19.md). Implications:
- Pruning does not transfer reliably; it needs a more conservative rule or a stronger duplicate sweep.
- Duplicate merges need a second check.
- Cost targets should be stated relative to W1.
- Any revised method needs new held-out data, such as expert-verified park data.

Results were committed at the user's request (not pushed).

**(c) frozen and pre-registered before any selection photo was read.** The freeze ran on calibration only, in 305 s.

- **Frozen configuration fingerprint:** `f24028291c0c02f9e901ee035ccc3ef23e1291695bdee15307ae893e5828951c`.
- **`config.json` SHA-256:** `6e4c9898a21caff9b19c354a8de39656dc3fb1394f94ad1e0a0b54f8477e2fea`. A copy is committed as [czechlynx-frozen-review-config-2026-10-19.json](research/czechlynx-frozen-review-config-2026-10-19.json).
- **Models:** W4 confidence model on 1,790 replay records, P(match) model on 56,987 records, facing classifier on 1,115 pose-labelled photos; frozen risk references for the four signals. All are JSON.
- **Operating points:** F1 embedding, P(match) order, ε 0.001, τ 0.99 primary and 0.98 secondary, default safeguards.
- **Calibration reference for S2:** 333.1 decisions per 1,000 photos, so the limit is ≤ 416.4.
- **In-sample sanity check** (plumbing only, not evidence), τ 0.99 vs cross-fitted: 48.8% vs 48.4% automated, 0.19% vs 0.24% misassigned, count 0 vs +1, 2,064 vs 2,116 decisions.

Code and the config copy were committed before the selection run. The run verifies these hashes, and any later code change invalidates the freeze.

**(c) approved.** The user approved the proposal as written: τ 0.99 primary, τ 0.98 secondary, and the no-selection rule lifted for this one run. The assistant may run the CPU tests and the freeze, commit the freeze record, and run the GPU selection evaluation.

**(c) proposed: freeze one operating point and evaluate it once on selection.** The user chose this next step.

[Proposal](czechlynx-frozen-selection-proposal-2026-10-19.md):
- **Frozen point:** the F1 embedding; P(match) order with ε 0.001; W4 auto-accept at **τ 0.99 primary** (τ 0.98 pre-declared secondary); all (a) safeguards.
- **Models:** trained on all of calibration and frozen as JSON. Risk percentiles are frozen against calibration distributions.
- **Pre-registration:** the configuration fingerprint is committed before any selection photo is read.
- **One-shot run on selection** (4,725 photos, 55 unseen lynx; earlier use was only the 16 Sep public-MiewID recognition run): simulations at 0% and 3% reviewer error, the W1 reference and protocol recognition. Breakdowns per animal, with and without lynx_287 (22% of photos), and by source.
- **Declared criteria:** S1, ≤1% misassigned and |count error| ≤1; S2, ≤416 decisions per 1,000 photos and ≥50% saving vs W1; S3, audit coverage; S4, robustness at 3% reviewer error.
- **Locks:** one-shot locks in code; no retuning on selection.

Approval would lift the standing no-count-configuration / no-selection rule for this one run. Awaiting approval and authorization; no code or runs yet.

**(b) committed (`019c839b`, not pushed).** The user then asked for the assistant's view on next steps.

**Assistant recommendation (not approved; the order is a suggestion):**
1. **Freeze and test once on selection.** Freeze one operating point: F1 embedding + P(match) order, ε 0.001, W4 auto-accept τ 0.98–0.99, and the safeguards. Run it once on the reserved selection pool for honest, non-calibration numbers. This needs a count-configuration export, which has been forbidden so far, so it needs explicit approval.
2. **Time real reviewers.** Measure real reviewer time on the pruned ranked workflow in the existing sibling review app (or a minimal extension). The fast-click cost model already meets ≤ 0.5 h per 1,000 photos; placeholder costs do not. Real timing decides whether the time-saving claim holds.
3. **Make audits work with an imperfect auditor:** consensus or bias-corrected estimates. Without this, count uncertainty is not defensible once reviewers make errors.
4. **Run end to end on raw frames.** The current pipeline starts from annotation-assisted masked crops. Run raw camera frames → detection/segmentation → species → re-ID review on CzechLynx raw images and Medvednica, and measure the degradation.
5. **Later, data rather than loss tweaks:** more or different lynx training data and night/IR handling. The upstream-exposure question remains open.

Readiness for outreach still depends on 1–4.

### 2026-10-18

**(b) facing-aware re-adaptation run and reviewed: a documented negative under the declared rules; F1 is a modest real improvement.** The user unpaused (b) and authorized the GPU steps.

Runs:
- **Fitting facing labels** (94 s): 2,415 from pose, 11,431 predicted, 16.1% "other"; pose AUC .9996.
- **Training:** F1 (358 animal × facing classes; 2,813 s; epoch 7) and F2 (sub-center ArcFace, K = 3; 2,843 s; epoch 9). Both use v2's exact inner split.
- **Evaluation:** calibration comparisons and the unchanged (a) review-cost simulation for each arm. Adapted weights were verified loaded by state hash.

Results:
- **R1 not met:** same-side macro NN top-1 67.6% (v2), 67.6% (F1), 67.9% (F2).
- **R2 not met:** at the declared best setting, decisions are 2,211 (v2), 2,116 (F1, −4%) and 2,173 (F2). Max safe automation is 45% / 54% / 47%. F1's best admissible point is 2,015 decisions (−9%) at 53.5% automation.
- **R3:** F1 meets it (3% reviewer error: count +1, 1.9% misassigned vs v2's 2.3%); F2 fails it (count +3).
- **Recognition:** protocol top-1 is 71.3% (F1) and 69.8% (F2) vs 68.6% (v2).

Interpretation: merging flanks into one class was not what limited the embedding; the limit lies elsewhere, with data and image conditions plausible but unproven. Per the declared rule, no further training variants follow without a new proposal. [Results](research/czechlynx-facing-adaptation-results-2026-10-18.md).

Next options, not approved:
- carry F1 forward as the working embedding;
- a timed reviewer UI trial;
- engineered audits;
- more or different training data;
- end-to-end integration.

Nothing is committed; no count configuration, selection or outreach.

**(b) implemented and tested.** Five new files:
- `reid_demo/benchmark_facing_adaptation.py`
- `tools/label_czechlynx_fitting_facing.py`
- `utils/finetune_czechlynx_miewid_facing.py` (the v2 FP32 loop, copied)
- `tools/evaluate_czechlynx_facing_adaptation.py`
- `tests/test_benchmark_facing_adaptation.py`

All 16 new tests pass; across eight related suites, 164 pass; Ruff F is clean.

Implementation detail: R1 is computed with **common facing labels** for all three models (pose, else the public-MiewID cross-fitted prediction), so the v2 same-side baseline is recomputed under those labels. The proposal's 66.7% used each model's own predicted facing.

**(b) approved.** The user approved the facing-aware re-adaptation proposal as written. They authorized the assistant to run the CPU tests and the GPU steps: fitting facing labels, both training arms (F1 and F2) and calibration extraction. They also authorized the CPU review-cost evaluation.

**(a) and the minimum-review round committed (`9e589d27`, not pushed; disk submodule excluded); (b) facing-aware re-adaptation claimed and proposed.** The user asked for the commit and then the (b) proposal.

Read-only checks on saved calibration vectors (no inference):
- Public MiewID predicts facing at pose AUC .9995, with 83% of photos confident; adapted v2 at .9946, with 74% confident. The identity-only adaptation reduced facing information.
- Cross-encounter nearest neighbours are the same animal for v2 at macro 67.1% (any side), 66.7% (same side) and 18.2% (opposite side); public MiewID: 62.4% / 62.1% / 14.9%.
- Only 2,415 of 16,504 fitting photos (14.6%) have pose keypoints.

[Proposal](czechlynx-facing-adaptation-proposal-2026-10-18.md):
- **Facing labels for fitting:** pose first, then an identity-cross-fitted facing classifier on public-MiewID fitting vectors (GPU extraction), else "other".
- **Two arms, otherwise identical to v2:** F1 explicit (animal, facing) ArcFace classes and F2 sub-center ArcFace (K = 3). Both start from public MiewID on the same inner split, in FP32 and with no flips; checkpoints are selected by same-facing inner top-1.
- **Evaluation:** the calibration comparison composed like v2, then the unchanged (a) review-cost simulation.
- **Declared rules:** R1, same-side macro nearest-neighbour top-1 +3 points; R2, ≥15% fewer decisions than 2,211, or ≥55% safe automation; R3, no worse under 3% reviewer error.
- **Size:** five new files and about 2 GPU-hours.

Awaiting approval and GPU authorization.

**(a) Non-match pruning run and reviewed: expert time for an exact count roughly halves; targets still not all met.** The run (`evaluations/czechlynx/nonmatch_pruning_v2_fp32/`) completed in 440 s with 148 simulations. The reference check passed: W1 and the saved full-scan W4 rows reproduce exactly.

- **P(match) model:** identity-cross-fitted, both sides excluded from training. AUC .998 with space-time features vs .988 appearance-only; rank-1 share 93.2% vs 82.9% in cosine order.
- **Re-ranking is the main lever:** P(match) order cuts hard-match comparisons 2,379 → 417. Pruning at ε = .001 cuts new-animal comparisons by 58% with no extra duplicates; larger ε adds 2–16 duplicates.
- **Best exact-count setting** (P(match) order, ε .001, τ .99, safeguards on): count error 0, 0.19% misassigned, 45% automatic. It takes 2,211 decisions: 0.77 h per 1,000 photos at placeholder costs and 0.35 at fast-click costs. That is −58% vs W1 and −50% vs the full-scan W4. Of those decisions, 990 are one-click confirmations.
- **3% reviewer error:** count +1 and 2.3% misassigned, vs 8–9.5% before. Without safeguards, the count errors are +15 to +37. The disagreement check and double confirmation do most of the work. Duplicate merges lack a second check and cause wrong merges.
- **Appearance-only models:** count errors of +10 to +59. Space-time context is essential.
- **Audits:** they cover the truth with a perfect auditor, but with a 3% auditor the automation audit covers it in 50% of runs and the count audit in 0–3%.
- **Post-hoc (outside the grid):** gating auto-accept on P(match) is worse than W4 (41% automatic at 0.82% vs 45% at 0.19%).

**Declared outcomes:**
- Targets: ≥70% automatic and ≤0.5 h at placeholder costs are not met. Misassignment, count and audit-coverage targets are met by the best setting. There is no candidate operating point.
- D1 pruning works: not met (58% < 75%).
- D2 re-ranking helps hard matches: met (−82%).
- D3 safeguards earn their cost: met in 2 of 4 combinations.
- D4 cold-start guard: mechanically met but uninformative (no collapse at τ .99 even without the guard).

[Results](research/czechlynx-nonmatch-pruning-results-2026-10-18.md). Next options, not approved:
- (b) facing-aware re-adaptation, already agreed as next, to raise automation and the rank-1 share;
- a timed reviewer UI trial to settle the cost model;
- consensus or bias-corrected audits;
- a second check on duplicate merges.

Nothing is committed; no count configuration, selection or outreach.

**(a) Non-match pruning approved.** The user approved the proposal as written, kept the carried-over targets, and authorized the assistant to implement it and run the CPU tests and the calibration run.

**(a) implemented and tested.** Three new files: `reid_demo/benchmark_nonmatch_pruning.py`, `tools/simulate_czechlynx_nonmatch_pruning.py` and `tests/test_benchmark_nonmatch_pruning.py`. No existing source was edited. All 18 new synthetic tests pass; across the eight related suites, 167 pass; Ruff F is clean. The tests cover:
- identity-free features;
- both-sides identity exclusion in P(match) training;
- W1 reproduction;
- automation error propagation, with no automatic new animals;
- pruning-induced duplicates, with full-scan, duplicate-check and double-confirmation recovery;
- the cold-start guard and the disagreement check;
- merges, ledger costs and count-audit arithmetic;
- the CLI contracts, including a failed manifest kept on a reference mismatch.

Implementation choices within the approved scope:
- The count audit scans each sampled animal against the older catalogue animals only, ranked and stopping at a "yes"; duplicates are defined against an older entry.
- The pruned scan always shows at least one candidate.
- The CLI compares the references with a live W1 and, on real data, with the saved W1 and full-scan W4 numbers. If they differ, the run is marked failed.

**(a) Non-match pruning claimed; proposal drafted.** A fresh session claimed item (a) from the handoff, read the five 18 October results and the reusable code, and drafted the [proposal](czechlynx-nonmatch-pruning-proposal-2026-10-18.md). No code, tests or runs yet.

**Ceiling arithmetic** (from saved W1 numbers; indicative, not a result):
- W1's 5,269 comparisons split into 1,485 for new-animal scans, about 1,451 rank-1 confirmations and about 2,333 for roughly 299 hard matches.
- Pruning can remove most of the new-animal cost, but not hard-match cost, because a scan stops at the first "yes".
- At placeholder costs, ≤ 0.5 h per 1,000 photos therefore needs hard-match comparisons cut by about two thirds, through better ranking. At "fast-click" costs, about 20% would be enough.
- ≥ 70% automation looks out of reach for the unchanged W4 confidence model.

**Design:**
- **P(match) model:** a per-candidate model with appearance, rank, reciprocal-rank, facing, risk and space-time features, including a same-day distance conflict. It is identity-cross-fitted, excluding both the part's animal and the candidate's animal from training.
- **Pruned scan:** in cosine or P(match) order, stopped when no remaining candidate has P ≥ ε (ε 0–0.03).
- **Automation:** the existing W4 auto-accept (τ grid).
- **Safeguards:** a cold-start guard (N = 10 animals, B = 100 bundles); second-reviewer double confirmation of new animals; a model–expert disagreement check (added); periodic duplicate checks; a second-reviewer automation audit; and a count audit by full-scan sampling (added).
- **Reviewer error and costs:** 0% and 3% reviewer error; two cost settings, with raw counts.
- **Integrity checks:** W1 and the post-hoc full-scan W4 must be reproduced exactly as references.

The carried-over targets and new rules D1–D4 are declared in the proposal. Questions for the user:
- approve the scope;
- keep or revise the targets, given the ceiling arithmetic;
- authorize CPU tests and the run after implementation.

**Plan agreed: (a) non-match pruning + safeguards, then (b) facing-aware re-adaptation.** After the gated-matching results, the user decided to do (a) first and (b) after it. (a) will start in a fresh session from the [handoff prompt](czechlynx-nonmatch-pruning-handoff-2026-10-18.md). Both items are listed in Active work as unclaimed. The minimum-review round (`benchmark_gated_matching`, its CLI/tests, results doc and these log updates) is still uncommitted in the working tree; a fresh session in this checkout sees it.

**Minimum-review gated matching run: accurate automation, small time saving; no configuration met all targets.** The user approved the scope and authorized the assistant to commit the review-effort round first (`0a25c2c8`) and then to implement and run this. Three new files and 11 tests were added (158 pass across related suites). The run took 135 s.
- **Confidence model:** identity-cross-fitted AUC .979 with space-time features, .924 with appearance only.
- **Automation:** in the sequential simulation, τ=.99 automates 49–51% of bundles with 0.30–0.41% misassigned photos.
- **Count:** the approved design (expert checks only the top-K before declaring a new animal) leaves +28 (K=20) to +70 (K=10) duplicate animals even without automation. The ≤50-pair duplicate sweep barely helps.
- **Post-hoc variant** (full-scan new-animal confirmation, run from a separate script with evidence saved next to the main output): count error 0 at τ .98–.995 with 46–61% automated and 0.22–1.57% misassigned. Expert time is only 1.48–1.59 vs 1.85 h per 1,000 photos, because automated bundles were one-click decisions and the cost sits in new-animal confirmations and hard matches.
- **Reviewer error:** 3% reviewer error gives 8–11% misassigned photos.
- **Cold start:** appearance-only τ=.95 collapsed everything into one animal, so a cold-start guard is needed.
- **Audit intervals:** covered the truth in all perfect-reviewer space-time runs and in 73% of all automated runs.

Declared targets were not all met; there is no candidate operating point. [Results](research/czechlynx-gated-matching-results-2026-10-18.md). Next options (not approved): calibrated non-match pruning plus safeguards; facing-aware re-adaptation; end-to-end integration.

**User decision recorded; minimum-review proposal drafted.** The user accepted expert-confirmed ranked matching as a route to reliable counts, provided the bulk is automated within an all-in-one ingestion → classification → re-ID/counting system with minimal expert review. This is recorded as a requirement update near the top of this log.

Read-only in-sample check: auto-accepting the top-1 match when its margin over the runner-up is ≥ .15 decides 53% of bundles with 0.42% wrong; margin ≥ .10 decides 65% with 1.37% wrong. Margin AUC is .934 for top-1 correctness; new-vs-known AUC from s1 is .87, so new animals stay expert-confirmed. 88% of calibration photos have coordinates for space-time features.

[Proposal](czechlynx-minimum-review-proposal-2026-10-18.md):
- W4 confidence-gated ranked matching: identity-cross-fitted confidence model with appearance and space-time features.
- Expert queue with a top-K new-animal check, a duplicate sweep, and an audit sample that estimates the error rate with a 95% interval.
- Simulated sequentially so that automatic errors propagate.
- Declared targets: ≥70% automatic; ≤1% misassigned photos; |count error| ≤ 1; ≤0.5 expert-h per 1,000 photos; the audit interval covers the true error.
- CPU only. Awaiting approval; the earlier review-effort round is still uncommitted.

**Review-effort simulation run: ranked catalogue matching wins; grouping-first costs more.** The user approved the scope and authorized the assistant to run it. Three new files and 15 tests were added. The first run completed all 88 simulations but failed while writing the report (numpy int64 in JSON); it is preserved. After a fix and a regression test, the `_r2` run completed in 131 s with identical numbers.
- **Starting groupings** reproduce the saved edge, flank and legacy runs exactly.
- **Main setting** (perfect reviewer, trap-day bundles):
  - W0 unassisted: 29,930 decisions.
  - **W1 embedding-ranked catalogue: 5,283 decisions (≈11.7 h; 1.8 h per 1,000 photos at placeholder costs), verified F1 1.0, 55/55, count error 0.**
  - W2 group-then-correct: 1.27× (edge), 1.91× (flank) and 2.92× (legacy) W1's decisions. From the edge and flank starts it is left with about 100 surplus fragments.
- **Ranking:** the correct animal ranks first for 82.9% of known-animal bundles (top-10: 95.9%); the 55 new animals cost 1,485 full-catalogue scans.
- **Bundling:** without trap-day bundles, W1 needs 3.1× more decisions.
- **Reviewer error:** 3% wrong answers degrade W1 to F1 .916, count +5, and W2 further.
- **Declared rules:** grouping saves work, strong saving and robust to reviewer error were all not met.

[Results](research/czechlynx-review-effort-results-2026-10-18.md). The user needs to decide whether expert-confirmed ranked matching satisfies the reliable-count requirement. Real reviewer timing and error rates are unmeasured.

**Review-effort measurement proposed (user choice after the flank results).**
- **User questions answered:** asked which next option the assistant would choose and whether local matching + geometric verification could still help. The assistant recommended measuring expert review effort first, keeping facing-aware re-adaptation as the fallback.
- **Local matching:** ruled out only in the tested form (generic DISK+LightGlue, homography/affine), per the same-flank re-check above. Pose-rectified flank spot matching remains the one untested variant.
- **Crop resolution:** crops are already native resolution (median animal crop width 521 px).
- **[Proposal](czechlynx-review-effort-proposal-2026-10-18.md):** a simulated expert compares W0 unassisted catalogue matching, W1 embedding-ranked matching and W2 group-then-correct review. W2 starts from the best edge-grouping, flank-grouping and legacy groupings, with trap-day bundles, a declared time-cost model, reviewer-error and unidentifiable-photo sensitivities, and effort–quality curves.
- **Declared rules:** W2 reaches F1 ≥ .98 and |count error| ≤ 1 with ≤ 50% (strong: ≤ 25%) of W1's decisions; F1 ≥ .95 with 3% reviewer error.
- Acceptable absolute effort and whether bounded review satisfies "reliable counts" remain the user's decisions. CPU only. Awaiting approval; no code or runs yet.

**Checkpoint committed; candidate-coverage/identifiability diagnostic proposed.** At the user's request, the CzechLynx work since June was committed as `606082f2` on `feat/reid-demo-pipeline` and pushed. The local `disk` submodule edit was not included. The assistant inspected the 16 saved adapted review examples, which are deterministic extremes and not representative. None is a clear left-side vs right-side pair. Most involve frontal/rear views, partial crops, washed-out or dark IR, tiny crops or fragmented masks. One clear same-side, same-camera `lynx_036` pair scores 0.04. 193 of the 321 audited images are calibration images with human flank/visibility/usable-markings labels. [Proposal](czechlynx-coverage-diagnostics-proposal-2026-10-18.md): three new files, no existing-source edits, calibration only, frozen adapted cache, no inference. It tests whether missing oracle links come from unidentifiable images (abstention curve vs random-drop control), split encounters, narrow candidate lists (k=100, mutual kNN, query expansion) or opposite flanks (audited cross-tabs), with decision rules declared before running. Awaiting approval; no code, tests or runs yet.

**Coverage diagnostic approved, implemented and run: missing links are largely recoverable; wrong links are now the bottleneck.** The user approved the scope as written and authorized the assistant to run the new tests and the read-only diagnostic. Three new files were added; 21 new tests pass (90 with the existing clustering suites). The run took 151.6 s, and its self-check reproduces the calibration graphs and the 10/55 oracle baseline exactly. Encounter links raise oracle connectivity 10 → 28/55 (0.81% of encounter pairs join different animals, from two encounters). kNN k=100 raises it to 23/55 at precision .41. A post-hoc check of encounters plus k=150 reaches 44/55, but 64% of those candidate edges are wrong. Opposite-flank same-animal pairs are almost never in each other's top-50 (2.6% vs 29% same-flank). The label-free quality score matches human audit labels (AUC .73), but abstaining on risky images does not connect animals. Query expansion and mutual kNN do not help. Caveat: encounter IDs are dataset-provided and unverified for deployment (day-only timestamps; 113 cross-camera encounters). [Results](research/czechlynx-coverage-diagnostics-results-2026-10-18.md). Next: propose grouping on kNN k=100–150 ∪ encounter links with calibrated global + local edge verification, plus a test of camera/time-derived encounters. No count configuration, selection or outreach.

**Coverage diagnostic committed (`9065277d`, not pushed); edge-verification grouping proposed.** The user asked for the next proposal: grouping on kNN k=100–150 plus encounter links, with calibrated global + local edge verification and a test of derived encounters. Read-only checks before drafting found:
- The saved local evidence is weak: DISK+LightGlue homography inliers AUC 0.70, RoMa 0.45, on hard shortlist pairs.
- CzechLynx has no EXIF times and only day-level dates. Annotated encounters (18,782) are finer than trap-day groups (10,540).
- Medvednica timestamps lack camera IDs and identities.
- The raw metadata has left/right pose keypoints for 5,206 images (not used yet).

[Proposal](czechlynx-edge-grouping-proposal-2026-10-18.md):
- Stage A (CPU): label-free edge features (cosine, ranks, k-reciprocal Jaccard, camera/trap-day/location-day/encounter flags, quality risk) and an identity-cross-fitted logistic edge model. Three grouping families: threshold, average-linkage that resists single bridges, and Leiden. Two encounter arms: annotated vs day-level derived.
- Stage B (GPU, gated): DISK+LightGlue pilot on 20k stratified edges with raw match, homography and partial-affine features. It runs on the full graph only if it adds ≥0.02 AUC overall or ≥0.05 in the ambiguous band.

Outcome rules are declared in the proposal. Awaiting approval; no code or runs yet.

**Edge grouping approved and run: better grouping, still not reliable; local features fail the gate.** The user approved the scope and authorized the assistant to run the tests, Stage A and the Stage B GPU pilot, asking to be consulted before any full Stage B run. Three tools/modules and 29 tests were added (144 pass across related suites).

- **Stage A (253 s, CPU):** identity-cross-fitted edge-model AUC 0.87–0.89 (cosine alone 0.48–0.64). Best grouping (derived arm, k=100, G2 average-linkage τ=.5) reaches pairwise F1 .585 vs .264 for the best saved control, but only 1/55 exact recoveries (38 merges, 49 splits).
- **Day-level links:** same trap + day joins different animals in 0.92% of pairs and reaches 46/55 oracle connectivity at k=150 (annotated encounters: 44/55).
- **Stage B pilot (20k edges, 607 s GPU):** local evidence adds ≈0.000 AUC. Its weak pooled signal (≈.60) is confounded by animal: local-only AUC .60 with random CV vs .35 identity-cross-fitted.
- **Declared rules met:** only day-level deployability. Edge model, grouping progress, count candidate and local gate were not met.
- **Run history:** the first pilot run failed on a mask-cache bug (preserved; fixed with a regression test; rerun as `_r2`). One Stage A run with only a "running" manifest was stopped after a code edit and removed.
- **Deviations:** Louvain instead of Leiden (leidenalg not installed; installing it would alter the frozen legacy control); G2 support min(s,|A||B|); derived links use trap-day or location-day; affine threshold .03.

[Results](research/czechlynx-edge-grouping-results-2026-10-18.md). Full Stage B is not recommended. Pairwise evidence quality is now the binding constraint. Candidate next directions (not approved): flank-aware comparison using the 5,206 pose-keypoint images, a learned pair verifier trained on fitting data, and a measured review-effort baseline. No count configuration, selection or outreach.

**Edge grouping committed (`3f5a0f11`, not pushed); flank-aware grouping proposed.** At the user's request, read-only checks used saved vectors, raw metadata and audit labels, with no inference.
- **Pose geometry:** pose keypoints exist for 5,206 images. Facing from head vs tail-base geometry agrees with all 56 audited side views, but with the names swapped: the audit's "right" means "faces right". The user needs to confirm the convention.
- **Facing from embeddings:** a logistic classifier on the adapted MiewID vectors predicts facing at AUC .995 vs pose and .989 vs audit (n=145, including 103 images without pose), 94.5% accuracy, identity-cross-fitted. An unscaled first attempt looked unconfident because of a regularization artifact. Public MegaDescriptor is nearly flip-invariant (.60).
- **Structure of the problem:** about half of same-animal pairs are opposite-facing (37,443 vs 36,283). Opposite-facing pairs at the same trap on the same day are 98.95% the same animal, and 50/55 animals have one.

[Proposal](czechlynx-flank-grouping-proposal-2026-10-18.md): group within each facing (per-facing graphs and cross-fitted edge models), join flank clusters by one-to-one matching on trap-day co-occurrence, and report bilateral count bounds [pairs + max(unmatched), pairs + sum(unmatched)]. Rules are declared. CPU only. Awaiting approval; no code or runs yet.

**Flank-aware grouping approved, implemented and run: facing works, grouping does not improve materially.** The user approved the scope, authorized the assistant to run it, and asked that every step be documented for multi-agent work. Since then, AGENTS.md carries a coordination rule and this log has a live "Active work" section.
- **Implementation:** three new files and 13 tests; 132 pass across related suites. The run took 215 s on CPU.
- **Facing:** identity-cross-fitted predicted facing reaches pose AUC .995 and audit AUC .989 (n=145; 103 without pose); 74% of images are confident.
- **Grouping:** per-side edge AUC is .85–.90, no better than without the split. Best animal-level F1 is .602 (edge grouping .585), with 0/55 exact recoveries. Flank-level precision is .6–.7, with at most 8/110 flank units recovered.
- **Counts:** 33/144 count intervals contain 55, but all are ≥19 wide and rest on about 30 merge errors.
- **Declared rules:** only "facing classifier usable" is met.
- **Interpretation:** the frozen embedding does not separate same-side photos of different lynx well enough. Further graph or grouping engineering on these vectors is unlikely to reach reliable counts.

[Results](research/czechlynx-flank-grouping-results-2026-10-18.md). A follow-up read-only check (user question: could local matching + GV still help?) found local evidence equally weak on same-facing and opposite-facing pilot pairs (affine-inlier AUC .61 vs .59; 5–6 inliers). Its weak signal is therefore image-condition confounding, not spot matching. This negative is limited to generic DISK+LightGlue with homography/affine verification; see the [addendum](research/czechlynx-edge-grouping-results-2026-10-18.md#addendum-18-oct-after-the-flank-run-local-evidence-on-same-flank-pairs). Candidate next steps (not approved): facing-aware re-adaptation, a review-effort baseline, a high-precision core count. The audit left/right convention is still unconfirmed by the user.

### 2026-10-17

**Adapted full calibration reviewed: recognition improves, reliable counts remain unachieved.** User-run comparison completed in 172.195s on all 6,352 calibration photos / 55 identities, all 25 controls and 20,170,776 pairs. Top-1 rises 877/1,336 → 917/1,336 (65.6437% → 68.6377%), macro top-1 56.0883% → 60.3884%, mAP@50 31.0078% → 40.1974%; 118 corrected / 78 worsened known queries. Every grouping control still has zero exact identity-group recovery. Legacy gives 26 groups with 26.603% pair precision; threshold .35 gives 1,545 groups including a 4,119-photo/50-identity component; threshold .55 gives 4,236 groups with 99.802% precision but 2.471% recall. Same numerical cutoffs have changed operating points after adaptation. k50 oracle connectivity remains 10/55 identities, fragments improve 196 → 187; truth-filtered graphs are not counts. Verified 647 artifacts, both immutable vector caches, 50 saved assignment/count/pair profiles and saved recognition/histogram arithmetic; all 22 protected hashes remain unchanged. Two histogram figures and three preselected pairs/six thumbnails inspected; no causal/night/flank/annotation claims. [Full review/receipt and next direction](research/czechlynx-miewid-fp32-calibration-results-2026-10-17.md). Preserve adapted recognition baseline and all original evidence; no count config/selection/outreach. Next is a concrete local/global candidate/pair-calibration/robust-grouping scope, with code approval still required. No assistant-run tests/models/training/fresh candidate/pair benchmark or source/data changes.

**Completed FP32 adaptation reviewed; calibration comparison next:** The user reported epoch 9 completion and selection of `best.pt`. Saved v2 status is `complete`, elapsed 2,397.940s (39.97 min), epochs 0–9; epoch 6 correctly selected and epochs 7–9 trigger patience three. Macro recall@50/top-1 improve 92.6351%/65.4234% → 95.3157%/71.3711%. Existing completed-trial validation, current code/environment/protocol/audit bindings, all fitting image/mask bytes and six artifact hashes pass. CPU tensor-safe inspection checks both best/last provenance, finite FP32 weights/head, exact initial architecture keys/shapes/dtypes, unchanged BN buffers/unused final layer and native update proof. All 22 protected hashes and frozen baseline/cache contracts match. [Review/receipt](research/czechlynx-miewid-fp32-training-results-2026-10-17.md). Inner k50 oracle connectivity improves 12/34 → 19/34 fully connected identities, fragments 70 → 58; this truth-filtered diagnostic is not a deployed count. The completed/improved trial is eligible for the user-run full calibration comparison on 6,352 images / 55 identities and the same 25 controls, using confirmed-absent adapted cache/report paths. No source/data/trial changes, model instantiation/inference, tests/training/benchmarks or outreach by the assistant. Local/global integration, independent/night/flank validation, reliable counts and park readiness remain open; earlier running/pending statuses are historical.

**User reaffirms local/global direction, readiness question and working diary:** The user expects the thesis local and global methods to be combined in the final system and suspects the earlier combination was flawed. Recorded this as the intended direction and a diagnostic hypothesis; no specific fault or fusion benefit is established. Added the current snapshot and next-decision sequence above so the diary shows both completed work and remaining gates. Read-only saved-artifact inspection finds v2 still `running`, epochs 0–6 saved; macro top-1 rises from 0.6542339389 at epoch zero to 0.7137113657 at epoch 6 (+5.9477 percentage points), and macro recall@50 from 0.9263512632 to 0.9531567744 (+2.6806 points). These are provisional inner-holdout recognition results, not final checkpoint qualification or count gains. [Progress snapshot](research/czechlynx-fp32-progress-2026-10-17.json). No change to the running trial, source, inputs, masks, splits or audit; no assistant-run tests/models/training/benchmarks/outreach. Readiness remains evidence-based: complete training review, grouping/count comparison, controlled local/global integration and independent/field qualification with agreed count/review criteria. No unsupported calendar commitment; keep reliable photographed-individual counts before park approach as the user's target.

**FP32 v2 preparation reviewed; user training handoff:** The user completed model-free prepare-only with the expected 14,044 training images / 131 identities / 123 atomic groups and 2,460 validation images / 34 identities / 33 groups. Metadata-only inspection confirms exactly two prepared files, the exact v1/reconstructed inner plan, fixed FP32 settings, current source/environment/protocol/review bindings and fitting digest. Zero cross-role identity/encounter/source/prepared-photo hash overlap; all original fitting images remain; no outer calibration/selection/test inclusion. All 2,460 holdout images have legal positive comparisons. All 22 protected hashes match, including failed v1/audit evidence. [Review and receipt](research/czechlynx-miewid-fp32-preparation-results-2026-10-16.md). Preparation/review began on 16 October; final preservation check and handoff occurred on 17 October. The 37-test post-correction suite is already user-reported passed. Next: user runs the same prepared v2 command without `--prepare-only`, then shares the completed/failed manifest and epoch history before calibration. No source changes or assistant-run tests/models/preparation/training/benchmarks/outreach. Real FP32 training and reliable count gains remain unvalidated; earlier preparation-pending entries are historical.

### 2026-10-16

**FP32 correction focused suite passed; fresh preparation next:** The user supplied the two-suite pytest result: **37 passed, 121 warnings in 11.61s** (65 audit / 56 adaptation warnings, all upstream pycocotools NumPy deprecations). This supersedes the correction-test-pending entry below and covers the synthetic precision/provenance, numerical-failure/no-update and composed-report regressions. It does not demonstrate real FP32 training, improved embedding accuracy or reliable counts. Read-only checks confirm `models/czechlynx/miewid_adaptation_v2_fp32` does not exist; failed-v1 manifest SHA256 remains `46d2fc8d3faebc95ad87ab09ca6b18b8b600d7b4ac3784c86cf952b9b552b58b`, audit receipt remains `ffa4000e8cccc219dc3e53da82df2714263b12f3e878d448416f2031c63c93f3`. Next: user-run model-free prepare-only in fresh v2, then review its exact split/current precision and source bindings before handing over the training command. Documentation only; no source changes or assistant-run tests/models/preparation/training/benchmarks/outreach. [Preparation command](czechlynx-benchmark.md#3-prepare-and-inspect-the-training-plan).

**Approved FP32 correction implemented; user validation pending:** In the requested fresh session, the user reaffirmed the three-file correction and instructed implementation without repeating approval. Added the fixed `training_precision: float32` recipe/provenance contract; FP32 model/head/inputs with training autocast and loss scaling disabled; separate effective evaluation precision metadata; and structured failure context plus failure-only gradient diagnostics. Invalid entries and overflow while reducing finite entries are distinct. Strict checks stop before the failed batch's optimizer step. Diagnostic/save errors preserve the original exception. Extended the focused regressions for precision enforcement under inherited autocast, stale preparation/checkpoint rejection, saved runtime metadata, invalid gradients and individual/aggregate norm overflow. The previous 26-test receipt predates this correction. No tests, toy/model runs, preparation, training, benchmarks or outreach were run by the assistant. The completed audit/receipt, data/splits, evaluation extraction, baseline source and failed v1 are preserved. Next: user runs the focused tests, then fresh v2 preparation and exact split/binding review before receiving the training command. No calibration or count improvement is established. [Current handoff](czechlynx-fp32-session-handoff-2026-10-16.md).

**FP32 correction approved, to be implemented in a fresh session:** The user explicitly approved the proposed three-file correction and asked to defer execution while documenting the work and drafting a continuation prompt. Recorded the approval in repository instructions, failure review, experiment proposal and benchmark status; created [the complete handoff and copyable prompt](czechlynx-fp32-session-handoff-2026-10-16.md). This supersedes pending-approval notes below. The fresh session should announce and implement the approved FP32 training contract, numerical-failure diagnostics and focused regressions without asking again, then provide user-run validation commands. Source changes beyond that scope still need approval. The current session changed documentation only; implementation still matches failed-v1 hashes, the audit receipt is unchanged and v2 does not exist. No tests, models, preparation retry, training, benchmarks or outreach. Preserve the existing dirty/untracked workspace and failed evidence.

**First real adaptation failed; numerical correction proposed:** The user started the prepared v1 trial. Frozen holdout macro recall@50/top-1 is 92.6351%/65.4234%; head-only epoch 1 finished in 89.800s with exactly unchanged full metrics. A native embedding update was verified at epoch-2 batch 1, but a later non-finite total gradient norm raised before that batch's optimizer update. Saved status is failed after 173.218s; no completed epoch-2 evaluation/checkpoint. FP16 instability is plausible, not demonstrated; existing logs cannot distinguish invalid gradient entries from norm-reduction overflow. Do not use the error message's suggested disabled check as a fix, restart v1 or commission calibration. [Failure evidence and three-file correction proposal](research/czechlynx-miewid-training-failure-2026-10-16.md): recorded FP32 training recipe, precise failure diagnostics and focused regressions, followed by user-run tests and a fresh prepared v2 trial. Source-change approval remains pending under AGENTS.md. Failed evidence/audit/source/data preserved; assistant performed read-only diagnosis/documentation, no source patch, tests, models or retry. This supersedes the ready-to-start-v1 handoff; actual adaptation/count gains remain unestablished.

**User-run MiewID training preparation complete and reviewed:** Saved trial `models/czechlynx/miewid_adaptation_v1` is prepared with 14,044 training images / 131 identities / 123 linked groups and 2,460 validation images / 34 identities / 33 groups. Exact reconstruction, run/source/environment/review bindings and fitting metadata digest match. Zero cross-role identity, encounter, source-photo or prepared-photo hash overlap; no outer calibration/selection or official-test images. Every validation image has a legal metadata-defined cross-encounter positive. [Preparation review and hashes](research/czechlynx-miewid-training-preparation-results-2026-10-16.md). This supersedes preparation-pending status. Next is user-run actual training with the same command minus `--prepare-only`: at most 10 x 1,756 balanced batches, head-only epoch 1, embedding adaptation from epoch 2, early stopping and no resume. Review the completed inner-holdout evidence before calibration; frozen epoch zero can remain best. No source/data changes, assistant-run tests, models, training or performance benchmarks. Input-quality and unresolved upstream exposure still limit interpretation; reliable counts remain unachieved.

**User completed the human input audit; review receipt created:** Confirmed 321/321 completed app records and exact CSV hash, payload and revision agreement with the user's final export. Ran the existing model-free summarizer successfully, producing `evaluations/czechlynx/input_audit_review_v1/receipt.json` and `reviews.json`. This closes the audit review/receipt prerequisite; earlier pending entries are historical. The export remains outside the repository, bound by absolute path and SHA256 `4e5285142551e66a7f1b93301032bcdf2e1b0e54da05515eac9636812af2244e`; preserve its bytes. [Results and evidence](research/czechlynx-input-audit-review-results-2026-10-16.md).

Uniform fitting/calibration samples contain 55/128 and 27/128 unusable-marking judgments, 24/128 and 20/128 crop/mask issues, and 12/128 and 5/128 no-clear-animal judgments. Findings support an input-quality concern but do not quantify its causal contribution to grouping errors. Do not combine purposive cohorts into a dataset-wide error rate. Several verified IR/night entries rely on appearance rather than independently checked capture evidence; preserve reviewer assertions and do not claim verified night/IR robustness. Labels remain sidecar data and do not enter the approved adaptation loss or inference.

Next is the existing user-run model-free `--prepare-only` step and inner-plan review. The approved first trial retains original fitting membership and the complete eligible calibration collection. Any correction/exclusion/cleaned-data comparison needs a separate explicit scope and lineage. No source/app code, images, masks, identities or membership were changed. No tests, models, training, performance benchmarks, clustering selection or outreach were run. Reliable counts and park readiness remain open.

### 2026-10-15

**User-authorized AI draft review pass completed:** The user asked for the purpose of the audit and authorized AI drafts followed by human review. Clarified that visual quality review informs input/representation experiments; the review receipt is an implemented workflow prerequisite, not a mathematical necessity for fine-tuning. Flank/verified illumination are diagnostic additions, with unknown valid. Inspected 321 original/crop/mask preview sets across 54 contact sheets and checked 12 selected cases again using larger source/prepared-image panels. Saved the displayed sheets, label TSV and input hashes in the sibling app's evidence archive. No identity labels entered the visual decisions. Imported **318 AI drafts**, preserving two completed user reviews and one existing draft. Exact revision checks prevented overwriting touched records; two own unchanged drafts were refined after the larger views. Each AI draft retains blank reviewer/reviewed_at, `completed=false` and an `AI suggestion:` rationale. Final observed state: **2 complete / 319 drafts**, including all 318 unmodified AI suggestions. Human confirmation and the summarize receipt remain pending.

Potential concerns include retained snow-covered rock/tree regions, apparent prey/ungulate subjects, an omitted visible head/neck, severe blur/low resolution and saturated/indistinct close frames. These are tentative assistant visual judgments for the user to verify. They establish neither definitive source-label errors nor a population-wide error rate; the sample includes purposive diagnostic cohorts. No removals, identity changes or training selection followed these flags. Image-only illumination suggestions use colour/monochrome/unknown and provide no verified night/IR labels. The existing app and benchmark source were unchanged; temporary analysis/import scripts operated through the approved app API. No tests, local neural feature extraction, training, calibration, benchmark, clustering selection or outreach execution. The attached server stopped during drafting; restarted in the background on the same LAN URL, with saved SQLite state intact.

Catalogue: `/home/mm-workstation/re-id/czechlynx-review-app/exports/ai_draft_catalogue_20261014T224307731683Z.json` (SHA256 `4631988a7c3ef69a6e4c1ee65ac1bd7486933b298a45a87949c334ef94afd15d`). Human-facing handoff: sibling `exports/ai_draft_handoff_20261014T224307731683Z.md`; retained visual evidence: sibling `data/ai_draft_evidence_20261014T224307731683Z`. UTC filenames begin 14 October; the work completed on 15 October in Europe/Zagreb. Next: refresh the app, inspect/correct each outstanding draft and save complete; export a dated completed CSV before summarizing. Reliable clustering/count outcomes remain pending.

### 2026-10-14

**Standalone browser review app approved, created and started:** User requested a simple LAN-accessible audit review app outside the main project and confirmed the concrete seven-file scope. Created `/home/mm-workstation/re-id/czechlynx-review-app` with a standard-library Python server, SQLite review/history storage, HTML/CSS/JS interface, launch script, README and 13 focused tests. Original/crop/mask overlay views use full-resolution pixels, with zoom; controls retain the existing schema, unknowns and illumination-evidence rule. Draft autosaves are separate from explicit completion. Revision checks prevent stale browser edits from silently replacing another review. Exact-schema CSV exports are dated snapshots in the sibling app, with JSON bindings; the existing summarize command still creates the training review receipt. Port 8765 is occupied, so the server uses `http://192.168.8.110:8766`. Startup audit integrity checks, Python/JS/shell syntax and Ruff F checks pass. Read-only LAN requests returned HTTP 200 for page/static/session/first-image assets; 321 records, 0 completed/0 drafts at verification. Browser workflow and focused tests remain user-validation steps; no unittest/pytest, model, training, benchmark or outreach execution by the assistant. Audit, benchmark source and the manual working CSV were not edited. Review annotations do not alter identities, splits or inclusion. Human audit completion, real adaptation and reliable clustering/count evidence remain pending. [Browser review instructions](czechlynx-benchmark.md#browser-review-on-the-local-network).

**Real input-audit preparation completed and reviewed:** User reports 321 crops in `evaluations/czechlynx/input_audit_v1`. Read-only checks validate complete status, deterministic sample/source bindings, 966 artifacts, original-photo and sampled prepared/mask bytes. All 321 recorded annotation/preparation alignment checks are exact. Cohorts retain 128 fitting uniform, 128 calibration uniform, all 55 calibration identities and 11 diagnostic crops; one calibration image belongs to two cohorts. No selection or non-official-time_open-train sample. Inspected two uniform crop panels and the known suspect source-mask overlay, with no completed human labels or verified night/flank judgments. These checks establish preparation fidelity; identifiability/source annotation correctness need manual review. Created a no-overwrite working review CSV with labels/reviewer fields unfilled. Next is user review of all 321 rows, then the summarize receipt before training preparation. [Review and instructions](research/czechlynx-input-audit-preparation-results-2026-10-14.md). No new source, tests, audit/model/training/benchmark runs or outreach by the assistant; saved-artifact review and documentation only, plus working CSV copy. Training/calibration, clustering selection and count outcomes remain open.

**Audit/adaptation focused rerun passed:** The latest user-provided output supersedes the preceding failed transcript: **26 passed, 93 warnings in 9.35s**. All five failures in the first 21-pass/5-fail run shared the corrected pandas index/column ambiguity. The passing rerun exercises the existing indexed-input regression, model-free preparation, protected outputs, native-shaped CPU toy training/checkpoint contracts and composed count/pair-report interfaces on a small fixture. Warnings are upstream pycocotools NumPy deprecations; no source change or extra run is needed for them. This is implementation regression evidence, not real MiewID adaptation or count improvement. Next: user-run bounded input audit, review its original/crop/mask views and completed review receipt before training preparation. Real audit/training/calibration and clustering selection remain pending. No tests/models/benchmarks/outreach executed by the assistant. [Audit command](czechlynx-benchmark.md#2-prepare-the-input-audit).

**First audit/adaptation test run failed; shared cause corrected:** User reports the two new suites returned 21 passed, five failed and 93 upstream pycocotools NumPy deprecation warnings in 6.46s. All five failures reach the same `build_inner_plan` statement: selecting with `set_index("image_id", drop=False)` retains both the index name and column, making `sort_values("image_id")` ambiguous. Reset the temporary selection index before sorting and extended the existing split/group/row-invariance test for an already-indexed input. No role assignments or model settings are intentionally changed. The failed paths include prepare-only/source protection and toy training/composed calibration, so passing downstream execution is not yet established. The warnings are separate from the reported exception. Tests remain user-run; assistant performed code/static review and documentation only. Rerun the same two suites before the real audit. Original frozen source/evidence is preserved.

**Approved seven-file audit/adaptation scope implemented:** The user explicitly approved the [concrete scope](czechlynx-input-audit-adaptation-proposal-2026-10-14.md) and reaffirmed continuation after refreshing AGENTS instructions. Added only its seven source/test files: model-free original/crop/mask audit with deterministic, separate cohorts and validated human-review receipts; fitting-only approximately 80/20 linked-group inner holdout; native 2,152-D MiewID adaptation with a separate ArcFace head, fixed BN running statistics, head warmup, verified embedding gradients/updates and strict best-checkpoint provenance; composed full calibration recognition, 25 grouping/count profiles and all-pair/connectivity comparison against immutable baseline vectors. Static review hardened re-signed sample/membership rejection, source/review/output protection, warmup state preservation, actual initialization-weight equality, label-free extraction, partial-cache reuse and incomplete-stage failure reporting. Tests cover these behaviors with temporary input artifacts and CPU toy models. AST syntax and Ruff F checks pass; no pytest, audit command, training, inference or benchmark was run by the assistant. Original clustering-bound source hashes remain unchanged. [Exact user commands](czechlynx-benchmark.md#input-audit-and-native-miewid-adaptation). Empirical validation, quality rejection, candidate/pair/robust merging work, clustering selection and reliable counts remain open; no config exported or outreach performed. Prior approval-pending notes/feasibility receipt describe the earlier proposal state.

**User agrees with the improvement recommendations; first trial presented:** Prepared the [input-audit/MiewID-adaptation scope](czechlynx-input-audit-adaptation-proposal-2026-10-14.md), listing seven additive source/test files. Audit first; then one declared supervised recipe on the existing fitting pool with a separate linked-group inner holdout, followed by full calibration recognition/pair/connectivity/group/count comparison. Broader candidates and calibrated global/local robust merging follow the result. Exact code-scope approval remains pending under AGENTS.md; no source implementation, tests, training, model/benchmark runs or outreach. Read-only fitting inventory finds 165 identities in 156 atomic groups, including five linked groups (largest six), and one single-encounter identity. Direct source-annotation decoding and pixel comparison confirms the suspect lynx_280 prepared RGB/mask exactly match the supplied annotation/source crop; the visual identifiability concern remains, with no preparation mismatch in that case. [Feasibility receipt](research/czechlynx-adaptation-feasibility-2026-10-14.json). Original frozen evidence/code remains preserved. Clustering selection, quality rejection, appropriately independent validation and reliable count outcomes remain open.

**User-run pair/connectivity diagnostic reviewed:** User reports 41 focused tests passed in 5.19s. The real diagnostic completed all 20,170,776 unordered pairs and 6,352 vectors in 14.913s, with zero unresolved observations; 50 artifact hashes, current bindings/cache bytes and saved-table arithmetic verified. Truth-filtered unthresholded k15 connects 2/55 identities in 382 fragments; k50 connects 10/55 in 196 fragments, with 100 isolated images. At k50, largest truth-selected components cover 95.88% of images but 81.22% on an identity-average basis; these oracle figures are not model accuracy or deployed counts. Raw cutoff .55 gives 98.17% pair precision / 1.16% recall; candidate k50/.55 gives 98.16% direct-edge precision while threshold components give 16.03%. Inspected six preselected pairs (11 unique prepared crops), one raw frame and one plot. The shared extreme low-score endpoint `clx_e981b521482a803cb7855493` retains 99.6864% of its crop as foreground; source image/annotation/identifiability needs verification, with no confirmed decoder bug or metadata change. Similar poses/collars in wrong pairs remain hypotheses, not attribution findings. Recommend auditing inputs, improving representation across views, candidate diversity and calibrated pair/robust graph evidence; further code requires presentation/approval. Selection/config export and reliable count outcomes remain open. The assistant performed read-only result/visual analysis and documentation only. [Review/receipt](research/czechlynx-pair-connectivity-results-2026-10-14.md).

**Approved pair/connectivity diagnostic implemented:** The user explicitly approved the three additions presented in the [diagnostic scope](czechlynx-clustering-diagnostics-proposal-2026-10-14.md). Added `reid_demo/benchmark_cluster_diagnostics.py`, `tools/diagnose_czechlynx_clusters.py` and `tests/test_benchmark_cluster_diagnostics.py`. The CLI accepts completed calibration only, reads immutable frozen vectors and completed appearance evidence, reconstructs the exact saved candidate budgets, and checks all candidate counts and retained score/ID prefixes. Streaming all-pair histograms/threshold metrics, macro-identity positive recall, missing-metadata subgroups, per-identity positive-score distributions, direct-edge/component comparisons and truth-filtered fixed-graph connectivity distinguish candidate, representation and propagation limitations. Ground truth enters after candidate inference; oracle fragments are not achieved identities/counts. Zero vectors remain unresolved, unscorable positive pairs remain in recall denominators, undefined metrics are null and incomplete oracle totals are null. Reports bind source/artifact/code/cache bytes and provide bounded deterministic examples. Static review corrected protocol protection to cover the source file while allowing independent sibling output directories. All original calibration-bound source hashes remain unchanged. Only static syntax/whitespace review was performed; no tests, diagnostic benchmarks, models or outreach were executed by the assistant. User-run validation and the count outcome remain pending. [Commands](czechlynx-benchmark.md#calibration-pair-and-connectivity-diagnostic).

**User-run clustering calibration reviewed:** All 25 profiles completed on 6,352 images / 55 identities in 39.497s. Verified 595 saved artifact hashes, current implementation/source contracts and every profile's assignment membership/count-pair arithmetic. Legacy Louvain finds 18 groups (14.70% pair precision, 23.54% recall). Threshold .35 finds 86 groups but one contains 6,250 images and all 55 animals (6.60% precision, 97.17% recall). Threshold .65 gives 4,623 groups (98.87%/1.32%). Consistent k15/.35 gives 1,607 groups (90.26%/2.28%). All profiles have zero unresolved observations and zero exact single-pure-group identity recovery. Threshold k15/k50 assignment hashes are identical at all six thresholds. Eight thumbnails were inspected; visible pose/illumination/visibility differences remain hypotheses, with flank and actual night/IR labels unknown. Scikit-learn's many-classes warnings accompany fragmentation; the saved run is complete. The earlier runtime estimate was too high. No configuration was exported; selection remains unrun. [Review and receipt](research/czechlynx-clustering-calibration-results-2026-10-14.md). [Next diagnostic proposal](czechlynx-clustering-diagnostics-proposal-2026-10-14.md) presents three additive source/test files, requiring approval before implementation. The assistant performed read-only result analysis and documentation updates, with no tests, clustering/model runs or outreach.

**User-run focused tests:** The user ran `venv/bin/python -m pytest tests/test_benchmark_clustering.py tests/test_benchmark.py tests/test_benchmark_protocol.py -q` and reported **62 passed, 1 warning in 8.50s**. The warning comes from upstream LightGlue's deprecated `torch.cuda.amp.custom_fwd` decorator during `test_partial_local_extraction_is_not_trusted_as_complete`. No source changes were made in response. The regression receipt does not establish real-data clustering/count accuracy. Next is the predefined 25-profile calibration run on the full eligible calibration collection using cached vectors, followed by review and explicit configuration export before frozen selection. The assistant did not execute tests or benchmarks.

**Subsequent user priority correction:** The user wants to focus on the clustering/count milestone before contacting parks, to have stronger demonstrated evidence. The outreach-first recommendation is superseded. The [concrete first implementation scope](czechlynx-clustering-count-proposal-2026-10-14.md) was subsequently approved and implemented: full eligible calibration/selection collection evaluation, existing frozen-vector reuse, global-only clustering controls, count/merge/fragmentation diagnostics and frozen selection configuration. The focused tests are user-reported passed and calibration results are reviewed; reliable count outcomes and selection validation remain open.

Approved implementation adds the three proposed source/test files and documented handoff commands. Inference receives no truth or target count; calibration profiles 25 configurations and selection requires one explicit checksummed export. Zero-vector images remain unresolved, component totals include singleton proposals, count errors are withheld for incomplete assignments and undefined pair metrics are null. Reports expose exact-group recovery, cross-encounter/camera evidence and deterministic error examples. Only static syntax/whitespace review was performed; no tests, neural runs, clustering/count benchmarks or outreach occurred. Reliable count outcomes, night/IR and opposite-flank treatment remain unvalidated.

Read-only metadata/cache feasibility checks: the full collection sizes are 6,352 eligible calibration images and 4,725 selection images. The matching masked MiewID cache has all required IDs (11,953 IDs total); individual vector bytes were not revalidated. All 39,760 prepared flank labels are unknown; the raw CSV has no validated viewpoint field. The collection has 22 mixed-identity encounters, so encounter IDs cannot imply same-animal constraints. No code, tests, model runs or benchmarks were performed for this priority/proposal update.

The user requested a deep project/practical-readiness assessment and identified **Risnjak, initially counting lynx because of distinctive markings**, as the first collaboration goal. The required clustering/count outcome remains unchanged. No Risnjak archive, current camera inventory or partner agreement was supplied.

Completed a [dated assessment](research/project-risnjak-assessment-2026-10-14.md) and [evidence receipt](research/project-risnjak-assessment-2026-10-14.json), using static code review, saved results and primary-source research. Phase 2 selection remains 63.79% top-1, 84.86% accepted precision at 36.89% correct-known coverage, with upstream exposure unresolved and clustering disabled. Historical full-funnel grouping remains eight mixed clusters for 60 individuals. Neither establishes reliable individual counts.

Extended the completed RoMa review: 2,000 saved rows contain the same 1,000 pairs per matcher (247 same-identity/753 different-identity), and all nine recorded source hashes match current files. Inlier-count AUC is 0.6958 for DISK/LightGlue and 0.4463 for RoMa; coverage AUC is 0.6340/0.6086. Only 52/68 known queries have a correct top-ten candidate. Inspected both matchers on the first preselected true and false rank-1 pairs (four images). This bounded review does not establish downstream gains or complete the full visual/per-identity audit. Focused-test confirmation remains open.

Static review adds potential pilot hazards: demo completion tests any non-null row; fused runs force reclustering of reviewed records; fresh cluster labels can share the namespace of preserved reviewed assignments; embedding skips use reference presence; MiewID is available through the benchmark adapter rather than the ordinary demo model path. The additional findings were not reproduced through new tests/fixtures. Historical filename collisions and ingestion resets remain present in the inspected code.

Proposed work centres on safe persistent identity state, encounter/view-aware evaluation, calibrated pair evidence, conservative clustering with unresolved cases, verified day/IR groups and count uncertainty. The assessment also distinguishes finite-inner-sample NIS bias and simulated oracle judgments from qualified field counting. Suggested gates, timing and collaboration descriptions are proposals; acceptable count error and review effort remain unagreed. Current Croatian AI-identification research and AnimalCLEF 2026 graph methods provide relevant collaboration/method references, with no claim of independent Risnjak performance.

Only assessment/decision documentation was changed. No source-code changes, tests, training, inference, benchmark runs or outreach were performed. Further code still requires prior presentation/approval; user-run validation remains the working agreement.

### 2026-09-22

User reaffirmed clustering and reliable individual-number assessment, questioned whether roughly 63% recognition reflects training leakage, highlighted missing local-feature gains and unresolved nighttime photographs, and requested persistent logging. Recorded these requirements and linked this log from repository instructions and active planning/readiness documents. No source code, tests, benchmarks or outreach performed for this update.
