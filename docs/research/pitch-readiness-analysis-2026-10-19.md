# Pitch readiness: lynx re-identification and counting for Croatian national parks

Date: 19 October 2026. Work item (d) in the [decision log](../lynx-demo-decision-log.md). Read-only analysis: no code, data, caches, tests, models or evaluation outputs were changed or run. Web sources were accessed on 19 October 2026.

**Labels used throughout.**
- Split: *development* (historical 750/560 dev sample), *fitting*, *calibration*, *selection* (now spent), *test* (official CzechLynx time_open test, locked), *external* (published by others).
- Claim status: **[M]** measured (a saved receipt or a published number), **[I]** inferred (arithmetic or reasoning from measured facts), **[H]** hypothesis (untested).

---

## Executive summary

**Verdict.** An automated lynx count is not pitchable to a national park now, and on current evidence it is not reachable within the next few months. What is honestly pitchable now is a **research collaboration for an expert-verified identification workflow**. The system ranks and pre-sorts photographs, the expert confirms them, and every decision is logged. It needs about 59% fewer expert decisions than full manual matching in simulation. However, that saving is not yet measured with real people or on Croatian photographs.

**What has been achieved** (CzechLynx, simulated reviewer):
- **Frozen one-shot test on 55 lynx never used for fitting or tuning** (*selection*, [M]): 51.5% of photo bundles decided automatically, 0.80% of photos misassigned, and a 59% saving in decisions and placeholder hours vs full manual matching (W1).
- **The count was +6 (61 groups for 55 animals).** Under 3% reviewer error, 3.6% of photos were misassigned.
- **Three of the four pre-registered criteria failed** (S1, S2, S4); only the self-audit criterion (S3) passed. Source: `evaluations/czechlynx/frozen_review_selection_v1/summary.json` → `criteria`.
- **The limiting factor is the embedding.** It does not separate same-side photographs of different lynx well enough: cross-encounter nearest-neighbour top-1 is ≈67.6% macro on *calibration* [M].
  - Grouping, flank-aware grouping, facing-aware training and generic local matching (DISK+LightGlue, RoMa) have all failed to fix this in the forms tested [M].
  - What does work is space-time-aware ranking and automating the easy matches [M].

**Three facts from outside research change how to pitch:**
1. **Volume is small in Croatia** [M external, I]. The national team analysed **4,201 lynx photographs over 14 seasons (225 individuals)** ([Bini et al., *J. Zool.*](https://www.citedrive.com/en/discovery/integrating-systematic-and-opportunistic-camera-trap-data-to-monitor-eurasian-lynx-demography/)). That is roughly 300 per season.
   - At our simulated W1 cost (2.46 h per 1,000 photos on *selection*), manual matching takes only a few expert-hours a season.
   - **"Saves you time on lynx matching" is therefore a weak selling point in Croatia.** "Fewer identification errors, consistent and auditable counts, and data ready for SCR and Article 17 reporting" is a stronger one.
2. **The Croatian experts have already tested AI and published.** [Blašković et al. 2025, *J. Zool.*](https://zslpublications.onlinelibrary.wiley.com/doi/abs/10.1111/jzo.70095) found:
   - MiewID operated by an expert reached 71.7% accuracy vs **92.7% for an experienced human** working manually;
   - novices struggled even with AI;
   - Whiskerbook caught one cross-border misidentification [M external].

   Any pitch to this audience must start from "human expertise remains essential". An automation claim would contradict their own published result.
3. **Expert identification is itself error-prone, which creates a real niche.**
   - Observers misclassified 12.5% of snow-leopard capture occasions and inflated abundance by about 35% ([Johansson et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7156508/)).
   - A computer-assisted re-check cut leopard individual counts from 59 to 46 (22%) and cheetah counts from 43 to 40 (7%) ([Verschueren et al. 2022](https://cheetah.org/resource-library/reducing-identification-errors-of-african-carnivores-from-photographs-through-computer-assisted-workflow/)).
   - A **"ghost and duplicate audit" of an existing expert catalogue** is a concrete, low-risk first pilot that plays to the system's measured strengths: ranking and pair evidence [I].

**Recommended acceptance criteria** are in Part 2. They are pitched against the measured human baseline rather than against perfection:
- ≤ 2% misassigned photos;
- |count error| ≤ max(1, 5% of individuals), with split and merge errors reported separately;
- ≥ 50% fewer expert decisions and ≥ 40% less measured wall-clock time than the partner's current practice;
- the same robustness when real reviewer error is measured;
- all of it on blind, partner-held data photographed after 15 October 2024.

Current evidence meets the misassignment figure in simulation. It fails the count, has no real-reviewer timing, no Croatian data and no night/IR stratification.

**The shortest honest path** (Part 4):
1. **Now, about 6 weeks:**
   - fix the two reproduced data-integrity bugs;
   - add the duplicate-merge second check and a conservative pruning floor;
   - build a timed review mode in the existing review app and time real reviewers on *calibration*;
   - open a research conversation with the Zagreb Veterinary Faculty lynx team about a blind retrospective test and a catalogue audit.
2. **Next, 2–4 months:** blind evaluation on Croatian post-October-2024 photographs held by the partner, including raw frames and night/IR strata.
3. **Later:** automated counts with intervals, via SCR models that tolerate identity uncertainty. These are gated on the embedding reaching ≥ 80% same-side top-1, or on a partner camera protocol with paired flank cameras.

---

## Part 1 — Audit of what has been achieved

### 1.1 Experiment timeline

Development-era work (before 16 Sep) used a 750-gallery / 560-query sample; later work used the phase 2 identity-disjoint pools (fingerprint `e3401b26…`).

| Date | Question | Split | Headline metric(s) | Verdict | Receipt |
| --- | --- | --- | --- | --- | --- |
| 12 Sep | Do bbox vs masked inputs and the thesis local+global ensemble work on lynx? | development | Full funnel top-1: bbox 104/303 (34.3%), mask 89/303 (29.4%) [M] | Thesis ensemble weak; masking alone hurt | `docs/research/czechlynx-analysis-2026-09-12.json`; AGENTS.md CzechLynx notes |
| 13 Sep | Is frozen MiewID-msv3 better than MegaDescriptor? | development | 156/303 (51.5%) top-1; +106/−11 queries vs MegaDescriptor [M] | MiewID adopted as baseline | `czechlynx-miewid-results-2026-09-13.md` / `.json` |
| 13 Sep | Does Fisher+GV add to MiewID? Does clustering work? | development | Global/+Fisher/full: 156/155/153; 8 mixed clusters for 60 animals, pair precision 7.42% [M] | Local adds nothing; clustering unusable | `czechlynx-miewid-full-pipeline-results-2026-09-13.md` |
| 13 Sep | Was MiewID trained on CzechLynx? | external | v2 card: 2,251 lynx training annotations; v3 inventory not public [M] | **Unresolved** | `miewid-czechlynx-training-provenance-2026-09-13.md` |
| 16 Sep | Global baseline on identity-disjoint pools | calibration / selection | Top-1 877/1,336 (65.6%) / 562/881 (63.8%); macro 56.1% / 47.3% [M] | Baseline holds on new animals | `czechlynx-phase2-{calibration,selection}-results-2026-09-16.json` → `metrics.global` |
| 22 Sep | Does RoMa dense matching discriminate identity? | calibration (1,000 pairs) | Inlier AUC 0.446 RoMa vs 0.696 DISK+LightGlue [M] | RoMa negative | `project-risnjak-assessment-2026-10-14.md` § RoMa; `evaluations/czechlynx/roma_pairwise_calibration/summary.json` |
| 14 Oct | Do 25 global-only clustering controls give counts? | calibration | 0/55 exact identity recovery in every profile [M] | Negative | `czechlynx-clustering-calibration-results-2026-10-14.json` |
| 14 Oct | Is the problem missing links or wrong links? | calibration | Oracle k50 connects 10/55; .55 cut: 98.2% precision / 1.2% recall [M] | Both | `czechlynx-pair-connectivity-results-2026-10-14.json` |
| 16 Oct | How good are the inputs? | fitting / calibration samples | Unusable markings 55/128 fitting, 27/128 calibration; crop/mask issue 24/128, 20/128 [M] | Real input-quality problem | `czechlynx-input-audit-review-results-2026-10-16.md`; `evaluations/czechlynx/input_audit_review_v1/receipt.json` |
| 16 Oct | First MiewID fine-tune (FP16) | fitting | Failed: non-finite gradient norm [M] | Corrected to FP32 | `czechlynx-miewid-training-failure-2026-10-16.md` |
| 17 Oct | FP32 v2 fine-tune | fitting (inner) → calibration | Calibration top-1 65.6% → 68.6%, macro 56.1% → 60.4%, mAP@50 .310 → .402; still 0 exact groups [M] | Recognition up; counts no | `czechlynx-miewid-fp32-calibration-results-2026-10-17.json` → `recognition` |
| 18 Oct | Missing links: images, encounters or candidate width? | calibration | Encounter links 10 → 28/55 oracle; k=100 → 23/55 [M] | Coverage fixable; wrong links now bind | `czechlynx-coverage-diagnostics-results-2026-10-18.md` |
| 18 Oct | Calibrated edge model + local evidence gate | calibration | Edge AUC .87–.89; best pairwise F1 .585, 1/55 exact; DISK+LightGlue adds ≈0.000 AUC, local-only identity-cross-fitted AUC .35 [M] | Local gate failed | `czechlynx-edge-grouping-results-2026-10-18.md` |
| 18 Oct | Flank-aware grouping | calibration | Facing AUC .995; F1 .602; 0/55 exact [M] | Negative for grouping | `czechlynx-flank-grouping-results-2026-10-18.md` |
| 18 Oct | Expert effort for a verified count | calibration | W1 5,283 decisions (1.8 h/1,000 placeholder), exact; W0 5.7×; group-first 1.27–2.92× W1 [M] | Ranked matching wins | `czechlynx-review-effort-results-2026-10-18.md` |
| 18 Oct | Confidence-gated automation | calibration | τ .99: ≈50% automated at 0.3–0.4% misassigned; top-K design leaves +28 to +70 duplicates [M] | Automation accurate; count design flawed | `czechlynx-gated-matching-results-2026-10-18.md` |
| 18 Oct | P(match) re-ranking + pruning + safeguards | calibration | Exact count, 0.19% misassigned, 2,211 decisions (−58% vs W1), 45% automated; 3% reviewer error → +1 / 2.3% [M] | Best development result; 70% automation and 0.5 h targets missed | `czechlynx-nonmatch-pruning-results-2026-10-18.md` |
| 18 Oct | Facing-aware re-adaptation (F1, F2) | fitting → calibration | Same-side NN macro 67.6 / 67.6 / 67.9% (v2/F1/F2); F1 protocol top-1 71.3%, safe automation 54% [M] | R1/R2 negative; F1 modest gain | `czechlynx-facing-adaptation-results-2026-10-18.md` |
| 19 Oct | Frozen pre-registered point, one shot | **selection (spent)** | 51.5% automated, 0.80% misassigned, **count +6**, 2,141 vs 5,244 W1 decisions; S1/S2/S4 fail [M] | Partial generalization | `czechlynx-frozen-selection-results-2026-10-19.md`; `evaluations/czechlynx/frozen_review_selection_v1/summary.json` |

### 1.2 Capability inventory

Maturity scale: **research script** < **tested module** (unit tests, real-data benchmark) < **demo-ready** (runs end-to-end on new data with a UI) < **field-validated** (measured on partner data).

| Capability | Where | Maturity | Evidence and caveats |
| --- | --- | --- | --- |
| Ingestion / storage | `reid_demo/ingest.py`, `store.py` | Research script, with **known blocking bugs** | Basename collisions and review-resetting upserts were reproduced 10 Sep (`docs/pilot-readiness-2026-09-10.md` §Gaps 1–2). Static check today: `ingest.py:307` still uses `Path(filepath).name`, and `store.make_record_id` still uses the stem [M] |
| Empty/species filtering | `reid_demo/species_filter.py` | Research script | Medvednica report has implausible species labels (pilot-readiness §3); untested for lynx |
| Segmentation | `segmentation/`, `reid_demo/benchmark_masks.py` | Tested module **only with supplied CzechLynx masks** | No lynx prompt in Grounded SAM2; predicted masks on raw frames are unevaluated (Risnjak assessment, engineering table) |
| Global embedding | `reid_demo/benchmark_embeddings.py`, `utils/finetune_czechlynx_miewid*.py` | Tested module | Public MiewID v3, v2 FP32 and F1 adapted. Upstream overlap unresolved. **MiewID is not wired into the field demo path** (Risnjak assessment) |
| Local features / GV | `feature_extraction.py`, `geometric_verification.py`, `benchmark_edge_grouping.py` | Tested module; **negative for lynx** in the tested form | See 1.4 |
| Retrieval + P(match) re-ranking | `benchmark_nonmatch_pruning.py`, `benchmark_frozen_review.py` | Tested module (*calibration* + one-shot *selection*) | Relies on CzechLynx day-level dates and coordinates; 88% of calibration photos have coordinates (decision log, 18 Oct) |
| Pruning / safeguards | same | Tested module, **fragile on new animals** | 3 pruning duplicates on calibration became 10 on selection |
| Clustering / grouping | `benchmark_clustering.py`, `benchmark_edge_grouping.py`, `benchmark_flank_grouping.py`, `reid_demo/cluster.py` | Tested module; **not usable for counts** | Best 1/55 exact recoveries |
| Counting | Sequential review simulation; thesis HITL-NIS (`nested_importance_sampling.py`) | Research script | Counts come from *simulated* reviewers. NIS CLI needs ground truth (Risnjak assessment) |
| Human review UI | `reid_demo/review.py`; sibling `../czechlynx-review-app/` | Demo-ready **for input audit only** | The sibling app audits image quality (321 images). It has no matching or catalogue workflow and no per-decision timing. `review.py` only checks within-cluster |
| Capture-history / SCR export | none | Absent | Required for Article 17 and SCR use (Part 2) |

**Overall [I]:** strong evaluation infrastructure (fingerprints, one-shot locks, identity-disjoint pools). The field path from raw camera frames to a verified catalogue is not demo-ready.

### 1.3 Best current operating point on unseen animals

Frozen configuration fingerprint `f2402829…` (commit `97571909`): F1 embedding; P(match) order; ε 0.001; W4 auto-accept at τ 0.99; all safeguards. Split: **selection** (4,725 photos, 55 lynx, 1,700 camera-day bundles). Simulated reviewer. Source: `evaluations/czechlynx/frozen_review_selection_v1/summary.json` [M].

| Measure | Value | JSON key |
| --- | ---: | --- |
| Automated bundles | 51.5% (876/1,700) | `simulations[0].automation` |
| Misassigned photos | 0.80% (38) | `criteria.S1.misassigned_photo_rate` |
| Count error | **+6** (61 groups) | `criteria.S1.count_error` |
| Exact animal recoveries | 40/55 | `simulations[0]`; results doc table |
| Expert decisions | 2,141 (453 per 1,000 photos) | `criteria.S2` |
| Expert-hours per 1,000 photos | 1.01 at placeholder costs (5 s glance / 8 s comparison / 10 s pair question); 0.47 at fast-click costs | `simulations[0].hours_per_1000_photos*` |
| W1 (full manual ranked matching) | 5,244 decisions, 41,859 s = 11.6 h = **2.46 h per 1,000 photos** | `w1."0.0"` |
| Saving vs W1 | **59.2%** of decisions; 59.1% of placeholder hours [I: 1 − 4.76/11.63] | `criteria.S2.saving_vs_w1` |
| 3% reviewer error | 3.58% misassigned, count 0 (17 split and 14 merge errors cancel) | `criteria.S4`; results doc |
| By source | snpa +6 / 1.41%; foe_bohemia +1 / 0.57%; foe_carpaths 0 / 0.29% | `breakdowns.by_source` |

The pre-registered criteria, as declared in [the proposal](../czechlynx-frozen-selection-proposal-2026-10-19.md):

- **S1** (≤1% misassigned **and** |count error| ≤1): **failed.** The cause is pruning: 10 new animals were declared while the true animal was in the catalogue but below ε; the duplicate checks merged only 3. The likely mechanism is a P(match) location prior that does not transfer to snpa [H, stated as such in the results doc].
- **S2** (≤416 decisions per 1,000 photos and ≥50% saving): **failed** on the absolute limit; the relative saving passed. Selection bundles are smaller (2.8 vs 3.5 photos), so even W1 costs 1,110 vs 838 decisions per 1,000 photos [M]. The absolute limit was badly posed [I].
- **S3** (both audit intervals cover the truth): **passed** with a perfect auditor. The count-audit interval is [44, 61] around a true 55 [M]. That is about ±15%, too wide to support a count claim [I].
- **S4** (3% reviewer error: |count| ≤2 and ≤3% misassigned): **failed** (3.6%). Duplicate merges have no second check: 6 of 10 merges were wrong [M].

**Context [I]:** the simulated 3% reviewer error is optimistic. Experienced manual identification in Croatia measured 92.7% accuracy (Blašković et al.). The denominators differ (identifying an encounter against a catalogue vs answering a yes/no pair question), but robustness should be tested at 5–10% too.

### 1.4 Ruled-out approaches: what each negative rules out and what it leaves open

| Approach | Result [M] | Rules out | Leaves open |
| --- | --- | --- | --- |
| Global-only clustering (25 controls; Louvain, thresholds, consistent merging) | 0/55 exact on calibration | Fixed-threshold or kNN-graph clustering of MiewID vectors as a count method | Constrained clustering with expert pair labels; encounter-level units |
| Thesis Fisher + GV funnel (summed Fisher vectors, homography GV) | 156 → 153/303 (development) | That configuration as implemented | Calibrated fusion. The Risnjak assessment lists 6 integration flaws: uncalibrated averaging, summed Fisher spaces, homography on deforming bodies, mask-centre-only filtering, top-10 shortlist, generic matcher training |
| DISK + LightGlue as edge evidence | +0.000 AUC; local-only identity-cross-fitted AUC .35; equally "predictive" on opposite-facing pairs (.585 vs .608), so it encodes image conditions, not spots | Generic keypoints with homography/affine on masked crops | **Pose-rectified flank patches**, spot-specific descriptors (HotSpotter/SIFT), non-rigid verification, same-observation diagnostic (§3B) |
| RoMa dense matching | Inlier AUC .446; 97 inliers on a false pair vs 80 on a true pair | Dense correspondence counts as identity evidence | Dense matching restricted to identity-bearing regions |
| Flank-aware grouping | Facing AUC .995, but grouping F1 .602, 0/55 exact | Splitting the graph by facing as a grouping fix | Linking flanks via synchronized paired cameras (needs partner protocol) |
| Facing-aware re-adaptation (F1, F2) | Same-side NN macro 67.6–67.9% vs 67.6% | Loss or class-definition changes on the same fitting data | More or different training data; pose/part models; night-specific training |
| Abstaining on low-quality images | Risk AUC .73, but abstention does not connect animals | Quality filtering as the main lever | Quality as a reviewer-routing signal |
| Top-K-only new-animal check | +28 to +70 duplicates | Declaring new animals after a short scan | Full-scan or calibrated pruning with a floor |
| Aggressive pruning (ε 0.001, P(match)-ranked) | Calibration 0–3 extra animals; selection 10 | The calibrated "not this animal" model as a free cost cut | Conservative floors; cheap duplicate sweeps |

**The user's integration-flaw hypothesis [H]** is not refuted for *calibrated fusion of identity-bearing local evidence*. It is refuted for *generic DISK+LightGlue homography inliers as edge evidence on these crops*. External evidence agrees:
- In AnimalCLEF 2026, catalogue anchoring for lynx was carried by global descriptors. "Anchoring on LightGlue alone is much weaker" (0.038 vs 0.149 ARI; [Colling 2026](https://ceur-ws.org/Vol-4283/paper332.pdf), working notes).
- Lynx top-1 catalogue accuracy there was only 0.232 [M external].

### 1.5 Open risks

| Risk | Status | Evidence |
| --- | --- | --- |
| Upstream MiewID overlap with CzechLynx | Unresolved; the author email was policy-blocked | Provenance doc. **Croatian lynx photos were on Whiskerbook before v3**, so Croatian data is not automatically clean [M] |
| Night/IR | No verified labels. Uniform audit samples (reviewer assertions): calibration 55/128 monochrome/IR/night-flagged, fitting 59/128. Unusable markings: monochrome 14/47 vs colour 12/67 (calibration), 31/54 vs 23/67 (fitting) [M, from `reviews.json` × `manifest.json` cohorts] | Roughly 40–45% of photos look monochrome. Their markings are more often unusable; same-condition and cross-condition performance is unmeasured |
| Flank handling | Facing predictable (AUC .995); opposite flanks almost never link by appearance (2.6% top-50) [M]; left/right audit convention unconfirmed | Counts need either paired cameras or explicit bilateral bounds |
| Input quality | Unusable markings 43% of fitting and 21% of calibration uniform samples [M] | Training data are noisier than field experts' curated sets |
| Single-dataset dependence | All method evidence is from CzechLynx masked, annotation-assisted crops with day-level dates | No transfer test to new sites, raw frames or Croatian cameras |
| No Croatian data | — | Every number in this document is non-Croatian |
| Pruning fragility | 3 → 10 duplicate events from calibration to selection [M] | P(match) priors are site-specific |
| Audits under a noisy auditor | The count audit covers the truth in only 0–3% of runs with a 3% auditor (calibration) [M] | No defensible uncertainty statement yet |
| Engineering integrity | Collisions and review resets still in code [M static] | Blocking for any partner archive |
| Licence | MiewID-msv3 licence disputed on Hugging Face ([discussion](https://huggingface.co/james-burgess/miewid/discussions/1)) [M external] | Commercial use needs clarification with Conservation X Labs [I] |

### 1.6 Gap table

| Pitch requirement (Part 2) | Current state | Evidence | Gap |
| --- | --- | --- | --- |
| Misassignment ≤ 2% after review | 0.80% (simulated perfect reviewer, selection) | `frozen_review_selection_v1/summary.json` | Small in simulation; **unknown with real reviewers** |
| \|count error\| ≤ max(1, 5% of N) | +6 / 55 = +10.9% | same | **Large** |
| ≥ 50% fewer decisions than W1 | 59.2% | same | Met (simulated) |
| ≥ 40% less wall-clock time than the partner's practice | Never measured | — | **Large (no data)** |
| Robust at measured real reviewer error | 3.6% at an assumed 3% | `criteria.S4` | **Large** |
| Uncertainty interval that covers the truth with noisy auditors | Fails | nonmatch-pruning results | **Large** |
| Night/IR stratified results | None | — | **Large** |
| Raw frames → crops → re-ID | Not evaluated | — | **Large** |
| Independent data (post-Oct-2024 or partner-blind) | None | provenance doc | **Large** |
| Collision-safe ingestion; review preservation | Bugs present | `reid_demo/ingest.py:307` | Medium (known fix) |
| SCR / capture-history export | Absent | — | Medium |
| Review UI for matching with timing | Absent | sibling app README | Medium |

---

## Part 2 — What "ready to pitch" means

### 2.1 How lynx are monitored today

**Croatia and Slovenia (Dinaric).**
- Coordinated camera-trap monitoring since 2018 under LIFE Lynx (July 2017 – March 2024; €6.83 M total, €4.08 M EU) [M]. Results are published annually.
- **Headline counts are usually minimum population sizes** (adult lynx identified from camera traps). SCR models are used for more realistic abundance ([LIFE Lynx](https://www.lifelynx.eu/?p=25946)) [M].
- Surveys run **August–February** ([Dryad, Slovenia 2018–2023](https://datadryad.org/dataset/doi:10.5061/dryad.cnp5hqcgm)) [M].
- Detection is highest at **marking sites**, then forest roads, then other locations: scent-marking sites have 1.6× and 2.5× higher baseline detection ([Fležar et al. 2023](https://link.springer.com/doi/10.1007/s10531-023-02646-3)) [M].
- Northern Dinaric density: 0.83 / 100 km² (95% CI 0.60–1.16), about 96 lynx over 11,536 km² [M]. A later LIFE figure for 2022–23 is 1.27 ± 0.15 / 100 km², 156 ± 19 adults in Slovenia and Croatia [M, different analysis].
- Many cameras are run by hunters and volunteers [M].
- **Croatia-wide** ([Bini, Topličanec, Sindičić, Gomerčić, *J. Zool.*](https://www.citedrive.com/en/discovery/integrating-systematic-and-opportunistic-camera-trap-data-to-monitor-eurasian-lynx-demography/)) [M]:
  - 4,201 lynx photographs, 2010–2024, of which 41% were opportunistic;
  - 225 individuals, mean 17.4 records each;
  - 201 individuals in 2018–2023; the minimum population rose from 64 to 89;
  - 31% annual turnover; sex unknown for 33% of animals.
- **Who identifies:** the University of Zagreb Veterinary Faculty team (Sindičić, Gomerčić, Topličanec, Blašković) [M]. They have used Whiskerbook (Blašković et al.).
- Croatia adopted a ten-year lynx management plan in May 2024 ([LIFE Lynx](https://www.lifelynx.eu/?p=28644)) [M].
- A national large-carnivore monitoring programme for Habitats Directive reporting was developed by the Veterinary Faculty with Geonatura under a Ministry project ([LIFE Lynx](https://www.lifelynx.eu/?p=27173)) [M].
- An After-LIFE plan (July 2024) covers five years of surveillance ([LIFE Lynx](https://www.lifelynx.eu/with-the-after-life-conservation-plan-we-are-safeguarding-the-future-of-lynxes-in-dinaric-mountains-and-se-alps/)) [M]. No funded successor LIFE project was found [M, absence in search].
- Plitvice: 123 historical camera locations; 26 lynx 2014–2022; 7 observed in 2022 ([NP Plitvice](https://np-plitvicka-jezera.hr/ris/)) [M, from the Risnjak assessment].

**Switzerland (KORA):** the "deterministic" design runs reference areas every 3–4 years for 60 nights with **paired cameras at each site to photograph both flanks**. "For a definitive identification, both flanks … must be photographed." Example: Southern Jura 2022/23, 76 sites / 152 cameras, abundance 28 (27–45), 3.61 / 100 km² ([KORA](https://www.kora.ch/en/projects/large-carnivore-monitoring/deterministic-camera-trap-monitoring)) [M].

**France (OFB Réseau Loup-Lynx):** more than 4,600 photo events (tens of thousands of images). "Each photo event is examined with an identification aid program, then validated by a human" ([OFB](https://ofb.gouv.fr/methode-de-photo-identification-du-lynx)) [M]. They use Whiskerbook with MiewID ([Wild Me forum](https://community.wildme.org/t/de-prioritize-hotspotter-for-eurasian-lynx-in-whiskerbook/2626)) [M].

**Czechia (CzechLynx sources):** 39,760 images, 319 individuals, three sources, 2009–2025. Identities were cross-checked by ≥2 independent observers, and **ambiguous cases were excluded** ([Picek et al.](https://arxiv.org/html/2506.04931v2)) [M].

Implication [I]: the benchmark contains only identifiable, expert-agreed photographs. Field data also contain unidentifiable ones, so benchmark accuracy overstates field accuracy unless "unresolved" is handled.

**Night/IR share:** no published lynx figure was found. CzechLynx mixes white-flash and IR cameras [M]. Our uniform audit samples flag about 43–46% as monochrome/IR/night by appearance or reviewer assertion [M, not verified labels].

**Time per season:** no published expert-hours figure was found [M absence]. Rough scale [I]: roughly 300 Croatian lynx photos per season. Even at 10× our placeholder W1 cost (24.6 h per 1,000 photos), that is about 7 expert-hours. The Croatian bottleneck is therefore expertise, consistency and the sorting of non-lynx images, not lynx-matching hours.

### 2.2 What parks and agencies must report

- **Minimum number of individuals** per season or area. This is the LIFE Lynx annual headline [M]. It is sensitive to **split errors (ghosts)**, which inflate it.
- **SCR density / abundance with a CI** for the defined area [M]. Ghosts bias it upwards. Even 1–5% identification error can bias population parameters ([Augustine seminar](https://www.gla.ac.uk/schools/mathematicsstatistics/events/details/?id=10852)) [M external, abstract-level].
- **Reproduction, sex, turnover**: females with kittens, age (Bini et al.) [M].
- **EU Habitats Directive Article 17** (6-year cycles; the 2019–2024 report was due in 2025): population size and trend feed the conservation-status assessment. The exact Croatian 2019–2024 submission figures were not retrieved [M absence]. The relevant tolerance is that of the SCR / minimum-count inputs.

**Tolerance [I].** Agencies already accept counts produced with human error rates in the 7–22% range for leopards and cheetahs, and inflation of about 35% for snow leopards, mostly without knowing it. A tool that **measures and reports** its error, and keeps it below the human baseline, improves the reported number even if it is not perfect.

### 2.3 Human-expert baseline error, turned into targets

| Source | Species | Measure | Value |
| --- | --- | --- | --- |
| [Johansson et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7156508/), *Sci. Rep.* (peer-reviewed) | Snow leopard (captive, 16 animals, 8 observers) | Capture occasions misclassified; abundance bias | 12.5%; +35 ± 21% |
| [Verschueren et al. 2022](https://cheetah.org/resource-library/reducing-identification-errors-of-african-carnivores-from-photographs-through-computer-assisted-workflow/), *Mamm. Res.* (peer-reviewed) | Cheetah / leopard | Individuals removed after a computer-assisted re-check | 7% / 22% |
| [Blašković et al. 2025](https://zslpublications.onlinelibrary.wiley.com/doi/abs/10.1111/jzo.70095), *J. Zool.* (peer-reviewed; abstract-level access) | Eurasian lynx, Croatia | Experienced manual vs MiewID operated by an expert; MiewID vs HotSpotter encounters matched | 92.7% vs 71.7%; 64.0% vs 45.5% |
| [Gooliaff & Hodges 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6262731/), *Ecol. Evol.* (peer-reviewed) | Bobcat vs Canada lynx (*species*, not individuals) | Expert agreement | Fleiss κ 0.64 |

**Derived targets [I]:**
- **Misassigned photos ≤ 2%** after review. That is three to six times better than the 7–12.5% human misclassification rates above; it is achievable in simulation (0.8%) and leaves room for real-reviewer degradation.
- **Count error ≤ max(1, 5% of individuals), reported as split and merge counts.** This is at least four times smaller than the ghost inflation documented above. A 55-animal set allows ±2.75 animals (round to 3), so the current +6 fails.
- **Non-inferiority to the partner's own manual result** on the same blind set. This is the honest comparison a park will ask for.

### 2.4 Existing tools and what they do not do

| Tool | What it does | Gap the product could fill |
| --- | --- | --- |
| **Whiskerbook / Wildbook for Lynx** (Wild Me / Conservation X Labs) | Catalogue with HotSpotter + MiewID ranked matching; 25,652 encounters / 1,126 individuals listed; cross-organization sharing ([Wild Me](https://www.wildme.org/wildbook-for-lynx.html)) [M]. MiewID v4 announced 30 Jan 2026: +12% for Eurasian lynx, about 90 species; weights not found on Hugging Face ([announcement](https://community.wildme.org/t/miewid-v4-announcement/5406)) [M] | No count with measured error; no automation gate; no audit trail of the count; hosted only; reported lynx-matching degradation episodes ([forum](https://community.wildme.org/t/miewid-is-feeling-tired-a-lot/4709)) [M] |
| HotSpotter | SIFT-based ranked matching | Weaker than MiewID for lynx (45.5% vs 64.0%) [M] |
| Wildlife Insights, Agouti, TRAPPER, Camelot | Camera-trap data management and species classification | No evidence of individual lynx identification features was found [M absence; not exhaustively verified] |
| Spotted ([Kelebek et al. 2026](https://arxiv.org/abs/2607.00804), preprint) | Location-informed re-ID + active expert queries for hyena/leopard; up to 69 pp fewer queries | Research code; same idea as our P(match). Confirms the approach, removes novelty [I] |

**Positioning [I]:** compete on *measured, auditable counts and an end-to-end pipeline that runs on local hardware*, not on matching accuracy. Whiskerbook is free to partners and already used by these teams. Interoperability (import and export of Whiskerbook catalogues) beats replacement.

### 2.5 Buyer reality

- **Who pays [M/I]:**
  - The Ministry (MZOZT / formerly MINGOR) funds monitoring programmes and Article 17 reporting, through the Institute for Nature Protection.
  - Scientific execution sits with the Zagreb Veterinary Faculty.
  - Parks are public institutions with small discretionary budgets: Risnjak's March 2026 ministry allocation for priority activities was **€21,863** ([MZOZT decision](https://mzozt.gov.hr/UserDocsImages/UPRAVA%20ZA%20ZA%C5%A0TITU%20PRIRODE/ODLUKE_ZASTICENA%20PODRUCJA/Odluka_prioritetne%20aktivnosti_ozujak%202026.pdf)) [M].
  - Risnjak has a €1.99 M EU-cohesion project, "Poboljšano upravljanje NP Risnjak" (2021–2027 programme; approved Nov 2025) [M]. Whether it covers monitoring services is unknown [H].
  - LIFE / Interreg calls fund multi-partner projects. LIFE Lynx ended in March 2024 [M].
- **Procurement path [I]:** a small direct service contract (simple procurement) for a pilot, or a subcontract or partner role in a research grant. A commercial licence sale to a park is unlikely to be the first step.
- **What a pilot must show [I]:** an independent result on *their* photographs; non-inferiority to their experts; a clear time or error benefit; data stays under their control; exportable capture histories.

### 2.6 Acceptance criteria (revision of S1–S4)

S1–S4 were sound for a simulation gate. For a pitch they need three changes:
- an absolute cost limit is replaced by measured wall-clock time;
- reviewer error is measured rather than assumed;
- independence and stratification are required.

Proposed **P-criteria** for framing (a). All are evaluated **once** on blind, partner-held data photographed after 15 October 2024; the partner keeps the labels.

| ID | Criterion | Justification |
| --- | --- | --- |
| P1 | Final reviewed catalogue: ≤ 2% misassigned photos **and** \|count error\| ≤ max(1, ⌈0.05·N⌉); split and merge errors each reported | §2.3 human baseline |
| P2 | ≥ 50% fewer expert decisions than W1 **and** ≥ 40% less measured wall-clock time than the partner's current workflow on the same photos | The real buyer value; the 18 Sep pilot gate targeted 30% |
| P3 | At the partner experts' **measured** pair-question error rate, P1 still holds with safeguards on (and at a 5% stress setting, ≤ 3% misassigned, \|count\| ≤ 2) | S4 generalized; human error is 7%+ |
| P4 | Count-audit and automation-audit intervals cover the truth with two-person consensus auditing | S3 made robust to auditor noise |
| P5 | Results reported separately for day-colour vs monochrome/IR (verified via EXIF or camera mode), and for same-camera vs other-camera links; no stratum's misassignment above 2× the overall rate | Night/IR risk |
| P6 | Raw camera frames in, including empties and other species; lynx detection recall ≥ 95% on an expert-checked sample | End-to-end claim |
| P7 | No data-integrity failure (collisions, review loss) over the pilot | Blocking engineering |

**Framing (a), human-verified count with an audit trail at X% of manual effort:** supported **in simulation** by current evidence for the misassignment and decision-saving parts. X ≈ 41% of W1 decisions. Not yet supported for the count, real time, real reviewers or field data.

**Framing (b), automated count with an uncertainty interval:** **not supported.**
- The best automated grouping recovers 1/55 animals exactly.
- The count interval [44, 61] is ±15% and fails with a noisy auditor.
- The embedding ceiling (same-side top-1 ≈ 68%) means an automated count without paired-flank cameras would need an SCR-type model with latent identities. That model is not built, and identity uncertainty is known to make such estimates imprecise and biased when data are sparse ([Augustine et al., categorical SPIM](https://www.usgs.gov/publications/spatial-capture-recapture-categorically-marked-populations-application-genetic-capture)) [M external].

---

## Part 3 — Improvement research mapped to the codebase

**Rules for every experiment below:**
- Method choices may be made on *fitting* and *calibration* (identity-cross-fitted).
- Confirmation needs **new held-out data D_new**. Options are listed in §3G.
- Nothing touches *selection* (spent) or the official *test* (locked).
- Effect sizes are guesses with stated confidence, not promises.

### A. Embedding discrimination (the binding constraint)

| Candidate | Hypothesis | Evidence | Plugs into | Metric, expected effect, confidence | Cost | Risks | Decisive experiment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **A1. MiewID v4** | More lynx training data raises same-side discrimination | Publisher reports +12% for Eurasian lynx ([announcement](https://community.wildme.org/t/miewid-v4-announcement/5406), vendor claim, not peer-reviewed) | `reid_demo/benchmark_embeddings.py` (new pinned revision) | Same-side NN macro top-1: +3–8 pts, **low confidence**; overlap risk *increases* | ~1 GPU-h if weights are obtainable | Weights not public; licence; more overlap with CzechLynx and Croatian data | Frozen v4 vs v3 vs F1 on calibration same-side NN macro top-1. Success: ≥ +3 pts **and** no gain concentrated in pre-2024 identities. Confirm on D_new |
| **A2. Pose-rectified flank patches** | Matching the torso-flank region normalized by the 20-point skeleton isolates spots from pose and background | CzechLynx provides pose keypoints for 5,206 images [M]; part-pose methods help tigers ([Liu et al. 2019](https://openaccess.thecvf.com/content_ICCVW_2019/html/CVWC/Liu_Part-Pose_Guided_Amur_Tiger_Re-Identification_ICCVW_2019_paper.html), peer-reviewed workshop); part tokens (PAW-ViT, [WACV 2026](https://www.lirmm.fr/~chaumont/publications/WACV2026_DIAS-RIBERO-NETO_CHAUMONT_SUBSOL_DE-GARINE-WICHATITSKY_GUIS_Animal-Part-Based-ReId.pdf)) | New `reid_demo/benchmark_flank_patches.py`; reuses facing classifier | Same-side pair AUC on confident same-facing pairs: +0.03–0.08, **low–medium**. It is the one untested local variant | 1–2 weeks; ~5 GPU-h; pose model for unannotated images | Keypoint coverage only 14.6% of fitting photos; predicted pose may fail at night | On the 7,744 same-facing confident calibration edges: identity-cross-fitted AUC of patch embedding + global vs global alone. Success: ≥ +0.03 AUC overall and ≥ +0.05 in the ambiguous band |
| **A3. Encounter-level aggregation** | Pooling the best 3–5 frames of a camera-day bundle reduces per-photo noise | Bundles already used (W1 is 3.1× cheaper with them) [M]; quality risk score AUC .73 [M] | `benchmark_nonmatch_pruning.py` features (bundle-mean or quality-weighted vector) | Rank-1 share of known bundles: +2–5 pts, **medium** | CPU only, days | Mixed-animal bundles (22 encounters) | Calibration replay: quality-weighted bundle vectors vs current. Success: rank-1 share ≥ +2 pts and no rise in misassignment at τ .99 |
| **A4. Test-time flips/crops** | Cheap invariance gains | General re-ID practice [I] | Embedding extraction | +0–2 pts, **low**. Horizontal flips change facing semantics; do not flip | ~1 GPU-h | Destroys flank information | Same as A1 protocol; stop if < +1 pt |
| **A5. DINOv2/v3 + metric head trained on lynx** | A general backbone fine-tuned on lynx beats a specialist | Mixed: DINOv2 triplet tuning did not help lynx in AnimalCLEF 2026 ([Colling](https://ceur-ws.org/Vol-4283/paper332.pdf)); a frozen DINOv2-G head beat MegaDescriptor elsewhere (Research Square preprint) | `utils/train_megadescriptor_scratch.py` pattern | **Low** confidence; useful mainly as a **leakage-free reference** (H) | ~10 GPU-h | Likely worse than MiewID | Run only as part of H |
| **A6. More, different lynx training data** | The ceiling is data, not loss (the (b) conclusion) | F1/F2 negative [M] | Fitting pool expansion | Unknown; **the most likely real lever [H]** | Data agreements | Licence/consent; overlap | Retrain v2 recipe with + partner data (only data photographed before the D_new cutoff); evaluate on calibration same-side top-1 |

### B. Local evidence done right

The decisive **same-observation diagnostic** for the integration-flaw hypothesis [H] should run on calibration only, CPU plus a small GPU budget, as an additive proposal.
- **Sample:** a fixed set of about 2,000 calibration pairs, stratified by:
  - same/different animal;
  - same/opposite facing (predicted, confident);
  - day-colour/monochrome;
  - same/different camera;
  - global-rank bin.
- **Five stages on the identical pairs:**
  1. **Candidate:** is the true animal in the global top-K? Separates shortlist misses from evidence failure.
  2. **Mask/ROI:** keypoint retention inside the mask vs inside a pose-defined flank polygon; boundary-distance histogram.
  3. **Geometry:** inliers under homography, partial affine and thin-plate/TPS-RANSAC; reprojection-error spread; inlier spatial coverage on the flank.
  4. **Descriptor:** DISK, ALIKED and SIFT-RootSIFT (HotSpotter-like) on the same ROI.
  5. **Fusion:** identity-cross-fitted logistic/isotonic calibration of each signal (WildFusion-style; [Čermák et al. 2024/25](https://arxiv.org/abs/2408.12934), peer-reviewed LNCS) vs the existing raw averaging in `reid_demo/benchmark.py::rank_queries`.
- **Pre-registered outcome:**
  - The **hypothesis is supported** if any ROI × geometry × descriptor combination gives identity-cross-fitted AUC ≥ 0.70 on same-facing confident pairs **and** adds ≥ 0.02 AUC to the global+space-time edge model.
  - It is **refuted for this data** if all combinations stay ≤ 0.62 or show the same AUC on opposite-facing pairs (the image-condition signature).
- **Plugs into:** new `reid_demo/benchmark_local_diagnostic.py` + CLI; reuses `benchmark_edge_grouping.py` features and the `edge_local_pilot` SQLite store.
- **Expected:** refutation is more likely (**medium confidence**), given the AnimalCLEF lynx result and the opposite-facing test. Its value is to close the question honestly.
- **Cost:** about 1 week; about 2 GPU-h.

Other B candidates: **ALIKED+LightGlue** (WildFusion standard; low confidence of change), **MASt3R-style matchers** (dense correspondence again; RoMa failed, so low priority [I]). Spot-specific SIFT on rectified patches is covered by A2.

### C. Encounter, sequence and space-time reasoning

| Candidate | Hypothesis | Evidence | Plugs into | Metric / effect / confidence | Cost | Risks | Decisive experiment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **C1. Conservative pruning floor** (always show the top-m candidates; ε with a margin; disable the space-time prior for a cold site) | Removes most of the 10 selection duplicates | 3 vs 10 duplicate events [M] | `benchmark_nonmatch_pruning.py` (new rule) | Duplicates −70%, at +10–20% decisions, **medium** | CPU, days | Erodes the saving | Calibration with **leave-one-source-out** P(match) fitting (simulates a new site). Success: pruning duplicates ≤ 1 per 1,000 photos in every held-out source with decisions ≤ 0.5× W1 |
| **C2. Second check on duplicate merges** | Fixes most S4 failures | 6/10 merges wrong under 3% error [M] | Review simulation | Misassigned at 3% error 3.6% → ≤ 2%, **medium–high** | CPU, days | Small cost | Calibration simulation at 3%, 5% and 7% error. Success: P3 holds at 5% |
| **C3. Spotted-style speed feasibility** | A minimum-travel-speed feasibility score generalizes across sites better than learned location priors | [Kelebek et al. 2026](https://arxiv.org/abs/2607.00804) (preprint): top-5 +2–9 pts, up to 69 pp fewer queries | Replace location-cell features in P(match) with a speed/feasibility feature | Cross-source transfer of P(match) calibration, **medium** | CPU, 1 week | Day-level CzechLynx dates weaken it; exact timestamps in park data help | Leave-one-source-out calibration: Brier score and duplicate rate vs current P(match). Success: ≥ 30% lower held-out-source Brier |
| **C4. Paired-camera flank linking** | Synchronized opposite-flank photos link left and right catalogues deterministically | KORA protocol [M]; SPIM uses double-camera stations ([Augustine et al. 2018](https://projecteuclid.org/euclid.aoas/1520564465), *Ann. Appl. Stat.*) | Ingestion: deployment metadata with a station ID and both camera IDs | Bilateral-count interval width → near 0 when coverage is good, **high** if the protocol exists | Partner protocol; no GPU | Croatian systematic surveys may be single-camera [unknown] | Ask the partner; measure the fraction of individuals with a simultaneous two-flank capture |

### D. Counting with uncertainty

- **D1. Report the bilateral bounds the data justify [I].** Count = left-flank catalogue ∪ right-flank catalogue with links; interval [pairs + max(unmatched), pairs + sum(unmatched)]. This is already implemented in `benchmark_flank_grouping.py` (33/144 intervals contained the truth, but width ≥ 19 [M]). Good audit-trail output; poor precision.
- **D2. Expert-verified catalogue → SCR with residual misidentification sensitivity.** Export capture histories (individual × station × occasion) and give the partner's ecologist a sensitivity analysis: refit SCR after injecting the measured split and merge rates. Repo: new `reid_demo/capture_history.py`. **High value, low risk** [I]. Decisive check: on CzechLynx calibration, the injected +6-style split errors change the minimum count by +11%. Report it as a sensitivity table, not a result.
- **D3. SPIM / latent-identity SCR** ([Augustine et al. 2018](https://projecteuclid.org/euclid.aoas/1520564465); [categorical SPIM](https://www.usgs.gov/publications/spatial-capture-recapture-categorically-marked-populations-application-genetic-capture)): the route to framing (b). P(match) probabilities could serve as soft identity evidence. Imprecise when data are sparse [M external]. **Later**; needs an ecologist co-author.
- **D4. HITL-NIS (thesis).** An unbiased count estimator with a perfect oracle (Risnjak assessment). Robustness to noisy oracles is thesis-simulated only. Keep as a cross-check on catalogue counts [I].
- **Decisive experiment for (b)**, using D_new only: on two independent sites, the 95% interval covers the expert-verified count in both and has width ≤ ±10%. Kill (b) for this horizon if, on calibration, no interval method reaches ≥ 90% coverage at ≤ ±15% width with a 5% noisy auditor.

### E. Night/IR

- **E1. Verified labels first [I].** Derive day/night from EXIF capture time + sun elevation, and IR mode from EXIF/maker notes or camera model where the raw files exist. CzechLynx has only day-level dates [M], so verified labels must come from partner raw files. Cost: hours of scripting, no labelling.
- **E2. Stratified evaluation:** same-condition vs cross-condition same-side NN top-1 and misassignment (P5).
- **E3. Grayscale-consistent inference** (convert everything to luminance before embedding). Cheap. Blašković et al. found AI did *better* on black-and-white images [M external], so there may be little gap to close. Expected effect: **low confidence** either way.
- **E4. Cross-condition positives in fine-tuning:** only after E1. Literature on visible-to-IR animal re-ID is thin [M absence]. IR infrared-video snow leopard pose-part results (94.5% mAP) are reported in secondary search snippets and were not verified.
- **Decisive:** on D_new, stratified P1/P5. Kill night automation (route all night bundles to an expert) if night misassignment exceeds 2× day.

### F. Human-in-the-loop

- **F1. Timed matching review mode** in the sibling app: catalogue view, ranked candidates, yes/no/unsure, per-decision timestamps. **This is the single most important missing measurement [I].** Proposal needed; the app is outside the repo.
  - Decisive experiment: two reviewers, 300 calibration bundles each, randomized W1 vs frozen-workflow order.
  - Success: measured median seconds per decision lets the frozen workflow reach ≥ 40% wall-clock saving vs W1, with measured error rates recorded for P3.
- **F2. Pair-label active learning on top-K shortlists** (AGENTS.md "Future" note; [Guo et al.](https://www.engr.colostate.edu/~aam/pdf/journals/120.pdf) report a 77% cut in human involvement for leopards at manual-baseline accuracy [M external, dissertation/under review]). Use partner pair labels to recalibrate the W4/P(match) models per site; this addresses pruning fragility.
- **F3. Risk-controlled automation.** [Conformal risk control](https://arxiv.org/abs/2208.02814) (Angelopoulos et al., ICLR 2024, peer-reviewed) gives finite-sample bounds on expected misassignment for the auto-accept threshold. It requires exchangeability between calibration and deployment; a new site breaks it [I]. Use it per site, with a small labelled burn-in (first 100 bundles fully reviewed).
- **F4. Noise-robust audits:** two-auditor consensus, or a bias-corrected estimator using a measured auditor confusion rate. CPU simulation on calibration. Success: count-audit coverage ≥ 90% at a 5% auditor error.

### G. Data: new held-out sets

| Source | What | Independence | Status |
| --- | --- | --- | --- |
| **Croatian partner, photographed after 15 Oct 2024** (2024/25 and 2025/26 seasons) | Raw frames + expert IDs, held blind by the partner | Image-clean for MiewID v3 by date [I]; individuals may be seen (pre-v3 Whiskerbook uploads) | **Primary D_new.** Needs a data-sharing agreement |
| Croatian archive before Oct 2024 | Fitting data (A6) and catalogue-audit pilot | Not clean for testing | Secondary |
| CzechLynx official time_open test (locked) | 1,614 photos dated after the v3 upload, 13 identities [M] | Note: 82 of 126 official test identities appear in official train [M, paper]; our fitting pool used official-train identities, so most of these are **known animals** to the adapted models [I] | Keep locked. Spend once only, by explicit user decision, as a final confirmatory test |
| Other programmes (KORA, OFB, Bavarian Forest NP, Slovak/Polish Carpathians) | Paired-flank data (KORA) | Varies | Later; good for a site-transfer test |
| AnimalCLEF LynxID2025 | Derived from CzechLynx [M] | **Not independent** | Do not use for testing |

**Fresh locked-test procedure [I]:**
- The partner exports hashes of the photographs before anything is shared and keeps the identity labels.
- We freeze and commit the configuration fingerprint (the existing one-shot lock pattern in `benchmark_frozen_review.py`), then run once.
- The partner scores the output.
- A dated pre-registration document is committed before data access.

### H. Resolving upstream leakage

1. **Ask:** the Wild Me community forum (staff-monitored), rather than the blocked email. Ask for a CzechLynx / Whiskerbook-Czech inclusion statement for msv3 and v4.
2. **Temporal test:** performance on post-2024-photographed images, Croatian or CzechLynx locked test, compared with pre-2024. Separates image memorization, not individual exposure.
3. **Identity novelty test:** individuals *first* detected after Oct 2024 (kittens or immigrants; the 31% Croatian annual turnover gives many [M]) vs long-known individuals, same time window. A large gap in favour of old animals suggests individual-level exposure [H].
4. **Clean reference model:** A5 trained only on our fitting pool. If MiewID's advantage over it vanishes on new-born identities, exposure is likely.
5. Decisive criterion (pre-register): **exposure is "material"** if same-side top-1 on long-known individuals exceeds that on newly appearing ones by ≥ 10 pts, with overlapping quality strata.

---

## Part 4 — Roadmap

### 4.1 Now / Next / Later with gates

**NOW (≤ 6 weeks): review-assist foundations.** All code items need proposals and approval first; all runs are user-authorized.

| # | Item | Gate to pass | Kill or redirect criterion |
| --- | --- | --- | --- |
| N1 | Fix ingestion collisions and review-preserving upserts (`reid_demo/ingest.py`, `store.py`) | Regression tests from the pilot-readiness fixtures pass | — (mandatory) |
| N2 | C2 duplicate-merge second check + C1 conservative pruning floor, chosen by leave-one-source-out on calibration | P3-style target at 5% error on calibration; duplicates ≤ 1 / 1,000 in every held-out source | If the saving vs W1 drops below 45%, keep a full scan for new animals and drop pruning |
| N3 | F1 timed review mode; 2 reviewers × 300 calibration bundles | Measured seconds per decision and error rates | If measured saving < 25% wall-clock, reframe the pitch to "error reduction" only |
| N4 | B same-observation local/global diagnostic | Pre-registered outcome above | Refuted → close local features for lynx, except A2 |
| N5 | Research conversation with the Zagreb Veterinary Faculty lynx team (needs the user's go-ahead) | Agreement in principle on a blind D_new and a catalogue-audit pilot | No data access within 3 months → seek KORA/OFB or Czech partners |

**NEXT (2–4 months)**
- X1: blind D_new evaluation on frozen N2 settings (P1–P7).
- X2: raw-frame pipeline: detection, lynx confirmation, segmentation without supplied masks.
- X3: E1 verified night labels + stratified report.
- X4: catalogue ghost-audit pilot on the partner's existing catalogue.
- X5: A1 (v4) and A2 (flank patches) on calibration.
- X6: H temporal/novelty test using D_new.
- **Gate:** P1, P2, P3, P5 and P7 met on D_new.
- **Kill:** if P1 fails because count error > 2× the target with all safeguards, stop the automation pitch; offer catalogue tooling and the audit only.

**LATER (≥ 6 months)**
- D2 capture-history export and SCR sensitivity with the partner's ecologist.
- C4 paired-camera flank linking (partner protocol).
- D3 latent-identity SCR toward framing (b).
- A6 training on expanded data.
- **Gate for (b):** same-side top-1 ≥ 80% on D_new **or** ≥ 80% of individuals with two-flank links, **and** interval coverage ≥ 90% at width ≤ ±10% on two sites.

### 4.2 Two paths

**Review-assist pitch (shortest honest path, about 3–5 months):**
N1 → N2 → N3 → N5 → X4 (catalogue audit, can start before the D_new labels exist) → X1 + X2 + X3 → pitch.

**Automated-count pitch (≥ 9–12 months, uncertain):**
The review-assist path → A1/A2/A6 → C4 (paired cameras) → D3 → multi-site validation. It may never close without a camera-protocol change [I].

### 4.3 What to ask a park and its partner for in a pilot

- **Data:** one or two full seasons of **raw** camera-trap output (all frames, including empties), with deployment metadata: station ID, coordinates, camera model, active dates, paired-camera layout. Photos after 15 Oct 2024 are reserved as blind test data.
- **Labelled subset:** the expert lynx catalogue for those seasons, kept by the partner. Include "unidentifiable" flags, not just confident IDs.
- **Expert time:** about 2 × 4 hours for the timed review trial; 2–4 hours to adjudicate the system's flagged suspected ghosts or merges in the catalogue audit.
- **Protocol (optional, high value):** at 10–20 marking sites, paired opposite cameras for one season (the KORA design). EXIF time and IR mode left on.
- **Agreements:** data stays on partner premises or is processed locally; no upload to third parties; co-authorship terms.

### 4.4 Demo plan

**Show:**
1. A raw-frame-to-catalogue walk-through on CzechLynx calibration photos (not partner data).
2. The ranked review screen: one-click confirmations; the system's automatic decisions with their evidence; a new-animal check.
3. The audit report: split and merge counts, unresolved photos, the audit interval.
4. A "suspected duplicate" panel: a ghost-audit example.
5. The capture-history export.

**Numbers that are honest to claim** (always with the split label):
- "On 55 lynx the system had never seen (CzechLynx *selection* split, one pre-registered run, simulated reviewer), it decided 51% of photo bundles automatically with 0.8% of photos misassigned, and needed 59% fewer expert decisions than full manual matching."
- "In that same run, the final count was **6 too high (61 for 55)**. That is exactly the kind of error we will measure and fix with you."
- Recognition, *selection*: 69.9% top-1; 56.1% per-animal average.

**Never claim:** time saved in hours (not measured with people); any automated count; night or IR performance; accuracy on Croatian animals; "trained without seeing these animals" (upstream overlap unresolved).

**Caveat paragraph for a park audience:**
> This is a research prototype tested only on Czech lynx photographs prepared by experts, with computer-simulated reviewers. On animals it had never seen, it automated about half of the matching decisions with under 1% of photos misassigned, but its final count was six animals too high out of 55. We have not yet measured it with real reviewers, on Croatian cameras, on raw unsorted camera output, or separately for night and infrared images. The underlying model may have been trained on some of the same Czech animals. We therefore propose a joint, blind evaluation on your photographs. Your experts keep the identities and confirm every decision; we report every error, and you decide whether it helps.

---

## Proposals requiring the user's approval (no code was changed)

1. **N1** ingestion/store integrity fixes: edits to `reid_demo/ingest.py` and `reid_demo/store.py` + regression tests.
2. **N2** pruning floor + duplicate-merge second check: new rule in `reid_demo/benchmark_nonmatch_pruning.py` or a new module, with a leave-one-source-out calibration simulation.
3. **N3** timed matching review mode in `../czechlynx-review-app/`: separate app scope.
4. **N4** same-observation local/global diagnostic: new `reid_demo/benchmark_local_diagnostic.py` + CLI + tests.
5. **D2** capture-history export: new `reid_demo/capture_history.py`.
6. **N5** outreach: not authorized by this analysis; the user's decision.

---

## Sources

Repository receipts (paths relative to the repo root) are cited inline. Web sources, all accessed 19 October 2026:

1. Blašković S. et al. (2025). *Spotted correctly? Human expertise remains essential in AI-assisted identification of Eurasian lynx.* J. Zool. 328(3):223–233. Peer-reviewed; abstract via search only (publisher 403). https://zslpublications.onlinelibrary.wiley.com/doi/abs/10.1111/jzo.70095 ; code https://github.com/iratop-droid/Whiskerbook-ArgusWild
2. Bini F., Topličanec I., Sindičić M., Gomerčić T. *Integrating systematic and opportunistic camera trap data to monitor Eurasian lynx demography.* J. Zool., doi:10.1111/jzo.70144. Peer-reviewed; abstract via aggregator. https://www.citedrive.com/en/discovery/integrating-systematic-and-opportunistic-camera-trap-data-to-monitor-eurasian-lynx-demography/
3. Fležar U. et al. (2023). Biodiversity and Conservation 32:3199–3216. Peer-reviewed. https://link.springer.com/doi/10.1007/s10531-023-02646-3
4. LIFE Lynx, density/SCR news. https://www.lifelynx.eu/?p=25946 ; management plan https://www.lifelynx.eu/?p=28644 ; national monitoring programme https://www.lifelynx.eu/?p=27173 ; After-LIFE plan https://www.lifelynx.eu/with-the-after-life-conservation-plan-we-are-safeguarding-the-future-of-lynxes-in-dinaric-mountains-and-se-alps/
5. Dryad dataset, Slovenian lynx monitoring 2018–2023. https://datadryad.org/dataset/doi:10.5061/dryad.cnp5hqcgm
6. KORA deterministic camera-trap monitoring. https://www.kora.ch/en/projects/large-carnivore-monitoring/deterministic-camera-trap-monitoring ; https://www.kora.ch/en/species/lynx/camera-trap-monitoring
7. OFB, lynx photo-identification method. https://ofb.gouv.fr/methode-de-photo-identification-du-lynx
8. Johansson Ö. et al. (2020). Sci. Rep. Peer-reviewed. https://pmc.ncbi.nlm.nih.gov/articles/PMC7156508/
9. Verschueren S. et al. (2022). Mamm. Res. 68:121–125. Peer-reviewed. https://cheetah.org/resource-library/reducing-identification-errors-of-african-carnivores-from-photographs-through-computer-assisted-workflow/
10. Gooliaff T. J., Hodges K. E. (2018). Ecol. Evol. Measuring agreement among experts… Peer-reviewed. https://pmc.ncbi.nlm.nih.gov/articles/PMC6262731/
11. Augustine B. et al. (2018). Spatial capture–recapture with partial identity. Ann. Appl. Stat. Peer-reviewed. https://projecteuclid.org/euclid.aoas/1520564465 ; categorical SPIM https://www.usgs.gov/publications/spatial-capture-recapture-categorically-marked-populations-application-genetic-capture ; seminar abstract https://www.gla.ac.uk/schools/mathematicsstatistics/events/details/?id=10852
12. Picek L. et al. CzechLynx. arXiv v2 (Scientific Data 2026 per the repo notes). https://arxiv.org/html/2506.04931v2
13. Picek L., Neumann L., Matas J. (2024). Animal identification with independent foreground and background modelling. Preprint. https://arxiv.org/abs/2408.12930
14. Čermák V. et al. WildFusion. arXiv 2024; LNCS 2025. https://arxiv.org/abs/2408.12934
15. Colling G. (2026). Anchored graph clustering… AnimalCLEF 2026. CEUR-WS working notes. https://ceur-ws.org/Vol-4283/paper332.pdf
16. Elkerdany S. et al. Calibrated similarity and graph clustering for open-set animal re-ID (AnimalCLEF 2026). Working notes. https://huggingface.co/papers/2608.02469
17. Kelebek H. S. et al. (2026). Spotted: location-informed re-identification of hyenas and leopards. Preprint. https://arxiv.org/abs/2607.00804
18. Angelopoulos A. et al. Conformal risk control. ICLR 2024. https://arxiv.org/abs/2208.02814
19. Guo C., Miguel A., Maciejewski A. Human-in-the-loop leopard identification. Dissertation / under review. https://www.engr.colostate.edu/~aam/pdf/journals/120.pdf
20. Liu C. et al. (2019). Part-pose guided Amur tiger re-ID. ICCVW. https://openaccess.thecvf.com/content_ICCVW_2019/html/CVWC/Liu_Part-Pose_Guided_Amur_Tiger_Re-Identification_ICCVW_2019_paper.html ; PAW-ViT, WACV 2026. https://www.lirmm.fr/~chaumont/publications/WACV2026_DIAS-RIBERO-NETO_CHAUMONT_SUBSOL_DE-GARINE-WICHATITSKY_GUIS_Animal-Part-Based-ReId.pdf
21. Wild Me: Wildbook for Lynx https://www.wildme.org/wildbook-for-lynx.html ; MiewID v4 announcement (30 Jan 2026) https://community.wildme.org/t/miewid-v4-announcement/5406 ; lynx MiewID degradation thread https://community.wildme.org/t/miewid-is-feeling-tired-a-lot/4709 ; French team thread https://community.wildme.org/t/de-prioritize-hotspotter-for-eurasian-lynx-in-whiskerbook/2626 ; msv3 card https://huggingface.co/conservationxlabs/miewid-msv3 ; licence discussion https://huggingface.co/james-burgess/miewid/discussions/1
22. MZOZT decision on priority-activity funding, March 2026. https://mzozt.gov.hr/UserDocsImages/UPRAVA%20ZA%20ZA%C5%A0TITU%20PRIRODE/ODLUKE_ZASTICENA%20PODRUCJA/Odluka_prioritetne%20aktivnosti_ozujak%202026.pdf ; Risnjak cohesion project via search summary (Upravno vijeće, Nov 2025), not opened directly.
23. NP Plitvice lynx monitoring. https://np-plitvicka-jezera.hr/ris/ (cited via the 14 Oct Risnjak assessment)
24. DS@GT AnimalCLEF 2025 (DINOv2 vs MegaDescriptor fine-tuning). Preprint. https://arxiv.org/abs/2509.12353
