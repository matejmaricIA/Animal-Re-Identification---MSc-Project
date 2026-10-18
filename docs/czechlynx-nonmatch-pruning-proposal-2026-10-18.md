# Proposal: calibrated non-match pruning with review safeguards

Date: 18 October 2026. Work item (a) in the [decision log](lynx-demo-decision-log.md) "Active work" table. Status: **approved by the user 18 October 2026 as written, then implemented and run.** Best exact count: 2,211 decisions (0.77 h per 1,000 photos, −58% vs W1). The automation and hours targets were not met, so there is no candidate operating point. See [results](research/czechlynx-nonmatch-pruning-results-2026-10-18.md).

## Why

The [gated-matching results](research/czechlynx-gated-matching-results-2026-10-18.md) showed that confident matches can be automated accurately, but doing so saves only 14–20% of expert time. Most of the remaining cost comes from two sources:

1. **Confirming new animals.** Each confirmation is a full catalogue scan: 0 + 1 + … + 54 = **1,485 of W1's 5,269 comparisons**.
2. **Hard matches.** Using the W1 numbers ([review-effort results](research/czechlynx-review-effort-results-2026-10-18.md)), 1,750 known-animal parts cost 3,784 comparisons. About 1,451 of them are rank-1 one-click confirmations (82.9%), so roughly **299 hard parts cost about 2,333 comparisons** (mean rank ≈ 7.8).

The user's requirement (decision log, 18 October) is that the bulk of the work is automated and expert review is minimal. This proposal attacks both cost sources:
- A calibrated **"not this animal" model** lets the system stop a scan once every remaining candidate is very unlikely. Declaring a new animal then means checking a short list.
- The same model, with space-time history, can **re-rank** candidates so the true animal is shown earlier in hard matches.

It also adds safeguards against the risks found so far: reviewer error, cold-start collapse, and duplicates caused by pruning.

## Ceiling arithmetic before running (indicative, from saved W1 numbers; not a result)

The declared hours target is ≤ 0.5 expert-h per 1,000 photos, or about 11,430 s for 6,352 photos. Assume the best plausible case: 70% of bundles automated (mostly easy rank-1 ones), new-animal checks pruned to about 5 comparisons each including double confirmation, 100 audit glances and about 100 duplicate-check questions.

| Cost setting | Everything except hard matches | Hard matches at today's ranks | Total | h / 1,000 photos |
| --- | ---: | ---: | ---: | ---: |
| Placeholder (8 s per comparison) | ≈ 5,300 s | ≈ 18,700 s | ≈ 24,000 s | ≈ 1.05 |
| Fast-click (see below) | ≈ 2,400 s | ≈ 10,800 s | ≈ 13,200 s | ≈ 0.58 |

Implications:
- **At placeholder costs, the 0.5 h target needs hard-match comparisons cut by about two thirds** (2,333 → ≈ 770). Pruning alone cannot do that, because a scan stops at the first "yes". Only better ranking can, and a large enough gain may need the better embedding from item (b).
- **At fast-click costs, a re-ranking gain of about 20% would be enough.**
- **The ≥ 70% automation target is probably out of reach** for the unchanged W4 confidence model: it reached 61% at 1.57% misassigned, and 68% only at 3.4%. Pruning does not raise automation, because new animals are never created automatically.

The targets below are carried over unchanged, as the handoff asked. The user may revise them before approving (see "Questions for the user").

## Design (calibration only, CPU, frozen FP32 v2 vectors)

The setup is the same as W1/W4: 1,791 camera-day bundles in date order, the reviewer splits multi-animal bundles into parts, and automatic mistakes stay in the catalogue and affect later matching. Inputs come from the gated CLI's `prepare()`, so context, bundles, identity folds and W4 replay records are identical to the predecessor run.

### 1. P(match) model ("not this animal")

For a part (the photos of one animal in a bundle) against **every alive catalogue animal e**, the model uses label-free features:

| Group | Features |
| --- | --- |
| Appearance | s_e (best image-pair cosine); log rank of e; gap to top-1 (s1 − s_e); gap to the next-ranked animal; mean of the top-3 image-pair scores; reciprocal rank (where the part's best photo ranks among e's best-matching member's neighbours in the current catalogue); log size of e |
| Facing | best same-facing score; share of e's photos with the part's predicted facing |
| Identifiability | mean and minimum risk of the part's photos; log part size; log catalogue size |
| Space-time history | e seen at this camera / location cell before; km to e's nearest sighting (+ missing flag); days since e was last seen; **same-day conflict**: km between the part and any same-day sighting of e (+ missing flag), since one lynx cannot be at two distant traps on the same day |

**Training data.** Records come from a perfect-review W1 replay: every part against every alive entry at that moment, labelled match or no match. That gives an estimated 60–80k records.

**Model.** Standardized logistic regression, as for the W4 confidence model.

**Identity cross-fitting** uses the existing 5 identity folds and is stricter than in W4. The model that scores a part of animal A (fold f) is trained only on records where **neither the part's animal nor the candidate's founding animal** is in fold f. This matches the "endpoints avoid both folds" rule of the edge-grouping model.

**Reported:** AUC, Brier score, a 10-bin reliability table and, for each ε:
- the **miss rate**: true-match records with P < ε, which become potential duplicates;
- the share of non-match records pruned;
- the short-list length for truly new parts.

Variants: appearance + space-time (main) and appearance only (ablation).

### 2. Pruned expert path

The expert compares candidates in one of two orders:
- **cosine order** (as W1);
- **P(match) order**, which re-ranks by the calibrated probability.

The scan stops at the first "yes". It also stops when **no not-yet-shown candidate has P(match) ≥ ε**, after at least one candidate has been shown. When the scan stops without a "yes", the expert declares a new animal (subject to double confirmation below).

- ε = 0 is the full scan and the reference.
- ε grid: **0 (reference), 0.001, 0.003, 0.01, 0.03**.

### 3. Combination with W4 auto-accept

Auto-accept reuses the existing W4 confidence model unchanged (`fit_fold_models` and the W4 feature set). Whole bundles are auto-accepted to the top-1 animal when confidence ≥ τ.
- τ grid: **none, 0.98, 0.99, 0.995**.
- New animals are never created automatically.

### 4. Safeguards

| Safeguard | Rule | Default in the main grid |
| --- | --- | --- |
| Cold-start guard | No auto-accept and no pruning (full scan) until the catalogue has ≥ N animals **and** ≥ B expert-decided bundles | N = 10, B = 100 |
| Double confirmation of new animals | Before a new animal is created, a second, independent simulated reviewer re-checks the top-c candidates by P(match) | c = 3 |
| Disagreement check (added beyond the handoff list) | If the expert says "yes" to a candidate the model rates P(match) < δ, a second reviewer must agree. This targets false "yes" answers, the main source of the 8–11% misassignment at 3% reviewer error | δ = 0.05 |
| Periodic duplicate check | Every 100 bundles, and once at the end: each animal created since the last check is compared, as a pair question, with its top-r other animals by entry-vs-entry P(match), skipping pairs already asked. A "yes" merges it into the older animal, and merged animals are never shown again | r = 3 |
| Second-reviewer audit of automation | An independent reviewer glances at a random sample of automatic decisions (reuses `audit` and `wilson_interval`). It estimates the bundle error rate with a 95% Wilson interval and does not correct anything | n = 100 |
| Second-reviewer count audit (added) | An independent reviewer full-scans a random sample of expert new-animal declarations against the final catalogue. This estimates the duplicate share with a Wilson interval, giving a count interval, and is how pruning-induced duplicates would be detected in practice. Estimate only. Its cost is reported separately, because full scans are expensive (≈ 60 comparisons each) | n ∈ {0, 10, 20} (post-processing; no effect on the catalogue) |

Reviewer error is **0% and 3%**. Every reviewer role (expert, second confirmer, auditors) has its own seeded generator, so their errors are independent.

### 5. Grids

- **References** (safeguards and guard off, cosine order, ε = 0). These are integrity checks; if either does not reproduce, the run is marked failed.
  - τ = none must reproduce **W1 exactly** (5,283 decisions, 55/55).
  - τ ∈ {0.98, 0.99, 0.995} with a 100-glance audit must reproduce the **post-hoc full-scan W4** perfect-reviewer rows exactly: 4,279 / 4,449 / 4,578 decisions, count error 0.
- **Main grid G1** (safeguards on at the defaults, space-time features): order {cosine, P(match)} × ε 5 × τ 4 × reviewer error {0, 3%} = **80 simulations**.
- **Safeguard ablations G2.** Base settings: τ = 0.99, ε ∈ {0.003, 0.01}, both orders, both reviewer errors (8 combinations). Variants, one change at a time:
  - (i) guard off;
  - (ii) guard N = 20, B = 200;
  - (iii) no double confirmation;
  - (iv) no disagreement check;
  - (v) no duplicate checks;
  - (vi) all safeguards off;
  - (vii) appearance-only models with the guard on;
  - (viii) appearance-only models with the guard off, the condition that previously collapsed to one animal.

  8 variants × 8 = **64 simulations**.
- Total: about 150 sequential simulations. Audits are computed afterwards on copies of the ledger, as in the predecessor.

### 6. Cost model and raw counts

A new decision ledger records separate kinds:
- rank-1 comparison;
- further comparison;
- second-reviewer comparison;
- disagreement check;
- duplicate pair question;
- audit glance;
- count-audit comparison;
- extra animal in bundle.

Two cost settings are reported side by side, with all raw counts:

| Decision | Placeholder (as before) | Fast-click |
| --- | ---: | ---: |
| First candidate shown | 8 s | 2 s |
| Each further candidate | 8 s | 5 s |
| Second-reviewer / disagreement comparison | 8 s | 5 s |
| Pair question (duplicate check) | 10 s | 6 s |
| Audit glance | 5 s | 2 s |
| Extra animal in bundle | 5 s | 3 s |

A per-placement log (part, order, rank of the eventual "yes", candidates shown, outcome) is saved for G1, so any other cost model (for example, pages of 5 candidates per decision) can be applied later without rerunning.

## Measurements

For every simulation:
- automation share;
- decisions by kind;
- hours per 1,000 photos under both cost settings, with and without the count audit;
- saving vs W1 (5,283 decisions, 1.85 h) and vs the post-hoc full-scan W4 at the same τ;
- misassigned photos;
- duplicate and merged animals, with **pruning-induced duplicates** identified separately: the true animal was alive, but its P(match) was < ε and it was never shown;
- count error, pairwise F1 and exact recoveries;
- new-animal comparisons and hard-match comparisons;
- automation-audit and count-audit estimates with 95% intervals vs the truth, and coverage.

Plots:
- duplicates added vs time saved across ε;
- automation vs misassignment across τ.

## Declared targets and rules (fixed before running)

Carried over from the [minimum-review proposal](czechlynx-minimum-review-proposal-2026-10-18.md):

| Target | Rule |
| --- | --- |
| Mostly automatic | ≥ 70% of bundles decided without the expert |
| Reliable grouping | ≤ 1% of photos misassigned |
| Reliable count | \|count error\| ≤ 1 |
| Large time saving | ≤ 0.5 expert-h per 1,000 photos at **placeholder** costs (fast-click reported alongside) |
| Honest self-assessment | The automation audit's 95% interval contains the true bundle error |

A **candidate operating point** must meet all five in G1 with a perfect reviewer at placeholder costs. Its 3%-reviewer-error row is reported next to it. Hours exclude the optional count audit, for comparability with the predecessor; hours including it are shown in the same table.

Additional rules for this experiment:
- **D1 pruning works:** some ε > 0 keeps \|count error\| ≤ 1 (perfect reviewer, safeguards on) while cutting new-animal comparisons by ≥ 75% compared with ε = 0 at the same τ and order.
- **D2 re-ranking helps hard matches:** at ε = 0 and τ = none, P(match) order cuts comparisons on known parts that were not rank 1 by ≥ 25% compared with cosine order.
- **D3 safeguards earn their cost:** at 3% reviewer error, all safeguards on vs all off (G2 vi) at least halves the misassigned photos **or** brings \|count error\| to ≤ 1, at ≤ 25% extra expert time.
- **D4 the cold-start guard prevents collapse:** appearance-only with the guard keeps count error ≥ −5, while the same setting without the guard is reported as is.

Anything computed outside these grids is labelled post-hoc.

## Proposed code changes (new files only; no edits to existing source)

1. **`reid_demo/benchmark_nonmatch_pruning.py`** (~450 lines)
   - Grids and cost settings.
   - A `Ledger` with the decision kinds above and costing under both settings. The existing `Effort` hard-wires one cost table, so it is not modified.
   - A catalogue wrapper that supports merging and retires merged animals from ranking. It builds on `Catalogue` and does not modify it.
   - `candidate_features()`: per-animal features, label-free.
   - `replay_pair_records()`: the perfect-review replay.
   - `fit_pair_models()` with both-sides identity exclusion, and `cross_fitted_pmatch()`.
   - `simulate_pruned()`: the sequential simulation with the guard, auto-accept, pruned scan, double confirmation, disagreement check and periodic duplicate checks.
   - `count_audit()`, declared rules, plots and the report.
   - Reuses `Catalogue`, `Reviewer`, `bundle_keys`, `fast_quality` and `plain` (from `benchmark_review_simulation`), and `Context`, `match_features`, `replay_records`, `fit_fold_models`, `audit`, `wilson_interval`, `misassigned_photos` and `majority` (from `benchmark_gated_matching`).
2. **`tools/simulate_czechlynx_nonmatch_pruning.py`** (~200 lines). CLI with the usual contracts:
   - calibration only (`load_calibration` refuses selection);
   - new or empty output directory, with protected inputs;
   - raw-metadata, vector-cache and calibration-manifest SHA-256 checked before and after;
   - the cache path checked against its contract;
   - a `running` → `complete`/`failed` manifest, with failed runs preserved;
   - no neural imports at import time.

   It reuses the gated CLI's `prepare()` so the inputs are identical, and checks the two references before the grids.
3. **`tests/test_benchmark_nonmatch_pruning.py`** (synthetic; about 14 tests):
   - Candidate features and order are unchanged when founders or identity codes are relabelled (identity-free).
   - A pair model never scores an animal it was trained on, whether as part or as candidate.
   - Auto-accepted mistakes propagate to later features and decisions.
   - No new animal is created except by an expert declaration.
   - A true match with P < ε becomes a pruning-induced duplicate; ε = 0 avoids it; a duplicate check with r ≥ 1 merges it.
   - With ε = 0 and safeguards off, the simulation reproduces `simulate_catalogue` (W1) decision counts on a synthetic world.
   - The cold-start guard blocks auto-accept and pruning until N and B are reached.
   - Double confirmation and the disagreement check recover scripted reviewer errors.
   - Merged animals are never shown again.
   - Ledger costing under both settings.
   - Count-audit arithmetic and intervals.
   - The CLI refuses selection and non-empty output.
   - A failure manifest is written on error.
   - The cache hash is unchanged.

Untouched: existing source and tests, earlier outputs, data, models, caches, and the reserved selection/test pools. Output: `evaluations/czechlynx/nonmatch_pruning_v2_fp32/` (confirmed absent). Results go to `docs/research/czechlynx-nonmatch-pruning-results-<date>.md`, with dated decision-log entries at each milestone.

## Limits

- **Simulated reviewers;** both cost settings are placeholders, not measured times.
- **The P(match) and confidence models are cross-fitted on calibration** (development only). A deployable model needs fitting-pool training on embeddings not adapted on those same photos.
- **Training/simulation shift:** the models are trained on a perfect-review replay but applied to an imperfect, evolving catalogue.
- **Data:** day-level dates and coordinates on about 88% of photos. Night/IR and flank effects are inherited from the embedding.
- **No count configuration, selection evaluation or outreach follows from this.**

## Runtime and who runs what

- CPU only; about 150 simulations, estimated **20–60 min** (per-animal features for every part are heavier than W4's top-candidate features).
- No GPU and no model loading.
- If the user approves, the assistant implements the code, then shows the tests and the run command. Per the usual practice, the user may authorize the assistant to run the CPU tests and the run.

## Questions for the user

1. **Approve the scope as written?** That covers the ε, τ and safeguard grids, the two added safeguards (disagreement check, count audit), the P(match) re-ranking arm and the new-files-only code plan.
2. **Keep the carried-over targets?** The ceiling arithmetic suggests that ≥ 70% automation and ≤ 0.5 h at placeholder costs are unlikely without a better embedding. One option is to also declare the hours target at fast-click costs. Otherwise they stay as they are.
3. **May the assistant run the CPU tests and the calibration run after implementing?**
