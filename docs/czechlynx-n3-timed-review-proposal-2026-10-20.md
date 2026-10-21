# N3 proposal: timed reviewer trial of ranked matching (W1) vs the assisted N2 workflow, on calibration

Date: 20 October 2026. Work item N3 in the [decision log](lynx-demo-decision-log.md) ("Active work (live)"). Source: Now item N3 / F1 of the [pitch-readiness analysis](research/pitch-readiness-analysis-2026-10-19.md#41-now--next--later-with-gates) and the [handoff prompt](czechlynx-n3-timed-review-handoff-2026-10-19.md).

**Status: approved by the user on 20 Oct 2026 as written. Implemented on 20 Oct; user-run tests passed (repo `pytest tests/test_benchmark_timed_review.py tests/test_benchmark_pruning_safeguards.py tests/test_benchmark_nonmatch_pruning.py -q`: 54 passed, 57 upstream SciPy L-BFGS-B deprecation warnings, 5.02 s; app `unittest`: 12 tests OK, 1.89 s).** The user runs the tests and the reviewer sessions. **Task set built 20 Oct (assistant-run, authorized):** `evaluations/czechlynx/timed_review_tasks_v1_r2/`, fingerprint `580d922d…`. It was built from a `7ade3225` export because N1b changed a calibration-bound file. The leak scan was narrowed to image metadata after a false positive; that change was user-approved. See the decision log and the provenance receipt.

**Additions beyond this text (disclosed in the decision log):**
- a confirmation bar after the final answer (Enter continues, Backspace undoes); its time is not decision time;
- answers within 0.2 s of a screen appearing are ignored;
- the contact sheet shows practice images only;
- tests: 18 repo and 12 app, rather than about 14 and 16.

## Why

Every expert-time figure so far multiplies simulated decision counts by invented costs:
- placeholder: 8 s per comparison, 10 s per pair question, 5 s per glance;
- "fast-click": 2–6 s.

On *calibration* (6,352 photos, 55 lynx, 1,791 camera-day bundles, F1 vectors), the N2 candidate needs 2,350 decisions and full ranked matching (W1) needs 5,325, a 55.9% cut. Sources: `evaluations/czechlynx/pruning_safeguards_f1_v1/simulations.csv`, row `N2/region/eps0.001/min1/cos0/merge1/err0.0/rep0/tau0.99`, and `summary.json` → `reference_check.w1_live_decisions`. The N2 candidate is P(match) order, ε 0.001, W4 auto-accept τ 0.99, the (a) safeguards and the duplicate-merge second check, with region folds.

**The decision cut does not say how much time is saved.** The two workflows spend their decisions on different kinds of work [measured, saved rows]:

| Saved perfect-reviewer row | Rank-1 comparisons | Further comparisons | Second comparisons | Disagreement checks | Pair questions | Audit glances | Extra animals | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W1, cosine full scan (`nonmatch_pruning_facing_classes_v1`, `reference/full_scan_no_safeguards/err0.0/cosine/eps0.0/tauNone`) | 1,804 | 3,507 | 0 | 0 | 0 | 0 | 14 | **5,325** |
| N2 candidate (region folds) | 1,000 | 874 | 159 | 44 | 163 | 100 | 10 | **2,350** |

- 1,485 of W1's comparisons (28%) are **new-animal full scans**: every catalogue animal is shown before a new one is declared. Most of these are obvious non-matches and may take a second or two each.
- N2 removes most easy work: 44.7% of bundles are auto-accepted, and pruning cuts the obvious non-matches. **What is left is mostly hard comparisons**, which may take much longer each.

So the measured time saving could plausibly be anywhere between about 25% and 56% [inferred from the saved counts]:
- **Equal cost per decision:** 55.9%, by definition.
- **"Hard work left" scenario:** every remaining N2 comparison takes 8 s, pair questions 12 s, audit glances 5 s. In W1, rank-1 comparisons on known animals take 8 s, later known-animal comparisons 4 s and new-animal scan steps 2 s. The saving is then about **24%** (19,122 s vs 25,344 s).
  - The W1 split is known-animal rank-1 1,750, later known-animal 2,076, new-animal scans 1,485 (`known_part_comparisons`, `new_animal_comparisons` in the saved rows).

Pitch criterion P2 needs ≥ 40% less measured wall-clock time. P3 needs a **measured** reviewer error rate: the simulations assumed 0, 3 or 5%, and Croatian experts measured 92.7% manual identification accuracy (Blašković et al. 2025, a different denominator).

## Decisions

### 1. Where the UI lives: a new sibling app, not a mode in the existing one

**Proposal:** a new directory `/home/mm-workstation/re-id/czechlynx-matching-review-app/`, outside this repository, with its own empty `data/` and `exports/`. It reuses the existing app's server patterns, copied, not imported. **The existing input-audit app is not edited, restarted or read from by the new app.**

Why not a mode in `/home/mm-workstation/re-id/czechlynx-review-app`:
- Its database is bound to the input audit. `Store` refuses a different audit, and the README recommends one copy per audit.
- The completed audit receipt binds the export CSV's path and bytes (`exports/reviews_complete_20261016T130044652360Z_bccd027d.csv`, SHA-256 `4e528514…44e`).
- Its server has been running for 5 days (PID 980550, port 8766).
- The matching UI has a different data model (task sequences, timing events, no completion revisions). A mode would add branches to code whose 13 tests describe a different job.

**Protection baseline, recorded 20 Oct 2026 (read-only):**
- `data/reviews.sqlite3`: SHA-256 `0dd295d6c39417ba151cc5779e8317b77369ef96ed4dd0866950ec41eea83abe`.
- Completed export CSV: `4e5285142551e66a7f1b93301032bcdf2e1b0e54da05515eac9636812af2244e` (matches the receipt).
- Its JSON: `d3679ae33431f54ad822f472c1beb7f8d94344d4fae0a6e460aff84994661f61`.
- Combined digest of all app files except `data/server.log`: `8556e455a9eb1a90e5e63ff9de68f18d2aed50de14b1a5be843da6212c62212a`.

These are re-checked and reported at every milestone. The new app refuses to start if its `data/` or `exports/` would resolve inside the old app's directory. It listens on **port 8767** by default.

### 2. Task design: precomputed tasks from the existing simulations

**Source of tasks.** A new builder replays both workflows on calibration with a perfect reviewer, exactly as the saved runs did:
- **W1:** `simulate_catalogue(ranked=True)` from `benchmark_review_simulation.py`, which must reproduce 5,325 decisions (1,804 / 3,507 / 14).
- **N2 candidate:** `SafeguardSimulation` with region folds and the candidate settings, which must reproduce the saved row field for field (2,350 decisions, all kinds, automation, count, misassignment).

A **traced subclass** in a new file records what each decision showed:
- the bundle and the part (one animal's photos in the camera-day bundle);
- the decision kind;
- the catalogue animal shown, its rank and the list length;
- the reference truth.

The committed modules are not edited; their hashes bind completed runs.

**A task** is one screen sequence for one part, in the order the workflow would show it:
- the query photos on the left;
- one candidate catalogue animal at a time on the right;
- the reviewer answers **Same**, **Different** or **Unsure** (keys S / D / U).

A "Same" ends the task. Different or Unsure advances to the next candidate. After the last candidate, the part is declared a new animal; that declaration is implicit and needs no extra click. This one-candidate-per-screen design maps each click onto exactly one ledger decision.

| Ledger kind | Task type in the app | Reviewer role | Screen banner |
| --- | --- | --- | --- |
| `rank1_comparison` | first candidate of a W1 or N2 placement task | expert | "Find this lynx in the catalogue" |
| `further_comparison` | later candidates of the same task | expert | same |
| `second_comparison` | top-3 P(match) candidates after an N2 "new animal" | second reviewer | "Second check before a new animal is created" |
| `disagreement_check` | expert said Same but P(match) < 0.05 | second reviewer | "Second opinion: same lynx?" |
| `pair_question` | duplicate check between two catalogue animals (both sides show exemplars) | expert | "Are these two catalogue animals the same lynx?" |
| `merge_confirmation` | not sampled: 0 occur at perfect review in the candidate row; **costed as a pair question** (declared, as the existing cost settings do) | — | — |
| `audit_glance` | an auto-accepted bundle shown with its assigned animal | auditor | "The system assigned these photos to this lynx. Correct?" |
| `extra_animal` | not timed: 14 (W1) and 10 (N2) events, 0.3–0.4% of decisions; **5 s placeholder in both arms**, sensitivity at 25 s | — | — |

**Strata.** Stratified sampling makes rare but expensive work measurable. The builder counts decisions per stratum and kind in the full replay (N<sub>s,k</sub>). These counts sum exactly to the saved per-kind totals and later weight the measured times.

W1 populations below are approximate: v2's rank shares (82.9% at rank 1, 95.9% in the top 10) applied to F1's 1,750 known parts. The builder computes the exact F1 figures.

| Arm | Stratum | Population | Tasks per reviewer |
| --- | --- | --- | ---: |
| W1 | K1: known animal, true match at rank 1 | ≈ 1,450 parts | 60 |
| W1 | K2: known, true at rank 2–10 | ≈ 230 parts | 58 |
| W1 | K3: known, true at rank > 10 | ≈ 70 parts | 20 |
| W1 | N: new animal (full catalogue scan) | 55 parts | 12, spread over catalogue sizes |
| N2 | K1: expert part, true at P(match) rank 1 | from trace | 35 |
| N2 | K2: expert part, true at rank ≥ 2 | from trace | 35 |
| N2 | N: new animal after the pruned shortlist | 55 parts | 12 |
| N2 | S: second comparisons (up to 3 each) | 55 sets | 15 (3 injected, see below) |
| N2 | D: disagreement checks | 44 | 8 |
| N2 | P: duplicate-check pair questions | 163 | 25 (5 injected) |
| N2 | A: audit glances of auto-accepted bundles | 100 | 20 (4 injected) |
| | **Total** | | **300 (150 per arm)** |

Expected load per reviewer: about 1,000 W1 decisions (dominated by the 32 K3/N scan tasks) and about 350 N2 decisions [inferred from the stratum sizes].

**Injected tasks.** At perfect review, three task types have only one possible answer:
- pair questions and second comparisons are all truly different;
- audit glances are nearly all correct.

A reviewer who learns that pattern would make the error rates meaningless. About 20% of S, P and A tasks are therefore built to have the other answer:
- **S:** a known part whose true match is among the three shown;
- **P:** two halves of one animal's catalogue photos, split by encounter, posing as a duplicate;
- **A:** an auto-accepted bundle shown with its runner-up animal.

Injected tasks count for error rates only. They get **zero weight in the time estimate**, so time weights follow natural base rates. Declared deviation: injection changes the base rate the reviewer experiences in these three task types.

**Independence rules (enforced by the builder):**
- Each calibration part appears in at most one task, practice included. W1-N, N2-N and N2-S draw 39 of the 55 new-animal parts.
- Practice tasks use bundles outside the task set.
- **Both reviewers get the same 300 tasks** in independently randomized orders. This allows inter-reviewer agreement and two-person consensus error to be measured (needed for P4), and keeps reviewer differences separate from task differences.

**Catalogue state is precomputed** from the perfect-reviewer replay: each task shows the catalogue as it stood just before that bundle. **Limitation: reviewer mistakes do not propagate.** A wrong "Same" does not corrupt later tasks, and a missed match does not create a duplicate that later tasks see. Propagation is handled by feeding the measured error rates back into the sequential simulations (§6).

**Against a live sequential catalogue** (reviewers build the catalogue bundle by bundle):
- **Cost:** W1 alone is 5,325 decisions. At even 4 s per decision that is about 6 hours per reviewer for one arm, before N2, and a fraction would not be representative of the date-ordered sequence.
- **No clean comparison:** the same reviewer cannot do both arms on the same bundles without remembering the answers. Different bundles per arm would confound arm with bundle difficulty.
- **No stratification:** rare expensive work (new-animal scans, pair questions, second checks) could not be oversampled.
- **Its one advantage, realistic error propagation, is better obtained in simulation** with measured rates, and seeded replicates give its variability.

A short live-sequential pilot could validate the propagation model later; it is not proposed now.

### 3. What reviewers see

- **Image.** Default: the CzechLynx dataset image, which shows the animal with background. Note that CzechLynx ships lynx-centred images, not full camera frames; the annotation box covers a median 56% of the image [measured, prepared metadata]. Experts normally work from originals, and background helps humans (same snow, same camera). A toggle shows the masked crop (the model's input). Click any photo to zoom: Fit / 100% / slider, reused from the audit app. Toggles and zooms are logged.
- **Query:** all photos of the part (one animal in one camera-day bundle). Calibration bundles average 3.5 photos. Up to 6 are shown, with a "show all" expander.
- **Candidate animal:** **4 exemplar photos**, chosen without labels: the members with the highest F1 cosine to any query photo, the same quantity the ranking uses. "Show all" expands to at most 24, and the expander is logged. One rule for both arms. Four is a compromise: one photo often shows the wrong flank; more than four slows every comparison.
- **Labels shown:** date and camera code for every photo (both are available to experts in practice and identical in both arms), and the position ("candidate 3 of 41").
- **Never shown:**
  - identity labels or source paths (paths contain `lynx_NNN`), image IDs or file names;
  - model scores, P(match) or the route that produced the task.
  - Images are served by random per-task-set tokens. The builder scans every served file's bytes for identity strings (`lynx_`) and refuses the task set if any match. **No scores are shown in either arm, to avoid automation bias and keep arms comparable;** a score-showing UI is a later variant.
- **Arm awareness:** list lengths reveal the arm. That is inherent to the workflows and stated as a limitation. Automation bias from knowing a shortlist was model-pruned is part of what N2 is.

### 4. Protocol

- **Reviewers:** at least 2. R1 is the user. R2 is one more person if available (skill level to be recorded; a non-expert R2 is reported as such).
- **Experience questionnaire,** stored with the reviewer code:
  - years identifying individual animals from camera-trap photos; species;
  - lynx or felid experience;
  - prior exposure to CzechLynx photos (the user reviewed 321 in the input audit);
  - knowledge of the system's internals;
  - the device, screen size and pointing device (screen size is also logged automatically).
- **Practice:** 20 tasks (10 per arm, all kinds), with feedback on the correct answer, excluded from all analyses.
- **Main trial:** 12 blocks of 25 tasks. Each block is single-arm and stratum-balanced, in **ABBA order**: R1 W1·N2·N2·W1·W1·N2·N2·W1·W1·N2·N2·W1; R2 the mirror, starting with N2.
- **Sessions:**
  - 3 blocks per session, **4 sessions**;
  - a forced 2-minute paused break between blocks (clock stopped);
  - at most 2 sessions a day, at least 2 h apart.
- **Schedule (the user's time is the scarce resource):**
  - about 15 min for practice and the questionnaire, then 4 sessions of about 40–50 min;
  - **about 3.5 h per reviewer**, at placeholder-like speeds (≈1,350 decisions);
  - suggested pace: one session every other day, done within 1–2 weeks.
  - After session 1, the app shows the projected remaining time.
  - **Pre-declared early stop:** after 8 blocks (two full ABBA cycles, 200 tasks), the user may stop. The result is then labelled "reduced". No other mid-trial change is allowed.
- **Single reviewer:** if only the user takes part, the rule is evaluated on one reviewer and labelled "single reviewer"; agreement and consensus are not reported.
- **Network:** LAN only, as with the audit app. A remote R2 needs the user's decision on an access method; none is proposed.

### 5. Pre-registered decision rule (fixed before any session)

**Timing definitions:**
- **Decision time:** from "screen shown" to the answer, minus paused time. "Shown" means all of that screen's images are decoded; the client uses `performance.now()`.
- **Excluded from decision time:** image load time (recorded separately; the next screen is prefetched).
- **Pauses:** the Pause key, a hidden tab, window blur, or 120 s without input (pointer, wheel, key or zoom). Paused intervals are subtracted, and the decision is flagged.
- **Undo** (the previous answer in the same task) adds both intervals to one decision.
- **Primary cap:** 120 s per decision (winsorized), with sensitivity at 60 s and uncapped.
- **Practice** is excluded.

**Primary estimator (stratified per-kind re-costing):**

T<sub>arm</sub> = Σ<sub>s,k</sub> N<sub>s,k</sub> · t̄<sub>s,k</sub>, and saving S = 1 − T<sub>N2</sub> / T<sub>W1</sub>.

- N<sub>s,k</sub> are the full-replay decision counts per stratum and kind. They sum to the saved totals: W1 5,325, N2 2,350.
- t̄<sub>s,k</sub> is the mean capped decision time for kind k in stratum s, over both reviewers, from natural (non-injected) tasks.
- **The mean, not the median, because total time is a sum.**
- Untimed kinds: extra_animal at 5 s; merge_confirmation at the measured pair-question mean; count-audit comparisons excluded (as in N2's hours).
- 95% CI: stratified bootstrap over tasks within stratum, 2,000 resamples, seed 42, reviewers fixed. S is also reported per reviewer.

**Verdict (one evaluation, after both reviewers finish or after the declared early stop):**

| Outcome | Condition | Consequence |
| --- | --- | --- |
| **SUCCESS** | S ≥ 40% (pooled point estimate) **and** per-kind error rates are reported with 95% CIs for every sampled kind | P2's time component is met on calibration. Add the label "imprecise" if the CI lower bound is < 40%, and "reviewer-dependent" if either reviewer's S is < 40% |
| **Partial** | 25% ≤ S < 40% | P2 time not met. The pitch leads with error reduction and auditability and quotes the measured saving; no time headline |
| **Reframe** | S < 25% | Reframe the pitch to error reduction only (the §4.1 N3 kill criterion) |

**Always reported, whatever the verdict:**
- **Error rates** per kind and arm: false-same (Same | truly different), false-different (Different | truly same) and unsure rate. 95% CIs: Wilson per kind, plus a task-cluster bootstrap for multi-decision kinds.
- **Unsure handling:** an unsure counts neither as correct nor as an error (sensitivity: as an error). The workflow treats it as Different.
- **"Unidentifiable photo" flags** on queries. A one-click flag per task means the photos alone cannot identify the animal; it is recorded alongside the answer.
- **Second-check reliability:** whether second-reviewer decisions (S, D, injected S positives) are more error-prone than first-pass decisions on comparable pairs. Also where R1 and R2 disagree on shared tasks, which one matched the reference.
- Inter-reviewer agreement (Cohen's κ) and two-person consensus error.

**Secondary estimates (reported; they do not change the verdict):**
- the literal handoff formula: pooled per-kind mean × saved per-kind counts;
- a task-level Horvitz–Thompson total;
- medians; the 60 s cap and uncapped;
- a W1-favourable bound: W1 new-animal and K3 scan steps costed at their 25th percentile;
- counts from the re-simulation at measured error rates;
- hours per 1,000 photos per arm.

**Pre-registration mechanics.** The builder writes `pre_registration.json` with the task-set fingerprint, this rule and the estimator settings. With the user's permission, it and this proposal are committed before session 1, as with (c). After the first session, no task, stratum, cap or estimator may change.

**Reference labels** are CzechLynx identities as-is. The input audit found 20/321 photos with no clear animal, so some "errors" may be reference problems. A post-hoc review of reviewer–reference disagreements is optional and labelled post hoc.

### 6. Outputs

1. **App export (unique names, never overwritten):**
   - `exports/matching_decisions_<UTC stamp>_<id>.csv`: one row per decision, with reviewer, task, step, kind, arm, stratum, answer, ms, paused ms, flags, zooms, toggles and undo count; no truth;
   - an adjacent `.json` with all raw events, reviewer profiles, the task-set fingerprint and database revision.
2. **Re-costing (new script):** joins exports with `truth.json` and the saved rows. It computes the primary and secondary estimates, error rates, agreement and the verdict. It then **re-runs the sequential simulations at the measured error rates**:
   - W1 and the N2 candidate, region folds, 5 seeds;
   - a new `MeasuredReviewer` with separate false-same and false-different rates per kind, and Unsure treated as Different;
   - this reports misassignment, count error and decisions at measured reviewer quality, a calibration-only P3 check: ≤ 2% misassigned, |count| ≤ 3.
   - Output: `evaluations/czechlynx/timed_review_recost_v1/` (`summary.json`, `per_kind_times.csv`, `error_rates.csv`, `resimulation.csv`, `index.html`, `manifest.json`).
3. **Results doc** `docs/research/czechlynx-n3-timed-review-results-<date>.md`, plus a dated decision-log entry, proposal status, an AGENTS.md entry and a `docs/czechlynx-benchmark.md` command section.

### 7. Files, size and tests

**New app (outside the repo), `/home/mm-workstation/re-id/czechlynx-matching-review-app/`; standard-library Python plus static files:**

| File | Size | Content |
| --- | --- | --- |
| `server.py` | ~600 lines | See the breakdown below |
| `static/index.html`, `static/app.js`, `static/styles.css` | ~150 / ~700 / ~300 | See the breakdown below |
| `run.sh`, `README.md` | ~5 / ~150 | Reviewer instructions, protocol, schedule, export |
| `tests/test_matching_app.py` | ~16 tests | Synthetic task set in a temp directory, loopback server |

`server.py`:
- **Patterns copied from the audit app:** host allowlist, `X-Review-App` and origin guard on writes, CSP, `no-store`, a path allowlist, hash-verified image bytes on every request, `xb`-mode exports and SQLite `BEGIN IMMEDIATE`.
- **New `TaskSet`:** loads `manifest.json`, `tasks.json` and `images.json`, and verifies fingerprint, status and artifact hashes. It **never opens `truth.json`**.
- **New `Store`** in `data/matching.sqlite3`, bound to the task-set fingerprint, with these tables:
  - `reviewers` (code, profile);
  - `events`: append-only, idempotent on (reviewer, session, seq), stamped with server time;
  - `answers`: final answer per task step, with revision, so undo is possible;
  - `progress`.
- **Routes:**
  - `GET /api/session?reviewer=` returns the reviewer's ordered tasks without truth;
  - `GET /images/<token>/<original|masked>`;
  - `POST /api/reviewers`, `/api/events` (batches of up to 200), `/api/answer` and `/api/export`;
  - `GET /exports/<name>`.

The static front end provides:
- **Screens:** start (reviewer code and questionnaire), practice with feedback, the comparison screen, block breaks and progress;
- **Input:** keys S/D/U, Z zoom, Backspace undo, P pause;
- **Timing:** images decoded before "shown"; the next screen prefetched; pause on visibility change, blur or idle;
- **Events:** batched, retried and resumable after a reload or server restart.

The ~16 app tests cover:
- no truth, identity, path or score in any HTTP payload;
- image hash checks;
- refusal of a changed task set, and of `data/` or `exports/` inside the input-audit app or other protected paths;
- idempotent append-only events, undo and revisions, resume after restart;
- deterministic per-reviewer order and a practice flag;
- export immutability and exact CSV columns;
- the HTTP allowlist and origin guard.

**New repo files (no existing source edited):**

| File | Size | Content |
| --- | --- | --- |
| `reid_demo/benchmark_timed_review.py` | ~500 lines | See the breakdown below |
| `tools/build_czechlynx_timed_review_tasks.py` | ~220 lines | See the breakdown below |
| `tools/recost_czechlynx_timed_review.py` | ~250 lines | See the breakdown below |
| `tests/test_benchmark_timed_review.py` | ~14 tests | See the breakdown below |

`reid_demo/benchmark_timed_review.py` contains:
- **`TracedSafeguardSimulation(ps.SafeguardSimulation)`:** overrides `place` and `duplicate_check` with copies of the committed bodies plus trace records, and accepts per-kind reviewers for re-simulation.
- **`traced_w1()`:** a W1 replay recording rank, catalogue size and truth per comparison.
- **Task construction:** `strata()`, `sample_tasks()` (seeded, part-disjoint, with injected tasks), `exemplars()` (label-free top-4 by cosine), `opaque_tokens()` and `block_orders()` (ABBA).
- **Analysis:** `decision_times()` (events → times with pauses, undo and cap), `stratified_total()`, `bootstrap_saving()`, `error_rates()`, `MeasuredReviewer` and `decision_rule()`.

`tools/build_czechlynx_timed_review_tasks.py`:
- loads calibration exactly as `tools/simulate_czechlynx_pruning_safeguards.py` does: F1 cache, raw metadata, region folds;
- **fails unless it reproduces the saved N2 candidate row field for field and live W1 = 5,325 with the 1,804 / 3,507 / 14 split**;
- refuses non-calibration image IDs;
- samples the tasks and scans the served bytes for identity strings;
- writes `evaluations/czechlynx/timed_review_tasks_v1/`: `tasks.json`, `images.json`, `truth.json`, `strata_counts.csv`, `pre_registration.json`, `contact_sheet.html` (for the user's visual check before the trial) and `manifest.json`;
- runs on CPU only, with no model inference, in about 3–5 minutes [inferred from N2's model-fitting time of about 80 s and about 3 s per simulation].

`tools/recost_czechlynx_timed_review.py`: as described in §6. It verifies every binding: task-set fingerprint, export hashes, saved-row hashes and code digests.

`tests/test_benchmark_timed_review.py` (~14 tests) checks that:
- the traced N2 subclass equals `SafeguardSimulation` (ledger, catalogue, logs) on synthetic data across settings, including the merge check;
- traced W1 equals `simulate_catalogue`;
- strata sum back to the ledger;
- sampling is deterministic and part-disjoint, and payloads hold no identity, paths or truth;
- injected tasks carry correct truth;
- the estimator recovers synthetic known times, and the bootstrap is deterministic;
- the 40% and 25% rule boundaries behave as declared;
- `MeasuredReviewer` hits its rates;
- event-to-time conversion handles pauses, undo and caps.

**Not touched:**
- committed benchmark modules;
- `evaluations/czechlynx/pruning_safeguards_f1_v1/` and all other saved outputs (read only);
- *selection* (spent) and the official *test* (locked);
- the input-audit app and its data;
- the `disk` submodule.

Nothing is committed without the user's request.

## Commands (after approval; the user runs tests and reviewer sessions)

App tests:

```bash
python3 -m unittest discover -s /home/mm-workstation/re-id/czechlynx-matching-review-app/tests -v
```

Repo tests (new suite plus the suites it builds on):

```bash
venv/bin/python -m pytest tests/test_benchmark_timed_review.py tests/test_benchmark_pruning_safeguards.py tests/test_benchmark_nonmatch_pruning.py -q
```

Task build (CPU, a few minutes; the user runs it or authorizes the assistant):

```bash
venv/bin/python tools/build_czechlynx_timed_review_tasks.py --calibration-dir evaluations/czechlynx/facing_adaptation_calibration_classes_v1/clustering --raw-metadata data/czechlynx/raw/CzechLynxDataset-Metadata-Real.csv --reference-dir evaluations/czechlynx/pruning_safeguards_f1_v1 --out-dir evaluations/czechlynx/timed_review_tasks_v1 --threads 4 --seed 42
```

Start the app (port 8767; the audit app keeps 8766):

```bash
bash /home/mm-workstation/re-id/czechlynx-matching-review-app/run.sh --tasks-dir /home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/evaluations/czechlynx/timed_review_tasks_v1 --port 8767
```

Re-costing, after the sessions (one `--export` per final export):

```bash
venv/bin/python tools/recost_czechlynx_timed_review.py --tasks-dir evaluations/czechlynx/timed_review_tasks_v1 --export /home/mm-workstation/re-id/czechlynx-matching-review-app/exports/matching_decisions_STAMP_ID.csv --reference-dir evaluations/czechlynx/pruning_safeguards_f1_v1 --out-dir evaluations/czechlynx/timed_review_recost_v1
```

**Sequence after approval:**
1. Implement, and run static checks (py_compile, Ruff F, `node --check`).
2. The user runs both test commands.
3. Build the task set, after which the user checks the contact sheet.
4. Commit the pre-registration (only with the user's permission).
5. The user runs practice and 4 sessions; a second reviewer is the user's call.
6. Re-cost, then write the results and docs.

## Limits

- **Calibration only; CzechLynx, not Croatian photos.** Reviewers are the user and possibly one more person, not the partner's experts. P2 proper compares against the *partner's current workflow* on partner data, not W1. N3 measures the time component on our own workflows only.
- **Precomputed catalogue:** mistakes do not propagate in the app. Propagation comes from simulation with measured rates, which assumes errors are independent of history.
- **Truth-split queries:** each query is one animal's photos. The truth split of multi-animal bundles is given (14 or 10 events), as in the simulations.
- **Learning:** reviewers learn the 55 animals during the trial. This is realistic for experts and balanced across arms by ABBA. W1 shows more candidates, so W1 tasks give more exposure.
- **One-candidate-per-screen W1:** this may be slower per candidate than a gallery scan. The per-stratum costs measure what this UI actually costs; a gallery W1 is a possible later variant. The W1-favourable sensitivity (W1 deep-scan cost at its 25th percentile) is reported.
- **Small numbers:** with two reviewers, between-reviewer variation is described, not estimated. Pair-question errors rest on about 50 decisions (wide CIs: 0/50 still allows up to about 7%).
- **Reference labels** may contain errors.
