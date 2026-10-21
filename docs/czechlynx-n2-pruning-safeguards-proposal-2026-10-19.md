# N2 proposal: pruning floor and duplicate-merge second check, tested as a new-site transfer on calibration

Date: 19 October 2026. Work item (e) in the [decision log](lynx-demo-decision-log.md). Source: Now item N2 of the [pitch-readiness analysis](research/pitch-readiness-analysis-2026-10-19.md#41-now--next--later-with-gates).

**Status: approved by the user on 19 Oct 2026 as written, including assistant-run CPU tests and the calibration run.**

**Correction (19 Oct, before the run):** R2 originally quoted W1 = 5,283 (≤ 2,641). That is v2's W1. The F1 vectors used here give W1 = 5,325 (saved `nonmatch_pruning_facing_classes_v1` reference check), so the limit is 0.5 × live W1 = 2,662. The intent, a ≥ 50% saving vs W1 on the same vectors, is unchanged. The saved F1 run also used `--threads 4`, so the reference reproduction uses 4 threads.

## Why

The frozen one-shot run on *selection* (19 Oct; [results](research/czechlynx-frozen-selection-results-2026-10-19.md)) failed on two mechanisms.

**1. Pruning duplicates.** A new animal was declared while the true animal was already in the catalogue but scored below ε by P(match).

| | Events | Remaining after duplicate checks | Count error |
| --- | ---: | ---: | ---: |
| *Calibration* (F1, identity-cross-fitted) | 3 | 1 | +1 |
| *Selection* | 10 | 7 | +6 |

Sources: `evaluations/czechlynx/nonmatch_pruning_facing_classes_v1/simulations.csv`, row `G1/defaults/err0.0/pmatch/eps0.001/tau0.99`; `evaluations/czechlynx/frozen_review_selection_v1/summary.json`.

The surplus sat in snpa. The suspected cause is a space-time prior that does not transfer to new animals or regions [hypothesis]. The current scan always shows **at least one** candidate (`PrunedSimulation.place`, `if pruning and shown and suffix[pos] < eps: break`). Nothing protects a true match that the appearance embedding ranks high but P(match) ranks low.

**2. Unchecked duplicate merges.** A single "yes" from the expert merges two catalogue animals (`duplicate_check`). At 3% reviewer error, 4 of 6 merges were wrong on *calibration* and 6 of 10 on *selection*. Most of the S4 failure (3.6% misassigned) comes from this.

**Testing on *calibration* is still informative,** but only if it mimics a new site. Calibration contains two geographically separate regions [measured, metadata + `data/czechlynx/protocols/identity_pools_v1.json`]:

| Region | Animals | Photos |
| --- | ---: | ---: |
| Western Carpathians (`foe_carpaths`) | 17 | 2,795 |
| Bohemian Forest (`snpa` ∪ `foe_bohemia`; the two share 9 animals) | 38 | 3,563 |

Fitting the P(match) model, the W4 confidence model and the facing classifier on one region and simulating on the other is a **leave-one-region-out (LORO)** test. It is the closest available stand-in for "a new park" that does not touch *selection* (spent) or *test* (locked).

## What changes (3 new files; no edits to existing source)

Existing modules are not edited. `benchmark_nonmatch_pruning.py` and `benchmark_frozen_review.py` are bound by hash to completed, committed runs (the frozen fingerprint `f2402829…` lists their code digests), and editing them would break those bindings.

### 1. `reid_demo/benchmark_pruning_safeguards.py` (new, ~200 lines)

```python
VERSION = "calibration_pruning_safeguards_v1"
KINDS = nm.KINDS + ("merge_confirmation",)
COST_SETTINGS = {k: {**v, "merge_confirmation": v["pair_question"]} for k, v in nm.COST_SETTINGS.items()}
NEW_DEFAULTS = {"min_shown": 1, "cosine_floor": 0, "merge_confirm": False}   # == current behaviour

class SafeguardLedger(nm.Ledger):          # accepts the extra kind; same costing

class SafeguardSimulation(nm.PrunedSimulation):
    def place(self, identity, part):
        """As nm.PrunedSimulation.place, with two pruning floors:
        - min_shown m: never stop before m candidates have been shown (P(match) order);
        - cosine_floor k: when the P(match) suffix falls below eps, the scan also shows any
          still-unshown entries among the top-k by APPEARANCE cosine (cosine order), then stops.
          This shortlist is independent of space-time features, so it guards against a
          mis-transferred location prior. Shown entries are costed as further_comparison."""

    def duplicate_check(self):
        """As nm.PrunedSimulation.duplicate_check; if merge_confirm, an expert 'yes' is merged
        only if an independent second reviewer also says 'yes' (one merge_confirmation
        decision). A second-reviewer 'no' leaves both entries; the pair stays marked asked."""

def region_folds(identity_codes, sources) -> Dict[int, int]:
    """Animal -> region fold: 0 = foe_carpaths, 1 = snpa or foe_bohemia (any photo). Raises if
    an animal spans both regions (none do on calibration)."""

def summarize(sim, codes, n_images, n_bundles, audits, regions) -> dict:
    """nm.summarize + merge_confirmation counts + per-region rows: pruning-duplicate events per
    1,000 photos, count error, misassigned rate (by the part's region)."""

def declared_rules(rows) -> dict:   # mechanical checks of N2-R1..R4 below
```

With `NEW_DEFAULTS`, `SafeguardSimulation` must reproduce `nm.PrunedSimulation` **exactly**. This is checked live in the run and in a test.

### 2. `tools/simulate_czechlynx_pruning_safeguards.py` (new, ~180 lines)

- Takes the same inputs as `tools/simulate_czechlynx_nonmatch_pruning.py`: the F1 calibration directory (`evaluations/czechlynx/facing_adaptation_calibration_classes_v1/clustering`, manifest SHA-256 `0477e511…`, the same input as the saved F1 pruning run), raw metadata and a fresh output directory. Calibration only; it refuses a selection or test partition, as the existing CLIs do.
- Preparation repeats `gated_cli.prepare` with one change: folds come from `--folds {identity,region}`. Under `region`, the facing classifier, W4 confidence models and P(match) models are all fitted per region and applied to the other region's animals.
- **Reference check:** at `--folds identity` with `NEW_DEFAULTS`, the row `pmatch / ε 0.001 / τ 0.99` must equal the saved F1 rows. That means 2,116 decisions and count +1 at 0% error, and 2,485 decisions and count +1 at 3%. If it does not match, the run is marked failed.
- **Output:** `evaluations/czechlynx/pruning_safeguards_f1_v1/` containing `summary.json`, `simulations.csv`, `per_region.csv`, `index.html` and `manifest.json` (code, input and cache hashes before and after).

### 3. `tests/test_benchmark_pruning_safeguards.py` (new, ~12 synthetic tests)

- Defaults reproduce `PrunedSimulation` decisions and labels on the toy fixture.
- `min_shown` never stops early; `cosine_floor` adds exactly the missing top-k entries and costs them.
- A pruned true match inside the cosine top-k is found instead of being declared new.
- `merge_confirm`: a second "no" blocks the merge; both "yes" merges; the ledger kind and cost are applied.
- Region folds: an animal spanning both regions raises; models are never trained on the held-out region (for both sides of each pair).
- The CLI refuses non-calibration inputs and existing outputs; the reference mismatch leaves a failed manifest.
- `declared_rules` arithmetic.

## Experiment grid (all *calibration*, F1 vectors, P(match) order, τ 0.99)

| Factor | Levels |
| --- | --- |
| Folds | identity (reference), **region (primary)** |
| ε | 0.001 (frozen value), 0.0003 |
| `min_shown` | 1, 5, 10 |
| `cosine_floor` | 0, 5, 10 |
| `merge_confirm` | off, on |
| Reviewer error | 0, 3%, 5% |
| Seeds | 1 at 0% error; 5 at 3% and 5% |

That gives 72 configurations, run as 792 simulations. At about 3 s each, that is roughly 40 CPU-minutes. Other safeguards stay at their frozen defaults. No GPU or model inference: F1 vectors come from the existing calibration cache.

## Declared rules (fixed before running; primary = region folds)

| Rule | Criterion | Why |
| --- | --- | --- |
| **N2-R1** duplicates transfer | At 0% error, in **each** held-out region: pruning-duplicate events ≤ 0.5 per 1,000 photos **and** overall \|count error\| ≤ 1 | Selection had 2.1 per 1,000; calibration 0.47 |
| **N2-R2** saving survives | Decisions ≤ 0.5 × W1 (≤ 2,662; W1 = 5,325 on calibration with F1 vectors) at 0% error | P2 saving floor |
| **N2-R3** reviewer robustness | Median over 5 seeds: at 3% error, misassigned ≤ 2% and \|count\| ≤ 1; at 5%, misassigned ≤ 3% and \|count\| ≤ 2 | S4 / P3 |
| **N2-R4** merge check works | With `merge_confirm` on at 3% error: median wrong merges ≤ 1 and ≤ 25% of merges | 4/6 wrong on calibration today |

**Outcome:**
- **Candidate** = the cheapest configuration (fewest decisions) meeting R1–R4 under region folds. It becomes the method to freeze for the blind partner test (X1). **It is not run on selection.**
- If no configuration meets R1 with R2, the pitch analysis's redirect applies: keep a full scan for new animals (drop pruning) and report the cost.
- If R4 fails, a merge check alone is insufficient, and duplicate merges go to an adjudication queue instead.

## Commands (after approval; the user runs them or authorizes the assistant)

```bash
venv/bin/python -m pytest tests/test_benchmark_pruning_safeguards.py tests/test_benchmark_nonmatch_pruning.py tests/test_benchmark_frozen_review.py -q
```

```bash
venv/bin/python tools/simulate_czechlynx_pruning_safeguards.py --calibration-dir evaluations/czechlynx/facing_adaptation_calibration_classes_v1/clustering --raw-metadata data/czechlynx/raw/CzechLynxDataset-Metadata-Real.csv --folds region --out-dir evaluations/czechlynx/pruning_safeguards_f1_v1
```

The flags mirror `tools/simulate_czechlynx_nonmatch_pruning.py`, plus `--folds`. One invocation runs both fold settings for the live reference check; `--folds` selects which one is primary.

## Limits

- Two regions only, with 17 and 38 animals. A region holdout is a proxy for a new park, not a new park. Day-level dates and supplied masks remain.
- Simulated reviewers and placeholder costs. Real timing is N3.
- This is development evidence. Any claim needs the blind partner test (X1).
