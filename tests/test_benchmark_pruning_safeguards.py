"""Behavioural regressions for the N2 pruning floors and duplicate-merge second check."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from reid_demo.benchmark import file_digest, write_json
import reid_demo.benchmark_nonmatch_pruning as nm
import reid_demo.benchmark_pruning_safeguards as ps
from reid_demo.benchmark_review_simulation import Reviewer, bundle_keys
import tools.simulate_czechlynx_gated_matching as gated_cli
import tools.simulate_czechlynx_nonmatch_pruning as nm_cli
import tools.simulate_czechlynx_pruning_safeguards as cli
from test_benchmark_clustering import run_small_calibration
from test_benchmark_gated_matching import Always, context
from test_benchmark_nonmatch_pruning import (
    APPEARANCE, DECOY_BUNDLES, DECOY_CODES, DECOY_VECTORS, Scripted, both_outcomes, by_similarity, fake_prepare, setting)


def run(cls, vectors, codes, bundles, *, conf=None, pair=None, expert=None, second=None, **kw):
    ctx = context(vectors)
    fold = {int(c): 0 for c in codes}
    return cls(bundles, np.asarray(codes), nm.PairContext(ctx), {0: conf or Always(0.0)}, ["s1"],
               {0: pair or by_similarity()}, APPEARANCE, fold, setting(**kw),
               expert or Reviewer(0), second or Reviewer(0)).run()


def safeguarded(*args, **kw):
    return run(ps.SafeguardSimulation, *args, **kw)


# Animal 0 is photographed twice; a perverse P(match) (high only for DISSIMILAR candidates) ranks the
# true animal last, as a mis-transferred location prior would, while appearance ranks it first.
HIDDEN_VECTORS = [[1, 0], [0, 1], [1, .05]]
HIDDEN_CODES = [0, 1, 0]
HIDDEN_BUNDLES = [[0], [1], [2]]


def perverse(x):
    return np.where(x[:, 0] < .5, .9, 0.0)


class TopSimilarity:
    """Confidence = the top-1 similarity feature, so only near-identical photos are automated."""

    def predict_proba(self, x):
        p = np.clip(np.asarray(x, float)[:, 0], 0, 1)
        return np.column_stack([1 - p, p])


def test_defaults_reproduce_pruned_simulation_exactly():
    rng = np.random.default_rng(7)
    centres = rng.normal(size=(8, 6))
    codes = rng.integers(0, 8, 80)
    vectors = centres[codes] + rng.normal(scale=.8, size=(80, 6))
    bundles = [list(range(i, min(i + 2, 80))) for i in range(0, 80, 2)]
    kw = dict(order="pmatch", eps=.01, dup_every=5, dup_r=3, double_c=2, disagree_delta=.05,
              conf=TopSimilarity(), tau=.99)
    base = run(nm.PrunedSimulation, vectors, codes, bundles, expert=Reviewer(.1, 1), second=Reviewer(.1, 2), **kw)
    new = safeguarded(vectors, codes, bundles, expert=Reviewer(.1, 1), second=Reviewer(.1, 2), **kw)
    np.testing.assert_array_equal(base.catalogue.entry_of, new.catalogue.entry_of)
    assert base.catalogue.founder == new.catalogue.founder
    assert dict(base.ledger.counts) == dict(new.ledger.counts)
    assert [r["outcome"] for r in base.placements] == [r["outcome"] for r in new.placements]
    assert base.duplicate_log == new.duplicate_log and len(base.auto_log) == len(new.auto_log)
    assert new.blocked_merges == [] and not new.ledger.counts["merge_confirmation"]
    assert base.auto_log and base.duplicate_log and base.declarations  # The paths were exercised.


def test_hidden_true_match_becomes_a_pruning_duplicate_without_floors():
    sim = safeguarded(HIDDEN_VECTORS, HIDDEN_CODES, HIDDEN_BUNDLES, order="pmatch", eps=.01, pair=perverse)
    assert sim.declarations[-1]["outcome"] == "new_pruning_duplicate"
    assert len(sim.catalogue.founder) == 3


def test_min_shown_never_stops_before_m_candidates():
    sim = safeguarded(HIDDEN_VECTORS, HIDDEN_CODES, HIDDEN_BUNDLES, order="pmatch", eps=.01, pair=perverse,
                      min_shown=2)
    assert len(sim.catalogue.founder) == 2 and sim.placements[-1]["outcome"] == "matched_correct"
    assert sim.placements[-1]["floor_shown"] == 0  # Found inside the main scan, not by the cosine floor.
    with pytest.raises(ValueError):
        safeguarded(HIDDEN_VECTORS, HIDDEN_CODES, HIDDEN_BUNDLES, min_shown=0)


def test_cosine_floor_shows_the_appearance_top_k_before_declaring():
    sim = safeguarded(HIDDEN_VECTORS, HIDDEN_CODES, HIDDEN_BUNDLES, order="pmatch", eps=.01, pair=perverse,
                      cosine_floor=1)
    last = sim.placements[-1]
    assert len(sim.catalogue.founder) == 2
    assert last["outcome"] == "matched_correct" and last["matched_by_floor"] and last["floor_shown"] == 1
    # Bundle 2: one rank-1 comparison (shown animal 1) + one floor comparison (animal 0).
    assert sim.ledger.counts["further_comparison"] == 1
    assert sim.ledger.counts["rank1_comparison"] == 2


def test_cosine_floor_only_adds_entries_not_already_shown():
    # Cosine order: the decoys rank first, so a floor of 1 adds nothing and the duplicate remains.
    sim = safeguarded(DECOY_VECTORS, DECOY_CODES, DECOY_BUNDLES, eps=.01, cosine_floor=1)
    assert sim.declarations[-1]["outcome"] == "new_pruning_duplicate"
    assert all(r["floor_shown"] == 0 for r in sim.placements)
    wide = safeguarded(DECOY_VECTORS, DECOY_CODES, DECOY_BUNDLES, eps=.01, cosine_floor=3)
    assert len(wide.catalogue.founder) == 3 and wide.placements[-1]["matched_by_floor"]


def test_merge_confirm_second_no_blocks_the_merge():
    blocked = safeguarded(DECOY_VECTORS, DECOY_CODES, DECOY_BUNDLES, eps=.01, dup_every=10, dup_r=3,
                          merge_confirm=True, second=Scripted(flips={0}))
    assert len(blocked.catalogue.founder) == 4 and blocked.duplicate_log == []
    assert blocked.blocked_merges == [{"bundle": 3, "uids": [0, 3], "true_duplicate": True}]
    assert blocked.ledger.counts["merge_confirmation"] == 1
    merged = safeguarded(DECOY_VECTORS, DECOY_CODES, DECOY_BUNDLES, eps=.01, dup_every=10, dup_r=3,
                         merge_confirm=True)
    assert len(merged.catalogue.founder) == 3 and merged.duplicate_log[0]["correct"]
    assert merged.ledger.counts["merge_confirmation"] == 1


def test_merge_confirm_stops_an_expert_false_merge():
    # A wrong expert "yes" on the first pair question would merge two different animals.
    world = dict(eps=.01, dup_every=10, dup_r=1, pair=lambda x: np.full(len(x), .5))
    vectors, codes, bundles = [[1, 0], [0, 1]], [0, 1], [[0], [1]]
    flips = {1}  # Question 0 places bundle 1 (truthful "no"); question 1 is the pair question.
    unchecked = safeguarded(vectors, codes, bundles, expert=Scripted(flips), **world)
    assert unchecked.duplicate_log == [{"bundle": 1, "kept_uid": 0, "merged_uid": 1, "correct": False}]
    checked = safeguarded(vectors, codes, bundles, expert=Scripted(flips), merge_confirm=True, **world)
    assert checked.duplicate_log == [] and len(checked.catalogue.founder) == 2
    assert checked.blocked_merges[0]["true_duplicate"] is False


def test_ledger_costs_merge_confirmation_like_a_pair_question():
    ledger = ps.SafeguardLedger()
    ledger.add("merge_confirmation", 2)
    ledger.add("pair_question")
    assert ledger.decisions() == 3
    assert ledger.seconds("placeholder") == pytest.approx(30.0)
    assert ledger.seconds("fast_click") == pytest.approx(18.0)
    assert isinstance(ledger.copy(), ps.SafeguardLedger) and ledger.copy().as_dict()["merge_confirmation"] == 2
    with pytest.raises(ValueError):
        ledger.add("coffee_break")


def test_region_folds_follow_the_animal_and_refuse_spanning_animals():
    folds = ps.region_folds([0, 0, 1, 2], ["snpa", "foe_bohemia", "foe_carpaths", "foe_bohemia"])
    assert folds.tolist() == [1, 1, 0, 1]
    with pytest.raises(ValueError, match="both regions"):
        ps.region_folds([0, 0], ["snpa", "foe_carpaths"])
    with pytest.raises(ValueError, match="Unknown"):
        ps.region_of_sources(["elsewhere"])


def test_summarize_adds_region_and_safeguard_fields():
    sim = safeguarded(DECOY_VECTORS, DECOY_CODES, DECOY_BUNDLES, eps=.01)
    ledger = sim.ledger.copy()
    audits = {"ledger": ledger, "auto": {"estimate": None, "low": None, "high": None, "true_bundle_error": None,
                                         "covered": None}, "count": {}}
    row = ps.summarize(sim, DECOY_CODES, 4, 4, audits, np.asarray([1, 1, 0, 1]))
    assert row["bohemian_forest_pruning_duplicate_events"] == 1   # Animal 0's duplicate, region 1.
    assert row["western_carpathians_pruning_duplicate_events"] == 0
    assert row["bohemian_forest_pruning_duplicates_per_1000"] == pytest.approx(1 / (3 / 1000))
    assert row["bohemian_forest_count_error"] == 1 and row["western_carpathians_count_error"] == 0
    assert row["blocked_merges"] == 0 and row["floor_comparisons"] == 0


def test_check_reference_compares_every_field():
    base = {f: 1 for f in ps.REFERENCE_FIELDS}
    rows = [{"group": "reference", "reviewer_error": e, **base} for e in ps.REFERENCE_TAGS]
    saved = {tag: dict(base) for tag in ps.REFERENCE_TAGS.values()}
    assert ps.check_reference(rows, saved)["passed"]
    saved[ps.REFERENCE_TAGS[0.03]]["decisions"] = 2
    result = ps.check_reference(rows, saved)
    assert not result["passed"] and "decisions" in result["problems"][0]


def _row(folds, eps, m, k, confirm, error, replicate, **kw):
    row = {"group": "grid", "folds": folds, "eps": eps, "min_shown": m, "cosine_floor": k, "merge_confirm": confirm,
           "reviewer_error": error, "replicate": replicate, "decisions": 2000, "automation": .5, "count_error": 0,
           "misassigned_photo_rate": .005, "duplicate_merges": 4, "wrong_duplicate_merges": 0,
           "western_carpathians_pruning_duplicates_per_1000": 0.0,
           "bohemian_forest_pruning_duplicates_per_1000": 0.0}
    row.update(kw)
    return row


def test_declared_rules_pick_the_cheapest_setting_meeting_all_rules():
    rows = []
    for k, decisions, dup in ((0, 1800, 1.2), (5, 2100, 0.2), (10, 2300, 0.0)):
        for error, reps in ps.REPLICATES.items():
            for r in range(reps):
                rows.append(_row("region", .001, 1, k, True, error, r, decisions=decisions,
                                 bohemian_forest_pruning_duplicates_per_1000=dup))
    rules = ps.declared_rules(rows, w1_decisions=5000)
    assert rules["r2_limit"] == 2500
    assert [s["R1"] for s in rules["settings"]] == [False, True, True]
    assert rules["candidate"]["cosine_floor"] == 5 and rules["candidate"]["decisions"] == 2100
    assert not rules["redirect_full_scan"] and not rules["redirect_adjudication_queue"]
    noisy = [dict(r, wrong_duplicate_merges=3) if r["reviewer_error"] == .03 else r for r in rows]
    assert ps.declared_rules(noisy, 5000)["redirect_adjudication_queue"]


def test_experiment_grid_matches_the_proposal():
    configs = ps.experiment_configs()
    assert len(configs) == 2 * 3 * 3 * 2 * (1 + 5 + 5)
    tags = {ps.config_tag("region", c) for c in configs}
    assert len(tags) == len(configs)
    assert all(c["tau"] == .99 and c["order"] == "pmatch" and c["spacetime"] for c in configs)


def test_cli_plumbing_with_synthetic_context(tmp_path, monkeypatch):
    fixture, calibration = run_small_calibration(tmp_path, monkeypatch)
    monkeypatch.setattr(gated_cli, "prepare", fake_prepare)
    monkeypatch.setattr(cli, "prepare_region", lambda args, inputs: fake_prepare(args, None, inputs))
    monkeypatch.setattr(nm_cli, "bundle_keys", lambda frame, bundling: bundle_keys(frame, "none"))
    monkeypatch.setattr(nm, "replay_pair_records", both_outcomes(nm.replay_pair_records))
    monkeypatch.setattr(ps, "REGIONS", {"test": 0})
    for name, value in (("EPSILONS", (.001,)), ("MIN_SHOWN", (1,)), ("COSINE_FLOORS", (0, 1)),
                        ("MERGE_CONFIRM", (False, True)), ("REPLICATES", {0.0: 1, 0.03: 1, 0.05: 1})):
        monkeypatch.setattr(ps, name, value)
    saved = {tag: {} for tag in ps.REFERENCE_TAGS.values()}
    monkeypatch.setattr(cli, "load_saved_reference", lambda *a: (saved, {"seed": 42, "threads": 4}))
    monkeypatch.setattr(ps, "check_reference", lambda rows, saved: {"passed": True, "problems": []})
    out = tmp_path / "safeguards"
    raw = tmp_path / "raw" / "raw.csv"
    raw.parent.mkdir()
    raw.write_text("path,latitude,longitude\n")
    args = ["--calibration-dir", str(calibration), "--raw-metadata", str(raw), "--out-dir", str(out)]
    cache_before = {p.name: file_digest(p) for p in fixture["cache_dir"].iterdir()}
    assert cli.main(args) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["reference_check"]["passed"]
    assert {p.name: file_digest(p) for p in fixture["cache_dir"].iterdir()} == cache_before
    for name, digest in manifest["artifacts"].items():
        assert file_digest(out / name) == digest
    sims = pd.read_csv(out / "simulations.csv")
    assert len(sims) == 2 + 2 * len(ps.experiment_configs())
    assert set(sims.folds) == {"identity", "region"}
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary["declared_rules"]) == {"identity", "region"}
    assert cli.main(args) == 1  # Output directory is no longer empty.


def test_failed_reference_check_preserves_a_failed_manifest(tmp_path, monkeypatch):
    fixture, calibration = run_small_calibration(tmp_path, monkeypatch)
    monkeypatch.setattr(gated_cli, "prepare", fake_prepare)
    monkeypatch.setattr(nm_cli, "bundle_keys", lambda frame, bundling: bundle_keys(frame, "none"))
    monkeypatch.setattr(nm, "replay_pair_records", both_outcomes(nm.replay_pair_records))
    monkeypatch.setattr(ps, "REGIONS", {"test": 0})
    monkeypatch.setattr(cli, "load_saved_reference", lambda *a: ({}, {"seed": 42, "threads": 4}))
    out = tmp_path / "safeguards"
    raw = tmp_path / "raw" / "raw.csv"
    raw.parent.mkdir()
    raw.write_text("path,latitude,longitude\n")
    assert cli.main(["--calibration-dir", str(calibration), "--raw-metadata", str(raw), "--out-dir", str(out)]) == 1
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "failed" and "saved F1 pruning rows" in manifest["error"]


def test_saved_reference_must_bind_the_same_vectors(tmp_path):
    reference = tmp_path / "ref"
    write_json(reference / "manifest.json", {"status": "complete", "run_spec": {
        "cache_sha256": "a", "calibration_manifest_sha256": "b", "seed": 42, "threads": 4}})
    with pytest.raises(ValueError, match="different vectors"):
        cli.load_saved_reference(reference, "other", "b")


def test_selection_is_refused(tmp_path):
    directory = tmp_path / "selection"
    write_json(directory / "manifest.json", {"partition": "selection"})
    assert cli.main(["--calibration-dir", str(directory), "--raw-metadata", "r", "--out-dir", str(tmp_path / "o")]) == 1
    assert not (tmp_path / "o").exists()


def test_tool_import_does_not_import_neural_dependencies():
    code = ("import sys; import tools.simulate_czechlynx_pruning_safeguards; "
            "assert not {'torch', 'timm', 'transformers', 'lightglue', 'romatch'} & set(sys.modules)")
    completed = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
