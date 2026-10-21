"""Behavioural regressions for calibrated non-match pruning with review safeguards."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from reid_demo.benchmark import file_digest, write_json
import reid_demo.benchmark_gated_matching as gm
import reid_demo.benchmark_nonmatch_pruning as nm
from reid_demo.benchmark_review_simulation import Reviewer, bundle_keys, fast_quality, simulate_catalogue, split_by_truth
import tools.simulate_czechlynx_nonmatch_pruning as cli
import tools.simulate_czechlynx_gated_matching as gated_cli
from test_benchmark_clustering import run_small_calibration
from test_benchmark_gated_matching import Always, catalogue_with, context


APPEARANCE = list(nm.PAIR_APPEARANCE)


def setting(**kw):
    return {"order": "cosine", "eps": 0.0, "tau": None, **nm.DEFAULTS, **nm.OFF, **kw}


def by_similarity(cut=.9, high=.9):
    """Scripted P(match): high only for candidates whose best cosine exceeds `cut`."""
    return lambda x: np.where(x[:, 0] > cut, high, 0.0)


def simulation(vectors, codes, bundles, *, conf=None, pair=None, fold=None, expert=None, second=None, **kw):
    ctx = context(vectors)
    fold = fold or {int(c): 0 for c in codes}
    return nm.PrunedSimulation(bundles, np.asarray(codes), nm.PairContext(ctx), {0: conf or Always(0.0)}, ["s1"],
                               {0: pair or by_similarity()}, APPEARANCE, fold, setting(**kw),
                               expert or Reviewer(0), second or Reviewer(0)).run()


class Scripted:
    """Reviewer whose answers are flipped at the listed question numbers (0-based)."""

    def __init__(self, flips=()):
        self.flips, self.asked = set(flips), 0

    def same(self, truth):
        answer = bool(truth) != (self.asked in self.flips)
        self.asked += 1
        return answer


# A decoy-heavy world: animal 0's second photo looks more like two decoys than like animal 0.
DECOY_VECTORS = [[1, 0], [0, 1], [.2, 1], [.1, 1]]
DECOY_CODES = [0, 1, 2, 0]
DECOY_BUNDLES = [[0], [1], [2], [3]]


def test_candidate_features_are_identity_free_and_follow_catalogue_ranking():
    ctx = nm.PairContext(context([[1, 0], [0, 1], [1, .1], [.1, 1], [.7, .7]]))
    a = catalogue_with([(7, [0]), (8, [1]), (9, [4])], 5)
    b = catalogue_with([(99, [0]), (5, [1]), (3, [4])], 5)  # Same photos, different founders (labels).
    xa, ids_a, order_a = nm.candidate_features([2], a, ctx)
    xb, ids_b, order_b = nm.candidate_features([2], b, ctx)
    np.testing.assert_array_equal(xa, xb)
    assert order_a == order_b == a.ranked(ctx.ctx.scores, [2])
    assert xa.shape == (3, len(nm.PAIR_COLUMNS))
    x, ids, order = nm.candidate_features([2], a, ctx, exclude=0)
    assert ids.tolist() == [1, 2] and 0 not in order


def test_space_time_features_reflect_camera_distance_and_same_day_conflict():
    ctx = context([[1, 0], [1, .05], [0, 1]], cameras=["c1", "c1", "c2"],
                  days=["2020-01-01", "2020-01-01", "2020-01-01"])
    ctx.lat[:] = [45.0, 45.0, 46.0]
    ctx.lon[:] = [15.0, 15.0, 15.0]
    cat = catalogue_with([(0, [0]), (1, [2])], 3)
    x, ids, _ = nm.candidate_features([1], cat, nm.PairContext(ctx))
    col = {c: i for i, c in enumerate(nm.PAIR_COLUMNS)}
    assert x[0, col["seen_same_camera"]] == 1 and x[1, col["seen_same_camera"]] == 0
    assert x[0, col["log_km"]] == pytest.approx(0) and x[0, col["km_missing"]] == 0
    assert x[1, col["log_same_day_km"]] == pytest.approx(np.log1p(111.19), rel=1e-3)  # Same day, ~111 km apart.
    assert x[1, col["same_day_missing"]] == 0


def test_pair_models_never_train_on_the_scored_animal_as_part_or_candidate(monkeypatch):
    import sklearn.pipeline

    class Recorder:
        def fit(self, x, y):
            self.seen = set(x[:, 0].tolist()) | set(x[:, 1].tolist())
            return self

    monkeypatch.setattr(sklearn.pipeline, "make_pipeline", lambda *a, **k: Recorder())
    rng = np.random.default_rng(0)
    part, cand = rng.integers(0, 10, 400), rng.integers(0, 10, 400)
    records = pd.DataFrame({"pid": part.astype(float), "cid": cand.astype(float), "match": (part == cand).astype(int),
                            "part_identity": part, "candidate_identity": cand})
    fold = {i: i % 5 for i in range(10)}
    models = nm.fit_pair_models(records, fold, ["pid", "cid"])
    for f, model in models.items():
        assert not {float(i) for i in range(10) if fold[i] == f} & model.seen


def test_scorer_matches_pipeline_probabilities():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 4)) * [1, 5, .1, 3] + [0, 2, 1, -1]
    y = (x[:, 0] + .2 * x[:, 1] + rng.normal(size=200) > 0).astype(int)
    model = make_pipeline(StandardScaler(), LogisticRegression()).fit(x, y)
    np.testing.assert_allclose(nm.scorer(model)(x), model.predict_proba(x)[:, 1], atol=1e-10)


def test_replay_records_label_every_candidate_and_mark_known_parts():
    vectors, codes, bundles = DECOY_VECTORS, np.asarray(DECOY_CODES), DECOY_BUNDLES
    records = nm.replay_pair_records(bundles, codes, nm.PairContext(context(vectors)))
    assert len(records) == 1 + 2 + 3  # Catalogue sizes 1, 2, 3 at parts 2-4.
    assert records.match.sum() == 1 and records[records.match == 1].part_identity.tolist() == [0]
    assert records.groupby("part").known.first().tolist() == [0, 0, 1]


def test_safeguard_free_full_scan_reproduces_w1():
    rng = np.random.default_rng(3)
    centres = rng.normal(size=(6, 8))
    codes = rng.integers(0, 6, 60)
    vectors = centres[codes] + rng.normal(scale=.9, size=(60, 8))
    bundles = [list(range(i, min(i + 3, 60))) for i in range(0, 60, 3)]
    sim = simulation(vectors, codes, bundles, pair=lambda x: np.zeros(len(x)))
    units = [{"images": part, "identity": identity, "extra": int(j > 0)}
             for b in bundles for j, (identity, part) in enumerate(split_by_truth(b, codes))]
    labels, effort = simulate_catalogue(units, sim.ctx.scores, 60, Reviewer(0), ranked=True)
    np.testing.assert_array_equal(sim.catalogue.entry_of, labels)
    assert sim.ledger.decisions() == effort.decisions
    assert sim.ledger.seconds("placeholder") == pytest.approx(effort.seconds)


def test_automation_never_creates_animals_and_its_errors_propagate():
    sim = simulation([[1, 0], [1, .02], [1, .04]], [0, 1, 1], [[0], [1], [2]], conf=Always(1.0), tau=.5)
    assert len(sim.catalogue.founder) == 1 and len(sim.declarations) == 1
    assert [r["correct"] for r in sim.auto_log] == [False, False]
    wrong, _ = gm.misassigned_photos(sim.catalogue.entry_of, sim.catalogue, np.asarray([0, 1, 1]))
    assert wrong == 2 and sim.ledger.decisions() == 0


def test_pruning_cut_creates_a_duplicate_that_full_scan_avoids_and_duplicate_check_merges():
    pruned = simulation(DECOY_VECTORS, DECOY_CODES, DECOY_BUNDLES, eps=.01)
    assert len(pruned.catalogue.founder) == 4
    assert [r["outcome"] for r in pruned.declarations][-1] == "new_pruning_duplicate"
    assert pruned.ledger.counts["rank1_comparison"] + pruned.ledger.counts["further_comparison"] == 1 + 1 + 2
    full = simulation(DECOY_VECTORS, DECOY_CODES, DECOY_BUNDLES)
    assert len(full.catalogue.founder) == 3
    checked = simulation(DECOY_VECTORS, DECOY_CODES, DECOY_BUNDLES, eps=.01, dup_every=10, dup_r=3)
    assert len(checked.catalogue.founder) == 3 and checked.duplicate_log == [
        {"bundle": 3, "kept_uid": 0, "merged_uid": 3, "correct": True}]
    assert fast_quality(checked.catalogue.entry_of, np.asarray(DECOY_CODES))["pairwise_f1"] == 1
    rescued = simulation(DECOY_VECTORS, DECOY_CODES, DECOY_BUNDLES, eps=.01, double_c=3)
    assert len(rescued.catalogue.founder) == 3 and rescued.placements[-1]["outcome"] == "rescued_correct"


def test_cold_start_guard_blocks_automation_and_pruning():
    vectors = [[1, 0], [0, 1], [1, .01], [0, 1.01]]
    sim = simulation(vectors, [0, 1, 0, 1], [[0], [1], [2], [3]], conf=Always(1.0), tau=.5, guard_n=2, guard_b=2)
    assert [r["bundle"] for r in sim.auto_log] == [2, 3]
    # Pruning with P = 0 everywhere would stop after one candidate; the guard forces a full scan first.
    world = [[1, 0], [0, 1], [.1, 1]]
    guarded = simulation(world, [0, 1, 0], [[0], [1], [2]], pair=lambda x: np.zeros(len(x)), eps=.5, guard_n=3)
    assert len(guarded.catalogue.founder) == 2 and guarded.declarations[-1]["outcome"] == "new"
    open_ = simulation(world, [0, 1, 0], [[0], [1], [2]], pair=lambda x: np.zeros(len(x)), eps=.5)
    assert open_.declarations[-1]["outcome"] == "new_pruning_duplicate"


def test_double_confirmation_and_disagreement_check_recover_scripted_errors():
    world, codes, bundles = [[1, 0], [0, 1], [1, .05]], [0, 1, 0], [[0], [1], [2]]
    # Expert wrongly says "no" to animal 0 (question 1, i.e. bundle 2's first look); the second reviewer rescues it.
    lost = simulation(world, codes, bundles, expert=Scripted({1}))
    assert len(lost.catalogue.founder) == 3
    rescued = simulation(world, codes, bundles, expert=Scripted({1}), double_c=3)
    assert len(rescued.catalogue.founder) == 2 and rescued.placements[-1]["outcome"] == "rescued_correct"
    assert rescued.ledger.counts["second_comparison"] == 2  # Bundle 1's true new animal is re-checked too.
    # Expert wrongly says "yes" to a candidate the model rates unlikely; the disagreement check blocks it.
    decoy_first = [[1, 0], [.2, 1], [.15, 1]]
    wrong = simulation(decoy_first, codes, bundles, expert=Scripted({1}), pair=by_similarity(.9))
    assert wrong.placements[-1]["outcome"] == "matched_wrong"
    blocked = simulation(decoy_first, codes, bundles, expert=Scripted({1}), pair=by_similarity(.9, high=.01),
                         disagree_delta=.05)
    assert blocked.placements[-1]["outcome"] == "matched_correct"
    assert blocked.ledger.counts["disagreement_check"] == 2  # The true match (P = 0) is double-checked as well.


def test_merged_animals_are_never_shown_again():
    cat = nm.LiveCatalogue(4)
    for founder, images in [(0, [0]), (1, [1]), (0, [2])]:
        cat.create(images, founder)
    cat.merge(0, 2)
    assert cat.uid == [0, 1] and cat.founder == [0, 1]
    assert cat.entry_of.tolist() == [0, 1, 0, -1]
    scores = np.eye(4)
    assert cat.ranked(scores, [3]) == [0, 1]
    with pytest.raises(ValueError):
        cat.merge(1, 0)
    cat.create([3], 2)
    assert cat.uid == [0, 1, 3]


def test_ledger_costs_under_both_settings_and_count_audit_is_optional():
    ledger = nm.Ledger()
    ledger.add("rank1_comparison", 10)
    ledger.add("further_comparison", 4)
    ledger.add("glance", 2)
    ledger.add("count_audit_comparison", 5)
    assert ledger.counts["audit_glance"] == 2
    assert ledger.decisions() == 16 and ledger.decisions(count_audit=True) == 21
    assert ledger.seconds("placeholder") == 10 * 8 + 4 * 8 + 2 * 5
    assert ledger.seconds("fast_click") == 10 * 2 + 4 * 5 + 2 * 2
    assert ledger.seconds("placeholder", count_audit=True) == 122 + 40
    with pytest.raises(ValueError):
        ledger.add("coffee")


def test_count_audit_estimates_duplicates_and_count_interval():
    cat = nm.LiveCatalogue(6)
    for founder, images in [(0, [0]), (1, [1]), (2, [2]), (0, [3]), (1, [4]), (3, [5])]:
        cat.create(images, founder)
    ledger = nm.Ledger()
    result = nm.count_audit(cat, 6, Reviewer(0), ledger, np.eye(6), np.asarray([0, 1, 2, 0, 1, 3]))
    assert result["true_duplicate_share"] == pytest.approx(2 / 6) and result["estimate"] == pytest.approx(2 / 6)
    assert result["covered"] and result["count_covered"] and result["count_low"] <= 4 <= result["count_high"]
    assert ledger.counts["count_audit_comparison"] > 0
    assert nm.count_audit(cat, 0, Reviewer(0), nm.Ledger(), np.eye(6), np.zeros(6, int))["audited"] == 0


def test_experiment_grid_matches_the_proposal():
    configs = nm.experiment_configs()
    groups = pd.Series([c["group"] for c in configs]).value_counts().to_dict()
    assert groups == {"reference": 4, "G1": 80, "G2": 64}
    assert len({c["config"] for c in configs}) == len(configs)
    refs = [c for c in configs if c["group"] == "reference"]
    assert all(c["guard_n"] == 0 and c["double_c"] == 0 and c["dup_every"] == 0 and c["disagree_delta"] is None
               and c["eps"] == 0 and c["order"] == "cosine" for c in refs)


def fake_prepare(args, source, inputs):
    n = len(inputs.image_ids)
    ctx = gm.Context(inputs.frame.assign(location="L"), inputs.vectors @ inputs.vectors.T, np.zeros(n),
                     np.zeros(n, int))
    bundles = gm.ordered_bundles(inputs.frame, bundle_keys(inputs.frame, "none"))  # Per-photo bundles (tiny fixture).
    folds = {int(c): int(c) % 5 for c in inputs.identity_codes}
    records = gm.replay_records(bundles, inputs.identity_codes, ctx)
    records["top1_correct"] = np.arange(len(records)) % 2  # Both outcomes in every fold.
    return ctx, bundles, folds, records


def both_outcomes(original):
    def replay(*args):
        records = original(*args)
        records["match"] = np.arange(len(records)) % 2
        return records
    return replay


def cli_args(calibration, tmp_path, out):
    raw = tmp_path / "raw" / "raw.csv"
    raw.parent.mkdir(exist_ok=True)
    raw.write_text("path,latitude,longitude\n")
    return ["--calibration-dir", str(calibration), "--raw-metadata", str(raw), "--out-dir", str(out),
            "--no-expect-saved-references"]


def test_cli_plumbing_with_synthetic_context(tmp_path, monkeypatch):
    fixture, calibration = run_small_calibration(tmp_path, monkeypatch)
    monkeypatch.setattr(gated_cli, "prepare", fake_prepare)
    monkeypatch.setattr(cli, "bundle_keys", lambda frame, bundling: bundle_keys(frame, "none"))  # Live W1 alike.
    monkeypatch.setattr(nm, "replay_pair_records", both_outcomes(nm.replay_pair_records))
    out = tmp_path / "pruning"
    args = cli_args(calibration, tmp_path, out)
    cache_before = {p.name: file_digest(p) for p in fixture["cache_dir"].iterdir()}
    assert cli.main(args) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["reference_check"]["passed"]
    assert {p.name: file_digest(p) for p in fixture["cache_dir"].iterdir()} == cache_before
    for name, digest in manifest["artifacts"].items():
        assert file_digest(out / name) == digest
    sims = pd.read_csv(out / "simulations.csv")
    assert len(sims) == len(nm.experiment_configs())
    assert (sims[sims.tau.isna()].automation == 0).all()
    assert cli.main(args) == 1  # Output directory is no longer empty.


def test_failed_reference_check_preserves_a_failed_manifest(tmp_path, monkeypatch):
    fixture, calibration = run_small_calibration(tmp_path, monkeypatch)
    monkeypatch.setattr(gated_cli, "prepare", fake_prepare)
    monkeypatch.setattr(cli, "bundle_keys", lambda frame, bundling: bundle_keys(frame, "none"))  # Live W1 alike.
    monkeypatch.setattr(nm, "replay_pair_records", both_outcomes(nm.replay_pair_records))
    monkeypatch.setattr(nm, "check_references", lambda *a: {"passed": False, "problems": ["forced"]})
    out = tmp_path / "pruning"
    assert cli.main(cli_args(calibration, tmp_path, out)) == 1
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "failed" and "Reference simulations" in manifest["error"]


def test_selection_is_refused(tmp_path):
    directory = tmp_path / "selection"
    write_json(directory / "manifest.json", {"partition": "selection"})
    assert cli.main(["--calibration-dir", str(directory), "--raw-metadata", "r", "--out-dir", str(tmp_path / "o")]) == 1
    assert not (tmp_path / "o").exists()


def test_tool_import_does_not_import_neural_dependencies():
    code = ("import sys; import tools.simulate_czechlynx_nonmatch_pruning; "
            "assert not {'torch', 'timm', 'transformers', 'lightglue', 'romatch'} & set(sys.modules)")
    completed = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
