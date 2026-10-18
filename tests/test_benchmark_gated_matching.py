"""Behavioural regressions for confidence-gated ranked matching (W4)."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from reid_demo.benchmark import file_digest, write_json
import reid_demo.benchmark_gated_matching as gm
from reid_demo.benchmark_review_simulation import Catalogue, Effort, Reviewer, fast_quality
import tools.simulate_czechlynx_gated_matching as cli
from test_benchmark_clustering import run_small_calibration


def context(vectors, cameras=None, days=None, facing=None):
    vectors = np.asarray(vectors, float)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    n = len(vectors)
    frame = pd.DataFrame({"source": "s", "trap_id": cameras or [f"t{i}" for i in range(n)],
                          "timestamp": days or [f"2020-01-{i % 28 + 1:02d}" for i in range(n)],
                          "location": ["L"] * n})
    return gm.Context(frame, vectors @ vectors.T, np.zeros(n), facing if facing is not None else np.zeros(n, int))


def catalogue_with(entries, n):
    catalogue = Catalogue(n)
    for founder, images in entries:
        entry = catalogue.add(images)
        catalogue.founder[entry] = founder
    return catalogue


def test_match_features_rank_candidates_without_labels_and_measure_margin():
    ctx = context([[1, 0], [0, 1], [1, .1], [.1, 1]])
    a = catalogue_with([(7, [0]), (8, [1])], 4)
    b = catalogue_with([(99, [0]), (5, [1])], 4)  # Same images, different founders (labels).
    fa, order_a = gm.match_features([2], a, ctx)
    fb, order_b = gm.match_features([2], b, ctx)
    assert order_a == order_b == [0, 1] and fa == fb
    assert fa["margin"] > 0.5 and fa["s1"] > fa["s2"]


def test_space_time_features_reflect_camera_history_and_distance():
    ctx = context([[1, 0], [1, .05], [0, 1]], cameras=["c1", "c1", "c2"])
    ctx.lat[:] = [45.0, 45.0, 46.0]
    ctx.lon[:] = [15.0, 15.0, 15.0]
    catalogue = catalogue_with([(0, [0]), (1, [2])], 3)
    features, _ = gm.match_features([1], catalogue, ctx)
    assert features["seen_same_camera"] == 1 and features["runner_up_seen_same_camera"] == 0
    assert features["km_missing"] == 0 and features["log_km"] == pytest.approx(0)


def bundles_world():
    # Three animals; each seen in several single-photo bundles with distinct directions.
    rng = np.random.default_rng(0)
    centres = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    codes = np.repeat(np.arange(3), 6)
    vectors = centres[codes] + rng.normal(scale=.15, size=(18, 3))
    order = rng.permutation(18)
    return vectors, codes, [[int(i)] for i in order]


def test_replay_marks_new_animals_as_incorrect_and_known_ones_by_rank():
    vectors, codes, bundles = bundles_world()
    records = gm.replay_records(bundles, codes, context(vectors))
    assert len(records) == 17
    assert set(records.columns) >= set(gm.APPEARANCE_FEATURES) | set(gm.SPACETIME_FEATURES) | {"top1_correct"}
    assert records.top1_correct.sum() >= 12


def test_fold_models_never_train_on_their_own_animal(monkeypatch):
    import sklearn.pipeline

    class Recorder:
        def fit(self, x, y):
            self.seen = set(x[:, 0].tolist())
            return self

        def predict_proba(self, x):
            assert not set(x[:, 0].tolist()) & self.seen
            return np.column_stack([np.full(len(x), .4), np.full(len(x), .6)])

    monkeypatch.setattr(sklearn.pipeline, "make_pipeline", lambda *a, **k: Recorder())
    records = pd.DataFrame({"code": np.repeat(np.arange(10), 4).astype(float), "top1_correct": np.tile([0, 1], 20),
                            "identity": np.repeat(np.arange(10), 4)})
    fold = {i: i % 5 for i in range(10)}
    models = gm.fit_fold_models(records, fold, ["code"])
    assert (gm.cross_fitted_confidence(records, models, fold, ["code"]) == .6).all()


class Always:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, x):
        return np.column_stack([1 - np.full(len(x), self.p), np.full(len(x), self.p)])


def test_automation_never_creates_animals_and_its_errors_propagate():
    # Animal 1 looks like animal 0; with forced automation every bundle after the first joins entry 0.
    ctx = context([[1, 0], [1, .02], [1, .04]])
    codes = np.asarray([0, 1, 1])
    models = {0: Always(1.0)}
    catalogue, effort, log = gm.simulate_gated([[0], [1], [2]], codes, ctx, models, {0: 0, 1: 0}, ["s1"], .5, 10,
                                               Reviewer(0))
    assert len(catalogue.founder) == 1  # Animal 1 was never created.
    assert [r["correct"] for r in log] == [False, False]
    wrong, rate = gm.misassigned_photos(catalogue.entry_of, catalogue, codes)
    assert wrong == 2 and rate == pytest.approx(2 / 3)
    assert effort.decisions == 0


def test_expert_path_top_k_miss_creates_a_duplicate_that_the_sweep_merges():
    # Animal 0 seen first, then three decoy animals rank above its second photo; K=1 misses it.
    ctx = context([[1, 0], [0, 1], [.2, 1], [.3, 1], [.1, 1]])
    codes = np.asarray([0, 1, 2, 3, 0])
    models = {0: Always(0.0)}
    fold = {i: 0 for i in range(4)}
    catalogue, effort, log = gm.simulate_gated([[0], [1], [2], [3], [4]], codes, ctx, models, fold, ["s1"], .5, 1,
                                               Reviewer(0))
    assert not log and len(catalogue.founder) == 5  # A duplicate of animal 0.
    labels = gm.duplicate_sweep(catalogue, ctx, 10, Reviewer(0), effort)
    q = fast_quality(labels, codes)
    assert q["groups"] == 4 and q["pairwise_f1"] == 1 and effort.counts["pair_question"] > 0


def test_wilson_interval_and_audit_coverage():
    p, low, high = gm.wilson_interval(2, 100)
    assert p == .02 and 0 < low < .02 < high < .08
    assert gm.wilson_interval(0, 0) == (None, None, None)
    log = [{"correct": i % 10 != 0} for i in range(200)]
    effort = Effort()
    result = gm.audit(log, 100, Reviewer(0), effort, seed=3)
    assert result["audited"] == 100 and effort.counts["glance"] == 100
    assert result["true_bundle_error"] == pytest.approx(.1) and result["covered"]
    assert gm.audit([], 100, Reviewer(0), Effort())["audited"] == 0


def test_ordered_bundles_follow_date_then_key():
    frame = pd.DataFrame({"timestamp": ["d2", "d1", "d2", "d1"]})
    assert gm.ordered_bundles(frame, ["b", "a", "b", "c"]) == [[1], [3], [0, 2]]


def test_cli_plumbing_with_synthetic_context(tmp_path, monkeypatch):
    fixture, calibration = run_small_calibration(tmp_path, monkeypatch)

    def fake_prepare(args, source, inputs):
        n = len(inputs.image_ids)
        ctx = gm.Context(inputs.frame.assign(location="L"), inputs.vectors @ inputs.vectors.T, np.zeros(n),
                         np.zeros(n, int))
        bundles = [[i] for i in range(n)]
        folds = {int(c): int(c) % 5 for c in inputs.identity_codes}
        records = gm.replay_records(bundles, inputs.identity_codes, ctx)
        records["top1_correct"] = np.arange(len(records)) % 2  # Both outcomes in every fold.
        return ctx, bundles, folds, records
    monkeypatch.setattr(cli, "prepare", fake_prepare)
    raw = tmp_path / "raw" / "raw.csv"
    raw.parent.mkdir()
    raw.write_text("path,latitude,longitude\n")
    out = tmp_path / "gated"
    args = ["--calibration-dir", str(calibration), "--raw-metadata", str(raw), "--out-dir", str(out)]
    cache_before = {p.name: file_digest(p) for p in fixture["cache_dir"].iterdir()}
    assert cli.main(args) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert {p.name: file_digest(p) for p in fixture["cache_dir"].iterdir()} == cache_before
    for name, digest in manifest["artifacts"].items():
        assert file_digest(out / name) == digest
    sims = pd.read_csv(out / "simulations.csv")
    sequential = len(gm.REVIEWER_ERRORS) * len(gm.TOP_K) * (len(gm.THRESHOLDS) + 1 + len(gm.THRESHOLDS))
    assert len(sims) == sequential * len(gm.SWEEPS) * len(gm.AUDITS)
    assert (sims[sims.tau.isna()].automation == 0).all()
    assert cli.main(args) == 1


def test_selection_is_refused(tmp_path):
    directory = tmp_path / "selection"
    write_json(directory / "manifest.json", {"partition": "selection"})
    assert cli.main(["--calibration-dir", str(directory), "--raw-metadata", "r", "--out-dir", str(tmp_path / "o")]) == 1
    assert not (tmp_path / "o").exists()


def test_tool_import_does_not_import_neural_dependencies():
    code = ("import sys; import tools.simulate_czechlynx_gated_matching; "
            "assert not {'torch', 'timm', 'transformers', 'lightglue', 'romatch'} & set(sys.modules)")
    completed = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
