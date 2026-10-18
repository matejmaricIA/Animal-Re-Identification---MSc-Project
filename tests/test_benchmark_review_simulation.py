"""Behavioural regressions for the simulated review-effort experiment."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from reid_demo.benchmark import file_digest, write_json
import reid_demo.benchmark_review_simulation as rs
import tools.simulate_czechlynx_review as cli
from test_benchmark_clustering import run_small_calibration


def test_fast_quality_counts_pairs_exact_recoveries_and_unresolved():
    codes = np.asarray([0, 0, 0, 1, 1, 2])
    q = rs.fast_quality(np.asarray([0, 0, 1, 2, 2, -1]), codes)
    assert q["pairwise_precision"] == 1 and q["pairwise_recall"] == pytest.approx(2 / 4)
    assert q["exact_identity_recoveries"] == 1  # Only animal 1 is one pure group.
    assert q["groups"] == 3 and q["animals_with_resolved_images"] == 2 and q["count_error"] == 1
    assert q["unresolved_images"] == 1 and q["split_errors"] == 1 and q["merge_errors"] == 0
    merged = rs.fast_quality(np.asarray([0, 0, 0, 0, 0, 0]), codes)
    assert merged["merge_errors"] == 1 and merged["pairwise_recall"] == 1


def test_reviewer_is_truthful_without_error_and_reproducible_with_error():
    assert all(rs.Reviewer(0).same(t) == t for t in [True, False] * 50)
    first = rs.Reviewer(.3, seed=5)
    second = rs.Reviewer(.3, seed=5)
    answers = [first.same(True) for _ in range(200)]
    assert answers == [second.same(True) for _ in range(200)] and 0 < answers.count(False) < 200
    with pytest.raises(ValueError):
        rs.Reviewer(.6)


def frame(ids, traps, dates):
    return pd.DataFrame({"image_id": ids, "source": "s", "trap_id": traps, "timestamp": dates})


def test_bundles_group_camera_days_and_fall_back_to_single_images():
    f = frame(list("abcd"), ["t1", "t1", "t2", None], ["d1", "d1", "d1", "d1"])
    assert rs.bundle_keys(f) == ["s:t1|d1", "s:t1|d1", "s:t2|d1", "image|d"]
    assert rs.bundle_keys(f, "none") == list("abcd")
    with pytest.raises(ValueError):
        rs.bundle_keys(f, "week")


def test_units_follow_date_order_and_charge_extra_animals_in_a_bundle():
    f = frame(list("abcde"), ["t1", "t1", "t2", "t2", "t2"], ["d2", "d2", "d1", "d1", "d1"])
    codes = np.asarray([0, 0, 1, 2, 1])
    units = rs.ordered_units(f, codes, rs.bundle_keys(f))
    assert [u["images"] for u in units] == [[2, 4], [3], [0, 1]]
    assert [u["extra"] for u in units] == [0, 1, 0]


def test_ranked_catalogue_costs_one_comparison_when_the_right_animal_ranks_first():
    # Three animals with distinct directions; each later unit should match its animal first.
    vectors = np.asarray([[1, 0], [0, 1], [-1, 0], [1, .05], [0, 1.05], [-1, .05]], float)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    scores = vectors @ vectors.T
    codes = np.asarray([0, 1, 2, 0, 1, 2])
    units = [{"images": [i], "identity": int(codes[i]), "extra": 0} for i in range(6)]
    labels, effort = rs.simulate_catalogue(units, scores, 6, rs.Reviewer(0), ranked=True)
    # New animals cost 0 + 1 + 2 comparisons; each repeat costs exactly 1.
    assert effort.counts["catalogue_comparison"] == 3 + 3
    assert rs.fast_quality(labels, codes)["pairwise_f1"] == 1
    unassisted, effort0 = rs.simulate_catalogue(units, scores, 6, rs.Reviewer(0), ranked=False, seed=1)
    assert rs.fast_quality(unassisted, codes)["pairwise_f1"] == 1
    assert effort0.counts["catalogue_comparison"] >= effort.counts["catalogue_comparison"]


def test_catalogue_ranking_never_depends_on_labels():
    scores = np.asarray([[1, .9, .1], [.9, 1, .2], [.1, .2, 1]])
    first, second = rs.Catalogue(3), rs.Catalogue(3)
    for catalogue, founders in ((first, [5, 6]), (second, [6, 5])):
        for image, founder in zip([1, 2], founders):
            entry = catalogue.add([image])
            catalogue.founder[entry] = founder
    assert first.ranked(scores, [0]) == second.ranked(scores, [0]) == [0, 1]


def test_split_phase_moves_exactly_the_wrong_units_and_sets_aside_unresolvable_ones():
    codes = np.asarray([0, 0, 0, 1, 1, 2])
    start = np.asarray([0, 0, 0, 0, 1, 1])  # Image 3 (animal 1) wrongly in group 0.
    keys = ["k0", "k0", "k1", "k2", "k3", "k4"]
    effort = rs.Effort()
    labels, orphans, majority = rs.split_phase(start, codes, keys, rs.Reviewer(0), effort)
    assert majority == {0: 0, 1: 1}  # Group 1 ties 1:1; the first-seen animal is its majority.
    moved = sorted(i for u in orphans for i in u["images"])
    assert moved == [3, 5]
    assert labels.tolist() == [0, 0, 0, -1, 1, -1]
    assert effort.counts["glance"] == 5
    risk = np.asarray([0, 0, 0, 0, 0, 1.0])
    labels2, _, _ = rs.split_phase(start, codes, keys, rs.Reviewer(0), rs.Effort(), risk, .5)
    assert labels2[5] == -2


def test_merge_phase_stops_after_k_consecutive_no_answers():
    labels = np.asarray([0, 1, 2, 3])
    founders = [0, 1, 0, 2]
    candidates = [(3.0, (0, 1)), (2.0, (1, 3)), (1.0, (0, 2))]
    effort = rs.Effort()
    out = rs.merge_phase(labels, founders, candidates, rs.Reviewer(0), effort, 2, np.asarray([0, 1, 0, 2]))
    assert effort.counts["pair_question"] == 2 and out[0] != out[2]  # Stopped before the true pair.
    effort = rs.Effort()
    out = rs.merge_phase(labels, founders, candidates, rs.Reviewer(0), effort, 3, np.asarray([0, 1, 0, 2]))
    assert out[0] == out[2] and effort.counts["pair_question"] == 3


def test_perfect_reviewer_without_a_stopping_limit_reaches_the_truth():
    rng = np.random.default_rng(0)
    codes = np.repeat(np.arange(6), 5)
    vectors = rng.normal(size=(30, 8)) + np.eye(8)[codes] * 3
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    scores = vectors @ vectors.T
    start = rng.integers(0, 9, 30)  # A messy automatic grouping.
    pairs = np.asarray([(a, b) for a in range(30) for b in range(a + 1, 30)])
    p = np.clip((scores[pairs[:, 0], pairs[:, 1]] + 1) / 2, .01, .99)
    keys = [f"b{i}" for i in range(30)]
    curve = []
    labels, effort = rs.simulate_group_first(start, codes, keys, scores, pairs, p, rs.Reviewer(0), 10_000, curve=curve)
    q = rs.fast_quality(labels, codes)
    assert q["pairwise_f1"] == 1 and q["exact_identity_recoveries"] == 6 and q["count_error"] == 0
    assert curve and curve[0]["phase"] == "split" and effort.decisions > 0


def test_report_values_are_json_safe():
    assert rs.plain({"a": np.int64(3), "b": [np.float64(.5)], "c": np.arange(2)}) == {"a": 3, "b": [.5], "c": [0, 1]}


def test_effort_rejects_unknown_kinds_and_costs_seconds():
    effort = rs.Effort()
    effort.add("glance", 2)
    effort.add("pair_question")
    assert effort.decisions == 3 and effort.seconds == 2 * rs.COSTS["glance"] + rs.COSTS["pair_question"]
    with pytest.raises(ValueError):
        effort.add("coffee")


def test_reproduction_check_rejects_a_mismatch():
    assert cli.check_reproduced("x", {"groups": 3, "pairwise_precision": .5}, {"groups": 3, "pairwise_precision": .5})
    with pytest.raises(ValueError, match="does not reproduce"):
        cli.check_reproduced("x", {"groups": 4, "pairwise_precision": .5}, {"groups": 3, "pairwise_precision": .5})


def test_cli_plumbing_with_synthetic_starting_groupings(tmp_path, monkeypatch):
    fixture, calibration = run_small_calibration(tmp_path, monkeypatch)

    def fake_starts(args, source, inputs, matrix, scores, risk):
        n = len(inputs.image_ids)
        pairs = np.asarray([(a, b) for a in range(n) for b in range(a + 1, n)])
        p = np.clip((scores[pairs[:, 0], pairs[:, 1]] + 1) / 2, .01, .99)
        starts = {"edge_grouping": np.arange(n) % 3, "flank_grouping": np.zeros(n, int),
                  "legacy_control": inputs.identity_codes.copy()}
        return starts, pairs, p, {k: {"reproduced_saved": {}, **rs.fast_quality(v, inputs.identity_codes)}
                                  for k, v in starts.items()}
    monkeypatch.setattr(cli, "starting_groupings", fake_starts)
    edge_dir, flank_dir = tmp_path / "edge", tmp_path / "flank"
    edge_dir.mkdir()
    flank_dir.mkdir()
    (edge_dir / "grouping_profiles.csv").write_text("x\n1\n")
    (flank_dir / "profiles.csv").write_text("x\n1\n")
    raw = tmp_path / "raw" / "raw.csv"
    raw.parent.mkdir()
    raw.write_text("path,pose\n")
    out = tmp_path / "review"
    args = ["--calibration-dir", str(calibration), "--edge-grouping-dir", str(edge_dir),
            "--flank-grouping-dir", str(flank_dir), "--raw-metadata", str(raw), "--out-dir", str(out)]
    cache_before = {p.name: file_digest(p) for p in fixture["cache_dir"].iterdir()}
    assert cli.main(args) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert {p.name: file_digest(p) for p in fixture["cache_dir"].iterdir()} == cache_before
    for name, digest in manifest["artifacts"].items():
        assert file_digest(out / name) == digest
    sims = pd.read_csv(out / "simulations.csv")
    assert len(sims) == 2 * 2 * 2 * (2 + 3 * len(rs.STOP_AFTER_NO))
    perfect = sims[(sims.reviewer_error == 0) & (~sims.unresolved_variant)]
    assert (perfect[perfect.workflow.isin(["W0_unassisted", "W1_ranked"])].pairwise_f1 == 1).all()
    summary = json.loads((out / "summary.json").read_text())
    assert {"grouping_saves_work", "strong_saving", "robust_to_reviewer_error"} <= set(summary["declared_rules"])
    assert cli.main(args) == 1  # Existing evidence is not overwritten.


def test_selection_is_refused(tmp_path):
    directory = tmp_path / "selection"
    write_json(directory / "manifest.json", {"partition": "selection"})
    assert cli.main(["--calibration-dir", str(directory), "--edge-grouping-dir", "e", "--flank-grouping-dir", "f",
                     "--raw-metadata", "r", "--out-dir", str(tmp_path / "out")]) == 1
    assert not (tmp_path / "out").exists()


def test_tool_import_does_not_import_neural_dependencies():
    code = ("import sys; import tools.simulate_czechlynx_review; "
            "assert not {'torch', 'timm', 'transformers', 'lightglue', 'romatch'} & set(sys.modules)")
    completed = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
