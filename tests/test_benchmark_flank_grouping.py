"""Behavioural regressions for flank-aware grouping."""
import builtins
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from reid_demo.benchmark import file_digest, write_json
import reid_demo.benchmark_flank_grouping as fg
import tools.evaluate_czechlynx_flank_grouping as cli
from test_benchmark_clustering import run_small_calibration


def pose(head_x, tail_x, *, y=100, drop=()):
    points = {"nose": [head_x, y], "left-eye": [head_x - 5, y - 5], "right-eye": [head_x + 5, y - 5],
              "tailbase": [tail_x, y + 10], "withers": [(head_x + tail_x) / 2, y]}
    return json.dumps({k: v for k, v in points.items() if k not in drop})


def test_pose_facing_geometry():
    assert fg.pose_facing(pose(400, 100)) == 1.0  # Head right of tail: faces right in frame.
    assert fg.pose_facing(pose(100, 400)) == 0.0
    assert np.isnan(fg.pose_facing(pose(200, 195)))  # Frontal/rear/oblique.
    assert np.isnan(fg.pose_facing(pose(400, 100, drop=("tailbase",))))
    assert np.isnan(fg.pose_facing(pose(400, 100, drop=("nose", "left-eye", "right-eye"))))
    assert np.isnan(fg.pose_facing(None)) and np.isnan(fg.pose_facing(""))


def test_pose_facing_aligns_by_source_path(tmp_path):
    raw = tmp_path / "raw.csv"
    pd.DataFrame({"path": ["a.jpg", "b.jpg", "c.jpg"], "pose": [pose(400, 100), pose(100, 400), None]}).to_csv(raw, index=False)
    frame = pd.DataFrame({"source_path": ["c.jpg", "a.jpg", "missing.jpg", "b.jpg"]})
    labels = fg.load_pose_facing(raw, frame)
    assert np.isnan(labels[0]) and labels[1] == 1 and np.isnan(labels[2]) and labels[3] == 0
    with pytest.raises(ValueError):
        fg.load_pose_facing(raw, pd.DataFrame({"path": ["a.jpg"]}))


def test_facing_classifier_never_predicts_an_animal_it_trained_on(monkeypatch):
    import sklearn.linear_model

    class Recorder:
        def __init__(self, **kwargs):
            pass

        def fit(self, x, y):
            self.seen = set(np.round(x[:, 0]).tolist())
            return self

        def predict_proba(self, x):
            assert not set(np.round(x[:, 0]).tolist()) & self.seen
            return np.column_stack([np.full(len(x), .3), np.full(len(x), .7)])

    monkeypatch.setattr(sklearn.linear_model, "LogisticRegression", Recorder)
    codes = np.repeat(np.arange(10), 4)
    vectors = np.column_stack([codes, np.zeros(len(codes))]).astype(float)
    labels = np.tile([0.0, 1.0, np.nan, 1.0], 10)
    p = fg.facing_probabilities(vectors, labels, fg.eg.identity_folds(codes, seed=2))
    assert (p == .7).all()


def test_facing_validation_uses_the_verified_audit_convention():
    ids = ["a", "b", "c", "d"]
    p = np.asarray([.9, .1, .95, .5])
    pose_labels = np.asarray([1.0, 0.0, np.nan, np.nan])
    reviews = pd.DataFrame({"image_id": ["a", "b", "c", "x"], "flank": ["right", "left", "right", "left"]})
    result = fg.facing_validation(p, pose_labels, ids, reviews)
    assert result["pose_accuracy"] == 1 and result["audit_accuracy"] == 1 and result["audit_images"] == 3
    assert result["audit_images_without_pose"] == 1
    assert result["confident_share_all"] == .75


def test_one_to_one_matching_refuses_a_chain_through_a_shared_trap_day():
    # Images: A-left cluster 0 (side 0), A-right cluster 0 (side 1), B-left 1, B-right 1.
    side_of = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    cluster_of = np.asarray([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    keys = ["d1", "d1", "d2", "d2", "d3", "d3", "d4", "d4", "d5", None]
    together = [None] * 10
    together[0] = together[7] = "d9"  # A-left and B-right once on the same trap-day.
    counts = fg.cooccurrence(cluster_of, side_of, [keys, together])
    assert counts[(0, 0)] == 3 and counts[(1, 1)] == 1 and counts[(0, 1)] == 1
    matched = fg.match_one_to_one(counts, 1)
    assert sorted((a, b) for a, b, _ in matched) == [(0, 0), (1, 1)]
    assert [(a, b) for a, b, _ in fg.match_one_to_one(counts, 2)] == [(0, 0)]
    with pytest.raises(ValueError):
        fg.match_one_to_one(counts, 0)


def test_bilateral_bounds_and_label_combination():
    bounds = fg.bilateral_bounds({0: 5, 1: 4, 2: 1}, {0: 3, 1: 2, 2: 2, 3: 7}, [(0, 0, 4), (2, 3, 1)])
    # Left clusters >= 2 images: {0, 1}; right: {0, 1, 2, 3}; (2, 3) involves a singleton left cluster.
    assert bounds["paired"] == 1 and bounds["unmatched_left"] == 1 and bounds["unmatched_right"] == 3
    assert (bounds["lower"], bounds["upper"]) == (4, 5)
    side_of = np.asarray([0, 1, 0, 1, 0])
    cluster_of = np.asarray([0, 0, 1, 1, -1])
    labels = fg.combine_labels(5, side_of, cluster_of, [(0, 0, 2)])
    assert labels[0] == labels[1] and len({labels[0], labels[2], labels[3]}) == 3 and labels[4] == -1


def test_flank_unit_evaluation_rewards_correct_flank_clusters():
    frame = pd.DataFrame({"image_id": list("abcd"), "identity": ["A", "A", "A", "A"], "path": "x",
                          "source": "s", "trap_id": "t", "encounter_id": list("abcd")})
    result = fg.evaluate_flank_units(list("abcd"), np.asarray([0, 0, 0, 0]), np.asarray([0, 0, 1, 1]), frame, {})
    assert result["flank_units"] == 2 and result["flank_exact_recoveries"] == 2
    assert result["flank_pairwise_precision"] == 1


def synthetic_facing(monkeypatch):
    def fake_pose(raw, frame):
        return np.asarray([float(i % 2) if i % 3 else np.nan for i in range(len(frame))])

    def fake_probabilities(vectors, labels, folds, **kwargs):
        return np.asarray([.95 if i % 2 else .05 for i in range(len(vectors))])
    monkeypatch.setattr(cli, "load_pose_facing", fake_pose)
    monkeypatch.setattr(cli, "facing_probabilities", fake_probabilities)


def write_audit(directory, ids):
    directory.mkdir(parents=True, exist_ok=True)
    reviews = [{"image_id": rid, "flank": "right" if i % 2 else "left", "usable_markings": "yes",
                "animal_visibility": "clear", "crop_mask": "correct", "illumination": "colour"}
               for i, rid in enumerate(ids)]
    path = directory / "reviews.json"
    path.write_text(json.dumps(reviews))
    write_json(directory / "receipt.json", {"status": "complete", "fingerprint": "f",
        "artifacts": {"reviews.json": file_digest(path)}, "sample": []})
    return path


def snapshot(directory):
    return {str(p.relative_to(directory)): file_digest(p) for p in Path(directory).rglob("*") if p.is_file()}


def test_cli_end_to_end_is_model_free_and_preserves_sources(tmp_path, monkeypatch):
    fixture, calibration = run_small_calibration(tmp_path, monkeypatch)
    synthetic_facing(monkeypatch)
    ids = pd.read_csv(calibration / "collection.csv", dtype=str).image_id.sort_values().tolist()
    audit = write_audit(tmp_path / "audit", ids[:6])
    raw = tmp_path / "raw" / "raw.csv"
    raw.parent.mkdir()
    raw.write_text("path,pose\n")
    before = snapshot(fixture["cache_dir"]), snapshot(calibration)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "timm", "transformers", "lightglue", "romatch"}:
            raise AssertionError(f"Neural import attempted: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    out = tmp_path / "flank"
    args = ["--calibration-dir", str(calibration), "--raw-metadata", str(raw), "--audit-reviews", str(audit),
            "--out-dir", str(out)]
    assert cli.main(args) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["partition"] == "calibration"
    assert (snapshot(fixture["cache_dir"]), snapshot(calibration)) == before
    for name, digest in manifest["artifacts"].items():
        assert file_digest(out / name) == digest
    profiles = pd.read_csv(out / "profiles.csv")
    expected = (len(fg.eg.ARMS) * len(fg.BUDGETS) * len(fg.AMBIGUITY_POLICIES)
                * len(fg.WITHIN_PROFILES) * len(fg.MIN_COOCCURRENCE))
    assert len(profiles) == expected
    assert (profiles.count_lower <= profiles.count_upper).all()
    summary = json.loads((out / "summary.json").read_text())
    assert {"facing_classifier_usable", "flank_level_progress", "animal_level_progress",
            "count_candidate"} <= set(summary["declared_rules"])
    facing = pd.read_csv(out / "facing.csv")
    assert len(facing) == len(ids) and set(facing.predicted) == {"faces_left", "faces_right"}
    snap = snapshot(out)
    assert cli.main(args) == 1
    assert snapshot(out) == snap


def test_selection_is_refused(tmp_path):
    directory = tmp_path / "selection"
    write_json(directory / "manifest.json", {"partition": "selection"})
    assert cli.main(["--calibration-dir", str(directory), "--raw-metadata", "x", "--audit-reviews", "y",
                     "--out-dir", str(tmp_path / "out")]) == 1
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("extra", [["--threads", "0"], ["--seed", "-1"], ["--partition", "selection"]])
def test_cli_rejects_invalid_settings(extra):
    with pytest.raises(SystemExit):
        cli.main(["--calibration-dir", "u", "--raw-metadata", "u", "--audit-reviews", "u", "--out-dir", "u", *extra])


def test_tool_import_does_not_import_neural_dependencies():
    code = ("import sys; import tools.evaluate_czechlynx_flank_grouping; "
            "assert not {'torch', 'timm', 'transformers', 'lightglue', 'romatch'} & set(sys.modules)")
    completed = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
