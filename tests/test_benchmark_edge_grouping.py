"""Behavioural regressions for calibrated edge evidence and grouping (Stage A)."""
import builtins
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from reid_demo.benchmark import file_digest
from reid_demo.benchmark_cluster_diagnostics import prepare_evaluation
from reid_demo.benchmark_coverage_diagnostics import score_matrix
import reid_demo.benchmark_edge_grouping as eg
import tools.evaluate_czechlynx_edge_grouping as cli
from test_benchmark_clustering import run_small_calibration


def vectors_at_angles(angles):
    angles = np.deg2rad(angles)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def frame_for(ids, identities, **columns):
    base = {"image_id": ids, "identity": identities, "path": "unused", "source": "s",
            "trap_id": ["t0"] * len(ids), "timestamp": ["2020-01-01"] * len(ids),
            "encounter_id": ids, "location": ["L"] * len(ids)}
    base.update(columns)
    return pd.DataFrame(base)


def snapshot(directory):
    return {str(p.relative_to(directory)): file_digest(p) for p in Path(directory).rglob("*") if p.is_file()}


def test_metadata_keys_group_by_camera_day_and_location_day_and_skip_missing():
    frame = frame_for(list("abcd"), list("AABB"), trap_id=["t0", "t0", "t1", None],
                      location=["L1", "L1", "L1", "L1"], timestamp=["d1", "d1", "d1", "d1"])
    keys = eg.metadata_keys(frame)
    valid = np.ones(4, bool)
    assert eg.group_pairs(keys["trap_day"], valid) == {(0, 1)}
    assert eg.group_pairs(keys["location_day"], valid) == {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
    assert eg.group_pairs(keys["trap_day"], np.asarray([True, False, True, True])) == set()


def test_derived_arm_links_trap_day_or_location_day():
    ids = list("abcd")
    frame = frame_for(ids, list("ABCD"), trap_id=["t0", "t0", "t1", "t2"],
                      location=[None, None, "L1", "L1"], timestamp=["d1"] * 4)
    inputs = prepare_evaluation(ids, vectors_at_angles([0, 90, 180, 270]), frame)
    scores = score_matrix(inputs.vectors)
    pairs, in_knn = eg.candidate_graph(scores, inputs.valid, eg.metadata_keys(inputs.frame), "derived", 1)
    links = {tuple(p) for p, k in zip(pairs, in_knn) if not k}
    assert {(0, 1), (2, 3)} >= links and len({tuple(p) for p in pairs} & {(0, 1), (2, 3)}) == 2
    with pytest.raises(ValueError):
        eg.candidate_graph(scores, inputs.valid, eg.metadata_keys(inputs.frame), "oracle", 1)


def test_edge_features_never_depend_on_identities():
    ids = [f"img{i:02d}" for i in range(10)]
    vectors = vectors_at_angles(np.arange(10) * 33.0)
    feats = []
    for identities in (list("AAAAABBBBB"), list("ABABABABAB")):
        inputs = prepare_evaluation(ids, vectors, frame_for(ids, identities))
        scores = score_matrix(inputs.vectors)
        keys = eg.metadata_keys(inputs.frame)
        pairs, _ = eg.candidate_graph(scores, inputs.valid, keys, "annotated", 3)
        ranks = eg.rank_matrix(scores, inputs.valid)
        feats.append(eg.edge_features(pairs, scores, ranks, inputs.valid, keys, inputs.appearance_codes,
                                      np.linspace(0, 1, 10), annotated=True, k=3))
    pd.testing.assert_frame_equal(*feats)
    assert set(eg.GRAPH_FEATURES) <= set(feats[0].columns)


def test_rank_matrix_excludes_self_and_zero_vectors():
    scores = score_matrix(np.asarray([[1, 0], [0.9, 0.1], [0, 0], [0, 1]], float) /
                          np.maximum(np.linalg.norm([[1, 0], [0.9, 0.1], [0, 0], [0, 1]], axis=1), 1e-12)[:, None])
    valid = np.asarray([True, True, False, True])
    ranks = eg.rank_matrix(scores, valid)
    assert ranks[0, 1] == 0 and ranks[0, 3] == 1
    assert (ranks[2] == len(scores)).all()  # Invalid rows are never ranked.


def test_cross_fitting_never_scores_an_edge_with_a_model_that_saw_its_animals(monkeypatch):
    import sklearn.pipeline

    class Recorder:
        def fit(self, x, y):
            self.seen = set(x[:, 0].tolist()) | set(x[:, 1].tolist())
            return self

        def predict_proba(self, x):
            assert not (set(x[:, 0].tolist()) | set(x[:, 1].tolist())) & self.seen
            return np.full((len(x), 2), .5)

    monkeypatch.setattr(sklearn.pipeline, "make_pipeline", lambda *a, **k: Recorder())
    rng = np.random.default_rng(0)
    codes = np.repeat(np.arange(10), 4)
    folds = eg.identity_folds(codes, folds=5, seed=3)
    pairs = np.asarray([(a, b) for a in range(40) for b in range(a + 1, 40) if rng.random() < .3])
    features = pd.DataFrame({"code_a": codes[pairs[:, 0]], "code_b": codes[pairs[:, 1]]})
    labels = eg.labels_for(pairs, codes)
    p, models = eg.cross_fit_probabilities(features, labels, pairs, folds, columns=("code_a", "code_b"))
    assert models == 15 and (p == .5).all()


def test_identity_folds_are_deterministic_and_keep_animals_whole():
    codes = np.repeat(np.arange(12), 3)
    first, second = eg.identity_folds(codes, seed=1), eg.identity_folds(codes, seed=1)
    assert (first == second).all()
    for code in range(12):
        assert len(set(first[codes == code])) == 1
    assert set(first) == set(range(5))


def two_animals_with_a_bridge():
    within = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
    cross = [(2, 3), (0, 3), (1, 4), (2, 5)]
    pairs = np.asarray(within + cross)
    p = np.asarray([.95] * 6 + [.95, .05, .05, .05])
    return pairs, p


def test_average_linkage_refuses_a_single_bridge_that_threshold_accepts():
    pairs, p = two_animals_with_a_bridge()
    g1 = eg.run_grouping({"algorithm": "G1_threshold", "tau": .9}, 6, pairs, p)
    assert len(set(g1)) == 1
    for support in (1, 3):
        g2 = eg.run_grouping({"algorithm": "G2_average_linkage", "tau": .9, "support": support}, 6, pairs, p)
        assert len(set(g2)) == 2 and g2[0] == g2[2] and g2[3] == g2[5] and g2[0] != g2[3]


def test_average_linkage_merges_consistently_supported_groups_and_respects_support():
    pairs, p = two_animals_with_a_bridge()
    strong = p.copy()
    strong[6:] = .95
    assert len(set(eg.run_grouping({"algorithm": "G2_average_linkage", "tau": .9, "support": 3},
                                   6, pairs, strong))) == 1
    two = strong.copy()
    two[8:] = .9 - 1e-9  # Mean stays high, but only two cross-edges reach tau.
    labels = eg.run_grouping({"algorithm": "G2_average_linkage", "tau": .5, "support": 3}, 6, pairs, two)
    assert len(set(labels)) == 1  # Under tau=.5 all four count as support.
    labels = eg.group_average_linkage(6, pairs, two, .9, 3)
    assert len(set(labels)) == 2


def test_louvain_separates_weakly_joined_cliques_and_labels_are_contiguous():
    pairs, p = two_animals_with_a_bridge()
    labels = eg.run_grouping({"algorithm": "G3_louvain", "resolution": 1.0}, 6, pairs, p)
    assert sorted(set(labels)) == [0, 1]
    assert labels[0] == labels[1] == labels[2] != labels[3] == labels[4] == labels[5]
    with pytest.raises(ValueError):
        eg.run_grouping({"algorithm": "magic"}, 6, pairs, p)


def test_encounter_link_audit_reports_different_animal_links():
    ids = list("abcdef")
    frame = frame_for(ids, list("AABBCC"), encounter_id=["e1", "e1", "e2", "e2", "e3", "e3"],
                      trap_id=["t0", "t0", "t0", "t1", "t1", "t1"])
    frame.loc[3, "identity"] = "C"
    inputs = prepare_evaluation(ids, vectors_at_angles([0, 90, 180, 270, 45, 135]), frame)
    rows = {r["rule"]: r for r in eg.encounter_link_audit(
        inputs.identity_codes, inputs.valid, eg.metadata_keys(inputs.frame), score_matrix(inputs.vectors), 1)}
    assert rows["annotated"]["linked_pairs"] == 3
    assert rows["annotated"]["pairs_joining_different_identities"] == 1
    assert rows["trap_day"]["linked_pairs"] == 6  # Two trap-day groups of three.


def test_pilot_sample_is_label_free_stratified_and_bounded():
    ids = [f"img{i:03d}" for i in range(200)]
    pairs = np.asarray([(i, i + 1) for i in range(199)])
    cosine = pd.DataFrame({"cosine": np.linspace(-1, 1, 199)})
    kinds = ["knn" if i % 3 else "derived_link" for i in range(199)]
    sample = eg.pilot_sample(ids, pairs, cosine, kinds, size=60, seed=1)
    assert len(sample) == 60 and not sample.duplicated(["image_id_a", "image_id_b"]).any()
    assert set(sample.edge_type) == {"knn", "derived_link"}
    assert sample.equals(eg.pilot_sample(ids, pairs, cosine, kinds, size=60, seed=1))
    assert len(eg.pilot_sample(ids, pairs, cosine, kinds, size=10_000)) == 199


def test_local_features_map_onto_edges_with_coverage():
    ids = ["a", "b", "c"]
    local = pd.DataFrame([{"image_id_a": "a", "image_id_b": "b", "status": "verified", "matches": 30,
                           "confidence_mean": .6, "homography_inliers": 12, "homography_coverage_min": .3,
                           "affine_inliers": 15}])
    features, covered = eg.local_feature_frame(local, ids, np.asarray([(0, 1), (1, 2)]))
    assert covered.tolist() == [True, False]
    assert features.loc[0, "lg_log_matches"] == pytest.approx(np.log1p(30))
    assert features.loc[0, "lg_failed"] == 0 and list(features.columns) == list(eg.LOCAL_FEATURES)


def run_cli(calibration, out, *extra):
    return cli.main(["--calibration-dir", str(calibration), "--out-dir", str(out), *extra])


def test_cli_is_model_free_preserves_sources_and_writes_pilot_and_report(tmp_path, monkeypatch):
    fixture, calibration = run_small_calibration(tmp_path, monkeypatch)
    cache_before, calibration_before = snapshot(fixture["cache_dir"]), snapshot(calibration)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "timm", "transformers", "lightglue", "romatch"}:
            raise AssertionError(f"Neural import attempted: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    out = tmp_path / "edge"
    assert run_cli(calibration, out, "--pilot-edges", "50") == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["partition"] == "calibration"
    assert snapshot(fixture["cache_dir"]) == cache_before and snapshot(calibration) == calibration_before
    for name, digest in manifest["artifacts"].items():
        assert file_digest(out / name) == digest
    summary = json.loads((out / "summary.json").read_text())
    profiles = pd.read_csv(out / "grouping_profiles.csv")
    assert len(profiles) == len(eg.ARMS) * len(eg.BUDGETS) * len(eg.grouping_profiles())
    assert {"edge_model_works", "grouping_progress", "count_candidate",
            "day_level_encounters_deployable"} <= set(summary["declared_rules"])
    pilot = pd.read_csv(out / "pilot_edges.csv")
    assert 0 < len(pilot) <= 50 and set(pilot.columns) >= {"image_id_a", "image_id_b", "edge_type"}
    assert (pilot.image_id_a < pilot.image_id_b).all()
    before = snapshot(out)
    assert run_cli(calibration, out) == 1
    assert snapshot(out) == before


def local_rows(edges, rng):
    return pd.DataFrame([{"image_id_a": a, "image_id_b": b, "status": "verified",
                          "matches": int(rng.integers(0, 50)), "confidence_mean": float(rng.random()),
                          "homography_inliers": int(rng.integers(0, 20)), "homography_coverage_min": float(rng.random()),
                          "affine_inliers": int(rng.integers(0, 20))} for a, b in edges])


def test_cli_gate_mode_and_full_local_mode(tmp_path, monkeypatch):
    _, calibration = run_small_calibration(tmp_path, monkeypatch)
    out = tmp_path / "stage_a"
    assert run_cli(calibration, out, "--pilot-edges", "200") == 0
    pilot = pd.read_csv(out / "pilot_edges.csv", dtype=str)
    rng = np.random.default_rng(0)
    partial = tmp_path / "pilot_local.csv"
    local_rows(zip(pilot.image_id_a, pilot.image_id_b), rng).to_csv(partial, index=False)
    monkeypatch.setattr(cli, "MIN_GATE_EDGES", 20)
    gate = tmp_path / "gate"
    assert run_cli(calibration, gate, "--local-evidence", str(partial)) == 0
    summary = json.loads((gate / "summary.json").read_text())
    assert summary["local_gate"]["rows"] and "met" in summary["local_gate"]
    assert summary["groupings"][0]["profile"] == "gate_mode_no_grouping"
    ids = pd.read_csv(calibration / "collection.csv", dtype=str).image_id.sort_values().tolist()
    everything = tmp_path / "full_local.csv"
    local_rows([(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]], rng).to_csv(everything, index=False)
    full = tmp_path / "full"
    assert run_cli(calibration, full, "--local-evidence", str(everything)) == 0
    summary = json.loads((full / "summary.json").read_text())
    assert any(name.startswith("full_local_") for name in summary["edge_models"])
    assert len(summary["groupings"]) == len(eg.ARMS) * len(eg.BUDGETS) * len(eg.grouping_profiles())


def test_selection_partition_is_refused(tmp_path):
    from reid_demo.benchmark import write_json
    directory = tmp_path / "selection"
    write_json(directory / "manifest.json", {"partition": "selection"})
    assert run_cli(directory, tmp_path / "out") == 1
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("extra", [["--partition", "selection"], ["--threads", "0"], ["--seed", "-1"],
                                    ["--pilot-edges", "0"]])
def test_cli_rejects_invalid_settings(extra):
    with pytest.raises(SystemExit):
        cli.main(["--calibration-dir", "unused", "--out-dir", "unused", *extra])


def test_tool_import_does_not_import_neural_dependencies():
    code = ("import sys; import tools.evaluate_czechlynx_edge_grouping; "
            "assert not {'torch', 'timm', 'transformers', 'lightglue', 'romatch'} & set(sys.modules)")
    completed = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
