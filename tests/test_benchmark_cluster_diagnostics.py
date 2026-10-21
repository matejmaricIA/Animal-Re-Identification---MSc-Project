"""Behavioural regressions for calibration pair and oracle connectivity evidence."""
import builtins
import copy
import csv
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from reid_demo.benchmark import file_digest, write_json
from reid_demo.benchmark_clustering import candidate_neighbours, load_frozen_vectors
from reid_demo.benchmark_cluster_diagnostics import (
    all_pair_diagnostics, candidate_diagnostics, check_diagnostic_output, load_calibration,
    oracle_connectivity, pair_review_rows, prepare_evaluation, verify_candidates,
)
import tools.diagnose_czechlynx_clusters as cli
import tools.evaluate_czechlynx_clusters as calibration_cli
from test_benchmark_clustering import evaluation_frame, make_fixture, parameters, run_small_calibration


def vectors_at_angles(angles):
    angles = np.deg2rad(angles)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def evaluate(ids, vectors, identities, thresholds=(.93,), *, k=3, frame=None, appearance=None):
    frame = frame if frame is not None else evaluation_frame(ids, identities)
    candidates = candidate_neighbours(ids, vectors, [k], chunk_size=2)
    inputs = prepare_evaluation(ids, vectors, frame, appearance)
    accumulators, low_pairs = all_pair_diagnostics(inputs, thresholds, chunk_size=2, examples_per_kind=2)
    result = candidate_diagnostics(inputs, candidates, {k: list(thresholds)}, accumulators, examples_per_kind=2)
    return inputs, accumulators, low_pairs, result


def overall(accumulators, threshold_index=0):
    return accumulators["all"].threshold_rows("all")[threshold_index]


def test_false_bridge_has_good_direct_edges_and_bad_transitive_component_pairs():
    ids = list("abcd")
    vectors = vectors_at_angles([-20, -10, 10, 20])
    inputs, pairs, _, result = evaluate(ids, vectors, ["A", "A", "B", "B"])
    direct = next(r for r in result["candidate_thresholds"] if r["group"] == "all")
    assert direct["accepted_same_edges"] == 2
    assert direct["accepted_different_edges"] == 1
    assert direct["direct_edge_precision"] == pytest.approx(2 / 3)
    assert overall(pairs)["recall"] == 1
    filtered = next(r for r in result["oracle_connectivity"] if r["threshold"] == .93)
    assert filtered["fully_connected_identities"] == 2
    assert filtered["oracle_fragments"] == 2
    source = {"reference": {"parameters": parameters(k=3), "metrics": {
        "pairwise_precision": 1 / 3, "pairwise_recall": 1, "provisional_component_count": 1,
        "merge_errors": 1, "split_errors": 0}}}
    candidates = candidate_neighbours(ids, vectors, [3])
    compared = candidate_diagnostics(inputs, candidates, {3: [.93]}, pairs, source_profiles=source)
    row = compared["component_comparisons"][0]
    assert row["direct_edge_precision"] > row["component_pair_precision"]
    assert row["admitted_candidate_edges"] == 3


def test_perfect_filter_cannot_connect_an_identity_with_no_correct_candidates():
    ids = list("abcd")
    _, _, _, result = evaluate(ids, vectors_at_angles([0, 5, 180, 185]), ["A", "B", "A", "B"], k=1)
    row = next(r for r in result["oracle_connectivity"] if r["threshold"] is None)
    assert row["fully_connected_identities"] == 0
    assert row["oracle_fragments"] == 4
    assert row["multi_image_identity_valid_images_without_correct_neighbour"] == 4
    assert len(result["disconnected_pairs"]) == 2


def test_strict_threshold_can_fragment_adequate_unthresholded_candidates():
    _, _, _, result = evaluate(list("abc"), vectors_at_angles([0, 60, 180]), ["A", "A", "B"], (.9,), k=2)
    loose, strict = result["oracle_connectivity"]
    assert loose["threshold"] is None
    assert loose["fully_connected_multi_image_identities"] == 1
    assert loose["oracle_fragments"] == 2
    assert strict["fully_connected_multi_image_identities"] == 0
    assert strict["oracle_fragments"] == 3


def test_zero_vectors_are_unscorable_and_stay_in_positive_recall_denominators():
    inputs, pairs, _, result = evaluate(list("abc"), [[1, 0], [0, 0], [0, 1]], ["A", "A", "B"], (.5,), k=2)
    row = overall(pairs)
    assert row["population_pairs"] == 3
    assert row["scored_pairs"] == 1
    assert row["unscorable_pairs"] == 2
    assert row["unscorable_same_pairs"] == 1
    assert row["precision"] is None
    assert row["recall"] == 0
    assert row["macro_identity_positive_recall"] == 0
    assert pairs["all"].hist_same.sum() == 0
    assert pairs["all"].hist_different.sum() == 1
    for row in result["oracle_connectivity"]:
        assert row["unresolved_images"] == 1
        assert row["complete_oracle_fragment_total"] is None
        assert row["oracle_fragments"] == 2
    oracle, rows, labels = oracle_connectivity(inputs, [])
    assert labels[1] == -1
    assert next(r for r in rows if r["identity"] == "A")["fully_connected"] is False
    assert oracle["fully_connected_identities"] == 1  # The valid, single-photo B.


def test_all_zero_vectors_do_not_produce_a_completed_oracle_total():
    _, pairs, _, result = evaluate(list("ab"), np.zeros((2, 2)), ["A", "A"], (.5,), k=1)
    assert overall(pairs)["recall"] == 0
    assert overall(pairs)["precision"] is None
    assert result["oracle_connectivity"][0]["oracle_fragments"] == 0
    assert result["oracle_connectivity"][0]["complete_oracle_fragment_total"] is None
    assert result["oracle_connectivity"][0]["fully_connected_identities"] == 0


def test_macro_recall_exposes_identity_size_imbalance_and_histogram_endpoints():
    ids = list("abcdef")
    vectors = [[1, 0], [1, 0], [1, 0], [0, 1], [0, -1], [1, 1]]
    _, pairs, _, _ = evaluate(ids, vectors, ["A", "A", "A", "B", "B", "singleton"], (.9,), k=2)
    row = overall(pairs)
    assert row["true_same_pairs"] == 4
    assert row["recall"] == .75
    assert row["macro_identity_positive_recall"] == .5
    assert row["macro_recall_eligible_identities"] == 2
    assert pairs["all"].hist_same[0] == 1  # Cosine -1 is in the first bin.
    assert pairs["all"].hist_same[-1] == 3  # Cosine +1 is in the last bin.
    np.testing.assert_array_equal(pairs["all"].identity_histogram.sum(axis=1), [3, 1, 0])


def test_subgroups_keep_missing_metadata_and_do_not_impose_encounter_links():
    ids = list("abcd")
    frame = evaluation_frame(ids, ["A", "A", "B", "B"])
    frame["encounter_id"] = ["one", "two", "one", None]
    frame["trap_id"] = ["one", "two", "one", None]
    appearance = {"a": {"near_monochrome": False}, "b": {"near_monochrome": True},
                  "c": {"near_monochrome": False}}
    _, pairs, _, result = evaluate(ids, np.eye(4), frame.identity.tolist(), (0.0,), frame=frame, appearance=appearance)
    cross = pairs["across_encounter"].threshold_rows("across_encounter")[0]
    assert cross["population_pairs"] == 2
    assert cross["true_same_pairs"] == 1
    assert cross["precision"] == .5
    assert cross["recall"] == 1
    within = pairs["within_encounter"].threshold_rows("within_encounter")[0]
    assert within["true_same_pairs"] == 0
    assert within["recall"] is None
    assert within["macro_identity_positive_recall"] is None
    assert within["macro_recall_eligible_identities"] == 0
    assert pairs["missing_encounter"].population_pairs == 3
    assert pairs["missing_camera"].population_pairs == 3
    assert pairs["missing_appearance_proxy"].population_pairs == 3
    assert pairs["colour_colour"].population_pairs == 1
    assert pairs["across_appearance_proxy"].positive_pairs == 1
    # A and B share encounter 'one'; oracle filtering keeps them different.
    assert result["oracle_connectivity"][0]["oracle_fragments"] == 2


def test_single_photo_identity_has_undefined_pair_metrics_and_valid_oracle_singleton():
    _, pairs, low, result = evaluate(["a"], [[1, 0]], ["A"], (.5,), k=1)
    assert overall(pairs)["precision"] is None
    assert overall(pairs)["recall"] is None
    assert overall(pairs)["macro_identity_positive_recall"] is None
    assert low == []
    assert result["oracle_connectivity"][0]["fully_connected_identities"] == 1
    assert result["oracle_connectivity"][0]["multi_image_identities"] == 0
    assert result["oracle_connectivity"][0]["fraction_fully_connected_multi_image_identities"] is None


def test_statistics_examples_and_connectivity_are_row_and_chunk_invariant():
    ids = list("abcdef")
    vectors = np.array([[1, 0], [1, 0], [0, 1], [-1, 0], [0, 1], [0, 0]])
    frame = evaluation_frame(ids, ["A", "A", "B", "A", "B", "B"])
    appearance = {rid: {"near_monochrome": bool(i % 2)} for i, rid in enumerate(ids)}
    one = prepare_evaluation(ids, vectors, frame, appearance)
    two = prepare_evaluation(ids[::-1], vectors[::-1], frame.iloc[::-1], appearance)
    a, low_a = all_pair_diagnostics(one, [.1, .9], chunk_size=1, examples_per_kind=3)
    b, low_b = all_pair_diagnostics(two, [.1, .9], chunk_size=4, examples_per_kind=3)
    assert low_a == low_b
    for group in a:
        assert a[group].threshold_rows(group) == b[group].threshold_rows(group)
        np.testing.assert_array_equal(a[group].hist_same, b[group].hist_same)
        np.testing.assert_array_equal(a[group].hist_different, b[group].hist_different)
        np.testing.assert_array_equal(a[group].identity_score_sum, b[group].identity_score_sum)
    first = candidate_neighbours(ids, vectors, [1, 3], chunk_size=1)
    second = candidate_neighbours(ids[::-1], vectors[::-1], [1, 3], chunk_size=4)
    assert first == second
    grid = {1: [.1, .9], 3: [.1, .9]}
    r1 = candidate_diagnostics(one, first, grid, a, examples_per_kind=3)
    r2 = candidate_diagnostics(two, second, grid, b, examples_per_kind=3)
    assert r1 == r2
    assert pair_review_rows(one, low_a, r1) == pair_review_rows(two, low_b, r2)


@pytest.mark.parametrize("thresholds", [[], [.5, .5], [True], [np.nan], [-.1], [1.1]])
def test_invalid_thresholds_are_rejected(thresholds):
    inputs = prepare_evaluation(["a"], [[1, 0]], evaluation_frame(["a"], ["A"]))
    with pytest.raises(ValueError):
        all_pair_diagnostics(inputs, thresholds)


def test_evaluation_rejects_missing_truth_and_invented_appearance_labels():
    frame = evaluation_frame(list("ab"), ["A", "A"])
    with pytest.raises(ValueError, match="exactly"):
        prepare_evaluation(list("ab"), np.eye(2), frame.iloc[:1])
    frame.loc[0, "identity"] = None
    with pytest.raises(ValueError, match="identities"):
        prepare_evaluation(list("ab"), np.eye(2), frame)
    with pytest.raises(ValueError, match="colour proxies"):
        prepare_evaluation(["a"], [[1, 0]], evaluation_frame(["a"], ["A"]), {"a": {"near_monochrome": "night"}})


@pytest.mark.parametrize("edges", [[(0, 1, .5)], [(0, 4, .5)], [(2, 0, .5)],
                                   [(0, 2, np.nan)], [(0, 2, .5), (0, 2, .5)]])
def test_malformed_or_zero_vector_candidate_edges_are_rejected(edges):
    inputs = prepare_evaluation(list("abc"), [[1, 0], [0, 0], [0, 1]],
                                evaluation_frame(list("abc"), ["A", "A", "B"]))
    pairs, _ = all_pair_diagnostics(inputs, [.5])
    with pytest.raises(ValueError, match="candidate edge"):
        candidate_diagnostics(inputs, {1: edges}, {1: [.5]}, pairs)


def snapshot(directory):
    return {str(p.relative_to(directory)): file_digest(p) for p in Path(directory).rglob("*") if p.is_file()}


def test_model_free_cli_preserves_cache_and_calibration_and_writes_checksummed_evidence(tmp_path, monkeypatch):
    fixture, calibration = run_small_calibration(tmp_path, monkeypatch)
    cache_before, calibration_before = snapshot(fixture["cache_dir"]), snapshot(calibration)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "timm", "transformers", "lightglue", "romatch"}:
            raise AssertionError(f"Neural import attempted: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    out = tmp_path / "diagnostics"
    assert cli.main(["--calibration-dir", str(calibration), "--out-dir", str(out), "--chunk-size", "3"]) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["partition"] == "calibration"
    assert manifest["candidate_audit"]["saved_candidate_counts_and_retained_score_prefixes_match"]
    assert manifest["source_cache_sha256_after"] == manifest["run_spec"]["cache_sha256"]
    assert manifest["population"]["unscorable_pairs"] == 0
    assert manifest["vector_cache"]["access"] == "sqlite_mode_ro"
    assert snapshot(fixture["cache_dir"]) == cache_before
    assert snapshot(calibration) == calibration_before
    for name, digest in manifest["artifacts"].items():
        assert file_digest(out / name) == digest
    for name in ("summary.json", "pair_thresholds.csv", "pair_histograms.csv", "identity_pair_statistics.csv",
                 "identity_pair_histograms.csv", "candidate_thresholds.csv", "oracle_connectivity.csv",
                 "component_comparisons.csv", "review_pairs.json", "pair_score_histograms.png", "index.html"):
        assert name in manifest["artifacts"]
    summary = json.loads((out / "summary.json").read_text())
    n = summary["images"]
    assert summary["unordered_population_pairs"] == n * (n - 1) // 2
    assert len(summary["oracle_connectivity"]) == 2  # Unthresholded and the one saved cutoff.
    assert any("not achieved accuracy" in note for note in summary["notes"])
    with (out / "identity_pair_statistics.csv").open(newline="") as stream:
        statistics = list(csv.DictReader(stream))
    assert statistics and all(float(row["score_mean"]) == 1 for row in statistics)
    assert all(float(row["positive_recall"]) == 1 for row in statistics)
    before = snapshot(out)
    assert cli.main(["--calibration-dir", str(calibration), "--out-dir", str(out)]) == 1
    assert snapshot(out) == before  # Existing evidence cannot be overwritten.


def test_selection_is_rejected_before_any_source_reconstruction(tmp_path, monkeypatch):
    directory = tmp_path / "selection"
    write_json(directory / "manifest.json", {"partition": "selection"})
    import reid_demo.benchmark_cluster_diagnostics as diagnostic

    def forbidden(*args, **kwargs):
        raise AssertionError("Selection source was accessed")

    monkeypatch.setattr(diagnostic, "load_collection", forbidden)
    with pytest.raises(ValueError, match="selection is forbidden"):
        load_calibration(directory)
    out = tmp_path / "out"
    assert cli.main(["--calibration-dir", str(directory), "--out-dir", str(out)]) == 1
    assert not out.exists()


@pytest.mark.parametrize("name", ["appearance.json", "collection.csv", "profiles.csv"])
def test_changed_calibration_artifacts_are_rejected(tmp_path, monkeypatch, name):
    _, calibration = run_small_calibration(tmp_path, monkeypatch)
    (calibration / name).write_text((calibration / name).read_text() + "tampered")
    with pytest.raises(ValueError, match="artifact"):
        load_calibration(calibration)


@pytest.mark.parametrize("change", ["model", "metadata", "implementation"])
def test_changed_source_contracts_are_rejected(tmp_path, monkeypatch, change):
    fixture, calibration = run_small_calibration(tmp_path, monkeypatch)
    if change == "model":
        path = fixture["references"]["calibration"] / "manifest.json"
        reference = json.loads(path.read_text())
        reference["global_model"]["weights_sha256"] = "b" * 64
        write_json(path, reference)
    elif change == "metadata":
        fixture["metadata"].write_text(fixture["metadata"].read_text() + "\n")
    else:
        import reid_demo.benchmark_clustering as clustering
        monkeypatch.setattr(clustering, "implementation_binding", lambda: {"changed": True})
    with pytest.raises(ValueError):
        load_calibration(calibration)


def test_reconstructed_candidate_counts_and_saved_score_prefixes_are_checked(tmp_path, monkeypatch):
    _, calibration = run_small_calibration(tmp_path, monkeypatch)
    source = load_calibration(calibration)
    matrix, _ = load_frozen_vectors(source.collection.frame, source.collection.reference["global_model"],
                                    source.manifest["run_spec"]["cache_dir"])
    inputs = prepare_evaluation(source.collection.frame.image_id, matrix, source.collection.frame, source.appearance)
    candidates = candidate_neighbours(inputs.image_ids, matrix, list(source.thresholds_by_budget))
    assert verify_candidates(source, inputs, candidates)["saved_candidate_counts_and_retained_score_prefixes_match"]
    budget = next(iter(candidates))
    short = copy.deepcopy(candidates)
    short[budget].pop()
    with pytest.raises(ValueError, match="counts differ"):
        verify_candidates(source, inputs, short)
    altered = {k: [(a, b, float(score * .9)) for a, b, score in edges] for k, edges in candidates.items()}
    with pytest.raises(ValueError, match="scores/IDs differ"):
        verify_candidates(source, inputs, altered)


@pytest.mark.parametrize("kind", ["image", "missing_vector", "vector_digest"])
def test_vector_validation_failure_writes_failed_diagnostic_manifest(tmp_path, monkeypatch, kind):
    fixture, calibration = run_small_calibration(tmp_path, monkeypatch)
    source = load_calibration(calibration)
    rid = source.collection.frame.image_id.iloc[0]
    if kind == "image":
        Path(source.collection.frame.path.iloc[0]).write_bytes(b"changed pixels")
    else:
        conn = sqlite3.connect(fixture["cache"])
        if kind == "missing_vector":
            conn.execute("DELETE FROM vectors WHERE image_id=?", (rid,))
        else:
            conn.execute("UPDATE vectors SET digest=? WHERE image_id=?", ("0" * 64, rid))
        conn.commit()
        conn.close()
    before = snapshot(calibration)
    out = tmp_path / "failed"
    assert cli.main(["--calibration-dir", str(calibration), "--out-dir", str(out)]) == 1
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert rid in manifest["error"]
    assert "artifacts" not in manifest
    assert snapshot(calibration) == before


def test_changed_mask_bytes_are_not_reused(tmp_path, monkeypatch):
    fixture = make_fixture(tmp_path, input_mode="masked")
    monkeypatch.setattr(calibration_cli, "calibration_profiles", lambda seed, *args: [parameters(k=2, threshold=.99)])
    calibration = tmp_path / "calibration"
    assert calibration_cli.main(["--metadata", str(fixture["metadata"]), "--protocol-plan", str(fixture["plan_path"]),
        "--cache-dir", str(fixture["cache_dir"]), "--partition", "calibration",
        "--reference-dir", str(fixture["references"]["calibration"]), "--out-dir", str(calibration)]) == 0
    source = load_calibration(calibration)
    Path(source.collection.frame.mask_path.iloc[0]).write_bytes(b"changed mask")
    out = tmp_path / "diagnostic"
    assert cli.main(["--calibration-dir", str(calibration), "--out-dir", str(out)]) == 1
    failed = json.loads((out / "manifest.json").read_text())
    assert failed["status"] == "failed"
    assert "Mask changed" in failed["error"]


def test_output_cannot_overlap_calibration_cache_images_or_protocol(tmp_path, monkeypatch):
    fixture, calibration = run_small_calibration(tmp_path, monkeypatch)
    source = load_calibration(calibration)
    for path in (calibration / "nested", fixture["cache_dir"] / "nested", tmp_path / "images" / "nested",
                 fixture["metadata"].parent / "nested", tmp_path):
        with pytest.raises(ValueError, match="protected"):
            check_diagnostic_output(path, source)


@pytest.mark.parametrize("extra", [["--partition", "selection"], ["--knn-k", "100"],
                                    ["--cosine-thresholds", ".2"], ["--chunk-size", "0"], ["--threads", "0"]])
def test_cli_cannot_access_selection_or_override_diagnostic_grid(extra):
    with pytest.raises(SystemExit):
        cli.main(["--calibration-dir", "unused", "--out-dir", "unused", *extra])


def test_tool_import_does_not_import_neural_dependencies():
    code = ("import sys; import tools.diagnose_czechlynx_clusters; "
            "assert not {'torch', 'timm', 'transformers', 'lightglue', 'romatch'} & set(sys.modules)")
    completed = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
