"""Behavioural regressions for candidate-coverage and identifiability diagnostics."""
import builtins
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from reid_demo.benchmark import file_digest, write_json
from reid_demo.benchmark_clustering import candidate_neighbours, inference_inputs
from reid_demo.benchmark_cluster_diagnostics import prepare_evaluation
from reid_demo.benchmark_coverage_diagnostics import (
    abstention_curve, audit_crosstabs, candidate_variants, encounter_analysis, identifiability_risk,
    image_quality_features, knn_edges, load_audit, oracle_components, risk_auc, score_matrix,
)
import tools.diagnose_czechlynx_coverage as cli
from test_benchmark_clustering import evaluation_frame, run_small_calibration


def vectors_at_angles(angles):
    angles = np.deg2rad(angles)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def inputs_for(ids, vectors, identities, **frame_options):
    return prepare_evaluation(ids, vectors, evaluation_frame(ids, identities, **frame_options))


def snapshot(directory):
    return {str(p.relative_to(directory)): file_digest(p) for p in Path(directory).rglob("*") if p.is_file()}


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_knn_builder_matches_the_calibration_builder_including_ties_and_zero_vectors(k):
    ids = [f"img{i}" for i in range(7)]
    vectors = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1], [1, 1], [0, 0], [-1, 0]], dtype=float)
    reference = candidate_neighbours(ids, vectors, [k])[k]
    _, x, valid = inference_inputs(ids, vectors)
    assert knn_edges(score_matrix(x), valid, k) == reference


def test_mutual_knn_keeps_only_reciprocal_links():
    # c's nearest neighbour is b, but b's nearest neighbour is a.
    _, x, valid = inference_inputs(list("abc"), vectors_at_angles([0, 10, 30]))
    scores = score_matrix(x)
    union = {(a, b) for a, b, _ in knn_edges(scores, valid, 1)}
    mutual = {(a, b) for a, b, _ in knn_edges(scores, valid, 1, mutual=True)}
    assert union == {(0, 1), (1, 2)}
    assert mutual == {(0, 1)}


def test_candidate_variants_build_the_same_edges_whatever_the_identities():
    ids = [f"img{i:02d}" for i in range(12)]
    vectors = vectors_at_angles(np.arange(12) * 25.0)
    first = inputs_for(ids, vectors, list("AAABBBCCCDDD"))
    second = inputs_for(ids, vectors, list("ABCDABCDABCD"))
    edges = lambda inputs: [(r["variant"], r["edges"]) for r in candidate_variants(
        score_matrix(inputs.vectors), inputs, budgets=(2, 3), qe_neighbours=(2,))[0]]
    assert edges(first) == edges(second)
    assert [name for name, _ in edges(first)] == [
        "knn_k2", "knn_k3", "mutual_knn_k3", "query_expansion_m2_knn_k3"]


def test_oracle_components_handle_retained_subsets():
    ids = list("abcde")
    inputs = inputs_for(ids, vectors_at_angles([0, 5, 90, 95, 180]), list("AABBC"))
    keep = np.asarray([True, True, True, False, False])
    edges = [(0, 1, 1.0), (2, 3, 1.0), (1, 2, 1.0)]  # The A-B edge is removed by the oracle.
    _, degree, summary = oracle_components(inputs.identity_codes, keep, edges)
    assert summary["retained_images"] == 3
    assert summary["identities_without_retained_images"] == 1  # C was fully abstained.
    assert summary["fully_connected_identities"] == 2  # A, and B with a single retained image.
    assert summary["fully_connected_multi_image_identities"] == 1
    assert summary["oracle_fragments"] == 2
    assert degree.tolist() == [1, 1, 0, 0, 0]


def test_abstention_drops_highest_risk_first_and_random_control_is_reproducible():
    ids = [f"img{i}" for i in range(8)]
    # Animal A's fourth photo looks like B; dropping it lets A connect at k=1.
    vectors = vectors_at_angles([0, 3, 6, 120, 200, 203, 206, 209])
    inputs = inputs_for(ids, vectors, list("AAAABBBB"))
    scores = score_matrix(inputs.vectors)
    risk = np.asarray([.1, .1, .1, .9, .2, .2, .2, .2])
    rows = abstention_curve(scores, inputs, risk, budget=1, fractions=(0, .125), repeats=3, seed=7)
    baseline = next(r for r in rows if r["fraction"] == 0)
    gated = next(r for r in rows if r["fraction"] == .125 and r["policy"] == "risk")
    assert baseline["fully_connected_multi_image_identities"] < gated["fully_connected_multi_image_identities"]
    assert gated["retained_images"] == 7 and gated["dropped_images"] == 1
    assert gated["fully_connected_multi_image_identities"] == 2
    random_rows = [r for r in rows if r["policy"] == "random"]
    assert len(random_rows) == 3
    again = abstention_curve(scores, inputs, risk, budget=1, fractions=(0, .125), repeats=3, seed=7)
    assert again == rows
    with pytest.raises(ValueError):
        abstention_curve(scores, inputs, risk, fractions=(1.0,))


def test_encounter_links_are_label_free_and_mixed_encounters_are_reported():
    ids = list("abcdef")
    frame = pd.DataFrame({"image_id": ids, "identity": list("AABBCC"), "path": "unused",
                          "encounter_id": ["e1", "e1", "e2", "e2", "e3", "e3"],
                          "source": "test", "trap_id": "cam"})
    frame.loc[3, "identity"] = "C"  # e2 now holds B and C.
    vectors = vectors_at_angles([0, 90, 180, 270, 45, 135])
    inputs = prepare_evaluation(ids, vectors, frame)
    result = encounter_analysis(score_matrix(inputs.vectors), inputs, budget=1)
    assert result["within_encounter_pairs"] == 3
    assert result["within_encounter_pairs_joining_different_identities"] == 1
    assert result["mixed_identity_encounters"] == 1
    assert result["fully_connected_identity_gain"] >= 1  # e1 joins A's two dissimilar photos.


def write_crop(directory, name, pixels, mask):
    path, mask_path = directory / f"{name}.png", directory / f"{name}_mask.png"
    Image.fromarray(pixels.astype(np.uint8)).save(path)
    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
    return {"image_id": name, "path": str(path), "mask_path": str(mask_path), "mask_sha256": file_digest(mask_path)}


def test_quality_features_detect_fragmented_truncated_and_washed_out_crops(tmp_path):
    rng = np.random.default_rng(0)
    textured = rng.integers(60, 190, (40, 40, 3))
    checker = (np.indices((40, 40)).sum(axis=0) % 2 * 255)[..., None].repeat(3, axis=2)
    centre = np.zeros((40, 40), bool)
    centre[8:32, 8:32] = True
    split = np.zeros((40, 40), bool)
    split[8:18, 8:32] = split[24:32, 8:32] = True
    edge = np.zeros((40, 40), bool)
    edge[0:30, 0:30] = True
    rows = [write_crop(tmp_path, "clean", checker, centre),
            write_crop(tmp_path, "fragmented", textured, split),
            write_crop(tmp_path, "truncated", textured, edge),
            write_crop(tmp_path, "washed_out", np.full((40, 40, 3), 240), centre)]
    features = image_quality_features(pd.DataFrame(rows)).set_index("image_id")
    assert features.loc["fragmented", "mask_components"] == 2
    assert features.loc["clean", "border_mask_fraction"] == 0
    assert features.loc["truncated", "border_mask_fraction"] > 0
    assert features.loc["washed_out", "luminance_std"] == 0
    risk, ranks = identifiability_risk(features.reset_index())
    assert list(ranks.columns) == ["small_animal", "low_contrast", "fragmented_mask", "truncated"]
    assert int(np.argmin(risk)) == 0  # The clean crop is least risky.


def test_changed_masks_are_rejected(tmp_path):
    row = write_crop(tmp_path, "a", np.zeros((8, 8, 3)), np.ones((8, 8)))
    row["mask_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Mask changed"):
        image_quality_features(pd.DataFrame([row]))


def test_bbox_inputs_use_the_whole_crop(tmp_path):
    path = tmp_path / "a.png"
    Image.fromarray(np.zeros((4, 5, 3), np.uint8)).save(path)
    features = image_quality_features(pd.DataFrame([{"image_id": "a", "path": str(path)}]))
    assert not features.mask_available[0] and features.mask_area_fraction[0] == 1


def test_risk_auc_edges():
    assert risk_auc([.1, .2, .8, .9], [False, False, True, True]) == 1
    assert risk_auc([.9, .8, .2, .1], [False, False, True, True]) == 0
    assert risk_auc([.5, .5], [True, False]) == .5
    assert risk_auc([.1, .2], [True, True]) is None


def write_audit(directory, reviews, sample=None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "reviews.json"
    path.write_text(json.dumps(reviews))
    write_json(directory / "receipt.json", {"status": "complete", "fingerprint": "f",
        "artifacts": {"reviews.json": file_digest(path)},
        "sample": sample or [{"image_id": r["image_id"], "cohorts": ["calibration_uniform"]} for r in reviews]})
    return path


def review(image_id, flank, usable="yes"):
    return {"image_id": image_id, "flank": flank, "usable_markings": usable, "animal_visibility": "clear",
            "crop_mask": "correct", "illumination": "colour"}


def test_audit_is_bound_to_its_receipt(tmp_path):
    path = write_audit(tmp_path / "audit", [review("a", "left")])
    reviews, binding = load_audit(path)
    assert reviews.cohorts.tolist() == ["calibration_uniform"]
    assert binding["reviews_sha256"] == file_digest(path)
    path.write_text(json.dumps([review("a", "right")]))
    with pytest.raises(ValueError, match="receipt"):
        load_audit(path)


def test_flank_pairs_separate_same_opposite_and_unknown(tmp_path):
    ids = list("abcde")
    inputs = inputs_for(ids, vectors_at_angles([0, 5, 60, 90, 95]), list("AAAAB"))
    images = pd.DataFrame({"image_id": ids, "identity": list("AAAAB"), "has_correct_candidate_neighbour": True,
                           "in_largest_fragment": True, "identifiability_risk": [.1, .2, .3, .05, .5]})
    reviews, _ = load_audit(write_audit(tmp_path / "audit", [
        review("a", "left"), review("b", "left"), review("c", "right", "no"), review("d", "unknown"),
        review("outside", "left")]))
    result = audit_crosstabs(images, reviews, score_matrix(inputs.vectors), inputs, budget=1)
    assert result["audited_calibration_images"] == 4
    assert result["audit_images_outside_calibration"] == 1
    summary = {r["relation"]: r for r in result["flank_summary"]}
    assert summary["same_flank"]["pairs"] == 1  # a-b
    assert summary["opposite_flank"]["pairs"] == 2  # a-c, b-c
    assert summary["unknown"]["pairs"] == 3  # Pairs involving d.
    assert summary["same_flank"]["median_cosine"] > summary["opposite_flank"]["median_cosine"]
    assert result["risk_auc"]["usable_markings"]["auc"] == 1  # c has the highest risk among yes/no images.


def test_cli_is_model_free_preserves_sources_and_writes_checksummed_evidence(tmp_path, monkeypatch):
    fixture, calibration = run_small_calibration(tmp_path, monkeypatch)
    collection = pd.read_csv(calibration / "collection.csv", dtype=str)
    chosen = collection.sort_values("image_id").groupby("identity").head(2).head(6)
    flanks = ["left", "right", "left", "left", "unknown", "right"]
    audit = write_audit(tmp_path / "audit", [review(r, f) for r, f in zip(chosen.image_id, flanks)])
    cache_before, calibration_before = snapshot(fixture["cache_dir"]), snapshot(calibration)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "timm", "transformers", "lightglue", "romatch"}:
            raise AssertionError(f"Neural import attempted: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    out = tmp_path / "coverage"
    args = ["--calibration-dir", str(calibration), "--audit-reviews", str(audit), "--out-dir", str(out)]
    assert cli.main(args) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["partition"] == "calibration"
    assert all(c["identical_pairs"] for c in manifest["candidate_self_check"].values())
    assert manifest["source_cache_sha256_after"] == manifest["run_spec"]["cache_sha256"]
    assert snapshot(fixture["cache_dir"]) == cache_before
    assert snapshot(calibration) == calibration_before
    for name, digest in manifest["artifacts"].items():
        assert file_digest(out / name) == digest
    for name in ("summary.json", "image_fragments.csv", "fragments.csv", "abstention_curve.csv",
                 "abstention_runs.csv", "candidate_variants.csv", "audit_crosstabs.csv",
                 "audit_flank_pairs.csv", "abstention_curve.png", "index.html"):
        assert name in manifest["artifacts"]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["audit"]["audited_calibration_images"] == 6
    assert set(summary["declared_rules"]) >= {"H1_abstention", "H2_flanks", "H3_encounters", "H4_candidates"}
    before = snapshot(out)
    assert cli.main(args) == 1
    assert snapshot(out) == before  # Existing evidence cannot be overwritten.


def test_cli_rejects_an_audit_that_differs_from_its_receipt(tmp_path, monkeypatch):
    _, calibration = run_small_calibration(tmp_path, monkeypatch)
    audit = write_audit(tmp_path / "audit", [review("x", "left")])
    audit.write_text("[]")
    out = tmp_path / "coverage"
    assert cli.main(["--calibration-dir", str(calibration), "--audit-reviews", str(audit),
                     "--out-dir", str(out)]) == 1
    assert not out.exists()


@pytest.mark.parametrize("extra", [["--partition", "selection"], ["--threads", "0"], ["--seed", "-1"]])
def test_cli_rejects_selection_and_invalid_settings(extra):
    with pytest.raises(SystemExit):
        cli.main(["--calibration-dir", "unused", "--audit-reviews", "unused", "--out-dir", "unused", *extra])


def test_tool_import_does_not_import_neural_dependencies():
    code = ("import sys; import tools.diagnose_czechlynx_coverage; "
            "assert not {'torch', 'timm', 'transformers', 'lightglue', 'romatch'} & set(sys.modules)")
    completed = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
