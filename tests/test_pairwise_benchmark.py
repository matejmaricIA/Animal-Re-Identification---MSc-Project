import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from reid_demo.benchmark import file_digest, write_json
from reid_demo.benchmark_masks import MASKED_PREPROCESSING
from reid_demo.pairwise_benchmark import (
    EvidenceCache, load_calibration_reference, pair_cache_spec, roc_auc, select_pairs,
    summarize, validate_predictions, visual_pair_ids,
)


def inputs():
    gallery = pd.DataFrame([dict(image_id="ga", identity="a"), dict(image_id="gb", identity="b")])
    query = pd.DataFrame([dict(image_id=q, identity=i, encounter_id=q)
                          for q, i in (("qa", "a"), ("qa2", "a"), ("qb", "b"), ("qu", "u"))])
    predictions = [dict(query_image_id=q, candidate_ids=["ga", "gb"], candidate_scores=[.8, .6])
                   for q in query.image_id]
    groups = {q: ["appearance/colour", "camera/same_camera_available"] for q in query.image_id}
    groups["qu"] = ["appearance/near_monochrome", "camera/unknown_identity"]
    return gallery, query, predictions, groups


def test_pair_selection_is_reproducible_order_independent_and_does_not_inject_positives():
    g, q, predictions, groups = inputs()
    pairs, queries = select_pairs(g, q, predictions, groups, num_queries=4, top_k=1)
    again = select_pairs(g.iloc[::-1], q.iloc[::-1], predictions[::-1], groups, num_queries=4, top_k=1)
    assert (pairs, queries) == again
    assert all(p["gallery_id"] == "ga" for p in pairs)
    qb = next(r for r in queries if r["query_id"] == "qb")
    assert qb["known"] and not qb["shortlist_hit"]
    assert any(not r["known"] for r in queries)


@pytest.mark.parametrize("mutation", ["duplicate_query", "foreign_candidate", "nan", "duplicate_candidate"])
def test_invalid_predictions_rejected(mutation):
    g, q, predictions, _ = inputs()
    if mutation == "duplicate_query":
        predictions[1] = predictions[0]
    elif mutation == "foreign_candidate":
        predictions[0]["candidate_ids"][0] = "selection_image"
    elif mutation == "nan":
        predictions[0]["candidate_scores"][0] = float("nan")
    else:
        predictions[0]["candidate_ids"] = ["ga", "ga"]
    with pytest.raises(ValueError):
        validate_predictions(predictions, g, q)


def test_requested_depth_must_be_available():
    with pytest.raises(ValueError, match="depth"):
        select_pairs(*inputs(), top_k=3)


def reference_fixture(tmp_path, monkeypatch):
    g, q, predictions, groups = inputs()
    frame = pd.concat([g, q], ignore_index=True)
    for column, value in {"path": "image.png", "image_sha256": "image", "mask_path": "mask.png",
                          "mask_sha256": "mask", "source_sha256": "source", "timestamp": "2020-01-01",
                          "input_mode": "masked", "split-time_open": "train"}.items():
        frame[column] = value
    frame["encounter_id"] = frame.image_id
    metadata = tmp_path / "metadata.csv"
    frame.to_csv(metadata, index=False)
    write_json(tmp_path / "manifest.json", {"prepared_metadata_sha256": file_digest(metadata),
                                            "masked_preprocessing": MASKED_PREPROCESSING})
    # Membership reconstruction is covered by test_benchmark_protocol; these
    # mocks isolate reference-role validation, which must happen before inference.
    plan = {"version": "test", "fingerprint": "protocol", "pools": {
        "fitting": {"image_ids": []}, "calibration": {
            "gallery_ids": g.image_id.tolist(), "query_ids": q.image_id.tolist()}}}
    monkeypatch.setattr("reid_demo.pairwise_benchmark.load_protocol", lambda *a: plan)
    monkeypatch.setattr("reid_demo.pairwise_benchmark.protocol_frames", lambda *a: (None, g, q, {}))
    from reid_demo.benchmark_protocol import protocol_binding
    reference = tmp_path / "reference"
    reference.mkdir()
    manifest = {"status": "complete", "partition": "calibration", "stage": "development", "split": "time_open",
                "input_mode": "masked", "metadata_sha256": file_digest(metadata), "protocol": protocol_binding(plan),
                "gallery_ids": g.image_id.tolist(), "query_ids": q.image_id.tolist(),
                "global_model": {"kind": "pretrained_miewid_msv3"}}
    write_json(reference / "manifest.json", manifest)
    write_json(reference / "predictions_global.json", predictions)
    write_json(reference / "diagnostics.json", {"query_measurements": [
        {"image_id": qid, "groups": value} for qid, value in groups.items()]})
    return metadata, tmp_path / "protocol.json", reference, manifest


@pytest.mark.parametrize("field,value", [("partition", "selection"), ("stage", "test"),
                                         ("status", "failed"), ("metadata_sha256", "wrong")])
def test_only_matching_completed_calibration_reference_allowed(tmp_path, monkeypatch, field, value):
    metadata, plan, reference, manifest = reference_fixture(tmp_path, monkeypatch)
    manifest[field] = value
    write_json(reference / "manifest.json", manifest)
    with pytest.raises(ValueError, match="calibration reference"):
        load_calibration_reference(metadata, plan, reference)


def test_reference_loader_accepts_matching_inputs(tmp_path, monkeypatch):
    metadata, plan, reference, _ = reference_fixture(tmp_path, monkeypatch)
    frame, g, q, predictions, groups, provenance = load_calibration_reference(metadata, plan, reference)
    assert len(frame) == 6 and len(predictions) == len(q) == 4
    assert provenance["files"]["predictions_global.json"]


def test_evidence_cache_binds_order_masks_weights_and_settings_and_rejects_corruption(tmp_path):
    cache = EvidenceCache(tmp_path)
    a = dict(image_id="a", image_sha256="a-bytes", mask_sha256="a-mask")
    b = dict(image_id="b", image_sha256="b-bytes", mask_sha256="b-mask")
    spec = pair_cache_spec({"weights": "one"}, a, b, {"budget": 512})
    evidence = {"status": "insufficient_matches", "n_inliers": 0}
    cache.put(spec, evidence)
    assert cache.get(spec) == evidence
    for changed in (pair_cache_spec({"weights": "two"}, a, b, {"budget": 512}),
                    pair_cache_spec({"weights": "one"}, b, a, {"budget": 512}),
                    pair_cache_spec({"weights": "one"}, {**a, "mask_sha256": "new"}, b, {"budget": 512}),
                    pair_cache_spec({"weights": "one"}, a, b, {"budget": 128})):
        assert cache.get(changed) is None
    path = next(tmp_path.glob("*.json"))
    damaged = json.loads(path.read_text())
    damaged["evidence"]["n_inliers"] = 99
    write_json(path, damaged)
    with pytest.raises(ValueError, match="Damaged"):
        cache.get(spec)


def test_runtime_errors_are_not_cached(tmp_path):
    cache = EvidenceCache(tmp_path)
    cache.put({"pair": "a-b"}, {"status": "matcher_error"})
    assert cache.get({"pair": "a-b"}) is None


def test_auc_ties_and_single_class_are_handled():
    assert roc_auc([True, False], [1, 1]) == .5
    assert roc_auc([True, False], [2, 1]) == 1
    assert roc_auc([True, True], [1, 2]) is None


def test_summary_exposes_failures_and_missing_shortlist_matches():
    base = {"matcher": "roma", "query_groups": [], "query_identity": "a", "cached": False, "seconds": 1}
    records = [{**base, "same_identity": True, "status": "matcher_error"},
               {**base, "same_identity": False, "status": "verified", "n_inliers": 10,
                "inlier_fraction": .5, "coverage_min": .2}]
    summary = summarize(records, [{"query_id": "q", "known": True, "shortlist_hit": False}])
    metric = summary["matchers"]["roma"]["groups"]["all"]
    assert metric["status_counts"]["matcher_error"] == 1
    assert metric["scores"]["n_inliers"]["auc_all_pairs_errors_as_zero"] == 0
    assert metric["scores"]["n_inliers"]["auc_successful_pairs"] is None
    assert summary["known_queries_without_match"] == ["q"]


def test_runner_uses_identical_pairs_and_resumes_without_matching(tmp_path, monkeypatch):
    import tools.evaluate_czechlynx_pairs as runner
    monkeypatch.setattr(runner, "check_optional_dependency", lambda: None)
    g, q, predictions, groups = inputs()
    frame = pd.concat([g, q], ignore_index=True)
    image_path, mask_path = tmp_path / "image.png", tmp_path / "mask.png"
    Image.new("RGB", (16, 16), "gray").save(image_path)
    Image.new("L", (16, 16), 255).save(mask_path)
    for column, value in {"path": str(image_path), "image_sha256": file_digest(image_path),
                          "mask_path": str(mask_path), "mask_sha256": file_digest(mask_path)}.items():
        frame[column] = value
    monkeypatch.setattr(runner, "load_calibration_reference", lambda *a: (frame, g, q, predictions, groups, {}))
    calls = {"roma": [], "disk_lightglue": []}
    class FakeMatcher:
        def __init__(self, name):
            self.name, self.spec = name, {"name": name, "weights": "fake"}
        def match(self, row0, row1, *masks):
            calls[self.name].append((row0["image_id"], row1["image_id"]))
            return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
    monkeypatch.setattr(runner, "RomaMatcher", lambda *a, **kw: FakeMatcher("roma"))
    monkeypatch.setattr(runner, "DiskLightGlueMatcher", lambda *a, **kw: FakeMatcher("disk_lightglue"))
    monkeypatch.setitem(sys.modules, "tools.evaluate_czechlynx", SimpleNamespace(
        local_feature_spec=lambda *a: {"method": "disk"},
        LocalFeatures=lambda *a: SimpleNamespace(close=lambda: None)))
    out = tmp_path / "out"
    argv = ["--out-dir", str(out), "--cache-dir", str(tmp_path / "cache"),
            "--num-queries", "2", "--top-k", "2", "--device", "cpu", "--visual-pairs", "2"]
    runner.main(argv)
    assert calls["roma"] == calls["disk_lightglue"] and len(calls["roma"]) == 4
    assert json.loads((out / "manifest.json").read_text())["status"] == "complete"
    assert (out / "index.html").exists() and len(list((out / "matches").glob("*.png"))) == 4
    runner.main(argv)
    assert len(calls["roma"]) == len(calls["disk_lightglue"]) == 4
    assert json.loads((out / "manifest.json").read_text())["cached_pairs"] == 8
    with pytest.raises(ValueError, match="another experiment"):
        runner.main(argv + ["--max-matches", "128"])


def test_visual_pairs_include_both_classes_without_exceeding_limit():
    pairs, _ = select_pairs(*inputs(), num_queries=4, top_k=2)
    chosen = visual_pair_ids(pairs, 2)
    labels = {(p["query_id"], p["gallery_id"]): p["same_identity"] for p in pairs}
    assert len(chosen) == 2 and {labels[key] for key in chosen} == {True, False}
