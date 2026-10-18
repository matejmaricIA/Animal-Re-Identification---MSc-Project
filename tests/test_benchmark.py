import json

import numpy as np
import pandas as pd
import pytest

from reid_demo.benchmark import (
    clustering_metrics, load_array_cache, make_protocol, partition_overlaps,
    rank_queries, rejection_threshold, retrieval_metrics, sample_encounters,
    save_array_cache, separate_partitions,
)
from tools.evaluate_czechlynx import VectorCache, validate_thresholds


def protocol_frame():
    rows = []
    for identity in range(10):
        for event in range(4):
            for image in range(2):
                rid = f"{identity}-{event}-{image}"
                rows.append(dict(image_id=rid, identity=str(identity), encounter_id=f"{identity}:{event}",
                                 timestamp=f"2020-0{event + 1}-01", image_sha256=rid,
                                 source_sha256=rid, **{"split-time_open": "train" if event < 3 else "test"}))
    return pd.DataFrame(rows)


def test_development_only_uses_official_train_and_whole_encounters():
    frame = protocol_frame()
    gallery, query, audit = make_protocol(frame, split="time_open")
    assert not partition_overlaps(gallery, query)
    assert set(pd.concat([gallery, query])["split-time_open"]) == {"train"}
    assert audit["unknown_query_images"] > 0
    assert set(frame[frame["split-time_open"] == "test"].image_id).isdisjoint(set(gallery.image_id) | set(query.image_id))
    again = make_protocol(frame.sample(frac=1, random_state=3), split="time_open")
    assert set(query.image_id) == set(again[1].image_id)


def test_shared_mixed_animal_encounter_cannot_cross_development_partition():
    frame = protocol_frame()
    frame.loc[frame.encounter_id.isin(["0:2", "1:2"]), "encounter_id"] = "shared"
    gallery, query, _ = make_protocol(frame, split="time_open")
    assert not partition_overlaps(gallery, query)
    assert not ("shared" in set(gallery.encounter_id) and "shared" in set(query.encounter_id))


def test_missing_dates_retained_without_inventing_time():
    frame = protocol_frame()
    frame["timestamp"] = None
    gallery, query, audit = make_protocol(frame, split="time_open")
    assert len(gallery) + len(query) == 60
    assert audit["missing_dates"] == len(frame)


def test_official_overlap_requires_explicit_recorded_exclusion():
    frame = protocol_frame()
    gallery = frame.iloc[:2].copy()
    query = frame.iloc[2:5].copy()
    query.loc[query.index[0], "source_sha256"] = gallery.source_sha256.iloc[0]
    with pytest.raises(ValueError, match="overlap"):
        separate_partitions(gallery, query)
    g, q, audit = separate_partitions(gallery, query, overlap_policy="purge-query")
    assert audit["excluded_query_ids"] == [query.image_id.iloc[0]]
    assert not partition_overlaps(g, q)


def test_seeded_sampling_retains_encounter_images():
    frame = protocol_frame()
    a = sample_encounters(frame, max_identities=5, max_images_per_identity=3)
    b = sample_encounters(frame.iloc[::-1], max_identities=5, max_images_per_identity=3)
    pd.testing.assert_frame_equal(a, b)
    assert a.identity.nunique() == 5
    assert (a.groupby("encounter_id").size() == 2).all()


def test_chunked_ranking_excludes_self_and_is_chunk_invariant():
    x = np.asarray([[1, 0], [.9, .1], [0, 1]], np.float32)
    small = list(rank_queries(["a", "c"], ["a", "b", "c"], x[[0, 2]], x,
                              shortlist_size=2, depth=3, chunk_size=1))
    large = list(rank_queries(["a", "c"], ["a", "b", "c"], x[[0, 2]], x,
                              shortlist_size=2, depth=3, chunk_size=10))
    assert small == large
    assert small[0]["candidate_ids"] == ["b", "c"]
    assert all(row["query_image_id"] not in row["candidate_ids"] for row in small)


def test_fisher_shortlist_union_and_geometric_rerank_are_label_free():
    kwargs = dict(query_ids=["q"], gallery_ids=["a", "b", "c"],
                  query_global=[[1, 0]], gallery_global=[[1, 0], [0, 1], [-1, 0]],
                  query_fisher=[[1, 0]], gallery_fisher=[[0, 1], [1, 0], [-1, 0]],
                  shortlist_size=1, rerank_size=2)
    fused = list(rank_queries(**kwargs, signal="global+fisher"))[0]
    assert set(fused["shortlist_ids"]) == {"a", "b"}
    calls = []
    def verify(q, g):
        calls.append((q, g))
        return {"attempted": True, "n_inliers": 20 if g == "b" else 0}
    full = list(rank_queries(**kwargs, signal="full-funnel", verifier=verify))[0]
    assert full["candidate_ids"] == ["b", "a"]
    assert len(calls) == 2
    assert full["confidence"] > 0


def test_skipped_verification_preserves_rank_and_score():
    kwargs = dict(query_ids=["q"], gallery_ids=["a", "b"],
                  query_global=[[1, 0]], gallery_global=[[1, 0], [.5, .5]],
                  query_fisher=[[1, 0]], gallery_fisher=[[1, 0], [.5, .5]])
    fused = list(rank_queries(**kwargs, signal="global+fisher"))[0]
    full = list(rank_queries(**kwargs, signal="full-funnel",
                             verifier=lambda q, g: {"attempted": False, "n_inliers": 0,
                                                     "reason": "missing_local_features"}))[0]
    assert full["candidate_ids"] == fused["candidate_ids"]
    assert full["confidence"] == fused["confidence"]


def test_known_accuracy_unknown_acceptance_and_truncated_ap():
    gallery = pd.DataFrame({"image_id": ["a1", "a2", "b"], "identity": ["a", "a", "b"]})
    query = pd.DataFrame({"image_id": ["q", "u"], "identity": ["a", "new"], "encounter_id": ["1", "2"]})
    rows = [dict(query_image_id=qid, candidate_ids=["a1", "b"], candidate_scores=[.8, .6],
                 shortlist_ids=["a1", "a2", "b"], confidence=conf, verification=[])
            for qid, conf in (("q", .8), ("u", .4))]
    stats, scored = retrieval_metrics(rows, query, gallery, depth=2, threshold=.5)
    assert stats["top1"] == 1.0
    assert stats["known_queries"] == stats["unknown_queries"] == 1
    assert stats["map_at_k"] == .5  # One of two relevant images retrieved at rank 1.
    assert stats["shortlist_recall"] == 1.0
    assert stats["unknown_false_accept_rate"] == 0
    assert stats["known_false_reject_rate"] == 0
    assert scored[1]["ap_at_k"] is None
    with pytest.raises(ValueError, match="exactly"):
        retrieval_metrics(rows[:1], query, gallery)


def test_threshold_ties_do_not_exceed_requested_false_acceptance():
    rows = [{"known": False, "confidence": x} for x in [0, .3, .5, .5, .5]]
    threshold = rejection_threshold(rows, target_false_accept_rate=.2)
    assert sum(row["confidence"] >= threshold for row in rows) / len(rows) <= .2
    with pytest.raises(ValueError, match="unknown"):
        rejection_threshold([{"known": True, "confidence": .8}])


def test_array_cache_rejects_changed_weights_order_and_corruption(tmp_path):
    path = tmp_path / "features.npz"
    matrix = np.eye(2, dtype=np.float32)
    save_array_cache(path, matrix, ["a", "b"], {"weights": "v1"})
    np.testing.assert_array_equal(load_array_cache(path, ["a", "b"], {"weights": "v1"}), matrix)
    with pytest.raises(ValueError, match="Incompatible"):
        load_array_cache(path, ["a", "b"], {"weights": "v2"})
    with pytest.raises(ValueError, match="ordering"):
        load_array_cache(path, ["b", "a"], {"weights": "v1"})
    path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="damaged"):
        load_array_cache(path, ["a", "b"], {"weights": "v1"})


def test_incremental_vector_cache_integrity(tmp_path):
    path = tmp_path / "vectors.sqlite"
    cache = VectorCache(path, {"weights": "v1"})
    assert cache.get("a", "imagehash") is None
    cache.put("a", "imagehash", [1, 2, 3])
    cache.close()
    cache = VectorCache(path, {"weights": "v1"})
    np.testing.assert_array_equal(cache.get("a", "imagehash"), [1, 2, 3])
    with pytest.raises(ValueError, match="Changed"):
        cache.get("a", "changedhash")
    cache.db.execute("UPDATE vectors SET data=? WHERE image_id='a'", (b"damaged",))
    with pytest.raises(ValueError, match="damaged"):
        cache.get("a", "imagehash")
    cache.close()
    with pytest.raises(ValueError, match="Incompatible"):
        VectorCache(path, {"weights": "v2"})


def test_threshold_provenance_rejects_test_labels_and_wrong_configuration(tmp_path):
    path = tmp_path / "thresholds.json"
    data = dict(stage="development", contract={"model": "v1"}, query_ids=["q"], gallery_ids=["g"],
                thresholds={"global": .6})
    path.write_text(json.dumps(data))
    assert validate_thresholds(path, {"model": "v1"}, ["g", "q"]) == data
    with pytest.raises(ValueError, match="compatible"):
        validate_thresholds(path, {"model": "v2"}, ["g", "q"])
    with pytest.raises(ValueError, match="outside"):
        validate_thresholds(path, {"model": "v1"}, ["g"])
    data["stage"] = "test"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="development"):
        validate_thresholds(path, {"model": "v1"}, ["g", "q"])


def test_noise_is_not_scored_as_one_merged_animal():
    stats = clustering_metrics(["a", "b", "c"], [-1, -1, -1], {"a": "a", "b": "b", "c": "c"})
    assert stats["adjusted_rand_index"] == 1
    assert stats["found_clusters"] == 0
    assert stats["unassigned_images"] == 3
    assert stats["merge_errors"] == 0


def test_runner_global_end_to_end_and_frozen_test(tmp_path, monkeypatch):
    from PIL import Image
    import tools.evaluate_czechlynx as runner
    from reid_demo.benchmark import file_digest, write_json
    frame = protocol_frame()
    prepared = tmp_path / "prepared"
    mode_dir = prepared / "frame"
    mode_dir.mkdir(parents=True)
    paths, hashes = [], []
    for index in range(len(frame)):
        path = tmp_path / f"image_{index}.png"
        Image.new("RGB", (8, 8), (index, 80, 40)).save(path)
        paths.append(str(path))
        hashes.append(file_digest(path))
    frame["path"], frame["image_sha256"] = paths, hashes
    frame["input_mode"] = "frame"
    metadata = mode_dir / "metadata.csv"
    frame.to_csv(metadata, index=False)
    write_json(mode_dir / "manifest.json", {"prepared_metadata_sha256": file_digest(metadata)})
    write_json(prepared / "data_audit.json", {"errors": []})
    def embeddings(rows, args, cache):
        x = np.asarray([[1., (int(rid.split("-")[0]) + 1) / 10] for rid in rows.image_id], dtype=np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x, {"weights_sha256": "fixture"}
    monkeypatch.setattr(runner, "global_vectors", embeddings)
    common = ["--metadata", str(metadata), "--signals", "global", "--cluster-limit", "0",
              "--cache-dir", str(tmp_path / "cache"), "--split", "time_open"]
    dev = tmp_path / "dev"
    assert runner.main(common + ["--out-dir", str(dev)]) == 0
    expected = json.loads((dev / "predictions_global.json").read_text())
    assert runner.main(common + ["--out-dir", str(dev)]) == 0
    assert json.loads((dev / "predictions_global.json").read_text()) == expected
    assert json.loads((dev / "manifest.json").read_text())["status"] == "complete"
    test = tmp_path / "test"
    assert runner.main(common + ["--stage", "test", "--out-dir", str(test),
                                 "--thresholds", str(dev / "thresholds.json")]) == 0
    assert len(pd.read_csv(test / "query.csv")) == 20
    assert json.loads((test / "metrics.json").read_text())["global"]["threshold_evaluation"] == "held-out test"
    with pytest.raises(ValueError, match="different run"):
        runner.main(common + ["--out-dir", str(dev), "--seed", "7"])


def test_memory_bounded_normalization_checks_and_same_rankings():
    from reid_demo.benchmark import validated_unit_matrix
    x = np.eye(3, dtype=np.float32)
    assert np.shares_memory(validated_unit_matrix(x), x)
    a = list(rank_queries(["q"], ["a", "b", "c"], x[:1], x))
    b = list(rank_queries(["q"], ["a", "b", "c"], x[:1], x, features_normalized=True))
    assert a == b
    with pytest.raises(ValueError, match="unit"):
        validated_unit_matrix(np.asarray([[2, 0]], dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        validated_unit_matrix(np.asarray([[np.nan, 0]], dtype=np.float32))


def test_frozen_fisher_artifact_checks_partition_settings_and_integrity(tmp_path, monkeypatch):
    import argparse
    import reid_demo.fisher as fisher_service
    from tools.evaluate_czechlynx import fisher_model
    rng = np.random.default_rng(4)
    descriptors = {rid: rng.normal(size=(25, 4)).astype(np.float32) for rid in ["a", "b", "q"]}
    fitting = pd.DataFrame({"image_id": ["a", "b"], "image_sha256": ["ha", "hb"]})
    args = argparse.Namespace(pca_dim=2, gmm_components=2, max_descriptors=40,
                              per_image_max=25, seed=42, split="time_open",
                              stage="development", fisher_model=None)
    model, info = fisher_model(descriptors, fitting, args, tmp_path, {"method": "fixture"}, "metadata-v1")
    assert model.fitting_ids == ["a", "b"]
    def no_fit(*args, **kwargs):
        raise AssertionError("A frozen or cached model must never refit")
    monkeypatch.setattr(fisher_service, "fit_fisher_model", no_fit)
    _, cached = fisher_model(descriptors, fitting, args, tmp_path, {"method": "fixture"}, "metadata-v1")
    assert cached["sha256"] == info["sha256"]
    args.stage, args.fisher_model = "test", info["path"]
    _, frozen = fisher_model(descriptors, fitting, args, tmp_path, {"method": "fixture"}, "metadata-v1")
    assert frozen["sha256"] == info["sha256"]
    with pytest.raises(ValueError, match="outside"):
        fisher_model(descriptors, fitting.iloc[:1], args, tmp_path, {"method": "fixture"}, "metadata-v1")
    with pytest.raises(ValueError, match="incompatible"):
        fisher_model(descriptors, fitting, args, tmp_path, {"method": "fixture"}, "metadata-v2")
    args.pca_dim = 3
    with pytest.raises(ValueError, match="incompatible"):
        fisher_model(descriptors, fitting, args, tmp_path, {"method": "fixture"}, "metadata-v1")
    args.pca_dim = 2
    from pathlib import Path
    Path(info["path"]).write_bytes(b"damaged")
    with pytest.raises(ValueError, match="Damaged"):
        fisher_model(descriptors, fitting, args, tmp_path, {"method": "fixture"}, "metadata-v1")


def test_partial_local_extraction_is_not_trusted_as_complete(tmp_path, monkeypatch):
    import h5py
    import feature_extraction
    from pathlib import Path
    from tools.evaluate_czechlynx import LocalFeatures
    rows = pd.DataFrame([{"image_id": rid, "path": f"{rid}.jpg", "image_sha256": rid}
                         for rid in ["a", "b"]])
    calls = []
    def extract(items, model_path, out_dir, max_keypoints):
        calls.append(list(items))
        root = Path(out_dir)
        root.mkdir(exist_ok=True)
        # First attempt mimics the extractor skipping an unreadable image.
        keep = items[:1] if len(calls) == 1 else items
        with h5py.File(root / "descriptors.h5", "w") as desc, h5py.File(root / "keypoints.h5", "w") as kp:
            for rid, _ in keep:
                desc[rid] = np.ones((3, 4), np.float32)
                kp[rid] = np.ones((3, 2), np.float32)
    monkeypatch.setattr(feature_extraction, "extract_features", extract)
    root = tmp_path / "local"
    with pytest.raises(ValueError, match="cover"):
        LocalFeatures(root, rows, {"max_keypoints": 4})
    assert not (root / "index.json").exists()
    local = LocalFeatures(root, rows, {"max_keypoints": 4})
    assert local["b"].shape == (3, 4)
    local.close()
    cached = LocalFeatures(root, rows, {"max_keypoints": 4})
    assert len(calls) == 2
    assert cached["a"].shape == (3, 4)
    shard = cached.index["images"]["a"]["shard"]
    cached.close()
    (root / shard / "descriptors.h5").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="Damaged"):
        LocalFeatures(root, rows, {"max_keypoints": 4})
