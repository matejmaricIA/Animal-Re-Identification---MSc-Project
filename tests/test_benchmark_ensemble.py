"""Ensemble contracts and a small development/test run without neural inference."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tools.evaluate_czechlynx as runner
from reid_demo.benchmark import file_digest, fingerprint, normalized, write_json


def test_method_alias_subsets_and_invalid_duplicates():
    assert runner.resolve_methods("ensamble") == (list(runner.LOCAL_METHODS), [1 / 3] * 3)
    assert runner.resolve_methods(["disk", "aliked"]) == (["disk", "aliked"], [.5, .5])
    assert runner.resolve_methods(["disk"]) == (["disk"], [1.0])
    for methods in (["disk", "disk"], ["unknown"], []):
        with pytest.raises(ValueError, match="distinct"):
            runner.resolve_methods(methods)


def test_combination_matches_legacy_vectors_and_is_chunk_invariant(tmp_path):
    from utility_functions import combine_fisher_vectors
    rng = np.random.default_rng(42)
    matrices = [normalized(rng.normal(size=(9, 8))) for _ in range(3)]
    # Missing descriptors produce zero vectors, including a completely empty image.
    matrices[1][0] = 0
    for matrix in matrices:
        matrix[-1] = 0
    weights = [1 / 3] * 3
    legacy = combine_fisher_vectors([dict(enumerate(m)) for m in matrices], weights)
    expected = normalized(np.stack([legacy[i] for i in range(9)]))
    for chunk_size in (1, 4, 20):
        combined = runner.combine_fisher_matrices(matrices, weights, tmp_path / f"{chunk_size}.npy",
                                                  chunk_size=chunk_size)
        assert isinstance(combined, np.memmap)
        np.testing.assert_allclose(combined, expected, atol=1e-6)
        assert np.count_nonzero(combined[-1]) == 0
    with pytest.raises(ValueError, match="equal shapes"):
        runner.combine_fisher_matrices([matrices[0], matrices[1][:2]], [.5, .5], tmp_path / "bad.npy")


@pytest.mark.parametrize("method", ["superpoint", "aliked"])
def test_local_extraction_batch_axis_empty_and_single_keypoint_caches(tmp_path, monkeypatch, method):
    import feature_extraction
    import h5py
    rows = pd.DataFrame([dict(image_id=str(n), path=f"{n}.jpg", image_sha256=f"hash{n}") for n in (0, 1, 3)])
    calls = []

    def extract(items, output_dir, feature_type, max_keypoints):
        calls.append(feature_type)
        root = Path(output_dir)
        root.mkdir(exist_ok=True)
        with h5py.File(root / "descriptors.h5", "w") as desc, h5py.File(root / "keypoints.h5", "w") as kp:
            for rid, _ in items:
                desc[rid] = np.ones((int(rid), 4), dtype=np.float32)
                kp[rid] = np.ones((1, int(rid), 2), dtype=np.float32)

    monkeypatch.setattr(feature_extraction, "extract_features_lightglue", extract)
    spec = {"method": method, "max_keypoints": 4}
    local = runner.LocalFeatures(tmp_path / "local", rows, spec)
    for n in (0, 1, 3):
        desc, kp = local.pair(str(n))
        assert desc.shape == (n, 4)
        assert kp.shape == (n, 2)
    local.close()
    cached = runner.LocalFeatures(tmp_path / "local", rows, spec)
    assert calls == [method]
    assert cached.pair("1")[1].shape == (1, 2)
    cached.close()
    with pytest.raises(ValueError, match="Incompatible local"):
        runner.LocalFeatures(tmp_path / "local", rows, {**spec, "method": "disk"})


def test_gv_uses_selected_extractor_and_separates_cached_evidence(tmp_path, monkeypatch):
    import geometric_verification as gv
    monkeypatch.setattr(gv, "_LIGHTGLUE_AVAILABLE", True)
    calls = []

    def similarity(*args, method, **kwargs):
        calls.append(method)
        return 1., 12

    monkeypatch.setattr(gv, "compute_geometric_similarity", similarity)

    class Local:
        def pair(self, image_id):
            return np.ones((4, 4), np.float32), np.ones((4, 2), np.float32)

    for method in ("disk", "aliked", "aliked"):
        verifier = runner.PairVerifier(tmp_path / "gv.sqlite", Local(), {"local": {"method": method}}, 42)
        assert verifier("q", "g")["n_inliers"] == 12
        verifier.close()
    assert calls == ["disk", "aliked"]


def test_bundle_rejects_changed_methods_weights_stage_and_corruption(tmp_path):
    methods, weights = runner.resolve_methods("ensamble")
    models = {}
    for method in methods:
        path = tmp_path / f"{method}.pkl"
        path.write_bytes(method.encode())
        models[method] = {"path": str(path), "sha256": file_digest(path)}
    data = dict(schema=1, kind="fisher_ensemble", stage="development", methods=methods,
                weights=weights, combination=runner.FISHER_COMBINATION, models=models)
    path = tmp_path / "fisher_bundle.json"
    write_json(path, data)
    assert runner.read_fisher_bundle(path, methods, weights) == data
    for changed in ({"methods": methods[::-1]}, {"weights": [.2, .3, .5]}, {"stage": "test"},
                    {"models": {"disk": models["disk"]}}, {"combination": "concatenation"}):
        write_json(path, {**data, **changed})
        with pytest.raises(ValueError, match="Incompatible"):
            runner.read_fisher_bundle(path, methods, weights)
    write_json(path, data)
    Path(models["aliked"]["path"]).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Damaged aliked"):
        runner.read_fisher_bundle(path, methods, weights)
    with pytest.raises(ValueError, match="fisher_bundle.json"):
        runner.read_fisher_bundle(models["disk"]["path"], methods, weights)


def test_bundle_members_must_share_the_development_gallery(tmp_path, monkeypatch):
    def model(local, fitting, args, cache, spec, metadata):
        return object(), {"fitting_ids": [spec["method"]]}

    monkeypatch.setattr(runner, "fisher_model", model)
    args = argparse.Namespace(method=["disk", "aliked"], fisher_model=None,
                              stage="development", gmm_components=256)
    specs = {method: {"method": method} for method in args.method}
    with pytest.raises(ValueError, match="same development fitting"):
        runner.fit_or_load_fisher_models(dict.fromkeys(args.method), specs, None, args, tmp_path, "metadata")


def test_threshold_contract_preserves_disk_and_rejects_ensemble_or_gv_changes(tmp_path):
    args = argparse.Namespace(method=["disk"], gv_features="disk", split="time_open",
                              shortlist_size=50, rerank_size=10)
    disk = runner.threshold_contract(args, "metadata", {"weights": "global"}, {"sha256": "disk-model"})
    assert "local_configuration" not in disk  # Previous DISK contract remains valid.
    path = tmp_path / "thresholds.json"
    write_json(path, dict(stage="development", contract=disk, gallery_ids=["g"], query_ids=["q"],
                         thresholds={"full-funnel": .5}))
    assert runner.validate_thresholds(path, disk, ["g", "q"])
    args.method = ["disk", "superpoint", "aliked"]
    ensemble = runner.threshold_contract(args, "metadata", {"weights": "global"}, {"sha256": "bundle"},
                                         {"local": {"method": "disk"}})
    with pytest.raises(ValueError, match="compatible"):
        runner.validate_thresholds(path, ensemble, ["g", "q"])
    write_json(path, dict(stage="development", contract=ensemble, gallery_ids=["g"], query_ids=["q"],
                         thresholds={"full-funnel": .5}))
    args.gv_features = "aliked"
    changed = runner.threshold_contract(args, "metadata", {"weights": "global"}, {"sha256": "bundle"},
                                        {"local": {"method": "aliked"}})
    with pytest.raises(ValueError, match="compatible"):
        runner.validate_thresholds(path, changed, ["g", "q"])


@pytest.mark.parametrize(("input_mode", "gv_only"), [("bbox", False), ("masked", False), ("bbox", True)])
def test_ensemble_development_cached_replay_and_frozen_test(tmp_path, monkeypatch, input_mode, gv_only):
    from PIL import Image
    import geometric_verification as gv
    import reid_demo.fisher as service
    from reid_demo.benchmark_masks import MASKED_PREPROCESSING

    mode_dir = tmp_path / "prepared" / input_mode
    mode_dir.mkdir(parents=True)
    rows, gv_rows = [], []
    for identity in range(10):
        for encounter in range(4):
            rid = f"{identity}-{encounter}"
            path = tmp_path / f"{rid}.png"
            image = Image.new("RGB", (8, 8), (identity * 20, encounter * 40, 50))
            mask_fields = {}
            if input_mode == "masked" or gv_only:
                if input_mode == "masked":
                    image.putpixel((0, 0), (128, 128, 128))
                mask = Image.new("L", (8, 8), 255)
                mask.putpixel((0, 0), 0)
                mask_path = tmp_path / f"{rid}_mask.png"
                mask.save(mask_path)
                mask_fields = {"mask_path": str(mask_path), "mask_sha256": file_digest(mask_path)}
            image.save(path)
            rows.append(dict(image_id=rid, identity=str(identity), encounter_id=rid,
                             timestamp=f"2020-0{encounter + 1}-01", path=str(path),
                             image_sha256=file_digest(path), source_sha256=rid, input_mode=input_mode,
                             bbox="[0,0,8,8]", **(mask_fields if input_mode == "masked" else {}),
                             **{"split-time_open": "train" if encounter < 3 else "test"}))
            if gv_only:
                masked_path = tmp_path / f"{rid}_masked.png"
                image.putpixel((0, 0), (128, 128, 128))
                image.save(masked_path)
                gv_rows.append({**rows[-1], **mask_fields, "input_mode": "masked",
                                "path": str(masked_path), "image_sha256": file_digest(masked_path)})
    frame = pd.DataFrame(rows)
    metadata = mode_dir / "metadata.csv"
    frame.to_csv(metadata, index=False)
    write_json(mode_dir / "manifest.json", {
        "prepared_metadata_sha256": file_digest(metadata),
        **({"masked_preprocessing": MASKED_PREPROCESSING} if input_mode == "masked" else {}),
    })
    write_json(mode_dir.parent / "data_audit.json", {"errors": []})
    if gv_only:
        gv_metadata = tmp_path / "gv_masks" / "metadata.csv"
        gv_metadata.parent.mkdir()
        pd.DataFrame(gv_rows).to_csv(gv_metadata, index=False)
        write_json(gv_metadata.with_name("manifest.json"), {
            "prepared_metadata_sha256": file_digest(gv_metadata),
            "masked_preprocessing": MASKED_PREPROCESSING,
        })

    class Local(dict):
        def __init__(self, root, images, spec):
            super().__init__()
            self.method = spec["method"]
            self.reads = []
            self.mask_counts = dict(images=len(images), raw_keypoints=16 * len(images),
                                    foreground_keypoints=16 * len(images), empty_images=0)
            assert set(images.input_mode) == {input_mode}
            for rid in images.image_id:
                rng = np.random.default_rng(int(fingerprint([self.method, rid])[:8], 16))
                self[rid] = rng.normal(size=(16, 4)).astype(np.float32)

        def __getitem__(self, rid):
            self.reads.append(rid)
            return super().__getitem__(rid)

        def pair(self, rid):
            points = np.ones((16, 2), np.float32)
            points[:8] = 0
            return self[rid], points

        def close(self):
            pass

    def embeddings(images, args, cache):
        assert set(images.input_mode) == {input_mode}
        if input_mode == "masked":
            with Image.open(images.path.iloc[0]) as im:
                assert im.getpixel((0, 0)) == (128, 128, 128)
        matrix = np.stack([np.random.default_rng(int(fingerprint(rid)[:8], 16)).random(3)
                           for rid in images.image_id]).astype(np.float32)
        return normalized(matrix), {"weights_sha256": "fixture", "input_mode": input_mode}

    monkeypatch.setattr(runner, "LocalFeatures", Local)
    monkeypatch.setattr(runner, "local_feature_spec", lambda method, args, mode: {"method": method, "input_mode": mode})
    monkeypatch.setattr(runner, "verification_spec", lambda spec, frame, args: {"local": spec, "seed": args.seed})
    monkeypatch.setattr(runner, "global_vectors", embeddings)
    monkeypatch.setattr(gv, "_LIGHTGLUE_AVAILABLE", True)
    gv_methods = []
    gv_filtered = False

    def similarity(*args, method, **kwargs):
        gv_methods.append(method)
        if gv_filtered:
            assert len(args[0]) == len(args[2]) == 8
            assert (args[1] == 1).all() and (args[3] == 1).all()
        else:
            assert len(args[0]) == len(args[2]) == 16
        return 1., 8

    monkeypatch.setattr(gv, "compute_geometric_similarity", similarity)
    fit = service.fit_fisher_model
    fitted = []
    dev = tmp_path / "dev"

    def fit_gallery_only(local, ids, **settings):
        expected = set(pd.read_csv(dev / "gallery.csv").image_id)
        assert set(ids) == expected
        local.reads.clear()
        model = fit(local, ids, **settings)
        assert set(local.reads) == expected
        fitted.append(local.method)
        return model

    monkeypatch.setattr(service, "fit_fisher_model", fit_gallery_only)
    common = ["--metadata", str(metadata), "--cache-dir", str(tmp_path / "cache"),
              "--method", "ensamble", "--gv_features", "aliked", "--cluster-limit", "0",
              "--pca-dim", "2", "--gmm-components", "2", "--max-descriptors", "64",
              "--per-image-max", "8", "--shortlist-size", "3", "--rerank-size", "2"]
    assert runner.main(common + ["--out-dir", str(dev)]) == 0
    assert fitted == list(runner.LOCAL_METHODS)
    assert set(gv_methods) == {"aliked"}
    if gv_only:
        bbox_dev = dev
        bbox_manifest = json.loads((bbox_dev / "manifest.json").read_text())
        dev = tmp_path / "gv_dev"
        gv_filtered = True
        common += ["--gv-mask-metadata", str(gv_metadata),
                   "--fisher-model", str(bbox_dev / "fisher_bundle.json")]

        def no_hybrid_refit(*args, **kwargs):
            raise AssertionError("GV-only masks must reuse bbox Fisher models")

        monkeypatch.setattr(service, "fit_fisher_model", no_hybrid_refit)
        assert runner.main(common + ["--out-dir", str(dev)]) == 0
        hybrid = json.loads((dev / "manifest.json").read_text())
        for key in ("gallery_ids", "query_ids", "global_model", "fisher_model"):
            assert hybrid[key] == bbox_manifest[key]
        assert hybrid["local_features"]["extractors"] == bbox_manifest["local_features"]["extractors"]
        for signal in ("global", "global+fisher"):
            assert (dev / f"predictions_{signal}.json").read_bytes() == (bbox_dev / f"predictions_{signal}.json").read_bytes()
        assert hybrid["gv_cache"]["computed_pairs"] > 0
        retained = hybrid["local_features"]["gv_mask_filter_counts"]["aliked"]
        assert retained["images"] == len(hybrid["gallery_ids"]) + len(hybrid["query_ids"])
        assert retained["foreground_keypoints"] == 8 * retained["images"]
        contract = json.loads((dev / "thresholds.json").read_text())["contract"]
        with pytest.raises(ValueError, match="compatible"):
            runner.validate_thresholds(bbox_dev / "thresholds.json", contract, frame.image_id)
    manifest = json.loads((dev / "manifest.json").read_text())
    assert len({info["sha256"] for info in manifest["fisher_model"]["models"].values()}) == 3
    assert manifest["local_features"]["gmm_components_per_extractor"] == 2
    if input_mode == "masked":
        assert manifest["segmentation"]["preprocessing"] == MASKED_PREPROCESSING
        assert set(manifest["local_features"]["mask_filter_counts"]) == set(runner.LOCAL_METHODS)
    assert "DISK + SuperPoint + ALIKED" in (dev / "index.html").read_text()
    prediction = (dev / "predictions_full-funnel.json").read_text()

    def no_refit(*args, **kwargs):
        raise AssertionError("Cached/frozen ensemble models must not refit")

    monkeypatch.setattr(service, "fit_fisher_model", no_refit)
    assert runner.main(common + ["--out-dir", str(dev)]) == 0
    assert (dev / "predictions_full-funnel.json").read_text() == prediction
    assert json.loads((dev / "manifest.json").read_text())["gv_cache"]["computed_pairs"] == 0
    test = tmp_path / "test"
    assert runner.main(common + ["--stage", "test", "--out-dir", str(test),
                                 "--fisher-model", manifest["fisher_model"]["path"],
                                 "--thresholds", str(dev / "thresholds.json")]) == 0
    tested = json.loads((test / "manifest.json").read_text())
    assert tested["status"] == "complete"
    assert tested["fisher_model"]["sha256"] == manifest["fisher_model"]["sha256"]
    assert len(pd.read_csv(test / "query.csv")) == 10
    assert json.loads((test / "metrics.json").read_text())["full-funnel"]["threshold_evaluation"] == "held-out test"
    if input_mode == "masked":
        calibration = json.loads((dev / "thresholds.json").read_text())
        calibration["contract"]["global"]["input_mode"] = "bbox"
        write_json(tmp_path / "bbox_thresholds.json", calibration)
        contract = json.loads((dev / "thresholds.json").read_text())["contract"]
        with pytest.raises(ValueError, match="compatible"):
            runner.validate_thresholds(tmp_path / "bbox_thresholds.json", contract, frame.image_id)
