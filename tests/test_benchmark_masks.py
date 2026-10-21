"""Foreground filtering and cache integrity, without neural inference."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import tools.evaluate_czechlynx as runner
from reid_demo.benchmark import file_digest
from reid_demo.benchmark_masks import (
    MASKED_PREPROCESSING, filter_foreground_features, load_foreground_mask, local_input_hash,
)


def masked_row(tmp_path, rid="q"):
    path = tmp_path / f"{rid}.png"
    mask_path = tmp_path / f"{rid}_mask.png"
    Image.new("RGB", (5, 4), (128, 128, 128)).save(path)
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[1, 1] = mask[2, 3] = 255
    Image.fromarray(mask).save(mask_path)
    return dict(image_id=rid, path=str(path), image_sha256=file_digest(path), input_mode="masked",
                mask_path=str(mask_path), mask_sha256=file_digest(mask_path))


def test_filter_uses_xy_crop_coordinates_and_preserves_descriptor_alignment():
    mask = np.zeros((4, 5), dtype=bool)
    mask[1, 1] = mask[2, 3] = True
    kp = np.array([[1.1, 1.2], [2, 1], [3, 2], [4.8, 2], [-.1, 1], [1, 4.5]], np.float32)
    desc = np.arange(24, dtype=np.float32).reshape(6, 4)
    d, k = filter_foreground_features(desc, kp, mask)
    np.testing.assert_array_equal(d, desc[[0, 2]])
    np.testing.assert_array_equal(k, kp[[0, 2]])
    d, k = filter_foreground_features(desc[:0], kp[:0], mask)
    assert d.shape == (0, 4) and k.shape == (0, 2)
    with pytest.raises(ValueError, match="Invalid local"):
        filter_foreground_features(desc[:1], np.array([[np.nan, 1]]), mask)


@pytest.mark.parametrize("failure", ["size", "binary", "empty", "checksum"])
def test_mask_validation_rejects_misalignment_invalid_pixels_and_changed_bytes(tmp_path, failure):
    row = masked_row(tmp_path)
    if failure == "size":
        Image.new("L", (4, 5), 255).save(row["mask_path"])
    elif failure == "binary":
        Image.new("L", (5, 4), 127).save(row["mask_path"])
    else:
        Image.new("L", (5, 4), 0).save(row["mask_path"])
    if failure != "checksum":
        row["mask_sha256"] = file_digest(row["mask_path"])
    with pytest.raises(ValueError):
        load_foreground_mask(row)


@pytest.mark.parametrize("method", ["disk", "superpoint", "aliked"])
def test_local_mask_filter_cache_and_empty_features_are_used_by_fisher_and_gv(tmp_path, monkeypatch, method):
    import h5py
    import feature_extraction
    import geometric_verification as gv
    from reid_demo.fisher import transform_fisher_vectors
    rows = pd.DataFrame([masked_row(tmp_path, rid) for rid in ("q", "empty")])
    points = np.array([[1, 1], [2, 1], [3, 2]], np.float32)
    descs = np.arange(12, dtype=np.float32).reshape(3, 4)
    calls = []

    def write_features(items, output_dir):
        calls.append(list(items))
        root = Path(output_dir)
        root.mkdir(exist_ok=True)
        with h5py.File(root / "descriptors.h5", "w") as desc, h5py.File(root / "keypoints.h5", "w") as kp:
            for rid, _ in items:
                desc[rid] = descs
                points_for_image = points if rid == "q" else np.zeros((3, 2), np.float32)
                kp[rid] = points_for_image if method == "disk" else points_for_image[None]

    monkeypatch.setattr(feature_extraction, "extract_features", lambda items, model, out, **kw: write_features(items, out))
    monkeypatch.setattr(feature_extraction, "extract_features_lightglue", lambda items, out, **kw: write_features(items, out))
    spec = {"method": method, "input_mode": "masked", "max_keypoints": 3}
    root = tmp_path / "local"
    local = runner.LocalFeatures(root, rows, spec)
    np.testing.assert_array_equal(local["q"], descs[[0, 2]])
    assert local["empty"].shape == (0, 4)
    assert local.mask_counts == dict(images=2, raw_keypoints=6, foreground_keypoints=2, empty_images=1)
    zero = transform_fisher_vectors(local, ["empty"], argparse.Namespace(pca_dim=2, gmm_components=2))["empty"]
    np.testing.assert_array_equal(zero, np.zeros(8))
    monkeypatch.setattr(gv, "_LIGHTGLUE_AVAILABLE", True)

    def no_matching(*args, **kwargs):
        raise AssertionError("An empty foreground feature set must skip GV")

    monkeypatch.setattr(gv, "compute_geometric_similarity", no_matching)
    verifier = runner.PairVerifier(tmp_path / "gv.sqlite", local, {"local": spec}, 42)
    assert verifier("empty", "q") == dict(attempted=False, n_inliers=0, reason="missing_local_features")
    verifier.close()
    local.close()
    cached = runner.LocalFeatures(root, rows, spec)
    assert len(calls) == 1
    assert cached.mask_counts["empty_images"] == 1
    cached.close()
    # Even unchanged RGB pixels cannot hide a changed filtering mask.
    row = rows.iloc[0].to_dict()
    previous = local_input_hash(row, masked=True)
    Image.new("L", (5, 4), 255).save(row["mask_path"])
    rows.loc[0, "mask_sha256"] = file_digest(row["mask_path"])
    assert local_input_hash(rows.iloc[0].to_dict(), masked=True) != previous
    with pytest.raises(ValueError, match="Changed local-feature"):
        runner.LocalFeatures(root, rows, spec)


def test_mask_contract_changes_when_only_mask_changes_and_rejects_missing_provenance(tmp_path):
    frame = pd.DataFrame([masked_row(tmp_path)])
    manifest = {"masked_preprocessing": MASKED_PREPROCESSING}
    a = runner.masked_input_spec(frame, manifest)
    assert a == runner.masked_input_spec(frame.iloc[::-1], manifest)
    frame.loc[0, "mask_sha256"] = "different-mask"
    b = runner.masked_input_spec(frame, manifest)
    assert a["mask_dataset_sha256"] != b["mask_dataset_sha256"]
    with pytest.raises(ValueError, match="validated masked"):
        runner.masked_input_spec(frame, {})
    with pytest.raises(ValueError, match="validated masked"):
        runner.masked_input_spec(frame.drop(columns="mask_path"), manifest)
    frame["input_mode"] = "bbox"
    assert runner.masked_input_spec(frame, {}) is None


def test_fisher_vector_cache_binds_mask_even_when_rgb_and_model_are_unchanged(tmp_path, monkeypatch):
    import reid_demo.fisher as fisher
    frame = pd.DataFrame([masked_row(tmp_path)])
    monkeypatch.setattr(fisher, "transform_fisher_vectors", lambda local, ids, model: {ids[0]: np.ones(8, np.float32)})
    model = argparse.Namespace(pca_dim=2, gmm_components=2)
    runner.fisher_vectors(frame, {}, model, {"sha256": "fixture"}, None, tmp_path)
    Image.new("L", (5, 4), 255).save(frame.mask_path.iloc[0])
    frame.loc[0, "mask_sha256"] = file_digest(frame.mask_path.iloc[0])
    with pytest.raises(ValueError, match="Changed image"):
        runner.fisher_vectors(frame, {}, model, {"sha256": "fixture"}, None, tmp_path)
