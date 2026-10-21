"""GV-only masks must preserve retrieval inputs and separate verification evidence."""
import argparse
import numpy as np
import pandas as pd
import pytest
from PIL import Image

import tools.evaluate_czechlynx as runner
from reid_demo.benchmark import file_digest, write_json
from reid_demo.benchmark_gv_masks import ForegroundFeatures, prepare_gv_masks
from reid_demo.benchmark_masks import MASKED_PREPROCESSING


def fixture_metadata(tmp_path):
    rows = []
    for rid in ("q", "g", "empty"):
        path, mask_path = tmp_path / f"{rid}.png", tmp_path / f"{rid}_mask.png"
        Image.new("RGB", (5, 4), "green").save(path)
        mask = Image.new("L", (5, 4), 0)
        mask.putpixel((1, 1), 255)
        mask.putpixel((3, 2), 255)
        mask.save(mask_path)
        rows.append(dict(image_id=rid, identity=rid, path=str(path), image_sha256=file_digest(path),
                         source_sha256=rid, bbox="[0,0,5,4]", encounter_id=rid, timestamp="2020-01-01",
                         input_mode="masked", mask_path=str(mask_path), mask_sha256=file_digest(mask_path),
                         **{"split-time_open": "train"}))
    masks = pd.DataFrame(rows)
    path = tmp_path / "metadata.csv"
    save_masks(path, masks)
    bbox = masks.drop(columns=["mask_path", "mask_sha256"]).copy()
    bbox["input_mode"] = "bbox"
    return bbox, masks, path


def save_masks(path, frame):
    frame.to_csv(path, index=False)
    write_json(path.with_name("manifest.json"), {
        "prepared_metadata_sha256": file_digest(path), "masked_preprocessing": MASKED_PREPROCESSING})


def test_join_preserves_bbox_inputs_and_aligns_by_id(tmp_path):
    bbox, masks, path = fixture_metadata(tmp_path)
    before = bbox.copy(deep=True)
    save_masks(path, masks.iloc[::-1])
    joined, spec = prepare_gv_masks(bbox, path)
    pd.testing.assert_frame_equal(bbox, before)
    pd.testing.assert_frame_equal(joined[bbox.columns], bbox)
    assert joined.mask_path.tolist() == masks.mask_path.tolist()
    assert spec["scope"] == "geometric_verification_only"
    _, reordered = prepare_gv_masks(bbox.iloc[::-1], path)
    assert spec == reordered
    masks.loc[0, "mask_sha256"] = "changed"
    save_masks(path, masks)
    _, changed = prepare_gv_masks(bbox, path)
    assert spec["mask_dataset_sha256"] != changed["mask_dataset_sha256"]


def test_join_preserves_missing_dates_but_rejects_changed_dates(tmp_path):
    bbox, masks, path = fixture_metadata(tmp_path)
    bbox.loc[0, "timestamp"] = np.nan
    masks.loc[0, "timestamp"] = np.nan
    save_masks(path, masks)
    joined, _ = prepare_gv_masks(bbox, path)
    assert pd.isna(joined.timestamp.iloc[0])
    masks.loc[0, "timestamp"] = "2020-01-01"
    save_masks(path, masks)
    with pytest.raises(ValueError, match="timestamp"):
        prepare_gv_masks(bbox, path)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "source_sha256", "bbox", "identity",
                                     "encounter_id", "split-time_open", "manifest", "input_mode"])
def test_join_rejects_incompatible_mask_metadata(tmp_path, failure):
    bbox, masks, path = fixture_metadata(tmp_path)
    if failure == "missing":
        masks = masks.iloc[1:]
    elif failure == "duplicate":
        masks = pd.concat([masks, masks.iloc[:1]])
    elif failure == "bbox":
        masks.loc[0, failure] = "[1,0,6,4]"
    elif failure == "input_mode":
        bbox["input_mode"] = "masked"
    elif failure != "manifest":
        masks.loc[0, failure] = "different"
    save_masks(path, masks)
    if failure == "manifest":
        path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError):
        prepare_gv_masks(bbox, path)


def test_foreground_view_filters_both_sides_keeps_raw_features_and_skips_empty(tmp_path, monkeypatch):
    import geometric_verification as gv
    bbox, _, path = fixture_metadata(tmp_path)
    frame, spec = prepare_gv_masks(bbox, path)
    desc = np.arange(12, dtype=np.float32).reshape(3, 4)
    points = np.array([[1, 1], [2, 1], [3, 2]], np.float32)

    class Local:
        def pair(self, rid):
            return desc, np.zeros_like(points) if rid == "empty" else points

    raw = Local()
    filtered = ForegroundFeatures(raw, frame)
    np.testing.assert_array_equal(filtered.pair("q")[0], desc[[0, 2]])
    np.testing.assert_array_equal(raw.pair("q")[0], desc)
    np.testing.assert_array_equal(raw.pair("q")[1], points)
    assert filtered.mask_counts == dict(images=3, raw_keypoints=9, foreground_keypoints=4, empty_images=1)
    monkeypatch.setattr(gv, "_LIGHTGLUE_AVAILABLE", True)
    calls = []

    def match(qd, qk, gd, gk, *args, **kwargs):
        expected = [0, 2] if len(qd) == 2 else [0, 1, 2]
        for d, k in ((qd, qk), (gd, gk)):
            np.testing.assert_array_equal(d, desc[expected])
            np.testing.assert_array_equal(k, points[expected])
        calls.append(len(qd))
        return 1., len(qd)

    monkeypatch.setattr(gv, "compute_geometric_similarity", match)
    base_spec = {"local": {"method": "disk", "input_mode": "bbox"}}
    for local, verification in [(raw, base_spec), (filtered, {**base_spec, "foreground_mask": spec}),
                                (filtered, {**base_spec, "foreground_mask": spec})]:
        verifier = runner.PairVerifier(tmp_path / "pairs.sqlite", local, verification, 42)
        verifier("q", "g")
        if local is filtered:
            assert verifier("empty", "q")["attempted"] is False
        verifier.close()
    assert calls == [3, 2]
    filtered.close()
    # Pair-cache hits must not conceal changed masks on the next run.
    Image.new("L", (5, 4), 255).save(frame.mask_path.iloc[0])
    with pytest.raises(ValueError, match="Mask changed"):
        ForegroundFeatures(raw, frame)


def test_disk_thresholds_bind_gv_only_masks_and_reject_plain_bbox(tmp_path):
    args = argparse.Namespace(method=["disk"], gv_features="disk", split="time_open",
                              shortlist_size=50, rerank_size=10)
    gv = {"local": {"method": "disk", "input_mode": "bbox"}}
    plain = runner.threshold_contract(args, "metadata", {}, {"sha256": "model"}, gv)
    masked = runner.threshold_contract(args, "metadata", {}, {"sha256": "model"},
                                       {**gv, "foreground_mask": {"mask_dataset_sha256": "masks"}})
    assert "local_configuration" not in plain
    assert "local_configuration" in masked
    path = tmp_path / "thresholds.json"
    write_json(path, dict(stage="development", contract=plain, gallery_ids=["g"], query_ids=["q"],
                         thresholds={"full-funnel": .5}))
    with pytest.raises(ValueError, match="compatible"):
        runner.validate_thresholds(path, masked, ["g", "q"])


def test_gv_mask_flag_requires_full_funnel():
    with pytest.raises(SystemExit):
        runner.main(["--metadata", "unused.csv", "--out-dir", "unused", "--signals", "global",
                     "--gv-mask-metadata", "masks.csv"])
