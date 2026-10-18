import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from reid_demo.benchmark import file_digest
from reid_demo.benchmark_diagnostics import (
    appearance_measurements, build_diagnostics, measure_appearance,
)


def masked_row(tmp_path):
    pixels = np.zeros((128, 128, 3), dtype=np.uint8)
    pixels[:, :] = [255, 0, 0]
    pixels[32:96, 32:96] = 100
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[32:96, 32:96] = 255
    path, mask_path = tmp_path / "image.png", tmp_path / "mask.png"
    Image.fromarray(pixels).save(path)
    Image.fromarray(mask).save(mask_path)
    return {"image_id": "q", "path": str(path), "image_sha256": file_digest(path),
            "mask_path": str(mask_path), "mask_sha256": file_digest(mask_path), "input_mode": "masked"}


def test_foreground_only_measurement_and_mask_bound_cache(tmp_path):
    row = masked_row(tmp_path)
    measured = measure_appearance(row)
    assert measured["near_monochrome"] is True
    assert measured["foreground_median_intensity"] == 100
    assert measured["chroma_p90"] == 0
    cache = tmp_path / "cache"
    first = appearance_measurements(pd.DataFrame([row]), cache)
    assert appearance_measurements(pd.DataFrame([row]), cache) == first
    # Same pixels, different mask selects the red background, making a distinct cache.
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[:16, :] = 255
    Image.fromarray(mask).save(row["mask_path"])
    with pytest.raises(ValueError, match="Mask changed"):
        appearance_measurements(pd.DataFrame([row]), cache)
    row["mask_sha256"] = file_digest(row["mask_path"])
    second = appearance_measurements(pd.DataFrame([row]), cache)
    assert second["q"]["near_monochrome"] is False
    assert len(list((cache / "appearance").glob("*.json"))) == 2
    Image.new("RGB", (128, 128), "blue").save(row["path"])
    with pytest.raises(ValueError, match="Image changed"):
        appearance_measurements(pd.DataFrame([row]), cache)


def test_missing_masks_and_empty_resized_foreground_are_explicit(tmp_path):
    assert measure_appearance({"input_mode": "frame"})["near_monochrome"] is None
    row = masked_row(tmp_path)
    # Real nonempty mask whose only point is lost under nearest-neighbour downsampling.
    Image.new("RGB", (1024, 1024), "gray").save(row["path"])
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    mask[0, 0] = 255
    Image.fromarray(mask).save(row["mask_path"])
    row["image_sha256"], row["mask_sha256"] = file_digest(row["path"]), file_digest(row["mask_path"])
    assert measure_appearance(row)["reason"] == "empty_resized_foreground"


def diagnostic_fixture():
    gallery = pd.DataFrame([
        {"image_id": "g1", "identity": "a", "source": "park", "trap_id": "1"},
        {"image_id": "g2", "identity": "b", "source": "park", "trap_id": None},
    ])
    query = pd.DataFrame([
        {"image_id": "q1", "identity": "a", "source": "park", "trap_id": "2", "encounter_id": "e1"},
        {"image_id": "q2", "identity": "a", "source": "park", "trap_id": None, "encounter_id": "e2"},
        {"image_id": "q3", "identity": "b", "source": "park", "trap_id": "3", "encounter_id": "e3"},
        {"image_id": "q4", "identity": "unknown", "source": "<other>", "trap_id": "3", "encounter_id": "e4"},
    ])
    appearance = {rid: {"near_monochrome": mono, "foreground_median_intensity": 100.,
                        "foreground_contrast_p90_p10": 20., "chroma_p90": 0. if mono else 30.,
                        "reason": "measured"}
                  for rid, mono in [("g1", False), ("g2", True), ("q1", True),
                                    ("q2", False), ("q3", True), ("q4", False)]}
    rows = [
        {"query_image_id": q["image_id"], "identity": q["identity"], "encounter_id": q["encounter_id"],
         "known": q["identity"] != "unknown", "candidate_ids": ["g1", "g2"],
         "shortlist_ids": ["g1", "g2"], "shortlist_hit": q["identity"] != "unknown",
         "top1_correct": q["identity"] == "a", "top5_correct": q["identity"] != "unknown",
         "ap_at_k": 1. if q["identity"] == "a" else .5 if q["identity"] == "b" else None,
         "accepted": q["image_id"] in ("q1", "q4")}
        for q in query.to_dict("records")]
    return gallery, query, appearance, {"global": rows}


def test_diagnostics_denominators_unknowns_and_missing_camera_categories():
    gallery, query, appearance, predictions = diagnostic_fixture()
    result = build_diagnostics(predictions, gallery, query, appearance, depth=50)
    groups = result["stages"]["global"]
    assert groups["all"]["known_queries"] == 3
    assert groups["all"]["accepted_precision"] == .5
    assert groups["all"]["correct_known_coverage"] == 1 / 3
    assert groups["all"]["unknown_false_accepts"] == 1
    assert groups["all"]["recall_at"]["50"] == 1.
    assert groups["cross_appearance/opposite_appearance_only"]["queries"] == 1
    assert groups["camera/other_camera_only"]["queries"] == 1
    assert groups["camera/query_camera_missing"]["queries"] == 1
    assert groups["camera/reference_camera_missing"]["queries"] == 1
    assert groups["source/<other>"]["top1"] is None
    assert groups["source/<other>"]["unknown_queries"] == 1
    short = build_diagnostics(predictions, gallery, query, appearance, depth=10)
    assert short["stages"]["global"]["all"]["recall_at"]["50"] is None
    with pytest.raises(ValueError, match="exactly"):
        build_diagnostics({"global": predictions["global"][:-1]}, gallery, query, appearance)


def test_report_exposes_role_diagnostics_and_clustering_pair_quality(tmp_path):
    from reid_demo.benchmark_report import write_report
    gallery, query, appearance, predictions = diagnostic_fixture()
    diagnostics = build_diagnostics(predictions, gallery, query, appearance)
    inventory = {"inventory": {"images": 10, "identities": 3}}
    manifest = {"stage": "development", "partition": "selection", "protocol": {"fingerprint": "abc"},
                "protocol_inventory": {"fitting": inventory}, "diagnostics": diagnostics}
    # No example tiles needed to verify tables and escaping.
    metadata = pd.DataFrame(columns=["image_id", "identity", "path"])
    metrics = {"global": {"cluster_found_clusters": 2, "cluster_pairwise_precision": .1,
                          "cluster_pairwise_recall": .2, "threshold": .5}}
    page = write_report(tmp_path, manifest, metrics, {}, metadata)
    text = page.read_text()
    assert "Selection validation" in text and "frozen calibration thresholds" in text
    assert "Pair precision" in text and "Pair recall" in text
    assert "source/&lt;other&gt;" in text and "source/<other>" not in text
    assert "10.0%" in text and "20.0%" in text
    assert json.loads((tmp_path / "diagnostics.json").read_text()) == diagnostics

