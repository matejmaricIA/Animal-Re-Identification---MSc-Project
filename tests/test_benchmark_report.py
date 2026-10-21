import json
from pathlib import Path
import re
import shutil

import pandas as pd
from PIL import Image

from reid_demo.benchmark_report import select_examples, write_report


def test_report_remains_portable_after_move_and_escapes_labels(tmp_path):
    image = tmp_path / "input.jpg"
    Image.new("RGB", (60, 40), "green").save(image)
    metadata = pd.DataFrame([
        {"image_id": rid, "identity": "<lynx>", "path": str(image)} for rid in ["q", "g"]])
    predictions = {"global": [{
        "query_image_id": "q", "candidate_ids": ["g"], "known": True,
        "top1_correct": True, "identity": "<lynx>",
    }]}
    metrics = {"global": {"top1": 1., "top5": 1., "map_at_k": 1., "known_queries": 1, "unknown_queries": 0}}
    clusters = pd.DataFrame([dict(signal="global", image_id=rid, identity="<lynx>", cluster_id=0)
                             for rid in ["q", "g"]])
    root = tmp_path / "report"
    write_report(root, {"stage": "development", "input_mode": "bbox"}, metrics, predictions, metadata,
                 clusters=clusters)
    moved = tmp_path / "moved"
    shutil.move(root, moved)
    image.unlink()
    for page in [moved / "index.html", moved / "catalogue/index.html"]:
        contents = page.read_text()
        assert "<lynx>" not in contents
        assert "&lt;lynx&gt;" in contents
        for target in re.findall(r'(?:src|href)="([^"]+)"', contents):
            assert not Path(target).is_absolute()
            assert (page.parent / target).exists()
    assert pd.read_csv(moved / "comparison.csv").top1.iloc[0] == 1
    assert json.loads((moved / "metrics.json").read_text()) == metrics


def test_failure_examples_include_improvements_and_regressions_deterministically():
    def row(qid, correct):
        return dict(query_image_id=qid, known=True, top1_correct=correct)
    predictions = {
        "global": [row("improve", False), row("worsen", True)],
        "full-funnel": [row("improve", True), row("worsen", False)],
    }
    examples = select_examples(predictions)
    assert examples == select_examples(predictions)
    categories = {e["image_id"]: e["categories"] for e in examples}
    assert "full-funnel:improved" in categories["improve"]
    assert "full-funnel:worsened" in categories["worsen"]


def test_report_identifies_ensemble_components_and_separate_gv_choice(tmp_path):
    metadata = pd.DataFrame(columns=["image_id", "identity", "path"])
    manifest = {"stage": "development", "local_features": {
        "methods": ["disk", "superpoint", "aliked"], "weights": [1 / 3] * 3,
        "pca_dim": 128, "gmm_components_per_extractor": 256, "max_keypoints": 1024,
        "gv_features": "aliked",
    }}
    page = write_report(tmp_path, manifest, {}, {}, metadata)
    contents = page.read_text()
    assert "DISK + SuperPoint + ALIKED" in contents
    assert "256 components per extractor" in contents
    assert "128 dimensions per extractor" in contents
    assert "0.333, 0.333, 0.333" in contents
    assert "Geometric verification: ALIKED + LightGlue" in contents


def test_masked_report_explains_provided_masks_and_feature_retention(tmp_path):
    metadata = pd.DataFrame(columns=["image_id", "identity", "path"])
    manifest = {"stage": "development", "input_mode": "masked", "local_features": {
        "methods": ["disk"], "weights": [1.0], "pca_dim": 128,
        "gmm_components_per_extractor": 256, "max_keypoints": 1024, "gv_features": "disk",
        "mask_filter_counts": {"disk": {"images": 5, "raw_keypoints": 500,
                                        "foreground_keypoints": 180, "empty_images": 1}},
    }}
    text = write_report(tmp_path, manifest, {}, {}, metadata).read_text()
    assert "Provided segmentation masks" in text
    assert "annotation-assisted evaluation" in text
    assert "Original camera-trap frames" not in text
    assert "Inside mask" in text and "180" in text
    assert "1 / 5" in text


def test_gv_mask_report_distinguishes_unchanged_retrieval_from_filtered_verification(tmp_path):
    metadata = pd.DataFrame(columns=["image_id", "identity", "path"])
    manifest = {"stage": "development", "input_mode": "bbox", "gv_foreground_mask": {"scope": "geometric_verification_only"},
                "local_features": {
                    "methods": ["disk"], "weights": [1.0], "pca_dim": 128,
                    "gmm_components_per_extractor": 256, "max_keypoints": 1024, "gv_features": "disk",
                    "gv_mask_filter_counts": {"disk": {"images": 5, "raw_keypoints": 500,
                                                       "foreground_keypoints": 180, "empty_images": 1}},
                }}
    text = write_report(tmp_path, manifest, {}, {}, metadata).read_text()
    assert "unchanged bbox images and features" in text
    assert "geometric verification only (Fisher features unchanged)" in text
    assert "Inside mask" in text and "180" in text and "1 / 5" in text
    assert "replace background pixels with grey" not in text


def test_global_model_comparison_is_named_escaped_and_self_contained(tmp_path):
    metadata = pd.DataFrame(columns=["image_id", "identity", "path"])
    model = {"kind": "pretrained_miewid_msv3", "dimension": 2152,
             "transform": "RGB; resize 440x440 bilinear", "revision": "<unsafe>"}
    summary = {"top1": .5, "top5": .75, "map_at_k": .25,
               "recall_at_10": .8, "recall_at_20": .9, "recall_at_50": 1.}
    comparison = {"reference": {"global_model": {"kind": "pretrained_megadescriptor_l_384"}},
                  "current_model": model, "reference_metrics": summary, "current_metrics": summary,
                  "fixed_queries": 4, "worsened_queries": 2}
    manifest = {"stage": "development", "input_mode": "masked", "global_model": model,
                "global_comparison": comparison}
    page = write_report(tmp_path, manifest, {}, {}, metadata)
    text = page.read_text()
    assert "MiewID-msv3" in text and "MegaDescriptor-L-384" in text
    assert "2152 dimensions" in text and "440x440" in text
    assert "&lt;unsafe&gt;" in text and "<unsafe>" not in text
    assert "Corrected 4 known queries; worsened 2" in text
    assert json.loads((tmp_path / "global_model_comparison.json").read_text()) == comparison
    for target in re.findall(r'(?:src|href)="([^\"]+)"', text):
        assert (page.parent / target).exists()
