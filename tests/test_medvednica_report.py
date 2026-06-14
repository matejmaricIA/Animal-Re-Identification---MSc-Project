"""Unit + end-to-end tests for T09 (reid_demo.medvednica_report).

Covers the four contracted pure helpers on tiny in-memory fixtures plus one
end-to-end run on the real ``data/MedvednicaDS`` dump asserting the headline
funnel/species numbers. Heavy asserts are skipped (not failed) if the real dump
is absent, so the helper tests still run anywhere.
"""

from __future__ import annotations

import json
import os

import pytest

from reid_demo.medvednica_report import (
    compute_funnel,
    compute_species_counts,
    generate_medvednica_report,
    parse_crop_filename,
    species_from_classes,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "MedvednicaDS")
HAVE_REAL_DATA = os.path.exists(os.path.join(DATA_DIR, "megadetector_results.json"))


# --------------------------------------------------------------------------- #
# parse_crop_filename
# --------------------------------------------------------------------------- #

def test_parse_crop_filename_numeric_stem():
    assert parse_crop_filename("02020401_crop1_conf92.jpg") == ("02020401", 1, 92)


def test_parse_crop_filename_img_stem():
    assert parse_crop_filename("IMG_0066_crop1_conf78.jpg") == ("IMG_0066", 1, 78)


def test_parse_crop_filename_missing_conf_suffix():
    stem, idx, conf = parse_crop_filename("02020401_crop2.jpg")
    assert stem == "02020401"
    assert idx == 2
    assert conf is None


def test_parse_crop_filename_handles_full_path_and_case():
    # Basename only; extension is case-insensitive.
    assert parse_crop_filename("a/b/IMG_0066_crop3_conf81.JPG") == ("IMG_0066", 3, 81)


def test_parse_crop_filename_underscore_stem_with_index():
    # Stems with underscores must survive (everything before _cropN).
    assert parse_crop_filename("IMG_0066_crop10_conf50.jpg") == ("IMG_0066", 10, 50)


# --------------------------------------------------------------------------- #
# species_from_classes
# --------------------------------------------------------------------------- #

def test_species_from_classes_common_name():
    classes = ["uuid;mammalia;cetartiodactyla;suidae;sus;scrofa;wild boar"]
    assert species_from_classes(classes) == "wild boar"


def test_species_from_classes_empty():
    assert species_from_classes([]) == ""


def test_species_from_classes_no_semicolon():
    assert species_from_classes(["blank"]) == "blank"


def test_species_from_classes_empty_first_element():
    assert species_from_classes([""]) == ""


# --------------------------------------------------------------------------- #
# compute_funnel (tiny in-memory fixture)
# --------------------------------------------------------------------------- #

def _tiny_md():
    return {
        "detection_categories": {"1": "animal", "2": "person", "3": "vehicle"},
        "images": [
            {"file": "f1.jpg", "detections": []},                      # empty
            {"file": "f2.jpg", "detections": []},                      # empty
            {"file": "f3.jpg", "detections": [                          # animal
                {"category": "1", "conf": 0.9, "bbox": [0, 0, 1, 1]},
            ]},
            {"file": "f4.jpg", "detections": [                          # animal x2
                {"category": "1", "conf": 0.8, "bbox": [0, 0, 1, 1]},
                {"category": "1", "conf": 0.6, "bbox": [0, 0, 1, 1]},
            ]},
            {"file": "f5.jpg", "detections": [                          # person only
                {"category": "2", "conf": 0.7, "bbox": [0, 0, 1, 1]},
            ]},
            {"file": "f6.jpg", "detections": [                          # animal + person
                {"category": "1", "conf": 0.9, "bbox": [0, 0, 1, 1]},
                {"category": "2", "conf": 0.5, "bbox": [0, 0, 1, 1]},
            ]},
            {"file": "f7.jpg", "detections": [                          # vehicle only
                {"category": "3", "conf": 0.4, "bbox": [0, 0, 1, 1]},
            ]},
        ],
    }


def _tiny_cleaned():
    # On-disk kept set is authoritative (D3): two records, three kept detections.
    return {
        "detection_categories": {"1": "animal"},
        "predictions": [
            {"filepath": "animal_images/f3.jpg", "detections": [
                {"category": "1", "conf": 0.9, "bbox": [0, 0, 1, 1]},
            ]},
            {"filepath": "animal_images/f4.jpg", "detections": [
                {"category": "1", "conf": 0.8, "bbox": [0, 0, 1, 1]},
                {"category": "1", "conf": 0.6, "bbox": [0, 0, 1, 1]},
            ]},
            {"filepath": "animal_images/f1.jpg", "detections": []},  # no kept dets
        ],
    }


def test_compute_funnel_d7b_separation():
    funnel = compute_funnel(_tiny_md(), _tiny_cleaned())
    assert funnel["total_frames"] == 7
    assert funnel["frames_with_any_detection"] == 5  # f3,f4,f5,f6,f7
    # D7b: empty = ZERO detections of ANY category.
    assert funnel["empty_frames"] == 2  # f1,f2
    assert funnel["empty_frames"] == (
        funnel["total_frames"] - funnel["frames_with_any_detection"]
    )
    assert funnel["frames_with_animal"] == 3  # f3,f4,f6
    # person/vehicle-only, non-empty, no animal: f5, f7 (NOT f6 which has an animal).
    assert funnel["person_or_vehicle_frames"] == 2
    # Never folded into empty: empty_frames stays exactly the zero-detection count, so
    # the person/vehicle frames are NOT inside it even when the two tallies coincide.
    assert funnel["empty_frames"] == 2  # only f1,f2 (zero detections), never f5/f7
    assert (
        funnel["empty_frames"]
        + funnel["person_or_vehicle_frames"]
        + funnel["frames_with_animal"]
        == funnel["total_frames"]
    )  # the three frame buckets partition all frames, disjointly
    assert funnel["person_detections"] == 2  # f5, f6
    assert funnel["vehicle_detections"] == 1  # f7
    assert funnel["animal_detections_raw"] == 4  # f3(1)+f4(2)+f6(1)


def test_compute_funnel_d3_trust_on_disk():
    funnel = compute_funnel(_tiny_md(), _tiny_cleaned())
    # Straight from the cleaned fixture, not recomputed from raw MD.
    assert funnel["animal_detections_kept"] == 3
    assert funnel["kept_frames"] == 2


def test_compute_funnel_pct_empty():
    funnel = compute_funnel(_tiny_md(), _tiny_cleaned())
    assert funnel["pct_empty_removed"] == round(2 / 7 * 100, 1)


# --------------------------------------------------------------------------- #
# compute_species_counts (tiny in-memory fixture)
# --------------------------------------------------------------------------- #

def _tiny_classified():
    def cls(name):
        return {"classifications": {"classes": [f"uuid;x;{name}"], "scores": [0.9]}}

    return {
        "predictions": [
            {"filepath": "animal_images/a.jpg", "detections": [cls("wild boar"), cls("blank")]},
            {"filepath": "animal_images/b.jpg", "detections": [cls("wild boar")]},
            {"filepath": "animal_images/c.jpg", "detections": [cls("red fox"), cls("blank")]},
            {"filepath": "animal_images/d.jpg", "detections": []},  # no classifications
        ],
    }


def test_compute_species_counts_excludes_blank_by_default():
    counts = compute_species_counts(_tiny_classified())
    assert counts == {"wild boar": 2, "red fox": 1}
    assert "blank" not in counts


def test_compute_species_counts_include_blank():
    counts = compute_species_counts(_tiny_classified(), include_blank=True)
    assert counts["blank"] == 2
    assert counts["wild boar"] == 2
    assert counts["red fox"] == 1


# --------------------------------------------------------------------------- #
# End-to-end on the real Medvednica dump
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not HAVE_REAL_DATA, reason="real MedvednicaDS dump not present")
def test_end_to_end_real_data(tmp_path):
    out = str(tmp_path / "report")
    summary = generate_medvednica_report(
        DATA_DIR, out, top_k_species=12, n_example_crops=12, seed=0
    )

    f = summary["funnel"]
    assert f["total_frames"] == 8208
    assert f["frames_with_any_detection"] == 6177
    assert f["frames_with_animal"] == 5801
    assert f["person_detections"] == 876
    assert f["vehicle_detections"] == 4
    assert f["animal_detections_raw"] == 11795
    assert f["animal_detections_kept"] == 2049
    assert f["kept_frames"] == 1866
    # D7b
    assert f["empty_frames"] == 2031 == f["total_frames"] - f["frames_with_any_detection"]
    assert f["pct_empty_removed"] == round(2031 / 8208 * 100, 1)
    assert isinstance(f["person_or_vehicle_frames"], int)

    sp = summary["species"]
    assert sp["total_classified_detections"] == 2049
    assert sp["blank_detections"] == 445
    assert sp["real_species_detections"] == 1604
    assert sp["counts"]["wild boar"] == 638
    assert sp["counts"]["european roe deer"] == 349
    assert sp["counts"]["red fox"] == 101
    assert "blank" not in sp["counts"]
    assert len(sp["top_k"]) <= 12
    counts_seq = [e["count"] for e in sp["top_k"]]
    assert counts_seq == sorted(counts_seq, reverse=True)

    # Output files exist and are non-empty.
    for fn in (
        "medvednica_report.md",
        os.path.join("figures", "detection_funnel.png"),
        os.path.join("figures", "species_breakdown.png"),
        os.path.join("figures", "example_crops.png"),
        "medvednica_summary.json",
    ):
        p = os.path.join(out, fn)
        assert os.path.exists(p) and os.path.getsize(p) > 0

    # examples non-empty, each crop_path exists.
    assert summary["examples"]
    for ex in summary["examples"]:
        assert os.path.exists(ex["crop_path"])

    # JSON round-trips and headline number is in the Markdown.
    with open(os.path.join(out, "medvednica_summary.json")) as fh:
        on_disk = json.load(fh)
    assert on_disk["schema_version"] == 1
    assert on_disk["dataset"] == "MedvednicaDS"
    with open(os.path.join(out, "medvednica_report.md")) as fh:
        md = fh.read()
    assert "8208" in md or "8,208" in md
    assert "figures/detection_funnel.png" in md.replace(os.sep, "/")


@pytest.mark.skipif(not HAVE_REAL_DATA, reason="real MedvednicaDS dump not present")
def test_determinism_examples(tmp_path):
    a = generate_medvednica_report(DATA_DIR, str(tmp_path / "a"), seed=0)
    b = generate_medvednica_report(DATA_DIR, str(tmp_path / "b"), seed=0)
    assert [e["crop_path"] for e in a["examples"]] == [e["crop_path"] for e in b["examples"]]


@pytest.mark.skipif(not HAVE_REAL_DATA, reason="real MedvednicaDS dump not present")
def test_species_filter_target_block(tmp_path):
    summary = generate_medvednica_report(
        DATA_DIR, str(tmp_path / "lynx"), species_filter=["eurasian lynx"], use_store=False
    )
    target = summary["target_species"]
    assert target["filter"] == ["eurasian lynx"]
    assert isinstance(target["detections"], int)
    assert isinstance(target["frames"], int)
    with open(os.path.join(str(tmp_path / "lynx"), "medvednica_report.md")) as fh:
        md = fh.read()
    assert "eurasian lynx" in md.lower()


@pytest.mark.skipif(not HAVE_REAL_DATA, reason="real MedvednicaDS dump not present")
def test_missing_required_json_raises(tmp_path):
    # A data dir without the required artifacts must raise FileNotFoundError.
    empty = tmp_path / "empty_ds"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        generate_medvednica_report(str(empty), str(tmp_path / "out"))
