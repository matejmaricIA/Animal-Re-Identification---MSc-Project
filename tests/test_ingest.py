"""Unit tests for reid_demo.ingest (T02).

Self-contained: tiny synthetic detection JSONs + PIL-generated dummy images, and a
synthetic `load_dataset` stub (monkeypatched) for the B-track. NO network, NO model
download, NO torch / megadetector import required to run any of these.
"""

import json
import os
import subprocess
import sys

import pytest

# Make the repo root importable regardless of how pytest is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reid_demo.ingest import (  # noqa: E402
    ingest,
    ingest_from_images,
    ingest_wildlife_dataset,
    load_detection_frames,
    crop_for_detection,
    resolve_metadata,
    DEFAULT_MD_JSON,
    DEFAULT_IMAGES_DIR,
    DEFAULT_METADATA_CSV,
    DEFAULT_EXISTING_CROPS,
    DEFAULT_CROPS_OUT,
    DEFAULT_DATASET,
    DEFAULT_CONF_THRESHOLD,
    ANIMAL_CATEGORY_ID,
)
from reid_demo.store import (  # noqa: E402
    connect,
    query_records,
    get_record,
    make_record_id,
    ORIENTATIONS,
)

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures: synthetic images + detection JSONs
# --------------------------------------------------------------------------- #

def _make_image(path, size=(200, 100), color=(120, 90, 60)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", size, color).save(path, "JPEG", quality=90)


@pytest.fixture
def images_dir(tmp_path):
    d = tmp_path / "animal_images"
    d.mkdir()
    # Three frames: one with two animals, one with one animal, one empty-on-disk too.
    _make_image(str(d / "IMG_A.JPG"))
    _make_image(str(d / "IMG_B.JPG"))
    _make_image(str(d / "IMG_C.JPG"))
    return d


@pytest.fixture
def md_json(tmp_path):
    """MegaDetector-results format with animals, a person, a vehicle, an empty frame,
    and a sub-threshold animal."""
    payload = {
        "info": {"detector": "synthetic"},
        "detection_categories": {"1": "animal", "2": "person", "3": "vehicle"},
        "images": [
            # Empty frame (no detections) -> contributes to frames_empty.
            {"file": "Camera 1/IMG_EMPTY.JPG", "detections": []},
            # Two animals (both above 0.5) + one person -> 2 crops, 1 person.
            {"file": "Camera 1/IMG_A.JPG", "detections": [
                {"category": "1", "conf": 0.91, "bbox": [0.1, 0.1, 0.3, 0.4]},
                {"category": "2", "conf": 0.80, "bbox": [0.0, 0.0, 1.0, 1.0]},
                {"category": "1", "conf": 0.77, "bbox": [0.5, 0.2, 0.2, 0.3]},
            ]},
            # One animal above threshold + one below + a vehicle -> 1 crop, 1 below, 1 vehicle.
            {"file": "Camera 2/IMG_B.JPG", "detections": [
                {"category": "1", "conf": 0.60, "bbox": [0.2, 0.2, 0.25, 0.25]},
                {"category": "1", "conf": 0.10, "bbox": [0.9, 0.9, 0.05, 0.05]},
                {"category": "3", "conf": 0.95, "bbox": [0.0, 0.0, 0.5, 0.5]},
            ]},
            # A frame whose only animal is sub-threshold -> empty (no kept crop).
            {"file": "Camera 1/IMG_SUB.JPG", "detections": [
                {"category": "1", "conf": 0.30, "bbox": [0.1, 0.1, 0.2, 0.2]},
            ]},
        ],
    }
    p = tmp_path / "md_results.json"
    p.write_text(json.dumps(payload))
    return p


@pytest.fixture
def flat_json(tmp_path):
    """Flat animal_detections.json format."""
    payload = {
        "IMG_A.JPG": [
            {"bbox": [0.1, 0.1, 0.3, 0.4], "confidence": 0.91},
            {"bbox": [0.5, 0.2, 0.2, 0.3], "confidence": 0.40},  # below threshold
        ],
        "IMG_B.JPG": [
            {"bbox": [0.2, 0.2, 0.25, 0.25], "confidence": 0.88},
        ],
    }
    p = tmp_path / "animal_detections.json"
    p.write_text(json.dumps(payload))
    return p


@pytest.fixture
def metadata_csv(tmp_path):
    p = tmp_path / "trail_cam_data.csv"
    p.write_text(
        "filepath,camera,num_detections,datetime,temperature\n"
        "/x/animal_images/IMG_A.JPG,cam_alpha,2,2025-06-02 04:27:51,Not available\n"
        "/x/animal_images/IMG_B.JPG,unknown_camera,1,2025-06-03 04:05:20,Not available\n"
    )
    return p


# --------------------------------------------------------------------------- #
# load_detection_frames
# --------------------------------------------------------------------------- #

def test_load_md_results_filters_and_indexes(md_json):
    frames = load_detection_frames(str(md_json), conf_threshold=0.5)
    assert len(frames) == 4  # all source frames returned (empties included)

    by_name = {f["source_basename"]: f for f in frames}
    assert set(by_name) == {"IMG_EMPTY.JPG", "IMG_A.JPG", "IMG_B.JPG", "IMG_SUB.JPG"}

    # Empty frame.
    assert by_name["IMG_EMPTY.JPG"]["animal_dets"] == []

    # IMG_A: two animals kept, 1-based det_index, person counted separately.
    a = by_name["IMG_A.JPG"]
    assert [d["det_index"] for d in a["animal_dets"]] == [1, 2]
    assert all(d["conf"] >= 0.5 for d in a["animal_dets"])
    assert a["n_person"] == 1 and a["n_vehicle"] == 0 and a["n_below_threshold"] == 0
    assert a["camera_hint"] == "Camera 1"

    # IMG_B: 1 kept animal, 1 below threshold, 1 vehicle.
    b = by_name["IMG_B.JPG"]
    assert [d["det_index"] for d in b["animal_dets"]] == [1]
    assert b["n_vehicle"] == 1 and b["n_below_threshold"] == 1
    assert b["camera_hint"] == "Camera 2"

    # IMG_SUB: only animal is sub-threshold -> no kept det, counts as below.
    s = by_name["IMG_SUB.JPG"]
    assert s["animal_dets"] == [] and s["n_below_threshold"] == 1


def test_load_md_results_threshold_is_inclusive(md_json):
    # IMG_B has an animal at exactly conf 0.60; threshold 0.60 must KEEP it (>=).
    frames = load_detection_frames(str(md_json), conf_threshold=0.60)
    b = next(f for f in frames if f["source_basename"] == "IMG_B.JPG")
    assert [d["det_index"] for d in b["animal_dets"]] == [1]
    # threshold just above keeps nothing.
    frames2 = load_detection_frames(str(md_json), conf_threshold=0.601)
    b2 = next(f for f in frames2 if f["source_basename"] == "IMG_B.JPG")
    assert b2["animal_dets"] == []


def test_load_flat_format(flat_json):
    frames = load_detection_frames(str(flat_json), conf_threshold=0.5)
    by_name = {f["source_basename"]: f for f in frames}
    assert set(by_name) == {"IMG_A.JPG", "IMG_B.JPG"}
    # IMG_A: one above, one below threshold.
    a = by_name["IMG_A.JPG"]
    assert [d["det_index"] for d in a["animal_dets"]] == [1]
    assert a["n_below_threshold"] == 1
    assert a["camera_hint"] is None  # flat format has no camera info
    # IMG_B: single kept det.
    assert [d["det_index"] for d in by_name["IMG_B.JPG"]["animal_dets"]] == [1]


def test_load_unknown_format_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ValueError):
        load_detection_frames(str(p))


# --------------------------------------------------------------------------- #
# crop_for_detection
# --------------------------------------------------------------------------- #

def test_crop_writes_file(images_dir, tmp_path):
    src = str(images_dir / "IMG_A.JPG")
    out = str(tmp_path / "out" / "IMG_A__crop1.jpg")
    ret = crop_for_detection(src, (0.1, 0.1, 0.3, 0.4), out, write=True)
    assert ret == out and os.path.exists(out)
    # The crop is a valid, non-empty JPEG.
    with Image.open(out) as im:
        assert im.size[0] > 0 and im.size[1] > 0


def test_crop_no_write_creates_nothing(images_dir, tmp_path):
    src = str(images_dir / "IMG_A.JPG")
    out = str(tmp_path / "out" / "IMG_A__crop1.jpg")
    ret = crop_for_detection(src, (0.1, 0.1, 0.3, 0.4), out, write=False)
    assert ret == out and not os.path.exists(out)


def test_crop_reuses_existing(images_dir, tmp_path):
    src = str(images_dir / "IMG_A.JPG")
    existing = str(tmp_path / "legacy" / "IMG_A_crop1_conf92.jpg")
    _make_image(existing, size=(30, 30))
    out = str(tmp_path / "out" / "IMG_A__crop1.jpg")
    ret = crop_for_detection(src, (0.1, 0.1, 0.3, 0.4), out,
                             existing_crop_path=existing, write=True)
    assert ret == existing and not os.path.exists(out)


def test_crop_degenerate_box_raises(images_dir, tmp_path):
    src = str(images_dir / "IMG_A.JPG")
    out = str(tmp_path / "out" / "IMG_A__crop1.jpg")
    with pytest.raises(ValueError):
        crop_for_detection(src, (0.5, 0.5, 0.0, 0.0), out, write=True)


# --------------------------------------------------------------------------- #
# resolve_metadata
# --------------------------------------------------------------------------- #

def test_resolve_metadata(metadata_csv):
    meta = resolve_metadata(str(metadata_csv))
    assert meta["IMG_A.JPG"]["camera_id"] == "cam_alpha"
    assert meta["IMG_A.JPG"]["timestamp"] == "2025-06-02 04:27:51"
    assert meta["IMG_B.JPG"]["camera_id"] == "unknown_camera"


def test_resolve_metadata_missing_returns_empty():
    assert resolve_metadata(None) == {}
    assert resolve_metadata("/nonexistent/does_not_exist.csv") == {}


# --------------------------------------------------------------------------- #
# ingest (A-track) end-to-end on synthetic data
# --------------------------------------------------------------------------- #

def test_ingest_end_to_end(md_json, images_dir, metadata_csv, tmp_path):
    db = str(tmp_path / "t02.sqlite")
    crops_out = str(tmp_path / "crops")
    stats = ingest(
        md_json=str(md_json),
        images_dir=str(images_dir),
        metadata_csv=str(metadata_csv),
        existing_crops_dir=None,        # no legacy crops -> all freshly written
        crops_out_dir=crops_out,
        db_path=db,
        dataset="SynthDS",
        conf_threshold=0.5,
        write_crops=True,
    )

    # Stats: 4 frames, 2 empty (IMG_EMPTY + IMG_SUB), 3 animal crops, person/vehicle counted.
    # Below-threshold animals: IMG_B's 0.10 box + IMG_SUB's 0.30 box -> 2.
    assert stats["frames_total"] == 4
    assert stats["frames_empty"] == 2
    assert stats["frames_with_animals"] == 2
    assert stats["dets_animal"] == 3
    assert stats["dets_person"] == 1
    assert stats["dets_vehicle"] == 1
    assert stats["dets_below_threshold"] == 2
    assert stats["crops_written"] == 3
    assert stats["crops_reused"] == 0
    assert stats["records_upserted"] == 3
    assert stats["dataset"] == "SynthDS"
    pct_empty = stats["frames_empty"] / stats["frames_total"]
    assert abs(pct_empty - 0.5) < 1e-9

    # Records: A-track field-population contract.
    conn = connect(db)
    rows = query_records(conn, dataset="SynthDS")
    assert len(rows) == 3
    for r in rows:
        assert r.record_id == make_record_id(r.source_stem, r.det_index)
        assert os.path.exists(r.crop_path)
        assert r.species is None and r.species_conf is None and r.species_class is None
        assert r.embedding_ref is None and r.embedding_path is None
        assert r.cluster_id is None and r.is_candidate_new is None
        assert r.gt_identity is None
        assert r.review_status == "unreviewed"
        assert r.extra_json == "{}"
        assert r.orientation == "unknown" and r.orientation in ORIENTATIONS
        assert r.dataset == "SynthDS"
        # bbox values are the raw normalized detector values.
        assert 0.0 <= r.bbox_x <= 1.0 and 0.0 <= r.bbox_w <= 1.0

    # Camera/timestamp resolution: IMG_A -> CSV cam_alpha; det_index 1-based.
    a1 = get_record(conn, make_record_id("IMG_A", 1))
    assert a1 is not None
    assert a1.camera_id == "cam_alpha"
    assert a1.timestamp == "2025-06-02 04:27:51"
    assert a1.detector_conf == pytest.approx(0.91)
    a2 = get_record(conn, make_record_id("IMG_A", 2))
    assert a2 is not None and a2.det_index == 2

    # IMG_B falls back to CSV unknown_camera (CSV present for IMG_B).
    b1 = get_record(conn, make_record_id("IMG_B", 1))
    assert b1.camera_id == "unknown_camera"
    conn.close()


def test_ingest_idempotent(md_json, images_dir, metadata_csv, tmp_path):
    db = str(tmp_path / "t02.sqlite")
    crops_out = str(tmp_path / "crops")
    kwargs = dict(
        md_json=str(md_json), images_dir=str(images_dir),
        metadata_csv=str(metadata_csv), existing_crops_dir=None,
        crops_out_dir=crops_out, db_path=db, dataset="SynthDS",
    )
    s1 = ingest(**kwargs)
    conn = connect(db)
    n1 = len(query_records(conn, dataset="SynthDS"))
    u1 = get_record(conn, make_record_id("IMG_A", 1)).updated_at
    conn.close()

    s2 = ingest(**kwargs)
    conn = connect(db)
    n2 = len(query_records(conn, dataset="SynthDS"))
    conn.close()

    assert n1 == n2 == 3
    assert s1["records_upserted"] == s2["records_upserted"] == 3
    # updated_at advances (or at least does not error); created_at preserved.
    assert isinstance(u1, str)


def test_ingest_camera_hint_fallback(md_json, images_dir, tmp_path):
    """With no CSV, camera_id falls back to the JSON file-subfolder camera_hint."""
    db = str(tmp_path / "t02.sqlite")
    ingest(
        md_json=str(md_json), images_dir=str(images_dir),
        metadata_csv=None, existing_crops_dir=None,
        crops_out_dir=str(tmp_path / "crops"), db_path=db, dataset="SynthDS",
    )
    conn = connect(db)
    a1 = get_record(conn, make_record_id("IMG_A", 1))
    b1 = get_record(conn, make_record_id("IMG_B", 1))
    assert a1.camera_id == "Camera 1"   # from "Camera 1/IMG_A.JPG"
    assert b1.camera_id == "Camera 2"   # from "Camera 2/IMG_B.JPG"
    conn.close()


def test_ingest_reuses_legacy_crops(md_json, images_dir, tmp_path):
    """Legacy {stem}_crop{idx}_*.jpg in existing_crops_dir are reused, not re-cropped."""
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    # Provide a legacy crop for IMG_A crop1 only.
    _make_image(str(legacy / "IMG_A_crop1_conf91.jpg"), size=(40, 40))
    db = str(tmp_path / "t02.sqlite")
    crops_out = str(tmp_path / "crops")
    stats = ingest(
        md_json=str(md_json), images_dir=str(images_dir), metadata_csv=None,
        existing_crops_dir=str(legacy), crops_out_dir=crops_out,
        db_path=db, dataset="SynthDS",
    )
    assert stats["crops_reused"] == 1
    assert stats["crops_written"] == 2  # the other two are fresh
    conn = connect(db)
    a1 = get_record(conn, make_record_id("IMG_A", 1))
    assert a1.crop_path.endswith("IMG_A_crop1_conf91.jpg")  # reused legacy path
    assert os.path.exists(a1.crop_path)
    conn.close()


def test_ingest_limit_caps_working_frames(md_json, images_dir, tmp_path):
    """limit caps the number of frames that actually produce crops."""
    db = str(tmp_path / "t02.sqlite")
    stats = ingest(
        md_json=str(md_json), images_dir=str(images_dir), metadata_csv=None,
        existing_crops_dir=None, crops_out_dir=str(tmp_path / "crops"),
        db_path=db, dataset="SynthDS", limit=1,
    )
    # Only the first animal-bearing frame (IMG_A, 2 crops) is processed.
    assert stats["frames_with_animals"] == 2  # both still counted in the scan
    assert stats["records_upserted"] == 2     # but only IMG_A's crops written
    assert stats["frames_total"] == 4         # full scan for pct_empty


def test_ingest_no_crop_records_paths_without_files(md_json, images_dir, tmp_path):
    db = str(tmp_path / "t02.sqlite")
    crops_out = str(tmp_path / "crops")
    stats = ingest(
        md_json=str(md_json), images_dir=str(images_dir), metadata_csv=None,
        existing_crops_dir=None, crops_out_dir=crops_out,
        db_path=db, dataset="SynthDS", write_crops=False,
    )
    assert stats["records_upserted"] == 3
    # No crop files were written.
    assert not os.path.isdir(crops_out) or not os.listdir(crops_out)
    conn = connect(db)
    a1 = get_record(conn, make_record_id("IMG_A", 1))
    assert a1.crop_path.endswith("IMG_A__crop1.jpg")  # path recorded, file absent
    conn.close()


def test_ingest_missing_source_skips_record(md_json, tmp_path):
    """If the source frame is missing and there's no legacy crop, the record is skipped."""
    empty_images = tmp_path / "no_images"
    empty_images.mkdir()
    db = str(tmp_path / "t02.sqlite")
    stats = ingest(
        md_json=str(md_json), images_dir=str(empty_images), metadata_csv=None,
        existing_crops_dir=None, crops_out_dir=str(tmp_path / "crops"),
        db_path=db, dataset="SynthDS",
    )
    assert stats["records_upserted"] == 0
    assert stats["crops_missing_source"] == 3  # all 3 animal crops had no source


def test_ingest_flat_format_end_to_end(flat_json, images_dir, tmp_path):
    db = str(tmp_path / "t02.sqlite")
    stats = ingest(
        md_json=str(flat_json), images_dir=str(images_dir), metadata_csv=None,
        existing_crops_dir=None, crops_out_dir=str(tmp_path / "crops"),
        db_path=db, dataset="SynthDS",
    )
    # IMG_A: 1 kept (1 below); IMG_B: 1 kept -> 2 records.
    assert stats["records_upserted"] == 2
    assert stats["dets_below_threshold"] == 1


# --------------------------------------------------------------------------- #
# B-track: ingest_wildlife_dataset with a synthetic load_dataset stub
# --------------------------------------------------------------------------- #

@pytest.fixture
def wild_root(tmp_path, monkeypatch):
    """Build a fake WildlifeReID-10k image root + a stub utility_functions.load_dataset
    and constants.WILD_DATASET_PATH, so the B-track runs with NO real dataset."""
    pd = pytest.importorskip("pandas")
    root = tmp_path / "wild_root"
    (root / "imgs").mkdir(parents=True)

    rows = []
    # 4 identities, a few images each; one image has an empty orientation, one missing.
    spec = [
        ("idA", "left", "leopard", "imgs/a1.jpg"),
        ("idA", "right", "leopard", "imgs/a2.jpg"),
        ("idB", "front", "leopard", "imgs/b1.jpg"),
        ("idB", "", "leopard", "imgs/b2.jpg"),       # empty orientation -> 'unknown'
        ("idC", None, "leopard", "imgs/c1.jpg"),     # missing orientation -> 'unknown'
        ("idD", "down", "leopard", "imgs/d1.jpg"),
    ]
    for i, (ident, orient, sp, rel) in enumerate(spec):
        _make_image(str(root / rel), size=(60, 40))
        rows.append({
            "image_id": str(i),
            "identity": ident,
            "orientation": orient,
            "species": sp,
            "path": rel,
            "dataset": "FakeLeopard",
        })
    df = pd.DataFrame(rows)

    import utility_functions
    import constants
    monkeypatch.setattr(utility_functions, "load_dataset", lambda subset, **kw: df.copy())
    monkeypatch.setattr(constants, "WILD_DATASET_PATH", str(root))
    return root


def test_b_track_whole_frame_and_gt(wild_root, tmp_path):
    db = str(tmp_path / "t02b.sqlite")
    stats = ingest_wildlife_dataset("FakeLeopard", db_path=db)

    assert stats["images_total"] == 6
    assert stats["images_ingested"] == 6
    assert stats["identities_total"] == 4
    assert stats["identities_ingested"] == 4
    assert stats["records_upserted"] == 6
    assert stats["subset"] == "FakeLeopard"
    assert stats["dataset"] == "FakeLeopard"   # defaults to subset

    conn = connect(db)
    rows = query_records(conn, dataset="FakeLeopard")
    assert len(rows) == 6
    for r in rows:
        assert (r.bbox_x, r.bbox_y, r.bbox_w, r.bbox_h) == (0.0, 0.0, 1.0, 1.0)
        assert r.det_index == 1 and r.detector_conf == 1.0
        assert r.gt_identity is not None
        assert r.species == "leopard"
        assert r.orientation in ORIENTATIONS and r.orientation != ""
        assert r.embedding_ref is None and r.cluster_id is None
        assert r.review_status == "unreviewed"
        assert r.crop_path == r.source_image  # whole frame IS the crop
        assert os.path.exists(r.crop_path)
    conn.close()


def test_b_track_orientation_normalization(wild_root, tmp_path):
    db = str(tmp_path / "t02b.sqlite")
    ingest_wildlife_dataset("FakeLeopard", db_path=db)
    conn = connect(db)
    rows = {r.source_stem: r for r in query_records(conn, dataset="FakeLeopard")}
    # b2 had orientation '' and c1 had None -> both 'unknown'.
    assert rows["b2"].orientation == "unknown"
    assert rows["c1"].orientation == "unknown"
    # explicit ones preserved.
    assert rows["a1"].orientation == "left"
    assert rows["a2"].orientation == "right"
    assert rows["d1"].orientation == "down"
    conn.close()


def test_b_track_max_identities_cap(wild_root, tmp_path):
    db = str(tmp_path / "t02b.sqlite")
    stats = ingest_wildlife_dataset("FakeLeopard", max_identities=2, db_path=db)
    assert stats["identities_ingested"] == 2
    conn = connect(db)
    rows = query_records(conn, dataset="FakeLeopard")
    ids = {r.gt_identity for r in rows}
    assert len(ids) == 2
    # Deterministic: first 2 sorted identities are idA, idB.
    assert ids == {"idA", "idB"}
    conn.close()


def test_b_track_limit_after_max_identities(wild_root, tmp_path):
    db = str(tmp_path / "t02b.sqlite")
    stats = ingest_wildlife_dataset("FakeLeopard", max_identities=2, limit=3, db_path=db)
    assert stats["identities_ingested"] == 2
    assert stats["images_ingested"] == 3  # limit applied after identity filter
    assert stats["records_upserted"] == 3


def test_b_track_custom_dataset_label(wild_root, tmp_path):
    db = str(tmp_path / "t02b.sqlite")
    stats = ingest_wildlife_dataset("FakeLeopard", dataset="MyRun", db_path=db)
    assert stats["dataset"] == "MyRun"
    conn = connect(db)
    assert len(query_records(conn, dataset="MyRun")) == 6
    conn.close()


def test_b_track_enriches_missing_columns(tmp_path, monkeypatch):
    """If load_dataset returns rows WITHOUT orientation/species (the curated CSV case),
    the adapter still works: species->None, orientation->'unknown' when enrichment
    can't supply them."""
    pd = pytest.importorskip("pandas")
    root = tmp_path / "wild_root2"
    (root / "imgs").mkdir(parents=True)
    rows = []
    for i, ident in enumerate(["idX", "idY"]):
        rel = f"imgs/x{i}.jpg"
        _make_image(str(root / rel), size=(50, 50))
        rows.append({"image_id": str(i), "identity": ident, "path": rel,
                     "dataset": "BareDS"})
    df = pd.DataFrame(rows)  # NO orientation / species columns

    import utility_functions, constants
    monkeypatch.setattr(utility_functions, "load_dataset", lambda subset, **kw: df.copy())
    monkeypatch.setattr(constants, "WILD_DATASET_PATH", str(root))

    db = str(tmp_path / "t02b.sqlite")
    stats = ingest_wildlife_dataset("BareDS", db_path=db)
    assert stats["records_upserted"] == 2
    conn = connect(db)
    for r in query_records(conn, dataset="BareDS"):
        assert r.orientation == "unknown"   # missing column -> unknown
        assert r.species is None            # missing column -> None
        assert r.gt_identity in {"idX", "idY"}
    conn.close()


# --------------------------------------------------------------------------- #
# Import-surface: module must not pull in torch / megadetector
# --------------------------------------------------------------------------- #

def test_module_does_not_import_torch_or_megadetector():
    # Verify the lazy-import contract in a CLEAN interpreter: importing reid_demo.ingest
    # must not eagerly pull in torch/megadetector. Asserting on this process's sys.modules
    # is unreliable — sibling tests (e.g. test_embed/test_fisher) legitimately load torch
    # into the shared process first — so run a fresh subprocess.
    code = (
        "import sys; import reid_demo.ingest; "
        "heavy = [m for m in ('torch', 'megadetector') if m in sys.modules]; "
        "print(','.join(heavy)); "
        "sys.exit(1 if heavy else 0)"
    )
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=repo_root,
    )
    assert proc.returncode == 0, (
        f"reid_demo.ingest eagerly imported heavy deps: {proc.stdout.strip()!r}\n"
        f"stderr: {proc.stderr.strip()}"
    )


def test_constants_present():
    assert DEFAULT_MD_JSON == "data/MedvednicaDS/megadetector_results.json"
    assert DEFAULT_IMAGES_DIR == "data/MedvednicaDS/animal_images"
    assert DEFAULT_METADATA_CSV == "data/MedvednicaDS/trail_cam_data.csv"
    assert DEFAULT_EXISTING_CROPS == "data/MedvednicaDS/animal_crops"
    assert DEFAULT_CROPS_OUT == "data/reid_demo/crops"
    assert DEFAULT_DATASET == "MedvednicaDS"
    assert DEFAULT_CONF_THRESHOLD == 0.5
    assert ANIMAL_CATEGORY_ID == "1"
