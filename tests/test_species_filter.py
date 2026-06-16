"""Unit tests for reid_demo.species_filter (T03).

Self-contained: tiny in-repo SpeciesNet-shaped JSON fixtures + a hand-seeded T01
store. No model, no GPU, no network. One opportunistic smoke check against the real
``data/MedvednicaDS/animals_classified.json`` (skipped if absent).
"""

import json
import os
import sys

import pytest

# Make the repo root importable regardless of how pytest is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reid_demo.species_filter import (  # noqa: E402
    classify_and_filter,
    ingest_speciesnet_json,
    set_known_species,
    is_target_species,
    SpeciesFilterResult,
    TARGET_SPECIES_ALIASES,
    BBOX_MATCH_TOLERANCE,
)
from reid_demo.store import (  # noqa: E402
    connect,
    query_records,
    get_record,
    update_extra,
    make_record_id,
    DetectionRecord,
    upsert_record,
)

REAL_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "MedvednicaDS", "animals_classified.json",
)

BOAR = ("d372cda5-a8ca-4b7b-97ed-4e4fab9c9b4b;mammalia;cetartiodactyla;"
        "suidae;sus;scrofa;wild boar")
ROE = ("aaaa;mammalia;cetartiodactyla;cervidae;capreolus;capreolus;"
       "european roe deer")
LYNX = "bbbb;mammalia;carnivora;felidae;lynx;lynx;eurasian lynx"
BOBCAT = "cccc;mammalia;carnivora;felidae;lynx;rufus;bobcat"
LEOPARD = "dddd;mammalia;carnivora;felidae;panthera;pardus;leopard"
TIGER = "eeee;mammalia;carnivora;felidae;panthera;tigris;tiger"
BLANK = "f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank"


# --------------------------------------------------------------------------- #
# is_target_species
# --------------------------------------------------------------------------- #

def test_target_matching_common_name():
    assert is_target_species("eurasian lynx", "lynx")
    assert is_target_species("canada lynx", "lynx")
    assert is_target_species("EURASIAN LYNX", "lynx")  # case-insensitive
    assert not is_target_species("wild boar", "lynx")
    assert is_target_species("leopard", "leopard")
    assert is_target_species("tiger", "tiger")


def test_target_matching_full_taxonomy_genus():
    # bobcat is genus 'lynx' -> kept under target 'lynx'.
    assert is_target_species(BOBCAT, "lynx")
    assert is_target_species(LYNX, "lynx")
    assert is_target_species(LEOPARD, "leopard")
    assert is_target_species(TIGER, "tiger")
    # genus panthera + epithet must disambiguate leopard vs tiger.
    assert not is_target_species(LEOPARD, "tiger")
    assert not is_target_species(TIGER, "leopard")
    assert not is_target_species(BOAR, "lynx")
    assert not is_target_species(BLANK, "lynx")


def test_target_matching_unknown_target_raises():
    with pytest.raises(KeyError):
        is_target_species("eurasian lynx", "wolverine")


def test_target_aliases_shape():
    for key in ("lynx", "leopard", "tiger"):
        assert key in TARGET_SPECIES_ALIASES
        assert isinstance(TARGET_SPECIES_ALIASES[key], set)


# --------------------------------------------------------------------------- #
# Fixtures: a seeded store + matching SpeciesNet JSON
# --------------------------------------------------------------------------- #

def _seed_record(conn, stem, idx, bbox, dataset="TestDS"):
    rid = make_record_id(stem, idx)
    rec = DetectionRecord(
        record_id=rid, source_image=f"imgs/{stem}.JPG", source_stem=stem,
        det_index=idx, crop_path=f"crops/{rid}.jpg",
        bbox_x=bbox[0], bbox_y=bbox[1], bbox_w=bbox[2], bbox_h=bbox[3],
        detector_conf=0.9, dataset=dataset,
    )
    upsert_record(conn, rec)
    return rid


@pytest.fixture
def seeded(tmp_path):
    """A store with: one boar, one roe deer (two dets in ONE frame, stored in
    non-positional bbox order), one lynx, one blank, plus one row with NO matching
    JSON detection (stays NULL)."""
    db = str(tmp_path / "t03.sqlite")
    conn = connect(db)

    # IMG_1: two animals. Store them in REVERSED bbox order vs the JSON so a positional
    # join would mis-assign. crop1 = roe (right side), crop2 = boar (left side).
    roe_bbox = (0.60, 0.20, 0.20, 0.30)
    boar_bbox = (0.10, 0.10, 0.25, 0.40)
    rid_roe = _seed_record(conn, "IMG_1", 1, roe_bbox)
    rid_boar = _seed_record(conn, "IMG_1", 2, boar_bbox)

    # IMG_2: one lynx.
    lynx_bbox = (0.30, 0.30, 0.20, 0.20)
    rid_lynx = _seed_record(conn, "IMG_2", 1, lynx_bbox)

    # IMG_3: one blank.
    blank_bbox = (0.0, 0.0, 1.0, 1.0)
    rid_blank = _seed_record(conn, "IMG_3", 1, blank_bbox)

    # IMG_4: a row with NO matching JSON detection -> must stay species NULL.
    rid_null = _seed_record(conn, "IMG_4", 1, (0.5, 0.5, 0.1, 0.1))

    # SpeciesNet JSON. IMG_1's detections listed boar-first (the OPPOSITE of stored
    # crop order) so the join must be bbox-driven, not positional.
    payload = {"predictions": [
        {"filepath": "animal_images/IMG_1.JPG", "detections": [
            {"category": "1", "conf": 0.87, "bbox": list(boar_bbox),
             "classifications": {"classes": [BOAR], "scores": [0.87]}},
            {"category": "1", "conf": 0.71, "bbox": list(roe_bbox),
             "classifications": {"classes": [ROE], "scores": [0.71]}},
        ]},
        {"filepath": "animal_images/IMG_2.JPG", "detections": [
            {"category": "1", "conf": 0.66, "bbox": list(lynx_bbox),
             "classifications": {"classes": [LYNX], "scores": [0.66]}},
        ]},
        {"filepath": "animal_images/IMG_3.JPG", "detections": [
            {"category": "1", "conf": 0.40, "bbox": list(blank_bbox),
             "classifications": {"classes": [BLANK], "scores": [0.40]}},
        ]},
        # A detection for a frame/box with no stored row -> skipped_unmatched.
        {"filepath": "animal_images/IMG_9.JPG", "detections": [
            {"category": "1", "conf": 0.55, "bbox": [0.1, 0.1, 0.1, 0.1],
             "classifications": {"classes": [BOAR], "scores": [0.55]}},
        ]},
    ]}
    json_path = str(tmp_path / "preds.json")
    with open(json_path, "w") as fh:
        json.dump(payload, fh)

    return {
        "conn": conn, "json_path": json_path, "db": db,
        "rid_roe": rid_roe, "rid_boar": rid_boar, "rid_lynx": rid_lynx,
        "rid_blank": rid_blank, "rid_null": rid_null,
    }


# --------------------------------------------------------------------------- #
# ingest_speciesnet_json
# --------------------------------------------------------------------------- #

def test_ingest_bbox_join_not_positional(seeded):
    conn = seeded["conn"]
    res = ingest_speciesnet_json(
        conn, seeded["json_path"], dataset="TestDS", target_species="lynx",
    )
    # crop1 was stored as the ROE bbox; despite the JSON listing boar first, the
    # nearest-bbox join must assign roe to crop1 and boar to crop2.
    roe = get_record(conn, seeded["rid_roe"])
    boar = get_record(conn, seeded["rid_boar"])
    assert roe.species == "european roe deer", roe.species
    assert boar.species == "wild boar", boar.species
    assert roe.species_class == ROE
    assert boar.species_class == BOAR
    assert roe.species_conf == pytest.approx(0.71)
    assert boar.species_conf == pytest.approx(0.87)


def test_ingest_writes_fields_and_keep_flag(seeded):
    conn = seeded["conn"]
    res = ingest_speciesnet_json(
        conn, seeded["json_path"], dataset="TestDS", target_species="lynx",
    )
    lynx = get_record(conn, seeded["rid_lynx"])
    assert lynx.species == "eurasian lynx"
    assert lynx.species_conf == pytest.approx(0.66)
    assert json.loads(lynx.extra_json)["species_kept"] == 1

    boar = get_record(conn, seeded["rid_boar"])
    assert json.loads(boar.extra_json)["species_kept"] == 0

    blank = get_record(conn, seeded["rid_blank"])
    assert blank.species == "blank"
    assert json.loads(blank.extra_json)["species_kept"] == 0


def test_ingest_unmatched_and_unclassified_counts(seeded):
    conn = seeded["conn"]
    res = ingest_speciesnet_json(
        conn, seeded["json_path"], dataset="TestDS", target_species="lynx",
    )
    # IMG_9 detection has no stored row -> skipped_unmatched.
    assert res.skipped_unmatched == 1
    # IMG_4's row never received a classification -> NULL, counted unclassified.
    assert res.n_unclassified == 1
    null_row = get_record(conn, seeded["rid_null"])
    assert null_row.species is None


def test_ingest_count_invariants_and_breakdown(seeded):
    conn = seeded["conn"]
    res = ingest_speciesnet_json(
        conn, seeded["json_path"], dataset="TestDS", target_species="lynx",
    )
    assert res.n_classified == 4  # roe, boar, lynx, blank
    assert res.n_kept + res.n_dropped == res.n_classified
    kept_rows = [r for r in query_records(conn, dataset="TestDS")
                 if r.species is not None and json.loads(r.extra_json).get("species_kept") == 1]
    assert res.n_kept == len(kept_rows)
    assert res.n_kept == 1  # only the lynx
    assert seeded["rid_lynx"] in res.kept_record_ids
    # Breakdown over all classified rows, includes blank + non-target species.
    assert res.species_breakdown.get("wild boar") == 1
    assert res.species_breakdown.get("european roe deer") == 1
    assert res.species_breakdown.get("blank") == 1
    assert res.species_breakdown.get("eurasian lynx") == 1


def test_keep_threshold_honored(seeded):
    conn = seeded["conn"]
    # lynx conf is 0.66; threshold 0.8 must DROP it while still labeling.
    res = ingest_speciesnet_json(
        conn, seeded["json_path"], dataset="TestDS", target_species="lynx",
        keep_threshold=0.8,
    )
    lynx = get_record(conn, seeded["rid_lynx"])
    assert lynx.species == "eurasian lynx"  # still labeled
    assert json.loads(lynx.extra_json)["species_kept"] == 0
    assert res.n_kept == 0


def test_drop_nontarget_only_affects_kept_ids(seeded):
    conn = seeded["conn"]
    res = ingest_speciesnet_json(
        conn, seeded["json_path"], dataset="TestDS", target_species="lynx",
        drop_nontarget=True,
    )
    # No rows deleted.
    assert len(query_records(conn, dataset="TestDS")) == 5
    # kept_record_ids only contains kept (lynx).
    assert res.kept_record_ids == [seeded["rid_lynx"]]


def test_extra_json_preserves_existing_keys(seeded):
    conn = seeded["conn"]
    # Pre-set a key on the lynx row; ingest must preserve it.
    update_extra(conn, seeded["rid_lynx"], "pre_existing", "keepme")
    ingest_speciesnet_json(
        conn, seeded["json_path"], dataset="TestDS", target_species="lynx",
    )
    blob = json.loads(get_record(conn, seeded["rid_lynx"]).extra_json)
    assert blob["pre_existing"] == "keepme"
    assert blob["species_kept"] == 1


def test_ingest_idempotent(seeded):
    conn = seeded["conn"]
    r1 = ingest_speciesnet_json(
        conn, seeded["json_path"], dataset="TestDS", target_species="lynx",
    )
    n1 = len(query_records(conn, dataset="TestDS"))
    r2 = ingest_speciesnet_json(
        conn, seeded["json_path"], dataset="TestDS", target_species="lynx",
    )
    n2 = len(query_records(conn, dataset="TestDS"))
    assert n1 == n2 == 5
    assert (r1.n_classified, r1.n_kept, r1.n_dropped, r1.skipped_unmatched) == \
           (r2.n_classified, r2.n_kept, r2.n_dropped, r2.skipped_unmatched)
    # Same species values second time.
    assert get_record(conn, seeded["rid_lynx"]).species == "eurasian lynx"


def test_ingest_unknown_target_raises(seeded):
    with pytest.raises(KeyError):
        ingest_speciesnet_json(
            seeded["conn"], seeded["json_path"], dataset="TestDS",
            target_species="wolverine",
        )


def test_classify_and_filter_delegates_to_json(seeded):
    conn = seeded["conn"]
    res = classify_and_filter(
        conn, dataset="TestDS", target_species="lynx",
        reuse_existing_json=seeded["json_path"],
    )
    assert isinstance(res, SpeciesFilterResult)
    assert res.n_classified == 4


# --------------------------------------------------------------------------- #
# Live path degradation
# --------------------------------------------------------------------------- #

def test_live_path_raises_clear_runtimeerror(tmp_path, monkeypatch):
    """With no reuse_existing_json and no SpeciesNet CLI, a clear RuntimeError naming
    SpeciesNet is raised (not a raw traceback)."""
    db = str(tmp_path / "t03live.sqlite")
    conn = connect(db)
    # Seed one row with a real crop file so we reach the CLI invocation.
    pytest.importorskip("PIL")
    from PIL import Image
    crop = tmp_path / "c.jpg"
    Image.new("RGB", (20, 20), (1, 2, 3)).save(crop, "JPEG")
    rec = DetectionRecord(
        record_id=make_record_id("IMG_X", 1), source_image="x", source_stem="IMG_X",
        det_index=1, crop_path=str(crop), bbox_x=0, bbox_y=0, bbox_w=1, bbox_h=1,
        dataset="LiveDS",
    )
    upsert_record(conn, rec)

    # Force the CLI to be "missing".
    import subprocess as _sp
    def _boom(*a, **k):
        raise FileNotFoundError("speciesnet not installed")
    monkeypatch.setattr(_sp, "run", _boom)

    with pytest.raises(RuntimeError, match="SpeciesNet"):
        classify_and_filter(conn, dataset="LiveDS", target_species="lynx")


# --------------------------------------------------------------------------- #
# set_known_species
# --------------------------------------------------------------------------- #

def test_set_known_species(tmp_path):
    db = str(tmp_path / "t03known.sqlite")
    conn = connect(db)
    for i in range(3):
        rec = DetectionRecord(
            record_id=make_record_id(f"L{i}", 1), source_image="x",
            source_stem=f"L{i}", det_index=1, crop_path="x",
            bbox_x=0, bbox_y=0, bbox_w=1, bbox_h=1, dataset="LeopardID2022",
            gt_identity="ID7", orientation="left",
        )
        upsert_record(conn, rec)

    n = set_known_species(conn, dataset="LeopardID2022", species="leopard")
    assert n == 3
    for r in query_records(conn, dataset="LeopardID2022"):
        assert r.species == "leopard"
        assert r.species_class == "leopard"
        assert json.loads(r.extra_json)["species_kept"] == 1
        # gt_identity / orientation untouched (D1).
        assert r.gt_identity == "ID7"
        assert r.orientation == "left"


# --------------------------------------------------------------------------- #
# Smoke check against the real Medvednica JSON (no model needed)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not os.path.exists(REAL_JSON), reason="real Medvednica JSON absent")
def test_real_medvednica_smoke(tmp_path):
    """Seed a store directly from the real JSON's detections, then run T03 on it.
    Verifies the breakdown contains expected real species without any GPU/model."""
    db = str(tmp_path / "t03real.sqlite")
    conn = connect(db)
    with open(REAL_JSON) as fh:
        payload = json.load(fh)

    # Seed one T01 row per JSON detection that has a classification, using the JSON's
    # own bbox (so the nearest-match join is exact). det_index is a synthetic 1-based
    # per-frame counter here purely to build distinct record_ids (T03 does NOT rely on
    # it for the join).
    from pathlib import Path as _P
    seeded = 0
    for pred in payload["predictions"]:
        stem = _P(pred.get("filepath", "")).stem
        if not stem:
            continue
        idx = 0
        for det in pred.get("detections", []):
            if not det.get("classifications"):
                continue
            idx += 1
            bbox = det["bbox"]
            rec = DetectionRecord(
                record_id=make_record_id(stem, idx), source_image=pred["filepath"],
                source_stem=stem, det_index=idx, crop_path="x",
                bbox_x=bbox[0], bbox_y=bbox[1], bbox_w=bbox[2], bbox_h=bbox[3],
                dataset="MedvednicaDS",
            )
            upsert_record(conn, rec)
            seeded += 1
        if seeded > 1500:  # keep the test quick; a representative slice is enough
            break

    res = classify_and_filter(
        conn, dataset="MedvednicaDS", target_species="lynx",
        reuse_existing_json=REAL_JSON,
    )
    assert "wild boar" in res.species_breakdown
    assert "european roe deer" in res.species_breakdown
    assert res.n_classified > 0
    assert res.n_kept + res.n_dropped == res.n_classified
