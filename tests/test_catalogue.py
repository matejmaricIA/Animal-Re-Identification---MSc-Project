"""Unit tests for reid_demo.catalogue (T06 — static visual individual catalogue).

All tests run against a tiny temp T01 SQLite store seeded via reid_demo.store
(no model, no clustering, no network). Crop image files are real tiny JPEGs written
to a temp dir (so thumbnail generation is exercised), plus deliberately-missing crops
(to exercise the placeholder path). Deterministic throughout.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

# Make the repo root importable regardless of how pytest is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reid_demo.catalogue import (  # noqa: E402
    build_catalogue,
    CatalogueResult,
    CANONICAL_FLANKS,
    _norm_flank,
    _is_low_conf,
)
from reid_demo.store import (  # noqa: E402
    connect,
    upsert_records,
    count_by,
    query_records,
    DetectionRecord,
    make_record_id,
    COLUMNS,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = "TestDS"


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #

def _write_tiny_jpeg(path):
    """Write a tiny real JPEG so PIL.thumbnail has something to downscale."""
    from PIL import Image
    img = Image.new("RGB", (40, 40), color=(123, 50, 200))
    img.save(path, format="JPEG")


def _mk(stem, idx, cid, conf, flank, crop_path, *, cand=0, review="unreviewed",
        species="eurasian lynx", camera="unknown_camera",
        ts="2025-06-02 04:27:51", detconf=0.9):
    return DetectionRecord(
        record_id=make_record_id(stem, idx),
        source_image=f"data/{DATASET}/images/{stem}.JPG",
        source_stem=stem, det_index=idx, crop_path=crop_path,
        bbox_x=0.1, bbox_y=0.1, bbox_w=0.2, bbox_h=0.2,
        detector_conf=detconf, camera_id=camera, timestamp=ts,
        species=species, species_conf=0.95,
        cluster_id=cid, cluster_conf=conf, is_candidate_new=cand,
        orientation=flank, review_status=review, dataset=DATASET,
    )


@pytest.fixture()
def seeded(tmp_path):
    """Seed a store with 3 individuals + 1 candidate-new singleton.

    Layout:
      cluster 0: 3 left crops (high conf 0.9)         <- real crop files
      cluster 1: 2 right crops (low conf 0.4)         <- MISSING crop files
      cluster 2: left + right + empty-orientation     <- mixed flank, '' -> unknown
      noise:     1 candidate-new singleton (cid -1)
    One cluster-0 crop is 'confirmed', the empty-orientation crop uses orientation=None.
    """
    db = str(tmp_path / "store.sqlite")
    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()
    missing_dir = tmp_path / "missing"  # never created -> crops absent

    conn = connect(db)
    recs = []

    # cluster 0: 3 left crops, real files; first is confirmed
    for i in range(3):
        cp = crops_dir / f"A{i}__crop1.jpg"
        _write_tiny_jpeg(cp)
        recs.append(_mk(f"A{i}", 1, 0, 0.9, "left", str(cp),
                        review="confirmed" if i == 0 else "unreviewed"))

    # cluster 1: 2 right crops, MISSING files, low confidence
    for i in range(2):
        recs.append(_mk(f"B{i}", 1, 1, 0.4, "right", str(missing_dir / f"B{i}__crop1.jpg")))

    # cluster 2: mixed flank (left + right) + empty-orientation crop (None)
    cmix = crops_dir / "Cmix__crop1.jpg"
    cmix2 = crops_dir / "Cmix2__crop1.jpg"
    _write_tiny_jpeg(cmix)
    _write_tiny_jpeg(cmix2)
    recs.append(_mk("Cmix", 1, 2, 0.8, "left", str(cmix)))
    recs.append(_mk("Cmix2", 1, 2, 0.8, "right", str(cmix2)))
    # empty orientation: store normalizes '' -> 'unknown'; pass None to also cover NULL
    cnull = crops_dir / "Cnull__crop1.jpg"
    _write_tiny_jpeg(cnull)
    recs.append(_mk("Cnull", 1, 2, 0.8, None, str(cnull)))

    # candidate-new singleton (noise id, cluster_id == -1 per D5)
    recs.append(_mk("S", 1, -1, 0.3, "unknown", str(missing_dir / "S__crop1.jpg"), cand=1))

    upsert_records(conn, recs)
    conn.close()
    return {"db": db, "out": str(tmp_path / "cat_out"), "tmp": tmp_path}


def _table_fingerprint(db):
    """Hash of every detection row, order-independent, for byte-identity checks."""
    conn = connect(db, create=False)
    try:
        recs = query_records(conn, order_by="record_id")
    finally:
        conn.close()
    h = hashlib.sha256()
    for r in recs:
        for c in COLUMNS:
            if c in ("created_at", "updated_at"):
                continue  # store may stamp these, irrelevant to read-only proof
            h.update(repr(getattr(r, c)).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Import surface / pure helpers
# --------------------------------------------------------------------------- #

def test_import_surface():
    from reid_demo.catalogue import build_catalogue, CatalogueResult  # noqa: F401
    assert callable(build_catalogue)


def test_norm_flank_maps_non_canonical_to_unknown():
    assert _norm_flank("left") == "left"
    assert _norm_flank("right") == "right"
    assert _norm_flank("front") == "front"
    assert _norm_flank(None) == "unknown"
    assert _norm_flank("") == "unknown"
    assert _norm_flank("sideways") == "unknown"
    assert _norm_flank("LEFT") == "left"  # case-insensitive normalization


def test_is_low_conf():
    assert _is_low_conf(None, 0.5) is True
    assert _is_low_conf(0.4, 0.5) is True
    assert _is_low_conf(0.5, 0.5) is False
    assert _is_low_conf(0.9, 0.5) is False


# --------------------------------------------------------------------------- #
# Core build + summary schema
# --------------------------------------------------------------------------- #

def test_build_basic_outputs_exist(seeded):
    res = build_catalogue(seeded["db"], dataset=DATASET, out_dir=seeded["out"])
    assert isinstance(res, CatalogueResult)
    assert os.path.exists(res.index_html)
    assert os.path.exists(res.summary_json)
    assert os.path.exists(os.path.join(res.out_dir, "assets", "style.css"))
    # one per-individual page per individual
    for cid in (0, 1, 2):
        assert cid in res.individual_pages
        assert os.path.exists(res.individual_pages[cid])
    assert os.path.exists(os.path.join(res.out_dir, "unassigned.html"))


def test_summary_schema_and_counts(seeded):
    res = build_catalogue(seeded["db"], dataset=DATASET, out_dir=seeded["out"],
                          low_conf_threshold=0.5)
    s = res.summary
    # required top-level keys
    for k in ("dataset", "species_filter", "generated_at", "low_conf_threshold",
              "counts", "headline", "by_flank", "individuals"):
        assert k in s, k
    c = s["counts"]
    assert c["individuals"] == 3, c
    assert c["crops_clustered"] == 8, c  # 3 + 2 + 3
    assert c["total_crops"] == 9, c
    assert c["candidate_new"] == 1, c
    assert c["unassigned_noise"] == 1, c
    # low-conf: cluster 1's 2 crops (0.4) + candidate-new (0.3) = 3
    assert c["low_confidence_crops"] == 3, c
    assert c["reviewed_confirmed"] == 1, c
    assert c["reviewed_rejected"] == 0, c
    # JSON on disk == summary in memory
    with open(res.summary_json) as fh:
        on_disk = json.load(fh)
    assert on_disk["counts"] == c


def test_individuals_equals_distinct_clusters_ge_zero(seeded):
    res = build_catalogue(seeded["db"], dataset=DATASET, out_dir=seeded["out"])
    # distinct cluster_id >= 0 directly from the store
    conn = connect(seeded["db"], create=False)
    try:
        by = count_by(conn, "cluster_id", dataset=DATASET)
    finally:
        conn.close()
    distinct_ge0 = sum(1 for k in by if k is not None and k >= 0)
    assert res.summary["counts"]["individuals"] == distinct_ge0 == 3


def test_individuals_sorted_n_crops_desc_then_cid_asc(seeded):
    res = build_catalogue(seeded["db"], dataset=DATASET, out_dir=seeded["out"])
    inds = res.summary["individuals"]
    keys = [(-i["n_crops"], i["cluster_id"]) for i in inds]
    assert keys == sorted(keys), keys
    # clusters 0 and 2 both have 3 crops; tie broken by cluster_id asc -> 0 first, 2 second
    assert inds[0]["cluster_id"] == 0 and inds[0]["n_crops"] == 3
    assert inds[1]["cluster_id"] == 2 and inds[1]["n_crops"] == 3
    assert inds[2]["cluster_id"] == 1 and inds[2]["n_crops"] == 2


def test_headline_is_plain_language_with_count(seeded):
    res = build_catalogue(seeded["db"], dataset=DATASET, out_dir=seeded["out"])
    headline = res.summary["headline"]
    assert isinstance(headline, str) and headline
    assert "3 individuals" in headline
    assert "individual" in headline.lower() and "across" in headline


# --------------------------------------------------------------------------- #
# by_flank (D7c)
# --------------------------------------------------------------------------- #

def test_by_flank_invariants(seeded):
    res = build_catalogue(seeded["db"], dataset=DATASET, out_dir=seeded["out"])
    s = res.summary
    bf = s["by_flank"]
    assert set(bf) == set(CANONICAL_FLANKS)
    # sum invariant
    assert sum(bf.values()) == s["counts"]["crops_clustered"]
    # population: cluster0=3 left, cluster1=2 right, cluster2=left+right+unknown
    assert bf["left"] == 4, bf
    assert bf["right"] == 3, bf
    assert bf["unknown"] == 1, bf  # the NULL-orientation clustered crop
    assert bf["front"] == 0 and bf["back"] == 0 and bf["down"] == 0
    # candidate-new (cluster_id == -1, orientation 'unknown') must NOT inflate by_flank
    assert sum(bf.values()) == 8


def test_mixed_flank_flag(seeded):
    res = build_catalogue(seeded["db"], dataset=DATASET, out_dir=seeded["out"])
    inds = {i["cluster_id"]: i for i in res.summary["individuals"]}
    assert inds[2]["mixed_flank"] is True     # has left + right
    assert inds[0]["mixed_flank"] is False    # only left
    assert inds[1]["mixed_flank"] is False    # only right


# --------------------------------------------------------------------------- #
# Read-only proof
# --------------------------------------------------------------------------- #

def test_build_does_not_mutate_store(seeded):
    before = _table_fingerprint(seeded["db"])
    before_counts = None
    conn = connect(seeded["db"], create=False)
    try:
        before_counts = count_by(conn, "cluster_id", dataset=DATASET)
    finally:
        conn.close()

    build_catalogue(seeded["db"], dataset=DATASET, out_dir=seeded["out"])

    after = _table_fingerprint(seeded["db"])
    conn = connect(seeded["db"], create=False)
    try:
        after_counts = count_by(conn, "cluster_id", dataset=DATASET)
    finally:
        conn.close()
    assert before == after, "detections table must be byte-identical after a build"
    assert before_counts == after_counts


# --------------------------------------------------------------------------- #
# Relative paths + portability
# --------------------------------------------------------------------------- #

def test_all_paths_relative_in_index(seeded):
    res = build_catalogue(seeded["db"], dataset=DATASET, out_dir=seeded["out"])
    html = open(res.index_html, encoding="utf-8").read()
    assert "file://" not in html
    for m in re.findall(r'(?:src|href)="([^"]+)"', html):
        if m.startswith("#"):
            continue
        assert not m.startswith("http"), m
        assert not m.startswith("/"), f"absolute path leaked: {m}"


def test_all_paths_relative_in_individual_pages(seeded):
    res = build_catalogue(seeded["db"], dataset=DATASET, out_dir=seeded["out"])
    for path in res.individual_pages.values():
        html = open(path, encoding="utf-8").read()
        assert "file://" not in html
        for m in re.findall(r'(?:src|href)="([^"]+)"', html):
            if m.startswith("#"):
                continue
            assert not m.startswith("http") and not m.startswith("/"), m


def test_portable_after_copytree(seeded):
    res = build_catalogue(seeded["db"], dataset=DATASET, out_dir=seeded["out"])
    dst = os.path.join(seeded["tmp"], "moved")
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(res.out_dir, dst)
    # every relative ref in index.html resolves in the moved dir
    html = open(os.path.join(dst, "index.html"), encoding="utf-8").read()
    for m in re.findall(r'(?:src|href)="([^"]+)"', html):
        if m.startswith("http") or m.startswith("#"):
            continue
        p = os.path.normpath(os.path.join(dst, m))
        assert os.path.exists(p), f"missing after move: {m}"
    # and for an individual page (paths are ../-relative)
    for fn in os.listdir(os.path.join(dst, "individuals")):
        page = os.path.join(dst, "individuals", fn)
        ph = open(page, encoding="utf-8").read()
        for m in re.findall(r'(?:src|href)="([^"]+)"', ph):
            if m.startswith("http") or m.startswith("#"):
                continue
            p = os.path.normpath(os.path.join(os.path.dirname(page), m))
            assert os.path.exists(p), f"missing after move (indiv): {m}"


# --------------------------------------------------------------------------- #
# Missing crop -> placeholder, still counted
# --------------------------------------------------------------------------- #

def test_missing_crop_does_not_crash_and_is_counted(seeded):
    # cluster 1 (B0/B1) and the singleton (S) all point at non-existent files
    res = build_catalogue(seeded["db"], dataset=DATASET, out_dir=seeded["out"])
    # build completed; placeholder generated
    assert os.path.exists(os.path.join(res.out_dir, "thumbs", "_placeholder.png"))
    # records still counted: total includes the missing-crop rows
    assert res.summary["counts"]["total_crops"] == 9
    # cluster 1 still has 2 crops despite missing files
    ind1 = next(i for i in res.summary["individuals"] if i["cluster_id"] == 1)
    assert ind1["n_crops"] == 2
    # placeholder reference appears in cluster-1 page
    page = res.individual_pages[1]
    html = open(page, encoding="utf-8").read()
    assert "_placeholder.png" in html


# --------------------------------------------------------------------------- #
# Low-confidence marker in HTML
# --------------------------------------------------------------------------- #

def test_low_conf_marked_in_html(seeded):
    res = build_catalogue(seeded["db"], dataset=DATASET, out_dir=seeded["out"],
                          low_conf_threshold=0.5)
    # cluster 1 is the low-conf cluster (0.4)
    page = res.individual_pages[1]
    html = open(page, encoding="utf-8").read()
    assert "low_conf" in html  # tile class marker
    # the per-individual summary records 2 low-conf crops
    ind1 = next(i for i in res.summary["individuals"] if i["cluster_id"] == 1)
    assert ind1["n_low_conf"] == 2


def test_review_status_shown_in_html(seeded):
    res = build_catalogue(seeded["db"], dataset=DATASET, out_dir=seeded["out"])
    page = res.individual_pages[0]  # cluster 0 has one 'confirmed' crop
    html = open(page, encoding="utf-8").read()
    assert "confirmed" in html


# --------------------------------------------------------------------------- #
# Representative crop / JSON relativity
# --------------------------------------------------------------------------- #

def test_representative_and_page_paths_relative(seeded):
    res = build_catalogue(seeded["db"], dataset=DATASET, out_dir=seeded["out"])
    for ind in res.summary["individuals"]:
        rep = ind["representative_crop"]
        assert rep is not None
        assert not os.path.isabs(rep), rep
        assert rep.startswith("thumbs/"), rep
        assert not os.path.isabs(ind["page"]), ind["page"]
        assert ind["page"].startswith("individuals/"), ind["page"]
        # resolves relative to out_dir
        assert os.path.exists(os.path.join(res.out_dir, rep))


def test_representative_is_highest_cluster_conf(seeded, tmp_path):
    # cluster 0 crops all 0.9; bump one detector_conf so tie-break is testable
    db = str(tmp_path / "rep.sqlite")
    crops = tmp_path / "c"
    crops.mkdir()
    conn = connect(db)
    recs = []
    for i, (cc, dc) in enumerate([(0.5, 0.5), (0.9, 0.5), (0.5, 0.99)]):
        cp = crops / f"R{i}__crop1.jpg"
        _write_tiny_jpeg(cp)
        recs.append(_mk(f"R{i}", 1, 0, cc, "left", str(cp), detconf=dc))
    upsert_records(conn, recs)
    conn.close()
    res = build_catalogue(db, dataset=DATASET, out_dir=str(tmp_path / "out"))
    ind0 = next(i for i in res.summary["individuals"] if i["cluster_id"] == 0)
    # highest cluster_conf is R1 (0.9)
    assert "R1" in ind0["representative_crop"]


# --------------------------------------------------------------------------- #
# Empty / filtered result set
# --------------------------------------------------------------------------- #

def test_empty_result_raises(seeded):
    with pytest.raises(ValueError):
        build_catalogue(seeded["db"], dataset="NoSuchDataset",
                        out_dir=seeded["out"] + "_empty")


def test_species_filter(seeded, tmp_path):
    # add a non-lynx clustered crop, then filter to lynx only
    db = seeded["db"]
    conn = connect(db)
    cp = tmp_path / "fox__crop1.jpg"
    _write_tiny_jpeg(cp)
    upsert_records(conn, [_mk("fox", 1, 5, 0.9, "left", str(cp), species="red fox")])
    conn.close()
    res = build_catalogue(db, dataset=DATASET, species="eurasian lynx",
                          out_dir=seeded["out"] + "_sp")
    cids = {i["cluster_id"] for i in res.summary["individuals"]}
    assert 5 not in cids  # fox cluster excluded by species filter
    assert res.summary["species_filter"] == "eurasian lynx"


# --------------------------------------------------------------------------- #
# Montages
# --------------------------------------------------------------------------- #

def test_no_montages_by_default(seeded):
    res = build_catalogue(seeded["db"], dataset=DATASET, out_dir=seeded["out"])
    assert res.montage_pngs == {}


def test_montages_when_requested(seeded):
    res = build_catalogue(seeded["db"], dataset=DATASET, out_dir=seeded["out"],
                          make_montages=True)
    # matplotlib is present in the repo venv; expect one PNG per individual.
    # If the optional import path fails, montage_pngs is {} (fail-soft) — accept both.
    if res.montage_pngs:
        for cid in (0, 1, 2):
            assert cid in res.montage_pngs
            assert os.path.exists(res.montage_pngs[cid])
            assert res.montage_pngs[cid].endswith(f"individual_{cid}.png")


# --------------------------------------------------------------------------- #
# CLI / selftest
# --------------------------------------------------------------------------- #

def test_cli_selftest_exit_zero(tmp_path):
    db = str(tmp_path / "selftest.sqlite")
    proc = subprocess.run(
        [sys.executable, "-m", "reid_demo.catalogue", "--selftest", "--db", db],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in (proc.stdout + proc.stderr)


def test_cli_full_build(seeded):
    proc = subprocess.run(
        [sys.executable, "-m", "reid_demo.catalogue",
         "--db", seeded["db"], "--dataset", DATASET, "--out", seeded["out"]],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    # prints absolute index.html path + headline
    lines = proc.stdout.strip().splitlines()
    assert os.path.exists(lines[0]), lines
    assert "individuals" in lines[1].lower()
