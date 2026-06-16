"""Unit tests for reid_demo.fisher (T11).

Two tiers:

* Logic-level tests that need NO model weights / torch / lightglue — they import the
  module, check the cache-path / label contract, the import surface, the boundary
  (no banned imports), and the read-side ``get_fisher_matrix`` / ``load_fisher_vectors``
  using a SYNTHETIC Fisher pickle seeded directly into a temp store. These always run.

* Heavy end-to-end tests that DO extract DISK descriptors from real Medvednica crops and
  fit PCA/GMM. These are gated behind ``importorskip`` (torch + lightglue + the repo's
  feature pipeline) AND the presence of real crop files, so they skip gracefully on a
  machine without the venv / data.

Self-contained, deterministic, no network.
"""

import ast
import glob
import json
import os
import pickle
import sys

import pytest

# Make the repo root importable regardless of how pytest is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

np = pytest.importorskip("numpy")

from reid_demo.fisher import (  # noqa: E402
    build_fisher_records,
    build_fisher_vectors,
    load_fisher_vectors,
    get_fisher_matrix,
    FisherResult,
    fisher_cache_path,
    fisher_cache_label,
    DEFAULT_FISHER_DIR,
    DEFAULT_METHOD,
    DEFAULT_PCA_DIM,
)
from reid_demo.store import (  # noqa: E402
    connect,
    upsert_record,
    get_record,
    update_extra,
    make_record_id,
    DetectionRecord,
    COLUMNS,
    TABLE_NAME,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FISHER_PY = os.path.join(REPO_ROOT, "reid_demo", "fisher.py")
CROP_GLOB = os.path.join(REPO_ROOT, "data", "MedvednicaDS", "animal_crops", "*.jpg")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _seed_record(conn, rid, crop_path, *, dataset="MedvednicaDS", species="eurasian lynx"):
    upsert_record(conn, DetectionRecord(
        record_id=rid, source_image="x", source_stem=rid.split("__")[0],
        det_index=1, crop_path=crop_path,
        bbox_x=0.0, bbox_y=0.0, bbox_w=1.0, bbox_h=1.0,
        dataset=dataset, species=species,
    ))


def _real_crops(n):
    return sorted(glob.glob(CROP_GLOB))[:n]


def _cleanup_fitted_cache(ds_tag):
    """Remove the global fitted PCA/GMM/FV pickle dir load_or_train_fisher_vectors writes
    under data/{ds_tag}/ so heavy tests start from a clean fit (they share the repo root)."""
    import shutil
    d = os.path.join(REPO_ROOT, "data", ds_tag)
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)


# Heavy-pipeline availability: torch + lightglue + real crops.
def _pipeline_available():
    try:
        import torch  # noqa: F401
        import lightglue  # noqa: F401
    except Exception:
        return False
    return len(_real_crops(1)) > 0


PIPELINE = _pipeline_available()
heavy = pytest.mark.skipif(not PIPELINE,
                           reason="needs torch+lightglue and real Medvednica crops")


# --------------------------------------------------------------------------- #
# Logic-level tests (no heavy deps)
# --------------------------------------------------------------------------- #

def test_import_surface():
    # All contracted names import (also verified at module import above).
    assert callable(build_fisher_records)
    assert callable(build_fisher_vectors)
    assert callable(load_fisher_vectors)
    assert callable(get_fisher_matrix)
    assert DEFAULT_METHOD == "disk"
    assert isinstance(DEFAULT_PCA_DIM, int) and DEFAULT_PCA_DIM > 0
    assert DEFAULT_FISHER_DIR == "data/reid_demo/fisher"


def test_cache_label_and_path():
    assert fisher_cache_label("disk", 128) == "disk_pca128"
    assert fisher_cache_label("ALIKED", 64) == "aliked_pca64"
    p = fisher_cache_path("MedvednicaDS", "disk", 128)
    assert p.endswith("MedvednicaDS_disk_pca128.pkl")
    assert p.startswith(DEFAULT_FISHER_DIR)
    # dataset=None -> "all"
    assert fisher_cache_path(None, "disk", 128).endswith("all_disk_pca128.pkl")


def test_no_hardcoded_dim_literal():
    src = open(FISHER_PY).read()
    assert "65536" not in src, "Fisher dim must be derived, never the literal 65536"
    # no bare 2*256*128 style literal
    assert "256*128" not in src.replace(" ", "")
    assert "2*256" not in src.replace(" ", "")


def test_boundary_no_banned_imports():
    src = open(FISHER_PY).read()
    banned = {"reid_demo.cluster", "reid_demo.fusion", "reid_demo.embed",
              "calibration", "geometric_verification"}
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.ImportFrom):
            assert (n.module or "") not in banned, f"banned import: {n.module}"
        if isinstance(n, ast.Import):
            for a in n.names:
                assert a.name not in banned, f"banned import: {a.name}"


def test_load_fisher_vectors_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_fisher_vectors(str(tmp_path / "nope.pkl"))


def test_get_fisher_matrix_synthetic(tmp_path):
    """Read-side test using a synthetic FV pickle (no model weights needed)."""
    db = str(tmp_path / "syn.sqlite")
    conn = connect(db)
    cache = str(tmp_path / "MedvednicaDS_disk_pca128.pkl")

    # 3 records: two real-ish unit vectors + one all-zero (no-descriptor) vector.
    D = 8
    v1 = np.zeros(D, dtype=np.float32); v1[0] = 3.0   # not unit -> normalize should fix
    v2 = np.zeros(D, dtype=np.float32); v2[1] = 1.0   # already unit
    v0 = np.zeros(D, dtype=np.float32)                # zero vector
    rid1, rid2, rid0 = (make_record_id("a", 1), make_record_id("b", 1), make_record_id("c", 1))
    with open(cache, "wb") as fh:
        pickle.dump({rid1: v1, rid2: v2, rid0: v0}, fh)

    for rid in (rid1, rid2, rid0):
        _seed_record(conn, rid, "x")
        update_extra(conn, rid, "fisher_ref", rid)
        update_extra(conn, rid, "fisher_path", cache)

    M, ids = get_fisher_matrix(conn, dataset="MedvednicaDS", normalize=True)
    assert ids == sorted([rid1, rid2, rid0])
    assert M.shape == (3, D)
    norms = np.linalg.norm(M, axis=1)
    assert np.all((np.abs(norms - 1.0) < 1e-5) | (norms < 1e-9)), norms
    assert not np.isnan(M).any()


def test_get_fisher_matrix_stale_cache_raises(tmp_path):
    db = str(tmp_path / "stale.sqlite")
    conn = connect(db)
    cache = str(tmp_path / "MedvednicaDS_disk_pca128.pkl")
    rid = make_record_id("ghost", 1)
    with open(cache, "wb") as fh:
        pickle.dump({}, fh)   # ref'd record NOT in the pickle
    _seed_record(conn, rid, "x")
    update_extra(conn, rid, "fisher_ref", rid)
    update_extra(conn, rid, "fisher_path", cache)
    with pytest.raises(RuntimeError) as ei:
        get_fisher_matrix(conn, dataset="MedvednicaDS")
    assert rid in str(ei.value) and cache in str(ei.value)


def test_get_fisher_matrix_empty(tmp_path):
    db = str(tmp_path / "empty.sqlite")
    conn = connect(db)
    _seed_record(conn, make_record_id("x", 1), "x")   # no fisher_ref
    M, ids = get_fisher_matrix(conn, dataset="MedvednicaDS")
    assert ids == [] and M.shape[0] == 0


# --------------------------------------------------------------------------- #
# Heavy end-to-end tests (DISK + PCA/GMM on real crops)
# --------------------------------------------------------------------------- #

@heavy
def test_build_fisher_vectors_low_level(tmp_path):
    crops = _real_crops(6)
    rids = [make_record_id("probe%d" % i, 1) for i in range(len(crops))]
    crop_paths = dict(zip(rids, crops))
    cache = str(tmp_path / "fv.pkl")
    desc = str(tmp_path / "desc")

    fv = build_fisher_vectors(crop_paths, cache, dataset="t11test",
                              desc_dir=desc, method="disk", pca_dim=128)
    assert os.path.exists(cache)
    assert set(fv.keys()) == set(rids)
    dims = {v.shape[0] for v in fv.values()}
    assert len(dims) == 1
    dim = dims.pop()
    for v in fv.values():
        assert v.ndim == 1 and v.dtype == np.float32
    # dim is derived (== 2*gmm*pca). With defaults that is 65536, but assert structurally.
    assert dim % (2 * 128) == 0


@heavy
def test_build_fisher_records_end_to_end(tmp_path):
    # Unique dataset tag so the fitted-model cache (data/{dataset}/...) does not collide
    # with other tests' record sets.
    ds = "t11e2e"
    _cleanup_fitted_cache(ds)
    crops = _real_crops(5)
    db = str(tmp_path / "e2e.sqlite")
    conn = connect(db)
    rids = []
    for i, c in enumerate(crops, start=1):
        rid = make_record_id("e2e%d" % i, 1)
        rids.append(rid)
        _seed_record(conn, rid, c, dataset=ds)

    res = build_fisher_records(conn, dataset=ds, cache_dir=str(tmp_path / "fisher"))
    assert isinstance(res, FisherResult)
    assert res.n_fishered == len(crops)
    assert res.n_failed == 0
    assert res.fisher_dim == 2 * res.gmm_components * res.pca_dim

    # store side effects live in extra_json (no new column)
    for rid in rids:
        ex = json.loads(get_record(conn, rid).extra_json)
        assert ex.get("fisher_ref") == rid
        assert ex.get("fisher_path") == res.cache_path

    # no schema drift
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({TABLE_NAME})").fetchall()]
    assert cols == COLUMNS

    cache = load_fisher_vectors(res.cache_path)
    assert set(cache.keys()) >= set(rids)

    M, ids = get_fisher_matrix(conn, dataset=ds, normalize=True)
    assert M.shape[0] == len(ids) == len(crops)
    norms = np.linalg.norm(M, axis=1)
    assert np.all((np.abs(norms - 1.0) < 1e-5) | (norms < 1e-9))
    assert not np.isnan(M).any()

    # idempotent re-run: recompute nothing
    res2 = build_fisher_records(conn, dataset=ds, cache_dir=str(tmp_path / "fisher"))
    assert res2.n_fishered == 0
    assert res2.n_skipped == len(crops)
    _cleanup_fitted_cache(ds)


@heavy
def test_build_fisher_records_missing_crop(tmp_path):
    ds = "t11miss"
    _cleanup_fitted_cache(ds)
    crops = _real_crops(2)
    db = str(tmp_path / "miss.sqlite")
    conn = connect(db)
    good = [make_record_id("good%d" % i, 1) for i in range(len(crops))]
    for rid, c in zip(good, crops):
        _seed_record(conn, rid, c, dataset=ds)
    ghost = make_record_id("ghost", 1)
    _seed_record(conn, ghost, str(tmp_path / "NOPE.jpg"), dataset=ds)

    res = build_fisher_records(conn, dataset=ds, cache_dir=str(tmp_path / "fisher"))
    assert res.n_failed == 1
    assert ghost in res.failed_ids
    assert res.n_fishered == len(crops)
    _cleanup_fitted_cache(ds)


@heavy
def test_build_fisher_records_species_filter(tmp_path):
    ds = "t11sp"
    _cleanup_fitted_cache(ds)
    crops = _real_crops(2)
    db = str(tmp_path / "sp.sqlite")
    conn = connect(db)
    lynx = make_record_id("lynx", 1)
    other = make_record_id("other", 1)
    _seed_record(conn, lynx, crops[0], dataset=ds, species="Eurasian Lynx")   # mixed case
    _seed_record(conn, other, crops[1], dataset=ds, species="red fox")

    res = build_fisher_records(conn, dataset=ds, only_species="eurasian lynx",
                               cache_dir=str(tmp_path / "fisher"))
    assert res.n_fishered == 1
    assert json.loads(get_record(conn, lynx).extra_json).get("fisher_ref") == lynx
    assert json.loads(get_record(conn, other).extra_json).get("fisher_ref") is None
    _cleanup_fitted_cache(ds)
