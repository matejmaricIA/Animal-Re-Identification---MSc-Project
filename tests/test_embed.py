"""Unit tests for reid_demo.embed (T04).

Logic-level tests run WITHOUT model weights / torch by monkeypatching
``global_embedding.extract_global_embeddings`` with a deterministic fake that returns a
1536-dim RAW (un-normalized) float32 vector per image (mirroring the base-model output
shape and the D2 "stored raw" contract). A separate test that exercises the REAL
MegaDescriptor model is skipped unless torch + a real crop are available.

No network, no model download required for the logic-level suite.
"""

import os
import sys

import numpy as np
import pytest

# Make the repo root importable regardless of how pytest is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reid_demo.embed import (  # noqa: E402
    embed_records,
    embed_crops,
    load_embeddings,
    get_embedding_matrix,
    EmbedResult,
    embedding_cache_path,
    DEFAULT_EMB_DIR,
    DEFAULT_MODEL_NAME,
)
from reid_demo.store import (  # noqa: E402
    connect,
    upsert_record,
    query_records,
    get_record,
    DetectionRecord,
    make_record_id,
)

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

FAKE_DIM = 1536


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _make_image(path, size=(64, 64), color=(120, 90, 60)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", size, color).save(path, "JPEG", quality=90)


def _fake_vector(key: str, dim: int = FAKE_DIM) -> np.ndarray:
    """Deterministic per-key RAW (un-normalized) vector. Norm != 1 so we can verify
    read-time normalization actually does something."""
    rng = np.random.RandomState(abs(hash(key)) % (2**32))
    return (rng.randn(dim).astype(np.float32) * 7.0)  # scale -> norm well above 1


@pytest.fixture
def fake_extract(monkeypatch):
    """Patch the model call with a deterministic fake. Patches the symbol both where it
    is defined (``global_embedding``) and where embed.py imports it from inside the
    function, so the lazy ``from global_embedding import extract_global_embeddings`` picks
    up the fake."""
    def _fake(image_paths, model_name=DEFAULT_MODEL_NAME, device=None, checkpoint_path=None):
        out = {}
        for key, path in image_paths.items():
            # Honour the real contract: would open the file; here just trust pre-filter.
            out[str(key)] = _fake_vector(str(key))
        return out

    import global_embedding
    monkeypatch.setattr(global_embedding, "extract_global_embeddings", _fake)
    # cache label is used by embed.py; keep the real one (no torch needed for it beyond
    # global_embedding import, which we just performed successfully).
    return _fake


@pytest.fixture
def seeded_conn(tmp_path):
    """A store seeded with 3 records pointing at real (tiny) crop files, species set."""
    crops_dir = tmp_path / "crops"
    conn = connect(str(tmp_path / "reid.sqlite"))
    for i in range(1, 4):
        stem = f"probe{i}"
        crop = crops_dir / f"{stem}.jpg"
        _make_image(str(crop))
        rid = make_record_id(stem, 1)
        upsert_record(conn, DetectionRecord(
            record_id=rid, source_image=f"src/{stem}.JPG", source_stem=stem,
            det_index=1, crop_path=str(crop),
            bbox_x=0.1, bbox_y=0.1, bbox_w=0.2, bbox_h=0.2,
            dataset="MedvednicaDS", species="eurasian lynx",
        ))
    return conn


# --------------------------------------------------------------------------- #
# Import surface / constants
# --------------------------------------------------------------------------- #

def test_constants():
    assert DEFAULT_EMB_DIR == "data/reid_demo/embeddings"
    assert DEFAULT_MODEL_NAME == "megadescriptor-l-384"


def test_cache_path_deterministic():
    p1 = embedding_cache_path("MedvednicaDS", cache_dir="/tmp/x")
    p2 = embedding_cache_path("MedvednicaDS", cache_dir="/tmp/x")
    assert p1 == p2
    assert p1.endswith(".pkl")
    assert "MedvednicaDS" in p1
    # dataset=None -> "all"
    assert "all_" in os.path.basename(embedding_cache_path(None, cache_dir="/tmp/x"))


# --------------------------------------------------------------------------- #
# embed_crops
# --------------------------------------------------------------------------- #

def test_embed_crops_and_cache_reuse(fake_extract, tmp_path):
    crop = tmp_path / "r1.jpg"
    _make_image(str(crop))
    cache_path = str(tmp_path / "e.pkl")

    out = embed_crops({"r1": str(crop)}, cache_path)
    assert set(out.keys()) == {"r1"}
    v = out["r1"]
    assert isinstance(v, np.ndarray) and v.ndim == 1 and v.dtype == np.float32
    assert v.shape[0] == FAKE_DIM
    assert os.path.exists(cache_path)

    # Second call reuses cache, recomputes nothing (patch extract to blow up if called).
    import global_embedding
    def _boom(*a, **k):
        raise AssertionError("extract_global_embeddings should not be called on reuse")
    global_embedding.extract_global_embeddings = _boom  # type: ignore
    out2 = embed_crops({"r1": str(crop)}, cache_path)
    assert np.array_equal(out2["r1"], v)


def test_embed_crops_missing_file_omitted(fake_extract, tmp_path):
    good = tmp_path / "good.jpg"
    _make_image(str(good))
    cache_path = str(tmp_path / "e.pkl")
    out = embed_crops({"good": str(good), "ghost": str(tmp_path / "nope.jpg")}, cache_path)
    assert "good" in out and "ghost" not in out


# --------------------------------------------------------------------------- #
# embed_records end-to-end (store side effects, idempotency, matrix)
# --------------------------------------------------------------------------- #

def test_embed_records_end_to_end(fake_extract, seeded_conn, tmp_path):
    conn = seeded_conn
    cache_dir = str(tmp_path / "emb")
    res = embed_records(conn, dataset="MedvednicaDS", cache_dir=cache_dir)
    assert isinstance(res, EmbedResult)
    assert res.n_total == 3
    assert res.n_embedded == 3
    assert res.n_failed == 0
    assert res.embedding_dim == FAKE_DIM
    assert os.path.exists(res.cache_path)

    # store side effects
    rows = query_records(conn, dataset="MedvednicaDS", has_embedding=True)
    assert len(rows) == 3
    assert all(r.embedding_ref == r.record_id for r in rows)
    assert all(r.embedding_path == res.cache_path for r in rows)

    # pickle keyed by record_id
    cache = load_embeddings(res.cache_path)
    assert {r.record_id for r in rows} <= set(cache.keys())
    for r in rows:
        assert np.array_equal(cache[r.record_id], _fake_vector(r.record_id))

    # read-side matrix, normalized
    M, ids = get_embedding_matrix(conn, dataset="MedvednicaDS", normalize=True)
    assert M.shape[0] == len(ids) == 3
    assert M.shape[1] == FAKE_DIM
    norms = np.linalg.norm(M, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), norms
    # ids are sorted
    assert ids == sorted(ids)

    # un-normalized matrix has the RAW (>1) norms
    Mraw, _ = get_embedding_matrix(conn, dataset="MedvednicaDS", normalize=False)
    assert not np.allclose(np.linalg.norm(Mraw, axis=1), 1.0, atol=1e-3)


def test_embed_records_idempotent(fake_extract, seeded_conn, tmp_path):
    conn = seeded_conn
    cache_dir = str(tmp_path / "emb")
    embed_records(conn, dataset="MedvednicaDS", cache_dir=cache_dir)
    res2 = embed_records(conn, dataset="MedvednicaDS", cache_dir=cache_dir)
    assert res2.n_embedded == 0
    assert res2.n_skipped == 3


def test_embed_records_species_filter(fake_extract, seeded_conn, tmp_path):
    conn = seeded_conn
    # add a non-matching species record
    crop = tmp_path / "boar.jpg"
    _make_image(str(crop))
    rid = make_record_id("boar1", 1)
    upsert_record(conn, DetectionRecord(
        record_id=rid, source_image="src/boar1.JPG", source_stem="boar1",
        det_index=1, crop_path=str(crop), bbox_x=0.0, bbox_y=0.0, bbox_w=0.1, bbox_h=0.1,
        dataset="MedvednicaDS", species="wild boar",
    ))
    res = embed_records(conn, dataset="MedvednicaDS", cache_dir=str(tmp_path / "emb"),
                        only_species="Eurasian Lynx")  # case-insensitive
    assert res.n_embedded == 3
    # boar stays unembedded
    boar = get_record(conn, rid)
    assert boar.embedding_ref is None


def test_embed_records_missing_crop(fake_extract, tmp_path):
    conn = connect(str(tmp_path / "fail.sqlite"))
    rid = make_record_id("ghost", 1)
    upsert_record(conn, DetectionRecord(
        record_id=rid, source_image="x", source_stem="ghost", det_index=1,
        crop_path=str(tmp_path / "NOPE.jpg"),
        bbox_x=0.0, bbox_y=0.0, bbox_w=0.1, bbox_h=0.1, dataset="MedvednicaDS",
    ))
    res = embed_records(conn, dataset="MedvednicaDS", cache_dir=str(tmp_path / "emb"))
    assert res.n_failed == 1
    assert rid in res.failed_ids
    assert res.n_embedded == 0


def test_embed_records_missing_crop_does_not_block_others(fake_extract, seeded_conn, tmp_path):
    conn = seeded_conn
    rid = make_record_id("ghost", 1)
    upsert_record(conn, DetectionRecord(
        record_id=rid, source_image="x", source_stem="ghost", det_index=1,
        crop_path=str(tmp_path / "NOPE.jpg"),
        bbox_x=0.0, bbox_y=0.0, bbox_w=0.1, bbox_h=0.1, dataset="MedvednicaDS",
        species="eurasian lynx",
    ))
    res = embed_records(conn, dataset="MedvednicaDS", cache_dir=str(tmp_path / "emb"))
    assert res.n_embedded == 3
    assert res.n_failed == 1 and rid in res.failed_ids


def test_limit(fake_extract, seeded_conn, tmp_path):
    conn = seeded_conn
    res = embed_records(conn, dataset="MedvednicaDS", cache_dir=str(tmp_path / "emb"), limit=2)
    assert res.n_total == 2
    assert res.n_embedded == 2


# --------------------------------------------------------------------------- #
# get_embedding_matrix error paths
# --------------------------------------------------------------------------- #

def test_load_embeddings_missing():
    with pytest.raises(FileNotFoundError):
        load_embeddings("/tmp/definitely_not_here_reid_embed.pkl")


def test_get_matrix_stale_cache_raises(fake_extract, seeded_conn, tmp_path):
    conn = seeded_conn
    res = embed_records(conn, dataset="MedvednicaDS", cache_dir=str(tmp_path / "emb"))
    # Corrupt the pickle: drop one key that the store still references.
    import pickle
    with open(res.cache_path, "rb") as fh:
        cache = pickle.load(fh)
    victim = sorted(cache.keys())[0]
    del cache[victim]
    with open(res.cache_path, "wb") as fh:
        pickle.dump(cache, fh)

    with pytest.raises(RuntimeError) as exc:
        get_embedding_matrix(conn, dataset="MedvednicaDS")
    assert victim in str(exc.value)
    assert res.cache_path in str(exc.value)


def test_get_matrix_empty(fake_extract, tmp_path):
    conn = connect(str(tmp_path / "empty.sqlite"))
    M, ids = get_embedding_matrix(conn, dataset="MedvednicaDS")
    assert M.shape[0] == 0 and ids == []


# --------------------------------------------------------------------------- #
# Real-model smoke test (skipped without torch / real crop / weights)
# --------------------------------------------------------------------------- #

def test_real_megadescriptor_dim():
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    import glob
    crops = sorted(glob.glob("data/MedvednicaDS/animal_crops/*.jpg"))
    if not crops:
        pytest.skip("no real Medvednica crops available")
    try:
        from global_embedding import extract_global_embeddings
        e = extract_global_embeddings({"probe": crops[0]}, model_name="megadescriptor-l-384")
    except Exception as exc:  # model weights / network unavailable
        pytest.skip(f"MegaDescriptor unavailable: {exc!r}")
    v = np.asarray(e["probe"])
    assert v.ndim == 1
    assert v.shape[0] == 1536
