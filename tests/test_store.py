"""Unit tests for reid_demo.store (T01) — one assertion group per acceptance criterion."""

import json
import os
import sqlite3
import sys
import time

import pytest

# Make the repo root importable regardless of how pytest is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reid_demo.store import (  # noqa: E402
    DetectionRecord,
    SCHEMA_VERSION,
    DEFAULT_DB_PATH,
    TABLE_NAME,
    COLUMNS,
    REVIEW_STATUSES,
    ORIENTATIONS,
    connect,
    init_db,
    upsert_record,
    upsert_records,
    get_record,
    query_records,
    update_species,
    update_embedding,
    update_cluster,
    update_review,
    update_extra,
    count_by,
    make_record_id,
    export_records,
    import_records,
    to_dataframe,
)

EXPECTED_COLUMNS = [
    "record_id", "source_image", "source_stem", "det_index", "crop_path",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h", "detector_conf", "camera_id",
    "timestamp", "species", "species_conf", "species_class", "embedding_ref",
    "embedding_path", "cluster_id", "cluster_conf", "is_candidate_new",
    "orientation", "gt_identity", "review_status", "review_note", "dataset",
    "extra_json", "created_at", "updated_at",
]

EXPECTED_TYPES = {
    "record_id": "TEXT", "source_image": "TEXT", "source_stem": "TEXT",
    "det_index": "INTEGER", "crop_path": "TEXT", "bbox_x": "REAL", "bbox_y": "REAL",
    "bbox_w": "REAL", "bbox_h": "REAL", "detector_conf": "REAL", "camera_id": "TEXT",
    "timestamp": "TEXT", "species": "TEXT", "species_conf": "REAL",
    "species_class": "TEXT", "embedding_ref": "TEXT", "embedding_path": "TEXT",
    "cluster_id": "INTEGER", "cluster_conf": "REAL", "is_candidate_new": "INTEGER",
    "orientation": "TEXT", "gt_identity": "TEXT", "review_status": "TEXT",
    "review_note": "TEXT", "dataset": "TEXT", "extra_json": "TEXT",
    "created_at": "TEXT", "updated_at": "TEXT",
}


def _rec(stem="IMG_0066", idx=1, **kw):
    defaults = dict(
        record_id=make_record_id(stem, idx),
        source_image=f"data/MedvednicaDS/animal_images/{stem}.JPG",
        source_stem=stem, det_index=idx,
        crop_path=f"data/reid_demo/crops/{stem}__crop{idx}.jpg",
        bbox_x=0.49, bbox_y=0.04, bbox_w=0.05, bbox_h=0.17,
        detector_conf=0.78, camera_id="unknown_camera",
        timestamp="2025-06-02 04:27:51", dataset="MedvednicaDS",
    )
    defaults.update(kw)
    return DetectionRecord(**defaults)


@pytest.fixture()
def conn(tmp_path):
    return connect(str(tmp_path / "reid.sqlite"))


# --- contract surface ------------------------------------------------------- #

def test_columns_exact_order_and_count():
    assert len(COLUMNS) == 28
    assert COLUMNS == EXPECTED_COLUMNS


def test_constants_present():
    assert SCHEMA_VERSION == 1
    assert DEFAULT_DB_PATH == "data/reid_demo/reid_demo.sqlite"
    assert TABLE_NAME == "detections"
    assert REVIEW_STATUSES == {"unreviewed", "confirmed", "rejected", "merged", "split"}
    assert ORIENTATIONS == {"left", "right", "front", "back", "down", "unknown"}


def test_table_shape_matches_contract(conn):
    info = conn.execute(f"PRAGMA table_info({TABLE_NAME})").fetchall()
    cols = [r[1] for r in info]
    types = {r[1]: r[2] for r in info}
    assert cols == COLUMNS
    for col, sqltype in EXPECTED_TYPES.items():
        assert types[col] == sqltype, (col, types[col], sqltype)
    # PK is record_id
    pk = [r[1] for r in info if r[5]]  # r[5] == pk flag
    assert pk == ["record_id"]


def test_required_indexes_exist(conn):
    idx_rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?",
        (TABLE_NAME,),
    ).fetchall()
    names = " ".join(r[0] for r in idx_rows)
    for needle in ("dataset", "cluster_id", "species", "review_status", "dataset_cluster"):
        assert needle in names, (needle, names)


def test_make_record_id():
    assert make_record_id("IMG_0066", 1) == "IMG_0066__crop1"


# --- round-trip & idempotency ----------------------------------------------- #

def test_round_trip_equal_modulo_timestamps(conn):
    r = _rec()
    upsert_record(conn, r)
    got = get_record(conn, r.record_id)
    assert got is not None
    for col in COLUMNS:
        if col in ("created_at", "updated_at"):
            assert getattr(got, col)  # populated
            continue
        assert getattr(got, col) == getattr(r, col), col


def test_upsert_twice_in_place_preserves_created_advances_updated(conn):
    r = _rec()
    upsert_record(conn, r)
    first = get_record(conn, r.record_id)
    time.sleep(1.05)  # store timestamps are second-resolution
    r2 = _rec(detector_conf=0.99)
    upsert_record(conn, r2)
    n = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    assert n == 1
    second = get_record(conn, r.record_id)
    assert second.created_at == first.created_at        # preserved
    assert second.updated_at > first.updated_at         # advanced
    assert second.detector_conf == 0.99                 # field updated in place


def test_bulk_upsert_count(conn):
    n = upsert_records(conn, [_rec(idx=i) for i in range(1, 6)])
    assert n == 5
    assert conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0] == 5


# --- stage writes ----------------------------------------------------------- #

def test_stage_writes_persist_independently(conn):
    r = _rec()
    upsert_record(conn, r)
    update_species(conn, r.record_id, "eurasian lynx", 0.91, "uuid;...;eurasian lynx")
    update_embedding(conn, r.record_id, r.record_id, "data/reid_demo/emb.pkl")
    update_cluster(conn, r.record_id, 3, 0.88, is_candidate_new=0)
    update_review(conn, r.record_id, "confirmed", "looks like the same cat")
    got = get_record(conn, r.record_id)
    assert got.species == "eurasian lynx" and got.species_conf == 0.91
    assert got.species_class == "uuid;...;eurasian lynx"
    assert got.embedding_ref == r.record_id
    assert got.embedding_path == "data/reid_demo/emb.pkl"
    assert got.cluster_id == 3 and got.cluster_conf == 0.88 and got.is_candidate_new == 0
    assert got.review_status == "confirmed" and got.review_note == "looks like the same cat"


def test_update_review_invalid_status_raises(conn):
    r = _rec()
    upsert_record(conn, r)
    with pytest.raises(ValueError):
        update_review(conn, r.record_id, "bogus")


def test_update_review_can_reassign_cluster(conn):
    r = _rec()
    upsert_record(conn, r)
    update_cluster(conn, r.record_id, 3, 0.88)
    update_review(conn, r.record_id, "merged", "merge into 7", cluster_id=7)
    got = get_record(conn, r.record_id)
    assert got.cluster_id == 7 and got.review_status == "merged"


def test_stage_write_unknown_record_raises(conn):
    with pytest.raises(KeyError):
        update_species(conn, "nope__crop1", "x", 0.5)


# --- update_extra ----------------------------------------------------------- #

def test_update_extra_merges_and_advances_updated(conn):
    r = _rec()
    upsert_record(conn, r)
    update_extra(conn, r.record_id, "species_kept", True)
    before = get_record(conn, r.record_id)
    time.sleep(1.05)
    update_extra(conn, r.record_id, "note", "x")
    got = get_record(conn, r.record_id)
    blob = json.loads(got.extra_json)
    assert blob.get("species_kept") is True   # first key preserved (merge, not overwrite)
    assert blob.get("note") == "x"
    assert got.updated_at > before.updated_at


def test_update_extra_unknown_record_raises(conn):
    with pytest.raises(KeyError):
        update_extra(conn, "nope__crop1", "k", 1)


# --- orientation normalization & validation --------------------------------- #

def test_orientation_empty_normalizes_to_unknown(conn):
    r = _rec(idx=2, orientation="")
    upsert_record(conn, r)
    assert get_record(conn, r.record_id).orientation == "unknown"


def test_orientation_none_stays_none(conn):
    r = _rec(idx=3, orientation=None)
    upsert_record(conn, r)
    assert get_record(conn, r.record_id).orientation is None


def test_orientation_invalid_raises(conn):
    with pytest.raises(ValueError):
        upsert_record(conn, _rec(idx=4, orientation="sideways"))


# --- queries & count_by ----------------------------------------------------- #

def test_query_records_filters(conn):
    upsert_record(conn, _rec(stem="A", idx=1, dataset="MedvednicaDS", species="eurasian lynx"))
    upsert_record(conn, _rec(stem="B", idx=1, dataset="MedvednicaDS", species="red fox"))
    upsert_record(conn, _rec(stem="C", idx=1, dataset="LeopardID2022", species="leopard"))
    # embedding only on A
    update_embedding(conn, "A__crop1", "A__crop1", "e.pkl")
    update_cluster(conn, "A__crop1", 5, 0.9)

    assert {r.source_stem for r in query_records(conn, dataset="MedvednicaDS")} == {"A", "B"}
    assert [r.source_stem for r in query_records(conn, species="leopard")] == ["C"]
    assert [r.source_stem for r in query_records(conn, has_embedding=True)] == ["A"]
    assert {r.source_stem for r in query_records(conn, has_embedding=False)} == {"B", "C"}
    assert [r.source_stem for r in query_records(conn, cluster_id=5)] == ["A"]
    assert len(query_records(conn, dataset="MedvednicaDS", species="eurasian lynx",
                             has_embedding=True, cluster_id=5)) == 1
    assert len(query_records(conn, limit=2)) == 2


def test_count_by(conn):
    upsert_record(conn, _rec(stem="A", idx=1, species="eurasian lynx"))
    upsert_record(conn, _rec(stem="B", idx=1, species="eurasian lynx"))
    upsert_record(conn, _rec(stem="C", idx=1, species="red fox"))
    update_cluster(conn, "A__crop1", 1, 0.9)
    update_cluster(conn, "B__crop1", 1, 0.9)
    update_cluster(conn, "C__crop1", 2, 0.9)
    assert count_by(conn, "species") == {"eurasian lynx": 2, "red fox": 1}
    assert count_by(conn, "cluster_id") == {1: 2, 2: 1}


def test_count_by_unknown_column_raises(conn):
    with pytest.raises(ValueError):
        count_by(conn, "not_a_column")


# --- export / import -------------------------------------------------------- #

def test_export_csv_header_matches_columns_and_import_roundtrip(conn, tmp_path):
    upsert_record(conn, _rec(stem="A", idx=1))
    upsert_record(conn, _rec(stem="B", idx=1, orientation="left", cluster_id=2,
                             cluster_conf=0.7, is_candidate_new=0))
    csv_path = str(tmp_path / "out.csv")
    export_records(conn, csv_path, fmt="csv")
    import csv as _csv
    with open(csv_path, newline="") as fh:
        header = next(_csv.reader(fh))
    assert header == COLUMNS

    conn2 = connect(str(tmp_path / "rt.sqlite"))
    n = import_records(conn2, csv_path)
    assert n == 2
    a = get_record(conn2, "A__crop1")
    b = get_record(conn2, "B__crop1")
    assert a.bbox_x == 0.49 and a.det_index == 1
    assert b.orientation == "left" and b.cluster_id == 2 and b.is_candidate_new == 0


def test_export_import_parquet_when_available(conn, tmp_path):
    pytest.importorskip("pyarrow")
    upsert_record(conn, _rec(stem="A", idx=1))
    upsert_record(conn, _rec(stem="B", idx=1, species="red fox", cluster_id=9, cluster_conf=0.5))
    pq_path = str(tmp_path / "out.parquet")
    out = export_records(conn, pq_path, fmt="parquet")
    assert out == pq_path and os.path.exists(pq_path)
    conn2 = connect(str(tmp_path / "rt_pq.sqlite"))
    n = import_records(conn2, pq_path)
    assert n == 2
    assert get_record(conn2, "B__crop1").cluster_id == 9


def test_to_dataframe_columns_order(conn):
    upsert_record(conn, _rec())
    df = to_dataframe(conn)
    assert list(df.columns) == COLUMNS
    assert len(df) == 1


# --- versioning & guards ---------------------------------------------------- #

def test_connect_succeeds_on_supported_sqlite():
    # target env bundles >= 3.24
    assert sqlite3.sqlite_version_info >= (3, 24, 0)


def test_schema_version_mismatch_raises(tmp_path):
    path = str(tmp_path / "ver.sqlite")
    conn = connect(path)
    # Tamper with the stored schema_version.
    conn.execute("UPDATE meta SET value='999' WHERE key='schema_version'")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError):
        connect(path)


def test_connect_create_false_on_uninitialized_raises(tmp_path):
    path = str(tmp_path / "empty.sqlite")
    # touch an empty sqlite file with no schema
    sqlite3.connect(path).close()
    with pytest.raises(RuntimeError):
        connect(path, create=False)


def test_init_db_idempotent(tmp_path):
    path = str(tmp_path / "idem.sqlite")
    conn = connect(path)
    init_db(conn)
    init_db(conn)  # second call must not error
    assert conn.execute(
        "SELECT value FROM meta WHERE key='schema_version'"
    ).fetchone()[0] == str(SCHEMA_VERSION)
