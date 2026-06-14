"""reid_demo.store — canonical detection-record store & data contract (T01).

This module is the backbone of the open-set lynx re-ID demo pipeline. It defines
ONE per-crop "detection record" schema and a SQLite-backed store that every other
ticket (T02–T12) reads from and writes to through this single access API.

It contains NO detection, classification, embedding, or clustering logic — it only
stores and serves records. See ``reid_demo/DATA_CONTRACT.md`` for the human-readable
contract and ``STATUS_BOARD.md`` for the overall plan and binding design decisions.

Core store uses only the Python standard library (``sqlite3``). ``pandas``/``pyarrow``
are OPTIONAL and imported lazily only for ``to_dataframe`` / Parquet export.
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import os
import sqlite3
import sys
import warnings
from dataclasses import dataclass, fields as _dataclass_fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# --------------------------------------------------------------------------- #
# Module-level constants (exact names — downstream tickets import these)
# --------------------------------------------------------------------------- #

SCHEMA_VERSION: int = 1
DEFAULT_DB_PATH: str = "data/reid_demo/reid_demo.sqlite"
TABLE_NAME: str = "detections"
META_TABLE: str = "meta"

#: The 28 canonical columns, in order. ``export_records`` / ``to_dataframe`` and the
#: ``detections`` table both follow this order exactly.
COLUMNS: List[str] = [
    "record_id",          # 1  TEXT  PK   (T02)
    "source_image",       # 2  TEXT       (T02)
    "source_stem",        # 3  TEXT       (T02)
    "det_index",          # 4  INTEGER    (T02)
    "crop_path",          # 5  TEXT       (T02)
    "bbox_x",             # 6  REAL       (T02)
    "bbox_y",             # 7  REAL       (T02)
    "bbox_w",             # 8  REAL       (T02)
    "bbox_h",             # 9  REAL       (T02)
    "detector_conf",      # 10 REAL       (T02)
    "camera_id",          # 11 TEXT       (T02)
    "timestamp",          # 12 TEXT       (T02)
    "species",            # 13 TEXT       (T03)
    "species_conf",       # 14 REAL       (T03)
    "species_class",      # 15 TEXT       (T03)
    "embedding_ref",      # 16 TEXT       (T04)
    "embedding_path",     # 17 TEXT       (T04)
    "cluster_id",         # 18 INTEGER    (T05)
    "cluster_conf",       # 19 REAL       (T05)
    "is_candidate_new",   # 20 INTEGER    (T05)
    "orientation",        # 21 TEXT       (T02)
    "gt_identity",        # 22 TEXT       (T02)
    "review_status",      # 23 TEXT       (T08)
    "review_note",        # 24 TEXT       (T08)
    "dataset",            # 25 TEXT       (T02/T10)
    "extra_json",         # 26 TEXT       (any, via update_extra)
    "created_at",         # 27 TEXT       (store)
    "updated_at",         # 28 TEXT       (store)
]

REVIEW_STATUSES: set = {"unreviewed", "confirmed", "rejected", "merged", "split"}
ORIENTATIONS: set = {"left", "right", "front", "back", "down", "unknown"}

#: Column → python coercion class, for export/import round-trips.
_INT_COLS = {"det_index", "cluster_id", "is_candidate_new"}
_FLOAT_COLS = {
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "detector_conf", "species_conf", "cluster_conf",
}

#: Table DDL. Names/types match COLUMNS exactly. NOT NULL / DEFAULT constraints do
#: not appear in PRAGMA table_info's ``type`` column, so ``type`` stays a bare
#: TEXT/INTEGER/REAL as the contract requires.
_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    record_id        TEXT PRIMARY KEY,
    source_image     TEXT    NOT NULL,
    source_stem      TEXT    NOT NULL,
    det_index        INTEGER NOT NULL,
    crop_path        TEXT    NOT NULL,
    bbox_x           REAL    NOT NULL,
    bbox_y           REAL    NOT NULL,
    bbox_w           REAL    NOT NULL,
    bbox_h           REAL    NOT NULL,
    detector_conf    REAL,
    camera_id        TEXT,
    timestamp        TEXT,
    species          TEXT,
    species_conf     REAL,
    species_class    TEXT,
    embedding_ref    TEXT,
    embedding_path   TEXT,
    cluster_id       INTEGER,
    cluster_conf     REAL,
    is_candidate_new INTEGER,
    orientation      TEXT,
    gt_identity      TEXT,
    review_status    TEXT    NOT NULL DEFAULT 'unreviewed',
    review_note      TEXT,
    dataset          TEXT,
    extra_json       TEXT    DEFAULT '{{}}',
    created_at       TEXT    NOT NULL,
    updated_at       TEXT    NOT NULL
)
"""

_INDEXES = [
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_dataset       ON {TABLE_NAME}(dataset)",
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_cluster_id    ON {TABLE_NAME}(cluster_id)",
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_species       ON {TABLE_NAME}(species)",
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_review_status ON {TABLE_NAME}(review_status)",
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_dataset_cluster ON {TABLE_NAME}(dataset, cluster_id)",
]


# --------------------------------------------------------------------------- #
# Canonical detection record
# --------------------------------------------------------------------------- #

@dataclass
class DetectionRecord:
    """One crop = one row. Field order matches COLUMNS. See DATA_CONTRACT.md."""

    record_id: str
    source_image: str
    source_stem: str
    det_index: int
    crop_path: str
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    detector_conf: Optional[float] = None
    camera_id: Optional[str] = None
    timestamp: Optional[str] = None
    species: Optional[str] = None
    species_conf: Optional[float] = None
    species_class: Optional[str] = None
    embedding_ref: Optional[str] = None
    embedding_path: Optional[str] = None
    cluster_id: Optional[int] = None
    cluster_conf: Optional[float] = None
    is_candidate_new: Optional[int] = None
    orientation: Optional[str] = None
    gt_identity: Optional[str] = None
    review_status: str = "unreviewed"
    review_note: Optional[str] = None
    dataset: Optional[str] = None
    extra_json: str = "{}"
    created_at: Optional[str] = None   # set by store on insert
    updated_at: Optional[str] = None   # set by store on write


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _now() -> str:
    """Store timestamp, second resolution (contract: isoformat timespec='seconds')."""
    return datetime.now().isoformat(timespec="seconds")


def make_record_id(source_stem: str, det_index: int) -> str:
    """Return the canonical record_id: f'{source_stem}__crop{det_index}'.

    Single source of truth so T02 (writer) and T08 (reviewer) agree.
    """
    return f"{source_stem}__crop{det_index}"


def _normalize_orientation(value: Optional[str]) -> Optional[str]:
    """Normalize orientation at ingest (D4): '' / missing -> 'unknown', None stays None."""
    if value is None:
        return None
    if value == "":
        return "unknown"
    return value


def _validate_record(record: DetectionRecord) -> None:
    """Validate review_status and orientation. Raises ValueError on bad values."""
    if record.review_status not in REVIEW_STATUSES:
        raise ValueError(
            f"review_status {record.review_status!r} not in {sorted(REVIEW_STATUSES)}"
        )
    if record.orientation is not None and record.orientation not in ORIENTATIONS:
        raise ValueError(
            f"orientation {record.orientation!r} not in {sorted(ORIENTATIONS)} (or None)"
        )
    # Bbox ranges are NOT hard-validated (detectors emit slightly out-of-range values);
    # warn only, never crash ingestion.
    for name in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"):
        v = getattr(record, name)
        if v is not None and not (-0.01 <= v <= 1.01):
            warnings.warn(f"{name}={v} outside normalized [0,1] range for {record.record_id}")


def _row_to_record(row: sqlite3.Row) -> DetectionRecord:
    """Map a sqlite3.Row (all 28 columns) to a DetectionRecord."""
    return DetectionRecord(**{c: row[c] for c in COLUMNS})


# --------------------------------------------------------------------------- #
# Connect / schema
# --------------------------------------------------------------------------- #

def _verify_sqlite_version() -> None:
    """ON CONFLICT upserts need SQLite >= 3.24. Fail loudly elsewhere (D3)."""
    if sqlite3.sqlite_version_info < (3, 24, 0):
        raise RuntimeError(
            f"SQLite {sqlite3.sqlite_version} is too old for ON CONFLICT upserts; "
            "need >= 3.24.0 (Python 3.12 bundles >= 3.37)."
        )


def init_db(conn: sqlite3.Connection) -> None:
    """Create `detections` + `meta` tables and indexes if absent; stamp SCHEMA_VERSION.

    Idempotent. Does NOT overwrite an existing schema_version (so mismatches are
    detectable by connect()).
    """
    conn.execute(_CREATE_TABLE_SQL)
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {META_TABLE} (key TEXT PRIMARY KEY, value TEXT)"
    )
    for idx_sql in _INDEXES:
        conn.execute(idx_sql)
    conn.execute(
        f"INSERT OR IGNORE INTO {META_TABLE}(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()


def _verify_schema_version(conn: sqlite3.Connection) -> None:
    """Raise RuntimeError if stored schema_version != SCHEMA_VERSION (or store uninit)."""
    try:
        row = conn.execute(
            f"SELECT value FROM {META_TABLE} WHERE key = 'schema_version'"
        ).fetchone()
    except sqlite3.OperationalError:
        raise RuntimeError(
            "store is not initialized (no meta table); open with create=True first."
        )
    if row is None:
        raise RuntimeError(
            "store is not initialized (no schema_version); open with create=True first."
        )
    stored = int(row[0])
    if stored != SCHEMA_VERSION:
        raise RuntimeError(
            f"schema_version mismatch: store has {stored}, code expects {SCHEMA_VERSION}. "
            "This demo does not auto-migrate; recreate the DB."
        )


def connect(db_path: str = DEFAULT_DB_PATH, *, create: bool = True) -> sqlite3.Connection:
    """Open (and if create=True, initialize schema for) the SQLite store.

    Creates parent dirs, sets row_factory = sqlite3.Row, verifies the runtime SQLite
    version (>= 3.24 for ON CONFLICT upserts) and the stored SCHEMA_VERSION. Raises
    RuntimeError on version mismatch or (when create=False) an uninitialized store.
    """
    _verify_sqlite_version()
    if db_path != ":memory:":
        parent = Path(db_path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    if create:
        init_db(conn)
    _verify_schema_version(conn)
    return conn


# --------------------------------------------------------------------------- #
# Writes
# --------------------------------------------------------------------------- #

def _record_values(record: DetectionRecord, *, now: str) -> tuple:
    """Validate + normalize a record and produce a value tuple in COLUMNS order.

    Preserves a caller-supplied created_at (e.g. on import); otherwise stamps `now`.
    updated_at is always set to `now` (contract: always refresh updated_at).
    """
    record.orientation = _normalize_orientation(record.orientation)
    if record.review_status is None:
        record.review_status = "unreviewed"
    if record.extra_json is None:
        record.extra_json = "{}"
    _validate_record(record)
    created_at = record.created_at or now
    values = []
    for col in COLUMNS:
        if col == "created_at":
            values.append(created_at)
        elif col == "updated_at":
            values.append(now)
        else:
            values.append(getattr(record, col))
    return tuple(values)


# All columns except the PK and created_at are refreshed from the incoming row on
# conflict; created_at is preserved, updated_at advances.
_UPDATE_COLS = [c for c in COLUMNS if c not in ("record_id", "created_at")]
_UPSERT_SQL = (
    f"INSERT INTO {TABLE_NAME} ({', '.join(COLUMNS)}) "
    f"VALUES ({', '.join('?' for _ in COLUMNS)}) "
    f"ON CONFLICT(record_id) DO UPDATE SET "
    + ", ".join(f"{c} = excluded.{c}" for c in _UPDATE_COLS)
)


def upsert_record(conn: sqlite3.Connection, record: DetectionRecord) -> None:
    """Insert or replace one record by record_id.

    Sets created_at on first insert (preserved on conflict), always refreshes
    updated_at. Validates review_status and orientation. Commits.
    """
    conn.execute(_UPSERT_SQL, _record_values(record, now=_now()))
    conn.commit()


def upsert_records(conn: sqlite3.Connection, records: Iterable[DetectionRecord]) -> int:
    """Bulk upsert in a single transaction; returns count written."""
    now = _now()
    rows = [_record_values(r, now=now) for r in records]
    if not rows:
        return 0
    with conn:  # single transaction (commit on success, rollback on error)
        conn.executemany(_UPSERT_SQL, rows)
    return len(rows)


def _require_record(conn: sqlite3.Connection, record_id: str) -> None:
    exists = conn.execute(
        f"SELECT 1 FROM {TABLE_NAME} WHERE record_id = ?", (record_id,)
    ).fetchone()
    if exists is None:
        raise KeyError(f"no record with record_id={record_id!r}")


def update_species(conn, record_id: str, species: str, species_conf: float,
                   species_class: Optional[str] = None) -> None:
    """T03 stage write. Refreshes updated_at. Commits."""
    _require_record(conn, record_id)
    conn.execute(
        f"UPDATE {TABLE_NAME} SET species = ?, species_conf = ?, species_class = ?, "
        f"updated_at = ? WHERE record_id = ?",
        (species, species_conf, species_class, _now(), record_id),
    )
    conn.commit()


def update_embedding(conn, record_id: str, embedding_ref: str, embedding_path: str) -> None:
    """T04 stage write. Commits."""
    _require_record(conn, record_id)
    conn.execute(
        f"UPDATE {TABLE_NAME} SET embedding_ref = ?, embedding_path = ?, "
        f"updated_at = ? WHERE record_id = ?",
        (embedding_ref, embedding_path, _now(), record_id),
    )
    conn.commit()


def update_cluster(conn, record_id: str, cluster_id: int, cluster_conf: float,
                   is_candidate_new: int = 0) -> None:
    """T05 stage write. Commits."""
    _require_record(conn, record_id)
    conn.execute(
        f"UPDATE {TABLE_NAME} SET cluster_id = ?, cluster_conf = ?, "
        f"is_candidate_new = ?, updated_at = ? WHERE record_id = ?",
        (cluster_id, cluster_conf, int(is_candidate_new), _now(), record_id),
    )
    conn.commit()


def update_review(conn, record_id: str, review_status: str,
                  review_note: Optional[str] = None,
                  cluster_id: Optional[int] = None) -> None:
    """T08 stage write. Validates review_status. Optionally re-assigns cluster_id
    (when a human merges/splits). Commits."""
    if review_status not in REVIEW_STATUSES:
        raise ValueError(
            f"review_status {review_status!r} not in {sorted(REVIEW_STATUSES)}"
        )
    _require_record(conn, record_id)
    if cluster_id is None:
        conn.execute(
            f"UPDATE {TABLE_NAME} SET review_status = ?, review_note = ?, "
            f"updated_at = ? WHERE record_id = ?",
            (review_status, review_note, _now(), record_id),
        )
    else:
        conn.execute(
            f"UPDATE {TABLE_NAME} SET review_status = ?, review_note = ?, "
            f"cluster_id = ?, updated_at = ? WHERE record_id = ?",
            (review_status, review_note, cluster_id, _now(), record_id),
        )
    conn.commit()


def update_extra(conn, record_id: str, key: str, value) -> None:
    """Set a single key in this record's `extra_json` blob WITHOUT a schema change.

    Reads current extra_json (default '{}'), parses as a dict, sets dict[key]=value,
    re-serialises and writes back. Refreshes updated_at. Commits. `value` must be
    JSON-serialisable. Raises KeyError if record_id absent, ValueError if extra_json
    is not a JSON object.
    """
    row = conn.execute(
        f"SELECT extra_json FROM {TABLE_NAME} WHERE record_id = ?", (record_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"no record with record_id={record_id!r}")
    raw = row[0]
    try:
        blob = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"extra_json for {record_id!r} is not valid JSON: {exc}") from exc
    if not isinstance(blob, dict):
        raise ValueError(f"extra_json for {record_id!r} is not a JSON object")
    blob[key] = value  # merge: preserves existing keys
    conn.execute(
        f"UPDATE {TABLE_NAME} SET extra_json = ?, updated_at = ? WHERE record_id = ?",
        (json.dumps(blob), _now(), record_id),
    )
    conn.commit()


# --------------------------------------------------------------------------- #
# Reads
# --------------------------------------------------------------------------- #

def get_record(conn: sqlite3.Connection, record_id: str) -> Optional[DetectionRecord]:
    """Fetch one record or None."""
    row = conn.execute(
        f"SELECT * FROM {TABLE_NAME} WHERE record_id = ?", (record_id,)
    ).fetchone()
    return _row_to_record(row) if row is not None else None


def query_records(
    conn: sqlite3.Connection,
    *,
    dataset: Optional[str] = None,
    species: Optional[str] = None,
    cluster_id: Optional[int] = None,
    review_status: Optional[str] = None,
    has_embedding: Optional[bool] = None,
    orientation: Optional[str] = None,
    where_sql: Optional[str] = None,
    where_params: tuple = (),
    order_by: str = "record_id",
    limit: Optional[int] = None,
) -> List[DetectionRecord]:
    """Filtered fetch. All filters AND-combined. Returns dataclasses."""
    clauses: List[str] = []
    params: List[Any] = []
    if dataset is not None:
        clauses.append("dataset = ?")
        params.append(dataset)
    if species is not None:
        clauses.append("species = ?")
        params.append(species)
    if cluster_id is not None:
        clauses.append("cluster_id = ?")
        params.append(cluster_id)
    if review_status is not None:
        clauses.append("review_status = ?")
        params.append(review_status)
    if has_embedding is True:
        clauses.append("embedding_ref IS NOT NULL")
    elif has_embedding is False:
        clauses.append("embedding_ref IS NULL")
    if orientation is not None:
        clauses.append("orientation = ?")
        params.append(orientation)
    if where_sql:
        clauses.append(f"({where_sql})")
        params.extend(where_params)

    sql = f"SELECT * FROM {TABLE_NAME}"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += f" ORDER BY {order_by}"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    rows = conn.execute(sql, tuple(params)).fetchall()
    return [_row_to_record(r) for r in rows]


def count_by(conn, column: str, *, dataset: Optional[str] = None) -> dict:
    """GROUP BY helper: returns {value: count} for the given column.

    e.g. count_by(conn, 'species'), count_by(conn, 'cluster_id'). Used by T06/T07/T09.
    """
    if column not in COLUMNS:
        raise ValueError(f"unknown column {column!r}; must be one of COLUMNS")
    sql = f"SELECT {column} AS v, COUNT(*) AS n FROM {TABLE_NAME}"
    params: List[Any] = []
    if dataset is not None:
        sql += " WHERE dataset = ?"
        params.append(dataset)
    sql += f" GROUP BY {column}"
    return {row["v"]: row["n"] for row in conn.execute(sql, tuple(params)).fetchall()}


# --------------------------------------------------------------------------- #
# Portable fallback: export / import / dataframe
# --------------------------------------------------------------------------- #

def to_dataframe(conn, *, dataset: Optional[str] = None):
    """Return all (or dataset-filtered) records as a pandas DataFrame in COLUMNS order.

    Convenience for T06/T07/T09. Raises an informative ImportError if pandas missing.
    """
    try:
        import pandas as pd  # lazy, optional
    except ImportError as exc:  # pragma: no cover - exercised only without pandas
        raise ImportError(
            "to_dataframe requires pandas (optional dependency). "
            "Install pandas or use query_records()."
        ) from exc
    records = query_records(conn, dataset=dataset, order_by="record_id", limit=None)
    data = [{c: getattr(r, c) for c in COLUMNS} for r in records]
    return pd.DataFrame(data, columns=COLUMNS)


def export_records(conn, out_path: str, *, fmt: str = "parquet",
                   dataset: Optional[str] = None) -> str:
    """Dump rows to .parquet (if pandas/pyarrow available) or .csv. Returns out_path.

    fmt in {'parquet','csv'}; auto-fallback to csv (with a warning) if pyarrow/pandas
    is missing. The CSV header is exactly COLUMNS, in order.
    """
    if fmt not in ("parquet", "csv"):
        raise ValueError(f"fmt must be 'parquet' or 'csv', got {fmt!r}")

    if fmt == "parquet":
        try:
            import pandas  # noqa: F401  (presence check)
            import pyarrow  # noqa: F401  (presence check)
        except ImportError:
            warnings.warn(
                "pyarrow/pandas unavailable; falling back to CSV export.", RuntimeWarning
            )
            fmt = "csv"
            if out_path.endswith(".parquet"):
                out_path = out_path[: -len(".parquet")] + ".csv"

    parent = Path(out_path).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        df = to_dataframe(conn, dataset=dataset)
        df.to_parquet(out_path, index=False)
        return out_path

    # CSV via stdlib (header == COLUMNS exactly; None -> empty string).
    records = query_records(conn, dataset=dataset, order_by="record_id", limit=None)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = _csv.writer(fh)
        writer.writerow(COLUMNS)
        for r in records:
            writer.writerow(["" if getattr(r, c) is None else getattr(r, c) for c in COLUMNS])
    return out_path


def _isna(value) -> bool:
    """True for SQL/import NULL-ish values: None, '', or float NaN."""
    if value is None:
        return True
    if isinstance(value, str) and value == "":
        return True
    if isinstance(value, float) and value != value:  # NaN
        return True
    return False


def _coerce(value, col: str):
    """Coerce one imported cell to the python type implied by `col`."""
    if _isna(value):
        # review_status / extra_json have non-null defaults handled below
        return None
    if col in _INT_COLS:
        return int(float(value))  # tolerate "3" / "3.0" / 3.0
    if col in _FLOAT_COLS:
        return float(value)
    return str(value)


def _row_dict_to_record(row: Dict[str, Any]) -> DetectionRecord:
    """Build a DetectionRecord from an imported {column: value} dict (with coercion)."""
    kwargs: Dict[str, Any] = {}
    for col in COLUMNS:
        kwargs[col] = _coerce(row.get(col), col)
    if kwargs.get("review_status") is None:
        kwargs["review_status"] = "unreviewed"
    if kwargs.get("extra_json") is None:
        kwargs["extra_json"] = "{}"
    return DetectionRecord(**kwargs)


def import_records(conn, in_path: str) -> int:
    """Load .parquet/.csv produced by export_records back into the store (upsert).

    Returns count. Format inferred from extension.
    """
    suffix = Path(in_path).suffix.lower()
    rows: List[Dict[str, Any]] = []
    if suffix == ".parquet":
        try:
            import pandas as pd  # lazy, optional
        except ImportError as exc:
            raise ImportError(
                "importing a .parquet file requires pandas/pyarrow."
            ) from exc
        df = pd.read_parquet(in_path)
        rows = df.to_dict(orient="records")
    elif suffix == ".csv":
        with open(in_path, newline="", encoding="utf-8") as fh:
            rows = list(_csv.DictReader(fh))
    else:
        raise ValueError(f"unsupported import format {suffix!r}; use .parquet or .csv")

    records = [_row_dict_to_record(r) for r in rows]
    return upsert_records(conn, records)


# --------------------------------------------------------------------------- #
# Self-test / info CLI
# --------------------------------------------------------------------------- #

def _selftest(db_path: str) -> bool:
    """Create a temp DB, round-trip records and stage writes. Returns True on success."""
    # Start from a clean file so re-runs are deterministic.
    for p in (db_path, db_path + "-wal", db_path + "-shm"):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass

    conn = connect(db_path)

    # Round-trip a single record + all stage writes.
    rid = make_record_id("IMG_0066", 1)
    assert rid == "IMG_0066__crop1", rid
    rec = DetectionRecord(
        record_id=rid,
        source_image="data/MedvednicaDS/animal_images/IMG_0066.JPG",
        source_stem="IMG_0066", det_index=1,
        crop_path="data/reid_demo/crops/IMG_0066__crop1.jpg",
        bbox_x=0.49, bbox_y=0.04, bbox_w=0.05, bbox_h=0.17,
        detector_conf=0.78, camera_id="unknown_camera",
        timestamp="2025-06-02 04:27:51", dataset="MedvednicaDS",
    )
    upsert_record(conn, rec)
    got = get_record(conn, rid)
    assert got is not None and got.bbox_x == 0.49 and got.dataset == "MedvednicaDS"
    assert got.created_at and got.updated_at

    update_species(conn, rid, "eurasian lynx", 0.91, "uuid;...;eurasian lynx")
    update_embedding(conn, rid, rid, "data/reid_demo/emb.pkl")
    update_cluster(conn, rid, 3, 0.88, is_candidate_new=0)
    update_review(conn, rid, "confirmed", "looks like the same cat")
    update_extra(conn, rid, "species_kept", True)
    update_extra(conn, rid, "note", "x")
    got = get_record(conn, rid)
    assert got.species == "eurasian lynx" and got.cluster_id == 3
    assert got.review_status == "confirmed"
    blob = json.loads(got.extra_json)
    assert blob.get("species_kept") is True and blob.get("note") == "x", blob

    # orientation '' normalizes to 'unknown' at ingest (D4)
    rec2 = DetectionRecord(
        record_id=make_record_id("IMG_0066", 2),
        source_image="x", source_stem="IMG_0066", det_index=2, crop_path="x",
        bbox_x=0.0, bbox_y=0.0, bbox_w=1.0, bbox_h=1.0,
        orientation="", dataset="MedvednicaDS",
    )
    upsert_record(conn, rec2)
    assert get_record(conn, rec2.record_id).orientation == "unknown"

    # Invalid review_status / orientation must raise.
    try:
        update_review(conn, rid, "bogus")
        return False
    except ValueError:
        pass
    try:
        upsert_record(conn, DetectionRecord(
            record_id="bad", source_image="x", source_stem="x", det_index=1,
            crop_path="x", bbox_x=0, bbox_y=0, bbox_w=1, bbox_h=1,
            orientation="sideways"))
        return False
    except ValueError:
        pass

    # Idempotent upsert keeps row count at 1 per record_id.
    upsert_record(conn, rec)
    n = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    assert n == 2, n

    # Export/import round-trip (CSV path, always available).
    csv_path = db_path + ".csv"
    out = export_records(conn, csv_path, fmt="csv")
    with open(out, newline="", encoding="utf-8") as fh:
        header = next(_csv.reader(fh))
    assert header == COLUMNS, header
    conn2 = connect(db_path + ".roundtrip.sqlite")
    n_imported = import_records(conn2, out)
    assert n_imported == 2, n_imported

    print(f"[selftest] OK — {n} records round-tripped through {db_path}")
    return True


def _cmd_info(db_path: str) -> None:
    conn = connect(db_path, create=False)
    version = conn.execute(
        f"SELECT value FROM {META_TABLE} WHERE key='schema_version'"
    ).fetchone()[0]
    rows = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    print(f"db_path:        {db_path}")
    print(f"schema_version: {version}")
    print(f"row_count:      {rows}")
    print(f"count_by species:    {count_by(conn, 'species')}")
    print(f"count_by cluster_id: {count_by(conn, 'cluster_id')}")


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="reid_demo.store",
                                     description="Detection-record store (T01).")
    parser.add_argument("--selftest", action="store_true",
                        help="round-trip test on a temp DB; exit non-zero on failure")
    parser.add_argument("--info", action="store_true",
                        help="print schema_version, row count, species/cluster counts")
    parser.add_argument("--export", metavar="OUT",
                        help="export rows to OUT (.parquet or .csv)")
    parser.add_argument("--dataset", default=None,
                        help="restrict --export to one dataset")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="DB path")
    args = parser.parse_args(argv)

    if args.selftest:
        db = args.db if args.db != DEFAULT_DB_PATH else "/tmp/reid_selftest.sqlite"
        try:
            ok = _selftest(db)
        except AssertionError as exc:
            print(f"[selftest] FAILED: {exc}", file=sys.stderr)
            return 1
        return 0 if ok else 1

    if args.export:
        conn = connect(args.db, create=False)
        fmt = "parquet" if args.export.endswith(".parquet") else "csv"
        out = export_records(conn, args.export, fmt=fmt, dataset=args.dataset)
        print(f"exported -> {out}")
        return 0

    if args.info:
        _cmd_info(args.db)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
