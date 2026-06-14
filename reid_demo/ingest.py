"""reid_demo.ingest — ingestion + MegaDetector adapter (T02).

Front door of the open-set lynx re-ID demo pipeline. Takes raw camera-trap input,
keeps only animal detections above a confidence threshold (dropping empty frames,
persons and vehicles), crops each surviving box to a JPG, resolves camera/timestamp
metadata, and writes ONE T01 ``DetectionRecord`` per crop into the shared store via
``reid_demo.store`` (no direct SQL). Everything downstream (T03 species, T04
embeddings, T05 clustering, …) reads those rows.

Four input adapters:
  (a) ``ingest`` over a MegaDetector results JSON (the Medvednica primary format),
  (b) ``ingest`` over the flat ``animal_detections.json`` format,
  (c) ``ingest_from_images`` — run MegaDetector (repo venv) on a raw image folder,
  (d) ``ingest_wildlife_dataset`` — a labeled WildlifeReID-10k subset (LeopardID2022 /
      ATRW) ingested WHOLE-FRAME with ground-truth identity/orientation/species.

``det_index`` is **1-based**, assigned over the kept (animal, above-threshold)
detections in MegaDetector source-file order. This matches the existing
``…_crop1_conf…jpg`` naming and the T01 ``make_record_id`` contract:
``record_id == make_record_id(source_stem, det_index) == f"{source_stem}__crop{det_index}"``.

Field-population contract (which columns T02 sets):
  * A-track (MegaDetector / flat / raw-image): record_id, source_image, source_stem,
    det_index, crop_path, bbox_x/y/w/h, detector_conf, camera_id, timestamp, dataset,
    orientation="unknown". Everything else (species*/embedding*/cluster*/gt_identity/
    review_*) is left at the dataclass default (NULL / "unreviewed" / "{}").
  * B-track (ingest_wildlife_dataset): the above whole-frame fields PLUS gt_identity,
    species, and orientation populated from the dataset metadata. camera_id/timestamp
    stay None unless metadata supplies them.

Species labels on the A-track are T03's job (left NULL). Embeddings are T04's,
clustering is T05's, review fields are T08's — all left at their defaults here.

The module imports cleanly under plain ``python3`` WITHOUT pulling in torch or
megadetector: the megadetector import is lazy (inside ``ingest_from_images`` only),
and the WildlifeReID10k enrichment import is lazy (inside ``ingest_wildlife_dataset``).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from reid_demo import store
from reid_demo.store import (
    DetectionRecord,
    ORIENTATIONS,
    connect,
    make_record_id,
    upsert_records,
)

# --------------------------------------------------------------------------- #
# Module-level constants (exact names — downstream tickets / T10 import these)
# --------------------------------------------------------------------------- #

DEFAULT_MD_JSON: str = "data/MedvednicaDS/megadetector_results.json"
DEFAULT_IMAGES_DIR: str = "data/MedvednicaDS/animal_images"
DEFAULT_METADATA_CSV: str = "data/MedvednicaDS/trail_cam_data.csv"
DEFAULT_EXISTING_CROPS: str = "data/MedvednicaDS/animal_crops"
DEFAULT_CROPS_OUT: str = "data/reid_demo/crops"
DEFAULT_DATASET: str = "MedvednicaDS"
DEFAULT_CONF_THRESHOLD: float = 0.5
ANIMAL_CATEGORY_ID: str = "1"   # MegaDetector animal category

#: Fallback camera id (Medvednica's CSV yields exactly this).
UNKNOWN_CAMERA: str = "unknown_camera"


# --------------------------------------------------------------------------- #
# Detection parsing
# --------------------------------------------------------------------------- #

def _is_megadetector_results(data: Any) -> bool:
    """True if ``data`` is the MegaDetector-results format (top-level dict with an
    ``images`` or ``predictions`` list of per-frame records)."""
    return isinstance(data, dict) and (
        isinstance(data.get("images"), list)
        or isinstance(data.get("predictions"), list)
    )


def _frame_basename_and_camera(file_field: str) -> Tuple[str, Optional[str]]:
    """Split a MegaDetector ``file`` value (e.g. ``"Camera 1/IMG_0001.JPG"``) into
    ``(basename, camera_hint)``. The basename is the on-disk filename (flat under
    ``animal_images/``); the camera_hint is the leading subfolder (if any)."""
    p = Path(file_field)
    basename = p.name
    parent = str(p.parent)
    camera_hint: Optional[str] = None
    if parent not in ("", "."):
        # Use only the immediate leading folder as the camera hint.
        camera_hint = Path(file_field).parts[0]
    return basename, camera_hint


def load_detection_frames(
    md_json: str,
    *,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Parse EITHER the MegaDetector-results format (top-level ``images``/``predictions``)
    OR the flat ``animal_detections.json`` format into a normalized list of frame dicts:

        {"source_basename": "IMG_0066.JPG",
         "camera_hint": "Camera 1" | None,
         "timestamp": "2025-06-02 04:27:51" | None,
         "animal_dets": [ {"det_index": int (1-based), "bbox": [x,y,w,h],
                           "conf": float}, ... ],
         "n_person": int, "n_vehicle": int, "n_below_threshold": int}

    Persons/vehicles and empty frames are dropped from ``animal_dets`` (length-0 lists
    are allowed and contribute to ``frames_empty``). ``det_index`` is assigned 1-based
    over the kept animal detections in original file order, matching existing crop
    naming. The per-category drop counts are returned so ``ingest`` can build stats
    without re-parsing. The confidence filter is **per-detection** with ``>=``
    against ``conf_threshold`` (intentionally differs from the legacy whole-frame
    ``conf > 0.5`` cleaner in ``utils/clean_detections.py``).
    """
    with open(md_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    frames: List[Dict[str, Any]] = []

    if _is_megadetector_results(data):
        records = data.get("images")
        if records is None:
            records = data.get("predictions", [])
        # Category map: id -> name (e.g. "1" -> "animal"). Default to the documented
        # Medvednica map so a missing 'detection_categories' still works.
        cat_map = {
            str(k): v
            for k, v in (data.get("detection_categories") or {}).items()
        }
        if not cat_map:
            cat_map = {"1": "animal", "2": "person", "3": "vehicle"}

        for rec in records:
            file_field = rec.get("file") or rec.get("filepath") or ""
            basename, camera_hint = _frame_basename_and_camera(file_field)
            dets = rec.get("detections") or []

            animal_dets: List[Dict[str, Any]] = []
            n_person = n_vehicle = n_below = 0
            det_index = 0
            for det in dets:
                cat = str(det.get("category"))
                name = cat_map.get(cat, cat.lower())
                conf = float(det.get("conf", 0.0))
                if name == "animal":
                    if conf >= conf_threshold:
                        det_index += 1
                        animal_dets.append({
                            "det_index": det_index,
                            "bbox": list(det.get("bbox", [])),
                            "conf": conf,
                        })
                    else:
                        n_below += 1
                elif name == "person":
                    n_person += 1
                elif name == "vehicle":
                    n_vehicle += 1
                # unknown categories: ignored (not counted under animal stats)

            frames.append({
                "source_basename": basename,
                "camera_hint": camera_hint,
                "timestamp": rec.get("timestamp"),
                "animal_dets": animal_dets,
                "n_person": n_person,
                "n_vehicle": n_vehicle,
                "n_below_threshold": n_below,
            })
        return frames

    # Flat animal_detections.json format: {basename: [ {"bbox":[...],
    # "confidence": float}, ... ], ...}. Already animal-only; treat every box as
    # animal and read 'confidence'. No camera/person/vehicle info.
    if isinstance(data, dict):
        for basename, dets in data.items():
            animal_dets = []
            n_below = 0
            det_index = 0
            for det in (dets or []):
                conf = float(det.get("confidence", det.get("conf", 0.0)))
                if conf >= conf_threshold:
                    det_index += 1
                    animal_dets.append({
                        "det_index": det_index,
                        "bbox": list(det.get("bbox", [])),
                        "conf": conf,
                    })
                else:
                    n_below += 1
            frames.append({
                "source_basename": basename,
                "camera_hint": None,
                "timestamp": None,
                "animal_dets": animal_dets,
                "n_person": 0,
                "n_vehicle": 0,
                "n_below_threshold": n_below,
            })
        return frames

    raise ValueError(
        f"Unrecognized detection JSON structure in {md_json!r}: "
        "expected MegaDetector-results (top-level 'images'/'predictions') or a flat "
        "{basename: [{'bbox', 'confidence'}]} dict."
    )


# --------------------------------------------------------------------------- #
# Cropping
# --------------------------------------------------------------------------- #

def crop_for_detection(
    source_image_path: str,
    bbox: Tuple[float, float, float, float],
    crop_out_path: str,
    *,
    existing_crop_path: Optional[str] = None,
    write: bool = True,
) -> str:
    """Produce (or locate) the crop file for one detection and return its path.

    Reuse ``existing_crop_path`` if given and on disk (returned unchanged). Else, if
    ``write`` is True, crop the normalized bbox to pixels via
    ``(x*W, y*H, (x+w)*W, (y+h)*H)`` (the project's existing convention from
    ``deprecated/seminar_classify_species.py:26-33``), convert to RGB, and save a
    JPEG (quality 90) to ``crop_out_path`` (parents created). If ``write`` is False,
    return the path it WOULD write without creating the file.

    The source image is resolved to an ABSOLUTE path before opening. Pixel coords are
    clamped to ``[0,W]``/``[0,H]``; a degenerate box (<1px width/height) raises
    ``ValueError`` so the caller can skip it.
    """
    if existing_crop_path is not None and os.path.exists(existing_crop_path):
        return existing_crop_path

    if not write:
        return crop_out_path

    from PIL import Image  # lazy; Pillow is available in the venv

    abs_src = os.path.abspath(source_image_path)
    img = Image.open(abs_src).convert("RGB")
    W, H = img.size
    x, y, w, h = (float(v) for v in bbox)
    left = max(0.0, min(x * W, W))
    top = max(0.0, min(y * H, H))
    right = max(0.0, min((x + w) * W, W))
    bottom = max(0.0, min((y + h) * H, H))
    if (right - left) < 1.0 or (bottom - top) < 1.0:
        raise ValueError(
            f"degenerate crop box for {source_image_path}: "
            f"pixels=({left:.1f},{top:.1f},{right:.1f},{bottom:.1f}) bbox={bbox}"
        )
    crop = img.crop((left, top, right, bottom))
    parent = Path(crop_out_path).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)
    crop.save(crop_out_path, "JPEG", quality=90)
    return crop_out_path


def _find_legacy_crop(existing_crops_dir: Optional[str], stem: str, det_index: int) -> Optional[str]:
    """Return the single legacy crop matching ``{stem}_crop{det_index}_*.jpg`` in
    ``existing_crops_dir``, or None if zero/multiple matches (or dir missing)."""
    if not existing_crops_dir or not os.path.isdir(existing_crops_dir):
        return None
    pattern = os.path.join(existing_crops_dir, f"{stem}_crop{det_index}_*.jpg")
    matches = glob.glob(pattern)
    if len(matches) == 1:
        return matches[0]
    return None


# --------------------------------------------------------------------------- #
# Metadata resolution
# --------------------------------------------------------------------------- #

def resolve_metadata(
    metadata_csv: Optional[str],
) -> Dict[str, Dict[str, Optional[str]]]:
    """Return ``{image_basename: {"camera_id": str|None, "timestamp": str|None}}``
    parsed from ``trail_cam_data.csv`` (match by basename of the ``filepath`` column;
    ``camera`` -> camera_id, ``datetime`` -> timestamp). Returns ``{}`` if the csv is
    missing or None. Uses the stdlib ``csv`` module (no pandas dependency)."""
    out: Dict[str, Dict[str, Optional[str]]] = {}
    if not metadata_csv or not os.path.exists(metadata_csv):
        return out
    import csv as _csv
    with open(metadata_csv, newline="", encoding="utf-8") as fh:
        reader = _csv.DictReader(fh)
        for row in reader:
            filepath = row.get("filepath") or ""
            if not filepath:
                continue
            basename = Path(filepath).name
            camera = row.get("camera")
            camera = camera if camera not in (None, "") else None
            ts = row.get("datetime")
            ts = ts if ts not in (None, "") else None
            out[basename] = {"camera_id": camera, "timestamp": ts}
    return out


def _resolve_camera(meta_row: Optional[Dict[str, Optional[str]]],
                    camera_hint: Optional[str]) -> str:
    """camera_id precedence: CSV camera -> JSON camera_hint -> UNKNOWN_CAMERA."""
    if meta_row and meta_row.get("camera_id"):
        return meta_row["camera_id"]  # type: ignore[return-value]
    if camera_hint:
        return camera_hint
    return UNKNOWN_CAMERA


def _resolve_timestamp(meta_row: Optional[Dict[str, Optional[str]]],
                       frame_timestamp: Optional[str]) -> Optional[str]:
    """timestamp precedence: CSV datetime -> JSON frame timestamp -> None."""
    if meta_row and meta_row.get("timestamp"):
        return meta_row["timestamp"]
    if frame_timestamp:
        return frame_timestamp
    return None


# --------------------------------------------------------------------------- #
# A-track ingest
# --------------------------------------------------------------------------- #

def _empty_stats(dataset: str, db_path: str) -> Dict[str, Any]:
    return {
        "frames_total": 0,
        "frames_empty": 0,
        "frames_with_animals": 0,
        "dets_total": 0,
        "dets_animal": 0,
        "dets_person": 0,
        "dets_vehicle": 0,
        "dets_below_threshold": 0,
        "crops_written": 0,
        "crops_reused": 0,
        "crops_missing_source": 0,
        "records_upserted": 0,
        "dataset": dataset,
        "db_path": db_path,
    }


def ingest(
    *,
    md_json: Optional[str] = DEFAULT_MD_JSON,
    images_dir: str = DEFAULT_IMAGES_DIR,
    metadata_csv: Optional[str] = DEFAULT_METADATA_CSV,
    existing_crops_dir: Optional[str] = DEFAULT_EXISTING_CROPS,
    crops_out_dir: str = DEFAULT_CROPS_OUT,
    db_path: Optional[str] = None,          # None -> store.DEFAULT_DB_PATH
    dataset: str = DEFAULT_DATASET,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    write_crops: bool = True,               # False -> record paths only, no JPGs
    limit: Optional[int] = None,            # cap on number of SOURCE frames (debug)
) -> Dict[str, Any]:
    """Run the full A-track ingestion: load/filter detections, crop (reusing legacy
    crops where present), resolve camera/timestamp, and upsert one ``DetectionRecord``
    per kept crop into the store. Returns a stats dict (see ``IngestStats`` keys).
    Opens the store via ``reid_demo.store.connect()``.

    ``limit`` caps the number of source frames that contribute at least one kept
    animal detection (i.e. frames that actually produce work). Fully-empty frames are
    still scanned for the ``frames_total`` / ``frames_empty`` / ``pct_empty`` stats but
    do not consume the limit budget, so a small ``--limit`` on empty-heavy data (the
    Medvednica JSON opens with dozens of empty frames) still produces crops.
    """
    if md_json is None:
        raise ValueError("ingest() requires md_json (or use ingest_from_images).")
    if not os.path.exists(md_json):
        raise FileNotFoundError(f"detection JSON not found: {md_json}")

    resolved_db = db_path or store.DEFAULT_DB_PATH
    stats = _empty_stats(dataset, resolved_db)

    frames = load_detection_frames(md_json, conf_threshold=conf_threshold)
    meta = resolve_metadata(metadata_csv)

    records: List[DetectionRecord] = []
    frames_with_animals_consumed = 0

    for frame in frames:
        basename = frame["source_basename"]
        animal_dets = frame["animal_dets"]

        # Stats over ALL frames (limit-independent denominators for pct_empty).
        stats["frames_total"] += 1
        stats["dets_person"] += frame.get("n_person", 0)
        stats["dets_vehicle"] += frame.get("n_vehicle", 0)
        stats["dets_below_threshold"] += frame.get("n_below_threshold", 0)
        stats["dets_animal"] += len(animal_dets)
        stats["dets_total"] += (
            len(animal_dets) + frame.get("n_person", 0)
            + frame.get("n_vehicle", 0) + frame.get("n_below_threshold", 0)
        )
        if not animal_dets:
            stats["frames_empty"] += 1
            continue
        stats["frames_with_animals"] += 1

        # Apply the limit over frames that actually do work.
        if limit is not None and frames_with_animals_consumed >= limit:
            continue
        frames_with_animals_consumed += 1

        stem = Path(basename).stem
        source_image_path = os.path.join(images_dir, basename)
        meta_row = meta.get(basename)
        camera_id = _resolve_camera(meta_row, frame.get("camera_hint"))
        timestamp = _resolve_timestamp(meta_row, frame.get("timestamp"))
        source_exists = os.path.exists(source_image_path)

        for det in animal_dets:
            det_index = det["det_index"]
            bbox = det["bbox"]
            legacy = _find_legacy_crop(existing_crops_dir, stem, det_index)
            new_crop_path = os.path.join(crops_out_dir, f"{stem}__crop{det_index}.jpg")

            crop_path: Optional[str] = None
            if legacy is not None:
                crop_path = legacy
                stats["crops_reused"] += 1
            elif source_exists:
                try:
                    crop_path = crop_for_detection(
                        source_image_path, tuple(bbox), new_crop_path,
                        existing_crop_path=None, write=write_crops,
                    )
                    if write_crops:
                        stats["crops_written"] += 1
                    else:
                        # No file written, but the record points at the path it WOULD
                        # have; count it as written for record-completeness.
                        stats["crops_written"] += 1
                except (ValueError, OSError):
                    # Degenerate box or unreadable image: skip this detection.
                    stats["crops_missing_source"] += 1
                    crop_path = None
            else:
                # No legacy crop and the source frame is missing on disk: skip.
                stats["crops_missing_source"] += 1
                crop_path = None

            if crop_path is None:
                continue  # a record must always point at a real crop path

            bx, by, bw, bh = (float(v) for v in bbox)
            records.append(DetectionRecord(
                record_id=make_record_id(stem, det_index),
                source_image=source_image_path,
                source_stem=stem,
                det_index=det_index,
                crop_path=crop_path,
                bbox_x=bx, bbox_y=by, bbox_w=bw, bbox_h=bh,
                detector_conf=det["conf"],
                camera_id=camera_id,
                timestamp=timestamp,
                orientation="unknown",
                dataset=dataset,
            ))

    conn = connect(resolved_db)
    try:
        stats["records_upserted"] = upsert_records(conn, records)
    finally:
        conn.close()

    _print_stats(stats)
    return stats


# --------------------------------------------------------------------------- #
# Raw-image adapter (lazy MegaDetector)
# --------------------------------------------------------------------------- #

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def ingest_from_images(
    *,
    images_dir: str,
    out_md_json: Optional[str] = None,   # where to write the MegaDetector JSON; temp if None
    md_threshold: float = 0.1,           # detector output threshold (keep low, filter later)
    **ingest_kwargs,
) -> Dict[str, Any]:
    """Run MegaDetector (repo venv) over a raw image folder to produce a results JSON,
    then call :func:`ingest` on it. Requires the ``megadetector`` package; raises a
    clear ``RuntimeError`` with an install hint if it is unavailable.

    The megadetector import is performed HERE (lazily) so that importing
    ``reid_demo.ingest`` never pulls in torch / megadetector.
    """
    try:
        from megadetector.detection import run_detector_batch
        from megadetector.detection.run_detector import DEFAULT_DETECTOR_LABEL_MAP  # noqa: F401
    except Exception as exc:  # pragma: no cover - requires the venv + model
        raise RuntimeError(
            "ingest_from_images requires the 'megadetector' package, which is only "
            "installed in the repo venv. Activate it (`source venv/bin/activate`) or "
            "run via `venv/bin/python`. Original import error: " + repr(exc)
        ) from exc

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images_dir not found: {images_dir}")

    image_paths = [
        os.path.join(images_dir, f)
        for f in sorted(os.listdir(images_dir))
        if f.endswith(_IMAGE_EXTS)
    ]
    if not image_paths:
        raise FileNotFoundError(f"no images ({_IMAGE_EXTS}) under {images_dir}")

    if out_md_json is None:
        fd, out_md_json = tempfile.mkstemp(prefix="md_results_", suffix=".json")
        os.close(fd)

    # MDV5A is the default detector; download/caching is handled by megadetector.
    results = run_detector_batch(
        "MDV5A", image_paths, confidence_threshold=md_threshold, quiet=True,
    )
    # run_detector_batch returns a list of per-image result dicts with 'file' and
    # 'detections' (normalized bbox [x,y,w,h], 'category', 'conf') — the same shape
    # load_detection_frames expects under 'images'.
    md_payload = {
        "images": results,
        "detection_categories": {"1": "animal", "2": "person", "3": "vehicle"},
        "info": {"detector": "MDV5A", "detection_threshold": md_threshold},
    }
    parent = Path(out_md_json).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)
    with open(out_md_json, "w", encoding="utf-8") as fh:
        json.dump(md_payload, fh)

    ingest_kwargs.setdefault("images_dir", images_dir)
    return ingest(md_json=out_md_json, **ingest_kwargs)


# --------------------------------------------------------------------------- #
# B-track: labeled WildlifeReID-10k subset (whole-frame, GT-populated)
# --------------------------------------------------------------------------- #

def _normalize_bt_orientation(value: Any) -> str:
    """B-track orientation normalization: '' / None / NaN / out-of-set -> 'unknown'.
    Always returns a value in ORIENTATIONS."""
    if value is None:
        return "unknown"
    # pandas NaN (float that != itself) or the string 'nan'
    if isinstance(value, float) and value != value:
        return "unknown"
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return "unknown"
    if s not in ORIENTATIONS:
        return "unknown"
    return s


def _enrich_wildlife_metadata(df, subset: str):
    """If ``df`` lacks ``orientation``/``species`` (the curated all_datasets.csv does),
    enrich it from the authoritative WildlifeReID10k package metadata by joining on
    ``path`` (falling back to ``image_id``). Returns the (possibly enriched) DataFrame.

    The ``wildlife_datasets`` import is lazy and local so the module stays importable
    without it; if enrichment is impossible the columns are simply added as missing
    (orientation -> 'unknown', species -> None handled by the caller)."""
    needs_orientation = "orientation" not in df.columns
    needs_species = "species" not in df.columns
    if not (needs_orientation or needs_species):
        return df
    try:
        from wildlife_datasets.datasets import WildlifeReID10k
        import constants  # repo-root constants for WILD_DATASET_PATH
        ds = WildlifeReID10k(constants.WILD_DATASET_PATH, check_files=False)
        meta = ds.metadata
        sub = meta[meta["dataset"].astype(str).str.lower() == str(subset).lower()].copy()
        cols = [c for c in ("orientation", "species") if c in sub.columns]
        if not cols:
            return df
        join_key = "path" if ("path" in df.columns and "path" in sub.columns) else None
        if join_key is None and "image_id" in df.columns and "image_id" in sub.columns:
            join_key = "image_id"
        if join_key is None:
            return df
        right = sub[[join_key] + cols].drop_duplicates(subset=[join_key])
        # Coerce join keys to str to avoid int/str mismatches (image_id).
        df = df.copy()
        df[join_key] = df[join_key].astype(str)
        right = right.copy()
        right[join_key] = right[join_key].astype(str)
        df = df.merge(right, on=join_key, how="left", suffixes=("", "_wild"))
    except Exception:
        # Enrichment is best-effort; caller normalizes whatever is (or isn't) present.
        return df
    return df


def ingest_wildlife_dataset(
    subset: str,
    *,
    max_identities: Optional[int] = None,
    limit: Optional[int] = None,
    db_path: Optional[str] = None,   # None -> store.DEFAULT_DB_PATH
    dataset: Optional[str] = None,   # store --dataset label; defaults to `subset`
) -> Dict[str, Any]:
    """FOURTH adapter. Ingest a labeled WildlifeReID-10k subset (e.g. 'LeopardID2022',
    'ATRW') as GROUND-TRUTH re-id data, creating ONE WHOLE-FRAME ``DetectionRecord``
    per image:

      * bbox = (0.0, 0.0, 1.0, 1.0)   — the whole frame IS the crop
      * crop_path = the ORIGINAL full image path on disk (no cropping, no MegaDetector)
      * detector_conf = 1.0
      * det_index = 1, record_id = make_record_id(source_stem, 1)
      * gt_identity = metadata 'identity'
      * orientation = metadata 'orientation', with ''/missing/NaN -> 'unknown'
      * species     = metadata 'species'
      * dataset     = `dataset` or `subset`

    Loads the subset via ``utility_functions.load_dataset(subset)``; if that loader
    returns the curated metadata WITHOUT orientation/species columns, those fields are
    enriched from the authoritative WildlifeReID10k metadata (joined by path).

    ``max_identities`` caps the number of DISTINCT identities ingested (first N in
    sorted order, deterministic; used by T10 ``--smoke``). ``limit`` caps the number of
    images, applied AFTER the ``max_identities`` filter. Leaves embedding*/cluster*/
    review_* at their dataclass defaults. Imports neither torch nor megadetector.
    """
    import utility_functions  # repo-root loader; pandas-backed, no torch
    import constants

    resolved_db = db_path or store.DEFAULT_DB_PATH
    label = dataset or subset

    df = utility_functions.load_dataset(subset)
    if df is None or len(df) == 0:
        raise RuntimeError(f"load_dataset({subset!r}) returned no rows.")
    if "identity" not in df.columns or "path" not in df.columns:
        raise RuntimeError(
            f"load_dataset({subset!r}) is missing required columns 'identity'/'path'; "
            f"got {list(df.columns)}."
        )

    df = _enrich_wildlife_metadata(df, subset)

    identities_total = int(df["identity"].nunique())
    images_total = int(len(df))

    # max_identities: select the first N distinct identities in deterministic order.
    if max_identities is not None:
        keep_ids = sorted(df["identity"].dropna().astype(str).unique())[:max_identities]
        keep_set = set(keep_ids)
        df = df[df["identity"].astype(str).isin(keep_set)]
    identities_ingested = int(df["identity"].nunique())

    # Deterministic image order so `limit` is reproducible.
    sort_cols = [c for c in ("identity", "path") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="stable")

    if limit is not None:
        df = df.head(limit)

    image_root = constants.WILD_DATASET_PATH
    has_orientation = "orientation" in df.columns
    has_species = "species" in df.columns

    records: List[DetectionRecord] = []
    images_ingested = 0
    for _, row in df.iterrows():
        rel_path = row["path"]
        if rel_path is None or (isinstance(rel_path, float) and rel_path != rel_path):
            continue
        abs_path = os.path.join(image_root, str(rel_path))
        stem = Path(str(rel_path)).stem

        orientation = (
            _normalize_bt_orientation(row["orientation"]) if has_orientation else "unknown"
        )
        species = None
        if has_species:
            sp = row["species"]
            if sp is not None and not (isinstance(sp, float) and sp != sp):
                sp_str = str(sp).strip()
                species = sp_str if sp_str not in ("", "nan") else None

        gt_identity = row["identity"]
        gt_identity = None if gt_identity is None else str(gt_identity)

        records.append(DetectionRecord(
            record_id=make_record_id(stem, 1),
            source_image=abs_path,
            source_stem=stem,
            det_index=1,
            crop_path=abs_path,           # whole frame IS the crop
            bbox_x=0.0, bbox_y=0.0, bbox_w=1.0, bbox_h=1.0,
            detector_conf=1.0,
            orientation=orientation,
            gt_identity=gt_identity,
            species=species,
            dataset=label,
        ))
        images_ingested += 1

    conn = connect(resolved_db)
    try:
        records_upserted = upsert_records(conn, records)
    finally:
        conn.close()

    stats = {
        "images_total": images_total,
        "images_ingested": images_ingested,
        "identities_total": identities_total,
        "identities_ingested": identities_ingested,
        "records_upserted": records_upserted,
        "subset": subset,
        "dataset": label,
        "db_path": resolved_db,
    }
    _print_bt_stats(stats)
    return stats


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def _print_stats(stats: Dict[str, Any]) -> None:
    ft = stats["frames_total"] or 1
    pct_empty = stats["frames_empty"] / ft
    print("=== T02 ingest (A-track) ===")
    print(f"  dataset:              {stats['dataset']}")
    print(f"  db_path:              {stats['db_path']}")
    print(f"  frames_total:         {stats['frames_total']}")
    print(f"  frames_with_animals:  {stats['frames_with_animals']}")
    print(f"  frames_empty:         {stats['frames_empty']}  (pct_empty={pct_empty:.3f})")
    print(f"  dets_total:           {stats['dets_total']}")
    print(f"  dets_animal:          {stats['dets_animal']}")
    print(f"  dets_person:          {stats['dets_person']}")
    print(f"  dets_vehicle:         {stats['dets_vehicle']}")
    print(f"  dets_below_threshold: {stats['dets_below_threshold']}")
    print(f"  crops_written:        {stats['crops_written']}")
    print(f"  crops_reused:         {stats['crops_reused']}")
    print(f"  crops_missing_source: {stats['crops_missing_source']}")
    print(f"  records_upserted:     {stats['records_upserted']}")


def _print_bt_stats(stats: Dict[str, Any]) -> None:
    print("=== T02 ingest (B-track: labeled WildlifeReID-10k) ===")
    print(f"  subset:               {stats['subset']}")
    print(f"  dataset:              {stats['dataset']}")
    print(f"  db_path:              {stats['db_path']}")
    print(f"  images_total:         {stats['images_total']}")
    print(f"  images_ingested:      {stats['images_ingested']}")
    print(f"  identities_total:     {stats['identities_total']}")
    print(f"  identities_ingested:  {stats['identities_ingested']}")
    print(f"  records_upserted:     {stats['records_upserted']}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reid_demo.ingest",
        description="T02 ingestion + MegaDetector adapter for the lynx re-ID demo.",
    )
    parser.add_argument("--md-json", default=None,
                        help=f"MegaDetector results JSON or flat animal_detections.json "
                             f"(default {DEFAULT_MD_JSON} when no other source given).")
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR,
                        help="source frames dir (basename resolution). Without --md-json "
                             "this triggers the raw-image MegaDetector path.")
    parser.add_argument("--wildlife-subset", default=None,
                        help="labeled WildlifeReID-10k subset (e.g. LeopardID2022, ATRW) "
                             "-> B-track ingest_wildlife_dataset.")
    parser.add_argument("--metadata-csv", default=DEFAULT_METADATA_CSV,
                        help="trail_cam_data.csv for camera/timestamp resolution.")
    parser.add_argument("--existing-crops", default=DEFAULT_EXISTING_CROPS,
                        help="dir of pre-existing crops to reuse.")
    parser.add_argument("--crops-out", default=DEFAULT_CROPS_OUT,
                        help="output dir for newly written crops.")
    parser.add_argument("--db", default=None,
                        help=f"store DB path (default {store.DEFAULT_DB_PATH}).")
    parser.add_argument("--dataset", default=None,
                        help=f"--dataset label (A-track default {DEFAULT_DATASET}; "
                             "B-track defaults to the subset name).")
    parser.add_argument("--conf-threshold", type=float, default=DEFAULT_CONF_THRESHOLD,
                        help="per-detection animal confidence cutoff (>=).")
    parser.add_argument("--no-crop", action="store_true",
                        help="record crop paths without writing JPGs (dry run).")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap on number of source frames / images (debug).")
    parser.add_argument("--max-identities", type=int, default=None,
                        help="B-track: cap on distinct identities ingested.")
    parser.add_argument("--report-json", default=None,
                        help="also write the stats dict to this JSON path.")
    return parser


def _main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        # B-track: labeled WildlifeReID-10k subset.
        if args.wildlife_subset:
            stats = ingest_wildlife_dataset(
                args.wildlife_subset,
                max_identities=args.max_identities,
                limit=args.limit,
                db_path=args.db,
                dataset=args.dataset,   # defaults to subset inside the adapter
            )
        # Raw-image path: --images-dir given but no --md-json.
        elif args.md_json is None and args.images_dir and args.images_dir != DEFAULT_IMAGES_DIR:
            stats = ingest_from_images(
                images_dir=args.images_dir,
                metadata_csv=args.metadata_csv,
                existing_crops_dir=args.existing_crops,
                crops_out_dir=args.crops_out,
                db_path=args.db,
                dataset=args.dataset or DEFAULT_DATASET,
                conf_threshold=args.conf_threshold,
                write_crops=not args.no_crop,
                limit=args.limit,
            )
        # A-track JSON path (default).
        else:
            stats = ingest(
                md_json=args.md_json or DEFAULT_MD_JSON,
                images_dir=args.images_dir,
                metadata_csv=args.metadata_csv,
                existing_crops_dir=args.existing_crops,
                crops_out_dir=args.crops_out,
                db_path=args.db,
                dataset=args.dataset or DEFAULT_DATASET,
                conf_threshold=args.conf_threshold,
                write_crops=not args.no_crop,
                limit=args.limit,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[ingest] FATAL: {exc}", file=sys.stderr)
        return 1

    if args.report_json:
        parent = Path(args.report_json).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)
        print(f"[ingest] wrote stats -> {args.report_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
