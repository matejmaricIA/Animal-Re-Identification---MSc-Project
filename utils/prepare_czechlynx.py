#!/usr/bin/env python3
"""Download/audit CzechLynx real v2 metadata and prepare frame or annotated crops."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import stat
import sys
import zipfile
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from reid_demo.benchmark import file_digest, fingerprint, write_json
from reid_demo.benchmark_masks import MASKED_PREPROCESSING

RECORD_URL = "https://zenodo.org/api/records/17592004"
REAL_METADATA = "CzechLynxDataset-Metadata-Real.csv"
SPLITS = ("split-geo_aware", "split-time_open", "split-time_closed")
SOURCE_VERSION = "zenodo-17592004-real-v2-schema"


def download_real(root, *, extract=True):
    """Only real photographs/metadata; resume transfers and verify publisher MD5."""
    import requests
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    response = requests.get(RECORD_URL, timeout=60)
    response.raise_for_status()
    manifest = response.json()
    write_json(root / "source_record.json", manifest)
    entries = {f["key"]: f for f in manifest["files"]}
    for name in (REAL_METADATA, "CzechLynx.zip"):
        entry = entries[name]
        target = root / name
        algorithm, expected = entry["checksum"].split(":", 1)
        if target.exists():
            if target.stat().st_size != entry["size"] or file_digest(target, algorithm) != expected:
                raise ValueError(f"Existing download has wrong checksum: {target}")
            continue
        if shutil.disk_usage(root).free < entry["size"] * 2:
            raise RuntimeError("Insufficient space for archive and extraction.")
        partial = target.with_name(target.name + ".part")
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with requests.get(entry["links"]["self"], headers=headers, stream=True,
                          timeout=(30, 180)) as stream:
            stream.raise_for_status()
            if offset and stream.status_code == 206:
                if not stream.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Server returned an unexpected download range.")
            elif offset:
                offset = 0
            with partial.open("ab" if offset else "wb") as out:
                for block in stream.iter_content(4 * 1024 * 1024):
                    out.write(block)
        if partial.stat().st_size != entry["size"] or file_digest(partial, algorithm) != expected:
            raise ValueError(f"Download verification failed: {partial}")
        os.replace(partial, target)
    if extract:
        extract_archive(root / "CzechLynx.zip", root)
    return root / REAL_METADATA


def extract_archive(archive, destination):
    destination = Path(destination).resolve()
    digest = file_digest(archive)
    receipt = destination / "extraction.json"
    if receipt.exists() and json.loads(receipt.read_text()).get("archive_sha256") == digest:
        return
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        if shutil.disk_usage(destination).free < sum(m.file_size for m in members):
            raise RuntimeError("Insufficient extraction space.")
        for member in members:
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination) or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Unsafe archive member: {member.filename}")
        for member in members:
            z.extract(member, destination)  # ZipFile verifies CRC during extraction.
    write_json(receipt, {"archive_sha256": digest, "members": len(members)})


def canonical_metadata(frame):
    required = {"source", "path", "unique_name", "date", "encounter", *SPLITS}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing metadata columns: {sorted(required - set(frame.columns))}")
    df = frame.copy()
    if df[["source", "path", "unique_name", "encounter"]].isna().any().any():
        raise ValueError("Source, path, identity and encounter must be present.")
    df["source_path"] = df.path.astype(str).str.replace("\\", "/", regex=False)
    for path in df.source_path:
        p = Path(path)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"Unsafe metadata path: {path}")
    if df.source_path.duplicated().any():
        raise ValueError("Duplicate target paths in metadata.")
    df["identity"] = df.unique_name.astype(str)
    df["image_id"] = df.source_path.map(
        lambda p: "clx_" + fingerprint({"version": SOURCE_VERSION, "path": p})[:24])
    if df.image_id.duplicated().any():
        raise ValueError("Image ID collision.")
    # Shared encounters must stay shared even when several individuals were present.
    df["encounter_id"] = df.source.astype(str) + ":" + df.encounter.astype(str)
    df["timestamp"] = pd.to_datetime(df.date, format="%d-%m-%Y", errors="coerce")
    invalid = df.date.notna() & df.timestamp.isna()
    if invalid.any():
        raise ValueError(f"Unparseable dates: {df.loc[invalid, 'date'].head().tolist()}")
    df["timestamp"] = df.timestamp.dt.strftime("%Y-%m-%d")
    for column in SPLITS:
        if not df[column].isin(["train", "test"]).all():
            raise ValueError(f"Missing or unsupported split values in {column}.")
    df["orientation"] = "unknown"  # No validated flank field in this schema.
    df["dataset"] = "CzechLynx"
    return df.sort_values("image_id").reset_index(drop=True)


def mask_rle(value, width, height):
    from pycocotools import mask as mask_utils
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = ast.literal_eval(value)
    if not isinstance(value, dict) or "counts" not in value or "size" not in value:
        raise ValueError("Mask must be a COCO RLE dictionary.")
    if list(value["size"]) != [height, width]:
        raise ValueError("Mask dimensions do not match the image.")
    rle = dict(value)
    if isinstance(rle["counts"], list):
        if any(not isinstance(n, int) or n < 0 for n in rle["counts"]) or sum(rle["counts"]) != width * height:
            raise ValueError("Invalid uncompressed mask run lengths.")
        rle = mask_utils.frPyObjects(rle, height, width)
    elif isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("ascii")
    return rle


def mask_bbox(value, width, height):
    from pycocotools import mask as mask_utils
    x, y, w, h = [int(v) for v in mask_utils.toBbox(mask_rle(value, width, height))]
    if min(w, h) <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError("Invalid or empty mask bounding box.")
    return (x, y, x + w, y + h)


def decode_mask(value, width, height):
    from pycocotools import mask as mask_utils
    mask = mask_utils.decode(mask_rle(value, width, height))
    if mask.shape != (height, width) or not np.any(mask):
        raise ValueError("Invalid or empty decoded animal mask.")
    return mask != 0


def save_masked_crop(image, mask, box, source_sha, image_id, out_dir):
    """Lossless foreground pixels and an aligned mask, with resumable receipts."""
    x0, y0, x1, y1 = box
    cropped_mask = mask[y0:y1, x0:x1]
    spec = {"source_sha256": source_sha, "bbox": list(box), "preprocessing": MASKED_PREPROCESSING,
            "mask_pixels_sha256": hashlib.sha256(cropped_mask.tobytes()).hexdigest()}
    key = fingerprint(spec)
    root = out_dir / "masked_crops"
    root.mkdir(exist_ok=True)
    stem = f"{image_id}_{key[:16]}"
    image_path, mask_path = root / f"{stem}.png", root / f"{stem}_mask.png"
    receipt_path = root / f"{stem}.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("spec") != spec or file_digest(image_path) != receipt["image_sha256"] or
                file_digest(mask_path) != receipt["mask_sha256"]):
            raise ValueError(f"Damaged masked preparation cache: {image_id}")
    else:
        pixels = np.asarray(image.crop(box)).copy()
        pixels[~cropped_mask] = MASKED_PREPROCESSING["background_rgb"]
        for path, pixels_to_save in ((image_path, pixels), (mask_path, cropped_mask.astype(np.uint8) * 255)):
            temp = path.with_name(path.name + ".tmp")
            Image.fromarray(pixels_to_save).save(temp, format="PNG")
            os.replace(temp, path)
        receipt = {"spec": spec, "image_sha256": file_digest(image_path), "mask_sha256": file_digest(mask_path)}
        write_json(receipt_path, receipt)
    return {"path": str(image_path.resolve()), "image_sha256": receipt["image_sha256"],
            "mask_path": str(mask_path.resolve()), "mask_sha256": receipt["mask_sha256"]}


def prepare(metadata, images_root, out_dir, *, modes=("frame", "bbox"), on_error="error"):
    metadata, images_root, out_dir = Path(metadata), Path(images_root).resolve(), Path(out_dir)
    df = canonical_metadata(pd.read_csv(metadata))
    out_dir.mkdir(parents=True, exist_ok=True)
    if not modes or set(modes) - {"frame", "bbox", "masked"}:
        raise ValueError("Choose preparation modes from frame, bbox and masked.")
    audits = {
        "schema": 1, "source_version": SOURCE_VERSION,
        "metadata_sha256": file_digest(metadata), "original_images": len(df),
        "original_identities": int(df.identity.nunique()),
        "missing_dates": int(df.timestamp.isna().sum()),
        "official_encounter_overlap": {},
        "annotation_assisted_modes": [m for m in modes if m in {"bbox", "masked"}],
        "errors": [], "on_error": on_error,
    }
    for split in SPLITS:
        groups = df.groupby("encounter_id")[split].nunique()
        events = groups[groups > 1].index.tolist()
        audits["official_encounter_overlap"][split] = {
            "encounters": len(events), "images": int(df.encounter_id.isin(events).sum()),
            "encounter_ids": events,
        }
    outputs = {mode: [] for mode in modes}
    for index, row in enumerate(df.to_dict("records")):
        source = (images_root / row["source_path"]).resolve()
        try:
            if not source.is_relative_to(images_root):
                raise ValueError("Image path escaped the dataset root.")
            source_sha = file_digest(source)
            with Image.open(source) as original:
                original.load()
                image = original.convert("RGB")
                width, height = image.size
            box = mask_bbox(row.get("mask"), width, height) if set(modes) & {"bbox", "masked"} else None
            animal_mask = decode_mask(row.get("mask"), width, height) if "masked" in modes else None
            base = {k: row.get(k) for k in (
                "image_id", "identity", "source", "source_path", "encounter_id",
                "timestamp", "trap_id", "location", "orientation", "dataset", *SPLITS)}
            base.update(source_sha256=source_sha, width=width, height=height)
            prepared_rows = {}
            for mode in modes:
                mask_fields = {}
                if mode == "frame":
                    image_path, digest = source, source_sha
                elif mode == "bbox":
                    crop_key = fingerprint({"source": source_sha, "box": box, "jpeg_quality": 95})
                    image_path = out_dir / "crops" / f"{row['image_id']}_{crop_key[:16]}.jpg"
                    image_path.parent.mkdir(exist_ok=True)
                    if not image_path.exists():
                        temp = image_path.with_name(image_path.name + ".tmp")
                        image.crop(box).save(temp, format="JPEG", quality=95)
                        os.replace(temp, image_path)
                    digest = file_digest(image_path)
                elif mode == "masked":
                    mask_fields = save_masked_crop(image, animal_mask, box, source_sha, row["image_id"], out_dir)
                    image_path, digest = Path(mask_fields["path"]), mask_fields["image_sha256"]
                else:
                    raise ValueError(f"Unsupported image mode: {mode}")
                prepared_rows[mode] = {
                    **base, "path": str(image_path.resolve()), "image_sha256": digest,
                    "input_mode": mode, "bbox": json.dumps(box) if box else "",
                    **mask_fields,
                }
            for mode in modes:
                outputs[mode].append(prepared_rows[mode])
        except (OSError, ValueError, TypeError, SyntaxError) as exc:
            audits["errors"].append({"image_id": row["image_id"], "path": str(source),
                                      "error": str(exc)})
        if (index + 1) % 2000 == 0:
            print(f"Prepared {index + 1}/{len(df)} images", flush=True)
    audits["retained_images"] = len(outputs[modes[0]])
    audits["excluded_images"] = len(audits["errors"])
    write_json(out_dir / "data_audit.json", audits)
    if audits["errors"] and on_error == "error":
        raise ValueError(f"Preparation found {len(audits['errors'])} invalid images; see data_audit.json.")
    for mode, rows in outputs.items():
        dest = out_dir / mode
        dest.mkdir(exist_ok=True)
        table = pd.DataFrame(rows)
        if table.empty:
            raise ValueError("No images remain after preparation.")
        temp = dest / "metadata.csv.tmp"
        table.to_csv(temp, index=False)
        os.replace(temp, dest / "metadata.csv")
        write_json(dest / "manifest.json", {
            "schema": 1, "source_version": SOURCE_VERSION,
            "original_metadata_sha256": audits["metadata_sha256"],
            "prepared_metadata_sha256": file_digest(dest / "metadata.csv"),
            "input_mode": mode, "annotation_assisted": mode in {"bbox", "masked"},
            **({"masked_preprocessing": MASKED_PREPROCESSING} if mode == "masked" else {}),
            "images": len(table), "identities": int(table.identity.nunique()),
            "excluded_ids": [e["image_id"] for e in audits["errors"]],
        })
    return audits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--raw-dir", default="data/czechlynx/raw")
    parser.add_argument("--metadata")
    parser.add_argument("--images-root")
    parser.add_argument("--out-dir", default="data/czechlynx/prepared")
    parser.add_argument("--modes", nargs="+", choices=["frame", "bbox", "masked"], default=["frame", "bbox"])
    parser.add_argument("--on-error", choices=["error", "exclude"], default="error")
    args = parser.parse_args(argv)
    raw = Path(args.raw_dir)
    if args.download:
        download_real(raw, extract=not args.download_only)
    if args.download_only:
        return 0
    if args.extract_only:
        extract_archive(raw / "CzechLynx.zip", raw)
        return 0
    prepare(args.metadata or raw / REAL_METADATA, args.images_root or raw,
            args.out_dir, modes=tuple(dict.fromkeys(args.modes)), on_error=args.on_error)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
