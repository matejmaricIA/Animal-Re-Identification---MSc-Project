"""Model-free, immutable-input CzechLynx audit and human-review receipts."""
from __future__ import annotations

from collections import Counter
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from reid_demo.benchmark import file_digest, fingerprint, write_json
from reid_demo.benchmark_clustering import check_output_directory
from reid_demo.benchmark_masks import MASKED_PREPROCESSING
from reid_demo.benchmark_protocol import load_protocol

ROOT = Path(__file__).resolve().parents[1]
VERSION = "czechlynx_input_audit_v1"
DIAGNOSTIC_IDS = (
    "clx_0f13df5c525a6707d7f0a5bd", "clx_1ba0da5ae8a7c09f48dde743",
    "clx_7a72b79ebb863380cde8383a", "clx_a3be3c1935cd73479ff499e6",
    "clx_127cd5c697122707fb92d864", "clx_e981b521482a803cb7855493",
    "clx_bd14c3811f3b4d3df1f431d5", "clx_20a081e51a7f25b6d64bf4de",
    "clx_2a87c65d2467873d5257da36", "clx_3867314d7ed72edbc48c7027",
    "clx_481be9e63ffb085867ffd013",
)
REVIEW_CHOICES = {
    "animal_visibility": ("clear", "partial", "no_clear_animal", "unknown"),
    "usable_markings": ("yes", "no", "unknown"),
    "crop_mask": ("correct", "issue", "unknown"),
    "blur_occlusion": ("minor", "moderate", "severe", "unknown"),
    "flank": ("left", "right", "unknown"),
    "illumination": ("colour", "monochrome", "verified_day", "verified_night", "verified_ir", "unknown"),
}
REVIEW_COLUMNS = ("image_id", *REVIEW_CHOICES, "illumination_evidence", "reviewer", "reviewed_at", "notes")


def implementation():
    names = ("reid_demo/benchmark_input_audit.py", "tools/audit_czechlynx_inputs.py",
             "utils/prepare_czechlynx.py", "reid_demo/benchmark_protocol.py")
    return {name: file_digest(ROOT / name) for name in names}


def load_inputs(metadata, protocol_plan):
    metadata, protocol_plan = Path(metadata).resolve(), Path(protocol_plan).resolve()
    frame = pd.read_csv(metadata, dtype={"image_id": str, "identity": str})
    required = {"image_id", "identity", "path", "image_sha256", "mask_path", "mask_sha256",
                "source_path", "source_sha256", "bbox", "width", "height", "input_mode"}
    if not required.issubset(frame) or frame.empty or frame.image_id.duplicated().any():
        raise ValueError("Use complete, unique prepared masked CzechLynx metadata.")
    if frame[list(required)].isna().any().any() or set(frame.input_mode) != {"masked"}:
        raise ValueError("Audit/adaptation requires nonmissing masked input provenance.")
    prepared_path = metadata.with_name("manifest.json")
    prepared = json.loads(prepared_path.read_text())
    digest = file_digest(metadata)
    if (prepared.get("prepared_metadata_sha256") != digest
            or prepared.get("masked_preprocessing") != MASKED_PREPROCESSING):
        raise ValueError("Prepared manifest/metadata/masking policy differs.")
    plan = load_protocol(protocol_plan, frame, digest)
    binding = {"metadata": str(metadata), "metadata_sha256": digest,
               "prepared_manifest_sha256": file_digest(prepared_path),
               "protocol_plan": str(protocol_plan), "protocol_plan_sha256": file_digest(protocol_plan),
               "protocol_fingerprint": plan["fingerprint"]}
    return frame.sort_values("image_id").reset_index(drop=True), plan, binding


def verify_prepared(frame):
    """Read actual bytes, not presence flags, before training or extraction."""
    for row in frame.to_dict("records"):
        for path_key, hash_key in (("path", "image_sha256"), ("mask_path", "mask_sha256")):
            if file_digest(row[path_key]) != row[hash_key]:
                raise ValueError(f"Changed {path_key}: {row['image_id']}")


def protected_inputs(frame, binding):
    paths = [binding["metadata"], Path(binding["metadata"]).with_name("manifest.json"),
             binding["protocol_plan"]]
    paths.extend(sorted({str(Path(p).resolve().parent) for p in frame.path}))
    paths.extend(sorted({str(Path(p).resolve().parent) for p in frame.mask_path}))
    return paths


def choose_sample(frame, plan, *, sample_size=128, seed=42, diagnostic_ids=DIAGNOSTIC_IDS):
    if sample_size < 1 or seed < 0:
        raise ValueError("Sample size must be positive and seed nonnegative.")
    rng = np.random.default_rng(seed)
    members = {}
    def add(rid, cohort):
        members.setdefault(str(rid), []).append(cohort)
    indexed = frame.set_index("image_id", drop=False)
    for pool in ("fitting", "calibration"):
        entry = plan["pools"][pool]
        ids = (entry["image_ids"] if pool == "fitting" else
               sorted(entry["gallery_ids"] + entry["query_ids"]))
        for rid in rng.choice(sorted(ids), size=min(sample_size, len(ids)), replace=False):
            add(rid, pool + "_uniform")
        if pool == "calibration":
            for _, rows in indexed.loc[ids].groupby("identity", sort=True):
                add(rng.choice(sorted(rows.image_id)), "calibration_identity_coverage")
            for rid in diagnostic_ids:
                if rid in indexed.index:
                    if rid not in ids:
                        raise ValueError("Diagnostic example is outside eligible calibration.")
                    add(rid, "calibration_diagnostic_extremes")
    return [{"image_id": rid, "cohorts": sorted(cohorts)} for rid, cohorts in sorted(members.items())]


def inspect_annotation(row, raw_row, images_root):
    from utils.prepare_czechlynx import decode_mask, mask_bbox
    images_root = Path(images_root).resolve()
    source = (images_root / row["source_path"]).resolve()
    if not source.is_relative_to(images_root):
        raise ValueError("Source image escapes the original image root.")
    if file_digest(source) != row["source_sha256"]:
        raise ValueError(f"Changed source image: {row['image_id']}")
    if str(raw_row["unique_name"]) != row["identity"]:
        raise ValueError("Original and prepared identity labels differ.")
    verify_prepared(pd.DataFrame([row]))
    with Image.open(source) as im:
        rgb = np.asarray(im.convert("RGB"))
    height, width = rgb.shape[:2]
    if (width, height) != (int(row["width"]), int(row["height"])):
        raise ValueError("Source dimensions differ from prepared metadata.")
    box = mask_bbox(raw_row["mask"], width, height)
    if list(box) != json.loads(row["bbox"]):
        raise ValueError("Prepared bounding box differs from source annotation.")
    mask = decode_mask(raw_row["mask"], width, height)
    x0, y0, x1, y1 = box
    foreground = mask[y0:y1, x0:x1]
    expected = rgb[y0:y1, x0:x1].copy()
    expected[~foreground] = MASKED_PREPROCESSING["background_rgb"]
    with Image.open(row["mask_path"]) as im:
        actual_mask = np.asarray(im)
    with Image.open(row["path"]) as im:
        actual_rgb = np.asarray(im.convert("RGB"))
    if (not np.array_equal(actual_mask, foreground.astype(np.uint8) * 255)
            or not np.array_equal(actual_rgb, expected)):
        raise ValueError("Prepared pixels/mask do not match supplied source annotation.")
    return {"source_path": str(source), "source_sha256": row["source_sha256"],
            "image_sha256": row["image_sha256"], "mask_sha256": row["mask_sha256"],
            "bbox": list(box), "foreground_fraction": float(foreground.mean()),
            "annotation_alignment": "exact"}, rgb, mask


def _thumbnail(pixels, destination):
    im = Image.fromarray(pixels)
    im.thumbnail((700, 450))
    im.save(destination, quality=90)


def prepare_audit(metadata, protocol_plan, raw_metadata, images_root, out_dir, *, sample_size=128, seed=42):
    frame, plan, binding = load_inputs(metadata, protocol_plan)
    raw_metadata = Path(raw_metadata).resolve()
    binding.update(raw_metadata=str(raw_metadata), raw_metadata_sha256=file_digest(raw_metadata),
                   images_root=str(Path(images_root).resolve()))
    protected = protected_inputs(frame, binding) + [raw_metadata, images_root]
    check_output_directory(out_dir, protected)
    raw = pd.read_csv(raw_metadata)
    raw["source_path"] = raw.path.astype(str).str.replace("\\", "/", regex=False)
    if raw.source_path.duplicated().any():
        raise ValueError("Original source paths are not unique.")
    raw = raw.set_index("source_path")
    sample = choose_sample(frame, plan, sample_size=sample_size, seed=seed)
    indexed = frame.set_index("image_id")
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir()
    manifest = {"kind": VERSION, "status": "running", "binding": binding,
                "settings": {"sample_size": sample_size, "seed": seed}, "implementation": implementation(),
                "sample": sample, "checks": []}
    write_json(out / "manifest.json", manifest)
    try:
        articles = []
        active_id = None
        for sample_row in sample:
            rid = sample_row["image_id"]
            active_id = rid
            row = indexed.loc[rid].to_dict()
            row["image_id"] = rid
            info, rgb, mask = inspect_annotation(row, raw.loc[row["source_path"]].to_dict(), images_root)
            manifest["checks"].append({"image_id": rid, **info})
            _thumbnail(rgb, out / "images" / f"{rid}_original.jpg")
            with Image.open(row["path"]) as im:
                _thumbnail(np.asarray(im.convert("RGB")), out / "images" / f"{rid}_crop.jpg")
            overlay = rgb.copy()
            overlay[mask] = (overlay[mask].astype(float) * .65 + np.array([0, 255, 80]) * .35).astype(np.uint8)
            marked = Image.fromarray(overlay)
            ImageDraw.Draw(marked).rectangle(info["bbox"], outline="red", width=4)
            _thumbnail(np.asarray(marked), out / "images" / f"{rid}_mask.jpg")
            tiles = "".join(f'<figure><img loading="lazy" src="images/{rid}_{suffix}.jpg"><figcaption>{label}</figcaption></figure>'
                            for suffix, label in (("original", "Original"), ("crop", "Prepared"), ("mask", "Supplied mask")))
            articles.append(f'<article><h2>{rid}</h2><p>{html.escape(", ".join(sample_row["cohorts"]))}; '
                            f'foreground {info["foreground_fraction"]:.2%}</p><div>{tiles}</div></article>')
        reviews = [{"image_id": r["image_id"], **{key: "unknown" for key in REVIEW_CHOICES},
                    "illumination_evidence": "", "reviewer": "", "reviewed_at": "", "notes": ""} for r in sample]
        pd.DataFrame(reviews, columns=REVIEW_COLUMNS).to_csv(out / "reviews_template.csv", index=False)
        write_json(out / "review_schema.json", {"choices": REVIEW_CHOICES, "required": REVIEW_COLUMNS,
            "notes": "Unknown is valid. Verified illumination requires independent evidence; monochrome is insufficient."})
        body = '<h1>CzechLynx input audit</h1><p>Enter reviews in a copy of reviews_template.csv. '
        body += 'Identity truth is omitted from this inspection page. This audit cannot verify identity labels or count animals.</p>'
        body += '<p>Verified day/night/IR needs supporting source information. Keep uncertain flank/visibility unknown.</p>'
        page = '<!doctype html><meta charset="utf-8"><title>CzechLynx input audit</title><style>'
        page += 'body{font:16px system-ui; margin:30px}article{border-top:1px solid #aaa;padding:15px 0}'
        page += 'article div{display:flex;flex-wrap:wrap}figure{margin:8px}img{max-width:430px;max-height:300px}</style>'
        (out / "index.html").write_text(page + body + "".join(articles))
        manifest["artifacts"] = {str(p.relative_to(out)): file_digest(p) for p in sorted(out.rglob("*"))
                                 if p.is_file() and p.name != "manifest.json"}
        manifest["status"] = "complete"
        manifest["fingerprint"] = fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"})
        write_json(out / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="failed", failed_image_id=active_id, error=f"{type(exc).__name__}: {exc}")
        write_json(out / "manifest.json", manifest)
        raise
    return manifest


def load_audit(audit_dir, binding):
    root = Path(audit_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("kind") != VERSION or manifest.get("status") != "complete"
            or manifest.get("fingerprint") != fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"})
            or manifest.get("implementation") != implementation()
            or any(manifest.get("binding", {}).get(k) != v for k, v in binding.items())):
        raise ValueError("Incomplete, stale or incompatible input audit.")
    frame, plan, reconstructed_binding = load_inputs(binding["metadata"], binding["protocol_plan"])
    if reconstructed_binding != binding or manifest["sample"] != choose_sample(frame, plan, **manifest["settings"]):
        raise ValueError("Audit sample differs from deterministic role-bound selection.")
    if {r["image_id"] for r in manifest["checks"]} != {r["image_id"] for r in manifest["sample"]}:
        raise ValueError("Audit alignment checks do not cover the sample.")
    for name, digest in manifest["artifacts"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or file_digest(path) != digest:
            raise ValueError("Changed audit artifact.")
    for key in ("raw_metadata",):
        if file_digest(manifest["binding"][key]) != manifest["binding"][key + "_sha256"]:
            raise ValueError("Changed original annotation metadata.")
    for row in manifest["checks"]:
        if file_digest(row["source_path"]) != row["source_sha256"]:
            raise ValueError("Changed audited source photograph.")
    return manifest


def validate_reviews(reviews, sample):
    if tuple(reviews.columns) != REVIEW_COLUMNS:
        raise ValueError("Review CSV columns differ from the exported schema.")
    rows = reviews.fillna("").to_dict("records")
    if (len(rows) != len(sample) or reviews.image_id.duplicated().any()
            or set(reviews.image_id) != {r["image_id"] for r in sample}):
        raise ValueError("Reviews must cover the exact audit sample once.")
    for row in rows:
        if any(row[key] not in choices for key, choices in REVIEW_CHOICES.items()):
            raise ValueError("Unsupported review label; preserve unknowns explicitly.")
        date = pd.to_datetime(row["reviewed_at"], errors="coerce", utc=True)
        if not str(row["reviewer"]).strip() or pd.isna(date):
            raise ValueError("Every review needs a reviewer and valid review timestamp.")
        if row["illumination"].startswith("verified_") and not str(row["illumination_evidence"]).strip():
            raise ValueError("Verified illumination requires supporting evidence.")
    return sorted(rows, key=lambda r: r["image_id"])


def summarize_reviews(audit_dir, reviews_path, out_dir):
    audit_dir, reviews_path = Path(audit_dir).resolve(), Path(reviews_path).resolve()
    raw_manifest = json.loads((audit_dir / "manifest.json").read_text())
    frame, _, binding = load_inputs(raw_manifest["binding"]["metadata"], raw_manifest["binding"]["protocol_plan"])
    manifest = load_audit(audit_dir, binding)
    verify_prepared(frame[frame.image_id.isin([r["image_id"] for r in manifest["sample"]])])
    rows = validate_reviews(pd.read_csv(reviews_path, dtype=str), manifest["sample"])
    check_output_directory(out_dir, protected_inputs(frame, binding) + [audit_dir, reviews_path,
                                                                      manifest["binding"]["images_root"]])
    indexed = {r["image_id"]: r for r in rows}
    cohorts = sorted({c for r in manifest["sample"] for c in r["cohorts"]})
    summaries = {}
    for cohort in cohorts:
        selected = [indexed[r["image_id"]] for r in manifest["sample"] if cohort in r["cohorts"]]
        summaries[cohort] = {"images": len(selected), **{key: dict(Counter(r[key] for r in selected))
                                                       for key in REVIEW_CHOICES}}
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "reviews.json", rows)
    receipt = {"kind": VERSION + "_reviews", "status": "complete", "binding": binding,
               "audit_dir": str(audit_dir), "audit_manifest_sha256": file_digest(audit_dir / "manifest.json"),
               "reviews_source": str(reviews_path), "reviews_source_sha256": file_digest(reviews_path),
               "sample": manifest["sample"], "cohort_summaries": summaries,
               "artifacts": {"reviews.json": file_digest(out / "reviews.json")},
               "policy": "No label/membership changes or inferred quality rejection. Unknown labels retained."}
    receipt["fingerprint"] = fingerprint(receipt)
    write_json(out / "receipt.json", receipt)
    return receipt


def validate_review_receipt(path, frame, binding):
    path = Path(path).resolve()
    receipt = json.loads(path.read_text())
    if (receipt.get("kind") != VERSION + "_reviews" or receipt.get("status") != "complete"
            or receipt.get("binding") != binding
            or receipt.get("fingerprint") != fingerprint({k: v for k, v in receipt.items() if k != "fingerprint"})):
        raise ValueError("Invalid/stale audit review receipt.")
    manifest = load_audit(receipt["audit_dir"], binding)
    if (file_digest(Path(receipt["audit_dir"]) / "manifest.json") != receipt["audit_manifest_sha256"]
            or receipt["sample"] != manifest["sample"]):
        raise ValueError("Audit receipt sample/source changed.")
    for name, digest in receipt["artifacts"].items():
        target = (path.parent / name).resolve()
        if not target.is_relative_to(path.parent) or file_digest(target) != digest:
            raise ValueError("Changed imported review artifact.")
    if file_digest(receipt["reviews_source"]) != receipt["reviews_source_sha256"]:
        raise ValueError("Original review CSV changed; regenerate its receipt.")
    rows = json.loads((path.parent / "reviews.json").read_text())
    imported = validate_reviews(pd.DataFrame(rows, columns=REVIEW_COLUMNS), manifest["sample"])
    original = validate_reviews(pd.read_csv(receipt["reviews_source"], dtype=str), manifest["sample"])
    if imported != original:
        raise ValueError("Imported labels differ from the source review CSV.")
    verify_prepared(frame[frame.image_id.isin([r["image_id"] for r in manifest["sample"]])])
    return receipt
