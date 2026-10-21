"""Foreground appearance and saved-ranking diagnostics, without model inference."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from reid_demo.benchmark import file_digest, fingerprint, write_json
from reid_demo.benchmark_masks import load_foreground_mask

VERSION = "foreground_chroma128_v1"
FIELDS = ("foreground_median_intensity", "foreground_contrast_p90_p10", "chroma_p90")


def measure_appearance(row):
    if row.get("input_mode") != "masked":
        return {"near_monochrome": None, **dict.fromkeys(FIELDS),
                "reason": "foreground_mask_unavailable"}
    mask = load_foreground_mask(row)
    with Image.open(row["path"]) as image:
        pixels = np.asarray(image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR),
                            dtype=np.float32)
    mask = np.asarray(Image.fromarray(mask).resize((128, 128), Image.Resampling.NEAREST), dtype=bool)
    foreground = pixels[mask]
    if not len(foreground):
        return {"near_monochrome": None, **dict.fromkeys(FIELDS),
                "reason": "empty_resized_foreground"}
    chroma = np.percentile(foreground.max(axis=1) - foreground.min(axis=1), 90)
    intensity = foreground.mean(axis=1)
    return {
        "near_monochrome": bool(chroma <= 5), "chroma_p90": float(chroma),
        "foreground_median_intensity": float(np.median(intensity)),
        "foreground_contrast_p90_p10": float(np.percentile(intensity, 90) - np.percentile(intensity, 10)),
        "reason": "measured",
    }


def appearance_measurements(frame, cache_root):
    root = Path(cache_root) / "appearance"
    root.mkdir(parents=True, exist_ok=True)
    values = {}
    for row in frame.to_dict("records"):
        if file_digest(row["path"]) != row["image_sha256"]:
            raise ValueError(f"Image changed before diagnostics: {row['image_id']}")
        masked = row.get("input_mode") == "masked"
        if masked and file_digest(row["mask_path"]) != row["mask_sha256"]:
            raise ValueError(f"Mask changed before diagnostics: {row['image_id']}")
        spec = {"version": VERSION, "image": row["image_sha256"],
                "mask": row.get("mask_sha256") if masked else None,
                "input_mode": row.get("input_mode")}
        path = root / (fingerprint(spec) + ".json")
        if path.exists():
            cached = json.loads(path.read_text())
            if cached.get("spec") != spec or cached.get("digest") != fingerprint(cached.get("values")):
                raise ValueError("Damaged appearance cache.")
            result = cached["values"]
        else:
            result = measure_appearance(row)
            write_json(path, {"spec": spec, "values": result, "digest": fingerprint(result)})
        values[str(row["image_id"])] = result
    return values


def _camera(row):
    parts = [row.get("source"), row.get("trap_id")]
    if any(pd.isna(v) or not str(v).strip() for v in parts):
        return None
    return tuple(str(v) for v in parts)


def ranking_summary(rows, gallery_labels, *, depth):
    known = [row for row in rows if row["known"]]
    unknown = [row for row in rows if not row["known"]]
    accepted = [row for row in rows if row.get("accepted") is True]
    correct_accepts = sum(row["known"] and row["top1_correct"] for row in accepted)
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    recall = {}
    for k in (10, 20, 50):
        recall[str(k)] = (ratio(sum(any(gallery_labels[c] == row["identity"]
                                       for c in row["candidate_ids"][:k]) for row in known), len(known))
                         if depth >= k else None)
    return {
        "queries": len(rows), "known_queries": len(known), "unknown_queries": len(unknown),
        "identities": len({r["identity"] for r in rows}),
        "encounters": len({r["encounter_id"] for r in rows}),
        "known_identities": len({r["identity"] for r in known}),
        "known_encounters": len({r["encounter_id"] for r in known}),
        "top1": ratio(sum(r["top1_correct"] for r in known), len(known)),
        "top5": ratio(sum(r["top5_correct"] for r in known), len(known)),
        "map_at_k": ratio(sum(r["ap_at_k"] for r in known), len(known)),
        "recall_at": recall,
        "shortlist_union_recall": ratio(sum(r["shortlist_hit"] for r in known), len(known)),
        "shortlist_size_min": min((len(r["shortlist_ids"]) for r in rows), default=None),
        "shortlist_size_max": max((len(r["shortlist_ids"]) for r in rows), default=None),
        "accepted": len(accepted), "correct_accepts": correct_accepts,
        "accepted_precision": ratio(correct_accepts, len(accepted)),
        "correct_known_coverage": ratio(correct_accepts, len(known)),
        "unknown_false_accepts": sum(r.get("accepted") is True for r in unknown),
        "unknown_false_accept_rate": ratio(sum(r.get("accepted") is True for r in unknown), len(unknown)),
    }


def build_diagnostics(predictions, gallery, query, appearance, *, depth=50):
    labels = dict(zip(gallery.image_id, gallery.identity))
    references = {identity: rows.to_dict("records") for identity, rows in gallery.groupby("identity")}
    query_rows = {r["image_id"]: r for r in query.to_dict("records")}
    categories = {}
    measurements = []
    for qid, row in query_rows.items():
        value = appearance[qid]
        mono = value["near_monochrome"]
        refs = references.get(row["identity"], [])
        refs_mono = [appearance[r["image_id"]]["near_monochrome"] for r in refs]
        colour = "appearance_missing" if mono is None else ("near_monochrome" if mono else "colour")
        if not refs:
            cross = "unknown_identity"
        elif mono is None or any(v is None for v in refs_mono):
            cross = "reference_appearance_missing"
        elif any(v == mono for v in refs_mono):
            cross = "same_appearance_available"
        else:
            cross = "opposite_appearance_only"
        camera = _camera(row)
        ref_cameras = [_camera(r) for r in refs]
        if not refs:
            camera_group = "unknown_identity"
        elif camera is None:
            camera_group = "query_camera_missing"
        elif camera in ref_cameras:
            camera_group = "same_camera_available"
        elif None in ref_cameras:
            camera_group = "reference_camera_missing"
        else:
            camera_group = "other_camera_only"
        source = row.get("source")
        source = "missing" if pd.isna(source) or not str(source).strip() else str(source)
        categories[qid] = ["appearance/" + colour, "cross_appearance/" + cross,
                           "camera/" + camera_group, "source/" + source]
        measurements.append({"image_id": qid, **value, "groups": categories[qid]})
    stages = {}
    for signal, rows in predictions.items():
        if (len(rows) != len(query_rows)
                or {r["query_image_id"] for r in rows} != set(query_rows)):
            raise ValueError("Diagnostics require exactly the query predictions.")
        groups = {"all": rows}
        for row in rows:
            for category in categories[row["query_image_id"]]:
                groups.setdefault(category, []).append(row)
        stages[signal] = {name: ranking_summary(group, labels, depth=depth)
                          for name, group in sorted(groups.items())}
    continuous = {}
    for field in FIELDS:
        values = [r[field] for r in measurements if r[field] is not None]
        continuous[field] = {"measured_queries": len(values), "missing_queries": len(query) - len(values),
                             "p10": float(np.percentile(values, 10)) if values else None,
                             "median": float(np.median(values)) if values else None,
                             "p90": float(np.percentile(values, 90)) if values else None}
    return {"version": VERSION, "stages": stages, "query_measurements": measurements,
            "continuous_query_summaries": continuous,
            "notes": ["Near-monochrome is a colour proxy, not verified night/infrared.",
                      "Camera groups describe gallery availability, not camera-disjoint training.",
                      "Shortlist union can be larger than returned depth. Upstream exposure is unverified."]}

