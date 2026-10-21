"""Compare saved global rankings only when the evaluation inputs match."""
from __future__ import annotations

import json
from pathlib import Path

from reid_demo.benchmark import file_digest, retrieval_metrics


def _summary(predictions, query, gallery, depth):
    stats, rows = retrieval_metrics(predictions, query, gallery, depth=depth)
    summary = {key: stats[key] for key in
               ("top1", "top5", "map_at_k", "known_queries", "unknown_queries")}
    known = [row for row in rows if row["known"]]
    labels = gallery.set_index("image_id").identity.astype(str).to_dict()
    for k in (10, 20, 50):
        summary[f"recall_at_{k}"] = (
            sum(any(labels[cid] == row["identity"] for cid in row["candidate_ids"][:k]) for row in known) / len(known)
            if known and depth >= k else None)
    return summary, rows


def load_global_reference(path, gallery, query, *, metadata_digest, stage, split, input_mode, depth):
    """Validate a completed reference before any neural extraction is started."""
    root = Path(path)
    manifest_path = root / "manifest.json"
    predictions_path = root / "predictions_global.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "status": "complete", "stage": stage, "split": split,
        "input_mode": input_mode, "metadata_sha256": metadata_digest,
        "gallery_ids": gallery.image_id.tolist(), "query_ids": query.image_id.tolist(),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Global comparison requires the same completed protocol and inputs: {key} differs.")
    if manifest.get("config", {}).get("depth", 50) != depth:
        raise ValueError("Global comparison requires the same retrieval depth.")
    predictions = json.loads(predictions_path.read_text())
    metrics, rows = _summary(predictions, query, gallery, depth)
    return {
        "predictions": rows, "metrics": metrics,
        "provenance": {
            "path": str(root.resolve()), "manifest_sha256": file_digest(manifest_path),
            "predictions_sha256": file_digest(predictions_path),
            "global_model": manifest.get("global_model", {}),
        },
    }


def compare_global_predictions(reference, predictions, query, gallery, *, model_spec, depth):
    """Scores use labels only after ranking; acceptance calibration is not compared."""
    metrics, rows = _summary(predictions, query, gallery, depth)
    before = {row["query_image_id"]: row for row in reference["predictions"]}
    fixed, worsened = [], []
    for row in rows:
        if not row["known"]:
            continue
        qid = row["query_image_id"]
        if row["top1_correct"] and not before[qid]["top1_correct"]:
            fixed.append(qid)
        elif before[qid]["top1_correct"] and not row["top1_correct"]:
            worsened.append(qid)
    return {
        "schema": 1, "signal": "global", "same_inputs": True, "retrieval_depth": depth,
        "reference": reference["provenance"], "current_model": model_spec,
        "reference_metrics": reference["metrics"], "current_metrics": metrics,
        "fixed_query_ids": sorted(fixed), "worsened_query_ids": sorted(worsened),
        "fixed_queries": len(fixed), "worsened_queries": len(worsened),
        "note": "Known-query rankings on identical inputs; rejection calibration and clustering are separate.",
    }
