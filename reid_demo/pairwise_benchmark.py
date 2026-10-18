"""Calibration-only pair sampling, checksummed evidence and diagnostic reports."""
from __future__ import annotations

from collections import Counter, defaultdict
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from reid_demo.benchmark import file_digest, fingerprint, write_json
from reid_demo.benchmark_masks import MASKED_PREPROCESSING
from reid_demo.benchmark_protocol import load_protocol, protocol_binding, protocol_frames


ARRAY_FIELDS = {"points_query", "points_gallery", "confidence", "inliers"}


def load_calibration_reference(metadata, protocol_path, reference_dir):
    """Validate roles and all saved candidate IDs before constructing models."""
    metadata, root = Path(metadata), Path(reference_dir)
    digest = file_digest(metadata)
    frame = pd.read_csv(metadata, dtype={"image_id": str, "identity": str})
    required = {"image_id", "identity", "path", "image_sha256", "mask_path", "mask_sha256",
                "source_sha256", "encounter_id", "timestamp", "input_mode", "split-time_open"}
    if (not required.issubset(frame) or frame.image_id.duplicated().any()
            or frame[list(required - {"timestamp"})].isna().any().any()
            or set(frame.input_mode) != {"masked"}):
        raise ValueError("Use complete, unique prepared masked CzechLynx metadata.")
    prepared = json.loads(metadata.with_name("manifest.json").read_text())
    if (prepared.get("prepared_metadata_sha256") != digest
            or prepared.get("masked_preprocessing") != MASKED_PREPROCESSING):
        raise ValueError("Prepared metadata / preprocessing differs from its manifest.")
    plan = load_protocol(protocol_path, frame, digest)
    _, gallery, query, _ = protocol_frames(plan, frame, "calibration")
    manifest = json.loads((root / "manifest.json").read_text())
    expected = {"status": "complete", "partition": "calibration", "stage": "development",
                "split": "time_open", "input_mode": "masked", "metadata_sha256": digest,
                "protocol": protocol_binding(plan), "gallery_ids": gallery.image_id.tolist(),
                "query_ids": query.image_id.tolist()}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Pair comparison requires matching calibration reference: {key}.")
    if manifest.get("global_model", {}).get("kind") != "pretrained_miewid_msv3":
        raise ValueError("This experiment requires the frozen MiewID global reference.")
    predictions = json.loads((root / "predictions_global.json").read_text())
    validate_predictions(predictions, gallery, query)
    diagnostics = json.loads((root / "diagnostics.json").read_text())
    measurements = diagnostics["query_measurements"]
    if (len(measurements) != len(query)
            or {r["image_id"] for r in measurements} != set(query.image_id)):
        raise ValueError("Appearance diagnostics must cover the calibration queries exactly.")
    groups = {r["image_id"]: r["groups"] for r in measurements}
    provenance = {"metadata_sha256": digest, "protocol": protocol_binding(plan),
                  "reference_dir": str(root.resolve()), "global_model": manifest["global_model"],
                  "files": {name: file_digest(root / name) for name in
                            ("manifest.json", "predictions_global.json", "diagnostics.json")}}
    return frame, gallery, query, predictions, groups, provenance


def validate_predictions(predictions, gallery, query):
    qids, gids = set(query.image_id), set(gallery.image_id)
    if (len(predictions) != len(qids)
            or {r["query_image_id"] for r in predictions} != qids):
        raise ValueError("Predictions must cover calibration queries exactly once.")
    for row in predictions:
        ids, scores = row["candidate_ids"], row["candidate_scores"]
        if (not ids or len(ids) != len(set(ids)) or not set(ids).issubset(gids)
                or row["query_image_id"] in ids or len(scores) != len(ids)
                or not np.isfinite(np.asarray(scores, float)).all()
                or any(a < b for a, b in zip(scores, scores[1:]))):
            raise ValueError("Invalid saved global candidate IDs or scores.")


def select_pairs(gallery, query, predictions, groups, *, num_queries=100, top_k=10, seed=42):
    """Round-robin diagnostic strata, favouring underrepresented identities/encounters.

    Labels stratify queries and annotate outcomes. Candidate membership and order
    come exclusively from saved retrieval, with no injected positive pairs.
    """
    if num_queries < 1 or top_k < 1:
        raise ValueError("Query and candidate budgets must be positive.")
    validate_predictions(predictions, gallery, query)
    by_id = {r["image_id"]: r for r in query.to_dict("records")}
    labels = gallery.set_index("image_id").identity.to_dict()
    known = set(labels.values())
    strata = defaultdict(list)
    for qid, row in by_id.items():
        categories = tuple(sorted(g for g in groups[qid]
                                  if g.startswith(("appearance/", "cross_appearance/", "camera/"))))
        strata[(row["identity"] in known, categories)].append(qid)
    buckets = [strata[key] for key in sorted(strata)]
    identity_counts, encounter_counts = Counter(), Counter()
    selected = []
    while len(selected) < min(num_queries, len(query)):
        for bucket in buckets:
            if not bucket:
                continue
            qid = min(bucket, key=lambda q: (
                identity_counts[by_id[q]["identity"]], encounter_counts[by_id[q]["encounter_id"]],
                fingerprint([seed, q])))
            bucket.remove(qid)
            selected.append(qid)
            identity_counts[by_id[qid]["identity"]] += 1
            encounter_counts[by_id[qid]["encounter_id"]] += 1
            if len(selected) == min(num_queries, len(query)):
                break
    ranked = {r["query_image_id"]: r for r in predictions}
    pairs, queries = [], []
    for qid in selected:
        row, prediction = by_id[qid], ranked[qid]
        if len(prediction["candidate_ids"]) < top_k:
            raise ValueError("Requested top-k exceeds saved candidate depth.")
        ids = prediction["candidate_ids"][:top_k]
        is_known = row["identity"] in known
        hit = any(labels[gid] == row["identity"] for gid in ids)
        queries.append({"query_id": qid, "identity": row["identity"], "known": is_known,
                        "shortlist_hit": hit, "groups": groups[qid]})
        for rank, (gid, score) in enumerate(zip(ids, prediction["candidate_scores"]), 1):
            pairs.append({"query_id": qid, "gallery_id": gid, "rank": rank,
                          "global_score": float(score), "same_identity": labels[gid] == row["identity"],
                          "query_identity": row["identity"], "known_query": is_known,
                          "query_groups": groups[qid]})
    return pairs, queries


def pair_cache_spec(matcher_spec, row0, row1, settings):
    return {"matcher": matcher_spec, "settings": settings,
            "ordered_inputs": [{key: row[key] for key in
                                ("image_id", "image_sha256", "mask_sha256")}
                               for row in (row0, row1)]}


class EvidenceCache:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, spec):
        path = self.root / (fingerprint(spec) + ".json")
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        if payload.get("spec") != spec or payload.get("digest") != fingerprint(payload.get("evidence")):
            raise ValueError(f"Damaged pair evidence cache: {path}")
        return payload["evidence"]

    def put(self, spec, evidence):
        if evidence["status"] == "matcher_error":
            return  # Retry runtime failures; never persist them as negative evidence.
        write_json(self.root / (fingerprint(spec) + ".json"),
                   {"spec": spec, "evidence": evidence, "digest": fingerprint(evidence)})


def roc_auc(labels, scores):
    labels = np.asarray(labels, bool)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not positives or not negatives:
        return None
    ranks = pd.Series(scores, dtype=float).rank(method="average").to_numpy()
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def summarize(records, queries):
    def metrics(rows):
        successful = [r for r in rows if r["status"] != "matcher_error"]
        result = {"pairs": len(rows), "positive_pairs": sum(r["same_identity"] for r in rows),
                  "status_counts": dict(Counter(r["status"] for r in rows)), "scores": {}}
        for name in ("n_inliers", "inlier_fraction", "coverage_min"):
            result["scores"][name] = {
                "auc_successful_pairs": roc_auc([r["same_identity"] for r in successful],
                                                [r[name] for r in successful]),
                "auc_all_pairs_errors_as_zero": roc_auc([r["same_identity"] for r in rows],
                                                        [r.get(name, 0) for r in rows]),
                "same_median": _median([r[name] for r in successful if r["same_identity"]]),
                "different_median": _median([r[name] for r in successful if not r["same_identity"]]),
            }
        result["median_seconds_uncached"] = _median([r["seconds"] for r in rows if not r["cached"]])
        result["peak_gpu_allocated_mb"] = max((r.get("peak_gpu_allocated_mb") or 0 for r in rows), default=0)
        return result
    output = {"queries": len(queries), "known_queries": sum(q["known"] for q in queries),
              "known_shortlist_hits": sum(q["known"] and q["shortlist_hit"] for q in queries),
              "known_queries_without_match": [q["query_id"] for q in queries
                                              if q["known"] and not q["shortlist_hit"]], "matchers": {}}
    for matcher in sorted({r["matcher"] for r in records}):
        rows = [r for r in records if r["matcher"] == matcher]
        grouped = {"all": rows}
        for row in rows:
            for group in row["query_groups"]:
                grouped.setdefault(group, []).append(row)
        identities = defaultdict(list)
        for row in rows:
            identities[row["query_identity"]].append(row)
        identity_metrics = {key: metrics(value) for key, value in sorted(identities.items())}
        macro = {}
        for score in ("n_inliers", "inlier_fraction", "coverage_min"):
            aucs = [m["scores"][score]["auc_all_pairs_errors_as_zero"] for m in identity_metrics.values()]
            aucs = [a for a in aucs if a is not None]
            macro[score] = {"auc": float(np.mean(aucs)) if aucs else None, "eligible_identities": len(aucs)}
        output["matchers"][matcher] = {"groups": {k: metrics(v) for k, v in sorted(grouped.items())},
                                       "identities": identity_metrics, "macro_identity_auc": macro}
    output["notes"] = [
        "Exploratory, stratified calibration sample; not held-out retrieval accuracy or park abundance.",
        "No ranking, fusion, rejection threshold, Fisher model or clustering is changed.",
        "Pair ROC AUC is descriptive; pairs share queries/identities and are not independent trials.",
        "Confidence scales differ between matchers; neither is a probability of animal identity.",
        "Near-monochrome is an appearance proxy, not verified night/infrared.",
        "A single homography is only an approximation for animal deformation.",
        "Missing correct shortlist identities cannot be recovered by pair verification.",
        "Timing excludes model initialization and DISK pre-extraction, reported separately in manifest.",
        "Cached timings describe their original computation; uncached timing medians exclude cache hits.",
    ]
    return output


def _median(values):
    return float(np.median(values)) if values else None


def visual_pair_ids(pairs, limit=24):
    """Choose positive/negative examples before matching, avoiding visual cherry-picking."""
    buckets = [[p for p in pairs if p["same_identity"] == value] for value in (True, False)]
    chosen = []
    for i in range(max(map(len, buckets), default=0)):
        for bucket in buckets:
            if i < len(bucket) and len(chosen) < limit:
                chosen.append((bucket[i]["query_id"], bucket[i]["gallery_id"]))
    return chosen


def render_matches(path, row0, row1, evidence, title):
    """Render computed correspondences; green=inlier, orange=other match."""
    tiles, scales = [], []
    for row in (row0, row1):
        with Image.open(row["path"]) as image:
            tile = image.convert("RGB")
        original = tile.size
        tile.thumbnail((480, 360))
        scales.append((tile.width / original[0], tile.height / original[1]))
        tiles.append(tile)
    canvas = Image.new("RGB", (960, 410), "#171a20")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, fill="white")
    draw.text((12, 25), "Green: homography inlier | Orange: other correspondence | up to 100 shown", fill="white")
    for i, tile in enumerate(tiles):
        canvas.paste(tile, (i * 480, 50))
    n = len(evidence.get("inliers", []))
    indices = np.linspace(0, n - 1, min(100, n), dtype=int) if n else []
    for index in indices:
        coords = []
        for side, field in enumerate(("points_query", "points_gallery")):
            x, y = evidence[field][index]
            coords.append((side * 480 + x * scales[side][0], 50 + y * scales[side][1]))
        colour = "#56e39f" if evidence["inliers"][index] else "#ffad5a"
        draw.line(coords, fill=colour, width=1)
        for x, y in coords:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=colour)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def write_report(out, records, queries, pairs, visual_ids):
    out = Path(out)
    summary = summarize(records, queries)
    write_json(out / "summary.json", summary)
    write_json(out / "pairs.json", pairs)
    write_json(out / "queries.json", queries)
    write_json(out / "results.json", records)
    pd.DataFrame([{k: v for k, v in r.items() if k not in ARRAY_FIELDS} for r in records]).to_csv(
        out / "results.csv", index=False)
    parts = ["<!doctype html><meta charset='utf-8'><title>CzechLynx pair matching</title>",
             "<style>body{font:16px sans-serif;max-width:1500px;margin:24px auto;padding:16px}"
             "table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}"
             ".pair{display:flex;gap:12px}.pair figure{width:49%;margin:0}.pair img{width:100%}"
             "pre{white-space:pre-wrap}summary{cursor:pointer}</style>",
             "<h1>RoMa / DISK + LightGlue: calibration pair diagnostics</h1>",
             f"<p>{len(queries)} queries; {len(pairs)} identical candidate pairs per matcher. "
             f"Known shortlist hits: {summary['known_shortlist_hits']}/{summary['known_queries']}.</p>",
             "<p>Exploratory sample; higher pair discrimination does not establish better retrieval. "
             "Open manifest.json for run status, setup cost and model provenance.</p>",
             "<table><tr><th>Matcher</th><th>Positive / all pairs</th><th>Errors</th>"
             "<th>Inlier fraction AUC*</th><th>Coverage AUC*</th><th>Median uncached seconds</th></tr>"]
    for name, data in summary["matchers"].items():
        metrics = data["groups"]["all"]
        parts.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            name, f"{metrics['positive_pairs']}/{metrics['pairs']}",
            metrics["status_counts"].get("matcher_error", 0),
            metrics["scores"]["inlier_fraction"]["auc_all_pairs_errors_as_zero"],
            metrics["scores"]["coverage_min"]["auc_all_pairs_errors_as_zero"],
            metrics["median_seconds_uncached"])) + "</tr>")
    parts.append("</table><p>*Operational AUC treats matcher failures as zero evidence; "
                 "successful-pair AUC and subgroup/identity counts are in summary.json.</p>")
    indexed = {(r["query_id"], r["gallery_id"], r["matcher"]): r for r in records}
    for qid, gid in visual_ids:
        parts.append(f"<h2>{html.escape(qid)} → {html.escape(gid)}</h2><div class='pair'>")
        for matcher in ("roma", "disk_lightglue"):
            record = indexed[(qid, gid, matcher)]
            stem = fingerprint([qid, gid, matcher])[:24]
            caption = (f"{matcher}: same individual={record['same_identity']}; "
                       f"rank={record['rank']}; {record['status']}; "
                       f"inliers={record.get('n_inliers', 0)}/{record.get('matches', 0)}")
            parts.append(f"<figure><img loading='lazy' src='matches/{stem}.png' "
                         f"alt='{html.escape(caption, quote=True)}'><figcaption>"
                         f"{html.escape(caption)}</figcaption></figure>")
        parts.append("</div>")
    parts.append("<details><summary>Full metrics, subgroups and limitations</summary><pre>" +
                 html.escape(json.dumps(summary, indent=2)) + "</pre></details>")
    (out / "index.html").write_text("\n".join(parts), encoding="utf-8")
    return summary
