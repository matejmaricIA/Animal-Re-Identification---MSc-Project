#!/usr/bin/env python3
"""Compare a completed native MiewID trial on full CzechLynx calibration only."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reid_demo.benchmark import file_digest, fingerprint, rank_queries, rejection_threshold, retrieval_metrics, write_json
from reid_demo.benchmark_clustering import check_output_directory, load_frozen_vectors
from reid_demo.benchmark_cluster_diagnostics import load_calibration
from reid_demo.benchmark_comparison import compare_global_predictions, load_global_reference
from reid_demo.benchmark_diagnostics import build_diagnostics
from reid_demo.benchmark_input_audit import load_inputs, protected_inputs, verify_prepared, validate_review_receipt
from reid_demo.benchmark_masks import MASKED_PREPROCESSING
from reid_demo.benchmark_miewid_adaptation import (
    environment, extract_matrix, load_adapted_model, load_completed_trial,
    native_state_digest,
)
from reid_demo.benchmark_protocol import protocol_binding

VERSION = "czechlynx_adaptation_calibration_v1"


def checked_artifacts(root, manifest):
    root = Path(root).resolve()
    artifacts = manifest.get("artifacts")
    if manifest.get("status") != "complete" or not artifacts:
        raise ValueError("A source report is incomplete.")
    for name, digest in artifacts.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or file_digest(path) != digest:
            raise ValueError("Changed source report artifact.")


def rank_collection(frame, matrix, plan, *, chunk_size=64):
    indexed = frame.set_index("image_id", drop=False)
    pool = plan["pools"]["calibration"]
    gallery = indexed.loc[pool["gallery_ids"]].reset_index(drop=True)
    query = indexed.loc[pool["query_ids"]].reset_index(drop=True)
    positions = {rid: i for i, rid in enumerate(frame.image_id)}
    gi, qi = [positions[r] for r in gallery.image_id], [positions[r] for r in query.image_id]
    rankings = list(rank_queries(query.image_id, gallery.image_id, matrix[qi], matrix[gi], depth=50,
                                shortlist_size=50, chunk_size=chunk_size, features_normalized=True))
    return gallery, query, rankings


def extract_adapted(frame, trial, checkpoint, cache_dir, *, batch_size=8, workers=2):
    import torch
    from tools.evaluate_czechlynx import VectorCache
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, info, _ = load_adapted_model(checkpoint, trial, device)
    spec = {**info, "weights_sha256": native_state_digest(model.state_dict()), "input_mode": "masked",
            "masked_preprocessing": MASKED_PREPROCESSING, "expected_dimension": trial["settings"]["embedding_dim"],
            "transform": "RGB; resize 440x440 bilinear; ImageNet normalization; L2", "amp": device.type == "cuda",
            "environment": environment(), "trial_manifest_sha256": trial["_manifest_sha256"],
            "mask_inventory_sha256": fingerprint(frame[["image_id", "mask_sha256"]].to_dict("records")),
            "implementation": {name: file_digest(ROOT / name) for name in
                               ("tools/evaluate_czechlynx_miewid_adaptation.py", "reid_demo/benchmark_miewid_adaptation.py")}}
    cache = VectorCache(Path(cache_dir) / ("global_" + fingerprint(spec)[:20] + ".sqlite"), spec)
    try:
        vectors, missing = {}, []
        for row in frame.to_dict("records"):
            vector = cache.get(row["image_id"], row["image_sha256"])
            if vector is None:
                missing.append(row)
            else:
                if vector.shape != (spec["expected_dimension"],):
                    raise ValueError("Adapted cached vector dimension differs.")
                vectors[row["image_id"]] = vector
        print(f"Adapted global embeddings: {len(vectors)} cached, {len(missing)} to extract", flush=True)
        for start in range(0, len(missing), 256):
            chunk = pd.DataFrame(missing[start:start + 256])
            # Extraction receives dummy identities, never reference identity labels.
            chunk["identity"] = "unlabelled"
            matrix = extract_matrix(model, chunk, trial["settings"], device, batch_size=batch_size, workers=workers)
            for row, vector in zip(chunk.to_dict("records"), matrix):
                cache.put(row["image_id"], row["image_sha256"], vector)
                vectors[row["image_id"]] = vector
            cache.db.commit()
            print(f"Adapted vectors complete: {min(start + 256, len(missing))}/{len(missing)} missing inputs", flush=True)
        spec["dimension"] = spec["expected_dimension"]
        return np.stack([vectors[rid] for rid in frame.image_id]), spec
    finally:
        cache.close()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def write_global_reference(out, frame, plan, binding, matrix, spec, appearance, trial_fingerprint, chunk_size):
    out = Path(out)
    out.mkdir()
    gallery, query, rankings = rank_collection(frame, matrix, plan, chunk_size=chunk_size)
    stats, predictions = retrieval_metrics(rankings, query, gallery)
    threshold = rejection_threshold(predictions, target_false_accept_rate=.05)
    stats, predictions = retrieval_metrics(rankings, query, gallery, threshold=threshold)
    stats["threshold_evaluation"] = "adaptation calibration (in-sample)"
    manifest = {"schema": 1, "status": "running", "stage": "development", "split": "time_open",
                "partition": "calibration", "seed": plan["settings"]["seed"], "input_mode": "masked",
                "metadata_sha256": binding["metadata_sha256"], "protocol": protocol_binding(plan),
                "prepared_data": json.loads(Path(binding["metadata"]).with_name("manifest.json").read_text()),
                "gallery_ids": gallery.image_id.tolist(), "query_ids": query.image_id.tolist(),
                "audit": {"excluded": plan["pools"]["calibration"]["excluded"]},
                "config": {"signals": ["global"], "depth": 50, "shortlist_size": 50},
                "global_model": spec, "trial_fingerprint": trial_fingerprint,
                "run_fingerprint": fingerprint({"binding": binding, "model": spec, "trial": trial_fingerprint})}
    write_json(out / "manifest.json", manifest)
    gallery.to_csv(out / "gallery.csv", index=False)
    query.to_csv(out / "query.csv", index=False)
    write_json(out / "predictions_global.json", predictions)
    write_json(out / "metrics.json", {"global": stats})
    write_json(out / "diagnostics.json", build_diagnostics({"global": predictions}, gallery, query, appearance))
    write_json(out / "thresholds.json", {"threshold": threshold, "scope": "recognition calibration in-sample only",
                                         "trial_fingerprint": trial_fingerprint, "global_model": spec})
    manifest["status"] = "complete"
    manifest["artifacts"] = {p.name: file_digest(p) for p in out.iterdir() if p.name != "manifest.json"}
    write_json(out / "manifest.json", manifest)
    return gallery, query, predictions, stats


def write_comparison(out, baseline, current, global_comparison, baseline_pair, current_pair):
    rows = []
    keys = ("provisional_component_count", "count_error_signed", "merge_errors", "split_errors",
            "pairwise_precision", "pairwise_recall", "macro_identity_exact_group_recovery", "unresolved_images")
    if set(baseline["profiles"]) != set(current["profiles"]):
        raise ValueError("Adaptation count profiles differ from the frozen baseline.")
    for name in baseline["profiles"]:
        before, after = baseline["profiles"][name], current["profiles"][name]
        if before["parameters"] != after["parameters"]:
            raise ValueError("Clustering parameters changed in the representation comparison.")
        rows.append({"profile": name, "reference": before["metrics"], "current": after["metrics"]})
    comparison = {"global": global_comparison, "profiles": rows,
                  "reference_pair_diagnostics": baseline_pair, "current_pair_diagnostics": current_pair,
                  "notes": "Calibration development only. Oracle connectivity uses truth; photographed groups are not park abundance."}
    write_json(out / "comparison.json", comparison)
    body = '<h1>MiewID adaptation: full calibration comparison</h1><p>Inspect grouping and count errors alongside recognition. '
    body += 'These are development results; oracle connectivity is not deployed performance, and upstream exposure is unresolved.</p>'
    body += '<p><a href="clustering/index.html">Grouping/count profiles</a> | '
    body += '<a href="pair_diagnostics/index.html">Pair evidence and oracle connectivity</a> | '
    body += '<a href="global/diagnostics.json">Recognition camera/appearance diagnostics</a> | '
    body += '<a href="comparison.json">Complete comparison data</a></p>'
    body += '<h2>Recognition</h2><table><tr><th>Metric</th><th>Frozen</th><th>Adapted</th></tr>'
    for key in ("top1", "macro_identity_top1", "top5", "map_at_k", "recall_at_50"):
        body += '<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in
                               (key, global_comparison["reference_metrics"].get(key), global_comparison["current_metrics"].get(key))) + '</tr>'
    body += '</table><h2>Grouping/count controls</h2><table><tr><th>Profile</th><th>Metric</th><th>Frozen</th><th>Adapted</th></tr>'
    for row in rows:
        for key in keys:
            body += '<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in
                                   (row["profile"], key, row["reference"].get(key), row["current"].get(key))) + '</tr>'
    body += '</table><h2>Unthresholded oracle connectivity</h2><table><tr><th>k</th><th>Connected identities (frozen/adapted)</th><th>Fragments (frozen/adapted)</th></tr>'
    old = {r["knn_k"]: r for r in baseline_pair["oracle_connectivity"] if r["threshold"] is None}
    for row in current_pair["oracle_connectivity"]:
        if row["threshold"] is None:
            before = old[row["knn_k"]]
            body += f'<tr><td>{row["knn_k"]}</td><td>{before["fully_connected_identities"]} / {row["fully_connected_identities"]}</td>'
            body += f'<td>{before["oracle_fragments"]} / {row["oracle_fragments"]}</td></tr>'
    page = '<!doctype html><meta charset="utf-8"><title>MiewID adaptation calibration</title><style>'
    page += 'body{font:16px system-ui;margin:30px}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd}</style>'
    (out / "index.html").write_text(page + body + '</table>')


def run_comparison(args):
    frame, plan, binding = load_inputs(args.metadata, args.protocol_plan)
    trial = load_completed_trial(args.trial_dir, frame, plan, binding)
    if trial["best_epoch"] == 0 or not trial["inner_holdout_improved"]:
        raise ValueError("Frozen epoch zero remains best; review unsuccessful adaptation before calibration.")
    source = load_calibration(args.baseline_calibration_dir)
    baseline_model = source.collection.reference["global_model"]
    if (baseline_model.get("kind") != "pretrained_miewid_msv3"
            or baseline_model.get("dimension") != trial["settings"]["embedding_dim"]
            or baseline_model.get("revision") != trial["settings"]["revision"]
            or baseline_model.get("weights_sha256") != trial["initial_native_state_sha256"]):
        raise ValueError("Baseline must be the same pinned native frozen MiewID model.")
    if (source.collection.plan != plan or source.collection.provenance["metadata"] != binding["metadata"]
            or source.thresholds_by_budget != {15: [.35, .45, .55, .65, .75, .85], 50: [.35, .45, .55, .65, .75, .85]}
            or len(source.manifest["profiles"]) != 25):
        raise ValueError("Use the completed original 25-profile calibration on these exact inputs.")
    diagnostic_dir = Path(args.baseline_diagnostics_dir).resolve()
    baseline_diagnostic = json.loads((diagnostic_dir / "manifest.json").read_text())
    checked_artifacts(diagnostic_dir, baseline_diagnostic)
    if baseline_diagnostic.get("run_spec", {}).get("binding") != source.binding:
        raise ValueError("Baseline diagnostic belongs to a different calibration source.")
    baseline_cache = Path(baseline_diagnostic["vector_cache"]["path"]).resolve()
    baseline_cache_hash = file_digest(baseline_cache)
    if (baseline_cache_hash != baseline_diagnostic["run_spec"]["cache_sha256"]
            or baseline_cache_hash != baseline_diagnostic["source_cache_sha256_after"]):
        raise ValueError("Frozen baseline cache changed since its diagnostic.")
    cache_root = Path(source.manifest["run_spec"]["cache_dir"]).resolve()
    if baseline_cache != cache_root / ("global_" + source.collection.binding["vector_cache_spec_fingerprint"][:20] + ".sqlite"):
        raise ValueError("Baseline cache path differs from its recorded model specification.")
    baseline_matrix, _ = load_frozen_vectors(source.collection.frame, source.collection.reference["global_model"], cache_root)
    calibration_frame = source.collection.frame
    gallery, query, reconstructed = rank_collection(calibration_frame, baseline_matrix, plan,
                                                     chunk_size=source.collection.reference["config"].get("chunk_size", 64))
    reference = load_global_reference(source.collection.provenance["reference_dir"], gallery, query,
        metadata_digest=binding["metadata_sha256"], stage="development", split="time_open", input_mode="masked", depth=50)
    saved = {r["query_image_id"]: r for r in reference["predictions"]}
    for row in reconstructed:
        before = saved[row["query_image_id"]]
        if (before["candidate_ids"] != row["candidate_ids"] or before["shortlist_ids"] != row["shortlist_ids"]
                or not np.allclose(before["candidate_scores"], row["candidate_scores"], atol=1e-6, rtol=0)
                or not np.isclose(before["confidence"], row["confidence"], atol=1e-6, rtol=0)):
            raise ValueError("Frozen saved rankings do not match their immutable vectors.")
    review = validate_review_receipt(trial["audit_receipt"], frame, binding)
    audit_binding = json.loads((Path(review["audit_dir"]) / "manifest.json").read_text())["binding"]
    protected = protected_inputs(frame, binding) + [args.trial_dir, source.directory,
        source.collection.provenance["reference_dir"], diagnostic_dir, cache_root,
        trial["audit_receipt"], Path(trial["audit_receipt"]).parent, review["audit_dir"],
        review["reviews_source"], audit_binding["images_root"], audit_binding["raw_metadata"]]
    check_output_directory(args.out_dir, protected)
    for value in protected:
        p, destination = Path(value).resolve(), Path(args.cache_dir).resolve()
        if destination == p or destination.is_relative_to(p) or p.is_relative_to(destination):
            raise ValueError("Adapted cache overlaps protected source data.")
    out, cache = Path(args.out_dir).resolve(), Path(args.cache_dir).resolve()
    if out == cache or out.is_relative_to(cache) or cache.is_relative_to(out):
        raise ValueError("Adaptation report and vector cache must have separate directories.")
    verify_prepared(calibration_frame)
    checkpoint = Path(args.trial_dir).resolve() / "best.pt"
    trial_manifest_hash = file_digest(Path(args.trial_dir) / "manifest.json")
    trial = {**trial, "_manifest_sha256": trial_manifest_hash}
    baseline_manifest_hash = file_digest(source.directory / "manifest.json")
    started = time.perf_counter()
    manifest = {"kind": VERSION, "status": "running", "partition": "calibration", "binding": binding,
                "trial_manifest_sha256": trial_manifest_hash, "baseline_cache_sha256": baseline_cache_hash,
                "baseline_calibration_manifest_sha256": baseline_manifest_hash,
                "baseline_diagnostic_manifest_sha256": file_digest(diagnostic_dir / "manifest.json"),
                "settings": {"threads": args.threads, "chunk_size": args.chunk_size, "workers": args.workers,
                             "batch_size": args.batch_size}, "implementation_sha256": file_digest(__file__)}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "manifest.json", manifest)
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=args.threads):
            matrix, spec = extract_adapted(calibration_frame, trial, checkpoint, cache,
                                           batch_size=args.batch_size, workers=args.workers)
            gallery, query, predictions, current_stats = write_global_reference(out / "global", calibration_frame, plan, binding,
                matrix, spec, source.appearance, trial["run_fingerprint"], args.chunk_size)
            from tools.evaluate_czechlynx_clusters import run_experiment
            current = run_experiment(argparse.Namespace(metadata=args.metadata, protocol_plan=args.protocol_plan,
                reference_dir=str(out / "global"), partition="calibration", config=None, knn_k=None,
                cosine_thresholds=None, cache_dir=str(cache), out_dir=str(out / "clustering"),
                chunk_size=args.chunk_size, threads=args.threads, edge_evidence_limit=5000))
            from tools.diagnose_czechlynx_clusters import run_diagnostic
            diagnostic = run_diagnostic(argparse.Namespace(calibration_dir=str(out / "clustering"),
                out_dir=str(out / "pair_diagnostics"), threads=args.threads, chunk_size=args.chunk_size))
            if current["status"] != "complete" or diagnostic["status"] != "complete":
                raise ValueError("A composed calibration stage did not complete.")
            global_comparison = compare_global_predictions(reference, predictions, query, gallery, model_spec=spec, depth=50)
            reference_stats, _ = retrieval_metrics(reference["predictions"], query, gallery)
            global_comparison["reference_metrics"]["macro_identity_top1"] = reference_stats.get("macro_identity_top1")
            global_comparison["current_metrics"]["macro_identity_top1"] = current_stats.get("macro_identity_top1")
            write_comparison(out, source.manifest, current, global_comparison,
                json.loads((diagnostic_dir / "summary.json").read_text()),
                json.loads((out / "pair_diagnostics" / "summary.json").read_text()))
        if (file_digest(baseline_cache) != baseline_cache_hash
                or file_digest(source.directory / "manifest.json") != baseline_manifest_hash
                or file_digest(Path(args.trial_dir) / "manifest.json") != trial_manifest_hash
                or file_digest(diagnostic_dir / "manifest.json") != manifest["baseline_diagnostic_manifest_sha256"]):
            raise ValueError("An immutable source changed during calibration comparison.")
        verify_prepared(calibration_frame)
        _, _, current_binding = load_inputs(args.metadata, args.protocol_plan)
        if current_binding != binding:
            raise ValueError("Metadata/protocol changed during calibration comparison.")
        manifest["artifacts"] = {str(p.relative_to(out)): file_digest(p) for p in sorted(out.rglob("*"))
                                 if p.is_file() and p != out / "manifest.json"}
        manifest.update(status="complete", seconds=time.perf_counter() - started)
        write_json(out / "manifest.json", manifest)
        print(f"Full calibration comparison: {out / 'index.html'}", flush=True)
        return manifest
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", seconds=time.perf_counter() - started)
        write_json(out / "manifest.json", manifest)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("metadata", "protocol-plan", "trial-dir", "baseline-calibration-dir", "baseline-diagnostics-dir", "cache-dir", "out-dir"):
        parser.add_argument("--" + name, required=True)
    for name, default in (("threads", 4), ("chunk-size", 64), ("batch-size", 8), ("workers", 2)):
        parser.add_argument("--" + name, type=int, default=default)
    args = parser.parse_args(argv)
    if min(args.threads, args.chunk_size, args.batch_size) < 1 or args.workers < 0:
        parser.error("Invalid threads/chunk/batch/workers.")
    try:
        run_comparison(args)
        return 0
    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
