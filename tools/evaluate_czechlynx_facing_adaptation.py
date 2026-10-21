#!/usr/bin/env python3
"""Calibration comparison for a completed facing-aware MiewID trial (work item (b)), calibration only.

Composes the v2 comparison stages (global reference, 25 clustering controls, pair diagnostic) so the
output's clustering report feeds the committed non-match pruning simulation, and adds a
facing-stratified retrieval comparison with public MiewID and adapted v2 on identical photos."""
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

from reid_demo.benchmark import file_digest, fingerprint, retrieval_metrics, write_json
from reid_demo.benchmark_clustering import check_output_directory, load_frozen_vectors
from reid_demo.benchmark_cluster_diagnostics import load_calibration, prepare_evaluation
from reid_demo.benchmark_input_audit import load_inputs, protected_inputs, verify_prepared
from reid_demo.benchmark_masks import MASKED_PREPROCESSING
from reid_demo import benchmark_facing_adaptation as fm
from reid_demo import benchmark_miewid_adaptation as ad

VERSION = "czechlynx_facing_adaptation_calibration_v1"
R1_MIN_GAIN = 0.03


def extract_facing_adapted(frame, trial, checkpoint, cache_dir, *, batch_size=8, workers=2):
    import torch
    from tools.evaluate_czechlynx import VectorCache
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, info, _ = fm.load_adapted_model(checkpoint, trial, device)
    spec = {**info, "weights_sha256": ad.native_state_digest(model.state_dict()), "input_mode": "masked",
            "masked_preprocessing": MASKED_PREPROCESSING, "expected_dimension": trial["settings"]["embedding_dim"],
            "transform": "RGB; resize 440x440 bilinear; ImageNet normalization; L2", "amp": device.type == "cuda",
            "environment": ad.environment(), "trial_manifest_sha256": trial["_manifest_sha256"],
            "mask_inventory_sha256": fingerprint(frame[["image_id", "mask_sha256"]].to_dict("records")),
            "implementation": {name: file_digest(ROOT / name) for name in
                               ("tools/evaluate_czechlynx_facing_adaptation.py", "reid_demo/benchmark_facing_adaptation.py")}}
    cache = VectorCache(Path(cache_dir) / ("global_" + fingerprint(spec)[:20] + ".sqlite"), spec)
    try:
        vectors, missing = {}, []
        for row in frame.to_dict("records"):
            vector = cache.get(row["image_id"], row["image_sha256"])
            if vector is None:
                missing.append(row)
            else:
                vectors[row["image_id"]] = vector
        print(f"Facing-adapted embeddings: {len(vectors)} cached, {len(missing)} to extract", flush=True)
        for start in range(0, len(missing), 512):
            chunk = pd.DataFrame(missing[start:start + 512]).assign(identity="unlabelled")
            matrix = ad.extract_matrix(model, chunk, trial["settings"], device, batch_size=batch_size, workers=workers)
            for row, vector in zip(chunk.to_dict("records"), matrix):
                cache.put(row["image_id"], row["image_sha256"], vector)
                vectors[row["image_id"]] = vector
            cache.db.commit()
            print(f"Extracted {min(start + 512, len(missing))}/{len(missing)}", flush=True)
        spec["dimension"] = spec["expected_dimension"]
        return np.stack([vectors[rid] for rid in frame.image_id]), spec
    finally:
        cache.close()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def facing_comparison(frame, matrices, appearance, raw_metadata, *, seed=42):
    """Common facing labels (pose, else public-MiewID cross-fitted prediction), then per-model retrieval."""
    from reid_demo import benchmark_edge_grouping as eg
    from reid_demo import benchmark_flank_grouping as fg
    from reid_demo.benchmark_coverage_diagnostics import risk_auc
    inputs = {name: prepare_evaluation(frame.image_id.tolist(), m, frame, appearance) for name, m in matrices.items()}
    first = inputs["public"]
    codes, encounters = first.identity_codes, first.frame.encounter_id.astype(str).to_numpy()
    folds = eg.identity_folds(codes, seed=seed)
    pose = fg.load_pose_facing(raw_metadata, first.frame)
    side, _ = fm.facing_labels(pose, fg.facing_probabilities(first.vectors, pose, folds))
    labelled = ~np.isnan(pose)
    result = {"common_facing": {"left": int((side == 0).sum()), "right": int((side == 1).sum()),
                                "other": int((side == -1).sum()), "pose": int(labelled.sum())}, "models": {}}
    for name, inp in inputs.items():
        if inp.identity_codes.tolist() != codes.tolist():
            raise ValueError("Model inputs are misaligned.")
        own = fg.facing_probabilities(inp.vectors, pose, folds)
        result["models"][name] = {"nn_top1": fm.nn_top1_by_facing(inp.vectors, codes, encounters, side),
                                  "facing_pose_auc_own_vectors": risk_auc(own[labelled], pose[labelled] == 1),
                                  "facing_confident_share_own_vectors": float(((own < .2) | (own > .8)).mean())}
    gain = (result["models"]["facing"]["nn_top1"]["same"]["macro_top1"]
            - result["models"]["v2"]["nn_top1"]["same"]["macro_top1"])
    result["R1"] = {"rule": f"same-side macro NN top-1 gain over v2 >= {R1_MIN_GAIN}", "gain": gain,
                    "met": bool(gain >= R1_MIN_GAIN)}
    return result


def recognition(frame, matrix, plan, chunk_size):
    from tools.evaluate_czechlynx_miewid_adaptation import rank_collection
    gallery, query, rankings = rank_collection(frame, matrix, plan, chunk_size=chunk_size)
    stats, _ = retrieval_metrics(rankings, query, gallery)
    keep = ("top1", "top5", "map_at_50", "recall_at_50", "macro_identity_top1", "known_queries")
    return {k: stats.get(k) for k in keep if k in stats} or stats


def write_report(out, comparison):
    write_json(out / "comparison.json", comparison)
    rows = []
    for name, m in comparison["facing"]["models"].items():
        nn = m["nn_top1"]
        rows.append(f"<tr><td>{html.escape(name)}</td>" + "".join(
            f"<td>{nn[k]['micro_top1']:.4f} / {nn[k]['macro_top1']:.4f} (n={nn[k]['queries']})</td>" for k in ("any", "same", "opposite"))
            + f"<td>{m['facing_pose_auc_own_vectors']:.4f}</td><td>{html.escape(json.dumps(comparison['recognition'][name]))}</td></tr>")
    (out / "index.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>CzechLynx facing adaptation</title>"
        "<style>body{font:14px system-ui;max-width:1400px;margin:30px auto;padding:16px}table{border-collapse:collapse}"
        "td,th{border:1px solid #bbb;padding:3px 6px}pre{white-space:pre-wrap}</style>"
        "<h1>Facing-aware MiewID: calibration comparison</h1><p>Calibration development evidence only.</p>"
        "<table><tr><th>Model</th><th>Any side NN top-1 (micro / macro)</th><th>Same side</th><th>Opposite side</th>"
        "<th>Facing pose AUC (own vectors)</th><th>Protocol recognition</th></tr>" + "".join(rows) + "</table>"
        "<h2>R1</h2><pre>" + html.escape(json.dumps(comparison["facing"]["R1"], indent=1)) + "</pre>"
        "<h2>Trial</h2><pre>" + html.escape(json.dumps(comparison["trial"], indent=1)) + "</pre></html>")


def run(args):
    started = time.perf_counter()
    frame, plan, binding = load_inputs(args.metadata, args.protocol_plan)
    trial = fm.load_completed_trial(args.trial_dir, frame, plan, binding)
    if trial["best_epoch"] == 0 or not trial["inner_holdout_improved"]:
        raise ValueError("Frozen epoch zero remains best; review unsuccessful adaptation before calibration.")
    public, v2 = load_calibration(args.public_calibration_dir), load_calibration(args.v2_calibration_dir)
    if public.collection.reference["global_model"].get("kind") != "pretrained_miewid_msv3":
        raise ValueError("The public baseline must be the frozen pinned MiewID calibration.")
    if (public.collection.plan != plan or v2.collection.plan != plan
            or public.collection.frame.image_id.tolist() != v2.collection.frame.image_id.tolist()):
        raise ValueError("Public and v2 calibrations must cover identical photos under this protocol.")
    caches = {}
    for name, source in (("public", public), ("v2", v2)):
        path = Path(source.manifest["vector_cache"]["path"]).resolve()
        caches[name] = (path, file_digest(path))
    out, cache = Path(args.out_dir).resolve(), Path(args.cache_dir).resolve()
    protected = protected_inputs(frame, binding) + [args.trial_dir, public.directory, v2.directory,
        public.collection.provenance["reference_dir"], v2.collection.provenance["reference_dir"],
        Path(args.raw_metadata).resolve().parent] + [str(p.parent) for p, _ in caches.values()]
    check_output_directory(out, protected)
    for value in protected:
        p = Path(value).resolve()
        if cache == p or cache.is_relative_to(p) or p.is_relative_to(cache):
            raise ValueError("New cache overlaps protected inputs or frozen caches.")
    if out == cache or out.is_relative_to(cache) or cache.is_relative_to(out):
        raise ValueError("Report and vector cache need separate directories.")
    calibration_frame = public.collection.frame
    verify_prepared(calibration_frame)
    trial = {**trial, "_manifest_sha256": file_digest(Path(args.trial_dir) / "manifest.json")}
    manifest = {"kind": VERSION, "status": "running", "partition": "calibration", "binding": binding,
                "trial_manifest_sha256": trial["_manifest_sha256"], "arm": trial["settings"]["arm"],
                "frozen_caches": {k: {"path": str(p), "sha256": h} for k, (p, h) in caches.items()},
                "raw_metadata_sha256": file_digest(args.raw_metadata),
                "implementation_sha256": file_digest(__file__)}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "manifest.json", manifest)
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=args.threads):
            matrix, spec = extract_facing_adapted(calibration_frame, trial, Path(args.trial_dir) / "best.pt", cache,
                                                  batch_size=args.batch_size, workers=args.workers)
            from tools.evaluate_czechlynx_miewid_adaptation import write_global_reference
            write_global_reference(out / "global", calibration_frame, plan, binding, matrix, spec, public.appearance,
                                   trial["run_fingerprint"], args.chunk_size)
            from tools.evaluate_czechlynx_clusters import run_experiment
            clustering = run_experiment(argparse.Namespace(metadata=args.metadata, protocol_plan=args.protocol_plan,
                reference_dir=str(out / "global"), partition="calibration", config=None, knn_k=None,
                cosine_thresholds=None, cache_dir=str(cache), out_dir=str(out / "clustering"),
                chunk_size=args.chunk_size, threads=args.threads, edge_evidence_limit=5000))
            from tools.diagnose_czechlynx_clusters import run_diagnostic
            diagnostic = run_diagnostic(argparse.Namespace(calibration_dir=str(out / "clustering"),
                out_dir=str(out / "pair_diagnostics"), threads=args.threads, chunk_size=args.chunk_size))
            if clustering["status"] != "complete" or diagnostic["status"] != "complete":
                raise ValueError("A composed calibration stage did not complete.")
            matrices = {"public": load_frozen_vectors(calibration_frame, public.collection.reference["global_model"],
                                                      public.manifest["run_spec"]["cache_dir"])[0],
                        "v2": load_frozen_vectors(calibration_frame, v2.collection.reference["global_model"],
                                                  v2.manifest["run_spec"]["cache_dir"])[0],
                        "facing": matrix}
            comparison = {"trial": {"arm": trial["settings"]["arm"], "best_epoch": trial["best_epoch"],
                                    "best_inner": {k: trial["best_metrics"].get(k) for k in
                                                   ("macro_same_facing_top1", "macro_recall_at_50", "macro_top1")},
                                    "baseline_inner": {k: trial["baseline_metrics"].get(k) for k in
                                                       ("macro_same_facing_top1", "macro_recall_at_50", "macro_top1")}},
                          "recognition": {k: recognition(calibration_frame, m, plan, args.chunk_size) for k, m in matrices.items()},
                          "facing": facing_comparison(calibration_frame, matrices, public.appearance, args.raw_metadata,
                                                      seed=args.seed)}
            write_report(out, json.loads(json.dumps(comparison, default=lambda o: o.item() if hasattr(o, "item") else str(o))))
        for name, (path, digest) in caches.items():
            if file_digest(path) != digest:
                raise ValueError(f"Frozen {name} cache changed during the comparison.")
        verify_prepared(calibration_frame)
        manifest["artifacts"] = {str(p.relative_to(out)): file_digest(p) for p in sorted(out.rglob("*"))
                                 if p.is_file() and p != out / "manifest.json"}
        manifest.update(status="complete", seconds=time.perf_counter() - started, R1=comparison["facing"]["R1"])
        write_json(out / "manifest.json", manifest)
        print(json.dumps({"R1": comparison["facing"]["R1"],
                          "same": {k: v["nn_top1"]["same"] for k, v in comparison["facing"]["models"].items()}},
                         indent=1, default=str), flush=True)
        print(f"Report: {out / 'index.html'}; clustering report for review-cost: {out / 'clustering'}", flush=True)
        return manifest
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", seconds=time.perf_counter() - started)
        write_json(out / "manifest.json", manifest)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("metadata", "protocol-plan", "trial-dir", "public-calibration-dir", "v2-calibration-dir",
                 "raw-metadata", "cache-dir", "out-dir"):
        parser.add_argument("--" + name, required=True)
    for name, default in (("threads", 8), ("chunk-size", 64), ("workers", 2), ("batch-size", 16), ("seed", 42)):
        parser.add_argument("--" + name, type=int, default=default)
    args = parser.parse_args(argv)
    try:
        run(args)
        return 0
    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
