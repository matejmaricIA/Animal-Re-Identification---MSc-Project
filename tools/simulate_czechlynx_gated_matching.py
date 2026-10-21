#!/usr/bin/env python3
"""Simulate confidence-gated ranked matching (W4) on calibration: automation vs error vs expert time."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from reid_demo.benchmark import file_digest, fingerprint, write_json
from reid_demo.benchmark_clustering import check_output_directory, load_frozen_vectors
from reid_demo.benchmark_cluster_diagnostics import check_diagnostic_output, load_calibration, prepare_evaluation
from reid_demo.benchmark_coverage_diagnostics import identifiability_risk, image_quality_features, risk_auc, score_matrix
from reid_demo import benchmark_edge_grouping as eg
from reid_demo import benchmark_flank_grouping as fg
from reid_demo.benchmark_review_simulation import Reviewer, bundle_keys, fast_quality
from reid_demo.benchmark_gated_matching import (
    APPEARANCE_FEATURES, AUDITS, NOTES, REVIEWER_ERRORS, SPACETIME_FEATURES, SWEEPS, THRESHOLDS, TOP_K, VERSION,
    Context, audit, copy_effort, cross_fitted_confidence, declared_rules, duplicate_sweep, fit_fold_models,
    implementation_binding, misassigned_photos, ordered_bundles, replay_records, simulate_gated, write_report,
)


def _seed(*parts):
    return int(fingerprint(list(parts))[:8], 16)


def coordinates(raw_metadata, frame):
    raw = pd.read_csv(raw_metadata, usecols=["path", "latitude", "longitude"])
    table = raw.set_index("path")
    lat = frame.source_path.map(table.latitude).to_numpy(float)
    lon = frame.source_path.map(table.longitude).to_numpy(float)
    return lat, lon


def prepare(args, source, inputs):
    """Label-free context plus replay records for the confidence model."""
    scores = score_matrix(inputs.vectors)
    quality = image_quality_features(inputs.frame)
    if quality.image_id.tolist() != inputs.image_ids:
        raise ValueError("Quality features are misaligned with vectors.")
    risk, _ = identifiability_risk(quality)
    folds = eg.identity_folds(inputs.identity_codes, seed=args.seed)
    pose = fg.load_pose_facing(args.raw_metadata, inputs.frame)
    facing = (fg.facing_probabilities(inputs.vectors, pose, folds) > .5).astype(int)
    lat, lon = coordinates(args.raw_metadata, inputs.frame)
    ctx = Context(inputs.frame, scores, risk, facing, lat, lon)
    bundles = ordered_bundles(inputs.frame, bundle_keys(inputs.frame, "trap_day"))
    identity_fold = {int(c): int(f) for c, f in zip(inputs.identity_codes, folds)}
    records = replay_records(bundles, inputs.identity_codes, ctx)
    return ctx, bundles, identity_fold, records


def run(args):
    started = time.perf_counter()
    source = load_calibration(args.calibration_dir)
    out = Path(args.out_dir).resolve()
    check_diagnostic_output(out, source)
    check_output_directory(out, [Path(args.raw_metadata).resolve().parent])
    cache_dir = Path(source.manifest["run_spec"]["cache_dir"]).resolve()
    cache_path = Path(source.manifest["vector_cache"]["path"]).resolve()
    if cache_path != cache_dir / f"global_{source.collection.binding['vector_cache_spec_fingerprint'][:20]}.sqlite":
        raise ValueError("Calibration vector cache path differs from its source/model contract.")
    cache_sha256 = file_digest(cache_path)
    spec = {"kind": VERSION, "partition": "calibration", "binding": source.binding,
            "implementation": implementation_binding(),
            "raw_metadata": {"path": str(Path(args.raw_metadata).resolve()), "sha256": file_digest(args.raw_metadata)},
            "cache_path": str(cache_path), "cache_sha256": cache_sha256, "seed": args.seed, "threads": args.threads}
    manifest = {"schema": 1, "kind": VERSION, "status": "running", "partition": "calibration",
                "run_fingerprint": fingerprint(spec), "run_spec": spec, "timing_seconds": {}, "notes": NOTES}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "manifest.json", manifest)
    timer = time.perf_counter
    try:
        t = timer()
        frame = source.collection.frame
        matrix, cache_audit = load_frozen_vectors(frame, source.collection.reference["global_model"], cache_dir)
        manifest["vector_cache"] = {**cache_audit,
            "loaded_matrix_sha256": hashlib.sha256(memoryview(matrix).cast("B")).hexdigest()}
        inputs = prepare_evaluation(frame.image_id.tolist(), matrix, frame, source.appearance)
        codes, n = inputs.identity_codes, len(inputs.image_ids)
        ctx, bundles, identity_fold, records = prepare(args, source, inputs)
        manifest["timing_seconds"]["inputs_and_replay"] = timer() - t
        print(f"{len(bundles)} camera-day bundles; {len(records)} replay records "
              f"({records.top1_correct.mean():.3f} top-1 correct).", flush=True)

        feature_sets = {True: list(APPEARANCE_FEATURES) + list(SPACETIME_FEATURES), False: list(APPEARANCE_FEATURES)}
        models, model_rows = {}, []
        for spacetime, columns in feature_sets.items():
            models[spacetime] = fit_fold_models(records, identity_fold, columns)
            p = cross_fitted_confidence(records, models[spacetime], identity_fold, columns)
            y = records.top1_correct.to_numpy()
            row = {"spacetime": spacetime, "records": len(records), "auc": risk_auc(p, y == 1),
                   "brier": float(np.mean((p - y) ** 2))}
            for tau in THRESHOLDS:
                accepted = p >= tau
                row[f"replay_share_ge_{tau}"] = float(accepted.mean())
                row[f"replay_error_ge_{tau}"] = float(1 - y[accepted].mean()) if accepted.any() else None
            model_rows.append(row)
            print(f"Confidence model ({'space-time' if spacetime else 'appearance only'}): cross-fitted AUC "
                  f"{row['auc']:.3f}; " + ", ".join(f"tau {tau}: {row[f'replay_share_ge_{tau}']:.2f} accepted, "
                  f"{(row[f'replay_error_ge_{tau}'] or 0):.4f} wrong" for tau in THRESHOLDS), flush=True)

        rows = []
        for spacetime, columns in feature_sets.items():
            for error in REVIEWER_ERRORS:
                for tau in (*THRESHOLDS, None):
                    if tau is None and not spacetime:
                        continue  # The no-automation baseline does not use the confidence model.
                    for k in TOP_K:
                        t = timer()
                        tag = f"{'st' if spacetime else 'app'}/err{error}/tau{tau}/K{k}"
                        reviewer = Reviewer(error, _seed("w4", tag, args.seed))
                        catalogue, effort, auto_log = simulate_gated(
                            bundles, codes, ctx, models[spacetime], identity_fold, columns,
                            np.inf if tau is None else tau, k, reviewer)
                        for sweep in SWEEPS:
                            for audit_n in AUDITS:
                                e = copy_effort(effort)
                                labels = duplicate_sweep(catalogue, ctx, sweep, Reviewer(error, _seed("sweep", tag, sweep)), e)
                                checked = audit(auto_log, audit_n, Reviewer(error, _seed("audit", tag, audit_n)), e,
                                                seed=_seed("sample", tag, audit_n))
                                wrong, rate = misassigned_photos(labels, catalogue, codes)
                                q = fast_quality(labels, codes)
                                d = e.as_dict()
                                rows.append({"config": f"{tag}/sweep{sweep}/audit{audit_n}", "spacetime": spacetime,
                                             "reviewer_error": error, "tau": tau, "top_k": k, "sweep": sweep,
                                             "audit_n": audit_n, "bundles": len(bundles),
                                             "auto_bundles": len(auto_log), "automation": len(auto_log) / len(bundles),
                                             "auto_photos": int(sum(len(r["images"]) for r in auto_log)),
                                             **d, "hours": d["seconds"] / 3600,
                                             "hours_per_1000_photos": d["seconds"] / 3600 / (n / 1000),
                                             "misassigned_photos": wrong, "misassigned_photo_rate": rate, **q,
                                             "audit_estimate": checked["estimate"], "audit_low": checked["low"],
                                             "audit_high": checked["high"], "true_bundle_error": checked["true_bundle_error"],
                                             "audit_covered": checked["covered"]})
                        r = next(x for x in rows[::-1] if x["config"] == f"{tag}/sweep25/audit100")
                        print(f"{tag}: auto {r['automation']:.2f}, misassigned {100 * r['misassigned_photo_rate']:.2f}%, "
                              f"count error {r['count_error']}, {r['hours_per_1000_photos']:.2f} h/1000 photos, "
                              f"audit {r['audit_estimate']} vs true {r['true_bundle_error']} "
                              f"({timer() - t:.0f}s)", flush=True)
        results = {"population": {"images": n, "reference_individuals": len(inputs.identities), "bundles": len(bundles),
                                  "replay_records": len(records)},
                   "confidence_models": model_rows, "simulations": rows}
        results["declared_rules"] = declared_rules(rows, n)
        manifest["artifacts"] = write_report(out, results)
        if file_digest(cache_path) != cache_sha256:
            raise ValueError("Frozen vector cache changed during the run; use a new output directory.")
        manifest["declared_rules"] = json.loads(json.dumps(results["declared_rules"], default=str))
        manifest["source_cache_sha256_after"] = cache_sha256
        manifest["timing_seconds"]["total"] = timer() - started
        manifest["status"] = "complete"
        write_json(out / "manifest.json", manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["timing_seconds"]["total"] = timer() - started
        write_json(out / "manifest.json", manifest)
        raise
    print(json.dumps(manifest["declared_rules"], indent=1), flush=True)
    print(f"Report: {out / 'index.html'}", flush=True)
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", required=True, help="Completed full-collection calibration report.")
    parser.add_argument("--raw-metadata", required=True, help="Raw CzechLynx metadata (pose keypoints, coordinates).")
    parser.add_argument("--out-dir", required=True, help="New/empty output directory.")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.threads < 1 or args.seed < 0:
        parser.error("Threads must be positive and the seed nonnegative.")
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=args.threads):
            run(args)
        return 0
    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
