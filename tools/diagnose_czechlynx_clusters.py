#!/usr/bin/env python3
"""Diagnose pair scores and candidate connectivity from completed calibration only."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reid_demo.benchmark import file_digest, fingerprint, write_json
from reid_demo.benchmark_clustering import candidate_neighbours, load_frozen_vectors
from reid_demo.benchmark_cluster_diagnostics import (
    NOTES, VERSION, all_pair_diagnostics, candidate_diagnostics, check_diagnostic_output,
    load_calibration, prepare_evaluation, verify_candidates, write_diagnostic_report,
)


def run_diagnostic(args):
    started = time.perf_counter()
    source = load_calibration(args.calibration_dir)
    out = Path(args.out_dir).resolve()
    check_diagnostic_output(out, source)
    cache_dir = Path(source.manifest["run_spec"]["cache_dir"]).resolve()
    cache_path = Path(source.manifest["vector_cache"]["path"]).resolve()
    key = source.collection.binding["vector_cache_spec_fingerprint"]
    if cache_path != cache_dir / f"global_{key[:20]}.sqlite":
        raise ValueError("Calibration vector cache path differs from its source/model contract.")
    cache_sha256 = file_digest(cache_path)
    spec = {"kind": VERSION, "partition": "calibration", "binding": source.binding,
            "cache_path": str(cache_path), "cache_sha256": cache_sha256,
            "thresholds_by_budget": {str(k): values for k, values in source.thresholds_by_budget.items()},
            "threads": args.threads, "chunk_size": args.chunk_size}
    manifest = {"schema": 1, "kind": VERSION, "status": "running", "partition": "calibration",
                "run_fingerprint": fingerprint(spec), "run_spec": spec,
                "timing_seconds": {}, "notes": NOTES}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "manifest.json", manifest)
    try:
        timer = time.perf_counter()
        frame = source.collection.frame
        matrix, cache_audit = load_frozen_vectors(frame, source.collection.reference["global_model"], cache_dir)
        manifest["vector_cache"] = {**cache_audit,
            "loaded_matrix_sha256": hashlib.sha256(memoryview(matrix).cast("B")).hexdigest()}
        manifest["timing_seconds"]["vector_validation"] = time.perf_counter() - timer
        print(f"Validated {len(frame)} frozen calibration vectors; no models loaded.", flush=True)
        timer = time.perf_counter()
        # This builder receives no identities, encounter labels or requested animal count.
        candidates = candidate_neighbours(frame.image_id.tolist(), matrix,
                                          sorted(source.thresholds_by_budget), chunk_size=args.chunk_size)
        inputs = prepare_evaluation(frame.image_id.tolist(), matrix, frame, source.appearance)
        manifest["candidate_audit"] = verify_candidates(source, inputs, candidates)
        manifest["timing_seconds"]["candidate_reconstruction"] = time.perf_counter() - timer
        print(f"Full candidate graphs match saved calibration evidence: "
              f"{manifest['candidate_audit']['budgets']}.", flush=True)
        timer = time.perf_counter()
        accumulators, low_pairs = all_pair_diagnostics(
            inputs, source.thresholds, chunk_size=args.chunk_size,
            progress=lambda done, total: print(f"Pair-score diagnostics: {done}/{total} unordered pairs.", flush=True))
        manifest["timing_seconds"]["all_pair_diagnostics"] = time.perf_counter() - timer
        timer = time.perf_counter()
        candidate_results = candidate_diagnostics(inputs, candidates, source.thresholds_by_budget, accumulators,
                                                  source_profiles=source.manifest["profiles"])
        manifest["timing_seconds"]["candidate_diagnostics"] = time.perf_counter() - timer
        for row in candidate_results["oracle_connectivity"]:
            if row["threshold"] is None:
                print(f"Oracle diagnostic, k={row['knn_k']}: {row['fully_connected_identities']}/"
                      f"{row['identities']} reference identities connected; {row['oracle_fragments']} fragments; "
                      f"{row['unresolved_images']} unresolved. This uses truth, not deployable inference.", flush=True)
        timer = time.perf_counter()
        summary, artifacts = write_diagnostic_report(out, inputs, accumulators, candidate_results, low_pairs)
        manifest["timing_seconds"]["report_generation"] = time.perf_counter() - timer
        # Detect concurrent cache/source edits as well as preserving our read-only access.
        if file_digest(cache_path) != cache_sha256:
            raise ValueError("Frozen vector cache changed during the diagnostic; use a new output directory.")
        if file_digest(source.directory / "manifest.json") != source.binding["calibration_manifest_sha256"]:
            raise ValueError("Source calibration manifest changed during the diagnostic.")
        manifest["artifacts"] = artifacts
        manifest["population"] = {k: summary[k] for k in
                                  ("images", "reference_individuals", "valid_vectors", "unresolved_images",
                                   "unordered_population_pairs", "scored_pairs", "unscorable_pairs")}
        manifest["source_cache_sha256_after"] = cache_sha256
        manifest["timing_seconds"]["total"] = time.perf_counter() - started
        manifest["status"] = "complete"
        write_json(out / "manifest.json", manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["timing_seconds"]["total"] = time.perf_counter() - started
        write_json(out / "manifest.json", manifest)
        raise
    print(f"Report: {out / 'index.html'}", flush=True)
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", required=True, help="Completed full-collection calibration report.")
    parser.add_argument("--out-dir", required=True, help="New/empty separate diagnostic directory.")
    parser.add_argument("--threads", type=int, default=4, help="Numerical-library thread limit (default: 4).")
    parser.add_argument("--chunk-size", type=int, default=64, help="Bounded pair-score block rows (default: 64).")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.threads < 1 or args.chunk_size < 1:
        parser.error("Threads and chunk size must be positive.")
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=args.threads):
            run_diagnostic(args)
        return 0
    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
