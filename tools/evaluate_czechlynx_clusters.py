#!/usr/bin/env python3
"""Full-collection, cache-only clustering profiles and frozen count evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reid_demo.benchmark import file_digest, fingerprint, write_json
from reid_demo.benchmark_clustering import (
    DEFAULT_BUDGETS, DEFAULT_THRESHOLDS, NOTES, VERSION, appearance_evidence,
    calibration_profiles, candidate_neighbours, check_output_directory, cluster_control,
    evaluate_clustering, export_configuration, load_collection, load_frozen_configuration,
    load_frozen_vectors, profile_id, write_overview, write_profile_report,
)


def run_experiment(args):
    started = time.perf_counter()
    collection = load_collection(args.metadata, args.protocol_plan, args.reference_dir, args.partition)
    frozen = None
    if args.partition == "selection":
        frozen = load_frozen_configuration(args.config, collection)
        parameters = [frozen["parameters"]]
    else:
        parameters = calibration_profiles(collection.provenance["seed"],
                                          args.knn_k or DEFAULT_BUDGETS,
                                          args.cosine_thresholds or DEFAULT_THRESHOLDS)
    spec = {
        "kind": VERSION, "partition": args.partition, "binding": collection.binding,
        "collection": collection.provenance, "profile_parameters": parameters,
        "cache_dir": str(Path(args.cache_dir).resolve()), "chunk_size": args.chunk_size,
        "edge_evidence_limit": args.edge_evidence_limit, "threads": args.threads,
    }
    if frozen:
        spec["frozen_config_sha256"] = file_digest(args.config)
        spec["calibration_run_fingerprint"] = frozen["calibration_run_fingerprint"]
    run_hash = fingerprint(spec)
    protected = [args.reference_dir, args.cache_dir, Path(args.metadata).parent]
    protected.extend({str(Path(p).parent) for p in collection.frame.path})
    if "mask_path" in collection.frame:
        protected.extend({str(Path(p).parent) for p in collection.frame.mask_path.dropna()})
    if frozen:
        protected.append(frozen["calibration_report_dir"])
    check_output_directory(args.out_dir, protected, run_fingerprint=run_hash)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": 1, "kind": VERSION, "status": "running", "partition": args.partition,
        "binding": collection.binding, "collection": collection.provenance,
        "run_fingerprint": run_hash, "run_spec": spec, "profiles": {},
        "notes": NOTES, "timing_seconds": {},
    }
    write_json(out / "manifest.json", manifest)
    try:
        timer = time.perf_counter()
        matrix, cache_audit = load_frozen_vectors(collection.frame, collection.reference["global_model"], args.cache_dir)
        manifest["vector_cache"] = cache_audit
        manifest["timing_seconds"]["vector_validation"] = time.perf_counter() - timer
        print(f"Validated {len(collection.frame)} frozen vectors; collection is complete.", flush=True)
        collection.frame.to_csv(out / "collection.csv", index=False)
        timer = time.perf_counter()
        appearance = appearance_evidence(collection.frame.drop(columns=["identity"]), args.cache_dir, out)
        manifest["timing_seconds"]["appearance_evidence"] = time.perf_counter() - timer
        ids = collection.frame.image_id.tolist()
        timer = time.perf_counter()
        budgets = sorted({p["knn_k"] for p in parameters if p["algorithm"] != "legacy_neighbour"})
        candidates = candidate_neighbours(ids, matrix, budgets, chunk_size=args.chunk_size)
        manifest["timing_seconds"]["candidate_retrieval"] = time.perf_counter() - timer
        print(f"Candidate retrieval complete for budgets {budgets}.", flush=True)
        for p in parameters:
            name = profile_id(p)
            print(f"Clustering {name} ...", flush=True)
            timer = time.perf_counter()
            result = cluster_control(ids, matrix, p, edges=candidates.get(p["knn_k"]),
                                     chunk_size=args.chunk_size, evidence_limit=args.edge_evidence_limit)
            metrics, rows = evaluate_clustering(result, collection.frame, appearance=appearance)
            elapsed = time.perf_counter() - timer
            artifacts = write_profile_report(out / "profiles" / name, result, p, metrics, rows)
            manifest["profiles"][name] = {
                "parameters": p, "metrics": metrics, "edge_audit": result.audit,
                "seconds": elapsed, "artifacts": artifacts,
            }
            write_json(out / "manifest.json", manifest)
            print(f"  {metrics['provisional_component_count']} provisional components; "
                  f"count error {metrics['count_error_signed']}; "
                  f"merges {metrics['merge_errors']}; fragmented identities {metrics['split_errors']}.", flush=True)
        write_overview(out, manifest["profiles"], args.partition)
        names = ["collection.csv", "appearance.json", "profiles.csv", "metrics.json", "index.html"]
        if args.partition == "calibration":
            names.append("calibration_sensitivity.png")
        manifest["artifacts"] = {name: file_digest(out / name) for name in names}
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
    parser.add_argument("--metadata", help="Prepared CSV (default: provided-mask metadata).")
    parser.add_argument("--protocol-plan", help="Existing identity_pools_v1.json.")
    parser.add_argument("--reference-dir", help="Completed phase 2 global report for this partition.")
    parser.add_argument("--partition", choices=("calibration", "selection"))
    parser.add_argument("--cache-dir", help="Frozen global cache directory; opened read only.")
    parser.add_argument("--out-dir", help="Separate clustering report directory.")
    parser.add_argument("--config", help="Exported clustering configuration required for selection.")
    parser.add_argument("--knn-k", nargs="+", type=int, help="Calibration grid (default: 15 50).")
    parser.add_argument("--cosine-thresholds", nargs="+", type=float,
                        help="Calibration grid (default: .35 .45 .55 .65 .75 .85).")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--edge-evidence-limit", type=int, default=5000)
    parser.add_argument("--calibration-dir", help="Completed clustering report for configuration export.")
    parser.add_argument("--export-profile", help="Explicit profile ID selected from calibration profiles.csv.")
    parser.add_argument("--config-out", help="New frozen configuration JSON path.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    export = any((args.calibration_dir, args.export_profile, args.config_out))
    if export:
        if not all((args.calibration_dir, args.export_profile, args.config_out)):
            parser.error("Export requires --calibration-dir, --export-profile and --config-out.")
        if any((args.partition, args.reference_dir, args.out_dir, args.config, args.metadata,
                args.protocol_plan, args.cache_dir, args.knn_k, args.cosine_thresholds)):
            parser.error("Configuration export cannot be combined with clustering/evaluation options.")
        try:
            export_configuration(args.calibration_dir, args.export_profile, args.config_out)
            print(f"Frozen configuration: {args.config_out}")
            return 0
        except (ValueError, OSError, KeyError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    if not all((args.partition, args.reference_dir, args.out_dir)):
        parser.error("A run requires --partition, --reference-dir and --out-dir.")
    if args.partition == "selection":
        if not args.config:
            parser.error("Selection requires --config exported from calibration.")
        if args.knn_k is not None or args.cosine_thresholds is not None:
            parser.error("Selection cannot profile thresholds or override candidate budgets.")
    elif args.config:
        parser.error("Calibration profiles configurations; do not supply --config.")
    if args.chunk_size < 1 or args.threads < 1 or args.edge_evidence_limit < 0:
        parser.error("Chunk size/threads must be positive; edge-evidence limit must be nonnegative.")
    if args.knn_k and (min(args.knn_k) < 1 or len(set(args.knn_k)) != len(args.knn_k)):
        parser.error("Candidate budgets must be distinct positive integers.")
    args.metadata = args.metadata or "data/czechlynx/prepared_masked/masked/metadata.csv"
    args.protocol_plan = args.protocol_plan or "data/czechlynx/protocols/identity_pools_v1.json"
    args.cache_dir = args.cache_dir or "data/czechlynx/cache/masked"
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=args.threads):
            run_experiment(args)
        return 0
    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
