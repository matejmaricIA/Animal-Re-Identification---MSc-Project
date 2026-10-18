#!/usr/bin/env python3
"""Diagnose candidate coverage and identifiability from completed calibration only."""
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
from reid_demo.benchmark_clustering import candidate_neighbours, check_output_directory, load_frozen_vectors
from reid_demo.benchmark_cluster_diagnostics import (
    check_diagnostic_output, load_calibration, oracle_connectivity, prepare_evaluation,
)
from reid_demo.benchmark_coverage_diagnostics import (
    NOTES, PRIMARY_BUDGET, VERSION, abstention_curve, audit_crosstabs, candidate_variants,
    encounter_analysis, fragment_risk_summary, fragment_tables, identifiability_risk,
    image_quality_features, implementation_binding, knn_edges, load_audit, score_matrix,
    summarise_abstention, write_coverage_report,
)


def declared_rules(results):
    """The proposal's decision rules, evaluated mechanically; interpretation stays with the reviewer."""
    curve = {r["fraction"]: r for r in results["abstention"]}
    base = next(r for r in results["candidate_variants"] if r["variant"] == f"knn_k{PRIMARY_BUDGET}")
    usable_auc = results["audit"]["risk_auc"]["usable_markings"]["auc"]
    h1 = [f for f, r in curve.items() if 0 < f <= .2
          and r["risk_fully_connected_identities"] >= 2 * r["random_mean_fully_connected_identities"]]
    encounter = results["encounters"]
    h4 = [r["variant"] for r in results["candidate_variants"]
          if r["fully_connected_identities"] - base["fully_connected_identities"] >= 5
          and r["direct_edge_precision"] is not None and base["direct_edge_precision"]
          and r["direct_edge_precision"] > base["direct_edge_precision"] * 2 / 3]
    flank = {r["relation"]: r for r in results["audit"]["flank_summary"]}
    same, opposite = flank["same_flank"]["median_cosine"], flank["opposite_flank"]["median_cosine"]
    return {
        "H1_abstention": {"fractions_meeting_2x_random": h1, "risk_auc_usable_markings": usable_auc,
                          "met": bool(h1) and usable_auc is not None and usable_auc >= .70},
        "H3_encounters": {"fully_connected_gain": encounter["fully_connected_identity_gain"],
                          "different_identity_fraction": encounter["fraction_encounter_pairs_joining_different_identities"],
                          "met": encounter["fully_connected_identity_gain"] >= 5
                          and (encounter["fraction_encounter_pairs_joining_different_identities"] or 0) <= .01},
        "H4_candidates": {"variants_meeting_rule": h4, "met": bool(h4)},
        "H2_flanks": {"same_flank_median": same, "opposite_flank_median": opposite,
                      "met": same is not None and opposite is not None and same - opposite >= .15},
        "policy": "Mechanical checks of thresholds declared in the 2026-10-18 proposal; not conclusions.",
    }


def run_diagnostic(args):
    started = time.perf_counter()
    source = load_calibration(args.calibration_dir)
    out = Path(args.out_dir).resolve()
    check_diagnostic_output(out, source)
    reviews, audit_binding = load_audit(args.audit_reviews)
    check_output_directory(out, [Path(args.audit_reviews).resolve().parent])
    cache_dir = Path(source.manifest["run_spec"]["cache_dir"]).resolve()
    cache_path = Path(source.manifest["vector_cache"]["path"]).resolve()
    key = source.collection.binding["vector_cache_spec_fingerprint"]
    if cache_path != cache_dir / f"global_{key[:20]}.sqlite":
        raise ValueError("Calibration vector cache path differs from its source/model contract.")
    cache_sha256 = file_digest(cache_path)
    spec = {"kind": VERSION, "partition": "calibration", "binding": source.binding,
            "implementation": implementation_binding(), "audit": audit_binding,
            "cache_path": str(cache_path), "cache_sha256": cache_sha256,
            "threads": args.threads, "seed": args.seed}
    manifest = {"schema": 1, "kind": VERSION, "status": "running", "partition": "calibration",
                "run_fingerprint": fingerprint(spec), "run_spec": spec, "timing_seconds": {}, "notes": NOTES}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "manifest.json", manifest)
    try:
        timer = time.perf_counter()
        frame = source.collection.frame
        matrix, cache_audit = load_frozen_vectors(frame, source.collection.reference["global_model"], cache_dir)
        manifest["vector_cache"] = {**cache_audit,
            "loaded_matrix_sha256": hashlib.sha256(memoryview(matrix).cast("B")).hexdigest()}
        inputs = prepare_evaluation(frame.image_id.tolist(), matrix, frame, source.appearance)
        scores = score_matrix(inputs.vectors)
        manifest["timing_seconds"]["vectors_and_scores"] = time.perf_counter() - timer
        print(f"Validated {len(frame)} frozen calibration vectors; no models loaded.", flush=True)

        timer = time.perf_counter()
        # Self-check: this module's builder must reproduce the calibration candidate graphs exactly.
        reference = candidate_neighbours(frame.image_id.tolist(), matrix, [15, PRIMARY_BUDGET])
        check = {}
        for k, edges in reference.items():
            ours = {(a, b): s for a, b, s in knn_edges(scores, inputs.valid, k)}
            theirs = {(a, b): s for a, b, s in edges}
            same_pairs = ours.keys() == theirs.keys()
            gap = max((abs(ours[p] - theirs[p]) for p in theirs), default=0.0) if same_pairs else None
            check[str(k)] = {"edges": len(edges), "identical_pairs": same_pairs, "max_score_difference": gap}
            if not same_pairs or gap > 1e-6:
                raise ValueError(f"Candidate builder differs from calibration at k={k}.")
        existing, _, _ = oracle_connectivity(inputs, reference[PRIMARY_BUDGET])
        manifest["candidate_self_check"] = check
        manifest["timing_seconds"]["self_check"] = time.perf_counter() - timer

        timer = time.perf_counter()
        features = image_quality_features(inputs.frame)
        if features.image_id.tolist() != inputs.image_ids:
            raise ValueError("Quality features are misaligned with vectors.")
        risk, risk_ranks = identifiability_risk(features)
        features = features.join(risk_ranks.add_prefix("risk_rank_"))
        manifest["timing_seconds"]["quality_features"] = time.perf_counter() - timer
        print("Measured label-free crop/mask quality for every image.", flush=True)

        timer = time.perf_counter()
        variants, (edges, labels, degree) = candidate_variants(scores, inputs)
        primary = next(r for r in variants if r["variant"] == f"knn_k{PRIMARY_BUDGET}")
        if (primary["fully_connected_identities"] != existing["fully_connected_identities"]
                or primary["oracle_fragments"] != existing["oracle_fragments"]):
            raise ValueError("Oracle connectivity differs from the existing diagnostic implementation.")
        manifest["timing_seconds"]["candidate_variants"] = time.perf_counter() - timer
        for row in variants:
            print(f"{row['variant']}: {row['fully_connected_identities']} connected identities, "
                  f"{row['oracle_fragments']} fragments, precision {row['direct_edge_precision']:.4f}", flush=True)

        timer = time.perf_counter()
        images, fragments = fragment_tables(inputs, labels, degree, features, risk)
        abstention_rows = abstention_curve(scores, inputs, risk, seed=args.seed)
        manifest["timing_seconds"]["abstention"] = time.perf_counter() - timer
        timer = time.perf_counter()
        results = {"images": images, "fragments": fragments, "abstention_rows": abstention_rows,
                   "abstention": summarise_abstention(abstention_rows), "candidate_variants": variants,
                   "encounters": encounter_analysis(scores, inputs),
                   "fragment_risk": fragment_risk_summary(images),
                   "audit": audit_crosstabs(images, reviews, scores, inputs),
                   "population": {"images": len(inputs.image_ids), "reference_individuals": len(inputs.identities),
                                  "valid_vectors": int(inputs.valid.sum())}}
        results["declared_rules"] = declared_rules(results)
        manifest["artifacts"] = write_coverage_report(out, inputs, results)
        manifest["timing_seconds"]["report"] = time.perf_counter() - timer
        if file_digest(cache_path) != cache_sha256:
            raise ValueError("Frozen vector cache changed during the diagnostic; use a new output directory.")
        if file_digest(source.directory / "manifest.json") != source.binding["calibration_manifest_sha256"]:
            raise ValueError("Source calibration manifest changed during the diagnostic.")
        manifest["declared_rules"] = results["declared_rules"]
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
    parser.add_argument("--audit-reviews", required=True, help="Completed input-audit reviews.json (receipt-bound).")
    parser.add_argument("--out-dir", required=True, help="New/empty separate diagnostic directory.")
    parser.add_argument("--threads", type=int, default=4, help="Numerical-library thread limit (default: 4).")
    parser.add_argument("--seed", type=int, default=42, help="Random-abstention control seed (default: 42).")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.threads < 1 or args.seed < 0:
        parser.error("Threads must be positive and the seed nonnegative.")
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
