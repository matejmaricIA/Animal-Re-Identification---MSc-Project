#!/usr/bin/env python3
"""Flank-aware grouping on calibration: facing per image, within-facing grouping, one-to-one bridging."""
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
from reid_demo.benchmark_coverage_diagnostics import identifiability_risk, image_quality_features, load_audit, score_matrix
from reid_demo import benchmark_edge_grouping as eg
from reid_demo.benchmark_flank_grouping import (
    AMBIGUITY_POLICIES, AMBIGUOUS, BUDGETS, MIN_COOCCURRENCE, NOTES, SIDE_NAMES, VERSION, WITHIN_PROFILES,
    bilateral_bounds, combine_labels, cooccurrence, declared_rules, evaluate_animals, evaluate_flank_units,
    facing_probabilities, facing_validation, group_side, implementation_binding, link_keys, load_pose_facing,
    match_one_to_one, within_side_edges, write_report,
)


def _f(value):
    return "n/a" if value is None else f"{value:.3f}"


def run(args):
    started = time.perf_counter()
    source = load_calibration(args.calibration_dir)
    out = Path(args.out_dir).resolve()
    check_diagnostic_output(out, source)
    reviews, audit_binding = load_audit(args.audit_reviews)
    check_output_directory(out, [Path(args.audit_reviews).resolve().parent, Path(args.raw_metadata).resolve().parent])
    cache_dir = Path(source.manifest["run_spec"]["cache_dir"]).resolve()
    cache_path = Path(source.manifest["vector_cache"]["path"]).resolve()
    key = source.collection.binding["vector_cache_spec_fingerprint"]
    if cache_path != cache_dir / f"global_{key[:20]}.sqlite":
        raise ValueError("Calibration vector cache path differs from its source/model contract.")
    cache_sha256 = file_digest(cache_path)
    spec = {"kind": VERSION, "partition": "calibration", "binding": source.binding,
            "implementation": implementation_binding(), "audit": audit_binding,
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
        ids, valid, n = inputs.image_ids, inputs.valid, len(inputs.image_ids)
        scores = score_matrix(inputs.vectors)
        quality = image_quality_features(inputs.frame)
        if quality.image_id.tolist() != ids:
            raise ValueError("Quality features are misaligned with vectors.")
        risk, _ = identifiability_risk(quality)
        folds = eg.identity_folds(inputs.identity_codes, seed=args.seed)
        pose = load_pose_facing(args.raw_metadata, inputs.frame)
        p_facing = facing_probabilities(inputs.vectors, pose, folds)
        p_facing[~valid] = .5
        side_of = (p_facing > .5).astype(int)
        ambiguous = (p_facing >= AMBIGUOUS[0]) & (p_facing <= AMBIGUOUS[1])
        facing = facing_validation(p_facing, pose, ids, reviews)
        manifest["timing_seconds"]["inputs_and_facing"] = timer() - t
        print(f"Facing: pose AUC {_f(facing['pose_auc'])}, audit AUC {_f(facing.get('audit_auc'))} "
              f"(n={facing.get('audit_images')}), confident {facing['confident_share_all']:.2f}; "
              f"faces_right share {facing['predicted_faces_right_share']:.2f}", flush=True)
        facing_rows = [{"image_id": rid, "pose_faces_right": None if np.isnan(pose[i]) else int(pose[i]),
                        "p_faces_right": float(p_facing[i]), "predicted": SIDE_NAMES[int(side_of[i])],
                        "ambiguous": bool(ambiguous[i])} for i, rid in enumerate(ids)]

        profiles, side_models = [], []
        for arm in eg.ARMS:
            key_lists = link_keys(inputs.frame, arm)
            for budget in BUDGETS:
                t = timer()
                side_edges = {}
                for side in (0, 1):
                    index = np.flatnonzero(valid & (side_of == side))
                    pairs, p, labels, fits = within_side_edges(index, inputs, scores, risk, p_facing, arm, budget,
                                                               folds, seed=args.seed)
                    side_edges[side] = (index, pairs, p)
                    side_models.append({"arm": arm, "knn_k": budget, "side": SIDE_NAMES[side], "images": len(index),
                                        **{k: v for k, v in eg.calibration_summary(p, labels).items()
                                           if k != "reliability"}, "fits": fits})
                    print(f"{arm} k={budget} {SIDE_NAMES[side]}: {len(index)} images, {len(pairs)} edges, "
                          f"edge AUC {_f(side_models[-1]['auc'])}", flush=True)
                for policy in AMBIGUITY_POLICIES:
                    for profile in WITHIN_PROFILES:
                        cluster_of = np.full(n, -1)
                        sizes = {0: {}, 1: {}}
                        for side, (index, pairs, p) in side_edges.items():
                            keep = np.ones(len(pairs), bool)
                            if policy == "unresolved":
                                local_ambiguous = ambiguous[index]
                                keep = ~local_ambiguous[pairs[:, 0]] & ~local_ambiguous[pairs[:, 1]]
                            local = group_side(index, pairs[keep], p[keep], profile, seed=args.seed)
                            if policy == "unresolved":
                                local = np.where(ambiguous[index], -1, local)
                            cluster_of[index] = local
                            sizes[side] = dict(pd.Series(local[local >= 0]).value_counts())
                        flank = evaluate_flank_units(ids, cluster_of, side_of, inputs.frame, source.appearance)
                        counts = cooccurrence(cluster_of, side_of, key_lists)
                        for minimum in MIN_COOCCURRENCE:
                            matched = match_one_to_one(counts, minimum)
                            labels = combine_labels(n, side_of, cluster_of, matched)
                            bounds = bilateral_bounds(sizes[0], sizes[1], matched)
                            row = {"profile_id": f"{arm}/k{budget}/{policy}/{eg.profile_name(profile)}/m{minimum}",
                                   "arm": arm, "knn_k": budget, "ambiguity": policy, **profile,
                                   "min_cooccurrence": minimum, "matched_pairs_all_sizes": len(matched),
                                   **evaluate_animals(ids, labels, inputs.frame, source.appearance), **flank,
                                   **{k: bounds[k] for k in ("left_clusters", "right_clusters", "paired",
                                                             "unmatched_left", "unmatched_right")},
                                   "count_lower": bounds["lower"], "count_upper": bounds["upper"]}
                            profiles.append(row)
                        best = max((r for r in profiles if r["profile_id"].startswith(
                            f"{arm}/k{budget}/{policy}/{eg.profile_name(profile)}/")),
                            key=lambda r: (r["exact_identity_recoveries"], r["pairwise_f1"] or 0))
                        print(f"  {best['profile_id']}: F1 {_f(best['pairwise_f1'])}, exact "
                              f"{best['exact_identity_recoveries']}/{best['true_individuals']}, merges "
                              f"{best['merge_errors']}, flank exact {best['flank_exact_recoveries']}/"
                              f"{best['flank_units']}, count [{best['count_lower']}, {best['count_upper']}]", flush=True)
                manifest["timing_seconds"][f"{arm}_k{budget}"] = timer() - t

        results = {"population": {"images": n, "reference_individuals": len(inputs.identities),
                                  "valid_vectors": int(valid.sum()), "ambiguous_facing_images": int(ambiguous.sum())},
                   "facing": facing, "side_edge_models": side_models, "profiles": profiles,
                   "existing_controls": eg.existing_control_rows(source.directory)}
        results["declared_rules"] = declared_rules(results)
        t = timer()
        manifest["artifacts"] = write_report(out, results, facing_rows)
        manifest["timing_seconds"]["report"] = timer() - t
        if file_digest(cache_path) != cache_sha256:
            raise ValueError("Frozen vector cache changed during the run; use a new output directory.")
        if file_digest(source.directory / "manifest.json") != source.binding["calibration_manifest_sha256"]:
            raise ValueError("Source calibration manifest changed during the run.")
        manifest["declared_rules"] = results["declared_rules"]
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
    print(json.dumps(results["declared_rules"], indent=1), flush=True)
    print(f"Report: {out / 'index.html'}", flush=True)
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", required=True, help="Completed full-collection calibration report.")
    parser.add_argument("--raw-metadata", required=True, help="Raw CzechLynx metadata CSV with pose keypoints.")
    parser.add_argument("--audit-reviews", required=True, help="Completed input-audit reviews.json (validation only).")
    parser.add_argument("--out-dir", required=True, help="New/empty output directory.")
    parser.add_argument("--threads", type=int, default=8, help="Numerical-library thread limit (default: 8).")
    parser.add_argument("--seed", type=int, default=42, help="Fold seed (default: 42).")
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
