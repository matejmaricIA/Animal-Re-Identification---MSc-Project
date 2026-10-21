#!/usr/bin/env python3
"""Simulate expert review effort on calibration: W0/W1 catalogue matching vs W2 group-then-correct."""
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
from reid_demo.benchmark_clustering import check_output_directory, cluster_control, load_frozen_vectors
from reid_demo.benchmark_cluster_diagnostics import check_diagnostic_output, load_calibration, prepare_evaluation
from reid_demo.benchmark_coverage_diagnostics import identifiability_risk, image_quality_features, score_matrix
from reid_demo import benchmark_edge_grouping as eg
from reid_demo import benchmark_flank_grouping as fg
from reid_demo.benchmark_review_simulation import (
    NOTES, REVIEWER_ERRORS, STOP_AFTER_NO, UNRESOLVED_RISK_QUANTILE, VERSION, Reviewer, bundle_keys,
    declared_rules, fast_quality, implementation_binding, ordered_units, simulate_catalogue,
    simulate_group_first, write_report,
)

EDGE_PROFILE = {"algorithm": "G2_average_linkage", "tau": 0.5, "support": 1}
FLANK_PROFILE = {"algorithm": "G3_louvain", "resolution": 1.0}
LEGACY_ID = "legacy_k15_networkx_louvain"


def _f(value):
    return "n/a" if value is None else f"{value:.3f}"


def _seed(*parts):
    return int(fingerprint(list(parts))[:8], 16)


def check_reproduced(name, metrics, saved, keys=("groups", "pairwise_precision")):
    """Starting groupings must match the saved runs they claim to reproduce."""
    for key in keys:
        a, b = metrics[key], saved[key]
        if (a is None) != (b is None) or (a is not None and abs(float(a) - float(b)) > 1e-9):
            raise ValueError(f"{name} does not reproduce its saved {key}: {a} vs {b}")
    return {k: saved[k] for k in keys}


def starting_groupings(args, source, inputs, matrix, scores, risk):
    ids, valid, n = inputs.image_ids, inputs.valid, len(inputs.image_ids)
    folds = eg.identity_folds(inputs.identity_codes, seed=args.seed)
    keys = eg.metadata_keys(inputs.frame)
    # Stage A best: derived arm, k=100, G2 average-linkage tau .5, support 1. Its edges also rank merges.
    pairs, _ = eg.candidate_graph(scores, valid, keys, "derived", 100)
    features = eg.edge_features(pairs, scores, eg.rank_matrix(scores, valid), valid, keys,
                                inputs.appearance_codes, risk)
    p, _ = eg.cross_fit_probabilities(features, eg.labels_for(pairs, inputs.identity_codes), pairs, folds,
                                      columns=eg.GRAPH_FEATURES, seed=args.seed)
    edge = eg.run_grouping(EDGE_PROFILE, n, pairs, p, seed=args.seed)
    saved = pd.read_csv(Path(args.edge_grouping_dir) / "grouping_profiles.csv")
    row = saved[(saved.arm == "derived") & (saved.knn_k == 100)
                & (saved.profile == eg.profile_name(EDGE_PROFILE))].iloc[0].to_dict()
    checks = {"edge_grouping": check_reproduced("Edge grouping", eg.evaluate_grouping(
        ids, edge, inputs.frame, valid, source.appearance), row)}
    # Flank best: derived, k=100, assign, Louvain 1.0, one-to-one bridging with m=1.
    pose = fg.load_pose_facing(args.raw_metadata, inputs.frame)
    p_facing = fg.facing_probabilities(inputs.vectors, pose, folds)
    side_of = (p_facing > .5).astype(int)
    cluster_of = np.full(n, -1)
    for side in (0, 1):
        index = np.flatnonzero(valid & (side_of == side))
        s_pairs, s_p, _, _ = fg.within_side_edges(index, inputs, scores, risk, p_facing, "derived", 100,
                                                  folds, seed=args.seed)
        cluster_of[index] = fg.group_side(index, s_pairs, s_p, FLANK_PROFILE, seed=args.seed)
    matched = fg.match_one_to_one(fg.cooccurrence(cluster_of, side_of, fg.link_keys(inputs.frame, "derived")), 1)
    flank = fg.combine_labels(n, side_of, cluster_of, matched)
    saved = pd.read_csv(Path(args.flank_grouping_dir) / "profiles.csv")
    row = saved[saved.profile_id == f"derived/k100/assign/{eg.profile_name(FLANK_PROFILE)}/m1"].iloc[0].to_dict()
    checks["flank_grouping"] = check_reproduced("Flank grouping", fg.evaluate_animals(
        ids, flank, inputs.frame, source.appearance), row)
    # Legacy saved control, rebuilt with its frozen parameters.
    params = source.manifest["profiles"][LEGACY_ID]["parameters"]
    legacy_result = cluster_control(source.collection.frame.image_id.tolist(), matrix, params)
    if legacy_result.image_ids != ids:
        raise ValueError("Legacy control ordering differs from the evaluation inputs.")
    legacy = legacy_result.labels
    saved = pd.read_csv(source.directory / "profiles.csv").set_index("profile_id").loc[LEGACY_ID]
    checks["legacy_control"] = check_reproduced("Legacy control", eg.evaluate_grouping(
        ids, legacy, inputs.frame, valid, source.appearance),
        {"groups": saved.provisional_component_count, "pairwise_precision": saved.pairwise_precision})
    starts = {"edge_grouping": edge, "flank_grouping": flank, "legacy_control": legacy}
    summary = {name: {"reproduced_saved": checks[name], **fast_quality(labels, inputs.identity_codes)}
               for name, labels in starts.items()}
    return starts, pairs, p, summary


def run(args):
    started = time.perf_counter()
    source = load_calibration(args.calibration_dir)
    out = Path(args.out_dir).resolve()
    check_diagnostic_output(out, source)
    check_output_directory(out, [Path(args.raw_metadata).resolve().parent, Path(args.edge_grouping_dir).resolve(),
                                 Path(args.flank_grouping_dir).resolve()])
    cache_dir = Path(source.manifest["run_spec"]["cache_dir"]).resolve()
    cache_path = Path(source.manifest["vector_cache"]["path"]).resolve()
    if cache_path != cache_dir / f"global_{source.collection.binding['vector_cache_spec_fingerprint'][:20]}.sqlite":
        raise ValueError("Calibration vector cache path differs from its source/model contract.")
    cache_sha256 = file_digest(cache_path)
    saved_inputs = {name: file_digest(path) for name, path in (
        ("edge_grouping_profiles", Path(args.edge_grouping_dir) / "grouping_profiles.csv"),
        ("flank_grouping_profiles", Path(args.flank_grouping_dir) / "profiles.csv"),
        ("raw_metadata", Path(args.raw_metadata)))}
    spec = {"kind": VERSION, "partition": "calibration", "binding": source.binding,
            "implementation": implementation_binding(), "saved_inputs": saved_inputs,
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
        scores = score_matrix(inputs.vectors)
        quality = image_quality_features(inputs.frame)
        if quality.image_id.tolist() != inputs.image_ids:
            raise ValueError("Quality features are misaligned with vectors.")
        risk, _ = identifiability_risk(quality)
        cutoff = float(np.quantile(risk, UNRESOLVED_RISK_QUANTILE))
        starts, pairs, p, start_summary = starting_groupings(args, source, inputs, matrix, scores, risk)
        manifest["timing_seconds"]["inputs_and_starting_groupings"] = timer() - t
        print("Starting groupings reproduced: " + json.dumps(
            {k: [v["groups"], _f(v["pairwise_f1"])] for k, v in start_summary.items()}), flush=True)

        rows, curves = [], {}
        for bundling in ("trap_day", "none"):
            keys = bundle_keys(inputs.frame, bundling)
            units = ordered_units(inputs.frame, codes, keys)
            for error in REVIEWER_ERRORS:
                for unresolved in (False, True):
                    t = timer()
                    setting = {"bundling": bundling, "reviewer_error": error, "unresolved_variant": unresolved,
                               "bundles": len(set(keys)), "units": len(units)}
                    cut = cutoff if unresolved else None
                    main = bundling == "trap_day" and error == 0 and not unresolved
                    sims = []
                    for name, ranked in (("W0_unassisted", False), ("W1_ranked", True)):
                        reviewer = Reviewer(error, _seed(name, bundling, error, unresolved, args.seed))
                        labels, effort = simulate_catalogue(units, scores, n, reviewer, ranked=ranked, risk=risk,
                                                            cutoff=cut, seed=_seed("order", name, bundling, args.seed))
                        sims.append(({"workflow": name, "start": None, "stop_after_no": None}, labels, effort))
                    for start, labels0 in starts.items():
                        for k in STOP_AFTER_NO:
                            curve = [] if main and k == max(STOP_AFTER_NO) else None
                            reviewer = Reviewer(error, _seed("W2", start, k, bundling, error, unresolved, args.seed))
                            labels, effort = simulate_group_first(labels0, codes, keys, scores, pairs, p, reviewer, k,
                                                                  risk=risk, cutoff=cut, curve=curve)
                            if curve is not None:
                                curves[start] = curve
                            sims.append(({"workflow": f"W2_{start}", "start": start, "stop_after_no": k}, labels, effort))
                    for meta, labels, effort in sims:
                        e = effort.as_dict()
                        rows.append({**meta, **setting, **e, "hours": e["seconds"] / 3600,
                                     "hours_per_1000_photos": e["seconds"] / 3600 / (n / 1000),
                                     **fast_quality(labels, codes)})
                    label = f"{bundling}, error {error}, unresolved {unresolved}"
                    manifest["timing_seconds"][label] = timer() - t
                    for r in rows[-len(sims):]:
                        print(f"[{label}] {r['workflow']:22s} K={r['stop_after_no']}: {r['decisions']:6d} decisions "
                              f"({r['hours']:.1f} h), F1 {_f(r['pairwise_f1'])}, exact {r['exact_identity_recoveries']}, "
                              f"count error {r['count_error']}", flush=True)
        results = {"population": {"images": n, "reference_individuals": len(inputs.identities),
                                  "unresolved_risk_cutoff": cutoff},
                   "starting_groupings": start_summary, "simulations": rows}
        results["declared_rules"] = declared_rules(rows)
        manifest["artifacts"] = write_report(out, results, curves)
        if file_digest(cache_path) != cache_sha256:
            raise ValueError("Frozen vector cache changed during the run; use a new output directory.")
        for name, path in (("edge_grouping_profiles", Path(args.edge_grouping_dir) / "grouping_profiles.csv"),
                           ("flank_grouping_profiles", Path(args.flank_grouping_dir) / "profiles.csv")):
            if file_digest(path) != saved_inputs[name]:
                raise ValueError(f"Saved input changed during the run: {name}")
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
    parser.add_argument("--edge-grouping-dir", required=True, help="Completed Stage A edge-grouping output.")
    parser.add_argument("--flank-grouping-dir", required=True, help="Completed flank-grouping output.")
    parser.add_argument("--raw-metadata", required=True, help="Raw CzechLynx metadata CSV with pose keypoints.")
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
