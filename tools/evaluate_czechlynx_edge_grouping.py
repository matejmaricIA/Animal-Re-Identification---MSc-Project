#!/usr/bin/env python3
"""Stage A: calibrated edge evidence and grouping on a wider, encounter-linked calibration graph.

With --local-evidence covering only a pilot sample, it instead evaluates the
Stage B gate (does local evidence add to the Stage A edge model?) on those edges.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from reid_demo.benchmark import file_digest, fingerprint, write_json
from reid_demo.benchmark_clustering import load_frozen_vectors
from reid_demo.benchmark_cluster_diagnostics import check_diagnostic_output, load_calibration, prepare_evaluation
from reid_demo.benchmark_coverage_diagnostics import identifiability_risk, image_quality_features, score_matrix
from reid_demo.benchmark_edge_grouping import (
    AMBIGUOUS_BAND, ARMS, BUDGETS, GLOBAL_FEATURES, GRAPH_FEATURES, LOCAL_FEATURES, MIN_GATE_EDGES, NOTES, VERSION,
    calibration_summary, candidate_graph, count_by, cross_fit_probabilities, declared_rules, edge_features,
    encounter_link_audit, evaluate_grouping, existing_control_rows, grouping_profiles,
    identity_folds, implementation_binding, labels_for, local_feature_frame, metadata_keys, pilot_sample,
    profile_name, rank_matrix, run_grouping, summarise_types, write_report,
)


def _f(value):
    return "n/a" if value is None else f"{value:.3f}"


def local_gate(rows):
    """Proposal gate: local evidence must add >= .02 AUC overall or >= .05 in the ambiguous band."""
    passed = [r for r in rows if (r["delta_auc"] or 0) >= .02 or (r["delta_ambiguous_band_auc"] or 0) >= .05]
    return {"rows": rows, "met": bool(passed), "passing": [r["graph"] for r in passed],
            "policy": "Pilot-sample cross-fitted AUC with vs without local features, same edges."}


def run(args):
    started = time.perf_counter()
    source = load_calibration(args.calibration_dir)
    out = Path(args.out_dir).resolve()
    check_diagnostic_output(out, source)
    local = None
    if args.local_evidence:
        local = pd.read_csv(args.local_evidence, dtype={"image_id_a": str, "image_id_b": str})
    cache_dir = Path(source.manifest["run_spec"]["cache_dir"]).resolve()
    cache_path = Path(source.manifest["vector_cache"]["path"]).resolve()
    key = source.collection.binding["vector_cache_spec_fingerprint"]
    if cache_path != cache_dir / f"global_{key[:20]}.sqlite":
        raise ValueError("Calibration vector cache path differs from its source/model contract.")
    cache_sha256 = file_digest(cache_path)
    spec = {"kind": VERSION, "partition": "calibration", "binding": source.binding,
            "implementation": implementation_binding(), "cache_path": str(cache_path),
            "cache_sha256": cache_sha256, "seed": args.seed, "threads": args.threads,
            "local_evidence": {"path": str(Path(args.local_evidence).resolve()),
                               "sha256": file_digest(args.local_evidence)} if args.local_evidence else None}
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
        ranks = rank_matrix(scores, valid)
        quality = image_quality_features(inputs.frame)
        if quality.image_id.tolist() != ids:
            raise ValueError("Quality features are misaligned with vectors.")
        risk, _ = identifiability_risk(quality)
        keys = metadata_keys(inputs.frame)
        manifest["timing_seconds"]["inputs"] = timer() - t
        print(f"Prepared {n} calibration images (scores, ranks, quality); no models loaded.", flush=True)

        t = timer()
        link_rows = [row for k in BUDGETS for row in
                     encounter_link_audit(inputs.identity_codes, valid, keys, scores, k)]
        manifest["timing_seconds"]["encounter_links"] = timer() - t
        for row in link_rows:
            print(f"k={row['knn_k']} {row['rule']}: {row['linked_pairs']} pairs, "
                  f"{row['fraction_joining_different_identities']} different-animal fraction, oracle "
                  f"{row['fully_connected_identities_before']} -> {row['fully_connected_identities_after']}"
                  f"/{len(inputs.identities)}", flush=True)

        folds = identity_folds(inputs.identity_codes, seed=args.seed)
        edge_models, groupings, gate_rows, graphs = {}, [], [], {}
        for arm in ARMS:
            for k in BUDGETS:
                t = timer()
                pairs, in_knn = candidate_graph(scores, valid, keys, arm, k)
                features = edge_features(pairs, scores, ranks, valid, keys, inputs.appearance_codes, risk,
                                         annotated=arm == "annotated")
                labels = labels_for(pairs, inputs.identity_codes)
                columns = list(GRAPH_FEATURES) + (["same_annotated_encounter"] if arm == "annotated" else [])
                p_global, fits_g = cross_fit_probabilities(features, labels, pairs, folds,
                                                           columns=GLOBAL_FEATURES, seed=args.seed)
                band = (p_global >= AMBIGUOUS_BAND[0]) & (p_global <= AMBIGUOUS_BAND[1])
                p_full, fits_f = cross_fit_probabilities(features, labels, pairs, folds, columns=columns, seed=args.seed)
                name = f"{arm}_k{k}"
                types = summarise_types(in_knn, features, arm)
                graphs[name] = {"edges": len(pairs), "positive_edges": int(labels.sum()),
                                "edge_types": count_by(types.tolist())}
                edge_models[f"global_{name}"] = {**calibration_summary(p_global, labels, band_mask=band), "fits": fits_g}
                edge_models[f"full_{name}"] = {**calibration_summary(p_full, labels, band_mask=band), "fits": fits_f}
                p_group = p_full
                if local is not None:
                    local_x, covered = local_feature_frame(local, ids, pairs)
                    combined = pd.concat([features.reset_index(drop=True), local_x], axis=1)
                    if covered.all():
                        p_group, fits_l = cross_fit_probabilities(combined, labels, pairs, folds,
                            columns=columns + list(LOCAL_FEATURES), seed=args.seed)
                        edge_models[f"full_local_{name}"] = {**calibration_summary(p_group, labels, band_mask=band),
                                                             "fits": fits_l}
                    elif covered.sum() >= MIN_GATE_EDGES:
                        sub = np.flatnonzero(covered)
                        base, _ = cross_fit_probabilities(combined.iloc[sub], labels[sub], pairs[sub], folds,
                                                          columns=columns, seed=args.seed)
                        plus, _ = cross_fit_probabilities(combined.iloc[sub], labels[sub], pairs[sub], folds,
                                                          columns=columns + list(LOCAL_FEATURES), seed=args.seed)
                        b = calibration_summary(base, labels[sub], band_mask=band[sub])
                        p = calibration_summary(plus, labels[sub], band_mask=band[sub])
                        local_only, _ = cross_fit_probabilities(combined.iloc[sub], labels[sub], pairs[sub], folds,
                                                                columns=list(LOCAL_FEATURES), seed=args.seed)
                        gate_rows.append({"graph": name, "pilot_edges": int(len(sub)),
                            "pilot_positive_edges": int(labels[sub].sum()),
                            "stage_a_auc": b["auc"], "with_local_auc": p["auc"],
                            "local_only_auc": calibration_summary(local_only, labels[sub])["auc"],
                            "delta_auc": p["auc"] - b["auc"],
                            "stage_a_ambiguous_band_auc": b["ambiguous_band_auc"],
                            "with_local_ambiguous_band_auc": p["ambiguous_band_auc"],
                            "ambiguous_band_edges": b["ambiguous_band_edges"],
                            "delta_ambiguous_band_auc": (p["ambiguous_band_auc"] - b["ambiguous_band_auc"])
                                if p["ambiguous_band_auc"] is not None and b["ambiguous_band_auc"] is not None else None})
                        print(f"{name} pilot gate: Stage A AUC {_f(b['auc'])} -> with local {_f(p['auc'])}; "
                              f"band {b['ambiguous_band_auc']} -> {p['ambiguous_band_auc']}", flush=True)
                print(f"{name}: {len(pairs)} edges; global AUC {_f(edge_models[f'global_{name}']['auc'])}, "
                      f"full AUC {_f(edge_models[f'full_{name}']['auc'])}", flush=True)
                manifest["timing_seconds"][f"edge_model_{name}"] = timer() - t
                if local is not None and f"full_local_{name}" not in edge_models:
                    continue  # Gate mode: no grouping on partial local evidence.
                t = timer()
                for profile in grouping_profiles():
                    labels_g = run_grouping(profile, n, pairs, p_group, seed=args.seed)
                    row = {"arm": arm, "knn_k": k, "profile": profile_name(profile), **profile,
                           **evaluate_grouping(ids, labels_g, inputs.frame, valid, source.appearance)}
                    groupings.append(row)
                    print(f"  {name} {row['profile']}: {row['groups']} groups, P {_f(row['pairwise_precision'])} "
                          f"R {_f(row['pairwise_recall'])}, exact {row['exact_identity_recoveries']}/"
                          f"{len(inputs.identities)}, merges {row['merge_errors']}", flush=True)
                manifest["timing_seconds"][f"grouping_{name}"] = timer() - t

        results = {"population": {"images": n, "reference_individuals": len(inputs.identities),
                                  "valid_vectors": int(valid.sum())},
                   "graphs": graphs, "encounter_links": link_rows, "edge_models": edge_models,
                   "existing_controls": existing_control_rows(source.directory)}
        if groupings:
            results["groupings"] = groupings
            results["declared_rules"] = declared_rules(results)
        else:
            results["groupings"] = [{"arm": None, "knn_k": None, "profile": "gate_mode_no_grouping",
                                     "groups": None, "multi_image_groups": None, "pairwise_precision": None,
                                     "pairwise_recall": None, "exact_identity_recoveries": None,
                                     "merge_errors": None, "split_errors": None, "corrections_needed": None}]
            results["declared_rules"] = {"policy": "Gate mode; grouping rules not evaluated."}
        if gate_rows:
            results["local_gate"] = local_gate(gate_rows)
            results["declared_rules"]["local_evidence_helps"] = {
                k: v for k, v in results["local_gate"].items() if k != "rows"}
        if local is None:
            # Pilot edges for Stage B: label-free strata over the union of both k=150 graphs.
            k = max(BUDGETS)
            kinds = {}
            for arm in ARMS:
                pairs, in_knn = candidate_graph(scores, valid, keys, arm, k)
                for edge, knn in zip(map(tuple, pairs), in_knn):
                    kinds.setdefault(edge, set()).add("knn" if knn else f"{arm}_link")
            union = {e: "knn" if "knn" in v else "+".join(sorted(v)) for e, v in kinds.items()}
            pairs = np.asarray(sorted(union), dtype=int)
            cosine = pd.DataFrame({"cosine": scores[pairs[:, 0], pairs[:, 1]]})
            results["pilot"] = pilot_sample(ids, pairs, cosine, [union[tuple(p)] for p in pairs],
                                            size=args.pilot_edges, seed=args.seed)
            results["pilot_summary"] = {"edges": len(results["pilot"]),
                                        "edge_types": count_by(results["pilot"].edge_type.tolist())}
        t = timer()
        manifest["artifacts"] = write_report(out, results)
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
    print(f"Report: {out / 'index.html'}", flush=True)
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", required=True, help="Completed full-collection calibration report.")
    parser.add_argument("--out-dir", required=True, help="New/empty output directory.")
    parser.add_argument("--local-evidence", help="Stage B per-edge local evidence CSV (pilot or full).")
    parser.add_argument("--pilot-edges", type=int, default=20000, help="Stage B pilot sample size (default: 20000).")
    parser.add_argument("--threads", type=int, default=8, help="Numerical-library thread limit (default: 8).")
    parser.add_argument("--seed", type=int, default=42, help="Fold/sample seed (default: 42).")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.threads < 1 or args.seed < 0 or args.pilot_edges < 1:
        parser.error("Threads and pilot size must be positive and the seed nonnegative.")
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
