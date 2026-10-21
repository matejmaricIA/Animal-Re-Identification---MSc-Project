#!/usr/bin/env python3
"""Simulate calibrated non-match pruning with review safeguards on calibration: duplicates vs expert time."""
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

from reid_demo.benchmark import file_digest, fingerprint, write_json
from reid_demo.benchmark_clustering import check_output_directory, load_frozen_vectors
from reid_demo.benchmark_cluster_diagnostics import check_diagnostic_output, load_calibration, prepare_evaluation
from reid_demo.benchmark_review_simulation import Reviewer, bundle_keys, ordered_units, simulate_catalogue
from reid_demo import benchmark_gated_matching as gm
from reid_demo import benchmark_nonmatch_pruning as nm
import tools.simulate_czechlynx_gated_matching as gated_cli


def _seed(*parts):
    return int(fingerprint(list(parts))[:8], 16)


def live_w1(inputs, ctx):
    """W1 from the review-effort module on the same inputs (perfect reviewer)."""
    keys = bundle_keys(inputs.frame, "trap_day")
    units = ordered_units(inputs.frame, inputs.identity_codes, keys)
    labels, effort = simulate_catalogue(units, ctx.scores, len(inputs.image_ids), Reviewer(0.0), ranked=True)
    return labels, effort.decisions


def simulate(config, bundles, codes, pctx, conf_models, pair_models, identity_fold, seed):
    tag = config["config"]
    error = config["reviewer_error"]
    spacetime = config["spacetime"]
    conf_columns = list(gm.APPEARANCE_FEATURES) + (list(gm.SPACETIME_FEATURES) if spacetime else [])
    pair_columns = list(nm.PAIR_APPEARANCE) + (list(nm.PAIR_SPACETIME) if spacetime else [])
    scorers = {fold: nm.scorer(model) for fold, model in pair_models[spacetime].items()}
    sim = nm.PrunedSimulation(bundles, codes, pctx, conf_models[spacetime], conf_columns, scorers, pair_columns,
                              identity_fold, config, Reviewer(error, _seed("expert", tag, seed)),
                              Reviewer(error, _seed("second", tag, seed))).run()
    ledger = sim.ledger.copy()
    auto = gm.audit(sim.auto_log, nm.AUTO_AUDIT, Reviewer(error, _seed("audit", tag, seed)), ledger,
                    seed=_seed("sample", tag, seed))
    count = {}
    for n in nm.COUNT_AUDITS:
        extra = ledger.copy()
        result = nm.count_audit(sim.catalogue, n, Reviewer(error, _seed("count_audit", tag, n, seed)), extra,
                                pctx.ctx.scores, codes, seed=_seed("count_sample", tag, n, seed))
        count[n] = {**result, "ledger": extra}
    return sim, {"ledger": ledger, "auto": auto, "count": count}


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
    raw_sha256 = file_digest(args.raw_metadata)
    manifest_sha256 = file_digest(Path(args.calibration_dir) / "manifest.json")
    spec = {"kind": nm.VERSION, "partition": "calibration", "binding": source.binding,
            "implementation": nm.implementation_binding(),
            "raw_metadata": {"path": str(Path(args.raw_metadata).resolve()), "sha256": raw_sha256},
            "calibration_manifest_sha256": manifest_sha256, "cache_path": str(cache_path), "cache_sha256": cache_sha256,
            "seed": args.seed, "threads": args.threads, "expect_saved_references": args.expect_saved_references}
    manifest = {"schema": 1, "kind": nm.VERSION, "status": "running", "partition": "calibration",
                "run_fingerprint": fingerprint(spec), "run_spec": spec, "timing_seconds": {}, "notes": nm.NOTES}
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
        ctx, bundles, identity_fold, records = gated_cli.prepare(args, source, inputs)
        pctx = nm.PairContext(ctx)
        manifest["timing_seconds"]["inputs_and_replay"] = timer() - t

        t = timer()
        conf_models = {True: gm.fit_fold_models(records, identity_fold,
                                                list(gm.APPEARANCE_FEATURES) + list(gm.SPACETIME_FEATURES)),
                       False: gm.fit_fold_models(records, identity_fold, list(gm.APPEARANCE_FEATURES))}
        pair_records = nm.replay_pair_records(bundles, codes, pctx)
        pair_models, model_rows = {}, []
        for spacetime in (True, False):
            columns = list(nm.PAIR_APPEARANCE) + (list(nm.PAIR_SPACETIME) if spacetime else [])
            pair_models[spacetime] = nm.fit_pair_models(pair_records, identity_fold, columns)
            p = nm.cross_fitted_pmatch(pair_records, pair_models[spacetime], identity_fold, columns)
            row = {"spacetime": spacetime, **nm.evaluate_pair_model(pair_records, p)}
            model_rows.append(row)
            print(f"P(match) model ({'space-time' if spacetime else 'appearance only'}): {row['records']} records, "
                  f"cross-fitted AUC {row['auc']:.4f}, Brier {row['brier']:.4f}; "
                  + ", ".join(f"eps {e}: miss {row[f'miss_rate_eps_{e}']:.4f}, shortlist "
                              f"{row[f'new_part_shortlist_mean_eps_{e}']}" for e in nm.EPSILONS if e > 0), flush=True)
        manifest["timing_seconds"]["models"] = timer() - t

        rows, placements = [], []
        configs = nm.experiment_configs()
        w1_labels, w1_decisions = live_w1(inputs, ctx)
        for i, config in enumerate(configs):
            t = timer()
            sim, audits = simulate(config, bundles, codes, pctx, conf_models, pair_models, identity_fold, args.seed)
            row = {**{k: config[k] for k in ("config", "group", "variant", "reviewer_error", "order", "eps", "tau",
                                             "guard_n", "guard_b", "double_c", "disagree_delta", "dup_every",
                                             "dup_r", "spacetime")},
                   **nm.summarize(sim, codes, n, len(bundles), audits)}
            rows.append(row)
            if config["group"] == "G1":
                placements.extend({"config": config["config"], **p} for p in sim.placements)
            if config["group"] == "reference" and config["tau"] is None:
                w1_live = {"decisions": int(w1_decisions),
                           "labels_equal": bool(np.array_equal(sim.catalogue.entry_of, w1_labels))}
            if config["group"] == "reference" and i == sum(c["group"] == "reference" for c in configs) - 1:
                check = nm.check_references(rows, w1_live, args.expect_saved_references)
                manifest["reference_check"] = check
                write_json(out / "manifest.json", manifest)
                print(f"Reference check: {'passed' if check['passed'] else 'FAILED'} {check['problems']}", flush=True)
                if not check["passed"]:
                    raise ValueError("Reference simulations do not reproduce W1 / the saved full-scan W4 runs.")
            print(f"[{i + 1}/{len(configs)}] {config['config']}: auto {row['automation']:.2f}, misassigned "
                  f"{100 * row['misassigned_photo_rate']:.2f}%, count error {row['count_error']}, "
                  f"pruning dups {row['pruning_duplicate_events']}, {row['decisions']} decisions, "
                  f"{row['hours_per_1000_photos']:.2f} h/1000 ({row['hours_per_1000_photos_fast_click']:.2f} fast) "
                  f"({timer() - t:.0f}s)", flush=True)
        results = {"population": {"images": n, "reference_individuals": len(inputs.identities), "bundles": len(bundles),
                                  "pair_records": len(pair_records), "w4_replay_records": len(records)},
                   "reference_check": manifest["reference_check"], "pmatch_models": model_rows, "simulations": rows}
        results["declared_rules"] = nm.declared_rules(rows)
        manifest["artifacts"] = nm.write_report(out, results, placements)
        if file_digest(cache_path) != cache_sha256 or file_digest(args.raw_metadata) != raw_sha256 \
                or file_digest(Path(args.calibration_dir) / "manifest.json") != manifest_sha256:
            raise ValueError("Inputs changed during the run; use a new output directory.")
        manifest["declared_rules"] = json.loads(json.dumps(results["declared_rules"], default=str))
        manifest["source_cache_sha256_after"] = cache_sha256
        manifest["raw_metadata_sha256_after"] = raw_sha256
        manifest["timing_seconds"]["total"] = timer() - started
        manifest["status"] = "complete"
        write_json(out / "manifest.json", manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["timing_seconds"]["total"] = timer() - started
        write_json(out / "manifest.json", manifest)
        raise
    print(json.dumps({k: v for k, v in manifest["declared_rules"].items() if k != "D1_pruning_works"}, indent=1),
          flush=True)
    print(f"Report: {out / 'index.html'}", flush=True)
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", required=True, help="Completed full-collection calibration report.")
    parser.add_argument("--raw-metadata", required=True, help="Raw CzechLynx metadata (pose keypoints, coordinates).")
    parser.add_argument("--out-dir", required=True, help="New/empty output directory.")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expect-saved-references", action=argparse.BooleanOptionalAction, default=True,
                        help="Require the references to equal the saved W1 / full-scan W4 numbers (real data).")
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
