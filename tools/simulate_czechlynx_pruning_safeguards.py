#!/usr/bin/env python3
"""Pruning floors + duplicate-merge second check (N2) on calibration, with leave-one-region-out folds."""
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

import pandas as pd

from reid_demo.benchmark import file_digest, fingerprint, write_json
from reid_demo.benchmark_clustering import check_output_directory, load_frozen_vectors
from reid_demo.benchmark_cluster_diagnostics import check_diagnostic_output, load_calibration, prepare_evaluation
from reid_demo.benchmark_coverage_diagnostics import identifiability_risk, image_quality_features, score_matrix
from reid_demo.benchmark_gated_matching import Context, ordered_bundles, replay_records
from reid_demo.benchmark_review_simulation import Reviewer, bundle_keys
from reid_demo import benchmark_flank_grouping as fg
from reid_demo import benchmark_gated_matching as gm
from reid_demo import benchmark_nonmatch_pruning as nm
from reid_demo import benchmark_pruning_safeguards as ps
import tools.simulate_czechlynx_gated_matching as gated_cli
import tools.simulate_czechlynx_nonmatch_pruning as nm_cli

DEFAULT_REFERENCE_DIR = "evaluations/czechlynx/nonmatch_pruning_facing_classes_v1"
CONF_COLUMNS = list(gm.APPEARANCE_FEATURES) + list(gm.SPACETIME_FEATURES)
PAIR_COLUMNS = list(nm.PAIR_APPEARANCE) + list(nm.PAIR_SPACETIME)
ROW_FIELDS = ("config", "group", "folds", "reviewer_error", "replicate", "order", "eps", "tau", "min_shown",
              "cosine_floor", "merge_confirm", "guard_n", "guard_b", "double_c", "disagree_delta", "dup_every",
              "dup_r")


def prepare_region(args, inputs):
    """``gated_cli.prepare`` with region folds: facing, W4 and P(match) never see the held-out region."""
    scores = score_matrix(inputs.vectors)
    quality = image_quality_features(inputs.frame)
    if quality.image_id.tolist() != inputs.image_ids:
        raise ValueError("Quality features are misaligned with vectors.")
    risk, _ = identifiability_risk(quality)
    folds = ps.region_folds(inputs.identity_codes, inputs.frame.source.astype(str).tolist())
    pose = fg.load_pose_facing(args.raw_metadata, inputs.frame)
    facing = (fg.facing_probabilities(inputs.vectors, pose, folds) > .5).astype(int)
    lat, lon = gated_cli.coordinates(args.raw_metadata, inputs.frame)
    ctx = Context(inputs.frame, scores, risk, facing, lat, lon)
    bundles = ordered_bundles(inputs.frame, bundle_keys(inputs.frame, "trap_day"))
    identity_fold = {int(c): int(f) for c, f in zip(inputs.identity_codes, folds)}
    records = replay_records(bundles, inputs.identity_codes, ctx)
    return ctx, bundles, identity_fold, records


def simulate(config, tag, bundles, codes, pctx, conf_models, pair_scorers, identity_fold, seed):
    """As ``nm_cli.simulate`` (same seeding by tag), with ``SafeguardSimulation``."""
    error = config["reviewer_error"]
    sim = ps.SafeguardSimulation(bundles, codes, pctx, conf_models, CONF_COLUMNS, pair_scorers, PAIR_COLUMNS,
                                 identity_fold, config, Reviewer(error, nm_cli._seed("expert", tag, seed)),
                                 Reviewer(error, nm_cli._seed("second", tag, seed))).run()
    ledger = sim.ledger.copy()
    auto = gm.audit(sim.auto_log, nm.AUTO_AUDIT, Reviewer(error, nm_cli._seed("audit", tag, seed)), ledger,
                    seed=nm_cli._seed("sample", tag, seed))
    count = {}
    for n in nm.COUNT_AUDITS:
        extra = ledger.copy()
        result = nm.count_audit(sim.catalogue, n, Reviewer(error, nm_cli._seed("count_audit", tag, n, seed)), extra,
                                pctx.ctx.scores, codes, seed=nm_cli._seed("count_sample", tag, n, seed))
        count[n] = {**result, "ledger": extra}
    return sim, {"ledger": ledger, "auto": auto, "count": count}


def load_saved_reference(reference_dir, cache_sha256, manifest_sha256):
    """Saved F1 rows the defaults must reproduce; their run must bind the same cache and calibration."""
    reference_dir = Path(reference_dir)
    manifest = json.loads((reference_dir / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("The saved reference run is not complete.")
    spec = manifest["run_spec"]
    if spec["cache_sha256"] != cache_sha256 or spec["calibration_manifest_sha256"] != manifest_sha256:
        raise ValueError("The saved reference run used different vectors or a different calibration.")
    table = pd.read_csv(reference_dir / "simulations.csv")
    saved = {}
    for tag in ps.REFERENCE_TAGS.values():
        match = table[table.config == tag]
        if len(match) != 1:
            raise ValueError(f"Saved reference row missing or duplicated: {tag}")
        saved[tag] = {k: match.iloc[0][k].item() for k in ps.REFERENCE_FIELDS}
    return saved, {"dir": str(reference_dir.resolve()), "manifest_sha256": file_digest(reference_dir / "manifest.json"),
                   "simulations_sha256": file_digest(reference_dir / "simulations.csv"),
                   "threads": spec.get("threads"), "seed": spec.get("seed")}


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
    saved, reference_binding = load_saved_reference(args.reference_dir, cache_sha256, manifest_sha256)
    if reference_binding["seed"] != args.seed or reference_binding["threads"] != args.threads:
        raise ValueError(f"Use the saved run's seed/threads for an exact reference: {reference_binding}.")
    spec = {"kind": ps.VERSION, "partition": "calibration", "binding": source.binding,
            "implementation": ps.implementation_binding(),
            "raw_metadata": {"path": str(Path(args.raw_metadata).resolve()), "sha256": raw_sha256},
            "calibration_manifest_sha256": manifest_sha256, "cache_path": str(cache_path), "cache_sha256": cache_sha256,
            "reference": reference_binding, "seed": args.seed, "threads": args.threads}
    manifest = {"schema": 1, "kind": ps.VERSION, "status": "running", "partition": "calibration",
                "run_fingerprint": fingerprint(spec), "run_spec": spec, "timing_seconds": {}, "notes": ps.NOTES}
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
        regions = ps.region_of_sources(inputs.frame.source.astype(str).tolist())
        manifest["timing_seconds"]["inputs"] = timer() - t

        rows, model_rows, w1_decisions = [], [], None
        for scheme in ps.FOLD_SCHEMES:
            t = timer()
            ctx, bundles, identity_fold, records = (gated_cli.prepare(args, source, inputs) if scheme == "identity"
                                                    else prepare_region(args, inputs))
            pctx = nm.PairContext(ctx)
            conf_models = gm.fit_fold_models(records, identity_fold, CONF_COLUMNS)
            pair_records = nm.replay_pair_records(bundles, codes, pctx)
            pair_models = nm.fit_pair_models(pair_records, identity_fold, PAIR_COLUMNS)
            p = nm.cross_fitted_pmatch(pair_records, pair_models, identity_fold, PAIR_COLUMNS)
            model_rows.append({"folds": scheme, **nm.evaluate_pair_model(pair_records, p)})
            scorers = {fold: nm.scorer(model) for fold, model in pair_models.items()}
            manifest["timing_seconds"][f"models_{scheme}"] = timer() - t
            print(f"[{scheme}] P(match) cross-fitted AUC {model_rows[-1]['auc']:.4f}, "
                  f"miss rate at eps .001 {model_rows[-1]['miss_rate_eps_0.001']:.4f}", flush=True)

            def record(config, tag, group):
                t0 = timer()
                sim, audits = simulate(config, tag, bundles, codes, pctx, conf_models, scorers, identity_fold, args.seed)
                row = {**{k: config.get(k) for k in ROW_FIELDS}, "config": tag, "group": group, "folds": scheme,
                       **ps.summarize(sim, codes, n, len(bundles), audits, regions)}
                rows.append(row)
                print(f"[{len(rows)}] {tag}: auto {row['automation']:.3f}, misassigned "
                      f"{100 * row['misassigned_photo_rate']:.2f}%, count {row['count_error']:+d}, pruning dups "
                      f"{row['pruning_duplicate_events']}, merges {row['duplicate_merges']} "
                      f"(wrong {row['wrong_duplicate_merges']}), {row['decisions']} decisions "
                      f"({timer() - t0:.0f}s)", flush=True)
                return row

            if scheme == "identity":
                for error, tag in ps.REFERENCE_TAGS.items():
                    record({**ps.BASE, **ps.NEW_DEFAULTS, "reviewer_error": error, "replicate": 0}, tag, "reference")
                check = ps.check_reference(rows, saved)
                _, w1_decisions = nm_cli.live_w1(inputs, ctx)
                check["w1_live_decisions"] = int(w1_decisions)
                manifest["reference_check"] = check
                write_json(out / "manifest.json", manifest)
                print(f"Reference check: {'passed' if check['passed'] else 'FAILED'} {check['problems']}; "
                      f"live W1 {w1_decisions}", flush=True)
                if not check["passed"]:
                    raise ValueError("Default settings do not reproduce the saved F1 pruning rows.")
            t = timer()
            for config in ps.experiment_configs():
                record({**config}, ps.config_tag(scheme, config), "grid")
            manifest["timing_seconds"][f"grid_{scheme}"] = timer() - t

        rules = {scheme: ps.declared_rules(rows, w1_decisions, folds=scheme) for scheme in ps.FOLD_SCHEMES}
        results = {"population": {"images": n, "reference_individuals": len(inputs.identities),
                                  "bundles": len(bundles),
                                  "region_photos": {name: int((regions == r).sum())
                                                    for r, name in ps.REGION_NAMES.items()}},
                   "reference_check": manifest["reference_check"], "pmatch_models": model_rows,
                   "declared_rules": rules, "primary_folds": ps.PRIMARY_FOLDS, "simulations": rows}
        manifest["artifacts"] = ps.write_report(out, results)
        if file_digest(cache_path) != cache_sha256 or file_digest(args.raw_metadata) != raw_sha256 \
                or file_digest(Path(args.calibration_dir) / "manifest.json") != manifest_sha256:
            raise ValueError("Inputs changed during the run; use a new output directory.")
        primary = rules[ps.PRIMARY_FOLDS]
        manifest["declared_rules"] = {"candidate": primary["candidate"], "passing": primary["passing"],
                                      "redirect_full_scan": primary["redirect_full_scan"],
                                      "redirect_adjudication_queue": primary["redirect_adjudication_queue"]}
        manifest["source_cache_sha256_after"] = cache_sha256
        manifest["raw_metadata_sha256_after"] = raw_sha256
        manifest["timing_seconds"]["total"] = timer() - started
        manifest["status"] = "complete"
        write_json(out / "manifest.json", json.loads(json.dumps(manifest, default=str)))
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["timing_seconds"]["total"] = timer() - started
        write_json(out / "manifest.json", json.loads(json.dumps(manifest, default=str)))
        raise
    print(json.dumps(manifest["declared_rules"], indent=1, default=str), flush=True)
    print(f"Report: {out / 'index.html'}", flush=True)
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", required=True, help="Completed full-collection calibration report (F1).")
    parser.add_argument("--raw-metadata", required=True, help="Raw CzechLynx metadata (pose keypoints, coordinates).")
    parser.add_argument("--out-dir", required=True, help="New/empty output directory.")
    parser.add_argument("--reference-dir", default=DEFAULT_REFERENCE_DIR,
                        help="Completed F1 non-match pruning run whose default rows must be reproduced.")
    parser.add_argument("--threads", type=int, default=4)
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
