#!/usr/bin/env python3
"""One-shot evaluation of the frozen review operating point on the reserved selection pool (work item (c)).

Verifies the committed freeze record, extracts F1 vectors for the selection photos, builds the frozen
label-free context and runs the unchanged simulation once. A completed run, or any attempt that reached
simulation, blocks further runs with the same freeze fingerprint."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from reid_demo.benchmark import file_digest, rank_queries, retrieval_metrics, write_json
from reid_demo.benchmark_clustering import check_output_directory
from reid_demo.benchmark_cluster_diagnostics import _write_table, prepare_evaluation
from reid_demo.benchmark_input_audit import load_inputs, protected_inputs, verify_prepared
from reid_demo.benchmark_review_simulation import plain
from reid_demo import benchmark_facing_adaptation as fm
from reid_demo import benchmark_frozen_review as fr


def one_shot_check(out, fingerprint, retry_of=None):
    """Refuse if this freeze was already evaluated, or if an earlier attempt may have produced results."""
    earlier = []
    for path in sorted(Path(out).parent.glob("*/manifest.json")):
        try:
            saved = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if saved.get("kind") == fr.SELECTION_VERSION and saved.get("freeze_fingerprint") == fingerprint:
            earlier.append((path.parent.resolve(), saved))
    for directory, saved in earlier:
        if saved.get("status") == "complete":
            raise ValueError(f"This frozen configuration was already evaluated on selection: {directory}")
        stage = saved.get("stage", "verification")
        if fr.STAGES.index(stage) >= fr.STAGES.index("simulation"):
            raise ValueError(f"An earlier attempt reached '{stage}'; results may exist. No rerun: {directory}")
        if retry_of is None or Path(retry_of).resolve() != directory:
            raise ValueError(f"An earlier attempt exists ({directory}); pass --retry-of it and document the retry.")
    if retry_of is not None and not earlier:
        raise ValueError("--retry-of names no earlier attempt of this frozen configuration.")


def selection_frame(frame, plan):
    pool = plan["pools"]["selection"]
    ids = sorted(set(pool["gallery_ids"]) | set(pool["query_ids"]))
    selected = frame[frame.image_id.isin(ids)].sort_values("image_id").reset_index(drop=True)
    if len(selected) != len(ids):
        raise ValueError("Selection membership is incomplete.")
    return selected, pool


def extract_selection(selection, trial_dir, frame, plan, binding, cache_dir, *, batch_size, workers):
    from tools.evaluate_czechlynx_facing_adaptation import extract_facing_adapted
    trial = fm.load_completed_trial(trial_dir, frame, plan, binding)
    trial = {**trial, "_manifest_sha256": file_digest(Path(trial_dir) / "manifest.json")}
    return extract_facing_adapted(selection, trial, Path(trial_dir) / "best.pt", cache_dir,
                                  batch_size=batch_size, workers=workers)


def recognition(selection, vectors, pool):
    positions = {rid: i for i, rid in enumerate(selection.image_id)}
    indexed = selection.set_index("image_id", drop=False)
    gallery = indexed.loc[pool["gallery_ids"]].reset_index(drop=True)
    query = indexed.loc[pool["query_ids"]].reset_index(drop=True)
    gi, qi = [positions[r] for r in gallery.image_id], [positions[r] for r in query.image_id]
    rankings = list(rank_queries(query.image_id, gallery.image_id, vectors[qi], vectors[gi], depth=50,
                                 shortlist_size=50, chunk_size=64, features_normalized=True))
    stats, _ = retrieval_metrics(rankings, query, gallery)
    return {k: stats[k] for k in ("top1", "top5", "map_at_50", "recall_at_50", "macro_identity_top1", "known_queries")
            if k in stats} or stats


def write_report(out, results):
    results = plain(results)
    write_json(out / "summary.json", results)
    _write_table(out / "simulations.csv", results["simulations"], list(results["simulations"][0]))
    _write_table(out / "per_animal.csv", results["per_animal"], list(results["per_animal"][0]))
    keys = ["tau", "reviewer_error", "automation", "misassigned_photo_rate", "count_error", "exact_identity_recoveries",
            "decisions", "hours_per_1000_photos", "hours_per_1000_photos_fast_click", "audit_estimate",
            "true_bundle_error", "audit_covered"]
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else "" if v is None else str(v)
    table = "<table><tr>" + "".join(f"<th>{k}</th>" for k in keys) + "</tr>" + "".join(
        "<tr>" + "".join(f"<td>{html.escape(fmt(r.get(k)))}</td>" for k in keys) + "</tr>"
        for r in results["simulations"]) + "</table>"
    (out / "index.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>CzechLynx frozen review on selection</title>"
        "<style>body{font:14px system-ui;max-width:1400px;margin:30px auto;padding:16px}table{border-collapse:collapse}"
        "td,th{border:1px solid #bbb;padding:3px 6px}pre{white-space:pre-wrap}</style>"
        "<h1>Frozen review operating point: one-shot selection evaluation</h1>"
        f"<p>Freeze fingerprint {html.escape(results['freeze_fingerprint'])}. Unseen animals of the same dataset; "
        "simulated reviewers; not park validation.</p><h2>Criteria</h2><pre>"
        + html.escape(json.dumps(results["criteria"], indent=1)) + "</pre><h2>Simulations</h2>" + table
        + "<h2>W1 reference</h2><pre>" + html.escape(json.dumps(results["w1"], indent=1)) + "</pre>"
        "<h2>Breakdowns</h2><pre>" + html.escape(json.dumps(results["breakdowns"], indent=1)) + "</pre>"
        "<h2>Recognition</h2><pre>" + html.escape(json.dumps(results["recognition"], indent=1)) + "</pre></html>")
    return {p.name: file_digest(p) for p in sorted(out.iterdir()) if p.name != "manifest.json"}


def run(args):
    started = time.perf_counter()
    record = fr.load_record(args.frozen_config_dir)
    fr.verify_bindings(record, metadata=args.metadata, protocol_plan=args.protocol_plan,
                       raw_metadata=args.raw_metadata, trial_dir=args.trial_dir)
    out, cache = Path(args.out_dir).resolve(), Path(args.cache_dir).resolve()
    one_shot_check(out, record["fingerprint"], args.retry_of)
    frame, plan, binding = load_inputs(args.metadata, args.protocol_plan)
    selection, pool = selection_frame(frame, plan)
    if set(selection.identity.astype(str)) & set(record["bindings"]["calibration_identities"]):
        raise ValueError("Selection shares animals with the calibration data the models were fitted on.")
    protected = protected_inputs(frame, binding) + [Path(args.frozen_config_dir), Path(args.trial_dir),
                                                    Path(args.raw_metadata).resolve().parent]
    check_output_directory(out, protected)
    for value in protected + [out]:
        p = Path(value).resolve()
        if cache == p or cache.is_relative_to(p) or p.is_relative_to(cache):
            raise ValueError("Selection cache overlaps protected inputs or the report.")
    if cache == (ROOT / "data/czechlynx/cache").resolve():
        raise ValueError("Use a dedicated selection cache directory.")
    manifest = {"kind": fr.SELECTION_VERSION, "status": "running", "stage": "verification", "partition": "selection",
                "freeze_fingerprint": record["fingerprint"],
                "freeze_config_sha256": file_digest(Path(args.frozen_config_dir) / "config.json"),
                "retry_of": str(Path(args.retry_of).resolve()) if args.retry_of else None,
                "selection_images": len(selection), "selection_identities": int(selection.identity.nunique())}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "manifest.json", manifest)

    def stage(name):
        manifest["stage"] = name
        write_json(out / "manifest.json", manifest)
    try:
        verify_prepared(selection)
        stage("extraction")
        matrix, spec = extract_selection(selection, args.trial_dir, frame, plan, binding, cache,
                                         batch_size=args.batch_size, workers=args.workers)
        manifest["model_spec"] = {k: spec.get(k) for k in ("kind", "arm", "adaptation_epoch", "checkpoint_sha256",
                                                           "native_state_sha256", "weights_sha256")}
        if spec.get("native_state_sha256") != record["bindings"]["native_state_sha256"]:
            raise ValueError("Extraction did not use the frozen F1 weights.")
        stage("context")
        inputs = prepare_evaluation(selection.image_id.tolist(), matrix, selection)
        codes, n = inputs.identity_codes, len(inputs.image_ids)
        ctx, bundles = fr.build_context(inputs.frame, inputs.vectors, args.raw_metadata, record)
        stage("simulation")
        rows, sims = [], {}
        for tau in (fr.PRIMARY_TAU, fr.SECONDARY_TAU):
            for error in record["reviewer_errors"]:
                sim, row = fr.run_frozen(record, tau, error, bundles, codes, ctx, seed=args.seed, tag="selection")
                rows.append(row)
                sims[(tau, error)] = sim
                print(f"tau {tau}, reviewer error {error}: auto {row['automation']:.3f}, misassigned "
                      f"{100 * row['misassigned_photo_rate']:.2f}%, count error {row['count_error']}, "
                      f"{row['decisions']} decisions, {row['hours_per_1000_photos']:.2f} h/1000 "
                      f"({row['hours_per_1000_photos_fast_click']:.2f} fast)", flush=True)
        w1 = {}
        for error in record["reviewer_errors"]:
            _, w1[str(error)] = fr.w1_reference(inputs.frame, codes, ctx, error, seed=args.seed)
        primary = sims[(fr.PRIMARY_TAU, 0.0)]
        labels = primary.catalogue.entry_of.copy()
        identities = [str(i) for i in inputs.identities]
        sources = inputs.frame.source.astype(str).to_numpy()
        big = np.asarray([identities[c] == "lynx_287" for c in codes])
        breakdowns = {"without_lynx_287": fr.subset_quality(labels, primary.catalogue, codes, ~big),
                      "lynx_287_only": fr.subset_quality(labels, primary.catalogue, codes, big),
                      "by_source": {s: fr.subset_quality(labels, primary.catalogue, codes, sources == s)
                                    for s in sorted(set(sources))}}
        stage("report")
        results = {"freeze_fingerprint": record["fingerprint"], "population": {"images": n, "identities": len(identities),
                   "bundles": len(bundles)}, "simulations": rows, "w1": w1, "breakdowns": breakdowns,
                   "per_animal": fr.per_animal(labels, primary.catalogue, codes, identities),
                   "recognition": recognition(inputs.frame, inputs.vectors, pool),
                   "calibration_reference": record["calibration_reference"]}
        results["criteria"] = fr.criteria(rows, w1["0.0"], record["calibration_reference"], n)
        manifest["artifacts"] = write_report(out, results)
        fr.verify_bindings(record, metadata=args.metadata, protocol_plan=args.protocol_plan,
                           raw_metadata=args.raw_metadata, trial_dir=args.trial_dir)
        verify_prepared(selection)
        manifest.update(status="complete", stage="complete", criteria=plain(results["criteria"]),
                        seconds=time.perf_counter() - started)
        write_json(out / "manifest.json", manifest)
        print(json.dumps(plain(results["criteria"]), indent=1), flush=True)
        print(f"Report: {out / 'index.html'}", flush=True)
        return manifest
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", seconds=time.perf_counter() - started)
        write_json(out / "manifest.json", manifest)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("metadata", "protocol-plan", "raw-metadata", "frozen-config-dir", "trial-dir", "cache-dir", "out-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--retry-of", help="Earlier attempt that failed before simulation (document the retry).")
    for name, default in (("batch-size", 16), ("workers", 4), ("threads", 8), ("seed", 42)):
        parser.add_argument("--" + name, type=int, default=default)
    args = parser.parse_args(argv)
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
