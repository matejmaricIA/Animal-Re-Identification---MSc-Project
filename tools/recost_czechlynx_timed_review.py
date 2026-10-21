#!/usr/bin/env python3
"""Re-cost W1 vs the N2 candidate with measured reviewer times and errors (N3), and apply the pre-registered rule.

Reads a completed task set (tools/build_czechlynx_timed_review_tasks.py) and a matching-review app export.
Decision times come from the raw app events (the client's own values are a cross-check and a fallback).
Measured error rates are fed back into the sequential W1 and N2 simulations (5 seeds, region folds).
"""
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
import pandas as pd

from reid_demo.benchmark import file_digest, fingerprint, write_json
from reid_demo.benchmark_clustering import check_output_directory
from reid_demo.benchmark_gated_matching import misassigned_photos
from reid_demo.benchmark_review_simulation import fast_quality, plain
from reid_demo import benchmark_pruning_safeguards as ps
from reid_demo import benchmark_timed_review as tr
import tools.build_czechlynx_timed_review_tasks as builder
import tools.simulate_czechlynx_nonmatch_pruning as nm_cli

RECOST_VERSION = "calibration_timed_review_recost_v1"
P3_MISASSIGNED = 0.02


def records(frame):
    """JSON-safe rows (NaN becomes null)."""
    return frame.astype(object).where(frame.notna(), None).to_dict("records")


def load_task_set(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("kind") != tr.TASK_SET_KIND or manifest.get("status") != "complete":
        raise ValueError("The task set is not a complete timed-review task set.")
    for name, expected in manifest["artifacts"].items():
        if file_digest(directory / name) != expected:
            raise ValueError(f"Changed task-set artifact: {name}")
    artifacts = manifest["artifacts"]
    if manifest["task_set_fingerprint"] != fingerprint({k: artifacts[k] for k in
                                                        ("tasks.json", "images.json", "truth.json", "pre_registration.json")}):
        raise ValueError("Task-set fingerprint does not match its artifacts.")
    read = lambda name: json.loads((directory / name).read_text())
    return manifest, read("tasks.json"), read("truth.json")["tasks"], read("pre_registration.json")


def load_export(path, task_manifest):
    path = Path(path).resolve()
    json_path = path.with_suffix(".json") if path.suffix == ".csv" else path
    export = json.loads(json_path.read_text())
    if export.get("kind") != tr.EXPORT_KIND:
        raise ValueError("Not a matching-review export JSON.")
    if export.get("task_set_fingerprint") != task_manifest["task_set_fingerprint"]:
        raise ValueError("The export belongs to a different task set.")
    csv_path = json_path.with_suffix(".csv")
    if file_digest(csv_path) != export.get("csv_sha256"):
        raise ValueError("The export CSV differs from the hash recorded in its JSON.")
    return export, {"json": str(json_path), "json_sha256": file_digest(json_path), "csv": str(csv_path),
                    "csv_sha256": export["csv_sha256"]}


def statuses(tasks, export):
    completed = {}
    for c in export["completions"]:
        completed.setdefault(c["reviewer"], set()).add(c["task_id"])
    return {r: tr.completion_status(tasks["orders"], r, done) for r, done in sorted(completed.items())
            if done & set(tasks["main"])}


def secondary(table, counts, prereg):
    cap = tr.RULE["cap_seconds"]
    out = {"pooled_kind_formula": tr.pooled_kind_saving(table, prereg["saved_kinds"], cap=cap)}
    pops = {tuple(k.split("|")): v for k, v in prereg["populations"].items()}
    out["horvitz_thompson"] = tr.horvitz_thompson_saving(table, counts, pops, cap=cap)
    median = tr.cell_costs(table, counts, cap=cap, statistic="median")
    totals, saving = tr.totals_and_saving(median)
    out["median"] = {"totals_seconds": totals, "saving": saving}
    for alt in tr.RULE["sensitivity_caps"]:
        cells = tr.cell_costs(table, counts, cap=alt)
        totals, saving = tr.totals_and_saving(cells)
        out[f"cap_{'none' if alt is None else int(alt)}"] = {"totals_seconds": totals, "saving": saving}
    favourable = tr.cell_costs(table, counts, cap=cap, overrides=tr.w1_favourable_overrides(table, counts, cap=cap))
    totals, saving = tr.totals_and_saving(favourable)
    out["w1_favourable"] = {"totals_seconds": totals, "saving": saving}
    extra = {k: tr.RULE["extra_animal_sensitivity_seconds"] for k in counts if k[2] == "extra_animal"}
    totals, saving = tr.totals_and_saving(tr.cell_costs(table, counts, cap=cap, overrides=extra))
    out["extra_animal_25s"] = {"totals_seconds": totals, "saving": saving}
    return out


def per_kind_costs(cells, arm):
    """Average measured seconds per decision of each kind in an arm (weights: full-replay decisions)."""
    rows = cells[cells.arm == arm]
    return {kind: float(g.seconds.sum() / g.decisions.sum()) for kind, g in rows.groupby("kind") if g.decisions.sum()}


def resimulate(task_manifest, table, cells, seeds):
    """W1 and the N2 candidate at the measured error rates, one run per seed."""
    spec = task_manifest["run_spec"]
    data = builder.load_inputs(spec["calibration_dir"], spec["raw_metadata"]["path"])
    if (data.cache_sha256, data.raw_sha256, data.calibration_manifest_sha256) != (
            spec["cache_sha256"], spec["raw_metadata"]["sha256"], spec["calibration_manifest_sha256"]):
        raise ValueError("Calibration inputs changed since the task set was built.")
    prep = builder.prepare_candidate(data)
    units = builder.w1_units(data)
    rates = tr.measured_rates(table)
    costs = {arm: per_kind_costs(cells, arm) for arm in tr.ARMS}
    rows = []
    for seed in range(seeds):
        def reviewer(arm, kind, *extra):
            r = rates[(arm, kind)]
            return tr.MeasuredReviewer(r["false_same"], r["false_different"],
                                       seed=nm_cli._seed("measured", arm, kind, seed, *extra))
        kinds = {k: reviewer("N2", k) for k in tr.KINDS if k not in ("audit_glance", "count_audit_comparison",
                                                                     "extra_animal")}
        sim, audits = builder.simulate_candidate(prep, data.codes, 42, kind_reviewers=kinds,
                                                 glance=reviewer("N2", "audit_glance"),
                                                 count_auditor=lambda n: reviewer("N2", "count_audit_comparison", n))
        row = ps.summarize(sim, data.codes, data.n, len(prep.bundles), audits, data.regions)
        ledger = audits["ledger"].as_dict()
        rows.append({"arm": "N2", "seed": seed, "decisions": int(row["decisions"]),
                     "misassigned_photo_rate": row["misassigned_photo_rate"], "count_error": int(row["count_error"]),
                     "automation": row["automation"], "duplicate_merges": row["duplicate_merges"],
                     "wrong_duplicate_merges": row["wrong_duplicate_merges"], **ledger,
                     "measured_hours": sum(costs["N2"].get(k, 0) * v for k, v in ledger.items()
                                           if k != "count_audit_comparison") / 3600})
        cat, ledger, _ = tr.traced_w1(units, prep.ctx.scores, data.n,
                                      reviewers={k: reviewer("W1", k) for k in tr.FIRST_PASS})
        labels = cat.entry_of.copy()
        _, rate = misassigned_photos(labels, cat, data.codes)
        quality = fast_quality(labels, data.codes)
        w1 = {k: int(ledger.get(k, 0)) for k in tr.KINDS}
        rows.append({"arm": "W1", "seed": seed, "decisions": int(sum(ledger.values())), "misassigned_photo_rate": rate,
                     "count_error": int(quality["count_error"]), "automation": 0.0, **w1,
                     "measured_hours": sum(costs["W1"].get(k, 0) * v for k, v in w1.items()) / 3600})
    frame = pd.DataFrame(rows)
    count_limit = max(1, int(np.ceil(0.05 * len(set(data.codes.tolist())))))
    summary = {}
    for arm, g in frame.groupby("arm"):
        summary[arm] = {"median_decisions": float(g.decisions.median()),
                        "median_misassigned_photo_rate": float(g.misassigned_photo_rate.median()),
                        "median_abs_count_error": float(g.count_error.abs().median()),
                        "median_measured_hours": float(g.measured_hours.median())}
    n2 = summary["N2"]
    summary["p3_calibration_check"] = {
        "misassigned_limit": P3_MISASSIGNED, "count_limit": count_limit,
        "met": n2["median_misassigned_photo_rate"] <= P3_MISASSIGNED and n2["median_abs_count_error"] <= count_limit}
    summary["saving_at_measured_error"] = 1 - summary["N2"]["median_measured_hours"] / summary["W1"]["median_measured_hours"]
    summary["rates"] = {f"{a}|{k}": v for (a, k), v in rates.items()}
    return frame, summary


def write_report(out, results, cells, errors, table, resim):
    out = Path(out)
    write_json(out / "summary.json", plain(results))
    cells.to_csv(out / "per_kind_times.csv", index=False)
    errors.to_csv(out / "error_rates.csv", index=False)
    table.to_csv(out / "decisions.csv", index=False)
    if resim is not None:
        resim.to_csv(out / "resimulation.csv", index=False)
    rule = results["decision_rule"]
    fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else html.escape(json.dumps(v, default=str))
    table_html = lambda df: df.to_html(index=False, float_format=lambda v: f"{v:.4g}", na_rep="")
    body = (f"<p><b>Verdict: {html.escape(rule['verdict'])}</b> {html.escape(', '.join(rule['labels']))}. "
            f"Measured wall-clock saving (stratified, cap {tr.RULE['cap_seconds']:.0f} s): {fmt(rule['saving'])} "
            f"(95% CI {fmt(rule['ci95'][0])}–{fmt(rule['ci95'][1])}); per reviewer {fmt(rule['per_reviewer'])}.</p>"
            "<h2>Secondary estimates</h2><pre>" + html.escape(json.dumps(plain(results["secondary"]), indent=1))
            + "</pre><h2>Seconds per decision (cells)</h2>" + table_html(cells)
            + "<h2>Error rates</h2>" + table_html(errors)
            + "<h2>Agreement, second checks, unidentifiable</h2><pre>"
            + html.escape(json.dumps(plain({k: results[k] for k in ("agreement", "second_check_reliability",
                                                                     "unidentifiable", "timing_consistency")}),
                                     indent=1)) + "</pre>"
            + ("<h2>Re-simulation at measured error</h2><pre>" + html.escape(json.dumps(plain(results["resimulation"]),
                                                                                       indent=1)) + "</pre>"
               if results.get("resimulation") else "")
            + "<h2>Limitations</h2><ul>" + "".join(f"<li>{html.escape(n)}</li>" for n in tr.NOTES) + "</ul>")
    (out / "index.html").write_text("<!doctype html><html lang='en'><meta charset='utf-8'><title>CzechLynx timed review"
                                    "</title><style>body{font:14px system-ui;max-width:1500px;margin:30px auto;padding:"
                                    "16px}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:3px 6px}"
                                    "pre{white-space:pre-wrap}</style><h1>CzechLynx timed review (N3): measured "
                                    "re-costing</h1>" + body + "</html>")
    return {str(p.relative_to(out)): file_digest(p) for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "manifest.json"}


def run(args):
    started = time.perf_counter()
    task_manifest, tasks, truth, prereg = load_task_set(args.tasks_dir)
    export, export_binding = load_export(args.export, task_manifest)
    out = Path(args.out_dir).resolve()
    check_output_directory(out, [Path(args.tasks_dir), Path(export_binding["json"]).parent])
    spec = {"kind": RECOST_VERSION, "task_set_fingerprint": task_manifest["task_set_fingerprint"],
            "task_manifest_sha256": file_digest(Path(args.tasks_dir) / "manifest.json"), "export": export_binding,
            "implementation": tr.implementation_binding(), "rule": tr.RULE,
            "allow_incomplete": bool(args.allow_incomplete), "resimulation": not args.no_resimulation}
    manifest = {"schema": 1, "kind": RECOST_VERSION, "status": "running", "run_fingerprint": fingerprint(spec),
                "run_spec": spec, "timing_seconds": {}}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "manifest.json", manifest)
    try:
        status = statuses(tasks, export)
        if not status:
            raise ValueError("No reviewer has completed any main task.")
        incomplete = [r for r, s in status.items() if s == "incomplete"]
        if incomplete and not args.allow_incomplete:
            raise ValueError(f"Reviewers have not completed 8 or 12 blocks: {incomplete}. "
                             "Use --allow-incomplete only for an interim look (not the pre-registered evaluation).")
        reviewers = sorted(status)
        table = tr.decision_table(tasks["tasks"], truth, export)
        table = table[table.reviewer.isin(reviewers)].reset_index(drop=True)
        counts = tr.parse_counts(prereg["counts"])
        main = table[~table.practice]
        primary = tr.stratified_saving(main, counts)
        low, high, _ = tr.bootstrap_saving(main, counts)
        per_reviewer = {r: tr.stratified_saving(main[main.reviewer == r], counts)["saving"] for r in reviewers}
        errors = tr.error_rates(main)
        cluster = tr.cluster_bootstrap_rates(main)
        errors = errors.merge(cluster.drop(columns=["rate"]), on=["arm", "kind", "truth"], how="left")
        rule = tr.decision_rule(primary["saving"], (low, high), per_reviewer, tr.errors_reported(errors, main),
                                reduced="reduced" in status.values(), reviewers=len(reviewers))
        if incomplete:
            rule["verdict"] = "NOT EVALUATED (interim, incomplete reviewers)"
        events_rows = main[main.timing_source == "events"]
        results = {
            "status": status, "decision_rule": rule,
            "primary": {"totals_seconds": primary["totals_seconds"], "saving": primary["saving"],
                        "hours_per_1000_photos": {a: v / 3600 / (prereg["photos"] / 1000)
                                                  for a, v in primary["totals_seconds"].items()}},
            "secondary": secondary(main, counts, prereg),
            "errors_by_stratum": records(tr.error_rates(main, by=("arm", "stratum", "kind"))),
            "agreement": tr.agreement(main), "second_check_reliability": tr.second_check_reliability(main),
            "unidentifiable": tr.unidentifiable_rates(main),
            "timing_consistency": {
                "decisions": int(len(main)), "event_derived": int(len(events_rows)),
                "client_fallback": int((main.timing_source == "client").sum()),
                "max_abs_event_minus_client_seconds": float((events_rows.seconds_raw - events_rows.client_seconds)
                                                            .abs().max()) if len(events_rows) else None,
                "paused_decisions": int(main.paused.sum()), "undone_decisions": int((main.undos > 0).sum()),
                "capped_decisions": int((main.seconds_raw > tr.RULE["cap_seconds"]).sum())},
            "reviewer_profiles": {r: export["reviewers"].get(r, {}).get("profile") for r in reviewers},
            "notes": tr.NOTES}
        manifest["timing_seconds"]["analysis"] = time.perf_counter() - started
        resim = None
        if not args.no_resimulation:
            t = time.perf_counter()
            resim, results["resimulation"] = resimulate(task_manifest, main, primary["cells"],
                                                        tr.RULE["resimulation_seeds"])
            manifest["timing_seconds"]["resimulation"] = time.perf_counter() - t
        manifest["artifacts"] = write_report(out, results, primary["cells"], errors, table, resim)
        manifest["decision_rule"] = plain(rule)
        manifest["timing_seconds"]["total"] = time.perf_counter() - started
        manifest["status"] = "complete"
        write_json(out / "manifest.json", plain(manifest))
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["timing_seconds"]["total"] = time.perf_counter() - started
        write_json(out / "manifest.json", plain(manifest))
        raise
    print(json.dumps(plain(manifest["decision_rule"]), indent=1), flush=True)
    print(f"Report: {out / 'index.html'}", flush=True)
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", required=True, help="Completed task set from build_czechlynx_timed_review_tasks.py.")
    parser.add_argument("--export", required=True, help="App export CSV (its adjacent JSON is read) or the JSON itself.")
    parser.add_argument("--out-dir", required=True, help="New/empty output directory.")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Interim look with unfinished reviewers; the verdict is then not evaluated.")
    parser.add_argument("--no-resimulation", action="store_true", help="Skip the measured-error simulations.")
    parser.add_argument("--threads", type=int, default=4)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("Threads must be positive.")
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
