#!/usr/bin/env python3
"""Build the N3 timed-review task set (W1 vs the N2 candidate) from calibration replays. CPU only, no inference.

The perfect-reviewer replays must reproduce the saved N2 candidate row and the saved W1 split exactly,
or nothing is written. Outputs: tasks.json and images.json (read by the matching-review app),
truth.json (evaluation only; the app never opens it), strata_counts.csv, pre_registration.json,
contact_sheet.html (practice images only) and manifest.json.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import html
import io
import json
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace
import zlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from reid_demo.benchmark import file_digest, fingerprint, write_json
from reid_demo.benchmark_clustering import check_output_directory, load_frozen_vectors
from reid_demo.benchmark_cluster_diagnostics import check_diagnostic_output, load_calibration, prepare_evaluation
from reid_demo.benchmark_review_simulation import Reviewer, bundle_keys, ordered_units, simulate_catalogue
from reid_demo import benchmark_gated_matching as gm
from reid_demo import benchmark_nonmatch_pruning as nm
from reid_demo import benchmark_pruning_safeguards as ps
from reid_demo import benchmark_timed_review as tr
import tools.simulate_czechlynx_nonmatch_pruning as nm_cli
import tools.simulate_czechlynx_pruning_safeguards as ps_cli

DEFAULT_REFERENCE_DIR = "evaluations/czechlynx/pruning_safeguards_f1_v1"
DEFAULT_W1_REFERENCE_DIR = "evaluations/czechlynx/nonmatch_pruning_facing_classes_v1"
IDENTITY_BYTES = re.compile(rb"lynx[_-]?\d", re.IGNORECASE)
IDENTITY_TEXT = re.compile(r"lynx[_-]?\d|clx_[0-9a-f]|CzechLynx/|\.jpe?g|\.png", re.IGNORECASE)
CHECK_FIELDS = ps.REFERENCE_FIELDS + tuple(k for k in tr.KINDS if k not in ps.REFERENCE_FIELDS)


# --- Inputs, saved references and the candidate replay (shared with the re-costing tool) ------

def load_inputs(calibration_dir, raw_metadata):
    """Calibration inputs exactly as tools/simulate_czechlynx_pruning_safeguards.py loads them."""
    source = load_calibration(calibration_dir)
    cache_dir = Path(source.manifest["run_spec"]["cache_dir"]).resolve()
    cache_path = Path(source.manifest["vector_cache"]["path"]).resolve()
    if cache_path != cache_dir / f"global_{source.collection.binding['vector_cache_spec_fingerprint'][:20]}.sqlite":
        raise ValueError("Calibration vector cache path differs from its source/model contract.")
    frame = source.collection.frame
    matrix, _ = load_frozen_vectors(frame, source.collection.reference["global_model"], cache_dir)
    inputs = prepare_evaluation(frame.image_id.tolist(), matrix, frame, source.appearance)
    return SimpleNamespace(
        source=source, inputs=inputs, codes=np.asarray(inputs.identity_codes), n=len(inputs.image_ids),
        regions=ps.region_of_sources(inputs.frame.source.astype(str).tolist()), cache_path=cache_path,
        cache_sha256=file_digest(cache_path), raw_metadata=str(Path(raw_metadata).resolve()),
        raw_sha256=file_digest(raw_metadata),
        calibration_manifest_sha256=file_digest(Path(calibration_dir) / "manifest.json"),
        matrix_sha256=hashlib.sha256(memoryview(matrix).cast("B")).hexdigest())


def load_saved_row(directory, config, data):
    """One saved simulation row; its run must be complete and bind the same vectors and calibration."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    spec = manifest.get("run_spec", {})
    if manifest.get("status") != "complete":
        raise ValueError(f"Saved run is not complete: {directory}")
    if spec.get("cache_sha256") != data.cache_sha256 or \
            spec.get("calibration_manifest_sha256") != data.calibration_manifest_sha256:
        raise ValueError(f"Saved run {directory} used different vectors or a different calibration.")
    table = pd.read_csv(directory / "simulations.csv")
    match = table[table.config == config]
    if len(match) != 1:
        raise ValueError(f"Saved row missing or duplicated: {config} in {directory}")
    row = {k: (v.item() if hasattr(v, "item") else v) for k, v in match.iloc[0].items()}
    binding = {"dir": str(directory), "config": config, "manifest_sha256": file_digest(directory / "manifest.json"),
               "simulations_sha256": file_digest(directory / "simulations.csv"),
               "seed": spec.get("seed"), "threads": spec.get("threads")}
    return row, manifest, binding


def prepare_candidate(data):
    """Region-fold context, W4 confidence models and P(match) scorers, as in the saved N2 run."""
    ctx, bundles, identity_fold, records = ps_cli.prepare_region(SimpleNamespace(raw_metadata=data.raw_metadata),
                                                                 data.inputs)
    pctx = nm.PairContext(ctx)
    conf_models = gm.fit_fold_models(records, identity_fold, ps_cli.CONF_COLUMNS)
    pair_records = nm.replay_pair_records(bundles, data.codes, pctx)
    pair_models = nm.fit_pair_models(pair_records, identity_fold, ps_cli.PAIR_COLUMNS)
    scorers = {fold: nm.scorer(model) for fold, model in pair_models.items()}
    return SimpleNamespace(ctx=ctx, bundles=bundles, identity_fold=identity_fold, pctx=pctx,
                           conf_models=conf_models, scorers=scorers)


def simulate_candidate(prep, codes, seed, *, kind_reviewers=None, glance=None, count_auditor=None):
    """``ps_cli.simulate`` for the candidate with the traced class. With no measured reviewers it uses the
    saved run's seeded perfect reviewers, so its ledger must equal the saved row."""
    config, tag = dict(tr.CANDIDATE), tr.CANDIDATE_TAG
    error = config["reviewer_error"]
    sim = tr.TracedSafeguardSimulation(
        prep.bundles, codes, prep.pctx, prep.conf_models, ps_cli.CONF_COLUMNS, prep.scorers, ps_cli.PAIR_COLUMNS,
        prep.identity_fold, config, Reviewer(error, nm_cli._seed("expert", tag, seed)),
        Reviewer(error, nm_cli._seed("second", tag, seed)), kind_reviewers=kind_reviewers).run()
    ledger = sim.ledger.copy()
    auto = gm.audit(sim.auto_log, nm.AUTO_AUDIT, glance or Reviewer(error, nm_cli._seed("audit", tag, seed)),
                    ledger, seed=nm_cli._seed("sample", tag, seed))
    count = {}
    for n in nm.COUNT_AUDITS:
        extra = ledger.copy()
        auditor = count_auditor(n) if count_auditor else Reviewer(error, nm_cli._seed("count_audit", tag, n, seed))
        result = nm.count_audit(sim.catalogue, n, auditor, extra, prep.ctx.scores, codes,
                                seed=nm_cli._seed("count_sample", tag, n, seed))
        count[n] = {**result, "ledger": extra}
    return sim, {"ledger": ledger, "auto": auto, "count": count}


def w1_units(data):
    keys = bundle_keys(data.inputs.frame, "trap_day")
    return ordered_units(data.inputs.frame, data.codes, keys)


def compare(got, want, fields):
    problems = []
    for field in fields:
        g, w = got.get(field), want.get(field)
        same = abs(float(g) - float(w)) < 1e-9 if isinstance(w, float) or isinstance(g, float) else g == w
        if not same:
            problems.append(f"{field}: {g} vs saved {w}")
    return problems


# --- Images and leak checks -----------------------------------------------------------------

def readable_metadata(data):
    """Parts of an image file a person could read as text; compressed pixel data is excluded.

    JPEG: every APPn/COM marker segment (byte stuffing means FF E0-EF/FE never occur inside
    entropy-coded data, so progressive JPEGs are covered) plus any bytes after the last EOI.
    PNG: tEXt, iTXt and eXIf chunks, zTXt (and compressed iTXt) decompressed. Other formats: the whole file."""
    if data.startswith(b"\xff\xd8"):
        parts = []
        for m in re.finditer(rb"\xff([\xe0-\xef\xfe])", data):
            length = int.from_bytes(data[m.end():m.end() + 2], "big")
            parts.append(data[m.end() + 2:m.end() + length])
        end = data.rfind(b"\xff\xd9")
        parts.append(data[end + 2:] if end >= 0 else b"")
        return b"\n".join(parts)
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        parts, i = [], 8
        while i + 8 <= len(data):
            length, kind = int.from_bytes(data[i:i + 4], "big"), data[i + 4:i + 8]
            body = data[i + 8:i + 8 + length]
            if kind in (b"tEXt", b"eXIf"):
                parts.append(body)
            elif kind == b"zTXt":
                parts.append(zlib.decompress(body.split(b"\0", 1)[1][1:]))
            elif kind == b"iTXt":
                keyword, rest = body.split(b"\0", 1)
                text = rest[2:].split(b"\0", 2)[-1]
                parts.append(keyword + b"\0" + (zlib.decompress(text) if rest[:1] == b"\1" else text))
            i += 12 + length
        return b"\n".join(parts)
    return data


def image_records(data, images_root, indices, salt):
    """token -> served files. Hashes are verified and bytes scanned for identity strings."""
    frame = data.inputs.frame
    root = Path(images_root).resolve()
    out, tokens = {}, {}
    for i in sorted(indices):
        row = frame.iloc[i]
        original = (root / str(row["source_path"])).resolve()
        masked = Path(str(row["path"])).resolve()
        if not original.is_relative_to(root):
            raise ValueError("Original image path escapes the images root.")
        for path, expected in ((original, row["source_sha256"]), (masked, row["image_sha256"])):
            data_bytes = path.read_bytes()
            if hashlib.sha256(data_bytes).hexdigest() != expected:
                raise ValueError(f"Image bytes differ from the prepared metadata (image {i}).")
            if IDENTITY_BYTES.search(readable_metadata(data_bytes)):
                raise ValueError(f"An identity string is embedded in a served image file (image {i}).")
        token = tr.image_token(salt, row["image_id"])
        if token in out:
            raise ValueError("Image token collision.")
        tokens[i] = token
        out[token] = {"original": {"path": str(original), "sha256": str(row["source_sha256"])},
                      "masked": {"path": str(masked), "sha256": str(row["image_sha256"])}}
    return out, tokens


def check_browser_text(text, identities):
    if IDENTITY_TEXT.search(text):
        raise ValueError("Browser task payloads contain an identity label, image ID or file name.")
    for name in identities:
        if name and name in text:
            raise ValueError("Browser task payloads contain an identity label.")


def contact_sheet(path, images, practice_indices, limit=40):
    """Practice images only (original and masked thumbnails), embedded; no paths or labels."""
    from PIL import Image
    cells = []
    for i in sorted(practice_indices)[:limit]:
        pair = []
        for variant in ("original", "masked"):
            with Image.open(images[i][variant]["path"]) as im:
                im = im.convert("RGB")
                im.thumbnail((220, 220))
                buffer = io.BytesIO()
                im.save(buffer, "JPEG", quality=80)
            pair.append(f"<img alt='{variant}' src='data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}'>")
        cells.append("<figure>" + "".join(pair) + "</figure>")
    Path(path).write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>Timed review practice images</title><style>"
        "body{font:14px system-ui;margin:24px}figure{display:inline-flex;gap:4px;margin:6px;padding:4px;"
        "border:1px solid #ccc}img{max-height:220px}</style><h1>Practice images: original | masked</h1><p>"
        "Check that both variants render and show no identity overlays. Only practice-task images are shown, "
        "so the main tasks stay unseen.</p>" + "".join(cells) + "</html>")


# --- Build ----------------------------------------------------------------------------------

def run(args):
    started = time.perf_counter()
    data = load_inputs(args.calibration_dir, args.raw_metadata)
    out = Path(args.out_dir).resolve()
    images_root = Path(args.images_root or Path(args.raw_metadata).parent).resolve()
    check_diagnostic_output(out, data.source)
    check_output_directory(out, [images_root, Path(args.reference_dir).resolve(), Path(args.w1_reference_dir).resolve()])
    n2_saved, n2_manifest, n2_binding = load_saved_row(args.reference_dir, tr.CANDIDATE_TAG, data)
    w1_saved, _, w1_binding = load_saved_row(args.w1_reference_dir, tr.W1_REFERENCE_TAG, data)
    w1_live = n2_manifest.get("reference_check", {}).get("w1_live_decisions")
    if n2_binding["seed"] != args.seed or n2_binding["threads"] != args.threads:
        raise ValueError(f"Use the saved run's seed/threads for an exact reference: {n2_binding}.")
    spec = {"kind": tr.VERSION, "partition": "calibration", "binding": data.source.binding,
            "implementation": tr.implementation_binding(), "calibration_dir": str(Path(args.calibration_dir).resolve()),
            "calibration_manifest_sha256": data.calibration_manifest_sha256,
            "raw_metadata": {"path": data.raw_metadata, "sha256": data.raw_sha256}, "images_root": str(images_root),
            "cache_path": str(data.cache_path), "cache_sha256": data.cache_sha256,
            "references": {"n2": n2_binding, "w1": w1_binding, "w1_live_decisions": w1_live},
            "seed": args.seed, "threads": args.threads}
    manifest = {"schema": 1, "kind": tr.TASK_SET_KIND, "status": "running", "partition": "calibration",
                "run_fingerprint": fingerprint(spec), "run_spec": spec, "timing_seconds": {}, "notes": tr.NOTES}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "manifest.json", manifest)
    timer = time.perf_counter
    try:
        t = timer()
        prep = prepare_candidate(data)
        sim, audits = simulate_candidate(prep, data.codes, args.seed)
        row = ps.summarize(sim, data.codes, data.n, len(prep.bundles), audits, data.regions)
        n2_problems = compare(row, n2_saved, CHECK_FIELDS)
        manifest["timing_seconds"]["n2_replay"] = timer() - t

        t = timer()
        units = w1_units(data)
        w1_cat, w1_ledger, w1_parts = tr.traced_w1(units, prep.ctx.scores, data.n)
        labels, effort = simulate_catalogue(units, prep.ctx.scores, data.n, Reviewer(0.0), ranked=True)
        w1_row = {**{k: int(w1_ledger.get(k, 0)) for k in tr.KINDS}, "decisions": int(sum(w1_ledger.values()))}
        w1_problems = compare(w1_row, w1_saved, ("rank1_comparison", "further_comparison", "extra_animal", "decisions"))
        if w1_row["decisions"] != w1_live or effort.decisions != w1_live:
            w1_problems.append(f"W1 decisions {w1_row['decisions']} / {effort.decisions} vs live {w1_live}")
        if not np.array_equal(labels, w1_cat.entry_of):
            w1_problems.append("Traced W1 labels differ from simulate_catalogue.")
        manifest["timing_seconds"]["w1_replay"] = timer() - t

        glances = int(audits["ledger"].counts.get("audit_glance", 0))
        counts = tr.decision_counts(w1_parts, w1_ledger, sim, glances)
        for arm, saved in (("N2", n2_saved), ("W1", w1_saved)):
            totals = tr.kind_totals(counts, arm)
            for kind in tr.KINDS:
                if kind != "count_audit_comparison" and totals[kind] != int(saved.get(kind, 0) or 0):
                    (n2_problems if arm == "N2" else w1_problems).append(
                        f"strata sum {arm} {kind}: {totals[kind]} vs saved {saved.get(kind)}")
        manifest["reference_check"] = {"n2_problems": n2_problems, "w1_problems": w1_problems,
                                       "passed": not n2_problems and not w1_problems,
                                       "n2_fields": list(CHECK_FIELDS), "w1_live_decisions": w1_live}
        write_json(out / "manifest.json", manifest)
        print(f"Reference check: {'passed' if manifest['reference_check']['passed'] else 'FAILED'} "
              f"{n2_problems + w1_problems}", flush=True)
        if not manifest["reference_check"]["passed"]:
            raise ValueError("The traced replays do not reproduce the saved N2 and W1 rows.")

        t = timer()
        frame = data.inputs.frame
        dates = frame.timestamp.astype(str).tolist()
        cameras = frame.trap_id.astype(str).tolist()
        candidates = tr.candidate_units(w1_parts, sim, data.codes, dates)
        rng = np.random.default_rng(args.seed)
        chosen, shortfalls, used = tr.sample_units(candidates, tr.ALLOCATION, rng)
        practice, practice_shortfalls = tr.sample_practice(candidates, tr.PRACTICE, rng, used)
        main = [(key, u) for key, us in chosen.items() for u in us]
        main = [main[int(i)] for i in rng.permutation(len(main))]
        practice_list = [(key, u) for key, us in practice.items() for u in us]
        practice_list = [practice_list[int(i)] for i in rng.permutation(len(practice_list))]
        catalogues = {"W1": w1_cat, "N2": sim.catalogue}
        salt = fingerprint({"run": manifest["run_fingerprint"], "purpose": "image tokens"})

        def photo(i):
            return {"image": int(i), "date": dates[i], "camera": cameras[i]}

        payloads, truth, records, used_images, practice_images = {}, {}, [], set(), set()
        for prefix, items, is_practice in (("t", main, False), ("p", practice_list, True)):
            for k, (key, unit) in enumerate(items, 1):
                task_id = f"{prefix}{k:03d}"
                payload, info, images = tr.build_task(task_id, unit, catalogues[unit["arm"]], prep.ctx.scores, photo,
                                                      practice=is_practice)
                info["stratum_key"] = f"{key[1]}"
                payloads[task_id], truth[task_id] = payload, info
                used_images |= images
                if is_practice:
                    practice_images |= images
                else:
                    records.append({"task_id": task_id, "arm": unit["arm"], "stratum_key": key[1]})
        images, tokens = image_records(data, images_root, used_images, salt)

        def tokenize(p):
            return {"token": tokens[p["image"]], "date": p["date"], "camera": p["camera"]}

        for payload in payloads.values():
            payload["query"] = [tokenize(p) for p in payload["query"]]
            for screen in payload["screens"]:
                screen["exemplars"] = [tokenize(p) for p in screen["exemplars"]]
                screen["more"] = [tokenize(p) for p in screen["more"]]
        blocks = tr.make_blocks(records, rng)
        practice_ids = [f"p{k:03d}" for k in range(1, len(practice_list) + 1)]
        orders = tr.reviewer_orders(blocks, practice_ids, args.seed)
        tasks_doc = {"schema": 1, "kind": tr.TASK_SET_KIND, "tasks": payloads, "practice": practice_ids,
                     "main": [r["task_id"] for r in records], "orders": orders,
                     "settings": {"idle_pause_seconds": tr.RULE["idle_pause_seconds"],
                                  "block_size": max(len(b) for arm in blocks.values() for b in arm),
                                  "blocks": len(tr.ARM_PATTERN), "early_stop_blocks": tr.RULE["early_stop_blocks"],
                                  "break_seconds": 120, "blocks_per_session": 3}}
        tasks_text = json.dumps(tasks_doc, indent=1, sort_keys=True)
        check_browser_text(tasks_text, data.inputs.identities)
        (out / "tasks.json").write_text(tasks_text + "\n")
        write_json(out / "images.json", {"schema": 1, "kind": tr.TASK_SET_KIND, "images": images})
        image_ids = data.inputs.image_ids
        for info in truth.values():
            info["query_image_ids"] = [image_ids[i] for i in info.pop("query_images")]
            info["candidate_image_ids"] = [[image_ids[i] for i in c] for c in info.pop("candidate_images")]
        write_json(out / "truth.json", {"schema": 1, "kind": tr.TASK_SET_KIND, "tasks": truth})
        strata = pd.DataFrame([{"arm": a, "stratum": s, "kind": k, "decisions": n} for (a, s, k), n in counts.items()])
        strata.to_csv(out / "strata_counts.csv", index=False)
        pops = tr.populations(w1_parts, sim, glances)
        sampled = {k: int(v) for k, v in sorted(Counter(f"{r['arm']}|{r['stratum_key']}" for r in records).items())}
        prereg = {"schema": 1, "kind": tr.VERSION, "date": time.strftime("%Y-%m-%d"),
                  "rule": tr.RULE, "candidate_tag": tr.CANDIDATE_TAG, "w1_reference_tag": tr.W1_REFERENCE_TAG,
                  "counts": {tr._cell_key(*k): v for k, v in counts.items()},
                  "populations": {f"{a}|{s}": v for (a, s), v in pops.items()},
                  "saved_kinds": {"N2": {k: int(n2_saved.get(k, 0) or 0) for k in tr.KINDS},
                                  "W1": {k: int(w1_saved.get(k, 0) or 0) for k in tr.KINDS}},
                  "saved_decisions": {"N2": int(n2_saved["decisions"]), "W1": int(w1_live)},
                  "allocation": {f"{a}|{s}": n for (a, s), n in tr.ALLOCATION.items()},
                  "sampled": sampled, "shortfalls": shortfalls, "practice_shortfalls": practice_shortfalls,
                  "photos": data.n, "implementation": tr.implementation_binding(),
                  "artifacts": {name: file_digest(out / name) for name in ("tasks.json", "images.json", "truth.json")}}
        write_json(out / "pre_registration.json", prereg)
        contact_sheet(out / "contact_sheet.html", {i: images[tokens[i]] for i in practice_images}, practice_images)
        manifest["timing_seconds"]["tasks"] = timer() - t
        artifacts = {p.name: file_digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "manifest.json"}
        manifest["artifacts"] = artifacts
        manifest["task_set_fingerprint"] = fingerprint({k: artifacts[k] for k in
                                                        ("tasks.json", "images.json", "truth.json", "pre_registration.json")})
        manifest["summary"] = {"main_tasks": len(records), "practice_tasks": len(practice_ids),
                               "served_images": len(images), "sampled": sampled, "shortfalls": shortfalls,
                               "practice_shortfalls": practice_shortfalls,
                               "n2_decisions": int(row["decisions"]), "w1_decisions": int(w1_row["decisions"])}
        if file_digest(data.cache_path) != data.cache_sha256 or file_digest(args.raw_metadata) != data.raw_sha256 \
                or file_digest(Path(args.calibration_dir) / "manifest.json") != data.calibration_manifest_sha256:
            raise ValueError("Inputs changed during the build; use a new output directory.")
        manifest["timing_seconds"]["total"] = timer() - started
        manifest["status"] = "complete"
        write_json(out / "manifest.json", json.loads(json.dumps(manifest, default=str)))
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["timing_seconds"]["total"] = timer() - started
        write_json(out / "manifest.json", json.loads(json.dumps(manifest, default=str)))
        raise
    print(json.dumps(manifest["summary"], indent=1, default=str), flush=True)
    print(f"Task set fingerprint: {manifest['task_set_fingerprint']}", flush=True)
    print(f"Visual check (practice images only): {html.escape(str(out / 'contact_sheet.html'))}", flush=True)
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", required=True, help="Completed full-collection calibration report (F1).")
    parser.add_argument("--raw-metadata", required=True, help="Raw CzechLynx metadata (pose keypoints, coordinates).")
    parser.add_argument("--out-dir", required=True, help="New/empty output directory.")
    parser.add_argument("--reference-dir", default=DEFAULT_REFERENCE_DIR,
                        help="Completed N2 run holding the candidate row and the live W1 count.")
    parser.add_argument("--w1-reference-dir", default=DEFAULT_W1_REFERENCE_DIR,
                        help="Completed F1 non-match pruning run holding the W1 rank-1/further split.")
    parser.add_argument("--images-root", default=None, help="Root of the raw images (default: raw metadata folder).")
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
