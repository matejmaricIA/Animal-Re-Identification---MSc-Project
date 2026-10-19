#!/usr/bin/env python3
"""Freeze the review operating point on calibration only (work item (c)): fit the W4 confidence,
P(match) and facing models on all of calibration, store them with frozen risk references and every
binding, then run an in-sample calibration sanity check (plumbing only, not evidence)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from reid_demo.benchmark import file_digest, write_json
from reid_demo.benchmark_clustering import check_output_directory, load_frozen_vectors
from reid_demo.benchmark_cluster_diagnostics import check_diagnostic_output, load_calibration, prepare_evaluation
from reid_demo import benchmark_frozen_review as fr
from reid_demo import benchmark_gated_matching as gm
from reid_demo import benchmark_nonmatch_pruning as nm
import tools.simulate_czechlynx_gated_matching as gated_cli


def calibration_reference(summary_path, n_images):
    summary = json.loads(Path(summary_path).read_text())
    row = next(r for r in summary["simulations"]
               if r["config"] == f"G1/defaults/err0.0/pmatch/eps{fr.EPSILON}/tau{fr.PRIMARY_TAU}")
    return {"source": str(Path(summary_path).resolve()), "sha256": file_digest(summary_path),
            "config": row["config"], "decisions": row["decisions"], "images": n_images,
            "decisions_per_1000_photos": row["decisions"] / (n_images / 1000),
            "automation": row["automation"], "misassigned_photo_rate": row["misassigned_photo_rate"],
            "count_error": row["count_error"], "note": "Identity-cross-fitted calibration result for F1."}


def run(args):
    started = time.perf_counter()
    from reid_demo import benchmark_flank_grouping as fg
    from reid_demo.benchmark_coverage_diagnostics import image_quality_features
    source = load_calibration(args.calibration_dir)
    out = Path(args.out_dir).resolve()
    check_diagnostic_output(out, source)
    check_output_directory(out, [Path(args.raw_metadata).resolve().parent, Path(args.trial_dir).resolve()])
    provenance = source.collection.provenance
    global_model = source.collection.reference["global_model"]
    if global_model.get("kind") != "facing_adapted_miewid_msv3" or global_model.get("arm") != "classes":
        raise ValueError("Freeze from the F1 (facing classes) calibration report.")
    if global_model.get("checkpoint_sha256") != file_digest(Path(args.trial_dir) / "best.pt"):
        raise ValueError("The calibration report was not produced by this trial's checkpoint.")
    manifest = {"kind": fr.FREEZE_VERSION, "status": "running", "partition": "calibration", "seed": args.seed}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "manifest.json", manifest)
    try:
        frame = source.collection.frame
        matrix, _ = load_frozen_vectors(frame, global_model, source.manifest["run_spec"]["cache_dir"])
        inputs = prepare_evaluation(frame.image_id.tolist(), matrix, frame, source.appearance)
        codes = inputs.identity_codes
        # Training records exactly as in (a)/(b): cross-fitted facing inside the context, perfect-review replay.
        ctx, bundles, _, records = gated_cli.prepare(args, source, inputs)
        confidence = fr.fit_logistic(records, list(gm.APPEARANCE_FEATURES) + list(gm.SPACETIME_FEATURES), "top1_correct")
        pair_records = nm.replay_pair_records(bundles, codes, nm.PairContext(ctx))
        pmatch = fr.fit_logistic(pair_records, list(nm.PAIR_COLUMNS), "match")
        pose = fg.load_pose_facing(args.raw_metadata, inputs.frame)
        facing = fr.FrozenFacing.fit(inputs.vectors, pose)
        reference = fr.risk_reference(image_quality_features(inputs.frame))
        bindings = {"metadata": provenance["metadata"], "metadata_sha256": file_digest(provenance["metadata"]),
                    "protocol_plan": provenance["protocol_plan"],
                    "protocol_plan_sha256": file_digest(provenance["protocol_plan"]),
                    "raw_metadata_sha256": file_digest(args.raw_metadata),
                    "trial_dir": str(Path(args.trial_dir).resolve()),
                    "trial_manifest_sha256": file_digest(Path(args.trial_dir) / "manifest.json"),
                    "checkpoint_sha256": file_digest(Path(args.trial_dir) / "best.pt"),
                    "native_state_sha256": global_model.get("native_state_sha256"),
                    "calibration_report_manifest_sha256": file_digest(source.directory / "manifest.json"),
                    "calibration_images": len(codes), "calibration_identities": sorted(map(str, inputs.identities)),
                    "training_records": {"confidence": len(records), "pmatch": len(pair_records),
                                         "facing_pose_labelled": int((~np.isnan(pose)).sum())},
                    "code": fr.code_hashes()}
        record = fr.build_record({"confidence": confidence.payload(), "pmatch": pmatch.payload(),
                                  "facing": facing.payload(), "risk_reference": reference}, bindings,
                                 calibration_reference(args.calibration_review_summary, len(codes)))
        write_json(out / "config.json", record)
        # In-sample sanity: the frozen pipeline (frozen facing/risk/models) on the data it was fitted on.
        sanity_ctx, sanity_bundles = fr.build_context(inputs.frame, inputs.vectors, args.raw_metadata, record)
        sanity = []
        for tau in (fr.PRIMARY_TAU, fr.SECONDARY_TAU):
            _, row = fr.run_frozen(record, tau, 0.0, sanity_bundles, codes, sanity_ctx, seed=args.seed, tag="sanity")
            sanity.append({k: row[k] for k in ("tau", "automation", "misassigned_photo_rate", "count_error",
                                                "decisions", "hours_per_1000_photos", "hours_per_1000_photos_fast_click")})
            print(f"In-sample sanity tau {tau}: {sanity[-1]}", flush=True)
        write_json(out / "sanity_in_sample.json", {"label": "IN-SAMPLE plumbing check on calibration; not evidence",
                                                   "cross_fitted_reference": record["calibration_reference"],
                                                   "rows": nm.plain(sanity)})
        manifest.update(status="complete", fingerprint=record["fingerprint"], config_sha256=file_digest(out / "config.json"),
                        sanity_sha256=file_digest(out / "sanity_in_sample.json"), seconds=time.perf_counter() - started)
        write_json(out / "manifest.json", manifest)
        print(f"Frozen configuration fingerprint: {record['fingerprint']}", flush=True)
        return manifest
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", seconds=time.perf_counter() - started)
        write_json(out / "manifest.json", manifest)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("calibration-dir", "calibration-review-summary", "raw-metadata", "trial-dir", "out-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
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
