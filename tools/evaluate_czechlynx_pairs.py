#!/usr/bin/env python3
"""Compare RoMa and DISK/LightGlue on saved calibration shortlist pairs."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reid_demo.benchmark import file_digest, fingerprint, write_json
from reid_demo.benchmark_masks import load_foreground_mask
from reid_demo.pairwise_benchmark import (
    EvidenceCache, load_calibration_reference, pair_cache_spec, render_matches,
    select_pairs, visual_pair_ids, write_report,
)
from reid_demo.pairwise_matching import (
    ROMATCH_VERSION, SAMPLING_POLICY, DiskLightGlueMatcher, RomaMatcher,
    geometric_evidence, select_correspondences,
)


def check_optional_dependency():
    try:
        version = importlib.metadata.version("romatch")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(f"Install the optional dependency: venv/bin/python -m pip install romatch=={ROMATCH_VERSION}") from error
    if version != ROMATCH_VERSION:
        raise RuntimeError(f"This experiment requires romatch=={ROMATCH_VERSION}; found {version}.")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--metadata", default="data/czechlynx/prepared_masked/masked/metadata.csv")
    result.add_argument("--protocol-plan", default="data/czechlynx/protocols/identity_pools_v1.json")
    result.add_argument("--reference-dir", default="evaluations/czechlynx/phase2_calibration_miewid_global")
    result.add_argument("--out-dir", default="evaluations/czechlynx/roma_pairwise_calibration")
    result.add_argument("--cache-dir", default="data/czechlynx/cache_pairwise")
    result.add_argument("--num-queries", type=int, default=100)
    result.add_argument("--top-k", type=int, default=10)
    result.add_argument("--max-matches", type=int, default=512)
    result.add_argument("--max-keypoints", type=int, default=1024)
    result.add_argument("--roma-min-confidence", type=float, default=0.05)
    result.add_argument("--roma-coarse-res", type=int, default=560)
    result.add_argument("--roma-upsample-res", type=int, default=864)
    result.add_argument("--visual-pairs", type=int, default=24)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--threads", type=int, default=4)
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--prepare-only", action="store_true", help="Validate inputs and save pair list, without neural imports.")
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if (min(args.num_queries, args.top_k, args.max_matches, args.max_keypoints,
            args.roma_coarse_res, args.roma_upsample_res, args.threads) < 1
            or args.roma_coarse_res % 14 or args.visual_pairs < 0
            or not 0 <= args.roma_min_confidence < 1 or args.seed < 0):
        cli.error("Invalid budget, resolution, confidence, seed or thread setting.")
    frame, gallery, query, predictions, groups, provenance = load_calibration_reference(
        args.metadata, args.protocol_plan, args.reference_dir)
    pairs, queries = select_pairs(gallery, query, predictions, groups,
                                  num_queries=args.num_queries, top_k=args.top_k, seed=args.seed)
    ids = {p[key] for p in pairs for key in ("query_id", "gallery_id")}
    needed = frame[frame.image_id.isin(ids)].sort_values("image_id").reset_index(drop=True)
    rows = {r["image_id"]: r for r in needed.to_dict("records")}
    print(f"Validating {len(rows)} segmented images for {len(pairs)} calibration pairs.", flush=True)
    for row in rows.values():
        if file_digest(row["path"]) != row["image_sha256"]:
            raise ValueError(f"Image changed since preparation: {row['image_id']}")
        load_foreground_mask(row)
    code_paths = ["tools/evaluate_czechlynx_pairs.py", "reid_demo/pairwise_matching.py",
                  "reid_demo/pairwise_benchmark.py", "reid_demo/benchmark_masks.py",
                  "reid_demo/benchmark_protocol.py", "reid_demo/benchmark.py",
                  "tools/evaluate_czechlynx.py", "constants.py", "feature_extraction.py"]
    code = {name: file_digest(ROOT / name) for name in code_paths}
    config = {k: v for k, v in vars(args).items() if k not in ("prepare_only", "out_dir", "cache_dir")}
    run_spec = {"schema": 1, "config": config, "reference": provenance,
                "pairs_sha256": fingerprint(pairs), "code": code}
    run_id = fingerprint(run_spec)
    out, cache = Path(args.out_dir), Path(args.cache_dir)
    if out.resolve() == Path(args.reference_dir).resolve():
        raise ValueError("Output directory must differ from the reference.")
    previous_path = out / "manifest.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    if previous and previous.get("run_fingerprint") != run_id:
        raise ValueError("Output belongs to another experiment; use a new --out-dir.")
    if out.exists() and any(out.iterdir()) and not previous:
        raise ValueError("Refusing a nonempty output directory without this experiment's manifest.")
    if args.prepare_only and previous.get("status") in ("complete", "complete_with_errors"):
        print(f"Already completed with this pair specification: {out / 'index.html'}")
        return
    out.mkdir(parents=True, exist_ok=True)
    manifest = {**run_spec, "run_fingerprint": run_id, "partition": "calibration",
                "status": "prepared" if args.prepare_only else "running",
                "models": dict(previous.get("models", {})), "setup_seconds": {}}
    if "environment" in previous:
        manifest["environment"] = previous["environment"]
    write_json(previous_path, manifest)
    write_json(out / "pairs.json", pairs)
    write_json(out / "queries.json", queries)
    print(f"Prepared {len(queries)} queries x {args.top_k} candidates. Pair list: {out / 'pairs.json'}", flush=True)
    if args.prepare_only:
        return
    local = matcher = None
    try:
        check_optional_dependency()
        import cv2
        import numpy as np
        import torch
        from threadpoolctl import threadpool_limits
        from tools.evaluate_czechlynx import LocalFeatures, local_feature_spec

        device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        device = "cpu" if device == "auto" else device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        torch.set_num_threads(args.threads)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        threadpool_limits(limits=args.threads)
        dependencies = {}
        for package in ("torchvision", "Pillow", "kornia", "timm", "einops"):
            try:
                dependencies[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                dependencies[package] = None
        environment = {"torch": torch.__version__, "opencv": cv2.__version__, "numpy": np.__version__,
                       "device": device, "cuda": torch.version.cuda,
                       "python": sys.version, "threads": args.threads, "dependencies": dependencies,
                       "cudnn": torch.backends.cudnn.version(),
                       "gpu": torch.cuda.get_device_name() if device == "cuda" else None}
        if previous.get("environment") is not None and previous["environment"] != environment:
            raise ValueError("Execution environment changed; choose a new output directory.")
        manifest["environment"] = environment
        manifest["disk_extraction_device"] = "cuda" if torch.cuda.is_available() else "cpu"
        settings = {"budget": args.max_matches, "sampling": SAMPLING_POLICY, "seed": args.seed,
                    "roma_min_confidence": args.roma_min_confidence, "environment": environment,
                    "implementation": code, "geometry": "legacy_normalized_homography_RANSAC"}
        cache.mkdir(parents=True, exist_ok=True)
        evidence_cache = EvidenceCache(cache / "pairs")
        visuals = visual_pair_ids(pairs, args.visual_pairs)
        visual_set = set(visuals)
        records = []
        # One neural matcher resident at a time so peak GPU allocations are interpretable.
        for name in ("disk_lightglue", "roma"):
            started = time.perf_counter()
            if name == "disk_lightglue":
                spec = local_feature_spec("disk", args, "masked")
                local = LocalFeatures(cache / "local" / fingerprint(spec), needed, spec)
                matcher = DiskLightGlueMatcher(device, local, spec)
            else:
                matcher = RomaMatcher(device, coarse_res=args.roma_coarse_res,
                                      upsample_res=args.roma_upsample_res)
            manifest["setup_seconds"][name] = time.perf_counter() - started
            model_spec = matcher.spec
            old_spec = previous.get("models", {}).get(name)
            if old_spec is not None and old_spec != model_spec:
                raise ValueError("Matcher weights/code changed; choose a new output directory.")
            manifest["models"][name] = model_spec
            write_json(previous_path, manifest)
            for index, pair in enumerate(pairs):
                qid, gid = pair["query_id"], pair["gallery_id"]
                row0, row1 = rows[qid], rows[gid]
                spec = pair_cache_spec({"name": name, "fingerprint": fingerprint(model_spec)},
                                       row0, row1, settings)
                evidence = evidence_cache.get(spec)
                cached = evidence is not None
                if not cached:
                    mask0, mask1 = load_foreground_mask(row0), load_foreground_mask(row1)
                    pair_seed = int(fingerprint([qid, gid, args.seed])[:8], 16) % (2**31 - 1)
                    torch.manual_seed(pair_seed)
                    if device == "cuda":
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    try:
                        p0, p1, confidence = matcher.match(row0, row1, mask0, mask1)
                        p0, p1, confidence, counts = select_correspondences(
                            p0, p1, confidence, mask0, mask1, budget=args.max_matches,
                            min_confidence=args.roma_min_confidence if name == "roma" else 0.0)
                        evidence = {**geometric_evidence(p0, p1, confidence, mask0, mask1, seed=pair_seed), **counts}
                    except (RuntimeError, ValueError) as error:
                        evidence = {"status": "matcher_error", "error": f"{type(error).__name__}: {error}",
                                    "points_query": [], "points_gallery": [], "inliers": [], "confidence": []}
                        if device == "cuda":
                            torch.cuda.empty_cache()
                    if device == "cuda":
                        torch.cuda.synchronize()
                    evidence["seconds"] = time.perf_counter() - start
                    evidence["peak_gpu_allocated_mb"] = (torch.cuda.max_memory_allocated() / 2**20
                                                         if device == "cuda" else None)
                    evidence_cache.put(spec, evidence)
                records.append({**pair, **{k: v for k, v in evidence.items()
                                          if k not in ("points_query", "points_gallery", "confidence", "inliers")},
                                "matcher": name, "cached": cached,
                                "evidence_cache_key": fingerprint(spec)})
                if (qid, gid) in visual_set:
                    filename = fingerprint([qid, gid, name])[:24] + ".png"
                    render_matches(out / "matches" / filename, row0, row1, evidence,
                                   f"{name} | {evidence['status']} | same identity: {pair['same_identity']}")
                if (index + 1) % 25 == 0 or index + 1 == len(pairs):
                    print(f"{name}: {index + 1}/{len(pairs)} pairs", flush=True)
            if local is not None:
                local.close()
                local = None
            matcher = None
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        write_report(out, records, queries, pairs, visuals)
        manifest["matcher_errors"] = sum(r["status"] == "matcher_error" for r in records)
        manifest["status"] = "complete_with_errors" if manifest["matcher_errors"] else "complete"
        manifest["cached_pairs"] = sum(r["cached"] for r in records)
        write_json(previous_path, manifest)
        print(f"{manifest['status']}: {out / 'index.html'}", flush=True)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        write_json(previous_path, manifest)
        raise
    finally:
        if local is not None:
            local.close()


if __name__ == "__main__":
    main()
