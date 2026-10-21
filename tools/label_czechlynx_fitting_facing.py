#!/usr/bin/env python3
"""Facing labels for the CzechLynx fitting pool: pose first, then an identity-cross-fitted classifier on
public MiewID vectors (extracted once into a new cache), else 'other'. Fitting pool only."""
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
import pandas as pd

from reid_demo.benchmark import file_digest, fingerprint, write_json
from reid_demo.benchmark_clustering import check_output_directory
from reid_demo.benchmark_input_audit import load_inputs, protected_inputs, verify_prepared
from reid_demo import benchmark_facing_adaptation as fm
from reid_demo import benchmark_miewid_adaptation as ad

VERSION = "czechlynx_fitting_facing_labels_v1"


def fitting_frame(metadata, protocol_plan):
    frame, plan, binding = load_inputs(metadata, protocol_plan)
    fitting = frame[frame.image_id.isin(plan["pools"]["fitting"]["image_ids"])].sort_values("image_id")
    if set(fitting["split-time_open"]) != {"train"} or len(fitting) != len(plan["pools"]["fitting"]["image_ids"]):
        raise ValueError("Fitting photos must be the complete official time_open train fitting pool.")
    return fitting.reset_index(drop=True), plan, binding


def extract_public(frame, cache_dir, *, batch_size=16, workers=2):
    """Pinned public MiewID vectors in a dedicated cache; extraction never sees identities."""
    import torch
    from reid_demo.benchmark_embeddings import load_miewid
    from reid_demo.benchmark_masks import MASKED_PREPROCESSING
    from tools.evaluate_czechlynx import VectorCache
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, info, _ = load_miewid(device)
    settings = ad.DEFAULT_SETTINGS
    spec = {**info, "weights_sha256": ad.native_state_digest(model.state_dict()), "input_mode": "masked",
            "masked_preprocessing": MASKED_PREPROCESSING, "expected_dimension": settings["embedding_dim"],
            "transform": "RGB; resize 440x440 bilinear; ImageNet normalization; L2", "amp": device.type == "cuda",
            "environment": ad.environment(), "purpose": "fitting facing labels",
            "mask_inventory_sha256": fingerprint(frame[["image_id", "mask_sha256"]].to_dict("records"))}
    cache_path = Path(cache_dir) / ("global_" + fingerprint(spec)[:20] + ".sqlite")
    cache = VectorCache(cache_path, spec)
    try:
        vectors, missing = {}, []
        for row in frame.to_dict("records"):
            vector = cache.get(row["image_id"], row["image_sha256"])
            if vector is None:
                missing.append(row)
            else:
                vectors[row["image_id"]] = vector
        print(f"Public fitting vectors: {len(vectors)} cached, {len(missing)} to extract", flush=True)
        for start in range(0, len(missing), 512):
            chunk = pd.DataFrame(missing[start:start + 512]).assign(identity="unlabelled")
            matrix = ad.extract_matrix(model, chunk, settings, device, batch_size=batch_size, workers=workers)
            for row, vector in zip(chunk.to_dict("records"), matrix):
                cache.put(row["image_id"], row["image_sha256"], vector)
                vectors[row["image_id"]] = vector
            cache.db.commit()
            print(f"Extracted {min(start + 512, len(missing))}/{len(missing)}", flush=True)
        return np.stack([vectors[r] for r in frame.image_id]), spec, cache_path
    finally:
        cache.close()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def run(args):
    started = time.perf_counter()
    from reid_demo import benchmark_flank_grouping as fg
    frame, plan, binding = fitting_frame(args.metadata, args.protocol_plan)
    out, cache_dir = Path(args.out_dir).resolve(), Path(args.cache_dir).resolve()
    protected = protected_inputs(frame, binding) + [Path(args.raw_metadata).resolve().parent]
    check_output_directory(out, protected)
    if cache_dir == (ROOT / "data/czechlynx/cache").resolve():
        raise ValueError("Use a new cache subdirectory, not the shared frozen cache root.")
    for p in [Path(v).resolve() for v in protected]:
        if cache_dir == p or cache_dir.is_relative_to(p) or p.is_relative_to(cache_dir):
            raise ValueError("The new cache must not overlap protected inputs.")
    if out == cache_dir or out.is_relative_to(cache_dir) or cache_dir.is_relative_to(out):
        raise ValueError("Labels and cache need separate directories.")
    if cache_dir.exists() and any(p.suffix != ".sqlite" for p in cache_dir.iterdir()):
        raise ValueError("Cache directory holds unexpected files.")
    raw_sha = file_digest(args.raw_metadata)
    manifest = {"kind": VERSION, "status": "running", "binding": binding, "fitting_images": len(frame),
                "raw_metadata": {"path": str(Path(args.raw_metadata).resolve()), "sha256": raw_sha},
                "seed": args.seed, "implementation": {n: file_digest(ROOT / n) for n in
                    ("tools/label_czechlynx_fitting_facing.py", "reid_demo/benchmark_facing_adaptation.py",
                     "reid_demo/benchmark_flank_grouping.py", "reid_demo/benchmark_miewid_adaptation.py")}}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "manifest.json", manifest)
    try:
        verify_prepared(frame)
        vectors, spec, cache_path = extract_public(frame, cache_dir, batch_size=args.batch_size, workers=args.workers)
        pose = fg.load_pose_facing(args.raw_metadata, frame)
        table, validation = fm.build_facing_labels(frame, vectors, pose, seed=args.seed)
        table = table[fm.LABEL_COLUMNS]
        table.to_csv(out / "labels.csv", index=False)
        inner = ad.build_inner_plan(frame.assign(), plan, seed=42, holdout_fraction=.2)
        roles = {name: set(r["image_ids"]) for name, r in inner["roles"].items()}
        summary = {"validation": validation,
                   "counts": {name: table[table.image_id.isin(ids)].groupby(["facing_source", "facing"]).size()
                              .rename("n").reset_index().to_dict("records") for name, ids in
                              [("all", set(table.image_id)), *roles.items()]},
                   "pose_labelled": int((table.facing_source == "pose").sum()),
                   "other_share": float((table.facing == -1).mean())}
        write_json(out / "summary.json", json.loads(json.dumps(summary, default=lambda o: o.item() if hasattr(o, "item") else str(o))))
        if file_digest(args.raw_metadata) != raw_sha:
            raise ValueError("Raw metadata changed during labelling.")
        verify_prepared(frame)
        manifest.update(status="complete", labels_sha256=file_digest(out / "labels.csv"),
                        summary_sha256=file_digest(out / "summary.json"), model_spec=spec,
                        cache_path=str(cache_path), cache_sha256=file_digest(cache_path),
                        seconds=time.perf_counter() - started)
        write_json(out / "manifest.json", manifest)
        print(json.dumps({"validation": validation, "pose_labelled": summary["pose_labelled"],
                          "other_share": summary["other_share"]}, indent=1, default=str), flush=True)
        return manifest
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", seconds=time.perf_counter() - started)
        write_json(out / "manifest.json", manifest)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("metadata", "protocol-plan", "raw-metadata", "cache-dir", "out-dir"):
        parser.add_argument("--" + name, required=True)
    for name, default in (("batch-size", 16), ("workers", 2), ("seed", 42)):
        parser.add_argument("--" + name, type=int, default=default)
    args = parser.parse_args(argv)
    try:
        run(args)
        return 0
    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
