#!/usr/bin/env python3
"""Stage B: DISK+LightGlue local evidence for a list of candidate edges (pilot or full graph).

Writes one row of scalar evidence per edge. Evidence is cached in a compact SQLite
store keyed by model/settings/input hashes, so interrupted runs resume. Matcher
errors are reported but never cached as negative evidence. No identities are read.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import importlib.metadata
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from reid_demo.benchmark import file_digest, fingerprint, write_json
from reid_demo.benchmark_masks import load_foreground_mask
from reid_demo.pairwise_matching import _normalize, geometric_evidence, select_correspondences


VERSION = "edge_local_evidence_v1"
MASK_CACHE = 512
AFFINE_THRESHOLD = 0.03  # Normalized units (max radius 1); fixed before the pilot, looser than legacy 0.005.
CODE = ("tools/extract_czechlynx_edge_local.py", "reid_demo/pairwise_matching.py", "reid_demo/benchmark_masks.py",
        "tools/evaluate_czechlynx.py", "constants.py", "feature_extraction.py")
COLUMNS = ("image_id_a", "image_id_b", "status", "raw_matches", "foreground_matches", "unique_foreground_matches",
           "matches", "confidence_mean", "homography_inliers", "homography_inlier_fraction",
           "homography_coverage_min", "affine_inliers", "seconds", "cached")


def affine_inliers(points0, points1, *, threshold=AFFINE_THRESHOLD, seed=42):
    """Partial-affine (similarity) RANSAC in the same per-image normalized coordinates as the homography."""
    import cv2
    from constants import MIN_MATCHES
    if len(points0) < max(MIN_MATCHES, 3):
        return 0
    cv2.setRNGSeed(int(seed) % (2**31 - 1))
    try:
        model, mask = cv2.estimateAffinePartial2D(
            _normalize(np.asarray(points0, np.float64)).reshape(-1, 1, 2),
            _normalize(np.asarray(points1, np.float64)).reshape(-1, 1, 2),
            method=cv2.RANSAC, ransacReprojThreshold=threshold, maxIters=2000, confidence=0.99)
    except cv2.error:
        return 0
    return int(mask.sum()) if model is not None and mask is not None else 0


class EvidenceStore:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS evidence (key TEXT PRIMARY KEY, value TEXT, digest TEXT)")

    def get(self, key):
        row = self.db.execute("SELECT value, digest FROM evidence WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        value = json.loads(row[0])
        if fingerprint(value) != row[1]:
            raise ValueError(f"Damaged edge evidence cache entry: {key}")
        return value

    def put(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO evidence VALUES (?, ?, ?)",
                        (key, json.dumps(value, sort_keys=True), fingerprint(value)))

    def close(self):
        self.db.commit()
        self.db.close()


def build_matcher(args, frame, cache_dir):
    """Neural imports happen only here."""
    import cv2
    import torch
    from tools.evaluate_czechlynx import LocalFeatures, local_feature_spec
    from reid_demo.pairwise_matching import DiskLightGlueMatcher
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    spec = local_feature_spec("disk", args, "masked")
    local = LocalFeatures(Path(cache_dir) / "local" / fingerprint(spec), frame, spec)
    matcher = DiskLightGlueMatcher(device, local, spec)
    environment = {"torch": torch.__version__, "opencv": cv2.__version__, "numpy": np.__version__,
                   "device": device, "cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                   "lightglue": importlib.metadata.version("lightglue")}
    return matcher, matcher.spec, environment, local


def load_edges(path, frame):
    edges = pd.read_csv(path, dtype={"image_id_a": str, "image_id_b": str})
    if not {"image_id_a", "image_id_b"} <= set(edges) or edges.empty:
        raise ValueError("Edge list needs image_id_a and image_id_b rows.")
    if (edges.image_id_a >= edges.image_id_b).any() or edges.duplicated(["image_id_a", "image_id_b"]).any():
        raise ValueError("Edges must be unique and ordered image_id_a < image_id_b.")
    unknown = (set(edges.image_id_a) | set(edges.image_id_b)) - set(frame.image_id)
    if unknown:
        raise ValueError(f"Edges reference images outside the metadata: {sorted(unknown)[:3]}")
    return edges[["image_id_a", "image_id_b"]].reset_index(drop=True)


def run(args):
    started = time.perf_counter()
    out = Path(args.out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory must be new or empty; resume comes from the evidence cache.")
    metadata = pd.read_csv(args.metadata, dtype={"image_id": str})
    edges = load_edges(args.edges, metadata)
    needed = set(edges.image_id_a) | set(edges.image_id_b)
    frame = metadata[metadata.image_id.isin(needed)].sort_values("image_id").reset_index(drop=True)
    rows = {r["image_id"]: r for r in frame.to_dict("records")}
    for row in rows.values():
        if file_digest(row["path"]) != row["image_sha256"]:
            raise ValueError(f"Image changed since preparation: {row['image_id']}")
    print(f"Validated {len(rows)} images for {len(edges)} edges.", flush=True)
    code = {name: file_digest(ROOT / name) for name in CODE}
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": 1, "kind": VERSION, "status": "running", "edges_sha256": file_digest(args.edges),
                "edges": len(edges), "images": len(rows), "code": code,
                "config": {k: v for k, v in vars(args).items() if k not in ("out_dir",)},
                "affine_threshold": AFFINE_THRESHOLD, "timing_seconds": {}}
    write_json(out / "manifest.json", manifest)
    store = local = None
    try:
        t = time.perf_counter()
        matcher, model_spec, environment, local = build_matcher(args, frame, args.cache_dir)
        manifest["timing_seconds"]["setup_and_local_extraction"] = time.perf_counter() - t
        manifest.update(model=model_spec, environment=environment)
        write_json(out / "manifest.json", manifest)
        settings = {"version": VERSION, "budget": args.max_matches, "seed": args.seed, "code": code,
                    "affine_threshold": AFFINE_THRESHOLD, "geometry": "legacy_normalized_homography_RANSAC",
                    "environment": environment}
        store = EvidenceStore(Path(args.cache_dir) / "edge_local_evidence.sqlite")
        masks, results = OrderedDict(), []
        t = time.perf_counter()
        for index, (a, b) in enumerate(zip(edges.image_id_a, edges.image_id_b)):
            row_a, row_b = rows[a], rows[b]
            key = fingerprint({"model": fingerprint(model_spec), "settings": settings,
                               "inputs": [[r["image_id"], r["image_sha256"], r.get("mask_sha256")]
                                          for r in (row_a, row_b)]})
            evidence = store.get(key)
            cached = evidence is not None
            if not cached:
                # Evict before loading so neither endpoint of this pair can be dropped.
                while len(masks) > MASK_CACHE:
                    masks.popitem(last=False)
                for rid, row in ((a, row_a), (b, row_b)):
                    if rid not in masks:
                        masks[rid] = load_foreground_mask(row)
                    masks.move_to_end(rid)
                pair_seed = int(fingerprint([a, b, args.seed])[:8], 16) % (2**31 - 1)
                start = time.perf_counter()
                try:
                    p0, p1, confidence = matcher.match(row_a, row_b, masks[a], masks[b])
                    p0, p1, confidence, counts = select_correspondences(
                        p0, p1, confidence, masks[a], masks[b], budget=args.max_matches)
                    geometry = geometric_evidence(p0, p1, confidence, masks[a], masks[b], seed=pair_seed)
                    evidence = {"status": geometry["status"], **counts, "matches": geometry["matches"],
                                "confidence_mean": geometry["confidence_mean"],
                                "homography_inliers": geometry["n_inliers"],
                                "homography_inlier_fraction": geometry["inlier_fraction"],
                                "homography_coverage_min": geometry["coverage_min"],
                                "affine_inliers": affine_inliers(p0, p1, seed=pair_seed)}
                except (RuntimeError, ValueError) as error:
                    evidence = {"status": "matcher_error", "error": f"{type(error).__name__}: {error}",
                                "raw_matches": 0, "foreground_matches": 0, "unique_foreground_matches": 0,
                                "matches": 0, "confidence_mean": None, "homography_inliers": 0,
                                "homography_inlier_fraction": 0.0, "homography_coverage_min": 0.0,
                                "affine_inliers": 0}
                evidence["seconds"] = time.perf_counter() - start
                if evidence["status"] != "matcher_error":
                    store.put(key, evidence)
            results.append({"image_id_a": a, "image_id_b": b, **evidence, "cached": cached})
            if (index + 1) % 2000 == 0:
                store.db.commit()
                rate = (index + 1) / (time.perf_counter() - t)
                print(f"{index + 1}/{len(edges)} edges ({rate:.1f}/s)", flush=True)
        manifest["timing_seconds"]["matching"] = time.perf_counter() - t
        table = pd.DataFrame(results)
        table[[c for c in COLUMNS if c in table] + [c for c in table if c not in COLUMNS]].to_csv(
            out / "local_evidence.csv", index=False)
        manifest["status_counts"] = table.status.value_counts().to_dict()
        manifest["cached_edges"] = int(table.cached.sum())
        manifest["local_feature_counts"] = getattr(local, "mask_counts", None)
        manifest["artifacts"] = {"local_evidence.csv": file_digest(out / "local_evidence.csv")}
        manifest["timing_seconds"]["total"] = time.perf_counter() - started
        manifest["status"] = "complete"
        write_json(out / "manifest.json", manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["timing_seconds"]["total"] = time.perf_counter() - started
        write_json(out / "manifest.json", manifest)
        raise
    finally:
        if store is not None:
            store.close()
        if local is not None and hasattr(local, "close"):
            local.close()
    print(f"Local evidence: {out / 'local_evidence.csv'}", flush=True)
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", required=True, help="CSV with image_id_a < image_id_b rows.")
    parser.add_argument("--metadata", required=True, help="Prepared masked metadata.csv.")
    parser.add_argument("--out-dir", required=True, help="New/empty output directory.")
    parser.add_argument("--cache-dir", default="data/czechlynx/cache_pairwise", help="DISK and evidence caches.")
    parser.add_argument("--max-keypoints", type=int, default=1024)
    parser.add_argument("--max-matches", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if min(args.max_keypoints, args.max_matches, args.threads) < 1 or args.seed < 0:
        parser.error("Budgets and threads must be positive and the seed nonnegative.")
    try:
        run(args)
        return 0
    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
