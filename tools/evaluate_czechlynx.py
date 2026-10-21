#!/usr/bin/env python3
"""Reproducible CzechLynx retrieval and bounded clustering benchmark."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sqlite3
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from reid_demo.benchmark import (
    SIGNALS, clustering_metrics, file_digest, fingerprint, make_protocol,
    normalized, rank_queries, rejection_threshold, retrieval_metrics,
    sample_encounters, write_json,
)
from reid_demo.benchmark_masks import (
    MASKED_PREPROCESSING, MASK_KEYPOINT_POLICY, filter_foreground_features,
    load_foreground_mask, local_input_hash,
)
from reid_demo.benchmark_gv_masks import ForegroundFeatures, prepare_gv_masks
from reid_demo.benchmark_embeddings import GLOBAL_MODELS, load_global_model, validate_global_options
from reid_demo.benchmark_comparison import load_global_reference, compare_global_predictions
from reid_demo.benchmark_protocol import (
    load_protocol, protocol_frames, protocol_binding, validate_calibration,
)
from reid_demo.benchmark_diagnostics import appearance_measurements, build_diagnostics


LOCAL_METHODS = ("disk", "superpoint", "aliked")
FISHER_COMBINATION = "weighted_sum_l2_v1"


def resolve_methods(methods):
    """Match the thesis ensemble alias, with equal weights for explicit subsets."""
    methods = [methods] if isinstance(methods, str) else list(methods)
    if "ensamble" in methods:
        methods = list(LOCAL_METHODS)
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(LOCAL_METHODS):
        raise ValueError("Choose distinct methods from disk, superpoint and aliked.")
    return methods, [1.0 / len(methods)] * len(methods)


class VectorCache:
    """Transactional, checksummed per-image vectors; reusable across run subsets."""

    def __init__(self, path, spec):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
        self.db.execute("""CREATE TABLE IF NOT EXISTS vectors
            (image_id TEXT PRIMARY KEY, input_hash TEXT, dim INTEGER, digest TEXT, data BLOB)""")
        key = fingerprint(spec)
        old = self.db.execute("SELECT value FROM metadata WHERE key='spec'").fetchone()
        if old and old[0] != key:
            self.db.close()
            raise ValueError("Incompatible vector cache specification.")
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('spec', ?)", (key,))
        self.db.commit()

    def get(self, image_id, input_hash):
        row = self.db.execute("SELECT input_hash, dim, digest, data FROM vectors WHERE image_id=?",
                              (image_id,)).fetchone()
        if row is None:
            return None
        if row[0] != input_hash or hashlib.sha256(row[3]).hexdigest() != row[2]:
            raise ValueError(f"Changed image or damaged vector cache: {image_id}")
        vector = np.frombuffer(row[3], dtype=np.float32).copy()
        if len(vector) != row[1] or not np.isfinite(vector).all():
            raise ValueError(f"Invalid cached vector: {image_id}")
        return vector

    def put(self, image_id, input_hash, vector):
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
            raise ValueError("Invalid vector for cache.")
        data = vector.tobytes()
        self.db.execute("INSERT OR REPLACE INTO vectors VALUES (?, ?, ?, ?, ?)",
                        (image_id, input_hash, len(vector), hashlib.sha256(data).hexdigest(), data))

    def close(self):
        self.db.commit()
        self.db.close()


class LocalFeatures:
    """Validated immutable HDF5 shards with an index and bounded in-memory reads."""

    def __init__(self, root, frame, spec):
        import h5py
        from feature_extraction import extract_features, extract_features_lightglue
        method = spec.get("method", "disk")
        if method not in LOCAL_METHODS:
            raise ValueError(f"Unsupported local feature method: {method}")
        masked = spec.get("input_mode") == "masked"
        rows_by_id = {row["image_id"]: row for row in frame.to_dict("records")}
        input_hashes = {rid: local_input_hash(row, masked=masked) for rid, row in rows_by_id.items()}
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.handles = {}
        self.memory = OrderedDict()
        self.index_path = self.root / "index.json"
        index = json.loads(self.index_path.read_text()) if self.index_path.exists() else {
            "spec": fingerprint(spec), "images": {}, "shards": {}}
        if index["spec"] != fingerprint(spec):
            raise ValueError("Incompatible local feature cache.")
        self.index = index
        missing = []
        used_shards = set()
        for row in rows_by_id.values():
            old = index["images"].get(row["image_id"])
            if old is None:
                missing.append(row)
            elif old["input_hash"] != input_hashes[row["image_id"]]:
                raise ValueError(f"Changed local-feature image: {row['image_id']}")
            else:
                used_shards.add(old["shard"])
        for shard in used_shards:
            for name, digest in index["shards"][shard].items():
                if file_digest(self.root / shard / name) != digest:
                    raise ValueError(f"Damaged local feature shard: {shard}/{name}")
        for start in range(0, len(missing), 256):
            rows = missing[start:start + 256]
            shard = fingerprint([(r["image_id"], input_hashes[r["image_id"]]) for r in rows])[:24]
            destination = self.root / shard
            partial = self.root / (shard + ".partial")
            wanted = {r["image_id"] for r in rows}
            if not destination.exists():
                items = [(r["image_id"], r["path"]) for r in rows]
                if method == "disk":
                    extract_features(items, None, str(partial), max_keypoints=spec["max_keypoints"])
                else:
                    extract_features_lightglue(items, str(partial), feature_type=method,
                                              max_keypoints=spec["max_keypoints"])
                with h5py.File(partial / "descriptors.h5", "r+") as desc, h5py.File(partial / "keypoints.h5", "r+") as kp:
                    if set(desc) != wanted or set(kp) != wanted:
                        raise ValueError("Local extraction did not cover every requested image.")
                    for rid in wanted:
                        d, k = desc[rid][:], kp[rid][:]
                        # LightGlue's SuperPoint/ALIKED helper retains the batch axis.
                        if k.ndim == 3 and k.shape[0] == 1:
                            k = k[0]
                            del kp[rid]
                            kp.create_dataset(rid, data=k, compression="gzip")
                        if d.ndim != 2 or k.shape != (len(d), 2) or not np.isfinite(d).all() or not np.isfinite(k).all():
                            raise ValueError(f"Invalid local features: {rid}")
                        if masked:
                            before = len(k)
                            d, k = filter_foreground_features(d, k, load_foreground_mask(rows_by_id[rid]))
                            del desc[rid], kp[rid]
                            desc.create_dataset(rid, data=d, compression="gzip")
                            desc[rid].attrs["unfiltered_keypoints"] = before
                            kp.create_dataset(rid, data=k, compression="gzip")
                os.replace(partial, destination)
            index["shards"][shard] = {
                name: file_digest(destination / name) for name in ("descriptors.h5", "keypoints.h5")}
            for row in rows:
                index["images"][row["image_id"]] = {"shard": shard, "input_hash": input_hashes[row["image_id"]]}
            if masked:
                with h5py.File(destination / "descriptors.h5") as desc:
                    for row in rows:
                        rid = row["image_id"]
                        index["images"][rid].update(
                            raw_keypoints=int(desc[rid].attrs["unfiltered_keypoints"]),
                            foreground_keypoints=int(desc[rid].shape[0]))
            write_json(self.index_path, index)
            print(f"{method} local cache: {min(start + 256, len(missing))}/{len(missing)} new images", flush=True)
        self.mask_counts = None
        if masked:
            entries = [index["images"][rid] for rid in rows_by_id]
            self.mask_counts = {
                "images": len(entries), "raw_keypoints": sum(e["raw_keypoints"] for e in entries),
                "foreground_keypoints": sum(e["foreground_keypoints"] for e in entries),
                "empty_images": sum(e["foreground_keypoints"] == 0 for e in entries),
            }

    def __contains__(self, image_id):
        return image_id in self.index["images"]

    def pair(self, image_id):
        import h5py
        if image_id in self.memory:
            self.memory.move_to_end(image_id)
            return self.memory[image_id]
        shard = self.index["images"][image_id]["shard"]
        if shard not in self.handles:
            self.handles[shard] = tuple(h5py.File(self.root / shard / name, "r")
                                       for name in ("descriptors.h5", "keypoints.h5"))
        desc, kp = self.handles[shard]
        pair = desc[image_id][:], kp[image_id][:]
        self.memory[image_id] = pair
        if len(self.memory) > 64:
            self.memory.popitem(last=False)
        return pair

    def __getitem__(self, image_id):
        return self.pair(image_id)[0]

    def close(self):
        for handles in self.handles.values():
            for handle in handles:
                handle.close()
        self.handles.clear()
        self.memory.clear()


class PairVerifier:
    """Ordered-pair evidence cache: matching/RANSAC can depend on image order."""

    def __init__(self, path, local, spec, seed):
        import geometric_verification as gv
        if not gv._LIGHTGLUE_AVAILABLE:
            raise RuntimeError("Full-funnel requires the requested LightGlue matcher.")
        self.local, self.seed = local, seed
        self.method = spec["local"]["method"]
        self.namespace = fingerprint(spec)
        self.db = sqlite3.connect(path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS pairs
            (namespace TEXT, query_id TEXT, gallery_id TEXT, evidence TEXT,
             PRIMARY KEY(namespace, query_id, gallery_id))""")
        self.computed, self.cached = 0, 0

    def __call__(self, query_id, gallery_id):
        import cv2
        from geometric_verification import compute_geometric_similarity
        row = self.db.execute("SELECT evidence FROM pairs WHERE namespace=? AND query_id=? AND gallery_id=?",
                              (self.namespace, query_id, gallery_id)).fetchone()
        if row:
            self.cached += 1
            return json.loads(row[0])
        qd, qk = self.local.pair(query_id)
        gd, gk = self.local.pair(gallery_id)
        if not len(qd) or not len(gd):
            result = {"attempted": False, "n_inliers": 0, "reason": "missing_local_features"}
        else:
            pair_seed = int(fingerprint([query_id, gallery_id, self.seed])[:8], 16) % (2**31 - 1)
            cv2.setRNGSeed(pair_seed)
            _, n = compute_geometric_similarity(qd, qk, gd, gk, 1.0,
                                               use_lightglue=True, method=self.method, gv_matcher="lightglue")
            result = {"attempted": True, "n_inliers": int(n), "reason": f"gv_inliers={int(n)}"}
        self.db.execute("INSERT INTO pairs VALUES (?, ?, ?, ?)",
                        (self.namespace, query_id, gallery_id, json.dumps(result)))
        self.computed += 1
        if self.computed % 100 == 0:
            self.db.commit()
        return result

    def close(self):
        self.db.commit()
        self.db.close()


def source_hashes(names):
    return {name: file_digest(ROOT / name) for name in names}


def model_state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def local_feature_spec(method, args, input_mode):
    """Fingerprint each extractor; retain the original DISK cache contract."""
    import inspect
    import kornia
    import lightglue
    import torch
    extractor = {"disk": lightglue.DISK, "superpoint": lightglue.SuperPoint,
                 "aliked": lightglue.ALIKED}[method]
    weights = {"disk": "depth-save.pth", "superpoint": "superpoint_v1.pth",
               "aliked": lightglue.ALIKED.default_conf["model_name"] + ".pth"}
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / weights[method]
    if not checkpoint.exists():
        # Initialise on CPU to let the existing library fetch its own weights.
        instance = extractor(max_num_keypoints=args.max_keypoints)
        del instance
    spec = {
        "method": method, "max_keypoints": args.max_keypoints, "input_mode": input_mode,
        "weights_sha256": file_digest(checkpoint),
        "lightglue_source": file_digest(inspect.getfile(extractor)),
        "lightglue_utils": file_digest(inspect.getfile(lightglue.utils)),
        "torch": torch.__version__, "kornia_version": kornia.__version__,
        "implementation": source_hashes(["feature_extraction.py"]),
    }
    if method == "disk":
        spec["kornia_disk_source"] = file_digest(inspect.getfile(kornia.feature.DISK))
    else:
        spec["keypoint_layout"] = "batch_axis_removed_v1"
    if input_mode == "masked":
        spec["masked_preprocessing"] = MASKED_PREPROCESSING
        spec["mask_keypoint_policy"] = MASK_KEYPOINT_POLICY
        spec["mask_implementation"] = source_hashes(["reid_demo/benchmark_masks.py"])
    return spec


def verification_spec(local_spec, frame, args):
    import cv2
    import inspect
    import lightglue
    import torch
    from lightglue_singleton import get_lightglue
    method = local_spec["method"]
    version = lightglue.LightGlue.version.replace(".", "-")
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / f"{method}_lightglue_{version}.pth"
    if not checkpoint.exists():
        get_lightglue(method)
    return {
        "local": local_spec, "seed": args.seed,
        "matcher_source": file_digest(inspect.getfile(lightglue.LightGlue)),
        "opencv_version": cv2.__version__,
        "image_hashes": fingerprint(sorted(zip(frame.image_id, frame.image_sha256))),
        "weights_sha256": file_digest(checkpoint),
        "implementation": source_hashes(["geometric_verification.py", "constants.py"]),
    }


def global_vectors(frame, args, cache_root):
    import torch
    from torch.utils.data import DataLoader
    from tools.evaluate_reid_embeddings import MetadataImageDataset, _build_transform, _extract_embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, info, image_size = load_global_model(args, device)
    expected_dim = info.get("embedding_dim")
    spec = {
        "kind": info["kind"], "weights_sha256": model_state_digest(model),
        "input_mode": str(frame.input_mode.iloc[0]),
        "transform": f"RGB; resize {image_size}x{image_size} bilinear; ImageNet normalization; L2",
        "amp": device.type == "cuda", "torch": torch.__version__,
        "timm": importlib.metadata.version("timm"),
        "torchvision": importlib.metadata.version("torchvision"),
        "pillow": importlib.metadata.version("pillow"),
        "implementation": source_hashes(["tools/evaluate_reid_embeddings.py", "megadescriptor.py"]),
    }
    if getattr(args, "global_model", "megadescriptor") == "miewid":
        spec.update({key: info[key] for key in ("model_id", "revision", "remote_code_sha256")})
        spec["transform_size"] = image_size
        spec["expected_dimension"] = expected_dim
        spec["transformers"] = importlib.metadata.version("transformers")
        spec["implementation"] = source_hashes([
            "tools/evaluate_reid_embeddings.py", "reid_demo/benchmark_embeddings.py"])
    if spec["input_mode"] == "masked":
        spec["masked_preprocessing"] = MASKED_PREPROCESSING
    expected = getattr(args, "_expected_global_spec", None)
    if expected is not None and {k: v for k, v in expected.items() if k != "dimension"} != spec:
        raise ValueError("Loaded global model differs from calibration; refusing feature extraction.")
    cache = VectorCache(cache_root / ("global_" + fingerprint(spec)[:20] + ".sqlite"), spec)
    try:
        vectors, missing = {}, []
        for row in frame.to_dict("records"):
            v = cache.get(row["image_id"], row["image_sha256"])
            if v is None:
                missing.append(row)
            else:
                if expected_dim is not None and len(v) != expected_dim:
                    raise ValueError(f"Cached embedding dimension mismatch: {row['image_id']}")
                vectors[row["image_id"]] = v
        print(f"Global embeddings: {len(vectors)} cached, {len(missing)} to extract", flush=True)
        for start in range(0, len(missing), 256):
            chunk = pd.DataFrame(missing[start:start + 256])
            # Dummy labels: extraction has no need to see reference identities.
            chunk["identity"] = "unlabelled"
            dataset = MetadataImageDataset(chunk, images_root=ROOT, image_column="path",
                                           label_column="identity", transform=_build_transform(image_size),
                                           namespace_labels=False, skip_missing=False)
            loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                                shuffle=False, pin_memory=device.type == "cuda")
            matrix, _, ids, _, _ = _extract_embeddings(model, loader, device=device, amp=device.type == "cuda")
            if ids != chunk.image_id.tolist():
                raise ValueError("Embedding extractor changed image ordering/coverage.")
            if (matrix.ndim != 2 or len(matrix) != len(ids) or
                    (expected_dim is not None and matrix.shape[1] != expected_dim)):
                raise ValueError("Embedding model returned an unexpected batch shape or feature dimension.")
            hashes = chunk.set_index("image_id").image_sha256.to_dict()
            for rid, vector in zip(ids, matrix.numpy()):
                cache.put(rid, hashes[rid], vector)
                vectors[rid] = vector
            cache.db.commit()
        spec["dimension"] = int(len(next(iter(vectors.values()))))
        return np.stack([vectors[rid] for rid in frame.image_id]), spec
    finally:
        cache.close()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def fisher_model(local, fitting, args, cache_root, local_spec, metadata_digest):
    from reid_demo.fisher import fit_fisher_model
    settings = {name: getattr(args, name) for name in
                ("pca_dim", "gmm_components", "max_descriptors", "per_image_max", "seed")}
    spec = {
        "settings": settings, "local": local_spec, "split": args.split,
        "metadata_sha256": metadata_digest,
        "fitting_images": sorted(zip(fitting.image_id, fitting.image_sha256)),
        "implementation": source_hashes(["reid_demo/fisher.py", "feature_aggregation.py"]),
    }
    binding = getattr(args, "_protocol_binding", None)
    if binding:
        spec["protocol"] = binding
    if getattr(args, "partition", None) == "selection" and not args.fisher_model:
        raise ValueError("Selection cannot fit Fisher models.")
    if args.fisher_model:
        path = Path(args.fisher_model)
        receipt = json.loads(path.with_suffix(".json").read_text())
        stored = receipt["spec"]
        if binding and (stored.get("protocol") != binding or
                        stored.get("fitting_images") != json.loads(json.dumps(spec["fitting_images"]))):
            raise ValueError("Fisher artifact is incompatible with the protocol fitting pool.")
        # A frozen development model may use a bounded fitting subset of official train.
        current_ids = set(fitting.image_id)
        if (stored["settings"] != settings or stored["local"] != local_spec or
                stored["split"] != args.split or stored["metadata_sha256"] != metadata_digest or
                stored["implementation"] != spec["implementation"] or
                not set(i for i, _ in stored["fitting_images"]).issubset(current_ids)):
            raise ValueError("Fisher model is incompatible or used images outside the development fitting partition.")
    else:
        if args.stage == "test":
            raise ValueError("Test Fisher runs require --fisher-model from a development run.")
        path = cache_root / ("fisher_model_" + fingerprint(spec)[:20] + ".pkl")
        if path.exists():
            receipt = json.loads(path.with_suffix(".json").read_text())
            if receipt["spec"] != json.loads(json.dumps(spec)):
                raise ValueError("Incompatible fitted Fisher cache.")
        else:
            print(f"Fitting PCA/GMM on {len(fitting)} development images only", flush=True)
            model = fit_fisher_model(local, fitting.image_id, **settings)
            temp = path.with_suffix(".pkl.tmp")
            temp.write_bytes(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
            os.replace(temp, path)
            receipt = {"spec": spec, "sha256": file_digest(path)}
            write_json(path.with_suffix(".json"), receipt)
    if file_digest(path) != receipt["sha256"]:
        raise ValueError("Damaged Fisher model.")
    # Only locally generated benchmark artifacts should be passed as pickle models.
    model = pickle.loads(path.read_bytes())
    if sorted(model.fitting_ids) != sorted(i for i, _ in receipt["spec"]["fitting_images"]):
        raise ValueError("Fitted model provenance does not match its receipt.")
    return model, {"path": str(path.resolve()), "sha256": receipt["sha256"],
                   "settings": settings, "fitting_ids": model.fitting_ids,
                   "descriptor_digest": model.fitting_digest, "converged": bool(model.gmm.converged_),
                   "iterations": int(model.gmm.n_iter_)}


def fisher_vectors(frame, local, model, model_info, args, cache_root):
    from reid_demo.fisher import transform_fisher_vectors
    spec = {"model_sha256": model_info["sha256"], "input_mode": frame.input_mode.iloc[0]}
    cache = VectorCache(cache_root / ("fisher_" + fingerprint(spec)[:20] + ".sqlite"), spec)
    # A memory map prevents full-dataset Fisher vectors from doubling peak RAM.
    dim = 2 * model.gmm_components * model.pca_dim
    ids_digest = fingerprint(list(zip(frame.image_id, frame.image_sha256)))[:20]
    matrix = np.lib.format.open_memmap(cache_root / f"matrix_{fingerprint(spec)[:16]}_{ids_digest}.npy",
                                      mode="w+", dtype=np.float32, shape=(len(frame), dim))
    try:
        computed = 0
        for index, row in enumerate(frame.to_dict("records")):
            input_hash = local_input_hash(row, masked=frame.input_mode.iloc[0] == "masked")
            vector = cache.get(row["image_id"], input_hash)
            if vector is None:
                vector = transform_fisher_vectors(local, [row["image_id"]], model)[row["image_id"]]
                cache.put(row["image_id"], input_hash, vector)
                computed += 1
            matrix[index] = vector
            if (index + 1) % 100 == 0:
                cache.db.commit()
                print(f"Fisher vectors: {index + 1}/{len(frame)} ({computed} newly transformed)", flush=True)
        matrix.flush()
        return matrix
    finally:
        cache.close()


def read_fisher_bundle(path, methods, weights):
    """Validate the bundle before any extraction or fitting can start."""
    path = Path(path)
    if path.suffix != ".json":
        raise ValueError("Ensemble runs require --fisher-model <development>/fisher_bundle.json.")
    data = json.loads(path.read_text())
    if (data.get("schema") != 1 or data.get("kind") != "fisher_ensemble" or
            data.get("stage") != "development" or data.get("methods") != methods or
            data.get("weights") != weights or data.get("combination") != FISHER_COMBINATION or
            set(data.get("models", {})) != set(methods)):
        raise ValueError("Incompatible development Fisher ensemble bundle.")
    for method in methods:
        member = data["models"][method]
        if file_digest(member["path"]) != member["sha256"]:
            raise ValueError(f"Damaged {method} Fisher model in ensemble bundle.")
    return data


def fit_or_load_fisher_models(locals_by_method, specs, fitting, args, cache, metadata_digest):
    """Use separate models, all fitted on the same permitted fitting images."""
    methods, weights = resolve_methods(args.method)
    if len(methods) == 1:
        method = methods[0]
        model, info = fisher_model(locals_by_method[method], fitting, args, cache,
                                   specs[method], metadata_digest)
        return {method: model}, info
    bundle = read_fisher_bundle(args.fisher_model, methods, weights) if args.fisher_model else None
    if args.stage == "test" and bundle is None:
        raise ValueError("Test ensemble runs require --fisher-model from development.")
    models, members = {}, {}
    for method in methods:
        member_args = argparse.Namespace(**vars(args))
        member_args.fisher_model = bundle["models"][method]["path"] if bundle else None
        print(f"Fisher model: {method} ({args.gmm_components} GMM components)", flush=True)
        models[method], members[method] = fisher_model(
            locals_by_method[method], fitting, member_args, cache, specs[method], metadata_digest)
        if bundle and members[method]["sha256"] != bundle["models"][method]["sha256"]:
            raise ValueError(f"Changed {method} Fisher model in ensemble bundle.")
    fitting_ids = sorted(members[methods[0]]["fitting_ids"])
    if any(sorted(info["fitting_ids"]) != fitting_ids for info in members.values()):
        raise ValueError("Ensemble models must use the same development fitting images.")
    if bundle:
        path = Path(args.fisher_model)
    else:
        path = Path(args.out_dir) / "fisher_bundle.json"
        write_json(path, {
            "schema": 1, "kind": "fisher_ensemble", "stage": "development",
            "methods": methods, "weights": weights, "combination": FISHER_COMBINATION,
            "models": {method: {key: members[method][key] for key in ("path", "sha256")}
                       for method in methods},
        })
    return models, {
        "path": str(path.resolve()), "sha256": file_digest(path),
        "methods": methods, "weights": weights, "combination": FISHER_COMBINATION,
        "settings": members[methods[0]]["settings"], "fitting_ids": fitting_ids,
        "models": members,
    }


def combine_fisher_matrices(matrices, weights, path, *, chunk_size=64):
    """Legacy weighted Fisher sum, L2-normalised for cosine ranking, in blocks."""
    if not matrices or len(matrices) != len(weights) or chunk_size < 1:
        raise ValueError("Fisher matrices and weights must have matching nonempty lengths.")
    shape = matrices[0].shape
    if len(shape) != 2 or any(matrix.shape != shape for matrix in matrices):
        raise ValueError("Ensemble Fisher matrices must have equal shapes and image ordering.")
    if not np.isfinite(weights).all() or min(weights) < 0 or sum(weights) <= 0:
        raise ValueError("Ensemble weights must be finite, nonnegative and have positive sum.")
    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    for start in range(0, shape[0], chunk_size):
        stop = min(start + chunk_size, shape[0])
        block = np.zeros((stop - start, shape[1]), dtype=np.float32)
        for component, weight in zip(matrices, weights):
            block += weight * component[start:stop]
        matrix[start:stop] = normalized(block)
    matrix.flush()
    return matrix


def threshold_contract(args, metadata_digest, global_spec, model_info, gv_spec=None):
    contract = {
        "split": args.split, "metadata_sha256": metadata_digest, "global": global_spec,
        "fisher_sha256": model_info["sha256"] if model_info else None,
        "shortlist_size": args.shortlist_size, "rerank_size": args.rerank_size,
        "ranking_code": file_digest(ROOT / "reid_demo/benchmark.py"),
    }
    methods, weights = resolve_methods(args.method)
    # Keep existing DISK thresholds usable; new choices must bind their full configuration.
    masked_gv = gv_spec and (gv_spec["local"].get("input_mode") == "masked" or
                             gv_spec.get("foreground_mask"))
    if model_info and (methods != ["disk"] or args.gv_features != "disk" or masked_gv
                       or getattr(args, "_protocol_binding", None)):
        contract["local_configuration"] = {
            "methods": methods, "weights": weights, "combination": FISHER_COMBINATION,
            "gv": gv_spec,
        }
    return contract


def validate_thresholds(path, contract, official_train_ids):
    data = json.loads(Path(path).read_text())
    if data.get("stage") != "development" or data.get("contract") != contract:
        raise ValueError("Thresholds must come from a compatible development run.")
    ids = set(data["query_ids"]) | set(data["gallery_ids"])
    if not ids.issubset(set(official_train_ids)):
        raise ValueError("Threshold calibration used images outside official training data.")
    if set(data["query_ids"]) & set(data["gallery_ids"]):
        raise ValueError("Threshold calibration partitions overlap.")
    for value in data["thresholds"].values():
        if not np.isfinite(value):
            raise ValueError("Threshold must be finite.")
    return data


def masked_input_spec(frame, prepared_manifest):
    if frame.input_mode.iloc[0] != "masked":
        return None
    if (not {"mask_path", "mask_sha256"}.issubset(frame) or
            frame[["mask_path", "mask_sha256"]].isna().any().any() or
            prepared_manifest.get("masked_preprocessing") != MASKED_PREPROCESSING):
        raise ValueError("Use validated masked metadata from utils/prepare_czechlynx.py --modes masked.")
    return {
        "preprocessing": MASKED_PREPROCESSING, "keypoint_policy": MASK_KEYPOINT_POLICY,
        # Use all prepared IDs so development and test share the same cache contract.
        "mask_dataset_sha256": fingerprint(sorted(zip(frame.image_id, frame.mask_sha256))),
    }


def run_clustering(frame, matrix, fisher, args, verifier, signals):
    from reid_demo.cluster import cluster_by_flank
    from reid_demo.fusion import (PairScore, build_fused_affinity,
                                  refine_affinity_with_gv, select_borderline_pairs)
    if args.cluster_limit == 0:
        return {}, None
    rng = np.random.default_rng(args.seed)
    take = np.arange(len(frame))
    if len(take) > args.cluster_limit:
        take = np.sort(rng.choice(take, args.cluster_limit, replace=False))
    subset = frame.iloc[take].sort_values("image_id")
    positions = {rid: i for i, rid in enumerate(frame.image_id)}
    take = [positions[rid] for rid in subset.image_id]
    ids = subset.image_id.tolist()
    x = normalized(matrix[take])
    embeddings = dict(zip(ids, x))
    orientations = dict.fromkeys(ids, "unknown")
    truth = subset.set_index("image_id").identity.to_dict()
    stats, rows = {}, []
    fused = None
    for signal in signals:
        affinity = None
        coverage = {}
        if signal != "global":
            if fused is None:
                fused = build_fused_affinity(ids, x, fisher[take], orientations=orientations,
                                              flank_policy="ignore")
            affinity = fused
        if signal == "full-funnel":
            pairs, capped = select_borderline_pairs(ids, fused, orientations,
                                                    budget=args.cluster_gv_budget,
                                                    flank_policy="ignore", seed=args.seed)
            pair_scores = []
            pos = {rid: i for i, rid in enumerate(ids)}
            for a, b, bucket in pairs:
                evidence = verifier(a, b)
                base = float(fused[pos[a], pos[b]])
                pair_scores.append(PairScore(
                    record_id_a=a, record_id_b=b, fused_prob=base,
                    n_inliers=evidence["n_inliers"], gv_prob=None,
                    geom_score=min(evidence["n_inliers"] / 50, 1) if evidence["attempted"] else base,
                    bucket=bucket, reason=evidence["reason"]))
            affinity = refine_affinity_with_gv(fused, ids, pair_scores)
            coverage = {"gv_pairs": len(pairs), "gv_capped_pairs": capped,
                        "gv_skipped_pairs": sum(p.reason == "missing_local_features" for p in pair_scores)}
        result = cluster_by_flank(embeddings, orientations, backend="leiden",
                                  backend_params={"knn_k": args.knn_k, "resolution": args.resolution},
                                  seed=args.seed, flank_policy="ignore", affinity=affinity)
        actual = "leiden" if importlib.util.find_spec("leidenalg") and importlib.util.find_spec("igraph") else "networkx_louvain"
        stats[signal] = {**clustering_metrics(result.image_ids, result.labels, truth),
                         **coverage, "backend": actual, "sample_ids": ids,
                         "sampling": "seeded images from query partition; no target cluster count"}
        rows.extend({"signal": signal, "image_id": rid, "cluster_id": int(label), "identity": truth[rid]}
                    for rid, label in zip(result.image_ids, result.labels))
    return stats, pd.DataFrame(rows)



def protocol_request(args, plan):
    """Cheap provenance checks, before model loading or feature extraction."""
    names = ("global_model", "miewid_revision", "signals", "method", "gv_features",
             "shortlist_size", "rerank_size", "depth", "max_keypoints",
             "pca_dim", "gmm_components", "max_descriptors", "per_image_max", "seed")
    return {
        "protocol": plan["fingerprint"],
        "options": {name: getattr(args, name) for name in names},
        "checkpoint_sha256": file_digest(args.ckpt) if args.ckpt else None,
        "gv_mask_metadata_sha256": file_digest(args.gv_mask_metadata) if args.gv_mask_metadata else None,
        "implementation": source_hashes([
            "tools/evaluate_czechlynx.py", "reid_demo/benchmark.py",
            "reid_demo/benchmark_protocol.py", "reid_demo/benchmark_embeddings.py",
            "reid_demo/fisher.py", "feature_aggregation.py", "feature_extraction.py",
            "geometric_verification.py", "constants.py", "reid_demo/benchmark_masks.py",
            "reid_demo/benchmark_gv_masks.py"]),
    }


def preflight_protocol_artifacts(args, plan, fitting, metadata_digest, local_specs, gv_spec=None):
    request = protocol_request(args, plan)
    calibration = None
    if args.thresholds:
        calibration = validate_calibration(
            json.loads(Path(args.thresholds).read_text()), plan, args.signals)
        if calibration.get("request") != request:
            raise ValueError("Calibration model/ranking request differs from this selection run.")
        contract = calibration.get("contract", {})
        if not isinstance(contract.get("global"), dict) or not contract["global"]:
            raise ValueError("Calibration artifact lacks its global model contract.")
        model_info = {"sha256": file_digest(args.fisher_model)} if args.fisher_model else None
        expected = threshold_contract(args, metadata_digest, contract["global"], model_info, gv_spec)
        expected["protocol"] = protocol_binding(plan)
        if contract != expected:
            raise ValueError("Calibration artifact contract differs from this selection run.")
        args._expected_global_spec = contract["global"]
    if args.fisher_model:
        methods, weights = resolve_methods(args.method)
        if len(methods) > 1:
            bundle = read_fisher_bundle(args.fisher_model, methods, weights)
            members = bundle["models"]
        else:
            members = {methods[0]: {"path": args.fisher_model}}
        expected_images = sorted([str(i), str(h)] for i, h in zip(fitting.image_id, fitting.image_sha256))
        settings = {key: getattr(args, key) for key in
                    ("pca_dim", "gmm_components", "max_descriptors", "per_image_max", "seed")}
        for method, member in members.items():
            path = Path(member["path"])
            receipt = json.loads(path.with_suffix(".json").read_text())
            spec = receipt["spec"]
            if (spec.get("protocol") != protocol_binding(plan)
                    or spec.get("fitting_images") != expected_images
                    or spec.get("metadata_sha256") != metadata_digest
                    or spec.get("settings") != settings or spec.get("split") != args.split
                    or spec.get("local") != local_specs[method]
                    or spec.get("implementation") != source_hashes(["reid_demo/fisher.py", "feature_aggregation.py"])):
                raise ValueError("Fisher artifact is incompatible with this protocol fitting pool.")
            if file_digest(path) != receipt["sha256"]:
                raise ValueError("Damaged Fisher artifact.")
        if calibration and calibration["contract"].get("fisher_sha256") != file_digest(args.fisher_model):
            raise ValueError("Selection Fisher artifact differs from the calibration artifact.")
    elif calibration and calibration["contract"].get("fisher_sha256") is not None:
        raise ValueError("Selection is missing the Fisher artifact used for calibration.")
    return request


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cache-dir", default="data/czechlynx/cache")
    parser.add_argument("--stage", choices=["development", "test"], default="development")
    parser.add_argument("--protocol-plan", help="Versioned identity-disjoint development protocol JSON.")
    parser.add_argument("--partition", choices=["calibration", "selection"])
    parser.add_argument("--split", choices=["time_closed", "time_open"], default="time_open")
    parser.add_argument("--signals", nargs="+", choices=SIGNALS, default=list(SIGNALS))
    parser.add_argument("--method", nargs="+", choices=[*LOCAL_METHODS, "ensamble"], default=["disk"],
                        help="Fisher extractors; ensamble selects DISK + SuperPoint + ALIKED with equal weights.")
    parser.add_argument("--gv_features", "--gv-features", choices=LOCAL_METHODS,
                        help="GV extractor (default: DISK for an ensemble, otherwise --method).")
    parser.add_argument("--gv-mask-metadata",
                        help="Prepared masked metadata; filter only GV keypoints on unchanged bbox images.")
    parser.add_argument("--overlap-policy", choices=["error", "purge-query"], default="error")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-identities", type=int)
    parser.add_argument("--max-images-per-identity", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--shortlist-size", type=int, default=50)
    parser.add_argument("--rerank-size", type=int, default=10)
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--max-keypoints", type=int, default=1024)
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--gmm-components", type=int, default=256, help="GMM components per extractor.")
    parser.add_argument("--max-descriptors", type=int, default=50000)
    parser.add_argument("--per-image-max", type=int, default=128)
    parser.add_argument("--fisher-model", help="Frozen development .pkl, or fisher_bundle.json for an ensemble.")
    parser.add_argument("--thresholds")
    parser.add_argument("--target-far", type=float, default=0.05)
    parser.add_argument("--global-model", choices=GLOBAL_MODELS, default="megadescriptor",
                        help="Frozen global embedding model (default: megadescriptor).")
    parser.add_argument("--miewid-revision", help="Optional full commit hash; MiewID otherwise uses the pinned release.")
    parser.add_argument("--compare-to", help="Completed reference report directory; compare global rankings on identical inputs.")
    parser.add_argument("--ckpt", help="Optional MegaDescriptor checkpoint; incompatible with --global-model miewid.")
    parser.add_argument("--cluster-limit", type=int, default=2000)
    parser.add_argument("--cluster-gv-budget", type=int, default=2000)
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--resolution", type=float, default=1.0)
    args = parser.parse_args(argv)
    try:
        args.method, weights = resolve_methods(args.method)
        validate_global_options(args)
    except ValueError as exc:
        parser.error(str(exc))
    args.gv_features = args.gv_features or (args.method[0] if len(args.method) == 1 else "disk")
    if bool(args.protocol_plan) != bool(args.partition):
        parser.error("--protocol-plan and --partition must be supplied together.")
    if args.protocol_plan:
        if args.stage != "development" or args.split != "time_open":
            parser.error("Protocol plans currently support development time_open only.")
        if args.max_identities is not None or args.max_images_per_identity is not None:
            parser.error("Protocol plans fix image membership; sampling flags are incompatible.")
        if args.fisher_model and all(s == "global" for s in args.signals):
            parser.error("Global-only protocol runs do not use --fisher-model.")
        if args.partition == "calibration" and args.thresholds:
            parser.error("Calibration creates thresholds; do not pass --thresholds.")
        if args.partition == "selection":
            if not args.thresholds:
                parser.error("Selection requires --thresholds from this protocol's calibration run.")
            if any(s != "global" for s in args.signals) and not args.fisher_model:
                parser.error("Selection Fisher runs require the calibration --fisher-model.")
            if Path(args.out_dir).resolve() == Path(args.thresholds).resolve().parent:
                parser.error("Selection must use a different output directory from calibration.")
    elif args.thresholds:
        if json.loads(Path(args.thresholds).read_text()).get("protocol"):
            parser.error("Identity-pool thresholds require --protocol-plan and --partition selection.")
    if args.compare_to and "global" not in args.signals:
        parser.error("--compare-to requires the global signal.")
    if args.compare_to and Path(args.compare_to).resolve() == Path(args.out_dir).resolve():
        parser.error("--compare-to and --out-dir must be different directories.")
    if args.gv_mask_metadata and "full-funnel" not in args.signals:
        parser.error("--gv-mask-metadata requires the full-funnel signal.")
    if args.cluster_limit < 0 or args.cluster_limit > 5000:
        parser.error("--cluster-limit must be 0..5000 (clustering builds a dense sample affinity).")
    if args.stage == "test" and not args.thresholds:
        parser.error("Frozen test runs require --thresholds from development.")
    if args.stage == "test" and (args.max_identities or args.max_images_per_identity):
        parser.error("Official test runs use all images; sampling is development-only.")
    if any(signal != "global" for signal in args.signals):
        if args.stage == "test" and not args.fisher_model:
            parser.error("Test Fisher runs require --fisher-model from development.")
        if args.fisher_model and len(args.method) > 1:
            read_fisher_bundle(args.fisher_model, args.method, weights)
        elif args.fisher_model and Path(args.fisher_model).suffix != ".pkl":
            parser.error("A single extractor requires its development .pkl; ensembles use fisher_bundle.json.")
    for name in ("batch_size", "threads", "shortlist_size", "rerank_size", "depth", "chunk_size",
                 "max_keypoints", "pca_dim", "gmm_components", "max_descriptors", "per_image_max"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.workers < 0 or args.cluster_gv_budget < 0 or not 0 <= args.target_far < 1:
        parser.error("Invalid worker/budget/false-acceptance setting.")
    from threadpoolctl import threadpool_limits
    import torch
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    threadpool_limits(limits=args.threads)
    started = time.perf_counter()
    out, cache = Path(args.out_dir), Path(args.cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    metadata = Path(args.metadata)
    frame = pd.read_csv(metadata, dtype={"image_id": str, "identity": str})
    required = {"image_id", "identity", "path", "image_sha256", "source_sha256",
                "encounter_id", "timestamp", "input_mode", "split-" + args.split}
    if not required.issubset(frame) or frame.image_id.duplicated().any() or frame.input_mode.nunique() != 1:
        raise ValueError("Use validated metadata from utils/prepare_czechlynx.py.")
    metadata_digest = file_digest(metadata)
    prepared_manifest = json.loads(metadata.with_name("manifest.json").read_text())
    if prepared_manifest["prepared_metadata_sha256"] != metadata_digest:
        raise ValueError("Prepared metadata no longer matches its manifest.")
    mask_spec = masked_input_spec(frame, prepared_manifest)
    gv_mask_frame, gv_mask_spec = None, None
    if args.gv_mask_metadata:
        gv_mask_frame, gv_mask_spec = prepare_gv_masks(frame, args.gv_mask_metadata)
    official_train = frame[frame["split-" + args.split] == "train"]
    plan, request, protocol_specs, protocol_gv_spec = None, None, {}, None
    if args.protocol_plan:
        plan = load_protocol(args.protocol_plan, frame, metadata_digest)
        if args.seed != plan["settings"]["seed"]:
            raise ValueError("--seed must match the frozen protocol plan.")
        fitting, gallery, query, audit = protocol_frames(plan, frame, args.partition)
        args._protocol_binding = protocol_binding(plan)
        if any(s != "global" for s in args.signals):
            for method in set(args.method) | ({args.gv_features} if "full-funnel" in args.signals else set()):
                protocol_specs[method] = local_feature_spec(method, args, frame.input_mode.iloc[0])
                if mask_spec:
                    protocol_specs[method]["mask_dataset_sha256"] = mask_spec["mask_dataset_sha256"]
        if "full-funnel" in args.signals:
            protocol_gv_spec = verification_spec(protocol_specs[args.gv_features], frame, args)
            if gv_mask_spec:
                protocol_gv_spec["foreground_mask"] = gv_mask_spec
        request = preflight_protocol_artifacts(
            args, plan, fitting, metadata_digest, protocol_specs, protocol_gv_spec)
    else:
        selected = frame
        if args.stage == "development":
            selected = sample_encounters(official_train, max_identities=args.max_identities,
                                         max_images_per_identity=args.max_images_per_identity, seed=args.seed)
        gallery, query, audit = make_protocol(selected, split=args.split, stage=args.stage,
                                              seed=args.seed, overlap_policy=args.overlap_policy)
        fitting = gallery if args.stage == "development" else official_train
    reference = None
    if args.compare_to:
        reference = load_global_reference(
            args.compare_to, gallery, query, metadata_digest=metadata_digest,
            stage=args.stage, split=args.split, input_mode=str(frame.input_mode.iloc[0]), depth=args.depth)
    needed = pd.concat([gallery, query]).drop_duplicates("image_id").sort_values("image_id").reset_index(drop=True)
    local_needed = needed
    if plan and any(s != "global" for s in args.signals) and not args.fisher_model:
        local_needed = pd.concat([needed, fitting]).drop_duplicates("image_id").sort_values("image_id").reset_index(drop=True)
    if gv_mask_frame is not None:
        gv_mask_frame = gv_mask_frame.set_index("image_id").loc[needed.image_id].reset_index()
    print(f"{args.stage} {args.split}: gallery={len(gallery)}, query={len(query)}, "
          f"unknown={audit['unknown_query_images']}", flush=True)
    # Check bytes as well as the source path before trusting a reusable cache.
    for row in local_needed.to_dict("records"):
        if file_digest(row["path"]) != row["image_sha256"]:
            raise ValueError(f"Image changed since preparation: {row['image_id']}")
        if mask_spec:
            load_foreground_mask(row)
    if gv_mask_frame is not None:
        for row in gv_mask_frame.to_dict("records"):
            load_foreground_mask(row)
    config = {key: value for key, value in vars(args).items()
              if key not in {"out_dir", "cache_dir"} and not key.startswith("_")}
    if not plan:
        config.pop("protocol_plan")
        config.pop("partition")
    if not args.compare_to:
        config.pop("compare_to")
    if args.global_model == "megadescriptor":
        config.pop("global_model")
        config.pop("miewid_revision")  # Keep existing MegaDescriptor run fingerprints.
    if not args.gv_mask_metadata:
        config.pop("gv_mask_metadata")  # Preserve existing bbox/masked run fingerprints.
    if args.method == ["disk"] and args.gv_features == "disk":
        # Preserve fingerprints for interrupted runs made before ensemble support.
        config.pop("method")
        config.pop("gv_features")
    run_spec = {"config": config, "metadata_sha256": metadata_digest}
    if plan:
        run_spec["protocol_fingerprint"] = plan["fingerprint"]
        run_spec["request"] = request
        if args.thresholds:
            run_spec["thresholds_sha256"] = file_digest(args.thresholds)
    if reference:
        run_spec["comparison_reference"] = reference["provenance"]
    if gv_mask_spec:
        run_spec["gv_foreground_mask"] = gv_mask_spec
    identity = fingerprint(run_spec)
    previous = out / "manifest.json"
    if previous.exists() and json.loads(previous.read_text()).get("run_fingerprint") != identity:
        raise ValueError("Output directory belongs to a different run; choose a new --out-dir.")
    manifest = {
        "schema": 1, "stage": args.stage, "split": args.split, "seed": args.seed,
        "input_mode": frame.input_mode.iloc[0], "run_fingerprint": identity,
        "config": config, "prepared_data": prepared_manifest, "metadata_sha256": metadata_digest,
        "gallery_ids": gallery.image_id.tolist(), "query_ids": query.image_id.tolist(),
        "audit": audit, "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "code": source_hashes(["tools/evaluate_czechlynx.py", "reid_demo/benchmark.py",
                              "reid_demo/benchmark_report.py", "reid_demo/fisher.py",
                              "reid_demo/fusion.py", "reid_demo/cluster.py", "geometric_verification.py"]),
        "status": "running", "timing_seconds": {},
    }
    if plan:
        manifest["partition"] = args.partition
        manifest["protocol"] = protocol_binding(plan)
        manifest["protocol_inventory"] = {
            key: {k: v for k, v in pool.items() if k in
                  ("inventory", "gallery", "query", "known_queries", "unknown_queries", "excluded")}
            for key, pool in plan["pools"].items()}
        manifest["request"] = request
        manifest["code"].update(source_hashes(["reid_demo/benchmark_protocol.py",
                                               "reid_demo/benchmark_diagnostics.py"]))
    if mask_spec:
        manifest["segmentation"] = mask_spec
        manifest["code"].update(source_hashes(["reid_demo/benchmark_masks.py"]))
    if args.global_model == "miewid":
        manifest["code"].update(source_hashes(["reid_demo/benchmark_embeddings.py"]))
    if reference:
        manifest["code"].update(source_hashes(["reid_demo/benchmark_comparison.py"]))
    if gv_mask_spec:
        manifest["gv_foreground_mask"] = gv_mask_spec
        manifest["code"].update(source_hashes(["reid_demo/benchmark_gv_masks.py",
                                               "reid_demo/benchmark_masks.py"]))
    write_json(previous, manifest)
    gallery.to_csv(out / "gallery.csv", index=False)
    query.to_csv(out / "query.csv", index=False)
    if plan:
        fitting.to_csv(out / "fitting.csv", index=False)
    source_audit = metadata.parent.parent / "data_audit.json"
    write_json(out / "data_audit.json", {"preparation": json.loads(source_audit.read_text()),
                                       "protocol": audit})
    locals_by_method = {}
    verifier, fisher, model_info, gv_spec = None, None, None, None
    gv_foreground = None
    try:
        timer = time.perf_counter()
        global_matrix, global_spec = global_vectors(needed, args, cache)
        manifest["global_model"] = global_spec
        manifest["timing_seconds"]["global_features"] = time.perf_counter() - timer
        if any(signal != "global" for signal in args.signals):
            extraction_methods = list(args.method)
            if "full-funnel" in args.signals and args.gv_features not in extraction_methods:
                extraction_methods.append(args.gv_features)
            specs = {}
            timer = time.perf_counter()
            for method in extraction_methods:
                specs[method] = protocol_specs.get(method) or local_feature_spec(method, args, frame.input_mode.iloc[0])
                if mask_spec:
                    specs[method]["mask_dataset_sha256"] = mask_spec["mask_dataset_sha256"]
                locals_by_method[method] = LocalFeatures(
                    cache / ("local_" + fingerprint(specs[method])[:20]), local_needed, specs[method])
            manifest["local_features"] = {
                "methods": args.method, "weights": weights,
                "combination": FISHER_COMBINATION if len(args.method) > 1 else "single",
                "pca_dim": args.pca_dim, "gmm_components_per_extractor": args.gmm_components,
                "max_keypoints": args.max_keypoints,
                "gv_features": args.gv_features if "full-funnel" in args.signals else None,
                "extractors": specs,
            }
            if mask_spec:
                manifest["local_features"]["mask_filter_counts"] = {
                    method: locals_by_method[method].mask_counts for method in extraction_methods}
            manifest["timing_seconds"]["local_features"] = time.perf_counter() - timer
            timer = time.perf_counter()
            models, model_info = fit_or_load_fisher_models(
                locals_by_method, specs, fitting, args, cache, metadata_digest)
            manifest["fisher_model"] = model_info
            manifest["timing_seconds"]["fisher_fit"] = time.perf_counter() - timer
            timer = time.perf_counter()
            components = []
            for method in args.method:
                info = model_info["models"][method] if len(args.method) > 1 else model_info
                components.append(fisher_vectors(needed, locals_by_method[method], models[method], info, args, cache))
            if len(components) == 1:
                fisher = components[0]
            else:
                matrix_key = fingerprint({"bundle": model_info["sha256"],
                                          "images": list(zip(needed.image_id, needed.image_sha256)),
                                          "combination": FISHER_COMBINATION})[:24]
                fisher = combine_fisher_matrices(components, weights, cache / f"ensemble_{matrix_key}.npy",
                                                 chunk_size=args.chunk_size)
            del components
            manifest["timing_seconds"]["fisher_transform"] = time.perf_counter() - timer
            if "full-funnel" in args.signals:
                gv_spec = protocol_gv_spec or verification_spec(specs[args.gv_features], frame, args)
                gv_local = locals_by_method[args.gv_features]
                if gv_mask_spec:
                    gv_foreground = ForegroundFeatures(gv_local, gv_mask_frame)
                    gv_local = gv_foreground
                    gv_spec["foreground_mask"] = gv_mask_spec
                    manifest["local_features"]["gv_mask_filter_counts"] = {
                        args.gv_features: gv_foreground.mask_counts}
                verifier = PairVerifier(cache / "verification.sqlite", gv_local, gv_spec, args.seed)
                manifest["gv"] = gv_spec
        contract = threshold_contract(args, metadata_digest, global_spec, model_info, gv_spec)
        if plan:
            contract["protocol"] = protocol_binding(plan)
        thresholds = {}
        if args.thresholds:
            calibration = validate_thresholds(args.thresholds, contract, official_train.image_id)
            thresholds = calibration["thresholds"]
            if not plan and model_info and not set(model_info["fitting_ids"]).issubset(set(calibration["gallery_ids"])):
                raise ValueError("Fisher fitting images were outside the calibration gallery.")
            if set(args.signals) - thresholds.keys():
                raise ValueError("Development thresholds do not cover all requested signals.")
            manifest["threshold_provenance"] = {"path": str(Path(args.thresholds).resolve()),
                                                "sha256": file_digest(args.thresholds)}
        positions = {rid: i for i, rid in enumerate(needed.image_id)}
        gi = [positions[rid] for rid in gallery.image_id]
        qi = [positions[rid] for rid in query.image_id]
        metrics, predictions = {}, {}
        for signal in args.signals:
            timer = time.perf_counter()
            rankings = []
            for pred in rank_queries(
                    query.image_id, gallery.image_id, global_matrix[qi], global_matrix[gi],
                    signal=signal, query_fisher=fisher[qi] if fisher is not None and signal != "global" else None,
                    gallery_fisher=fisher[gi] if fisher is not None and signal != "global" else None,
                    shortlist_size=args.shortlist_size, rerank_size=args.rerank_size,
                    depth=args.depth, chunk_size=args.chunk_size, verifier=verifier, features_normalized=True):
                rankings.append(pred)
                if len(rankings) % 100 == 0:
                    print(f"{signal}: ranked {len(rankings)}/{len(query)} queries", flush=True)
            stats, rows = retrieval_metrics(rankings, query, gallery, depth=args.depth)
            if signal == "full-funnel" and stats["gv_attempted_pairs"] == 0:
                raise RuntimeError("Full-funnel did not perform any geometric verification.")
            if args.stage == "development" and args.partition != "selection" and stats["unknown_queries"]:
                thresholds[signal] = rejection_threshold(rows, target_false_accept_rate=args.target_far)
            if signal in thresholds:
                stats, rows = retrieval_metrics(rankings, query, gallery, depth=args.depth,
                                                threshold=thresholds[signal])
            stats["ranking_seconds"] = time.perf_counter() - timer
            stats["threshold_evaluation"] = (
                "selection validation (frozen calibration thresholds)" if args.partition == "selection"
                else "development calibration (in-sample)" if args.stage == "development" else "held-out test")
            metrics[signal], predictions[signal] = stats, rows
            write_json(out / f"predictions_{signal}.json", rows)
            write_json(out / "metrics.json", metrics)
            print(f"{signal}: top1={stats['top1']}, top5={stats['top5']}, mAP@{args.depth}={stats['map_at_k']}", flush=True)
        if args.stage == "development" and args.partition != "selection":
            write_json(out / "thresholds.json", {
                "stage": "development", "contract": contract, "thresholds": thresholds,
                "query_ids": query.image_id.tolist(), "gallery_ids": gallery.image_id.tolist(),
                "unknown_queries": audit["unknown_query_images"], "target_far": args.target_far,
                **({"partition": "calibration", "protocol": protocol_binding(plan), "request": request}
                   if plan else {})})
        timer = time.perf_counter()
        cluster_stats, clusters = run_clustering(query, global_matrix[qi],
                                                 fisher[qi] if fisher is not None else None,
                                                 args, verifier, args.signals)
        manifest["timing_seconds"]["clustering"] = time.perf_counter() - timer
        write_json(out / "clustering_metrics.json", cluster_stats)
        for signal, stats in cluster_stats.items():
            metrics[signal].update({"cluster_" + k: v for k, v in stats.items() if k != "sample_ids"})
        if verifier:
            manifest["gv_cache"] = {"computed_pairs": verifier.computed, "cached_pairs": verifier.cached}
        if reference:
            comparison = compare_global_predictions(
                reference, predictions["global"], query, gallery, model_spec=global_spec, depth=args.depth)
            manifest["global_comparison"] = comparison
            print(f"Global comparison: fixed={comparison['fixed_queries']}, "
                  f"worsened={comparison['worsened_queries']} known queries", flush=True)
        if plan:
            timer = time.perf_counter()
            appearance = appearance_measurements(needed, cache)
            diagnostics = build_diagnostics(predictions, gallery, query, appearance, depth=args.depth)
            write_json(out / "diagnostics.json", diagnostics)
            manifest["diagnostics"] = {k: v for k, v in diagnostics.items() if k != "query_measurements"}
            manifest["timing_seconds"]["diagnostics"] = time.perf_counter() - timer
        manifest["status"] = "complete"
        manifest["timing_seconds"]["total_before_report"] = time.perf_counter() - started
        from reid_demo.benchmark_report import write_report
        write_report(out, manifest, metrics, predictions, needed, clusters=clusters)
        print(f"Report: {(out / 'index.html').resolve()}", flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        write_json(previous, manifest)
        raise
    finally:
        if verifier:
            verifier.close()
        if gv_foreground:
            gv_foreground.close()
        for local in locals_by_method.values():
            local.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
