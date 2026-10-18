"""Fitting-only native MiewID adaptation; neural dependencies are lazy imports."""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from reid_demo.benchmark import file_digest, fingerprint, write_json
from reid_demo.benchmark_embeddings import MIEWID_EMBEDDING_DIM, MIEWID_IMAGE_SIZE, MIEWID_REVISION
from reid_demo.benchmark_input_audit import (
    load_inputs, protected_inputs, validate_review_receipt, verify_prepared,
)
from reid_demo.benchmark_clustering import check_output_directory
from reid_demo.benchmark_protocol import LINKS

ROOT = Path(__file__).resolve().parents[1]
VERSION = "czechlynx_native_miewid_adaptation_v1"
SELECTION_RULE = "macro_cross_encounter_recall_at_50; macro_top1; earliest_epoch"
DEFAULT_SETTINGS = {
    "seed": 42, "holdout_fraction": .2, "epochs": 10, "patience": 3,
    "identities_per_batch": 4, "images_per_identity": 2,
    "embedding_lr": 2e-5, "head_lr": 1e-3, "weight_decay": 1e-4,
    "arcface_scale": 30., "arcface_margin": .3, "grayscale_probability": .2,
    "jitter": .1, "clip_norm": 1., "image_size": MIEWID_IMAGE_SIZE,
    "embedding_dim": MIEWID_EMBEDDING_DIM, "revision": MIEWID_REVISION,
    "selection_rule": SELECTION_RULE, "warmup_epochs": 1, "training_precision": "float32",
}


def implementation():
    names = ("reid_demo/benchmark_miewid_adaptation.py", "utils/finetune_czechlynx_miewid.py",
             "reid_demo/benchmark_embeddings.py", "reid_demo/benchmark_protocol.py",
             "reid_demo/benchmark_input_audit.py", "reid_demo/benchmark_cluster_diagnostics.py",
             "reid_demo/benchmark_clustering.py")
    return {name: file_digest(ROOT / name) for name in names}


def environment():
    names = ("torch", "torchvision", "timm", "transformers", "numpy", "pandas", "Pillow")
    return {name: importlib.metadata.version(name) for name in names}


def validate_settings(settings):
    if set(settings) != set(DEFAULT_SETTINGS):
        raise ValueError("Unexpected adaptation settings.")
    for name in ("seed", "epochs", "patience", "identities_per_batch", "images_per_identity"):
        v = settings[name]
        if isinstance(v, bool) or not isinstance(v, int) or v < (0 if name == "seed" else 1):
            raise ValueError(f"Invalid {name}.")
    if settings["identities_per_batch"] < 2 or settings["images_per_identity"] < 2:
        raise ValueError("Use at least two identities and two observations per identity.")
    if not 0 < settings["holdout_fraction"] < 1 or settings["epochs"] <= settings["warmup_epochs"]:
        raise ValueError("Invalid holdout fraction or epoch budget.")
    for name in ("embedding_lr", "head_lr", "weight_decay", "arcface_scale", "clip_norm"):
        if not np.isfinite(settings[name]) or settings[name] <= 0:
            raise ValueError(f"Invalid {name}.")
    if not 0 <= settings["grayscale_probability"] <= 1 or not 0 <= settings["jitter"] <= .5:
        raise ValueError("Invalid photometric augmentation.")
    if not 0 < settings["arcface_margin"] < math.pi / 2:
        raise ValueError("Invalid ArcFace margin.")
    for name in ("revision", "image_size", "embedding_dim", "selection_rule", "warmup_epochs", "training_precision"):
        if settings[name] != DEFAULT_SETTINGS[name]:
            raise ValueError(f"This native trial fixes {name}.")
    return dict(settings)


def _inventory(frame):
    return {"images": len(frame), "identities": int(frame.identity.nunique()),
            "encounters": int(frame.encounter_id.nunique()),
            "sources": {str(k): int(v) for k, v in frame.groupby("source").size().items()}}


def build_inner_plan(frame, outer_plan, *, seed=42, holdout_fraction=.2):
    if not 0 < holdout_fraction < 1 or seed < 0:
        raise ValueError("Invalid inner split settings.")
    pool = outer_plan["pools"]["fitting"]
    indexed = frame.set_index("image_id", drop=False)
    fitting = indexed.loc[pool["image_ids"]].reset_index(drop=True).sort_values("image_id")
    if set(fitting["split-time_open"]) != {"train"}:
        raise ValueError("Fitting inputs must remain official time_open train.")
    groups = pool["identity_groups"]
    if len(groups) < 2:
        raise ValueError("Need at least two atomic fitting groups for inner validation.")
    sources = sorted(fitting.source.unique())
    weights = []
    for group in groups:
        rows = fitting[fitting.identity.isin(group)]
        weights.append(np.array([len(group), *[rows[rows.source == s].identity.nunique() for s in sources]], float))
    target = np.array([1 - holdout_fraction, holdout_fraction])[:, None] * np.sum(weights, axis=0)
    assigned, counts = [[], []], np.zeros_like(target)
    order = sorted(np.random.default_rng(seed).permutation(len(groups)), key=lambda i: -len(groups[i]))
    for position, i in enumerate(order):
        delta = np.sum(((counts + weights[i] - target) ** 2 - (counts - target) ** 2)
                       / np.maximum(target, 1), axis=1)
        role = int(np.argmin(delta))
        if position == len(order) - 1 and not all(assigned):
            role = 0 if not assigned[0] else 1
        assigned[role].append(groups[i])
        counts[role] += weights[i]
    roles = {}
    for name, identity_groups in zip(("train", "validation"), assigned):
        identities = {i for group in identity_groups for i in group}
        rows = fitting[fitting.identity.isin(identities)]
        roles[name] = {"identity_groups": sorted(identity_groups), "image_ids": sorted(rows.image_id),
                       "inventory": _inventory(rows)}
    train = indexed.loc[roles["train"]["image_ids"]]
    val = indexed.loc[roles["validation"]["image_ids"]]
    for field in ("identity", *LINKS):
        a, b = set(train[field].dropna().astype(str)), set(val[field].dropna().astype(str))
        if (a & b) - {""}:
            raise ValueError(f"Inner split overlap: {field}")
    if set(train.image_id) | set(val.image_id) != set(pool["image_ids"]):
        raise ValueError("Inner split must cover all fitting images.")
    result = {"kind": VERSION + "_inner_plan", "outer_protocol": outer_plan["fingerprint"],
              "seed": seed, "holdout_fraction": holdout_fraction, "roles": roles}
    result["fingerprint"] = fingerprint(result)
    return result


def prepare_trial(metadata, protocol_plan, audit_receipt, out_dir, settings):
    settings = validate_settings(settings)
    frame, outer, binding = load_inputs(metadata, protocol_plan)
    review = validate_review_receipt(audit_receipt, frame, binding)
    audit_binding = json.loads((Path(review["audit_dir"]) / "manifest.json").read_text())["binding"]
    protected = protected_inputs(frame, binding) + [Path(audit_receipt).parent, review["audit_dir"],
        review["reviews_source"], audit_binding["images_root"], audit_binding["raw_metadata"]]
    inner = build_inner_plan(frame, outer, seed=settings["seed"], holdout_fraction=settings["holdout_fraction"])
    if inner["roles"]["train"]["inventory"]["identities"] < settings["identities_per_batch"]:
        raise ValueError("Not enough training identities for the requested batch.")
    fitting = frame[frame.image_id.isin(outer["pools"]["fitting"]["image_ids"])]
    verify_prepared(fitting)
    spec = {"kind": VERSION, "binding": binding, "inner_plan": inner,
            "audit_receipt": str(Path(audit_receipt).resolve()), "audit_receipt_sha256": file_digest(audit_receipt),
            "settings": settings, "implementation": implementation(), "environment": environment(),
            "fitting_input_digest": fingerprint(fitting[["image_id", "image_sha256", "mask_sha256"]].to_dict("records"))}
    out = Path(out_dir).resolve()
    if not (out / "manifest.json").exists():
        check_output_directory(out, protected)
    # Output overlap is forbidden even when a prepared manifest exists.
    for value in protected:
        p = Path(value).resolve()
        if out == p or out.is_relative_to(p) or p.is_relative_to(out):
            raise ValueError("Trial output overlaps protected input/review data.")
    manifest = {**spec, "status": "prepared", "run_fingerprint": fingerprint(spec)}
    if (out / "manifest.json").exists():
        saved = json.loads((out / "manifest.json").read_text())
        if (saved != manifest or json.loads((out / "inner_plan.json").read_text()) != inner
                or {p.name for p in out.iterdir()} != {"manifest.json", "inner_plan.json"}):
            raise ValueError("Trial is stale, started or complete; choose a new directory. No training resume.")
    else:
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "inner_plan.json", inner)
        write_json(out / "manifest.json", manifest)
    return frame, outer, manifest


def _camera(row):
    value = row.get("trap_id")
    return None if pd.isna(value) or not str(value).strip() else (str(row["source"]), str(value))


class EncounterBatchSampler:
    """Balance labelled identities; encounters/cameras diversify, never label, examples."""
    def __init__(self, frame, identities_per_batch=4, images_per_identity=2, seed=42):
        self.frame = frame.reset_index(drop=True)
        self.p, self.k, self.seed = identities_per_batch, images_per_identity, seed
        self.by_id = {str(i): rows.index.to_numpy() for i, rows in self.frame.groupby("identity", sort=True)}
        if self.p < 2 or self.k < 2 or len(self.by_id) < self.p:
            raise ValueError("Invalid balanced batch or insufficient identities.")
        self.epoch, self.audit = 0, Counter()

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return max(1, math.ceil(len(self.frame) / (self.p * self.k)))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.audit = Counter()
        rows = self.frame.to_dict("records")
        for _ in range(len(self)):
            batch = []
            for identity in rng.choice(sorted(self.by_id), self.p, replace=False):
                candidates = self.by_id[identity].tolist()
                selected = [int(rng.choice(candidates))]
                for _ in range(self.k - 1):
                    unused = [i for i in candidates if i not in selected]
                    encounters = {rows[i]["encounter_id"] for i in selected}
                    cameras = {_camera(rows[i]) for i in selected}
                    diverse = [i for i in unused if rows[i]["encounter_id"] not in encounters]
                    different_camera = [i for i in diverse if _camera(rows[i]) is not None and _camera(rows[i]) not in cameras]
                    options = different_camera or diverse or unused or candidates
                    index = int(rng.choice(options))
                    if index in selected:
                        self.audit["repeated_image_observations"] += 1
                    if not diverse:
                        self.audit["cross_encounter_unavailable"] += 1
                    selected.append(index)
                batch.extend(selected)
                self.audit["identity_draws"] += 1
            yield batch


class ImageDataset:
    def __init__(self, frame, transform, labels=None):
        self.rows = frame.to_dict("records")
        self.transform, self.labels = transform, labels or {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as im:
            tensor = self.transform(im.convert("RGB"))
        return tensor, self.labels.get(str(row["identity"]), -1), str(row["image_id"])


def build_transform(settings, *, training=False):
    from torchvision import transforms as T
    ops = [T.Resize((settings["image_size"], settings["image_size"]), interpolation=T.InterpolationMode.BILINEAR)]
    if training:
        j = settings["jitter"]
        ops.extend([T.ColorJitter(brightness=j, contrast=j, saturation=j),
                    T.RandomGrayscale(p=settings["grayscale_probability"])])
    return T.Compose([*ops, T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])


def make_head(dimension, classes, *, scale=30., margin=.3):
    import torch
    from torch import nn
    from torch.nn import functional as F
    class ArcHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(classes, dimension))
            nn.init.xavier_uniform_(self.weight)

        def forward(self, features, labels):
            cosine = F.linear(F.normalize(features.float(), dim=1), F.normalize(self.weight.float(), dim=1)).clamp(-1, 1)
            sine = torch.sqrt((1 - cosine.square()).clamp(min=1e-7))
            phi = cosine * math.cos(margin) - sine * math.sin(margin)
            phi = torch.where(cosine > math.cos(math.pi - margin), phi, cosine - math.sin(math.pi - margin) * margin)
            one_hot = F.one_hot(labels, classes).to(cosine.dtype)
            return scale * (one_hot * phi + (1 - one_hot) * cosine)
    return ArcHead()


def training_mode(model, *, warmup=False):
    import torch
    model.train()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not warmup and not name.startswith("final."))
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def native_state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def selection_key(metrics):
    def number(key):
        value = metrics.get(key)
        return -1. if value is None else float(value)
    return number("macro_recall_at_50"), number("macro_top1")


def inner_metrics(frame, matrix, *, appearance=None, chunk_size=64):
    from reid_demo.benchmark_clustering import candidate_neighbours, inference_inputs
    from reid_demo.benchmark_cluster_diagnostics import prepare_evaluation, oracle_connectivity
    order = np.argsort(frame.image_id.astype(str).to_numpy(), kind="stable")
    frame = frame.iloc[order].reset_index(drop=True)
    matrix = np.asarray(matrix)[order]
    ids, vectors, valid = inference_inputs(frame.image_id.tolist(), matrix)
    identities = frame.identity.astype(str).to_numpy()
    fields = {name: frame[name].astype(str).to_numpy() for name in LINKS}
    rows, groups = [], {}
    for start in range(0, len(ids), chunk_size):
        block = vectors[start:start + chunk_size] @ vectors.T
        for offset, scores in enumerate(block):
            i = start + offset
            allowed = np.ones(len(ids), bool)
            for values in fields.values():
                allowed &= values != values[i]
            positives = allowed & (identities == identities[i])
            if not positives.any():
                rows.append({"image_id": ids[i], "identity": identities[i], "eligible": False,
                             "reason": "no_cross_encounter_nonduplicate_positive"})
                continue
            ranked = np.argsort(-np.where(allowed & valid, scores, -np.inf), kind="stable")
            ranked = ranked[(allowed & valid)[ranked]] if valid[i] else np.array([], int)
            hit = bool(any(positives[j] for j in ranked[:50]))
            top1 = bool(len(ranked) and positives[ranked[0]])
            row = {"image_id": ids[i], "identity": identities[i], "eligible": True,
                   "recall_at_50": hit, "top1": top1, "valid_vector": bool(valid[i])}
            rows.append(row)
            cameras = [_camera(r) for r in frame.loc[positives].to_dict("records")]
            camera = _camera(frame.iloc[i].to_dict())
            category = "camera_unknown" if camera is None else ("same_camera_positive_available" if camera in cameras else "other_camera_only")
            groups.setdefault(category, []).append(row)
            proxy = (appearance or {}).get(ids[i], {}).get("near_monochrome")
            ref = [(appearance or {}).get(ids[j], {}).get("near_monochrome") for j in np.flatnonzero(positives)]
            category = "appearance_unknown" if proxy is None or None in ref else ("same_proxy_positive_available" if proxy in ref else "opposite_proxy_only")
            groups.setdefault(category, []).append(row)
    eligible = [r for r in rows if r["eligible"]]
    def summary(values):
        if not values:
            return {"queries": 0, "identities": 0, "macro_recall_at_50": None, "macro_top1": None}
        df = pd.DataFrame(values)
        means = df.groupby("identity")[["recall_at_50", "top1"]].mean()
        return {"queries": len(values), "identities": len(means),
                "macro_recall_at_50": float(means.recall_at_50.mean()), "macro_top1": float(means.top1.mean())}
    metrics = {**summary(eligible), "images": len(frame), "unresolved_images": int((~valid).sum()),
               "ineligible_query_ids": [r["image_id"] for r in rows if not r["eligible"]],
               "ineligible_identities": sorted(set(identities) - {r["identity"] for r in eligible}),
               "groups": {key: summary(value) for key, value in groups.items()},
               "notes": "Inner unseen identities; colour proxies are not night/IR. Oracle uses truth only after inference."}
    candidates = candidate_neighbours(ids, matrix, [15, 50], chunk_size=chunk_size)
    inputs = prepare_evaluation(ids, matrix, frame, appearance)
    metrics["oracle_connectivity"] = {str(k): oracle_connectivity(inputs, edges)[0] for k, edges in candidates.items()}
    return metrics, rows


def extract_matrix(model, frame, settings, device, *, batch_size=8, workers=0):
    import torch
    from torch.utils.data import DataLoader
    from torch.nn import functional as F
    model.eval()
    loader = DataLoader(ImageDataset(frame, build_transform(settings)), batch_size=batch_size,
                        num_workers=workers, shuffle=False, pin_memory=device.type == "cuda")
    parts, ids = [], []
    with torch.no_grad():
        for images, _, batch_ids in loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                vectors = model(images.to(device, non_blocking=True))
            if vectors.ndim != 2 or vectors.shape[1] != settings["embedding_dim"] or not torch.isfinite(vectors).all():
                raise ValueError("Native embedding returned invalid values or dimension.")
            parts.append(F.normalize(vectors.float(), dim=1).cpu().numpy())
            ids.extend(batch_ids)
    if ids != frame.image_id.tolist() or not parts:
        raise ValueError("Extraction changed observation membership/order.")
    return np.concatenate(parts).astype(np.float32)


def save_checkpoint(path, model, head, manifest, epoch, metrics, model_info):
    import torch
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    payload = {"kind": VERSION + "_checkpoint", "schema": 1, "epoch": epoch, "model_state": state,
               "model_state_sha256": native_state_digest(state), "training_head_state": head.state_dict(),
               "trial_fingerprint": manifest["run_fingerprint"], "settings": manifest["settings"],
               "inner_plan_fingerprint": manifest["inner_plan"]["fingerprint"], "binding": manifest["binding"],
               "model_info": model_info, "metrics": metrics}
    temporary = Path(path).with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_completed_trial(trial_dir, frame, outer, binding):
    out = Path(trial_dir).resolve()
    manifest = json.loads((out / "manifest.json").read_text())
    settings = validate_settings(manifest["settings"])
    spec_keys = ("kind", "binding", "inner_plan", "audit_receipt", "audit_receipt_sha256", "settings",
                 "implementation", "environment", "fitting_input_digest")
    if (manifest.get("kind") != VERSION or manifest.get("status") != "complete"
            or manifest.get("binding") != binding or manifest.get("implementation") != implementation()
            or manifest.get("environment") != environment()
            or manifest.get("run_fingerprint") != fingerprint({k: manifest[k] for k in spec_keys})):
        raise ValueError("Incomplete/stale/incompatible training trial.")
    required_artifacts = {"inner_plan.json", "holdout_appearance.json", "inner_validation_latest.json",
                          "epochs.json", "best.pt", "last.pt"}
    if set(manifest.get("artifacts", {})) != required_artifacts:
        raise ValueError("Completed trial is missing required artifacts.")
    inner = build_inner_plan(frame, outer, seed=settings["seed"], holdout_fraction=settings["holdout_fraction"])
    if inner != manifest["inner_plan"] or json.loads((out / "inner_plan.json").read_text()) != inner:
        raise ValueError("Inner fitting membership changed.")
    validate_review_receipt(manifest["audit_receipt"], frame, binding)
    if file_digest(manifest["audit_receipt"]) != manifest["audit_receipt_sha256"]:
        raise ValueError("Training review receipt changed.")
    verify_prepared(frame[frame.image_id.isin(outer["pools"]["fitting"]["image_ids"])])
    for name, digest in manifest["artifacts"].items():
        path = (out / name).resolve()
        if not path.is_relative_to(out) or file_digest(path) != digest:
            raise ValueError("Changed trial artifact.")
    epochs = json.loads((out / "epochs.json").read_text())
    eligible = [r for r in epochs if r["epoch"] == 0 or r["epoch"] > settings["warmup_epochs"]]
    selected = max(eligible, key=lambda r: (*selection_key(r["metrics"]), -r["epoch"]))
    if (selected["epoch"] != manifest["best_epoch"] or selected["metrics"] != manifest["best_metrics"]
            or epochs[0]["metrics"] != manifest["baseline_metrics"] or not manifest.get("gradient_proof")):
        raise ValueError("Trial selection/gradient evidence differs from epoch history.")
    return manifest


def read_checkpoint(path, manifest):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("kind") != VERSION + "_checkpoint" or payload.get("schema") != 1
            or payload.get("trial_fingerprint") != manifest["run_fingerprint"]
            or payload.get("settings") != manifest["settings"] or payload.get("binding") != manifest["binding"]
            or payload.get("inner_plan_fingerprint") != manifest["inner_plan"]["fingerprint"]
            or payload.get("model_info") != manifest["model_info"]
            or native_state_digest(payload["model_state"]) != payload.get("model_state_sha256")):
        raise ValueError("Invalid native checkpoint/provenance.")
    return payload


def load_adapted_model(path, manifest, device):
    from reid_demo.benchmark_embeddings import load_miewid
    payload = read_checkpoint(path, manifest)
    if payload["epoch"] != manifest["best_epoch"] or selection_key(payload["metrics"]) != selection_key(manifest["best_metrics"]):
        raise ValueError("Use the trial's selected best checkpoint.")
    model, info, size = load_miewid(device, revision=manifest["settings"]["revision"])
    if info != payload["model_info"] or size != manifest["settings"]["image_size"]:
        raise ValueError("Publisher architecture/source differs from the training initialization.")
    try:
        model.load_state_dict(payload["model_state"], strict=True)
    except RuntimeError as exc:
        raise ValueError("Native checkpoint state keys/shapes differ from pinned architecture.") from exc
    model.eval().requires_grad_(False)
    return model, {**info, "kind": "adapted_miewid_msv3", "adaptation_epoch": payload["epoch"],
                   "checkpoint_sha256": file_digest(path), "trial_fingerprint": manifest["run_fingerprint"],
                   "native_state_sha256": payload["model_state_sha256"]}, size
