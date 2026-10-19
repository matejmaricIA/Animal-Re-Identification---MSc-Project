"""Facing-aware re-adaptation of native MiewID on the CzechLynx fitting pool (work item (b), 18 Oct 2026).

Two arms share the v2 recipe and inner split; only the class definition changes:
- "classes": one ArcFace class per (animal, facing in {left, right, other});
- "subcenter": one class per animal with K sub-centers (label-free view modes).
Fitting facing labels come from pose keypoints first, then an identity-cross-fitted facing classifier on
public MiewID vectors, else "other". Checkpoints are selected by same-facing cross-encounter top-1 on the
inner holdout. Calibration only downstream; selection stays reserved.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from reid_demo.benchmark import file_digest, fingerprint, write_json
from reid_demo.benchmark_clustering import check_output_directory
from reid_demo.benchmark_input_audit import load_inputs, protected_inputs, validate_review_receipt, verify_prepared
from reid_demo.benchmark_protocol import LINKS
from reid_demo import benchmark_miewid_adaptation as ad

ROOT = Path(__file__).resolve().parents[1]
VERSION = "czechlynx_facing_miewid_adaptation_v1"
ARMS = ("classes", "subcenter")
FACING_NAMES = {0: "left", 1: "right", -1: "other"}
CONFIDENT = (0.2, 0.8)
SUBCENTERS = 3
SELECTION_RULE = "macro_same_facing_top1; macro_recall_at_50; earliest_epoch"
V2_INNER_PLAN_FINGERPRINT = "ef8d64056a89ef7bce2e37db9d089bd86ce92ff12b922f9f4acbd951db0c88e7"
DEFAULT_SETTINGS = {**ad.DEFAULT_SETTINGS, "selection_rule": SELECTION_RULE, "arm": "classes",
                    "subcenters": SUBCENTERS, "facing_confident": list(CONFIDENT), "horizontal_flip": False}
LABEL_COLUMNS = ["image_id", "identity", "pose_facing", "p_faces_right", "facing", "facing_source"]


def implementation():
    names = ("reid_demo/benchmark_facing_adaptation.py", "utils/finetune_czechlynx_miewid_facing.py",
             "utils/finetune_czechlynx_miewid.py", "tools/label_czechlynx_fitting_facing.py")
    return {**ad.implementation(), **{n: file_digest(ROOT / n) for n in names}}


def validate_settings(settings):
    if set(settings) != set(DEFAULT_SETTINGS):
        raise ValueError("Unexpected facing-adaptation settings.")
    if settings["arm"] not in ARMS:
        raise ValueError(f"Arm must be one of {ARMS}.")
    for name in ("selection_rule", "subcenters", "facing_confident", "horizontal_flip"):
        if settings[name] != DEFAULT_SETTINGS[name]:
            raise ValueError(f"This trial fixes {name}.")
    base = {k: v for k, v in settings.items() if k in ad.DEFAULT_SETTINGS}
    ad.validate_settings({**base, "selection_rule": ad.SELECTION_RULE})
    return dict(settings)


# --- Fitting facing labels ------------------------------------------------------------------

def facing_labels(pose, p_right, confident=CONFIDENT):
    """Pose first, then a confident prediction, else 'other' (-1). Returns (labels, sources)."""
    pose, p_right = np.asarray(pose, float), np.asarray(p_right, float)
    has_pose = ~np.isnan(pose)
    predicted = np.where(p_right > confident[1], 1, np.where(p_right < confident[0], 0, -1))
    labels = np.where(has_pose, np.nan_to_num(pose, nan=-1).astype(int), predicted).astype(int)
    sources = np.where(has_pose, "pose", np.where(predicted >= 0, "predicted", "other"))
    return labels, sources


def build_facing_labels(frame, vectors, pose, *, seed=42):
    """Identity-cross-fitted facing for every fitting photo (a model never sees the photo's animal)."""
    from reid_demo import benchmark_edge_grouping as eg
    from reid_demo import benchmark_flank_grouping as fg
    codes = pd.factorize(frame.identity.astype(str), sort=True)[0]
    folds = eg.identity_folds(codes, seed=seed)
    p = fg.facing_probabilities(vectors, np.asarray(pose, float), folds)
    labels, sources = facing_labels(pose, p)
    table = pd.DataFrame({"image_id": frame.image_id.astype(str).to_numpy(), "identity": frame.identity.astype(str).to_numpy(),
                          "pose_facing": pose, "p_faces_right": p, "facing": labels, "facing_source": sources})
    validation = fg.facing_validation(p, np.asarray(pose, float), table.image_id.tolist())
    return table, validation


def load_label_table(labels_dir, image_ids):
    """Read a completed label directory; it must cover exactly the requested fitting photos."""
    labels_dir = Path(labels_dir).resolve()
    manifest = json.loads((labels_dir / "manifest.json").read_text())
    path = labels_dir / "labels.csv"
    if manifest.get("status") != "complete" or manifest.get("labels_sha256") != file_digest(path):
        raise ValueError("Facing labels are incomplete or changed.")
    table = pd.read_csv(path, dtype={"image_id": str, "identity": str})
    if list(table.columns) != LABEL_COLUMNS or set(table.facing) - {-1, 0, 1}:
        raise ValueError("Invalid facing label table.")
    if table.image_id.duplicated().any() or set(table.image_id) != set(map(str, image_ids)):
        raise ValueError("Facing labels must cover exactly the fitting photos.")
    return table, {"path": str(path), "sha256": manifest["labels_sha256"],
                   "manifest_sha256": file_digest(labels_dir / "manifest.json")}


def class_keys(frame, facing, arm):
    """Training class per photo: (animal, facing) for 'classes'; the animal for 'subcenter'."""
    identities = frame.identity.astype(str).to_numpy()
    if arm == "subcenter":
        return identities
    if arm != "classes":
        raise ValueError(f"Unknown arm: {arm}")
    return np.asarray([f"{i}|{FACING_NAMES[int(f)]}" for i, f in zip(identities, facing)])


# --- Heads ----------------------------------------------------------------------------------

def make_subcenter_head(dimension, classes, *, k=SUBCENTERS, scale=30., margin=.3):
    """Sub-center ArcFace (Deng et al. 2020): class cosine = max over k sub-centers; margin on the target."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    class SubCenterArcHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.k, self.classes = k, classes
            self.weight = nn.Parameter(torch.empty(classes * k, dimension))
            nn.init.xavier_uniform_(self.weight)

        def class_cosine(self, features):
            cosine = F.linear(F.normalize(features.float(), dim=1), F.normalize(self.weight.float(), dim=1))
            return cosine.view(-1, self.classes, self.k).max(dim=2).values.clamp(-1, 1)

        def forward(self, features, labels):
            cosine = self.class_cosine(features)
            sine = torch.sqrt((1 - cosine.square()).clamp(min=1e-7))
            phi = cosine * math.cos(margin) - sine * math.sin(margin)
            phi = torch.where(cosine > math.cos(math.pi - margin), phi, cosine - math.sin(math.pi - margin) * margin)
            one_hot = F.one_hot(labels, classes).to(cosine.dtype)
            return scale * (one_hot * phi + (1 - one_hot) * cosine)
    return SubCenterArcHead()


# --- Facing-stratified retrieval and selection ----------------------------------------------

def facing_retrieval(frame, matrix, facing, *, mode="same", depth=50):
    """Cross-encounter, non-duplicate retrieval among photos with known facing (0/1).

    mode 'same': candidates share the query's facing; 'opposite': the other facing; 'any': both."""
    if mode not in ("same", "opposite", "any"):
        raise ValueError("Invalid facing mode.")
    vectors = np.asarray(matrix, float)
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > 0
    vectors = np.where(valid[:, None], vectors / np.where(norms > 0, norms, 1)[:, None], 0)
    facing = np.asarray(facing, int)
    identities = frame.identity.astype(str).to_numpy()
    fields = [frame[name].astype(str).to_numpy() for name in LINKS]
    rows = []
    for i in np.flatnonzero((facing >= 0) & valid):
        allowed = valid & (facing >= 0)
        for values in fields:
            allowed &= values != values[i]
        if mode == "same":
            allowed &= facing == facing[i]
        elif mode == "opposite":
            allowed &= facing == 1 - facing[i]
        positives = allowed & (identities == identities[i])
        if not positives.any():
            continue
        candidates = np.flatnonzero(allowed)
        scores = vectors[candidates] @ vectors[i]
        order = candidates[np.argsort(-scores, kind="stable")]
        rows.append({"identity": identities[i], "top1": bool(positives[order[0]]),
                     "recall": bool(positives[order[:depth]].any())})
    if not rows:
        return {"queries": 0, "identities": 0, "micro_top1": None, "macro_top1": None, "macro_recall": None}
    df = pd.DataFrame(rows)
    per = df.groupby("identity")[["top1", "recall"]].mean()
    return {"queries": len(df), "identities": len(per), "micro_top1": float(df.top1.mean()),
            "macro_top1": float(per.top1.mean()), f"macro_recall_at_{depth}": float(per.recall.mean()),
            "macro_recall": float(per.recall.mean())}


def selection_key(metrics):
    def number(key):
        value = metrics.get(key)
        return -1. if value is None else float(value)
    return number("macro_same_facing_top1"), number("macro_recall_at_50")


def epoch_metrics(frame, matrix, facing, *, appearance=None, chunk_size=64):
    """v2 inner-holdout metrics plus facing-stratified retrieval; the selection key reads the first."""
    metrics, rows = ad.inner_metrics(frame, matrix, appearance=appearance, chunk_size=chunk_size)
    strata = {mode: facing_retrieval(frame, matrix, facing, mode=mode) for mode in ("same", "opposite", "any")}
    metrics.update(facing_strata=strata, macro_same_facing_top1=strata["same"]["macro_top1"],
                   v2_selection_key=list(ad.selection_key(metrics)))
    return metrics, rows


# --- Trial preparation, checkpoints and loading ---------------------------------------------

SPEC_KEYS = ("kind", "binding", "inner_plan", "audit_receipt", "audit_receipt_sha256", "settings",
             "implementation", "environment", "fitting_input_digest", "facing_labels")


def prepare_trial(metadata, protocol_plan, audit_receipt, labels_dir, out_dir, settings,
                  expected_inner=V2_INNER_PLAN_FINGERPRINT):
    settings = validate_settings(settings)
    frame, outer, binding = load_inputs(metadata, protocol_plan)
    review = validate_review_receipt(audit_receipt, frame, binding)
    audit_binding = json.loads((Path(review["audit_dir"]) / "manifest.json").read_text())["binding"]
    protected = protected_inputs(frame, binding) + [Path(audit_receipt).parent, review["audit_dir"],
        review["reviews_source"], audit_binding["images_root"], audit_binding["raw_metadata"], Path(labels_dir)]
    inner = ad.build_inner_plan(frame, outer, seed=settings["seed"], holdout_fraction=settings["holdout_fraction"])
    if expected_inner is not None and inner["fingerprint"] != expected_inner:
        raise ValueError("Inner fitting split differs from the v2 trial's.")
    fitting = frame[frame.image_id.isin(outer["pools"]["fitting"]["image_ids"])]
    _, label_binding = load_label_table(labels_dir, fitting.image_id)
    verify_prepared(fitting)
    spec = {"kind": VERSION, "binding": binding, "inner_plan": inner,
            "audit_receipt": str(Path(audit_receipt).resolve()), "audit_receipt_sha256": file_digest(audit_receipt),
            "settings": settings, "implementation": implementation(), "environment": ad.environment(),
            "fitting_input_digest": fingerprint(fitting[["image_id", "image_sha256", "mask_sha256"]].to_dict("records")),
            "facing_labels": label_binding}
    out = Path(out_dir).resolve()
    if not (out / "manifest.json").exists():
        check_output_directory(out, protected)
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


def save_checkpoint(path, model, head, manifest, epoch, metrics, model_info):
    import torch
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    payload = {"kind": VERSION + "_checkpoint", "schema": 1, "epoch": epoch, "model_state": state,
               "model_state_sha256": ad.native_state_digest(state), "training_head_state": head.state_dict(),
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
    if (manifest.get("kind") != VERSION or manifest.get("status") != "complete"
            or manifest.get("binding") != binding or manifest.get("implementation") != implementation()
            or manifest.get("environment") != ad.environment()
            or manifest.get("run_fingerprint") != fingerprint({k: manifest[k] for k in SPEC_KEYS})):
        raise ValueError("Incomplete/stale/incompatible facing trial.")
    required = {"inner_plan.json", "holdout_appearance.json", "inner_validation_latest.json", "epochs.json",
                "best.pt", "last.pt"}
    if set(manifest.get("artifacts", {})) != required:
        raise ValueError("Completed trial is missing required artifacts.")
    inner = ad.build_inner_plan(frame, outer, seed=settings["seed"], holdout_fraction=settings["holdout_fraction"])
    if inner != manifest["inner_plan"] or json.loads((out / "inner_plan.json").read_text()) != inner:
        raise ValueError("Inner fitting membership changed.")
    validate_review_receipt(manifest["audit_receipt"], frame, binding)
    if file_digest(manifest["audit_receipt"]) != manifest["audit_receipt_sha256"]:
        raise ValueError("Training review receipt changed.")
    if file_digest(manifest["facing_labels"]["path"]) != manifest["facing_labels"]["sha256"]:
        raise ValueError("Facing labels changed since training.")
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
            or ad.native_state_digest(payload["model_state"]) != payload.get("model_state_sha256")):
        raise ValueError("Invalid facing checkpoint/provenance.")
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
        raise ValueError("Checkpoint state keys/shapes differ from pinned architecture.") from exc
    model.eval().requires_grad_(False)
    return model, {**info, "kind": "facing_adapted_miewid_msv3", "arm": manifest["settings"]["arm"],
                   "adaptation_epoch": payload["epoch"], "checkpoint_sha256": file_digest(path),
                   "trial_fingerprint": manifest["run_fingerprint"],
                   "native_state_sha256": payload["model_state_sha256"]}, size


# --- Calibration comparison helpers ---------------------------------------------------------

def nn_top1_by_facing(vectors, codes, encounters, side):
    """Cross-encounter nearest-neighbour identity top-1 among confident-facing photos (any/same/opposite)."""
    x = np.asarray(vectors, float)
    codes, encounters, side = np.asarray(codes), np.asarray(encounters).astype(str), np.asarray(side, int)
    scores = x @ x.T
    out = {}
    for mode in ("any", "same", "opposite"):
        hits, ids = [], []
        for i in np.flatnonzero(side >= 0):
            mask = (encounters != encounters[i]) & (side >= 0)
            if mode == "same":
                mask &= side == side[i]
            elif mode == "opposite":
                mask &= side == 1 - side[i]
            mask[i] = False
            if not (mask & (codes == codes[i])).any():
                continue
            j = np.flatnonzero(mask)[np.argmax(scores[i, mask])]
            hits.append(codes[j] == codes[i])
            ids.append(codes[i])
        hits, ids = np.asarray(hits, bool), np.asarray(ids)
        out[mode] = {"queries": int(len(hits)),
                     "micro_top1": float(hits.mean()) if len(hits) else None,
                     "macro_top1": float(np.mean([hits[ids == c].mean() for c in np.unique(ids)])) if len(hits) else None}
    return out
