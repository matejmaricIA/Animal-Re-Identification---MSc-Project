"""Frozen review operating point and its one-shot selection evaluation (work item (c), 19 Oct 2026).

The freeze step (calibration only) fits the W4 confidence model, the P(match) model and a facing
classifier on all of calibration and stores them as plain JSON, together with frozen identifiability-risk
reference distributions and every binding. The selection runner verifies the freeze record, builds the
same label-free context on unseen animals and runs the unchanged (a) simulation once. Truth is read only
by the simulated reviewers and for evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from reid_demo.benchmark import file_digest, fingerprint
from reid_demo.benchmark_clustering import ROOT
from reid_demo import benchmark_gated_matching as gm
from reid_demo import benchmark_nonmatch_pruning as nm

FREEZE_VERSION = "czechlynx_frozen_review_config_v1"
SELECTION_VERSION = "czechlynx_frozen_review_selection_v1"
PRIMARY_TAU = 0.99
SECONDARY_TAU = 0.98
EPSILON = 0.001
REVIEWER_ERRORS = (0.0, 0.03)
AUTO_AUDIT = 100
COUNT_AUDIT = 10
RISK_SIGNALS = ("small_animal", "low_contrast", "fragmented_mask", "truncated")
FACING_SCALE, FACING_C = 10.0, 1.0
STAGES = ("verification", "extraction", "context", "simulation", "report", "complete")
CODE = ("reid_demo/benchmark_frozen_review.py", "tools/freeze_czechlynx_review_config.py",
        "tools/evaluate_czechlynx_frozen_review.py", "reid_demo/benchmark_nonmatch_pruning.py",
        "reid_demo/benchmark_gated_matching.py", "reid_demo/benchmark_review_simulation.py",
        "reid_demo/benchmark_facing_adaptation.py", "reid_demo/benchmark_coverage_diagnostics.py",
        "tools/evaluate_czechlynx_facing_adaptation.py")


def code_hashes():
    return {name: file_digest(ROOT / name) for name in CODE}


def operating_point(tau):
    return {**nm.DEFAULTS, "order": "pmatch", "eps": EPSILON, "tau": tau}


# --- Frozen models --------------------------------------------------------------------------

class FrozenLogistic:
    """A standardized logistic model stored as JSON; predict_proba matches the sklearn pipeline."""

    def __init__(self, payload):
        self.columns = list(payload["columns"])
        self.mean = np.asarray(payload["mean"], float)
        self.scale = np.asarray(payload["scale"], float)
        self.coef = np.asarray(payload["coef"], float)
        self.intercept = float(payload["intercept"])
        if not (len(self.columns) == len(self.mean) == len(self.scale) == len(self.coef)):
            raise ValueError("Frozen model dimensions differ.")

    @classmethod
    def from_pipeline(cls, model, columns):
        scaler, lr = model.steps[0][1], model.steps[-1][1]
        return cls({"columns": list(columns), "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                    "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])})

    def payload(self):
        return {"columns": self.columns, "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "coef": self.coef.tolist(), "intercept": self.intercept}

    def probability(self, x):
        z = ((np.asarray(x, float) - self.mean) / self.scale) @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    def predict_proba(self, x):
        p = self.probability(x)
        return np.column_stack([1 - p, p])


def fit_logistic(frame, columns, target, *, c=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    y = frame[target].to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("Training records lack both outcomes.")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=c)).fit(
        frame[list(columns)].to_numpy(float), y)
    return FrozenLogistic.from_pipeline(model, columns)


class FrozenFacing:
    """p(faces right) from L2-normalized vectors (same scaling as the flank work), fixed after fitting."""

    def __init__(self, payload):
        self.coef = np.asarray(payload["coef"], float)
        self.intercept = float(payload["intercept"])
        self.scale = float(payload["scale"])

    @classmethod
    def fit(cls, vectors, pose):
        from sklearn.linear_model import LogisticRegression
        labelled = ~np.isnan(np.asarray(pose, float))
        y = np.asarray(pose)[labelled]
        if len(np.unique(y)) < 2:
            raise ValueError("Facing labels lack both classes.")
        model = LogisticRegression(max_iter=5000, C=FACING_C).fit(np.asarray(vectors, float)[labelled] * FACING_SCALE, y)
        return cls({"coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]), "scale": FACING_SCALE})

    def payload(self):
        return {"coef": self.coef.tolist(), "intercept": self.intercept, "scale": self.scale}

    def probability(self, vectors):
        z = (np.asarray(vectors, float) * self.scale) @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-z))


def risk_signals(quality):
    return pd.DataFrame({
        "small_animal": -np.sqrt(quality.mask_pixels.astype(float)),
        "low_contrast": -quality.luminance_std.astype(float),
        "fragmented_mask": 1 - quality.largest_component_share.astype(float),
        "truncated": quality.border_mask_fraction.astype(float)})


def risk_reference(quality):
    signals = risk_signals(quality)
    return {name: np.sort(signals[name].to_numpy(float)).tolist() for name in RISK_SIGNALS}


def frozen_risk(quality, reference):
    """Mean mid-rank percentile of each signal against the frozen calibration distribution."""
    signals = risk_signals(quality)
    ranks = []
    for name in RISK_SIGNALS:
        ref = np.asarray(reference[name], float)
        values = signals[name].to_numpy(float)
        below = np.searchsorted(ref, values, side="left")
        at_or_below = np.searchsorted(ref, values, side="right")
        ranks.append((below + 0.5 * (at_or_below - below)) / len(ref))
    return np.mean(ranks, axis=0)


# --- Freeze record --------------------------------------------------------------------------

def build_record(models, bindings, calibration_reference):
    """The frozen configuration; its fingerprint covers every setting, model and binding."""
    record = {"kind": FREEZE_VERSION,
              "operating_points": {"primary": operating_point(PRIMARY_TAU), "secondary": operating_point(SECONDARY_TAU)},
              "reviewer_errors": list(REVIEWER_ERRORS), "auto_audit": AUTO_AUDIT, "count_audit": COUNT_AUDIT,
              "cost_settings": nm.COST_SETTINGS, "bundling": "trap_day", "pair_columns": list(nm.PAIR_COLUMNS),
              "models": models, "bindings": bindings, "calibration_reference": calibration_reference,
              "criteria": {"S1": "misassigned <= 1% and |count error| <= 1 (primary, perfect reviewer)",
                           "S2": "decisions per 1,000 photos <= 1.25 x calibration and >= 50% fewer than W1 on selection",
                           "S3": "automation audit covers true bundle error and count audit covers true count",
                           "S4": "3% reviewer error: |count error| <= 2 and misassigned <= 3%"}}
    record["fingerprint"] = fingerprint(record)
    return record


def load_record(directory):
    directory = Path(directory)
    record = json.loads((directory / "config.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    body = {k: v for k, v in record.items() if k != "fingerprint"}
    if (record.get("kind") != FREEZE_VERSION or fingerprint(body) != record.get("fingerprint")
            or manifest.get("status") != "complete" or manifest.get("fingerprint") != record["fingerprint"]
            or manifest.get("config_sha256") != file_digest(directory / "config.json")):
        raise ValueError("Frozen configuration is incomplete or was changed.")
    return record


def verify_bindings(record, *, metadata, protocol_plan, raw_metadata, trial_dir):
    """Every frozen input and the code must be byte-identical to the freeze."""
    b = record["bindings"]
    current = {"metadata_sha256": file_digest(metadata), "protocol_plan_sha256": file_digest(protocol_plan),
               "raw_metadata_sha256": file_digest(raw_metadata),
               "trial_manifest_sha256": file_digest(Path(trial_dir) / "manifest.json"),
               "checkpoint_sha256": file_digest(Path(trial_dir) / "best.pt"), "code": code_hashes()}
    problems = [k for k, v in current.items() if b.get(k) != v]
    if problems:
        raise ValueError(f"Frozen bindings differ: {problems}")


def models_from_record(record):
    m = record["models"]
    return FrozenLogistic(m["confidence"]), FrozenLogistic(m["pmatch"]), FrozenFacing(m["facing"])


# --- Label-free context and simulation ------------------------------------------------------

def coordinates(raw_metadata, frame):
    header = pd.read_csv(raw_metadata, nrows=0).columns
    if not {"path", "latitude", "longitude"}.issubset(header):
        return np.full(len(frame), np.nan), np.full(len(frame), np.nan)
    table = pd.read_csv(raw_metadata, usecols=["path", "latitude", "longitude"]).set_index("path")
    return (frame.source_path.map(table.latitude).to_numpy(float), frame.source_path.map(table.longitude).to_numpy(float))


def build_context(frame, vectors, raw_metadata, record):
    """Context exactly as deployed: frozen facing, frozen-reference risk, day-level space-time."""
    from reid_demo.benchmark_coverage_diagnostics import image_quality_features, score_matrix
    from reid_demo.benchmark_review_simulation import bundle_keys
    _, _, facing_model = models_from_record(record)
    quality = image_quality_features(frame)
    if quality.image_id.tolist() != frame.image_id.astype(str).tolist():
        raise ValueError("Quality features are misaligned.")
    risk = frozen_risk(quality, record["models"]["risk_reference"])
    facing = (facing_model.probability(vectors) > .5).astype(int)
    lat, lon = coordinates(raw_metadata, frame)
    ctx = gm.Context(frame, score_matrix(vectors), risk, facing, lat, lon)
    bundles = gm.ordered_bundles(frame, bundle_keys(frame, "trap_day"))
    return ctx, bundles


def run_frozen(record, tau, error, bundles, codes, ctx, *, seed=42, tag=""):
    """One sequential simulation of the frozen operating point plus both audits."""
    from reid_demo.benchmark_review_simulation import Reviewer
    confidence, pmatch, _ = models_from_record(record)
    setting = record["operating_points"]["primary" if tau == PRIMARY_TAU else "secondary"]
    if setting["tau"] != tau:
        raise ValueError("Unknown frozen operating point.")
    fold = {int(c): 0 for c in np.unique(codes)}
    seeds = [int(fingerprint([tag, role, tau, error, seed])[:8], 16) for role in ("expert", "second", "audit", "sample", "count", "count_sample")]
    sim = nm.PrunedSimulation(bundles, codes, nm.PairContext(ctx), {0: confidence}, list(confidence.columns),
                              {0: pmatch.probability}, list(pmatch.columns), fold, setting,
                              Reviewer(error, seeds[0]), Reviewer(error, seeds[1])).run()
    ledger = sim.ledger.copy()
    auto = gm.audit(sim.auto_log, record["auto_audit"], Reviewer(error, seeds[2]), ledger, seed=seeds[3])
    extra = ledger.copy()
    count = nm.count_audit(sim.catalogue, record["count_audit"], Reviewer(error, seeds[4]), extra, ctx.scores, codes,
                           seed=seeds[5])
    row = nm.summarize(sim, codes, len(codes), len(bundles),
                       {"ledger": ledger, "auto": auto, "count": {record["count_audit"]: {**count, "ledger": extra}}})
    return sim, {"tau": tau, "reviewer_error": error, **row}


def w1_reference(frame, codes, ctx, error=0.0, seed=42):
    from reid_demo.benchmark_review_simulation import Reviewer, bundle_keys, fast_quality, ordered_units, simulate_catalogue
    units = ordered_units(frame, codes, bundle_keys(frame, "trap_day"))
    labels, effort = simulate_catalogue(units, ctx.scores, len(codes), Reviewer(error, seed), ranked=True)
    return labels, {"reviewer_error": error, **effort.as_dict(), **fast_quality(labels, codes)}


def subset_quality(labels, catalogue, codes, keep):
    """Grouping quality restricted to photos in `keep` (a breakdown, not a separate simulation)."""
    from reid_demo.benchmark_review_simulation import fast_quality
    labels, codes = np.asarray(labels), np.asarray(codes)
    founder = np.asarray([catalogue.founder[l] if l >= 0 else -1 for l in labels])
    wrong = (labels >= 0) & (founder != codes)
    q = fast_quality(labels[keep], codes[keep])
    return {"photos": int(keep.sum()), "misassigned_photos": int(wrong[keep].sum()),
            "misassigned_photo_rate": float(wrong[keep].mean()) if keep.any() else None, **q}


def per_animal(labels, catalogue, codes, identities):
    labels, codes = np.asarray(labels), np.asarray(codes)
    founder = np.asarray([catalogue.founder[l] if l >= 0 else -1 for l in labels])
    rows = []
    for c in np.unique(codes):
        mine = codes == c
        groups = set(labels[mine].tolist())
        rows.append({"identity": identities[int(c)], "photos": int(mine.sum()),
                     "misassigned_photos": int((founder[mine] != c).sum()), "entries": len(groups),
                     "exact": bool(len(groups) == 1 and (labels == next(iter(groups))).sum() == mine.sum())})
    return rows


def criteria(rows, w1, calibration_reference, n_images):
    def find(tau, error):
        return next(r for r in rows if r["tau"] == tau and r["reviewer_error"] == error)
    primary, noisy = find(PRIMARY_TAU, 0.0), find(PRIMARY_TAU, 0.03)
    per_1000 = primary["decisions"] / (n_images / 1000)
    limit = 1.25 * calibration_reference["decisions_per_1000_photos"]
    saving = 1 - primary["decisions"] / w1["decisions"]
    count_cov = primary.get(f"count_audit{COUNT_AUDIT}_count_covered")
    return {
        "S1": {"met": bool(primary["misassigned_photo_rate"] <= .01 and abs(primary["count_error"]) <= 1),
               "misassigned_photo_rate": primary["misassigned_photo_rate"], "count_error": primary["count_error"]},
        "S2": {"met": bool(per_1000 <= limit and saving >= .5), "decisions_per_1000_photos": per_1000,
               "limit": limit, "saving_vs_w1": saving, "w1_decisions": w1["decisions"]},
        "S3": {"met": bool(primary["audit_covered"] and count_cov), "automation_audit_covered": primary["audit_covered"],
               "count_audit_covered": count_cov},
        "S4": {"met": bool(abs(noisy["count_error"]) <= 2 and noisy["misassigned_photo_rate"] <= .03),
               "count_error": noisy["count_error"], "misassigned_photo_rate": noisy["misassigned_photo_rate"]},
        "carried_over_targets": {"automation_ge_70": primary["automation"] >= .7,
                                 "misassigned_le_1pct": primary["misassigned_photo_rate"] <= .01,
                                 "count_error_le_1": abs(primary["count_error"]) <= 1,
                                 "hours_le_0_5": primary["hours_per_1000_photos"] <= .5,
                                 "audit_covered": bool(primary["audit_covered"])},
        "policy": "Mechanical checks of the 2026-10-19 frozen-selection proposal; not conclusions."}
