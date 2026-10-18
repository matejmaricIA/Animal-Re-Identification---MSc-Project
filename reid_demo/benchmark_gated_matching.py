"""Confidence-gated ranked matching (W4): automate confident matches, send the rest to an expert.

Photos arrive in date order as camera-day bundles. The system auto-accepts the
top-ranked catalogue animal when a calibrated, identity-cross-fitted confidence
model is sure enough; otherwise a simulated expert checks the top-K candidates
or confirms a new animal. New animals are never created automatically. A
duplicate sweep and an audit sample follow. Automatic mistakes stay in the
catalogue and affect later matching. Calibration development evidence only.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reid_demo.benchmark import file_digest, write_json
from reid_demo.benchmark_clustering import ROOT
from reid_demo.benchmark_cluster_diagnostics import _write_table
from reid_demo.benchmark_review_simulation import COSTS, Catalogue, Effort, plain


VERSION = "calibration_gated_matching_v1"
THRESHOLDS = (0.95, 0.98, 0.99, 0.995)
TOP_K = (10, 20)
SWEEPS = (0, 25, 50)
AUDITS = (100, 200)
REVIEWER_ERRORS = (0.0, 0.03)
APPEARANCE_FEATURES = ("s1", "s2", "margin", "top3_mean", "log_size", "risk_mean", "facing_agreement",
                       "s1_same_facing", "log_catalogue")
SPACETIME_FEATURES = ("seen_same_camera", "seen_same_cell", "runner_up_seen_same_camera", "log_km",
                      "km_missing", "log_days_since_seen")
NOTES = [
    "Simulated reviewer answers from reference identities; placeholder costs are not measured expert times.",
    "Automation acts on whole camera-day bundles; a bundle holding two animals is assigned whole and counted as errors.",
    "The confidence model is identity-cross-fitted on calibration (development only); a deployable model needs fitting-pool training.",
    "New animals are never created automatically; the expert confirms them after checking the top-K candidates.",
    "Space-time features use day-level dates and cell/camera coordinates where available (about 88% of photos).",
    "Calibration development evidence only; no selection evaluation, count configuration or outreach.",
]


def implementation_binding():
    names = ("reid_demo/benchmark_gated_matching.py", "tools/simulate_czechlynx_gated_matching.py",
             "reid_demo/benchmark_review_simulation.py")
    return {"version": VERSION, "code": {n: file_digest(ROOT / n) for n in names}, "thresholds": list(THRESHOLDS),
            "top_k": list(TOP_K), "sweeps": list(SWEEPS), "audits": list(AUDITS),
            "reviewer_errors": list(REVIEWER_ERRORS), "appearance_features": list(APPEARANCE_FEATURES),
            "spacetime_features": list(SPACETIME_FEATURES), "costs": COSTS}


# --- Context (label-free) -------------------------------------------------------------------

class Context:
    """Per-image label-free context: camera, cell, coordinates, day number, facing, risk."""

    def __init__(self, frame, scores, risk, facing, lat=None, lon=None):
        n = len(frame)
        self.scores = scores
        self.risk = np.asarray(risk, float)
        self.facing = np.asarray(facing, int)
        self.camera = [f"{s}:{t}" if isinstance(t, str) else None for s, t in zip(frame.source, frame.trap_id)]
        loc = frame["location"] if "location" in frame else pd.Series([None] * n)
        self.cell = [f"{s}:{c}" if isinstance(c, str) else None for s, c in zip(frame.source, loc)]
        days = pd.to_datetime(frame.timestamp, errors="coerce")
        self.day = ((days - pd.Timestamp("1970-01-01")).dt.days).to_numpy(dtype=float)
        self.lat = np.full(n, np.nan) if lat is None else np.asarray(lat, float)
        self.lon = np.full(n, np.nan) if lon is None else np.asarray(lon, float)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def match_features(images, catalogue, ctx):
    """Features of a bundle against the current catalogue, plus the top candidates. No labels are read."""
    assigned = np.asarray(catalogue.assigned, dtype=int)
    entries = catalogue.entry_of[assigned]
    block = ctx.scores[np.ix_(images, assigned)]
    best = np.full(len(catalogue.founder), -np.inf)
    np.maximum.at(best, entries, block.max(axis=0))
    order = np.argsort(-best, kind="stable")
    e1 = int(order[0])
    e2 = int(order[1]) if len(order) > 1 else None
    s1 = float(best[e1])
    s2 = float(best[e2]) if e2 is not None else -1.0
    members = assigned[entries == e1]
    facing = int(round(ctx.facing[images].mean()))
    same_facing = members[ctx.facing[members] == facing]
    s1_same = float(ctx.scores[np.ix_(images, same_facing)].max()) if len(same_facing) else -1.0
    cameras = {ctx.camera[i] for i in images} - {None}
    cells = {ctx.cell[i] for i in images} - {None}
    seen = lambda entry, attr, values: float(bool(values & ({getattr(ctx, attr)[i] for i in assigned[entries == entry]} - {None})))
    lat, lon = np.nanmean(ctx.lat[images]) if np.isfinite(ctx.lat[images]).any() else np.nan, \
        np.nanmean(ctx.lon[images]) if np.isfinite(ctx.lon[images]).any() else np.nan
    known = members[np.isfinite(ctx.lat[members])]
    km = (float(np.min(haversine_km(lat, lon, ctx.lat[known], ctx.lon[known])))
          if np.isfinite(lat) and len(known) else np.nan)
    day = np.nanmin(ctx.day[images]) if np.isfinite(ctx.day[images]).any() else np.nan
    last = np.nanmax(ctx.day[members]) if np.isfinite(ctx.day[members]).any() else np.nan
    days = day - last if np.isfinite(day) and np.isfinite(last) else np.nan
    top3 = best[order[:3]]
    features = {
        "s1": s1, "s2": s2, "margin": s1 - s2, "top3_mean": float(np.mean(top3)),
        "log_size": float(np.log1p(len(images))), "risk_mean": float(ctx.risk[images].mean()),
        "facing_agreement": float((ctx.facing[members] == facing).mean()), "s1_same_facing": s1_same,
        "log_catalogue": float(np.log1p(len(catalogue.founder))),
        "seen_same_camera": seen(e1, "camera", cameras), "seen_same_cell": seen(e1, "cell", cells),
        "runner_up_seen_same_camera": seen(e2, "camera", cameras) if e2 is not None else 0.0,
        "log_km": float(np.log1p(km)) if np.isfinite(km) else 0.0, "km_missing": float(not np.isfinite(km)),
        "log_days_since_seen": float(np.log1p(max(days, 0))) if np.isfinite(days) else 0.0,
    }
    return features, [int(e) for e in order]


# --- Bundles --------------------------------------------------------------------------------

def ordered_bundles(frame, keys):
    """Whole camera-day bundles (image index lists) in date order; no labels."""
    dates = frame.timestamp.astype(str).tolist()
    groups = defaultdict(list)
    for i, key in enumerate(keys):
        groups[key].append(i)
    order = sorted(groups, key=lambda k: (min(dates[i] for i in groups[k]), k))
    return [groups[k] for k in order]


def majority(images, codes):
    return Counter(codes[images].tolist()).most_common(1)[0][0]


# --- Confidence model -----------------------------------------------------------------------

def replay_records(bundles, codes, ctx):
    """Perfect-review W1 replay: features of each bundle vs the true catalogue, and whether top-1 is right."""
    catalogue, rows = Catalogue(len(codes)), []
    entry_of_identity = {}
    for images in bundles:
        identity = majority(images, codes)
        if catalogue.founder:
            features, order = match_features(images, catalogue, ctx)
            rows.append({**features, "top1_correct": int(catalogue.founder[order[0]] == identity),
                         "identity": int(identity)})
        for animal, part in _parts(images, codes):
            if animal in entry_of_identity:
                catalogue.add(part, entry_of_identity[animal])
            else:
                entry = catalogue.add(part)
                catalogue.founder[entry] = animal
                entry_of_identity[animal] = entry
    return pd.DataFrame(rows)


def _parts(images, codes):
    parts = defaultdict(list)
    for i in images:
        parts[int(codes[i])].append(int(i))
    return sorted(parts.items(), key=lambda item: min(item[1]))


def fit_fold_models(records, identity_fold, columns, *, c=1.0):
    """One model per identity fold, each trained on records of the other folds' animals."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    folds = records.identity.map(identity_fold).to_numpy()
    models = {}
    for fold in sorted(set(identity_fold.values())):
        train = folds != fold
        y = records.top1_correct.to_numpy()[train]
        if len(np.unique(y)) < 2:
            raise ValueError("A confidence fold lacks both outcomes.")
        models[fold] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=c)).fit(
            records.loc[train, list(columns)].to_numpy(float), y)
    return models


def cross_fitted_confidence(records, models, identity_fold, columns):
    out = np.full(len(records), np.nan)
    folds = records.identity.map(identity_fold).to_numpy()
    for fold, model in models.items():
        m = folds == fold
        if m.any():
            out[m] = model.predict_proba(records.loc[m, list(columns)].to_numpy(float))[:, 1]
    return out


# --- W4 simulation --------------------------------------------------------------------------

def simulate_gated(bundles, codes, ctx, models, identity_fold, columns, tau, top_k, reviewer):
    """Sequential W4. Returns catalogue, effort, auto-decision log and counts."""
    catalogue, effort, auto_log = Catalogue(len(codes)), Effort(), []
    for images in bundles:
        animal = majority(images, codes)  # Used only to pick the cross-fitted model and by the reviewer.
        if catalogue.founder:
            features, order = match_features(images, catalogue, ctx)
            model = models[identity_fold[animal]]
            confidence = float(model.predict_proba(np.asarray([[features[c] for c in columns]], float))[:, 1][0])
            if confidence >= tau:
                entry = order[0]
                catalogue.add(images, entry)
                auto_log.append({"images": list(images), "entry": entry, "confidence": confidence,
                                 "correct": bool(all(codes[i] == catalogue.founder[entry] for i in images))})
                continue
        parts = _parts(images, codes)
        effort.add("extra_animal", len(parts) - 1)
        for identity, part in parts:
            ranked = catalogue.ranked(ctx.scores, part)[:top_k] if catalogue.founder else []
            for entry in ranked:
                effort.add("catalogue_comparison")
                if reviewer.same(catalogue.founder[entry] == identity):
                    catalogue.add(part, entry)
                    break
            else:
                entry = catalogue.add(part)
                catalogue.founder[entry] = identity
    return catalogue, effort, auto_log


def copy_effort(effort):
    clone = Effort()
    clone.counts = Counter(effort.counts)
    return clone


def duplicate_sweep(catalogue, ctx, m, reviewer, effort):
    """Expert reviews the m most similar catalogue-animal pairs (label-free ranking) and merges duplicates."""
    labels = catalogue.entry_of.copy()
    if m <= 0 or len(catalogue.founder) < 2:
        return labels
    entries = sorted(set(labels[labels >= 0].tolist()))
    members = {e: np.flatnonzero(labels == e) for e in entries}
    pairs = []
    for i, a in enumerate(entries):
        for b in entries[i + 1:]:
            pairs.append((float(ctx.scores[np.ix_(members[a], members[b])].max()), a, b))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    parent = {e: e for e in entries}

    def root(e):
        while parent[e] != e:
            parent[e] = parent[parent[e]]
            e = parent[e]
        return e
    for _, a, b in pairs[:m]:
        ra, rb = root(a), root(b)
        if ra == rb:
            continue
        effort.add("pair_question")
        if reviewer.same(catalogue.founder[ra] == catalogue.founder[rb]):
            parent[max(ra, rb)] = min(ra, rb)
    return np.asarray([root(e) if e >= 0 else -1 for e in labels])


def wilson_interval(errors, n, z=1.96):
    if n == 0:
        return None, None, None
    p = errors / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0.0, centre - half), min(1.0, centre + half)


def audit(auto_log, n, reviewer, effort, seed=42):
    """Expert re-checks a random sample of automatic decisions; estimates the bundle error rate."""
    if not auto_log:
        return {"audited": 0, "estimate": None, "low": None, "high": None, "true_bundle_error": None, "covered": None}
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(auto_log), min(n, len(auto_log)), replace=False)
    found = 0
    for i in sample:
        effort.add("glance")
        found += int(not reviewer.same(auto_log[i]["correct"]))
    estimate, low, high = wilson_interval(found, len(sample))
    truth = 1 - float(np.mean([r["correct"] for r in auto_log]))
    return {"audited": int(len(sample)), "errors_found": found, "estimate": estimate, "low": low, "high": high,
            "true_bundle_error": truth, "covered": bool(low <= truth <= high)}


def misassigned_photos(labels, catalogue, codes):
    """Photos whose final animal's founding identity differs from their own (resolved photos only)."""
    founder_of = {}
    for i, label in enumerate(labels):
        if label >= 0 and label not in founder_of:
            founder_of[label] = catalogue.founder[label]
    wrong = sum(1 for i, label in enumerate(labels) if label >= 0 and founder_of[label] != codes[i])
    return wrong, wrong / max(1, int((labels >= 0).sum()))


# --- Rules and report -----------------------------------------------------------------------

def declared_rules(rows, n_images):
    def ok(r):
        return (r["automation"] >= .70 and r["misassigned_photo_rate"] <= .01 and abs(r["count_error"]) <= 1
                and r["hours_per_1000_photos"] <= .5 and bool(r["audit_covered"]))
    main = [r for r in rows if r["reviewer_error"] == 0 and r["spacetime"] and r["tau"] is not None]
    passing = [r for r in main if ok(r)]
    best = sorted(main, key=lambda r: (-(r["automation"]), r["misassigned_photo_rate"]))[:5]
    return {"candidate_operating_points": [r["config"] for r in passing], "met": bool(passing),
            "per_target_best": {
                "max_automation_with_error_le_1pct": max((r["automation"] for r in main
                                                          if r["misassigned_photo_rate"] <= .01), default=None),
                "min_hours_with_count_error_le_1": min((r["hours_per_1000_photos"] for r in main
                                                        if abs(r["count_error"]) <= 1), default=None)},
            "most_automated": [r["config"] for r in best],
            "policy": "Mechanical checks of the 2026-10-18 minimum-review proposal targets; not conclusions."}


def _fmt(value):
    return f"{value:.3f}" if isinstance(value, float) else "" if value is None else str(value)


def plot_tradeoff(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for spacetime in (True, False):
        for k in TOP_K:
            sel = sorted([r for r in rows if r["reviewer_error"] == 0 and r["spacetime"] == spacetime
                          and r["top_k"] == k and r["sweep"] == 25 and r["audit_n"] == 100 and r["tau"] is not None],
                         key=lambda r: r["tau"])
            label = f"{'space-time' if spacetime else 'appearance only'}, K={k}"
            axes[0].plot([r["automation"] for r in sel], [100 * r["misassigned_photo_rate"] for r in sel], "o-", label=label)
            axes[1].plot([r["automation"] for r in sel], [r["hours_per_1000_photos"] for r in sel], "o-", label=label)
    axes[0].axhline(1, color="#888", ls="--", lw=1)
    axes[1].axhline(.5, color="#888", ls="--", lw=1)
    for ax, title in zip(axes, ("Misassigned photos (%)", "Expert hours per 1,000 photos")):
        ax.axvline(.7, color="#888", ls=":", lw=1)
        ax.set_xlabel("Share of bundles decided automatically")
        ax.set_title(title)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle("Confidence-gated matching: automation vs error and expert time (perfect reviewer, sweep 25)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(out, results):
    out = Path(out)
    results = plain(results)
    rows = results["simulations"]
    write_json(out / "summary.json", {**results, "notes": NOTES})
    _write_table(out / "simulations.csv", rows, list(rows[0]))
    _write_table(out / "confidence_models.csv", results["confidence_models"], list(results["confidence_models"][0]))
    plot_tradeoff(out / "tradeoff.png", rows)
    keys = ["config", "automation", "misassigned_photo_rate", "count_error", "groups", "pairwise_f1",
            "exact_identity_recoveries", "decisions", "hours_per_1000_photos", "audit_estimate", "audit_low",
            "audit_high", "true_bundle_error", "audit_covered"]
    main = [r for r in rows if r["reviewer_error"] == 0 and r["sweep"] == 25 and r["audit_n"] == 100]
    table = "<table><tr>" + "".join(f"<th>{k}</th>" for k in keys) + "</tr>" + "".join(
        "<tr>" + "".join(f"<td>{html.escape(_fmt(r.get(k)))}</td>" for k in keys) + "</tr>" for r in main) + "</table>"
    body = ("<p>Simulated confidence-gated matching on calibration; reviewer answers from reference identities. "
            f"Placeholder costs (s): {html.escape(json.dumps(COSTS))}.</p>"
            "<h2>Declared targets</h2><pre>" + html.escape(json.dumps(results["declared_rules"], indent=1)) + "</pre>"
            "<h2>Confidence models (cross-fitted, replay records)</h2><pre>"
            + html.escape(json.dumps(results["confidence_models"], indent=1)) + "</pre>"
            "<h2>Trade-off</h2><img style='max-width:100%' src='tradeoff.png' alt='Trade-off'>"
            "<h2>Perfect reviewer, sweep 25, audit 100</h2>" + table +
            "<h2>Limitations</h2><ul>" + "".join(f"<li>{html.escape(n)}</li>" for n in NOTES) + "</ul>")
    (out / "index.html").write_text("<!doctype html><html lang='en'><meta charset='utf-8'><title>CzechLynx gated "
        "matching</title><style>body{font:14px system-ui;max-width:1400px;margin:30px auto;padding:16px}table{"
        "border-collapse:collapse}td,th{border:1px solid #bbb;padding:3px 6px}pre{white-space:pre-wrap}</style>"
        "<h1>CzechLynx confidence-gated matching</h1>" + body + "</html>")
    return {str(p.relative_to(out)): file_digest(p) for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "manifest.json"}
