"""Simulated expert review: how much work does a verified lynx grouping/count take?

A simulated reviewer answers identity questions from reference labels; the
workflows never read labels otherwise. W0 is unassisted catalogue matching, W1
embedding-ranked catalogue matching, and W2 group-then-correct review starting
from a fixed automatic grouping. Calibration development evidence only.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import html
import json
from pathlib import Path

import numpy as np

from reid_demo.benchmark import file_digest, write_json
from reid_demo.benchmark_clustering import ROOT
from reid_demo.benchmark_cluster_diagnostics import _write_table


VERSION = "calibration_review_effort_v1"
COSTS = {"glance": 5.0, "extra_animal": 5.0, "unresolved_glance": 5.0, "catalogue_comparison": 8.0,
         "pair_question": 10.0}
STOP_AFTER_NO = (10, 25, 50)
REVIEWER_ERRORS = (0.0, 0.03)
UNRESOLVED_RISK_QUANTILE = 0.8
CURVE_EVERY = 10
NOTES = [
    "A simulated reviewer answers from reference identities; real experts take variable time and cannot identify every photo.",
    "Workflows order and group photos without labels; only the reviewer's answers use truth.",
    "Reviewer noise flips yes/no identity answers at the stated rate; separating animals inside one bundle is assumed correct.",
    "Time costs are declared placeholders (seconds per decision); raw decision counts are reported for re-costing.",
    "Bundles are same camera + same day; CzechLynx dates are day-level.",
    "Calibration development evidence only; no selection evaluation, count configuration or outreach.",
]


def implementation_binding():
    names = ("reid_demo/benchmark_review_simulation.py", "tools/simulate_czechlynx_review.py")
    return {"version": VERSION, "code": {n: file_digest(ROOT / n) for n in names}, "costs": COSTS,
            "stop_after_no": list(STOP_AFTER_NO), "reviewer_errors": list(REVIEWER_ERRORS),
            "unresolved_risk_quantile": UNRESOLVED_RISK_QUANTILE, "curve_every": CURVE_EVERY}


class Effort:
    def __init__(self):
        self.counts = Counter()

    def add(self, kind, n=1):
        if kind not in COSTS:
            raise ValueError(f"Unknown decision kind: {kind}")
        self.counts[kind] += n

    @property
    def decisions(self):
        return int(sum(self.counts.values()))

    @property
    def seconds(self):
        return float(sum(COSTS[k] * v for k, v in self.counts.items()))

    def as_dict(self):
        return {**{k: int(self.counts.get(k, 0)) for k in COSTS}, "decisions": self.decisions,
                "seconds": self.seconds}


class Reviewer:
    """Answers 'same animal?' from truth, flipping answers at `error` with a seeded generator."""

    def __init__(self, error=0.0, seed=42):
        if not 0 <= error < 0.5:
            raise ValueError("Reviewer error must be in [0, 0.5).")
        self.error = error
        self.rng = np.random.default_rng(seed)

    def same(self, truth):
        return bool(truth) != bool(self.error and self.rng.random() < self.error)


# --- Units ----------------------------------------------------------------------------------

def bundle_keys(frame, bundling="trap_day"):
    """Label-free bundle key per image: camera|date, or the image itself."""
    if bundling == "none":
        return frame.image_id.astype(str).tolist()
    if bundling != "trap_day":
        raise ValueError("Bundling must be 'trap_day' or 'none'.")
    keys = []
    for row in frame.to_dict("records"):
        source, trap, date = row.get("source"), row.get("trap_id"), row.get("timestamp")
        ok = all(isinstance(v, str) and v.strip() for v in (source, trap, date))
        keys.append(f"{source}:{trap}|{date}" if ok else f"image|{row['image_id']}")
    return keys


def split_by_truth(members, codes):
    """The reviewer separates animals inside one bundle: (identity, image indices) parts."""
    parts = defaultdict(list)
    for i in members:
        parts[int(codes[i])].append(int(i))
    return sorted(parts.items(), key=lambda item: min(item[1]))


def ordered_units(frame, codes, keys):
    """(images, identity, extra_animals_in_bundle) in date / bundle order."""
    dates = frame.timestamp.astype(str).tolist() if "timestamp" in frame else [""] * len(frame)
    groups = defaultdict(list)
    for i, key in enumerate(keys):
        groups[key].append(i)
    order = sorted(groups, key=lambda k: (min(dates[i] for i in groups[k]), k))
    units = []
    for key in order:
        parts = split_by_truth(groups[key], codes)
        for j, (identity, images) in enumerate(parts):
            units.append({"images": images, "identity": identity, "extra": int(j > 0), "bundle": key})
    return units


def unresolvable(images, risk, cutoff):
    return cutoff is not None and bool(np.all(risk[images] >= cutoff))


# --- Catalogue matching (W0 unassisted, W1 ranked) ------------------------------------------

class Catalogue:
    """Entries hold image indices; an entry's truth is its founding unit's identity."""

    def __init__(self, n_images):
        self.founder = []
        self.entry_of = np.full(n_images, -1)
        self.assigned = []

    def add(self, images, entry=None):
        if entry is None:
            entry = len(self.founder)
            self.founder.append(None)
        self.entry_of[images] = entry
        self.assigned.extend(images)
        return entry

    def ranked(self, scores, images):
        if not self.founder:
            return []
        assigned = np.asarray(self.assigned, dtype=int)
        best = np.full(len(self.founder), -np.inf)
        np.maximum.at(best, self.entry_of[assigned], scores[np.ix_(images, assigned)].max(axis=0))
        return sorted(range(len(self.founder)), key=lambda e: (-best[e], e))


def place(unit, catalogue, order, reviewer, effort, kind="catalogue_comparison"):
    """Compare in order until the reviewer says yes; otherwise create a new entry."""
    for entry in order:
        effort.add(kind)
        if reviewer.same(catalogue.founder[entry] == unit["identity"]):
            return catalogue.add(unit["images"], entry)
    entry = catalogue.add(unit["images"])
    catalogue.founder[entry] = unit["identity"]
    return entry


def simulate_catalogue(units, scores, n_images, reviewer, *, ranked, risk=None, cutoff=None, seed=42):
    effort, catalogue = Effort(), Catalogue(n_images)
    rng = np.random.default_rng(seed)
    for unit in units:
        effort.add("extra_animal", unit["extra"])
        if unresolvable(unit["images"], risk, cutoff):
            effort.add("unresolved_glance")
            continue
        order = (catalogue.ranked(scores, unit["images"]) if ranked
                 else list(rng.permutation(len(catalogue.founder))))
        place(unit, catalogue, order, reviewer, effort)
    return catalogue.entry_of.copy(), effort


# --- W2: group first, then correct ----------------------------------------------------------

def majority_identity(labels, codes):
    out = {}
    for label in np.unique(labels[labels >= 0]):
        out[int(label)] = Counter(codes[labels == label].tolist()).most_common(1)[0][0]
    return out


def split_phase(start, codes, keys, reviewer, effort, risk=None, cutoff=None):
    """Glance at each (group, bundle, animal) unit; move out units not matching the group's animal."""
    labels = np.asarray(start, dtype=int).copy()
    majority = majority_identity(labels, codes)
    orphans = []
    groups = defaultdict(list)
    for i, (label, key) in enumerate(zip(labels, keys)):
        if label >= 0:
            groups[(int(label), key)].append(i)
    for (label, key) in sorted(groups):
        for j, (identity, images) in enumerate(split_by_truth(groups[(label, key)], codes)):
            effort.add("extra_animal", int(j > 0))
            if unresolvable(images, risk, cutoff):
                effort.add("unresolved_glance")
                labels[images] = -2  # Set aside as unresolved.
                continue
            effort.add("glance")
            if not reviewer.same(identity == majority[label]):
                labels[images] = -1
                orphans.append({"images": images, "identity": identity, "extra": 0, "bundle": key})
    return labels, orphans, majority


def place_orphans(labels, orphans, scores, codes, reviewer, effort):
    """Place moved-out units into corrected groups by ranked comparison (new group if none)."""
    catalogue = Catalogue(len(labels))
    for label in sorted(set(labels[labels >= 0].tolist())):
        images = np.flatnonzero(labels == label).tolist()
        entry = catalogue.add(images)
        catalogue.founder[entry] = Counter(codes[images].tolist()).most_common(1)[0][0]
    for unit in orphans:
        place(unit, catalogue, catalogue.ranked(scores, unit["images"]), reviewer, effort)
    # Set-aside (-2) and never-placed images stay unresolved (-1).
    return catalogue.entry_of.copy(), list(catalogue.founder)


def merge_candidates(labels, pairs, p):
    """Group pairs ranked by mean cross-edge log-odds (label-free)."""
    w = np.log(p / (1 - p))
    sums, counts = defaultdict(float), Counter()
    a, b = labels[pairs[:, 0]], labels[pairs[:, 1]]
    for x, y, weight in zip(a, b, w):
        if x >= 0 and y >= 0 and x != y:
            key = (min(x, y), max(x, y))
            sums[key] += weight
            counts[key] += 1
    return sorted(((sums[k] / counts[k], k) for k in counts), key=lambda item: (-item[0], item[1]))


def merge_phase(labels, founders, candidates, reviewer, effort, stop_after_no, codes, curve=None):
    """Ask 'same animal?' down the ranked list until `stop_after_no` consecutive no answers."""
    parent = {g: g for g in set(labels[labels >= 0].tolist())}

    def root(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g
    truth_of = {g: founders[g] for g in parent}
    misses = 0
    for _, (x, y) in candidates:
        rx, ry = root(x), root(y)
        if rx == ry:
            continue
        effort.add("pair_question")
        if reviewer.same(truth_of[rx] == truth_of[ry]):
            parent[max(rx, ry)] = min(rx, ry)
            misses = 0
        else:
            misses += 1
        if curve is not None and effort.counts["pair_question"] % CURVE_EVERY == 0:
            current = np.asarray([root(g) if g >= 0 else -1 for g in labels])
            curve.append({"decisions": effort.decisions, "seconds": effort.seconds, **fast_quality(current, codes)})
        if misses >= stop_after_no:
            break
    return np.asarray([root(g) if g >= 0 else -1 for g in labels])


def simulate_group_first(start, codes, keys, scores, pairs, p, reviewer, stop_after_no, *,
                         risk=None, cutoff=None, curve=None):
    effort = Effort()
    labels, orphans, _ = split_phase(start, codes, keys, reviewer, effort, risk, cutoff)
    if curve is not None:
        curve.append({"decisions": effort.decisions, "seconds": effort.seconds, "phase": "split",
                      **fast_quality(np.where(labels >= 0, labels, -1), codes)})
    labels, founders = place_orphans(labels, orphans, scores, codes, reviewer, effort)
    if curve is not None:
        curve.append({"decisions": effort.decisions, "seconds": effort.seconds, "phase": "placed",
                      **fast_quality(labels, codes)})
    labels = merge_phase(labels, founders, merge_candidates(labels, pairs, p), reviewer, effort,
                         stop_after_no, codes, curve)
    return labels, effort


# --- Quality (truth) ------------------------------------------------------------------------

def _pairs(n):
    return n * (n - 1) // 2


def fast_quality(labels, codes):
    """Pairwise precision/recall/F1, exact recoveries and counts over resolved images (label >= 0)."""
    labels, codes = np.asarray(labels), np.asarray(codes)
    keep = labels >= 0
    lab, cod = labels[keep], codes[keep]
    joint = Counter(zip(lab.tolist(), cod.tolist()))
    by_label, by_code = Counter(lab.tolist()), Counter(cod.tolist())
    tp = sum(_pairs(v) for v in joint.values())
    predicted, true = sum(_pairs(v) for v in by_label.values()), sum(_pairs(v) for v in by_code.values())
    precision = tp / predicted if predicted else None
    recall = tp / true if true else None
    f1 = (2 * precision * recall / (precision + recall) if precision and recall else 0.0
          if precision is not None and recall is not None else None)
    labels_of = defaultdict(set)
    codes_of = defaultdict(set)
    for (l, c) in joint:
        labels_of[c].add(l)
        codes_of[l].add(c)
    exact = sum(len(ls) == 1 and len(codes_of[next(iter(ls))]) == 1 for ls in labels_of.values())
    return {"pairwise_precision": precision, "pairwise_recall": recall, "pairwise_f1": f1,
            "exact_identity_recoveries": int(exact), "groups": len(by_label),
            "animals_with_resolved_images": len(by_code), "count_error": len(by_label) - len(by_code),
            "resolved_images": int(keep.sum()), "unresolved_images": int((~keep).sum()),
            "merge_errors": sum(len(c) > 1 for c in codes_of.values()),
            "split_errors": sum(len(l) > 1 for l in labels_of.values())}


# --- Rules and report -----------------------------------------------------------------------

def declared_rules(rows):
    main = [r for r in rows if r["reviewer_error"] == 0 and not r["unresolved_variant"] and r["bundling"] == "trap_day"]
    w1 = next(r for r in main if r["workflow"] == "W1_ranked")
    w2 = [r for r in main if r["workflow"].startswith("W2") and r["stop_after_no"] == 25]
    good = [r for r in w2 if (r["pairwise_f1"] or 0) >= .98 and abs(r["count_error"]) <= 1]
    ratio = {r["workflow"]: r["decisions"] / w1["decisions"] for r in w2}
    noisy = [r for r in rows if r["reviewer_error"] > 0 and not r["unresolved_variant"]
             and r["bundling"] == "trap_day" and r["workflow"].startswith("W2") and r["stop_after_no"] == 25]
    return {"w1_decisions": w1["decisions"], "w2_decision_ratio_vs_w1": ratio,
            "grouping_saves_work": {"met": any(ratio[r["workflow"]] <= .5 for r in good),
                                    "qualifying": [r["workflow"] for r in good if ratio[r["workflow"]] <= .5]},
            "strong_saving": {"met": any(ratio[r["workflow"]] <= .25 for r in good)},
            "robust_to_reviewer_error": {"met": any((r["pairwise_f1"] or 0) >= .95 for r in noisy),
                                         "f1": {r["workflow"]: r["pairwise_f1"] for r in noisy}},
            "policy": "Mechanical checks of the 2026-10-18 review-effort proposal thresholds; not conclusions."}


def _fmt(value):
    return f"{value:.3f}" if isinstance(value, float) else "" if value is None else str(value)


def plot_curves(path, curves, w1_decisions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, rows in curves.items():
        x = [r["decisions"] for r in rows]
        axes[0].plot(x, [r["pairwise_f1"] or 0 for r in rows], label=name)
        axes[1].plot(x, [r["exact_identity_recoveries"] for r in rows], label=name)
    for ax, title in zip(axes, ("Pairwise F1", "Animals recovered exactly")):
        ax.axvline(w1_decisions, color="#888", ls="--", lw=1)
        ax.set_xlabel("Expert decisions (dashed: W1 ranked matching total)")
        ax.set_title(title)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle("Simulated group-then-correct review (perfect reviewer, trap-day bundles)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plain(value):
    """Convert numpy scalars/arrays nested in results to JSON-safe Python values."""
    return json.loads(json.dumps(value, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else
                                 o.item() if isinstance(o, np.generic) else str(o)))


def write_report(out, results, curves):
    out = Path(out)
    results, curves = plain(results), plain(curves)
    rows = results["simulations"]
    write_json(out / "summary.json", {**results, "notes": NOTES})
    _write_table(out / "simulations.csv", rows, list(rows[0]))
    curve_rows = [{"start": k, **r} for k, v in curves.items() for r in v]
    _write_table(out / "curves.csv", curve_rows, sorted({k for r in curve_rows for k in r}))
    plot_curves(out / "curves.png", curves, results["declared_rules"]["w1_decisions"])
    keys = ["workflow", "bundling", "reviewer_error", "unresolved_variant", "stop_after_no", "decisions",
            "hours", "hours_per_1000_photos", "pairwise_precision", "pairwise_recall", "pairwise_f1",
            "exact_identity_recoveries", "groups", "count_error"]
    table = "<table><tr>" + "".join(f"<th>{k}</th>" for k in keys) + "</tr>" + "".join(
        "<tr>" + "".join(f"<td>{html.escape(_fmt(r.get(k)))}</td>" for k in keys) + "</tr>" for r in rows) + "</table>"
    body = ("<p>Simulated expert review on calibration data; reviewer answers from reference identities. "
            f"Declared costs (s): {html.escape(json.dumps(COSTS))}.</p>"
            "<h2>Declared rules</h2><pre>" + html.escape(json.dumps(results["declared_rules"], indent=1)) + "</pre>"
            "<h2>Starting groupings (reproduced)</h2><pre>" + html.escape(json.dumps(results["starting_groupings"], indent=1))
            + "</pre><h2>Effort vs quality</h2><img style='max-width:100%' src='curves.png' alt='Curves'>"
            "<h2>All simulations</h2>" + table +
            "<h2>Limitations</h2><ul>" + "".join(f"<li>{html.escape(n)}</li>" for n in NOTES) + "</ul>")
    (out / "index.html").write_text("<!doctype html><html lang='en'><meta charset='utf-8'><title>CzechLynx review effort"
        "</title><style>body{font:14px system-ui;max-width:1400px;margin:30px auto;padding:16px}table{border-collapse:"
        "collapse}td,th{border:1px solid #bbb;padding:3px 6px}pre{white-space:pre-wrap}</style>"
        "<h1>CzechLynx simulated review effort</h1>" + body + "</html>")
    return {str(p.relative_to(out)): file_digest(p) for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "manifest.json"}
