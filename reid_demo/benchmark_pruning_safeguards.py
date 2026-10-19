"""Pruning floors and a duplicate-merge second check for the pruned ranked review (N2).

Extends ``benchmark_nonmatch_pruning.PrunedSimulation`` without editing it (its code is bound
by hash to completed, committed runs, including the frozen selection evaluation):

* ``min_shown`` m: the pruned scan never stops before m candidates were shown (P(match) order);
* ``cosine_floor`` k: when the P(match) suffix falls below eps, any still-unshown entries among
  the top-k by APPEARANCE cosine are also shown before a new animal is declared. That shortlist
  ignores space-time features, so a mis-transferred location prior cannot hide them;
* ``merge_confirm``: an expert "same animal" on a duplicate-check pair is merged only if an
  independent second reviewer agrees (one ``merge_confirmation`` decision).

With ``NEW_DEFAULTS`` the simulation is identical to ``PrunedSimulation``.

Region folds (``region_folds``) fit every label-dependent model on one CzechLynx region and
apply it to the other region's animals, a stand-in for a new site. Calibration only.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from reid_demo.benchmark import file_digest
from reid_demo.benchmark_clustering import ROOT
from reid_demo.benchmark_frozen_review import subset_quality
from reid_demo import benchmark_nonmatch_pruning as nm


VERSION = "calibration_pruning_safeguards_v1"
KINDS = nm.KINDS + ("merge_confirmation",)
COST_SETTINGS = {name: {**costs, "merge_confirmation": costs["pair_question"]}
                 for name, costs in nm.COST_SETTINGS.items()}
NEW_DEFAULTS = {"min_shown": 1, "cosine_floor": 0, "merge_confirm": False}
TAU = 0.99
EPSILONS = (0.001, 0.0003)
MIN_SHOWN = (1, 5, 10)
COSINE_FLOORS = (0, 5, 10)
MERGE_CONFIRM = (False, True)
REVIEWER_ERRORS = (0.0, 0.03, 0.05)
REPLICATES = {0.0: 1, 0.03: 5, 0.05: 5}
FOLD_SCHEMES = ("identity", "region")
PRIMARY_FOLDS = "region"
#: CzechLynx sources by region. snpa and foe_bohemia overlap geographically and share animals.
REGIONS = {"foe_carpaths": 0, "snpa": 1, "foe_bohemia": 1}
REGION_NAMES = {0: "western_carpathians", 1: "bohemian_forest"}
#: The (a)/(b) operating point these settings modify, and its saved F1 rows (reference check).
BASE = {"order": "pmatch", "eps": 0.001, "tau": TAU, **nm.DEFAULTS}
REFERENCE_TAGS = {0.0: "G1/defaults/err0.0/pmatch/eps0.001/tau0.99",
                  0.03: "G1/defaults/err0.03/pmatch/eps0.001/tau0.99"}
REFERENCE_FIELDS = ("decisions", "automation", "misassigned_photos", "count_error", "pruning_duplicate_events",
                    "duplicate_merges", "wrong_duplicate_merges", "auto_bundles")
# Declared rules (proposal, 19 Oct 2026), evaluated on the primary (region) folds.
R1_EVENTS_PER_1000 = 0.5
R2_SHARE_OF_W1 = 0.5
R3_LIMITS = {0.03: {"misassigned": 0.02, "count": 1}, 0.05: {"misassigned": 0.03, "count": 2}}
R4_MAX_WRONG_MERGES = 1
R4_MAX_WRONG_SHARE = 0.25
NOTES = [
    "Calibration development evidence only (F1 vectors); selection is spent and the official test is locked.",
    "Region folds fit the facing classifier, the W4 confidence model and the P(match) model on the other "
    "region's animals; a two-region proxy for a new site, not a new park.",
    "Simulated reviewers answer from reference identities; costs are placeholders, raw decision counts are kept.",
    "With min_shown 1, cosine_floor 0 and merge_confirm off the simulation equals PrunedSimulation.",
    "R2 uses 0.5 x live W1 on the same F1 vectors (5,325 decisions on the saved F1 run); the proposal's "
    "2,641 quoted v2's W1 (5,283).",
]


def implementation_binding():
    names = ("reid_demo/benchmark_pruning_safeguards.py", "tools/simulate_czechlynx_pruning_safeguards.py",
             "reid_demo/benchmark_nonmatch_pruning.py", "reid_demo/benchmark_gated_matching.py",
             "reid_demo/benchmark_review_simulation.py", "reid_demo/benchmark_frozen_review.py")
    return {"version": VERSION, "code": {n: file_digest(ROOT / n) for n in names}, "tau": TAU,
            "epsilons": list(EPSILONS), "min_shown": list(MIN_SHOWN), "cosine_floors": list(COSINE_FLOORS),
            "merge_confirm": list(MERGE_CONFIRM), "reviewer_errors": list(REVIEWER_ERRORS),
            "replicates": {str(k): v for k, v in REPLICATES.items()}, "fold_schemes": list(FOLD_SCHEMES),
            "regions": REGIONS, "base": BASE, "cost_settings": COST_SETTINGS,
            "rules": {"R1_events_per_1000": R1_EVENTS_PER_1000, "R2_share_of_w1": R2_SHARE_OF_W1,
                      "R3_limits": {str(k): v for k, v in R3_LIMITS.items()},
                      "R4_max_wrong_merges": R4_MAX_WRONG_MERGES, "R4_max_wrong_share": R4_MAX_WRONG_SHARE}}


# --- Ledger and simulation ------------------------------------------------------------------

class SafeguardLedger(nm.Ledger):
    """``nm.Ledger`` plus the ``merge_confirmation`` kind, costed like a pair question."""

    def add(self, kind, n=1):
        kind = self.ALIASES.get(kind, kind)
        if kind not in KINDS:
            raise ValueError(f"Unknown decision kind: {kind}")
        self.counts[kind] += n

    def copy(self):
        clone = SafeguardLedger()
        clone.counts = Counter(self.counts)
        return clone

    def seconds(self, setting="placeholder", count_audit=False):
        costs = COST_SETTINGS[setting]
        return float(sum(costs[k] * v for k, v in self.counts.items() if count_audit or k != "count_audit_comparison"))

    def as_dict(self):
        return {k: int(self.counts.get(k, 0)) for k in KINDS}


class SafeguardSimulation(nm.PrunedSimulation):
    """``PrunedSimulation`` with the pruning floors and the merge second check (settings in ``setting``)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in NEW_DEFAULTS.items():
            self.s.setdefault(key, value)
        if self.s["min_shown"] < 1 or self.s["cosine_floor"] < 0:
            raise ValueError("min_shown must be >= 1 and cosine_floor >= 0.")
        self.ledger = SafeguardLedger()
        self.blocked_merges = []

    def place(self, identity, part):
        cat, s = self.catalogue, self.s
        true_entries = {e for e, f in enumerate(cat.founder) if f == identity}  # Evaluation log only.
        record = {"bundle": self.bundle, "identity": int(identity), "photos": len(part),
                  "true_in_catalogue": bool(true_entries), "guard_open": self.guard_open(), "shown": 0,
                  "disagreement_checks": 0, "second_comparisons": 0, "first_correct": None,
                  "floor_shown": 0, "matched_by_floor": False}
        if not cat.founder:
            self._declare(identity, part, record, cut_true=False)
            return
        pruning = s["eps"] > 0 and self.guard_open()
        need_p = s["order"] == "pmatch" or pruning or s["double_c"] or s["disagree_delta"] is not None
        if need_p:
            p, cosine = self._pmatch(part, identity)
        else:
            p, cosine = None, cat.ranked(self.ctx.scores, part)
        position = {e: i for i, e in enumerate(cosine)}
        order = cosine if s["order"] == "cosine" else sorted(cosine, key=lambda e: (-p[e], position[e]))
        if pruning:
            suffix = np.maximum.accumulate(np.asarray([p[e] for e in order])[::-1])[::-1]
        record["first_correct"] = bool(order[0] in true_entries)
        shown = set()

        def ask(e):
            self.ledger.add("rank1_comparison" if not shown else "further_comparison")
            shown.add(e)
            truth = cat.founder[e] == identity
            if self.expert.same(truth):
                if s["disagree_delta"] is not None and p[e] < s["disagree_delta"]:
                    self.ledger.add("disagreement_check")
                    record["disagreement_checks"] += 1
                    if not self.second.same(truth):
                        return False
                cat.add(part, e)
                self._log(record, shown, "matched_correct" if truth else "matched_wrong")
                return True
            return False

        stopped = False
        for pos, e in enumerate(order):
            if pruning and len(shown) >= s["min_shown"] and suffix[pos] < s["eps"]:
                stopped = True
                break
            if ask(e):
                return
        if stopped:
            for e in cosine[:s["cosine_floor"]]:
                if e in shown:
                    continue
                record["floor_shown"] += 1
                if ask(e):
                    record["matched_by_floor"] = True
                    return
        if s["double_c"]:
            for e in sorted(cosine, key=lambda e: (-p[e], position[e]))[:s["double_c"]]:
                self.ledger.add("second_comparison")
                record["second_comparisons"] += 1
                truth = cat.founder[e] == identity
                if self.second.same(truth):
                    cat.add(part, e)
                    self._log(record, shown, "rescued_correct" if truth else "rescued_wrong")
                    return
        record["shown"] = len(shown)
        self._declare(identity, part, record, cut_true=bool(true_entries) and not (true_entries & shown))

    def duplicate_check(self):
        cat = self.catalogue
        pending, self.pending = self.pending, []
        for uid in pending:
            if uid not in cat.uid or len(cat.founder) < 2:
                continue
            b = cat.uid.index(uid)
            p, cosine = self._pmatch(cat.members(b), cat.founder[b], exclude=b)
            position = {e: i for i, e in enumerate(cosine)}
            asked = 0
            for a in sorted(cosine, key=lambda e: (-p[e], position[e])):
                if asked >= self.s["dup_r"]:
                    break
                key = frozenset((cat.uid[a], uid))
                if key in self.asked:
                    continue
                self.asked.add(key)
                asked += 1
                self.ledger.add("pair_question")
                truth = cat.founder[a] == cat.founder[b]
                if self.expert.same(truth):
                    if self.s["merge_confirm"]:
                        self.ledger.add("merge_confirmation")
                        if not self.second.same(truth):
                            self.blocked_merges.append({"bundle": self.bundle, "uids": sorted(key),
                                                        "true_duplicate": bool(truth)})
                            continue
                    self.duplicate_log.append({"bundle": self.bundle, "kept_uid": cat.uid[min(a, b)],
                                               "merged_uid": cat.uid[max(a, b)], "correct": bool(truth)})
                    cat.merge(min(a, b), max(a, b))
                    break


# --- Folds, metrics and rules ---------------------------------------------------------------

def region_of_sources(sources):
    unknown = sorted(set(sources) - set(REGIONS))
    if unknown:
        raise ValueError(f"Unknown CzechLynx source(s) for region folds: {unknown}")
    return np.asarray([REGIONS[s] for s in sources], int)


def region_folds(identity_codes, sources):
    """Per-image region fold of each photo's ANIMAL (0 Carpathians, 1 Bohemian Forest).
    Raises if an animal has photos in both regions."""
    regions = region_of_sources(sources)
    codes = np.asarray(identity_codes, int)
    by_animal = {}
    for c, r in zip(codes.tolist(), regions.tolist()):
        by_animal.setdefault(c, set()).add(r)
    spanning = sorted(c for c, rs in by_animal.items() if len(rs) > 1)
    if spanning:
        raise ValueError(f"Animals with photos in both regions cannot be region-folded: {spanning}")
    return np.asarray([next(iter(by_animal[c])) for c in codes.tolist()], int)


def summarize(sim, codes, n_images, n_bundles, audits, regions):
    """``nm.summarize`` plus merge-check, floor and per-region figures."""
    row = nm.summarize(sim, codes, n_images, n_bundles, audits)
    codes, regions = np.asarray(codes), np.asarray(regions)
    animal_region = {int(c): int(r) for c, r in zip(codes.tolist(), regions.tolist())}
    blocked = sim.blocked_merges
    floor_records = [r for r in sim.placements if r.get("floor_shown")]
    row.update({"merge_confirmation": int(sim.ledger.counts.get("merge_confirmation", 0)),
                "blocked_merges": len(blocked),
                "blocked_true_duplicates": sum(b["true_duplicate"] for b in blocked),
                "floor_comparisons": int(sum(r["floor_shown"] for r in floor_records)),
                "floor_matches": int(sum(r["matched_by_floor"] for r in floor_records)),
                "floor_matches_correct": int(sum(r["matched_by_floor"] and r.get("outcome") == "matched_correct"
                                                 for r in floor_records))})
    labels = sim.catalogue.entry_of.copy()
    for region, name in REGION_NAMES.items():
        keep = regions == region
        q = subset_quality(labels, sim.catalogue, codes, keep)
        events = sum(r["outcome"] == "new_pruning_duplicate" and animal_region[r["identity"]] == region
                     for r in sim.declarations)
        row.update({f"{name}_photos": q["photos"], f"{name}_count_error": q["count_error"],
                    f"{name}_misassigned_photo_rate": q["misassigned_photo_rate"],
                    f"{name}_pruning_duplicate_events": int(events),
                    f"{name}_pruning_duplicates_per_1000": events / (q["photos"] / 1000) if q["photos"] else None})
    return row


def setting_key(row):
    return (row["eps"], row["min_shown"], row["cosine_floor"], bool(row["merge_confirm"]))


def experiment_configs():
    """The declared grid for one fold scheme (references are built separately)."""
    configs = []
    for eps in EPSILONS:
        for min_shown in MIN_SHOWN:
            for floor in COSINE_FLOORS:
                for confirm in MERGE_CONFIRM:
                    for error in REVIEWER_ERRORS:
                        for replicate in range(REPLICATES[error]):
                            configs.append({**BASE, "eps": eps, "min_shown": min_shown, "cosine_floor": floor,
                                            "merge_confirm": confirm, "reviewer_error": error,
                                            "replicate": replicate})
    return configs


def config_tag(folds, c):
    return (f"N2/{folds}/eps{c['eps']}/min{c['min_shown']}/cos{c['cosine_floor']}/"
            f"merge{int(c['merge_confirm'])}/err{c['reviewer_error']}/rep{c['replicate']}/tau{c['tau']}")


def check_reference(rows, saved):
    """The default settings on identity folds must equal the saved F1 rows field for field."""
    problems = []
    for error, tag in REFERENCE_TAGS.items():
        got = next((r for r in rows if r.get("group") == "reference" and r["reviewer_error"] == error), None)
        want = saved.get(tag)
        if got is None or want is None:
            problems.append(f"Missing reference row for {tag}.")
            continue
        for field in REFERENCE_FIELDS:
            g, w = got[field], want[field]
            same = abs(float(g) - float(w)) < 1e-9 if isinstance(w, float) else g == w
            if not same:
                problems.append(f"{tag} {field}: {g} vs saved {w}.")
    return {"fields": list(REFERENCE_FIELDS), "problems": problems, "passed": not problems}


def _median(values):
    return float(np.median(values)) if len(values) else None


def declared_rules(rows, w1_decisions, folds=PRIMARY_FOLDS):
    """Mechanical checks of N2-R1..R4 per setting on the given folds; candidate = cheapest passing."""
    grid = [r for r in rows if r.get("group") == "grid" and r["folds"] == folds]
    by_setting = {}
    for r in grid:
        by_setting.setdefault(setting_key(r), []).append(r)
    limit = R2_SHARE_OF_W1 * w1_decisions
    settings = []
    for key, rs in sorted(by_setting.items()):
        perfect = next(r for r in rs if r["reviewer_error"] == 0)
        noisy = {e: [r for r in rs if r["reviewer_error"] == e] for e in R3_LIMITS}
        per_region = {name: perfect[f"{name}_pruning_duplicates_per_1000"] for name in REGION_NAMES.values()}
        r1 = all(v is not None and v <= R1_EVENTS_PER_1000 for v in per_region.values()) \
            and abs(perfect["count_error"]) <= 1
        r2 = perfect["decisions"] <= limit
        r3_detail = {str(e): {"median_misassigned": _median([r["misassigned_photo_rate"] for r in noisy[e]]),
                              "median_abs_count_error": _median([abs(r["count_error"]) for r in noisy[e]]),
                              "replicates": len(noisy[e])} for e in R3_LIMITS}
        r3 = all(d["replicates"] and d["median_misassigned"] <= R3_LIMITS[float(e)]["misassigned"]
                 and d["median_abs_count_error"] <= R3_LIMITS[float(e)]["count"] for e, d in r3_detail.items())
        wrong = [r["wrong_duplicate_merges"] for r in noisy[0.03]]
        share = [r["wrong_duplicate_merges"] / r["duplicate_merges"] if r["duplicate_merges"] else 0.0
                 for r in noisy[0.03]]
        r4 = bool(wrong) and _median(wrong) <= R4_MAX_WRONG_MERGES and _median(share) <= R4_MAX_WRONG_SHARE
        settings.append({"eps": key[0], "min_shown": key[1], "cosine_floor": key[2], "merge_confirm": key[3],
                         "decisions": int(perfect["decisions"]), "automation": perfect["automation"],
                         "count_error": int(perfect["count_error"]),
                         "misassigned_photo_rate": perfect["misassigned_photo_rate"],
                         "pruning_duplicates_per_1000": per_region, "R1": r1, "R2": r2, "R3": r3,
                         "R3_detail": r3_detail, "R4": r4, "R4_median_wrong_merges": _median(wrong),
                         "R4_median_wrong_share": _median(share), "all": r1 and r2 and r3 and r4})
    passing = sorted((s for s in settings if s["all"]),
                     key=lambda s: (s["decisions"], -s["eps"], s["min_shown"], s["cosine_floor"], s["merge_confirm"]))
    candidate = passing[0] if passing else None
    r1_and_r2 = [s for s in settings if s["R1"] and s["R2"]]
    return {"folds": folds, "w1_decisions": int(w1_decisions), "r2_limit": limit, "settings": settings,
            "passing": [{k: s[k] for k in ("eps", "min_shown", "cosine_floor", "merge_confirm", "decisions")}
                        for s in passing],
            "candidate": candidate,
            "redirect_full_scan": not r1_and_r2,
            "redirect_adjudication_queue": bool(settings) and not any(s["R4"] for s in settings if s["merge_confirm"]),
            "policy": "Mechanical checks of the 19 Oct 2026 N2 proposal; not conclusions."}


# --- Report ---------------------------------------------------------------------------------

def write_report(out, results):
    import html
    import json
    from pathlib import Path
    from reid_demo.benchmark import write_json
    from reid_demo.benchmark_cluster_diagnostics import _write_table
    from reid_demo.benchmark_review_simulation import plain
    out = Path(out)
    results = plain(results)
    rows = results["simulations"]
    write_json(out / "summary.json", {**results, "notes": NOTES})
    fields = list(dict.fromkeys(k for r in rows for k in r))
    _write_table(out / "simulations.csv", rows, fields)
    settings = [{"folds": rules["folds"], **{k: v for k, v in s.items() if k != "R3_detail"},
                 "R3_detail": s["R3_detail"]} for rules in results["declared_rules"].values() for s in rules["settings"]]
    _write_table(out / "settings.csv", settings, list(settings[0]))
    keys = ["folds", "eps", "min_shown", "cosine_floor", "merge_confirm", "decisions", "automation", "count_error",
            "misassigned_photo_rate", "pruning_duplicates_per_1000", "R1", "R2", "R3", "R4",
            "R4_median_wrong_merges", "all"]

    def fmt(v):
        return f"{v:.4g}" if isinstance(v, float) else json.dumps(v) if isinstance(v, (dict, list)) else str(v)

    table = "<table><tr>" + "".join(f"<th>{k}</th>" for k in keys) + "</tr>" + "".join(
        "<tr>" + "".join(f"<td>{html.escape(fmt(s.get(k)))}</td>" for k in keys) + "</tr>" for s in settings) + "</table>"
    primary = results["declared_rules"][PRIMARY_FOLDS]
    body = ("<p>Pruning floors and duplicate-merge second check on calibration (F1 vectors), simulated reviewers. "
            f"Primary folds: {PRIMARY_FOLDS}. W1 (live): {primary['w1_decisions']} decisions; R2 limit "
            f"{primary['r2_limit']:.1f}.</p><h2>Reference check</h2><pre>"
            + html.escape(json.dumps(results["reference_check"], indent=1)) + "</pre><h2>Candidate (primary folds)</h2><pre>"
            + html.escape(json.dumps(primary["candidate"], indent=1)) + "</pre><p>Redirect to full scan: "
            f"{primary['redirect_full_scan']}; redirect merges to adjudication: {primary['redirect_adjudication_queue']}"
            "</p><h2>All settings (rules at perfect reviewer; R3/R4 medians over replicates)</h2>" + table
            + "<h2>Limitations</h2><ul>" + "".join(f"<li>{html.escape(n)}</li>" for n in NOTES) + "</ul>")
    (out / "index.html").write_text("<!doctype html><html lang='en'><meta charset='utf-8'><title>CzechLynx pruning "
        "safeguards</title><style>body{font:14px system-ui;max-width:1600px;margin:30px auto;padding:16px}table{"
        "border-collapse:collapse}td,th{border:1px solid #bbb;padding:3px 6px}pre{white-space:pre-wrap}</style>"
        "<h1>CzechLynx pruning floors and duplicate-merge check (N2)</h1>" + body + "</html>")
    return {str(p.relative_to(out)): file_digest(p) for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "manifest.json"}
