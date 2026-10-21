"""Timed reviewer trial (N3): traced workflow replays, task sampling and measured re-costing.

The builder replays full ranked matching (W1) and the N2 candidate on calibration with a perfect
reviewer, records what every decision showed, and samples a fixed, stratified task set for the
sibling matching-review app. The analysis joins the app's timed answers with the reference truth,
estimates wall-clock time per workflow and reviewer error rates, and applies the pre-registered rule.

The committed simulation modules are not edited (their hashes bind completed runs).
``TracedSafeguardSimulation`` overrides ``place`` and ``duplicate_check`` with copies of the
``SafeguardSimulation`` bodies plus trace records. With no per-kind reviewers it makes the same
reviewer calls in the same order, so it must reproduce the saved N2 row exactly.
Calibration development evidence only.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib

import numpy as np
import pandas as pd

from reid_demo.benchmark import file_digest
from reid_demo.benchmark_clustering import ROOT
from reid_demo.benchmark_gated_matching import wilson_interval
from reid_demo.benchmark_review_simulation import Reviewer
from reid_demo import benchmark_nonmatch_pruning as nm
from reid_demo import benchmark_pruning_safeguards as ps


VERSION = "calibration_timed_review_v1"
TASK_SET_KIND = "czechlynx_timed_review_tasks_v1"
EXPORT_KIND = "czechlynx_matching_review_export_v1"
KINDS = ps.KINDS
FIRST_PASS = ("rank1_comparison", "further_comparison")
ARMS = ("W1", "N2")
#: The N2 mechanical candidate (19 Oct 2026): frozen (c) operating point plus the merge check.
CANDIDATE = {**ps.BASE, "eps": 0.001, "min_shown": 1, "cosine_floor": 0, "merge_confirm": True,
             "reviewer_error": 0.0, "replicate": 0}
CANDIDATE_FOLDS = "region"
CANDIDATE_TAG = ps.config_tag(CANDIDATE_FOLDS, CANDIDATE)
#: Saved safeguard-free cosine full scan on the same F1 vectors (W1 with the rank-1/further split).
W1_REFERENCE_TAG = "reference/full_scan_no_safeguards/err0.0/cosine/eps0.0/tauNone"
W1_K2_MAX_RANK = 10
#: Main tasks per reviewer, by (arm, stratum); "+" marks injected opposite-answer tasks.
ALLOCATION = {
    ("W1", "N"): 12, ("N2", "N"): 12, ("N2", "S"): 12, ("N2", "D"): 8, ("W1", "K3"): 20, ("N2", "S+"): 3,
    ("W1", "K2"): 58, ("N2", "K2"): 35, ("N2", "A"): 16, ("N2", "A+"): 4, ("W1", "K1"): 60, ("N2", "K1"): 35,
    ("N2", "P"): 20, ("N2", "P+"): 5,
}
PRACTICE = {
    ("W1", "N"): 2, ("N2", "N"): 1, ("N2", "S"): 1, ("N2", "D"): 1, ("W1", "K3"): 1, ("W1", "K2"): 3,
    ("N2", "K2"): 2, ("N2", "A"): 2, ("W1", "K1"): 4, ("N2", "K1"): 2, ("N2", "P"): 1,
}
NEW_ANIMAL_TERCILES = 3
EXEMPLARS = 4
MORE_EXEMPLARS = 20  # "Show all" adds up to 20 more (24 in total).
QUERY_CAP = 24
BLOCKS_PER_ARM = 6
ARM_PATTERN = ("A", "B", "B", "A") * 3
REVIEWER_SLOTS = {"R1": "W1", "R2": "N2", "R3": "W1", "R4": "N2"}  # First arm of each slot's ABBA order.
TYPES = {
    "placement": {"banner": "Find this lynx in the catalogue",
                  "query_label": "New photos (one camera, one day)", "candidate_label": "Catalogue animal"},
    "second_check": {"banner": "Second check before a new animal is created",
                     "query_label": "New photos (one camera, one day)", "candidate_label": "Catalogue animal"},
    "disagreement": {"banner": "Second opinion: is this the same lynx?",
                     "query_label": "New photos (one camera, one day)", "candidate_label": "Catalogue animal"},
    "pair": {"banner": "Are these two catalogue animals the same lynx?",
             "query_label": "Catalogue animal A", "candidate_label": "Catalogue animal B"},
    "audit": {"banner": "The system assigned these photos to this lynx. Is that correct?",
              "query_label": "New photos (one camera, one day)", "candidate_label": "Assigned animal"},
}
SCREEN_KIND = {"second_check": "second_comparison", "disagreement": "disagreement_check",
               "pair": "pair_question", "audit": "audit_glance"}
RULE = {
    "success_saving": 0.40,
    "reframe_saving": 0.25,
    "estimator": "stratified mean capped decision time x full-replay decisions per (arm, stratum, kind)",
    "cap_seconds": 120.0,
    "sensitivity_caps": [60.0, None],
    "idle_pause_seconds": 120,
    "bootstrap_resamples": 2000,
    "bootstrap_seed": 42,
    "untimed_seconds": {"extra_animal": 5.0},
    "extra_animal_sensitivity_seconds": 25.0,
    "merge_confirmation": "costed as the measured pair-question mean",
    "count_audit_comparison": "excluded, as in the N2 hours",
    "injected_time_weight": 0,
    "unsure": "neither correct nor an error (sensitivity: an error); the workflow treats it as Different",
    "early_stop_blocks": 8,
    "w1_favourable_quantile": 0.25,
    "w1_favourable_strata": ["K3", "N"],
    "resimulation_seeds": 5,
}
NOTES = [
    "Calibration development evidence only (F1 vectors); selection is spent and the official test is locked.",
    "Tasks are precomputed from perfect-reviewer replays: a reviewer's mistake does not change later tasks. "
    "Propagation comes from re-running the sequential simulations at the measured error rates.",
    "Each query shows one animal's photos from a camera-day bundle (the simulations' truth split).",
    "Reviewers see no identity labels, file names, scores or routes; images are served by opaque tokens.",
    "Injected opposite-answer tasks (second checks, pair questions, audits) count for error rates only.",
    "Reviewers are the user and possibly one more person, not the partner's experts.",
]


def implementation_binding():
    names = ("reid_demo/benchmark_timed_review.py", "tools/build_czechlynx_timed_review_tasks.py",
             "tools/recost_czechlynx_timed_review.py", "reid_demo/benchmark_pruning_safeguards.py",
             "reid_demo/benchmark_nonmatch_pruning.py", "reid_demo/benchmark_gated_matching.py",
             "reid_demo/benchmark_review_simulation.py", "tools/simulate_czechlynx_pruning_safeguards.py")
    return {"version": VERSION, "code": {n: file_digest(ROOT / n) for n in names},
            "candidate_tag": CANDIDATE_TAG, "w1_reference_tag": W1_REFERENCE_TAG,
            "allocation": {f"{a}|{s}": n for (a, s), n in ALLOCATION.items()},
            "practice": {f"{a}|{s}": n for (a, s), n in PRACTICE.items()},
            "exemplars": EXEMPLARS, "more_exemplars": MORE_EXEMPLARS, "query_cap": QUERY_CAP, "rule": RULE}


# --- Traced catalogue and simulations -------------------------------------------------------

class TracedCatalogue(nm.LiveCatalogue):
    """``LiveCatalogue`` that can reconstruct any earlier state: every add and merge gets a sequence number."""

    def __init__(self, n_images):
        super().__init__(n_images)
        self.seq = 0
        self.added_seq = np.full(n_images, -1)
        self.first_uid = np.full(n_images, -1)
        self.founder_of_uid = {}
        self.merges = []

    def add(self, images, entry=None):
        entry = super().add(images, entry)
        images = np.asarray(images, int)
        self.added_seq[images] = self.seq
        self.first_uid[images] = self.uid[entry]
        self.seq += 1
        return entry

    def create(self, images, founder):
        entry = super().create(images, founder)
        self.founder_of_uid[int(self.uid[entry])] = founder
        return entry

    def merge(self, keep, drop):
        self.merges.append((self.seq, int(self.uid[keep]), int(self.uid[drop])))
        super().merge(keep, drop)
        self.seq += 1

    def owners_at(self, seq):
        """uid owning each image just before operation `seq` (-1: not yet in the catalogue)."""
        resolve = {dropped: kept for s, kept, dropped in self.merges if s < seq}
        owners = np.where((self.added_seq >= 0) & (self.added_seq < seq), self.first_uid, -1)
        if resolve:
            def root(u):
                while u in resolve:
                    u = resolve[u]
                return u
            owners = np.asarray([root(int(u)) if u >= 0 else -1 for u in owners])
        return owners

    def members_at(self, uid, seq):
        return np.flatnonzero(self.owners_at(seq) == uid)


class TracedSafeguardSimulation(ps.SafeguardSimulation):
    """``SafeguardSimulation`` that records each decision (kind, shown animal, step, truth, answer).

    ``kind_reviewers`` maps ledger kinds to reviewer objects (re-simulation at measured rates); kinds
    not listed use the expert or second reviewer exactly as the parent does."""

    def __init__(self, *args, kind_reviewers=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.catalogue = TracedCatalogue(len(self.codes))
        self.kind_reviewers = dict(kind_reviewers or {})
        unknown = sorted(set(self.kind_reviewers) - set(KINDS))
        if unknown:
            raise ValueError(f"Unknown decision kinds for reviewers: {unknown}")
        self.parts, self.pairs, self.autos = [], [], []

    def _rv(self, kind, default):
        return self.kind_reviewers.get(kind, default)

    def _maybe_check(self, b):
        if self.auto_log and self.auto_log[-1]["bundle"] == b and (not self.autos or self.autos[-1]["bundle"] != b):
            images = [int(i) for i in self.auto_log[-1]["images"]]
            cat = self.catalogue
            self.autos.append({"bundle": b, "images": images, "uid": int(cat.uid[cat.entry_of[images[0]]]),
                               "seq": int(cat.added_seq[images[0]]), "correct": bool(self.auto_log[-1]["correct"])})
        super()._maybe_check(b)

    def place(self, identity, part):
        # Copy of ps.SafeguardSimulation.place with trace records; reviewer calls are unchanged.
        cat, s = self.catalogue, self.s
        true_entries = {e for e, f in enumerate(cat.founder) if f == identity}  # Evaluation log only.
        record = {"bundle": self.bundle, "identity": int(identity), "photos": len(part),
                  "true_in_catalogue": bool(true_entries), "guard_open": self.guard_open(), "shown": 0,
                  "disagreement_checks": 0, "second_comparisons": 0, "first_correct": None,
                  "floor_shown": 0, "matched_by_floor": False}
        trace = {"bundle": self.bundle, "part": [int(i) for i in part], "identity": int(identity), "seq": cat.seq,
                 "known": bool(true_entries), "true_uids": sorted(int(cat.uid[e]) for e in true_entries),
                 "order": [], "shortlist": 0, "screens": [], "second": [], "decisions": []}
        self.parts.append(trace)

        def finish():
            trace["outcome"] = record["outcome"]

        if not cat.founder:
            self._declare(identity, part, record, cut_true=False)
            finish()
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
        uid = {e: int(cat.uid[e]) for e in order}
        shortlist = len(order)
        if pruning:
            shortlist = next((pos for pos in range(len(order)) if pos >= s["min_shown"] and suffix[pos] < s["eps"]),
                             len(order))
        listed = set(order[:shortlist])
        floor = [e for e in cosine[:s["cosine_floor"]] if e not in listed] if shortlist < len(order) else []
        trace.update(order=[uid[e] for e in order], shortlist=shortlist,
                     screens=[uid[e] for e in order[:shortlist] + floor])
        shown = set()

        def ask(e):
            kind = "rank1_comparison" if not shown else "further_comparison"
            self.ledger.add(kind)
            shown.add(e)
            truth = cat.founder[e] == identity
            answer = self._rv(kind, self.expert).same(truth)
            trace["decisions"].append({"kind": kind, "uid": uid[e], "step": len(shown), "truth": bool(truth),
                                       "answer": bool(answer)})
            if answer:
                if s["disagree_delta"] is not None and p[e] < s["disagree_delta"]:
                    self.ledger.add("disagreement_check")
                    record["disagreement_checks"] += 1
                    second = self._rv("disagreement_check", self.second).same(truth)
                    trace["decisions"].append({"kind": "disagreement_check", "uid": uid[e], "step": 1,
                                               "truth": bool(truth), "answer": bool(second)})
                    if not second:
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
                finish()
                return
        if stopped:
            for e in cosine[:s["cosine_floor"]]:
                if e in shown:
                    continue
                record["floor_shown"] += 1
                if ask(e):
                    record["matched_by_floor"] = True
                    finish()
                    return
        if s["double_c"]:
            second_list = sorted(cosine, key=lambda e: (-p[e], position[e]))[:s["double_c"]]
            trace["second"] = [int(cat.uid[e]) for e in second_list]
            for step, e in enumerate(second_list, 1):
                self.ledger.add("second_comparison")
                record["second_comparisons"] += 1
                truth = cat.founder[e] == identity
                answer = self._rv("second_comparison", self.second).same(truth)
                trace["decisions"].append({"kind": "second_comparison", "uid": int(cat.uid[e]), "step": step,
                                           "truth": bool(truth), "answer": bool(answer)})
                if answer:
                    cat.add(part, e)
                    self._log(record, shown, "rescued_correct" if truth else "rescued_wrong")
                    finish()
                    return
        record["shown"] = len(shown)
        self._declare(identity, part, record, cut_true=bool(true_entries) and not (true_entries & shown))
        finish()

    def duplicate_check(self):
        # Copy of ps.SafeguardSimulation.duplicate_check with trace records.
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
                pair = {"bundle": self.bundle, "seq": cat.seq, "uid_new": int(uid), "uid_other": int(cat.uid[a]),
                        "truth": bool(truth)}
                answer = self._rv("pair_question", self.expert).same(truth)
                self.pairs.append({"kind": "pair_question", **pair, "answer": bool(answer)})
                if answer:
                    if self.s["merge_confirm"]:
                        self.ledger.add("merge_confirmation")
                        confirm = self._rv("merge_confirmation", self.second).same(truth)
                        self.pairs.append({"kind": "merge_confirmation", **pair, "answer": bool(confirm)})
                        if not confirm:
                            self.blocked_merges.append({"bundle": self.bundle, "uids": sorted(key),
                                                        "true_duplicate": bool(truth)})
                            continue
                    self.duplicate_log.append({"bundle": self.bundle, "kept_uid": cat.uid[min(a, b)],
                                               "merged_uid": cat.uid[max(a, b)], "correct": bool(truth)})
                    cat.merge(min(a, b), max(a, b))
                    break


def traced_w1(units, scores, n_images, reviewers=None):
    """W1: embedding-ranked catalogue matching with a full scan before a new animal, with a decision trace.

    With no ``reviewers`` it is the perfect-reviewer replay of ``simulate_catalogue(ranked=True)``.
    ``reviewers`` maps rank1_comparison / further_comparison to reviewer objects.
    Returns (catalogue, ledger Counter, part traces)."""
    reviewers = dict(reviewers or {})
    unknown = sorted(set(reviewers) - set(FIRST_PASS))
    if unknown:
        raise ValueError(f"W1 has only first-pass comparisons, not {unknown}")
    perfect = Reviewer(0.0)
    cat, ledger, parts = TracedCatalogue(n_images), Counter(), []
    for unit in units:
        ledger["extra_animal"] += int(unit["extra"])
        images, identity = unit["images"], unit["identity"]
        order = cat.ranked(scores, images)
        uids = [int(cat.uid[e]) for e in order]
        trace = {"bundle": unit["bundle"], "part": [int(i) for i in images], "identity": int(identity), "seq": cat.seq,
                 "known": identity in cat.founder,
                 "true_uids": [int(cat.uid[e]) for e, f in enumerate(cat.founder) if f == identity],
                 "order": uids, "shortlist": len(order), "screens": list(uids), "second": [], "decisions": []}
        parts.append(trace)
        for step, e in enumerate(order, 1):
            kind = FIRST_PASS[0] if step == 1 else FIRST_PASS[1]
            ledger[kind] += 1
            truth = cat.founder[e] == identity
            answer = reviewers.get(kind, perfect).same(truth)
            trace["decisions"].append({"kind": kind, "uid": uids[step - 1], "step": step, "truth": bool(truth),
                                       "answer": bool(answer)})
            if answer:
                cat.add(images, e)
                trace["outcome"] = "matched_correct" if truth else "matched_wrong"
                break
        else:
            cat.create(images, identity)
            trace["outcome"] = "new_reviewer_duplicate" if trace["known"] else "new"
    return cat, ledger, parts


# --- Strata and counts ----------------------------------------------------------------------

def placement_stratum(arm, trace):
    """K1 matched at the first candidate, K2 later (W1: up to rank 10), K3 W1 beyond rank 10, N no match."""
    if not trace["outcome"].startswith("matched"):
        return "N"
    step = max(d["step"] for d in trace["decisions"] if d["kind"] in FIRST_PASS)
    if step == 1:
        return "K1"
    return "K2" if arm == "N2" or step <= W1_K2_MAX_RANK else "K3"


def decision_counts(w1_parts, w1_ledger, sim, audit_glances):
    """Full-replay decisions per (arm, stratum, kind); they sum to the ledgers' per-kind totals."""
    counts = Counter()
    for t in w1_parts:
        stratum = placement_stratum("W1", t)
        for d in t["decisions"]:
            counts[("W1", stratum, d["kind"])] += 1
    counts[("W1", "X", "extra_animal")] += int(w1_ledger["extra_animal"])
    for t in sim.parts:
        stratum = placement_stratum("N2", t)
        for d in t["decisions"]:
            counts[("N2", {"second_comparison": "S", "disagreement_check": "D"}.get(d["kind"], stratum), d["kind"])] += 1
    for q in sim.pairs:
        counts[("N2", "P" if q["kind"] == "pair_question" else "M", q["kind"])] += 1
    counts[("N2", "A", "audit_glance")] += int(audit_glances)
    counts[("N2", "X", "extra_animal")] += int(sim.ledger.counts.get("extra_animal", 0))
    return {k: int(v) for k, v in sorted(counts.items()) if v}


def kind_totals(counts, arm):
    totals = Counter()
    for (a, _, kind), n in counts.items():
        if a == arm:
            totals[kind] += n
    return {k: int(totals.get(k, 0)) for k in KINDS}


def populations(w1_parts, sim, audit_glances):
    """Task units per (arm, stratum) in the full replay (Horvitz-Thompson totals)."""
    pop = Counter()
    for t in w1_parts:
        if t["decisions"]:
            pop[("W1", placement_stratum("W1", t))] += 1
    for t in sim.parts:
        if any(d["kind"] in FIRST_PASS for d in t["decisions"]):
            pop[("N2", placement_stratum("N2", t))] += 1
        pop[("N2", "S")] += int(any(d["kind"] == "second_comparison" for d in t["decisions"]))
        pop[("N2", "D")] += sum(d["kind"] == "disagreement_check" for d in t["decisions"])
    pop[("N2", "P")] += sum(q["kind"] == "pair_question" for q in sim.pairs)
    pop[("N2", "M")] += sum(q["kind"] == "merge_confirmation" for q in sim.pairs)
    pop[("N2", "A")] += int(audit_glances)
    return {k: int(v) for k, v in sorted(pop.items()) if v}


# --- Task units -----------------------------------------------------------------------------

def _placement_unit(arm, i, t, screens, kinds, unit_type, stratum, injected=False):
    true = set(t["true_uids"])
    return {"arm": arm, "stratum": stratum, "injected": injected, "type": unit_type, "source_index": i,
            "query": list(t["part"]), "seq": t["seq"], "screens": list(screens), "kinds": list(kinds),
            "truth": [u in true for u in screens], "bundle": t["bundle"], "identity": t["identity"]}


def candidate_units(w1_parts, sim, codes, dates):
    """Every possible task unit per (arm, stratum[+]), from the traces. Truth is evaluation-only."""
    units = defaultdict(list)
    codes = np.asarray(codes)
    for i, t in enumerate(w1_parts):
        if t["screens"]:
            kinds = [FIRST_PASS[0]] + [FIRST_PASS[1]] * (len(t["screens"]) - 1)
            units[("W1", placement_stratum("W1", t))].append(
                _placement_unit("W1", i, t, t["screens"], kinds, "placement", placement_stratum("W1", t)))
    for i, t in enumerate(sim.parts):
        stratum = placement_stratum("N2", t)
        if t["screens"]:
            kinds = [FIRST_PASS[0]] + [FIRST_PASS[1]] * (len(t["screens"]) - 1)
            units[("N2", stratum)].append(_placement_unit("N2", i, t, t["screens"], kinds, "placement", stratum))
        if t["second"]:
            units[("N2", "S")].append(_placement_unit("N2", i, t, t["second"], ["second_comparison"] * len(t["second"]),
                                                      "second_check", "S"))
        for d in t["decisions"]:
            if d["kind"] == "disagreement_check":
                unit = _placement_unit("N2", i, t, [d["uid"]], ["disagreement_check"], "disagreement", "D")
                unit["truth"] = [bool(d["truth"])]
                units[("N2", "D")].append(unit)
        double_c = int(sim.s["double_c"])
        if stratum in ("K1", "K2") and double_c and set(t["true_uids"]) & set(t["order"][:double_c]):
            top = t["order"][:double_c]
            units[("N2", "S+")].append(_placement_unit("N2", i, t, top, ["second_comparison"] * len(top),
                                                       "second_check", "S", injected=True))
    for j, q in enumerate(sim.pairs):
        if q["kind"] == "pair_question":
            units[("N2", "P")].append({"arm": "N2", "stratum": "P", "injected": False, "type": "pair",
                                       "source_index": j, "left_uid": q["uid_new"], "seq": q["seq"],
                                       "screens": [q["uid_other"]], "kinds": ["pair_question"],
                                       "truth": [bool(q["truth"])], "bundle": q["bundle"]})
    cat = sim.catalogue
    final = cat.owners_at(cat.seq)
    for uid in sorted(set(final[final >= 0].tolist())):
        members = np.flatnonzero(final == uid)
        days = sorted({dates[m] for m in members})
        if len(days) >= 2:
            cut = days[len(days) // 2]
            left = [int(m) for m in members if dates[m] < cut]
            right = [int(m) for m in members if dates[m] >= cut]
            units[("N2", "P+")].append({"arm": "N2", "stratum": "P", "injected": True, "type": "pair",
                                        "source_index": int(uid), "left_members": left, "right_members": right,
                                        "seq": cat.seq, "screens": [int(uid)], "kinds": ["pair_question"],
                                        "truth": [True], "bundle": None})
    for j, a in enumerate(sim.autos):
        base = {"arm": "N2", "stratum": "A", "type": "audit", "source_index": j, "query": list(a["images"]),
                "seq": a["seq"], "kinds": ["audit_glance"], "bundle": a["bundle"]}
        units[("N2", "A")].append({**base, "injected": False, "screens": [a["uid"]], "truth": [a["correct"]]})
        owners = cat.owners_at(a["seq"])
        runner = _best_other(sim.ctx.scores, a["images"], owners, exclude=a["uid"])
        if runner is not None:
            founder = cat.founder_of_uid.get(runner)
            truth = founder is not None and bool(np.all(codes[a["images"]] == founder))
            units[("N2", "A+")].append({**base, "injected": True, "screens": [runner], "truth": [truth]})
    return units


def _best_other(scores, images, owners, exclude):
    present = owners >= 0
    if not present.any():
        return None
    best = scores[np.ix_(np.asarray(images, int), np.flatnonzero(present))].max(axis=0)
    by_uid = {}
    for m, value in zip(np.flatnonzero(present), best):
        u = int(owners[m])
        if u != exclude and value > by_uid.get(u, -np.inf):
            by_uid[u] = value
    return min(by_uid, key=lambda u: (-by_uid[u], u)) if by_uid else None


def _query_images(unit):
    return set(unit.get("query", []))


def sample_units(units, allocation, rng, used=None):
    """Seeded stratified sampling. Query photos never repeat across tasks; W1 new-animal tasks are spread
    over catalogue-size terciles. Returns ({(arm, stratum+): [unit]}, shortfalls, used photos)."""
    used = set() if used is None else used
    chosen, shortfalls = {}, {}
    for key, wanted in allocation.items():
        pool = list(units.get(key, []))
        if key == ("W1", "N"):
            pool.sort(key=lambda u: (len(u["screens"]), u["source_index"]))
            groups = [list(g) for g in np.array_split(np.arange(len(pool)), NEW_ANIMAL_TERCILES)]
            quotas = [wanted // NEW_ANIMAL_TERCILES] * NEW_ANIMAL_TERCILES
            quotas[-1] += wanted - sum(quotas)
            picked = []
            for group, quota in zip(groups, quotas):
                picked.extend(_take([pool[i] for i in group], quota, rng, used))
        else:
            picked = _take(pool, wanted, rng, used)
        chosen[key] = picked
        if len(picked) < wanted:
            shortfalls[f"{key[0]}|{key[1]}"] = {"wanted": wanted, "sampled": len(picked)}
    return chosen, shortfalls, used


def _take(pool, n, rng, used):
    picked = []
    for i in rng.permutation(len(pool)):
        if len(picked) >= n:
            break
        unit = pool[int(i)]
        images = _query_images(unit)
        if images & used:
            continue
        used |= images
        picked.append(unit)
    return picked


def sample_practice(units, practice, rng, used):
    """Practice tasks from units not used by the main set; W1 new-animal practice uses the smallest catalogues."""
    chosen, shortfalls = {}, {}
    for key, wanted in practice.items():
        pool = list(units.get(key, []))
        if key == ("W1", "N"):
            pool.sort(key=lambda u: (len(u["screens"]), u["source_index"]))
            picked = []
            for unit in pool:
                if len(picked) >= wanted:
                    break
                if not _query_images(unit) & used:
                    used |= _query_images(unit)
                    picked.append(unit)
        else:
            picked = _take(pool, wanted, rng, used)
        chosen[key] = picked
        if len(picked) < wanted:
            shortfalls[f"{key[0]}|{key[1]}"] = {"wanted": wanted, "sampled": len(picked)}
    return chosen, shortfalls


# --- Payloads -------------------------------------------------------------------------------

def image_token(salt, image_id):
    return hashlib.sha256(f"{salt}:{image_id}".encode()).hexdigest()[:24]


def rank_members(scores, members, against):
    """Members ordered by their best cosine to any image in `against` (label-free), ties by index."""
    members = np.asarray(sorted(int(m) for m in members), int)
    if not len(members):
        return []
    best = scores[np.ix_(np.asarray(list(against), int), members)].max(axis=0)
    return [int(members[i]) for i in np.lexsort((members, -best))]


def build_task(task_id, unit, catalogue, scores, photo, practice=False):
    """Browser payload (no identity, path, score, stratum or truth except practice feedback) and its
    evaluation-only truth record. `photo(i)` returns the display dict for image index i."""
    kind_info = TYPES[unit["type"]]
    owners = catalogue.owners_at(unit["seq"])
    if unit["type"] == "pair":
        left = (unit["left_members"] if unit["injected"] else np.flatnonzero(owners == unit["left_uid"]).tolist())
        right = (unit["right_members"] if unit["injected"] else np.flatnonzero(owners == unit["screens"][0]).tolist())
        query = rank_members(scores, left, right)[:QUERY_CAP]
        candidates = [rank_members(scores, right, left)]
    else:
        query = sorted(unit["query"])[:QUERY_CAP]
        candidates = [rank_members(scores, np.flatnonzero(owners == u), unit["query"]) for u in unit["screens"]]
    if any(not c for c in candidates) or not query:
        raise ValueError(f"Task {task_id} has an empty side.")
    n = len(unit["screens"])
    screens = [{"step": i + 1, "kind": unit["kinds"][i], "position": i + 1, "list_length": n,
                "exemplars": [photo(m) for m in c[:EXEMPLARS]],
                "more": [photo(m) for m in c[EXEMPLARS:EXEMPLARS + MORE_EXEMPLARS]],
                "members": len(c)} for i, c in enumerate(candidates)]
    payload = {"task_id": task_id, "arm": unit["arm"], "type": unit["type"], "practice": bool(practice),
               **kind_info, "query": [photo(m) for m in query], "screens": screens}
    if practice:
        payload["feedback"] = ["same" if t else "different" for t in unit["truth"]]
    truth = {"arm": unit["arm"], "type": unit["type"], "stratum": unit["stratum"], "injected": bool(unit["injected"]),
             "practice": bool(practice), "screens_truth": [bool(t) for t in unit["truth"]],
             "kinds": list(unit["kinds"]), "true_step": next((i + 1 for i, t in enumerate(unit["truth"]) if t), None),
             "query_images": [int(i) for i in query],
             "candidate_images": [[int(i) for i in c[:EXEMPLARS + MORE_EXEMPLARS]] for c in candidates],
             "seq": int(unit["seq"]), "bundle": unit.get("bundle"), "source_index": unit["source_index"]}
    images = set(query) | {m for c in candidates for m in c[:EXEMPLARS + MORE_EXEMPLARS]}
    return payload, truth, images


def make_blocks(main, rng):
    """Per arm: tasks dealt round-robin by stratum into BLOCKS_PER_ARM stratum-balanced blocks."""
    blocks = {}
    for arm in ARMS:
        ids = [(t["stratum_key"], t["task_id"]) for t in main if t["arm"] == arm]
        by_stratum = defaultdict(list)
        for stratum, task_id in sorted(ids):
            by_stratum[stratum].append(task_id)
        dealt, k = [[] for _ in range(BLOCKS_PER_ARM)], 0
        for stratum in sorted(by_stratum):
            group = by_stratum[stratum]
            for i in rng.permutation(len(group)):
                dealt[k % BLOCKS_PER_ARM].append(group[int(i)])
                k += 1
        blocks[arm] = dealt
    return blocks


def reviewer_orders(blocks, practice_ids, seed):
    """ABBA arm order per reviewer slot (R1/R3 start with W1, R2/R4 with N2); shuffled within blocks."""
    orders = {}
    for slot, first in REVIEWER_SLOTS.items():
        other = "N2" if first == "W1" else "W1"
        rng = np.random.default_rng([seed, int(slot[1:])])
        used = Counter()
        sequence = []
        for index, letter in enumerate(ARM_PATTERN):
            arm = first if letter == "A" else other
            tasks = list(blocks[arm][used[arm]])
            used[arm] += 1
            sequence.append({"block": index + 1, "arm": arm, "arm_block": used[arm],
                             "tasks": [tasks[int(i)] for i in rng.permutation(len(tasks))]})
        practice = [practice_ids[int(i)] for i in rng.permutation(len(practice_ids))]
        orders[slot] = {"practice": practice, "blocks": sequence}
    return orders


# --- Timing from app events -----------------------------------------------------------------

def attempt_events(events, task_id):
    """Events of the completed attempt: after the last task_start before the task_complete."""
    events = sorted(events, key=lambda e: e["seq"])
    end = next((i for i in range(len(events) - 1, -1, -1)
                if events[i]["type"] == "task_complete" and events[i].get("task_id") == task_id), None)
    if end is None:
        return []
    start = next((i for i in range(end, -1, -1)
                  if events[i]["type"] == "task_start" and events[i].get("task_id") == task_id), None)
    return [] if start is None else events[start:end + 1]


def paused_before(events, seq):
    """Whether the clock was paused just before event `seq` (a pause can start before a task does)."""
    state = False
    for e in sorted(events, key=lambda e: e["seq"]):
        if e["seq"] >= seq:
            break
        if e["type"] in ("pause", "resume"):
            state = e["type"] == "pause"
    return state


def step_times(events, paused_at_start=False):
    """Decision milliseconds per step: screen shown to answer, minus pauses. An interval that ends in an
    undo belongs to the step returned to. Returns {step: {"ms", "paused_ms", "undos", "intervals"}}."""
    out = defaultdict(lambda: {"ms": 0.0, "paused_ms": 0.0, "undos": 0, "intervals": 0})
    step = shown = paused_since = None
    paused, is_paused = 0.0, bool(paused_at_start)
    for e in sorted(events, key=lambda e: e["seq"]):
        kind, t = e["type"], float(e["t"])
        if kind == "screen_shown":
            # A screen that finishes loading during a pause starts paused (as the client does).
            step, shown, paused = int(e["step"]), t, 0.0
            paused_since = t if is_paused else None
        elif kind == "pause" and not is_paused:
            is_paused = True
            if shown is not None:
                paused_since = t
        elif kind == "resume" and is_paused:
            is_paused = False
            if paused_since is not None:
                paused += t - paused_since
                paused_since = None
        elif kind in ("answer", "undo"):
            if shown is None:
                continue  # E.g. an undo from the confirmation bar: no screen was showing.
            if paused_since is not None:
                paused += t - paused_since
                paused_since = None
            target = step if kind == "answer" else int(e["to_step"])
            out[target]["ms"] += max(0.0, t - shown - paused)
            out[target]["paused_ms"] += paused
            out[target]["intervals"] += 1
            out[target]["undos"] += int(kind == "undo")
            shown, paused = None, 0.0
    return dict(out)


def decision_table(tasks, truth, export):
    """One row per final answer: reviewer, task, step, arm, stratum, kind, truth, answer and seconds (event-
    derived when the attempt's events are complete, else the client's value; `timing_source` says which)."""
    if export.get("kind") != EXPORT_KIND:
        raise ValueError("Not a matching-review export.")
    events = defaultdict(list)
    for e in export["events"]:
        events[(e["reviewer"], e["session_id"])].append(e)
    completions = {(c["reviewer"], c["task_id"]): c for c in export["completions"]}
    rows = []
    for a in export["answers"]:
        key = (a["reviewer"], a["task_id"])
        task, info, done = tasks[a["task_id"]], truth[a["task_id"]], completions.get(key)
        if done is None:
            raise ValueError(f"Answer without a completed task: {key}")
        step = int(a["step"])
        session = events[(a["reviewer"], done["session_id"])]
        attempt = attempt_events(session, a["task_id"])
        times = step_times(attempt, paused_before(session, attempt[0]["seq"]) if attempt else False)
        derived = times.get(step)
        seconds = derived["ms"] / 1000 if derived and derived["intervals"] else float(a["decision_ms"]) / 1000
        rows.append({"reviewer": a["reviewer"], "task_id": a["task_id"], "step": step, "arm": info["arm"],
                     "type": info["type"], "stratum": info["stratum"], "injected": info["injected"],
                     "practice": info["practice"], "kind": task["screens"][step - 1]["kind"],
                     "truth": bool(info["screens_truth"][step - 1]), "answer": a["answer"], "seconds_raw": seconds,
                     "client_seconds": float(a["decision_ms"]) / 1000,
                     "paused": bool((derived or {}).get("paused_ms", 0) > 0 or a.get("paused_ms", 0) > 0),
                     "undos": int((derived or {}).get("undos", a.get("undo_count", 0))),
                     "timing_source": "events" if derived and derived["intervals"] else "client",
                     "unidentifiable": bool(done.get("unidentifiable", False))})
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("The export holds no answers.")
    return table


def capped(table, cap):
    return table.seconds_raw.clip(upper=cap) if cap is not None else table.seconds_raw


# --- Re-costing estimators ------------------------------------------------------------------

def _cell_key(arm, stratum, kind):
    return f"{arm}|{stratum}|{kind}"


def parse_counts(counts):
    return {tuple(k.split("|")): int(v) for k, v in counts.items()}


def cell_costs(table, counts, *, cap=RULE["cap_seconds"], statistic="mean", overrides=None):
    """Seconds per decision for every (arm, stratum, kind) cell with full-replay decisions; natural tasks only.
    Missing cells fall back to the arm's kind mean, then the kind mean over arms; `source` records it."""
    natural = table[~table.injected & ~table.practice].assign(seconds=lambda t: capped(t, cap))
    agg = (lambda s: s.mean()) if statistic == "mean" else (lambda s: s.median())
    by_cell = natural.groupby(["arm", "stratum", "kind"]).seconds.agg(agg).to_dict()
    by_arm_kind = natural.groupby(["arm", "kind"]).seconds.agg(agg).to_dict()
    by_kind = natural.groupby("kind").seconds.agg(agg).to_dict()
    n_cell = natural.groupby(["arm", "stratum", "kind"]).size().to_dict()
    rows = []
    for (arm, stratum, kind), n in sorted(counts.items()):
        if kind == "count_audit_comparison":
            continue
        if kind in RULE["untimed_seconds"]:
            cost, source = RULE["untimed_seconds"][kind], "declared"
        elif kind == "merge_confirmation":
            cost = by_cell.get((arm, "P", "pair_question"), by_kind.get("pair_question"))
            source = "pair_question"
        elif (arm, stratum, kind) in by_cell:
            cost, source = by_cell[(arm, stratum, kind)], "measured"
        elif (arm, kind) in by_arm_kind:
            cost, source = by_arm_kind[(arm, kind)], "arm_kind_fallback"
        else:
            cost, source = by_kind.get(kind), "kind_fallback"
        if overrides and (arm, stratum, kind) in overrides:
            cost, source = overrides[(arm, stratum, kind)], "override"
        rows.append({"arm": arm, "stratum": stratum, "kind": kind, "decisions": int(n),
                     "measured_decisions": int(n_cell.get((arm, stratum, kind), 0)),
                     "seconds_per_decision": None if cost is None else float(cost), "source": source,
                     "seconds": None if cost is None else float(cost) * n})
    return pd.DataFrame(rows)


def totals_and_saving(cells):
    if cells.seconds_per_decision.isna().any():
        missing = cells[cells.seconds_per_decision.isna()][["arm", "stratum", "kind"]].to_dict("records")
        raise ValueError(f"No measured time for cells: {missing}")
    totals = {arm: float(cells[cells.arm == arm].seconds.sum()) for arm in ARMS}
    return totals, 1 - totals["N2"] / totals["W1"]


def stratified_saving(table, counts, **kw):
    cells = cell_costs(table, counts, **kw)
    totals, saving = totals_and_saving(cells)
    return {"cells": cells, "totals_seconds": totals, "saving": saving}


def bootstrap_saving(table, counts, *, cap=RULE["cap_seconds"], resamples=RULE["bootstrap_resamples"],
                     seed=RULE["bootstrap_seed"]):
    """Stratified bootstrap: natural tasks resampled within (arm, stratum), all reviewers' decisions of a task
    kept together; reviewers fixed. Returns (low, high, savings array)."""
    natural = table[~table.injected & ~table.practice].assign(seconds=lambda t: capped(t, cap))
    point = cell_costs(table, counts, cap=cap)
    fallback = {(r.arm, r.stratum, r.kind): r.seconds_per_decision for r in point.itertuples()}
    rng = np.random.default_rng(seed)
    totals = {arm: np.zeros(resamples) for arm in ARMS}
    by_stratum = defaultdict(dict)
    for (arm, stratum, kind), n in sorted(counts.items()):
        if kind != "count_audit_comparison":
            by_stratum[(arm, stratum)][kind] = n
    for (arm, stratum), kinds in sorted(by_stratum.items()):
        cell = natural[(natural.arm == arm) & (natural.stratum == stratum)]
        tasks = sorted(cell.task_id.unique())
        # One draw per stratum: a task's decisions of every kind stay together.
        draw = rng.integers(0, len(tasks), size=(resamples, len(tasks))) if tasks else None
        index = {t: i for i, t in enumerate(tasks)}
        for kind, n in kinds.items():
            timed = (kind not in RULE["untimed_seconds"] and kind != "merge_confirmation"
                     and (cell.kind == kind).any())
            if not timed:
                totals[arm] += n * fallback[(arm, stratum, kind)]
                continue
            of_kind = cell[cell.kind == kind]
            position = of_kind.task_id.map(index).to_numpy()
            sums = np.bincount(position, weights=of_kind.seconds.to_numpy(), minlength=len(tasks))
            num = np.bincount(position, minlength=len(tasks)).astype(float)
            s, c = sums[draw].sum(axis=1), num[draw].sum(axis=1)
            mean = np.where(c > 0, s / np.where(c > 0, c, 1), fallback[(arm, stratum, kind)])
            totals[arm] += n * mean
    savings = 1 - totals["N2"] / totals["W1"]
    return float(np.quantile(savings, .025)), float(np.quantile(savings, .975)), savings


def pooled_kind_saving(table, saved_kinds, *, cap=RULE["cap_seconds"]):
    """The handoff's literal formula: pooled per-kind mean (natural tasks) x the saved per-kind counts."""
    natural = table[~table.injected & ~table.practice].assign(seconds=lambda t: capped(t, cap))
    totals = {}
    for arm in ARMS:
        means = natural[natural.arm == arm].groupby("kind").seconds.mean().to_dict()
        pair = natural[natural.kind == "pair_question"].seconds.mean()
        total = 0.0
        for kind, n in saved_kinds[arm].items():
            if not n or kind == "count_audit_comparison":
                continue
            cost = (RULE["untimed_seconds"].get(kind) if kind in RULE["untimed_seconds"]
                    else pair if kind == "merge_confirmation" else means.get(kind))
            if cost is None or (isinstance(cost, float) and np.isnan(cost)):
                raise ValueError(f"No measured time for {arm} {kind}.")
            total += n * cost
        totals[arm] = total
    return {"totals_seconds": totals, "saving": 1 - totals["N2"] / totals["W1"]}


def horvitz_thompson_saving(table, counts, pops, *, cap=RULE["cap_seconds"]):
    """Task-level totals: mean task time per (arm, stratum) over natural tasks (averaged over reviewers)
    x the replay's task units, plus untimed kinds at their declared costs."""
    natural = table[~table.injected & ~table.practice].assign(seconds=lambda t: capped(t, cap))
    per_task = natural.groupby(["arm", "stratum", "task_id", "reviewer"]).seconds.sum().groupby(
        ["arm", "stratum", "task_id"]).mean()
    stratum_mean = per_task.groupby(["arm", "stratum"]).mean().to_dict()
    pair = natural[natural.kind == "pair_question"].seconds.mean()
    totals = {arm: 0.0 for arm in ARMS}
    for (arm, stratum), units in pops.items():
        if stratum == "M":
            totals[arm] += units * pair
        elif (arm, stratum) in stratum_mean:
            totals[arm] += units * stratum_mean[(arm, stratum)]
        else:
            raise ValueError(f"No natural tasks for stratum {arm} {stratum}.")
    for (arm, _, kind), n in counts.items():
        if kind in RULE["untimed_seconds"]:
            totals[arm] += n * RULE["untimed_seconds"][kind]
    return {"totals_seconds": totals, "saving": 1 - totals["N2"] / totals["W1"]}


def w1_favourable_overrides(table, counts, *, cap=RULE["cap_seconds"]):
    natural = table[~table.injected & ~table.practice & (table.arm == "W1")].assign(seconds=lambda t: capped(t, cap))
    out = {}
    for (arm, stratum, kind) in counts:
        if arm == "W1" and stratum in RULE["w1_favourable_strata"] and kind == "further_comparison":
            cell = natural[(natural.stratum == stratum) & (natural.kind == kind)].seconds
            if len(cell):
                out[(arm, stratum, kind)] = float(cell.quantile(RULE["w1_favourable_quantile"]))
    return out


# --- Error rates, agreement and the rule ----------------------------------------------------

def _rate(errors, n):
    estimate, low, high = wilson_interval(int(errors), int(n))
    return estimate, low, high


def error_rates(table, by=("arm", "kind")):
    """Per group and truth class: error rate over decided answers (unsure excluded), unsure rate, and the
    sensitivity rate counting unsure as an error. Wilson 95% intervals."""
    rows = []
    main = table[~table.practice]
    for key, group in main.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        for truth, wrong_answer, label in ((True, "different", "false_different"), (False, "same", "false_same")):
            g = group[group.truth == truth]
            n, unsure = len(g), int((g.answer == "unsure").sum())
            wrong = int((g.answer == wrong_answer).sum())
            rate, low, high = _rate(wrong, n - unsure)
            u_rate, u_low, u_high = _rate(unsure, n)
            s_rate, s_low, s_high = _rate(wrong + unsure, n)
            rows.append({**dict(zip(by, key)), "truth": "same" if truth else "different", "error": label,
                         "decisions": n, "errors": wrong, "unsure": unsure, "rate": rate, "low": low, "high": high,
                         "unsure_rate": u_rate, "unsure_low": u_low, "unsure_high": u_high,
                         "rate_unsure_as_error": s_rate, "rate_unsure_as_error_low": s_low,
                         "rate_unsure_as_error_high": s_high})
    return pd.DataFrame(rows)


def cluster_bootstrap_rates(table, *, resamples=RULE["bootstrap_resamples"], seed=RULE["bootstrap_seed"]):
    """Task-cluster bootstrap 95% intervals for each (arm, kind, truth) error rate (unsure excluded)."""
    rng = np.random.default_rng(seed)
    rows = []
    main = table[~table.practice & (table.answer != "unsure")]
    for (arm, kind, truth), g in main.groupby(["arm", "kind", "truth"]):
        wrong = g.answer == ("different" if truth else "same")
        tasks = sorted(g.task_id.unique())
        index = g.task_id.map({t: i for i, t in enumerate(tasks)})
        errors = np.bincount(index, weights=wrong.astype(float), minlength=len(tasks))
        n = np.bincount(index, minlength=len(tasks)).astype(float)
        draw = rng.integers(0, len(tasks), size=(resamples, len(tasks)))
        e, c = errors[draw].sum(axis=1), n[draw].sum(axis=1)
        rates = e / np.where(c > 0, c, 1)
        rows.append({"arm": arm, "kind": kind, "truth": "same" if truth else "different", "tasks": len(tasks),
                     "rate": float(wrong.mean()), "cluster_low": float(np.quantile(rates, .025)),
                     "cluster_high": float(np.quantile(rates, .975))})
    return pd.DataFrame(rows)


def agreement(table):
    """Pairwise Cohen's kappa over {same, different, unsure} on steps answered by both reviewers, the
    two-person consensus false-same rate, and who matched the reference when they disagreed."""
    main = table[~table.practice]
    reviewers = sorted(main.reviewer.unique())
    out = []
    labels = ("same", "different", "unsure")
    for i, a in enumerate(reviewers):
        for b in reviewers[i + 1:]:
            x = main[main.reviewer == a].set_index(["task_id", "step"])
            y = main[main.reviewer == b].set_index(["task_id", "step"])
            both = x.join(y[["answer"]], rsuffix="_b", how="inner")
            if both.empty:
                continue
            agree = float((both.answer == both.answer_b).mean())
            pa = np.asarray([(both.answer == lab).mean() for lab in labels])
            pb = np.asarray([(both.answer_b == lab).mean() for lab in labels])
            expected = float(pa @ pb)
            kappa = (agree - expected) / (1 - expected) if expected < 1 else None
            different = both[~both.truth]
            consensus_same = int(((different.answer == "same") & (different.answer_b == "same")).sum())
            disagree = both[both.answer != both.answer_b]
            right = lambda col: ((disagree[col] == "same") == disagree.truth) & (disagree[col] != "unsure")
            out.append({"reviewers": [a, b], "shared_decisions": int(len(both)), "agreement": agree, "kappa": kappa,
                        "consensus_false_same": _rate(consensus_same, len(different)),
                        "disagreements": int(len(disagree)), "first_matched_reference": int(right("answer").sum()),
                        "second_matched_reference": int(right("answer_b").sum())})
    return out


def second_check_reliability(table):
    """Second-reviewer kinds vs N2 first-pass decisions, and the rescue rate on injected second checks."""
    main = table[~table.practice & (table.arm == "N2")]
    groups = {"first_pass": main[main.kind.isin(FIRST_PASS)],
              "second_role": main[main.kind.isin(["second_comparison", "disagreement_check"])]}
    out = {}
    for name, g in groups.items():
        for truth, wrong, label in ((True, "different", "false_different"), (False, "same", "false_same")):
            h = g[(g.truth == truth) & (g.answer != "unsure")]
            out[f"{name}_{label}"] = _rate(int((h.answer == wrong).sum()), len(h))
    injected = main[(main.type == "second_check") & main.injected & main.truth]
    out["injected_rescue"] = _rate(int((injected.answer == "same").sum()), len(injected))
    return out


def unidentifiable_rates(table):
    tasks = table[~table.practice].drop_duplicates(["reviewer", "task_id"])
    return {f"{arm}|{kind}": _rate(int(g.unidentifiable.sum()), len(g)) for (arm, kind), g in
            tasks.groupby(["arm", "type"])}


def measured_rates(table):
    """Re-simulation rates per (arm, kind): P(Same | different) and P(Different or Unsure | same), with
    fallbacks (kind over arms, then all first-pass decisions, then 0). Returns {(arm, kind): dict}."""
    main = table[~table.practice]

    def rates(g):
        same, diff = g[g.truth], g[~g.truth]
        fs = float((diff.answer == "same").mean()) if len(diff) else None
        fd = float((same.answer != "same").mean()) if len(same) else None
        return fs, fd

    first = rates(main[main.kind.isin(FIRST_PASS)])
    out = {}
    for arm in ARMS:
        for kind in KINDS:
            source_kind = "pair_question" if kind in ("merge_confirmation", "count_audit_comparison") else kind
            levels = [("arm_kind", main[(main.arm == arm) & (main.kind == source_kind)]),
                      ("kind", main[main.kind == source_kind])]
            fs = fd = None
            used = []
            for name, g in levels:
                a, b = rates(g)
                if fs is None and a is not None:
                    fs = a
                    used.append(f"false_same:{name}")
                if fd is None and b is not None:
                    fd = b
                    used.append(f"false_different:{name}")
            if fs is None:
                fs = first[0] if first[0] is not None else 0.0
                used.append("false_same:first_pass" if first[0] is not None else "false_same:zero")
            if fd is None:
                fd = first[1] if first[1] is not None else 0.0
                used.append("false_different:first_pass" if first[1] is not None else "false_different:zero")
            out[(arm, kind)] = {"false_same": fs, "false_different": fd, "sources": used}
    return out


class MeasuredReviewer:
    """Answers 'same animal?' with separate false-same and false-different (incl. unsure) rates."""

    def __init__(self, false_same=0.0, false_different=0.0, seed=42):
        if not (0 <= false_same <= 1 and 0 <= false_different <= 1):
            raise ValueError("Measured rates must lie in [0, 1].")
        self.false_same, self.false_different = float(false_same), float(false_different)
        self.rng = np.random.default_rng(seed)

    def same(self, truth):
        p = self.false_different if truth else self.false_same
        flip = bool(p) and self.rng.random() < p
        return bool(truth) != flip


def decision_rule(saving, ci, per_reviewer, errors_reported, *, reduced=False, reviewers=2):
    """The pre-registered verdict (see RULE)."""
    labels = []
    if saving >= RULE["success_saving"] and errors_reported:
        verdict = "SUCCESS"
        if ci[0] < RULE["success_saving"]:
            labels.append("imprecise")
        if any(v < RULE["success_saving"] for v in per_reviewer.values()):
            labels.append("reviewer-dependent")
    elif saving >= RULE["success_saving"]:
        verdict = "INCOMPLETE"  # The time threshold is met but some sampled kind has no error rate.
    elif saving >= RULE["reframe_saving"]:
        verdict = "PARTIAL"
    else:
        verdict = "REFRAME"
    if reduced:
        labels.append("reduced")
    if reviewers == 1:
        labels.append("single reviewer")
    return {"verdict": verdict, "labels": labels, "saving": saving, "ci95": list(ci),
            "per_reviewer": per_reviewer, "errors_reported": bool(errors_reported),
            "thresholds": {"success": RULE["success_saving"], "reframe": RULE["reframe_saving"]}}


def errors_reported(rates, table):
    """Every sampled (arm, kind) has at least one truth class with a computable rate and interval."""
    main = table[~table.practice]
    sampled = set(map(tuple, main[["arm", "kind"]].drop_duplicates().to_numpy().tolist()))
    have = set(map(tuple, rates[rates.rate.notna()][["arm", "kind"]].drop_duplicates().to_numpy().tolist()))
    return sampled <= have


def completion_status(orders, slot, completed):
    """'complete' (all blocks), 'reduced' (exactly the first early-stop blocks) or 'incomplete'."""
    blocks = orders[slot]["blocks"]
    done = set(completed)
    every = {t for b in blocks for t in b["tasks"]}
    first = {t for b in blocks[:RULE["early_stop_blocks"]] for t in b["tasks"]}
    main_done = done & every
    if main_done == every:
        return "complete"
    if main_done == first:
        return "reduced"
    return "incomplete"
