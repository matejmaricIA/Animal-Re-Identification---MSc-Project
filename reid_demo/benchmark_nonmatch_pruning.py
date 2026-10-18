"""Calibrated non-match pruning with review safeguards (work item (a), 18 October 2026).

Extends confidence-gated matching (W4). For each part (one animal's photos in a camera-day bundle)
an identity-cross-fitted P(match) model scores every catalogue animal from label-free features. The
expert's ranked scan stops once no unseen candidate has P(match) >= epsilon, so declaring a new
animal needs only a short list. Candidates can also be shown in P(match) order. Safeguards: a
cold-start guard, double confirmation of new animals, a model-expert disagreement check, periodic
duplicate checks, and second-reviewer audits of automation and of the count. Automatic mistakes stay
in the catalogue. New animals are never created automatically. Calibration development evidence only.
"""
from __future__ import annotations

from collections import Counter
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reid_demo.benchmark import file_digest, write_json
from reid_demo.benchmark_clustering import ROOT
from reid_demo.benchmark_cluster_diagnostics import _write_table
from reid_demo.benchmark_review_simulation import Catalogue, fast_quality, plain
from reid_demo import benchmark_gated_matching as gm


VERSION = "calibration_nonmatch_pruning_v1"
EPSILONS = (0.0, 0.001, 0.003, 0.01, 0.03)
THRESHOLDS = (None, 0.98, 0.99, 0.995)
ORDERS = ("cosine", "pmatch")
REVIEWER_ERRORS = (0.0, 0.03)
AUTO_AUDIT = 100
COUNT_AUDITS = (10, 20)
DEFAULTS = {"guard_n": 10, "guard_b": 100, "double_c": 3, "disagree_delta": 0.05, "dup_every": 100, "dup_r": 3,
            "spacetime": True}
OFF = {"guard_n": 0, "guard_b": 0, "double_c": 0, "disagree_delta": None, "dup_every": 0, "dup_r": 0}
ABLATION_TAU = 0.99
ABLATION_EPSILONS = (0.003, 0.01)
ABLATIONS = {
    "guard_off": {"guard_n": 0, "guard_b": 0},
    "guard_20_200": {"guard_n": 20, "guard_b": 200},
    "no_double_confirmation": {"double_c": 0},
    "no_disagreement_check": {"disagree_delta": None},
    "no_duplicate_checks": {"dup_every": 0, "dup_r": 0},
    "all_safeguards_off": dict(OFF),
    "appearance_only_guard_on": {"spacetime": False},
    "appearance_only_guard_off": {"spacetime": False, "guard_n": 0, "guard_b": 0},
}
KINDS = ("rank1_comparison", "further_comparison", "second_comparison", "disagreement_check", "pair_question",
         "audit_glance", "count_audit_comparison", "extra_animal")
COST_SETTINGS = {
    "placeholder": {"rank1_comparison": 8.0, "further_comparison": 8.0, "second_comparison": 8.0,
                    "disagreement_check": 8.0, "pair_question": 10.0, "audit_glance": 5.0,
                    "count_audit_comparison": 8.0, "extra_animal": 5.0},
    "fast_click": {"rank1_comparison": 2.0, "further_comparison": 5.0, "second_comparison": 5.0,
                   "disagreement_check": 5.0, "pair_question": 6.0, "audit_glance": 2.0,
                   "count_audit_comparison": 5.0, "extra_animal": 3.0},
}
# Saved perfect-reviewer outcomes the safeguard-free cosine references must reproduce exactly:
# W1 from review_effort_v2_fp32_r2 and the post-hoc full-scan W4 rows (100-glance audit included).
EXPECTED_REFERENCES = {None: {"decisions": 5283, "automation": 0.0},
                       0.98: {"decisions": 4279, "automation": 0.609},
                       0.99: {"decisions": 4449, "automation": 0.532},
                       0.995: {"decisions": 4578, "automation": 0.457}}
PAIR_APPEARANCE = ("s_e", "log_rank", "gap_top", "gap_next", "top3_mean", "log_reciprocal_rank", "log_entry_size",
                   "s_same_facing", "facing_agreement", "risk_mean", "risk_min", "log_part_size", "log_catalogue")
PAIR_SPACETIME = ("seen_same_camera", "seen_same_cell", "log_km", "km_missing", "log_days_since_seen",
                  "days_missing", "log_same_day_km", "same_day_missing")
PAIR_COLUMNS = PAIR_APPEARANCE + PAIR_SPACETIME
NOTES = [
    "Simulated reviewers answer from reference identities; both cost settings are placeholders, not measured times.",
    "Every reviewer role (expert, second confirmer, auditors) has its own seeded error generator.",
    "Automation acts on whole camera-day bundles; new animals are only ever created by an expert declaration.",
    "P(match) and confidence models are identity-cross-fitted on calibration (development only); a deployable "
    "model needs fitting-pool training. They are trained on a perfect-review replay and applied to an imperfect catalogue.",
    "Hours exclude the optional count audit unless stated; raw decision counts by kind are reported for re-costing.",
    "Space-time features use day-level dates and cell/camera coordinates where available (about 88% of photos).",
    "Calibration development evidence only; no selection evaluation, count configuration or outreach.",
]


def implementation_binding():
    names = ("reid_demo/benchmark_nonmatch_pruning.py", "tools/simulate_czechlynx_nonmatch_pruning.py",
             "reid_demo/benchmark_gated_matching.py", "tools/simulate_czechlynx_gated_matching.py",
             "reid_demo/benchmark_review_simulation.py")
    return {"version": VERSION, "code": {n: file_digest(ROOT / n) for n in names}, "epsilons": list(EPSILONS),
            "thresholds": list(THRESHOLDS), "orders": list(ORDERS), "reviewer_errors": list(REVIEWER_ERRORS),
            "auto_audit": AUTO_AUDIT, "count_audits": list(COUNT_AUDITS), "defaults": DEFAULTS,
            "ablation_tau": ABLATION_TAU, "ablation_epsilons": list(ABLATION_EPSILONS), "ablations": ABLATIONS,
            "cost_settings": COST_SETTINGS, "pair_columns": list(PAIR_COLUMNS),
            "expected_references": {str(k): v for k, v in EXPECTED_REFERENCES.items()}}


# --- Ledger ---------------------------------------------------------------------------------

class Ledger:
    """Decision counts by kind, costed under any declared setting."""
    ALIASES = {"glance": "audit_glance", "catalogue_comparison": "further_comparison"}

    def __init__(self):
        self.counts = Counter()

    def add(self, kind, n=1):
        kind = self.ALIASES.get(kind, kind)
        if kind not in KINDS:
            raise ValueError(f"Unknown decision kind: {kind}")
        self.counts[kind] += n

    def copy(self):
        clone = Ledger()
        clone.counts = Counter(self.counts)
        return clone

    def decisions(self, count_audit=False):
        return int(sum(v for k, v in self.counts.items() if count_audit or k != "count_audit_comparison"))

    def seconds(self, setting="placeholder", count_audit=False):
        costs = COST_SETTINGS[setting]
        return float(sum(costs[k] * v for k, v in self.counts.items() if count_audit or k != "count_audit_comparison"))

    def as_dict(self):
        return {k: int(self.counts.get(k, 0)) for k in KINDS}


# --- Catalogue with merges ------------------------------------------------------------------

class LiveCatalogue(Catalogue):
    """Catalogue whose entries can be merged. Entries stay compact and in creation order; uids are stable."""

    def __init__(self, n_images):
        super().__init__(n_images)
        self.uid = []
        self._next = 0

    def add(self, images, entry=None):
        if entry is None:
            self.uid.append(self._next)
            self._next += 1
        return super().add(images, entry)

    def create(self, images, founder):
        entry = self.add(images)
        self.founder[entry] = founder
        return entry

    def merge(self, keep, drop):
        """Move `drop`'s photos into the older entry `keep`; `drop` disappears from ranking."""
        if not keep < drop:
            raise ValueError("Merges keep the older (lower) entry.")
        self.entry_of[self.entry_of == drop] = keep
        self.entry_of[self.entry_of > drop] -= 1
        self.founder.pop(drop)
        self.uid.pop(drop)

    def members(self, entry):
        return np.flatnonzero(self.entry_of == entry)


# --- Label-free candidate features ----------------------------------------------------------

def _codes(values):
    mapping = {}
    return np.asarray([-1 if v is None else mapping.setdefault(v, len(mapping)) for v in values], int)


class PairContext:
    """Wraps the W4 Context with integer camera/cell codes for vectorised per-animal features."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.camera = _codes(ctx.camera)
        self.cell = _codes(ctx.cell)


def _seen(codes_part, codes_members, local, m):
    wanted = np.setdiff1d(np.unique(codes_part), [-1])
    if not len(wanted):
        return np.zeros(m)
    return (np.bincount(local, weights=np.isin(codes_members, wanted).astype(float), minlength=m) > 0).astype(float)


def candidate_features(images, catalogue, pctx, exclude=None):
    """Features of a part against every catalogue animal (except `exclude`). No labels are read.

    Returns (X with PAIR_COLUMNS, candidate entry indices, the same entries in cosine order)."""
    ctx = pctx.ctx
    images = np.asarray(images, int)
    assigned = np.asarray(catalogue.assigned, int)
    entries = catalogue.entry_of[assigned]
    if exclude is not None:
        keep = entries != exclude
        assigned, entries = assigned[keep], entries[keep]
    ids = np.unique(entries)
    m = len(ids)
    if not m:
        return np.empty((0, len(PAIR_COLUMNS))), ids, []
    local = np.searchsorted(ids, entries)
    colmax = ctx.scores[np.ix_(images, assigned)].max(axis=0)
    best = np.full(m, -np.inf)
    np.maximum.at(best, local, colmax)
    order = np.lexsort((ids, -best))  # Same order as Catalogue.ranked: (-best, entry).
    rank = np.empty(m)
    rank[order] = np.arange(m)
    ranked_best = best[order]
    gap_next = np.empty(m)
    gap_next[order] = np.append(ranked_best[:-1] - ranked_best[1:], 1.0)
    within = np.lexsort((-colmax, local))
    sorted_local = local[within]
    starts = np.searchsorted(sorted_local, np.arange(m))
    position = np.arange(len(within)) - starts[sorted_local]
    top = position < 3
    top3 = (np.bincount(sorted_local[top], weights=colmax[within][top], minlength=m)
             / np.bincount(sorted_local[top], minlength=m))
    size = np.bincount(local, minlength=m)
    best_member = assigned[within[starts]]
    others = local[None, :] != np.arange(m)[:, None]
    reciprocal = ((ctx.scores[np.ix_(best_member, assigned)] > best[:, None]) & others).sum(axis=1)
    facing = int(round(ctx.facing[images].mean()))
    same = ctx.facing[assigned] == facing
    s_same = np.full(m, -np.inf)
    np.maximum.at(s_same, local[same], colmax[same])
    s_same[~np.isfinite(s_same)] = -1.0
    agreement = np.bincount(local, weights=same.astype(float), minlength=m) / size

    lat = np.nanmean(ctx.lat[images]) if np.isfinite(ctx.lat[images]).any() else np.nan
    lon = np.nanmean(ctx.lon[images]) if np.isfinite(ctx.lon[images]).any() else np.nan
    if np.isfinite(lat):
        dist = gm.haversine_km(lat, lon, ctx.lat[assigned], ctx.lon[assigned])
        dist = np.where(np.isfinite(dist), dist, np.inf)
    else:
        dist = np.full(len(assigned), np.inf)
    km = np.full(m, np.inf)
    np.minimum.at(km, local, dist)
    days_all = ctx.day[assigned]
    day = np.nanmin(ctx.day[images]) if np.isfinite(ctx.day[images]).any() else np.nan
    last = np.full(m, -np.inf)
    np.maximum.at(last, local, np.where(np.isfinite(days_all), days_all, -np.inf))
    since = day - last
    same_day_km = np.full(m, np.inf)
    if np.isfinite(day):
        on_day = days_all == day
        np.minimum.at(same_day_km, local[on_day], dist[on_day])
    km_ok, since_ok, day_ok = np.isfinite(km), np.isfinite(since), np.isfinite(same_day_km)
    columns = {
        "s_e": best, "log_rank": np.log1p(rank), "gap_top": ranked_best[0] - best, "gap_next": gap_next,
        "top3_mean": top3, "log_reciprocal_rank": np.log1p(reciprocal), "log_entry_size": np.log1p(size),
        "s_same_facing": s_same, "facing_agreement": agreement,
        "risk_mean": np.full(m, float(ctx.risk[images].mean())), "risk_min": np.full(m, float(ctx.risk[images].min())),
        "log_part_size": np.full(m, float(np.log1p(len(images)))), "log_catalogue": np.full(m, float(np.log1p(m))),
        "seen_same_camera": _seen(pctx.camera[images], pctx.camera[assigned], local, m),
        "seen_same_cell": _seen(pctx.cell[images], pctx.cell[assigned], local, m),
        "log_km": np.where(km_ok, np.log1p(np.where(km_ok, km, 0)), 0.0), "km_missing": (~km_ok).astype(float),
        "log_days_since_seen": np.where(since_ok, np.log1p(np.clip(np.where(since_ok, since, 0), 0, None)), 0.0),
        "days_missing": (~since_ok).astype(float),
        "log_same_day_km": np.where(day_ok, np.log1p(np.where(day_ok, same_day_km, 0)), 0.0),
        "same_day_missing": (~day_ok).astype(float),
    }
    x = np.column_stack([columns[c] for c in PAIR_COLUMNS]).astype(float)
    return x, ids, [int(e) for e in ids[order]]


# --- P(match) model -------------------------------------------------------------------------

def replay_pair_records(bundles, codes, pctx):
    """Perfect-review W1 replay: every part against every catalogue animal at that moment, labelled."""
    catalogue, blocks, meta = LiveCatalogue(len(codes)), [], []
    entry_of_identity = {}
    part_index = 0
    for b, images in enumerate(bundles):
        for identity, part in gm._parts(images, codes):
            if catalogue.founder:
                x, ids, _ = candidate_features(part, catalogue, pctx)
                founders = np.asarray([catalogue.founder[e] for e in ids], int)
                blocks.append(x)
                meta.append(pd.DataFrame({"match": (founders == identity).astype(int), "part_identity": int(identity),
                                          "candidate_identity": founders, "part": part_index, "bundle": b,
                                          "known": int(identity in entry_of_identity)}))
            if identity in entry_of_identity:
                catalogue.add(part, entry_of_identity[identity])
            else:
                entry_of_identity[identity] = catalogue.create(part, identity)
            part_index += 1
    if not blocks:
        return pd.DataFrame(columns=list(PAIR_COLUMNS) + ["match", "part_identity", "candidate_identity", "part"])
    return pd.concat([pd.DataFrame(np.vstack(blocks), columns=PAIR_COLUMNS), pd.concat(meta, ignore_index=True)],
                     axis=1)


def fit_pair_models(records, identity_fold, columns, *, c=1.0):
    """One model per identity fold, trained only on records whose part AND candidate animals avoid that fold."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    part_fold = records.part_identity.map(identity_fold).to_numpy()
    candidate_fold = records.candidate_identity.map(identity_fold).to_numpy()
    models = {}
    for fold in sorted(set(identity_fold.values())):
        train = (part_fold != fold) & (candidate_fold != fold)
        y = records.match.to_numpy()[train]
        if len(np.unique(y)) < 2:
            raise ValueError("A P(match) fold lacks both outcomes.")
        models[fold] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=c)).fit(
            records.loc[train, list(columns)].to_numpy(float), y)
    return models


def scorer(model):
    """Fast P(match) for a scaler + logistic pipeline (identical maths); other models use predict_proba."""
    steps = getattr(model, "steps", None)
    if steps and hasattr(steps[0][1], "scale_") and hasattr(steps[-1][1], "coef_"):
        scale, lr = steps[0][1], steps[-1][1]
        w = lr.coef_[0] / scale.scale_
        b = float(lr.intercept_[0] - np.dot(scale.mean_, w))
        return lambda x: 1.0 / (1.0 + np.exp(-(np.asarray(x, float) @ w + b)))
    return lambda x: model.predict_proba(np.asarray(x, float))[:, 1]


def cross_fitted_pmatch(records, models, identity_fold, columns):
    out = np.full(len(records), np.nan)
    folds = records.part_identity.map(identity_fold).to_numpy()
    for fold, model in models.items():
        m = folds == fold
        if m.any():
            out[m] = model.predict_proba(records.loc[m, list(columns)].to_numpy(float))[:, 1]
    return out


def evaluate_pair_model(records, p, epsilons=EPSILONS):
    """AUC, Brier, reliability and per-epsilon miss / pruning rates on cross-fitted replay records."""
    from sklearn.metrics import roc_auc_score
    y = records.match.to_numpy().astype(bool)
    row = {"records": int(len(records)), "matches": int(y.sum()),
           "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
           "brier": float(np.mean((p - y) ** 2))}
    bins = np.clip((p * 10).astype(int), 0, 9)
    row["reliability"] = [{"bin": f"{b / 10:.1f}-{(b + 1) / 10:.1f}", "n": int((bins == b).sum()),
                           "mean_p": float(p[bins == b].mean()) if (bins == b).any() else None,
                           "observed": float(y[bins == b].mean()) if (bins == b).any() else None} for b in range(10)]
    new_parts = records.known.to_numpy() == 0
    frame = pd.DataFrame({"part": records.part.to_numpy(), "p": p, "y": y, "cos_rank": records.log_rank.to_numpy()})
    for eps in epsilons:
        if eps <= 0:
            continue
        shortlist = frame[new_parts].assign(keep=lambda f: f.p >= eps).groupby("part").keep.sum()
        row[f"miss_rate_eps_{eps}"] = float((p[y] < eps).mean()) if y.any() else None
        row[f"nonmatch_pruned_eps_{eps}"] = float((p[~y] < eps).mean()) if (~y).any() else None
        row[f"new_part_shortlist_mean_eps_{eps}"] = float(shortlist.mean()) if len(shortlist) else None
        row[f"new_part_shortlist_max_eps_{eps}"] = int(shortlist.max()) if len(shortlist) else None
    # Indicative replay rank of the true animal: cosine order vs P(match) order (perfect catalogue).
    known = frame[records.known.to_numpy() == 1].copy()
    known["p_rank"] = known.groupby("part").p.rank(ascending=False, method="first")
    known["c_rank"] = known.groupby("part").cos_rank.rank(method="first")
    true = known[known.y]
    row["replay_known_parts"] = int(len(true))
    row["replay_true_rank1_cosine"] = float((true.c_rank == 1).mean()) if len(true) else None
    row["replay_true_rank1_pmatch"] = float((true.p_rank == 1).mean()) if len(true) else None
    row["replay_comparisons_cosine"] = int(true.c_rank.sum()) if len(true) else None
    row["replay_comparisons_pmatch"] = int(true.p_rank.sum()) if len(true) else None
    return row


# --- Sequential simulation ------------------------------------------------------------------

class PrunedSimulation:
    """Sequential W4 + pruned expert scan + safeguards. Truth is read only by reviewers, to pick the
    cross-fitted model of the animal being scored, and for evaluation logs."""

    def __init__(self, bundles, codes, pctx, conf_models, conf_columns, pair_scorers, pair_columns, identity_fold,
                 setting, expert, second):
        self.bundles, self.codes, self.pctx, self.ctx = bundles, np.asarray(codes), pctx, pctx.ctx
        self.conf_models, self.conf_columns = conf_models, list(conf_columns)
        self.pair_scorers = pair_scorers
        self.pair_index = [PAIR_COLUMNS.index(c) for c in pair_columns]
        self.fold, self.s = identity_fold, setting
        self.expert, self.second = expert, second
        self.catalogue, self.ledger = LiveCatalogue(len(codes)), Ledger()
        self.auto_log, self.declarations, self.placements, self.duplicate_log = [], [], [], []
        self.expert_bundles, self.pending, self.asked = 0, [], set()
        self.bundle = -1

    def guard_open(self):
        return len(self.catalogue.founder) >= self.s["guard_n"] and self.expert_bundles >= self.s["guard_b"]

    def run(self):
        tau = self.s["tau"]
        for b, images in enumerate(self.bundles):
            self.bundle = b
            if self.catalogue.founder and tau is not None and self.guard_open():
                animal = gm.majority(images, self.codes)
                features, order = gm.match_features(images, self.catalogue, self.ctx)
                model = self.conf_models[self.fold[animal]]
                confidence = float(model.predict_proba(
                    np.asarray([[features[c] for c in self.conf_columns]], float))[:, 1][0])
                if confidence >= tau:
                    entry = order[0]
                    self.catalogue.add(images, entry)
                    self.auto_log.append({"images": list(images), "bundle": b, "confidence": confidence,
                                          "correct": bool(all(self.codes[i] == self.catalogue.founder[entry]
                                                              for i in images))})
                    self._maybe_check(b)
                    continue
            self.expert_bundles += 1
            parts = gm._parts(images, self.codes)
            self.ledger.add("extra_animal", len(parts) - 1)
            for identity, part in parts:
                self.place(identity, part)
            self._maybe_check(b)
        if self.s["dup_every"]:
            self.duplicate_check()
        return self

    def _maybe_check(self, b):
        if self.s["dup_every"] and (b + 1) % self.s["dup_every"] == 0:
            self.duplicate_check()

    def _pmatch(self, images, identity, exclude=None):
        x, ids, cosine = candidate_features(images, self.catalogue, self.pctx, exclude)
        p = self.pair_scorers[self.fold[identity]](x[:, self.pair_index]) if len(ids) else np.empty(0)
        return dict(zip(ids.tolist(), np.asarray(p, float).tolist())), cosine

    def place(self, identity, part):
        cat, s = self.catalogue, self.s
        true_entries = {e for e, f in enumerate(cat.founder) if f == identity}  # Evaluation log only.
        record = {"bundle": self.bundle, "identity": int(identity), "photos": len(part),
                  "true_in_catalogue": bool(true_entries), "guard_open": self.guard_open(), "shown": 0,
                  "disagreement_checks": 0, "second_comparisons": 0, "first_correct": None}
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
        for pos, e in enumerate(order):
            if pruning and shown and suffix[pos] < s["eps"]:
                break
            self.ledger.add("rank1_comparison" if not shown else "further_comparison")
            shown.add(e)
            truth = cat.founder[e] == identity
            if self.expert.same(truth):
                if s["disagree_delta"] is not None and p[e] < s["disagree_delta"]:
                    self.ledger.add("disagreement_check")
                    record["disagreement_checks"] += 1
                    if not self.second.same(truth):
                        continue
                cat.add(part, e)
                self._log(record, shown, "matched_correct" if truth else "matched_wrong")
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

    def _log(self, record, shown, outcome):
        record.update(shown=len(shown), outcome=outcome)
        self.placements.append(record)

    def _declare(self, identity, part, record, cut_true):
        entry = self.catalogue.create(part, identity)
        uid = self.catalogue.uid[entry]
        duplicate = record["true_in_catalogue"]
        record.update(outcome="new_pruning_duplicate" if cut_true else "new_reviewer_duplicate" if duplicate else "new",
                      uid=uid)
        self.placements.append(record)
        self.declarations.append(record)
        self.pending.append(uid)

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
                    self.duplicate_log.append({"bundle": self.bundle, "kept_uid": cat.uid[min(a, b)],
                                               "merged_uid": cat.uid[max(a, b)], "correct": bool(truth)})
                    cat.merge(min(a, b), max(a, b))
                    break


# --- Audits and metrics ---------------------------------------------------------------------

def count_audit(catalogue, n, auditor, ledger, scores, codes, seed=42):
    """A second reviewer checks sampled animals against all OLDER animals (ranked, stop at yes).
    Estimates the duplicate share and a count interval; corrects nothing."""
    entries = len(catalogue.founder)
    true_animals = int(len(np.unique(np.asarray(codes)[catalogue.entry_of >= 0])))
    seen, duplicate = set(), []
    for f in catalogue.founder:
        duplicate.append(f in seen)
        seen.add(f)
    truth = float(np.mean(duplicate)) if entries else None
    if n <= 0 or not entries:
        return {"audited": 0, "estimate": None, "low": None, "high": None, "true_duplicate_share": truth,
                "covered": None, "count_low": None, "count_high": None, "count_covered": None}
    rng = np.random.default_rng(seed)
    sample = rng.choice(entries, min(n, entries), replace=False)
    found = 0
    for x in sample:
        older = [e for e in catalogue.ranked(scores, catalogue.members(int(x))) if e < x]
        for e in older:
            ledger.add("count_audit_comparison")
            if auditor.same(catalogue.founder[e] == catalogue.founder[x]):
                found += 1
                break
    estimate, low, high = gm.wilson_interval(found, len(sample))
    count_low, count_high = entries * (1 - high), entries * (1 - low)
    return {"audited": int(len(sample)), "errors_found": found, "estimate": estimate, "low": low, "high": high,
            "true_duplicate_share": truth, "covered": bool(low <= truth <= high), "count_low": count_low,
            "count_high": count_high, "count_covered": bool(count_low <= true_animals <= count_high)}


def summarize(sim, codes, n_images, n_bundles, audits):
    cat, ledger = sim.catalogue, audits["ledger"]
    labels = cat.entry_of.copy()
    wrong, rate = gm.misassigned_photos(labels, cat, codes)
    q = fast_quality(labels, codes)
    per = n_images / 1000
    founders_seen, still_duplicate = set(), set()
    for f, uid in zip(cat.founder, cat.uid):
        if f in founders_seen:
            still_duplicate.add(uid)
        founders_seen.add(f)
    declared = sim.declarations
    known = [r for r in sim.placements if r["true_in_catalogue"]]
    comparisons = lambda r: r["shown"] + r["disagreement_checks"] + r["second_comparisons"]
    row = {"bundles": n_bundles, "auto_bundles": len(sim.auto_log), "automation": len(sim.auto_log) / n_bundles,
           "auto_photos": int(sum(len(r["images"]) for r in sim.auto_log)),
           "auto_wrong_bundles": int(sum(not r["correct"] for r in sim.auto_log)),
           **ledger.as_dict(), "decisions": ledger.decisions(),
           "hours": ledger.seconds("placeholder") / 3600,
           "hours_per_1000_photos": ledger.seconds("placeholder") / 3600 / per,
           "hours_per_1000_photos_fast_click": ledger.seconds("fast_click") / 3600 / per,
           "misassigned_photos": wrong, "misassigned_photo_rate": rate, **q,
           "declarations": len(declared),
           "pruning_duplicate_events": sum(r["outcome"] == "new_pruning_duplicate" for r in declared),
           "reviewer_duplicate_events": sum(r["outcome"] == "new_reviewer_duplicate" for r in declared),
           "pruning_duplicates_unresolved": sum(r["outcome"] == "new_pruning_duplicate" and r["uid"] in still_duplicate
                                                for r in declared),
           "duplicate_merges": len(sim.duplicate_log),
           "wrong_duplicate_merges": sum(not r["correct"] for r in sim.duplicate_log),
           "new_animal_comparisons": int(sum(comparisons(r) for r in declared)),
           "hard_match_comparisons": int(sum(comparisons(r) for r in known if not r["first_correct"])),
           "known_part_comparisons": int(sum(comparisons(r) for r in known)),
           "known_parts": len(known)}
    auto = audits["auto"]
    row.update({"audit_estimate": auto["estimate"], "audit_low": auto["low"], "audit_high": auto["high"],
                "true_bundle_error": auto["true_bundle_error"], "audit_covered": auto["covered"]})
    for n, result in audits["count"].items():
        extra = result["ledger"]
        row.update({f"count_audit{n}_comparisons": extra.counts["count_audit_comparison"],
                    f"count_audit{n}_estimate": result["estimate"], f"count_audit{n}_low": result["low"],
                    f"count_audit{n}_high": result["high"], f"count_audit{n}_covered": result["covered"],
                    f"count_audit{n}_count_low": result["count_low"], f"count_audit{n}_count_high": result["count_high"],
                    f"count_audit{n}_count_covered": result["count_covered"],
                    f"hours_per_1000_photos_with_count_audit{n}":
                        extra.seconds("placeholder", count_audit=True) / 3600 / per,
                    f"hours_per_1000_photos_fast_click_with_count_audit{n}":
                        extra.seconds("fast_click", count_audit=True) / 3600 / per})
        row["true_duplicate_share"] = result["true_duplicate_share"]
    return row


# --- Experiment grid ------------------------------------------------------------------------

def experiment_configs():
    """References, main grid G1 and safeguard ablations G2, as declared in the proposal."""
    configs = []
    for tau in THRESHOLDS:
        configs.append({"group": "reference", "variant": "full_scan_no_safeguards", "reviewer_error": 0.0,
                        "order": "cosine", "eps": 0.0, "tau": tau, **DEFAULTS, **OFF})
    for error in REVIEWER_ERRORS:
        for order in ORDERS:
            for eps in EPSILONS:
                for tau in THRESHOLDS:
                    configs.append({"group": "G1", "variant": "defaults", "reviewer_error": error, "order": order,
                                    "eps": eps, "tau": tau, **DEFAULTS})
    for name, change in ABLATIONS.items():
        for error in REVIEWER_ERRORS:
            for order in ORDERS:
                for eps in ABLATION_EPSILONS:
                    configs.append({"group": "G2", "variant": name, "reviewer_error": error, "order": order,
                                    "eps": eps, "tau": ABLATION_TAU, **DEFAULTS, **change})
    for c in configs:
        c["config"] = (f"{c['group']}/{c['variant']}/err{c['reviewer_error']}/{c['order']}/eps{c['eps']}/"
                       f"tau{c['tau']}")
    return configs


def check_references(rows, w1_live, expect_saved=True):
    """The safeguard-free cosine full scan must equal live W1 and, on the real data, the saved runs."""
    refs = {r["tau"]: r for r in rows if r["group"] == "reference"}
    w1 = refs[None]
    problems = []
    if w1["decisions"] != w1_live["decisions"] or not w1_live["labels_equal"]:
        problems.append(f"W1 reference differs from live W1: {w1['decisions']} vs {w1_live['decisions']} decisions, "
                        f"labels equal {w1_live['labels_equal']}.")
    if expect_saved:
        for tau, expected in EXPECTED_REFERENCES.items():
            got = refs[tau]
            if got["decisions"] != expected["decisions"] or round(got["automation"], 3) != expected["automation"] \
                    or got["count_error"] != 0:
                problems.append(f"Reference tau={tau}: {got['decisions']} decisions, automation "
                                f"{got['automation']:.3f}, count error {got['count_error']}; expected {expected}.")
    return {"w1_live": {k: v for k, v in w1_live.items()}, "expect_saved": expect_saved, "problems": problems,
            "passed": not problems}


def declared_rules(rows):
    g1 = [r for r in rows if r["group"] == "G1"]
    g2 = [r for r in rows if r["group"] == "G2"]
    perfect = [r for r in g1 if r["reviewer_error"] == 0]

    def target(r):
        return {"automation": r["automation"] >= .70, "misassigned": r["misassigned_photo_rate"] <= .01,
                "count": abs(r["count_error"]) <= 1, "hours": r["hours_per_1000_photos"] <= .5,
                "audit_covered": bool(r["audit_covered"])}
    passing = [r["config"] for r in perfect if all(target(r).values())]
    targets_met = {k: [r["config"] for r in perfect if target(r)[k]] for k in target(perfect[0])} if perfect else {}

    def find(group, **kw):
        return next((r for r in group if all(r[k] == v for k, v in kw.items())), None)
    d1 = []
    for r in perfect:
        if r["eps"] > 0:
            ref = find(perfect, order=r["order"], tau=r["tau"], eps=0.0)
            cut = 1 - r["new_animal_comparisons"] / ref["new_animal_comparisons"] if ref["new_animal_comparisons"] else None
            ok = abs(r["count_error"]) <= 1 and cut is not None and cut >= .75
            d1.append({"config": r["config"], "count_error": r["count_error"], "new_animal_cut": cut, "met": ok})
    cos = find(perfect, order="cosine", eps=0.0, tau=None)
    pm = find(perfect, order="pmatch", eps=0.0, tau=None)
    d2_cut = 1 - pm["hard_match_comparisons"] / cos["hard_match_comparisons"] if cos and pm and \
        cos["hard_match_comparisons"] else None
    d3 = []
    for r in g2:
        if r["variant"] == "all_safeguards_off" and r["reviewer_error"] > 0:
            on = find(g1, reviewer_error=r["reviewer_error"], order=r["order"], eps=r["eps"], tau=r["tau"])
            halves = on["misassigned_photo_rate"] <= .5 * r["misassigned_photo_rate"]
            cost = on["hours_per_1000_photos"] / r["hours_per_1000_photos"] - 1
            d3.append({"base": f"{r['order']}/eps{r['eps']}", "misassigned_on": on["misassigned_photo_rate"],
                       "misassigned_off": r["misassigned_photo_rate"], "count_error_on": on["count_error"],
                       "count_error_off": r["count_error"], "extra_time": cost,
                       "met": (halves or abs(on["count_error"]) <= 1) and cost <= .25})
    d4_on = [r for r in g2 if r["variant"] == "appearance_only_guard_on"]
    d4_off = [r for r in g2 if r["variant"] == "appearance_only_guard_off"]
    return {
        "candidate_operating_points": passing, "met": bool(passing),
        "targets_met_by": {k: len(v) for k, v in targets_met.items()},
        "per_target_best": {
            "max_automation_with_error_le_1pct": max((r["automation"] for r in perfect
                                                      if r["misassigned_photo_rate"] <= .01), default=None),
            "min_hours_with_count_error_le_1": min((r["hours_per_1000_photos"] for r in perfect
                                                    if abs(r["count_error"]) <= 1), default=None),
            "min_fast_click_hours_with_count_error_le_1": min((r["hours_per_1000_photos_fast_click"] for r in perfect
                                                               if abs(r["count_error"]) <= 1), default=None)},
        "D1_pruning_works": {"met": any(x["met"] for x in d1), "rows": d1},
        "D2_reranking_helps": {"met": d2_cut is not None and d2_cut >= .25, "hard_match_cut": d2_cut,
                               "cosine": cos and cos["hard_match_comparisons"],
                               "pmatch": pm and pm["hard_match_comparisons"]},
        "D3_safeguards_earn_cost": {"met_all": bool(d3) and all(x["met"] for x in d3),
                                    "met_count": sum(x["met"] for x in d3), "rows": d3},
        "D4_guard_prevents_collapse": {"met": bool(d4_on) and all(r["count_error"] >= -5 for r in d4_on),
                                       "guard_on_count_errors": [r["count_error"] for r in d4_on],
                                       "guard_off_count_errors": [r["count_error"] for r in d4_off]},
        "policy": "Mechanical checks of the 2026-10-18 non-match pruning proposal; not conclusions.",
    }


# --- Report ---------------------------------------------------------------------------------

def _fmt(value):
    return f"{value:.3f}" if isinstance(value, float) else "" if value is None else str(value)


def plot_pruning(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g1 = [r for r in rows if r["group"] == "G1" and r["reviewer_error"] == 0]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for order in ORDERS:
        for tau in THRESHOLDS:
            sel = sorted([r for r in g1 if r["order"] == order and r["tau"] == tau], key=lambda r: r["eps"])
            if not sel:
                continue
            x = [str(r["eps"]) for r in sel]
            label = f"{order}, tau={tau}"
            axes[0].plot(x, [r["count_error"] for r in sel], "o-", label=label)
            axes[1].plot(x, [r["hours_per_1000_photos"] for r in sel], "o-", label=label)
            axes[2].plot(x, [r["new_animal_comparisons"] for r in sel], "o-", label=label)
    for ax, title in zip(axes, ("Count error", "Expert h / 1,000 photos (placeholder)", "New-animal comparisons")):
        ax.set_xlabel("epsilon (0 = full scan)")
        ax.set_title(title)
        ax.grid(alpha=.2)
        ax.legend(fontsize=6)
    axes[1].axhline(.5, color="#888", ls="--", lw=1)
    fig.suptitle("Non-match pruning, perfect reviewer, safeguards on")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tradeoff(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g1 = [r for r in rows if r["group"] == "G1" and r["reviewer_error"] == 0 and r["tau"] is not None]
    fig, ax = plt.subplots(figsize=(7, 4))
    for order in ORDERS:
        for eps in EPSILONS:
            sel = sorted([r for r in g1 if r["order"] == order and r["eps"] == eps], key=lambda r: r["tau"])
            ax.plot([r["automation"] for r in sel], [100 * r["misassigned_photo_rate"] for r in sel], "o-",
                    label=f"{order}, eps={eps}")
    ax.axhline(1, color="#888", ls="--", lw=1)
    ax.axvline(.7, color="#888", ls=":", lw=1)
    ax.set_xlabel("Share of bundles decided automatically")
    ax.set_ylabel("Misassigned photos (%)")
    ax.grid(alpha=.2)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(out, results, placements):
    out = Path(out)
    results = plain(results)
    rows = results["simulations"]
    write_json(out / "summary.json", {**results, "notes": NOTES})
    _write_table(out / "simulations.csv", rows, list(rows[0]))
    model_rows = [{k: v for k, v in r.items() if k != "reliability"} for r in results["pmatch_models"]]
    _write_table(out / "pmatch_models.csv", model_rows, list(model_rows[0]))
    pd.DataFrame(placements).to_csv(out / "placements_g1.csv.gz", index=False, compression="gzip")
    plot_pruning(out / "pruning.png", rows)
    plot_tradeoff(out / "tradeoff.png", rows)
    keys = ["config", "automation", "misassigned_photo_rate", "count_error", "pruning_duplicate_events",
            "pruning_duplicates_unresolved", "duplicate_merges", "exact_identity_recoveries", "decisions",
            "new_animal_comparisons", "hard_match_comparisons", "hours_per_1000_photos",
            "hours_per_1000_photos_fast_click", "audit_estimate", "true_bundle_error", "audit_covered"]

    def table(selected):
        return "<table><tr>" + "".join(f"<th>{k}</th>" for k in keys) + "</tr>" + "".join(
            "<tr>" + "".join(f"<td>{html.escape(_fmt(r.get(k)))}</td>" for k in keys) + "</tr>" for r in selected) \
            + "</table>"
    body = ("<p>Simulated non-match pruning on calibration; reviewer answers from reference identities. "
            f"Cost settings (s): {html.escape(json.dumps(COST_SETTINGS))}.</p>"
            "<h2>Reference checks</h2><pre>" + html.escape(json.dumps(results["reference_check"], indent=1)) + "</pre>"
            "<h2>Declared targets and rules</h2><pre>" + html.escape(json.dumps(results["declared_rules"], indent=1))
            + "</pre><h2>P(match) models (cross-fitted replay records)</h2><pre>"
            + html.escape(json.dumps(results["pmatch_models"], indent=1)) + "</pre>"
            "<h2>Pruning</h2><img style='max-width:100%' src='pruning.png' alt='Pruning'>"
            "<h2>Automation trade-off</h2><img style='max-width:100%' src='tradeoff.png' alt='Trade-off'>"
            "<h2>References</h2>" + table([r for r in rows if r["group"] == "reference"]) +
            "<h2>G1, perfect reviewer</h2>" + table([r for r in rows if r["group"] == "G1" and r["reviewer_error"] == 0])
            + "<h2>G1, 3% reviewer error</h2>" + table([r for r in rows if r["group"] == "G1" and r["reviewer_error"] > 0])
            + "<h2>G2 safeguard ablations</h2>" + table([r for r in rows if r["group"] == "G2"]) +
            "<h2>Limitations</h2><ul>" + "".join(f"<li>{html.escape(n)}</li>" for n in NOTES) + "</ul>")
    (out / "index.html").write_text("<!doctype html><html lang='en'><meta charset='utf-8'><title>CzechLynx non-match "
        "pruning</title><style>body{font:14px system-ui;max-width:1600px;margin:30px auto;padding:16px}table{"
        "border-collapse:collapse}td,th{border:1px solid #bbb;padding:3px 6px}pre{white-space:pre-wrap}</style>"
        "<h1>CzechLynx calibrated non-match pruning</h1>" + body + "</html>")
    return {str(p.relative_to(out)): file_digest(p) for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "manifest.json"}
