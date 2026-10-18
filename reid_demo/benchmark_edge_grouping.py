"""Calibration-only grouping on a wider, encounter-linked graph with calibrated edge evidence.

Candidate graphs, edge features and groupings are built without identities.
Reference truth trains the edge model only through identity-cross-fitting (an
edge is scored by a model that never saw either endpoint's animal) and enters
the evaluators after grouping. Results are development evidence, not counts.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import heapq
import html
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from reid_demo.benchmark import file_digest, write_json
from reid_demo.benchmark_clustering import ROOT, ClusteringResult, _camera, _positive_int, _value, evaluate_clustering
from reid_demo.benchmark_cluster_diagnostics import _write_table
from reid_demo.benchmark_coverage_diagnostics import knn_edges, oracle_components, risk_auc


VERSION = "calibration_edge_grouping_v1"
BUDGETS = (100, 150)
ARMS = ("annotated", "derived")
RECIPROCAL_K = 20
FOLDS = 5
THRESHOLDS = (0.5, 0.7, 0.9)
SUPPORT = (1, 3)
RESOLUTIONS = (0.5, 1.0, 2.0)
AMBIGUOUS_BAND = (0.2, 0.8)
PILOT_EDGES = 20000
MIN_GATE_EDGES = 1000
GLOBAL_FEATURES = ("cosine",)
GRAPH_FEATURES = ("cosine", "log_rank_min", "log_rank_max", "mutual_top20", "shared_top20_jaccard",
                  "reciprocal_jaccard", "same_camera", "same_trap_day", "same_location_day",
                  "pair_colour_colour", "pair_mono_mono", "pair_mixed_appearance", "risk_max", "risk_min")
LOCAL_FEATURES = ("lg_log_matches", "lg_confidence_mean", "lg_log_homography_inliers",
                  "lg_homography_coverage_min", "lg_log_affine_inliers", "lg_failed")
NOTES = [
    "Calibration development evidence only; no selection or official test observations are evaluated.",
    "Edges and features are label-free; the edge model is identity-cross-fitted, never scoring its own animals.",
    "The annotated arm uses the dataset's encounter field (upper bound); the derived arm uses day-level camera/location.",
    "CzechLynx dates are day-level and EXIF times are absent; time-window encounters are untested here.",
    "G3 uses networkx Louvain because leidenalg is not installed; installing it would alter the frozen legacy control.",
    "Group counts include fragments and singletons; they are not park abundance or validated animal counts.",
]


def implementation_binding():
    names = ("reid_demo/benchmark_edge_grouping.py", "tools/evaluate_czechlynx_edge_grouping.py")
    return {"version": VERSION, "code": {name: file_digest(ROOT / name) for name in names},
            "budgets": list(BUDGETS), "arms": list(ARMS), "reciprocal_k": RECIPROCAL_K, "folds": FOLDS,
            "thresholds": list(THRESHOLDS), "support": list(SUPPORT), "resolutions": list(RESOLUTIONS),
            "ambiguous_band": list(AMBIGUOUS_BAND), "graph_features": list(GRAPH_FEATURES),
            "local_features": list(LOCAL_FEATURES)}


# --- Label-free metadata groups and candidate graphs --------------------------------------

def metadata_keys(frame):
    """Per-image grouping keys; None where metadata is missing."""
    rows = frame.to_dict("records")
    date = [_value(r.get("timestamp")) for r in rows]
    camera = [_camera(r) for r in rows]
    location = [(_value(r.get("source")), _value(r.get("location"))) for r in rows]
    return {
        "annotated": [_value(r.get("encounter_id")) for r in rows],
        "camera": camera,
        "trap_day": [f"{c}|{d}" if c and d else None for c, d in zip(camera, date)],
        "location_day": [f"{s}:{loc}|{d}" if s and loc and d else None for (s, loc), d in zip(location, date)],
    }


def group_pairs(keys, valid):
    """All unordered within-group pairs (a < b) of valid images sharing a non-missing key."""
    groups = defaultdict(list)
    for i, key in enumerate(keys):
        if key is not None and valid[i]:
            groups[key].append(i)
    return {(a, b) for members in groups.values() for a, b in combinations(sorted(members), 2)}


def candidate_graph(scores, valid, keys, arm, budget):
    """kNN(k) union the arm's encounter pairs, as sorted (a, b) index arrays plus a link flag."""
    if arm not in ARMS:
        raise ValueError(f"Unknown encounter arm: {arm}")
    knn = {(a, b) for a, b, _ in knn_edges(scores, valid, budget)}
    links = (group_pairs(keys["annotated"], valid) if arm == "annotated"
             else group_pairs(keys["trap_day"], valid) | group_pairs(keys["location_day"], valid))
    pairs = np.asarray(sorted(knn | links), dtype=int).reshape(-1, 2)
    in_knn = np.asarray([tuple(p) in knn for p in map(tuple, pairs)])
    return pairs, in_knn


def encounter_link_audit(identity_codes, valid, keys, scores, budget):
    """Oracle gain and different-animal rate for each metadata linking rule (truth used only here)."""
    base = knn_edges(scores, valid, budget)
    _, _, before = oracle_components(identity_codes, valid, base)
    rows = []
    for name in ("annotated", "trap_day", "location_day"):
        pairs = group_pairs(keys[name], valid)
        mixed = sum(identity_codes[a] != identity_codes[b] for a, b in pairs)
        _, _, after = oracle_components(identity_codes, valid, base + list(pairs))
        rows.append({"rule": name, "knn_k": budget, "linked_pairs": len(pairs),
                     "pairs_joining_different_identities": int(mixed),
                     "fraction_joining_different_identities": mixed / len(pairs) if pairs else None,
                     "groups": len({k for k in keys[name] if k is not None}),
                     "fully_connected_identities_before": before["fully_connected_identities"],
                     "fully_connected_identities_after": after["fully_connected_identities"],
                     "oracle_fragments_after": after["oracle_fragments"]})
    return rows


# --- Label-free edge features --------------------------------------------------------------

def rank_matrix(scores, valid):
    """rank[i, j] = position of j in i's descending score list (self and invalid excluded)."""
    n = len(scores)
    ranks = np.full((n, n), n, dtype=np.int32)
    for i in np.flatnonzero(valid):
        row = scores[i].copy()
        row[i] = -np.inf
        row[~valid] = -np.inf
        order = np.argsort(-row, kind="stable")
        ranks[i, order] = np.arange(n)
    return ranks


def edge_features(pairs, scores, ranks, valid, keys, appearance, risk, *, annotated=False, k=RECIPROCAL_K):
    """One row per candidate edge; nothing here reads identities."""
    a, b = pairs[:, 0], pairs[:, 1]
    n = len(scores)
    top = csr_matrix((ranks < k) & valid[:, None] & valid[None, :], dtype=np.float32)
    mutual = top.multiply(top.T)
    shared = np.asarray((top @ top.T)[a, b]).ravel()
    shared_mutual = np.asarray((mutual @ mutual.T)[a, b]).ravel()
    deg = np.asarray(mutual.sum(axis=1)).ravel()
    rank_ab, rank_ba = ranks[a, b], ranks[b, a]
    same = lambda name: np.asarray([keys[name][i] is not None and keys[name][i] == keys[name][j]
                                    for i, j in zip(a, b)], dtype=float)
    mono = np.asarray(appearance)
    features = pd.DataFrame({
        "cosine": scores[a, b],
        "log_rank_min": np.log1p(np.minimum(rank_ab, rank_ba)),
        "log_rank_max": np.log1p(np.maximum(rank_ab, rank_ba)),
        "mutual_top20": ((rank_ab < k) & (rank_ba < k)).astype(float),
        "shared_top20_jaccard": shared / np.maximum(2 * k - shared, 1),
        "reciprocal_jaccard": shared_mutual / np.maximum(deg[a] + deg[b] - shared_mutual, 1),
        "same_camera": same("camera"), "same_trap_day": same("trap_day"),
        "same_location_day": same("location_day"),
        "pair_colour_colour": ((mono[a] == 0) & (mono[b] == 0)).astype(float),
        "pair_mono_mono": ((mono[a] == 1) & (mono[b] == 1)).astype(float),
        "pair_mixed_appearance": (((mono[a] == 0) & (mono[b] == 1)) | ((mono[a] == 1) & (mono[b] == 0))).astype(float),
        "risk_max": np.maximum(risk[a], risk[b]), "risk_min": np.minimum(risk[a], risk[b]),
    })
    if annotated:
        features["same_annotated_encounter"] = same("annotated")
    if len(features) and not np.isfinite(features.to_numpy(float)).all():
        raise ValueError("Non-finite edge features.")
    if n and (a >= b).any():
        raise ValueError("Edges must be ordered a < b.")
    return features


def local_feature_frame(local, ids, pairs):
    """Map saved Stage B evidence onto edges; returns features and a coverage mask."""
    index = {(r.image_id_a, r.image_id_b): r for r in local.itertuples(index=False)}
    rows, covered = [], []
    for a, b in pairs:
        r = index.get((ids[a], ids[b]))
        covered.append(r is not None)
        if r is None:
            rows.append({name: 0.0 for name in LOCAL_FEATURES})
            continue
        failed = r.status not in ("verified",)
        confidence = 0.0 if pd.isna(r.confidence_mean) else float(r.confidence_mean)
        rows.append({"lg_log_matches": np.log1p(r.matches), "lg_confidence_mean": confidence,
                     "lg_log_homography_inliers": np.log1p(r.homography_inliers),
                     "lg_homography_coverage_min": r.homography_coverage_min,
                     "lg_log_affine_inliers": np.log1p(r.affine_inliers), "lg_failed": float(failed)})
    return pd.DataFrame(rows), np.asarray(covered)


# --- Identity-cross-fitted edge model ------------------------------------------------------

def identity_folds(identity_codes, folds=FOLDS, seed=42):
    identities = np.unique(identity_codes)
    order = np.random.default_rng(seed).permutation(identities)
    fold_of = {int(code): i % folds for i, code in enumerate(order)}
    return np.asarray([fold_of[int(c)] for c in identity_codes])


def cross_fit_probabilities(features, labels, pairs, image_fold, *, columns, seed=42):
    """Each edge is scored by a model trained only on edges whose endpoints avoid both its folds."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    x = features[list(columns)].to_numpy(float)
    fa, fb = image_fold[pairs[:, 0]], image_fold[pairs[:, 1]]
    lo, hi = np.minimum(fa, fb), np.maximum(fa, fb)
    output = np.full(len(pairs), np.nan)
    models = 0
    for f, g in sorted(set(zip(lo.tolist(), hi.tolist()))):
        target = (lo == f) & (hi == g)
        train = ~np.isin(fa, (f, g)) & ~np.isin(fb, (f, g))
        if len(np.unique(labels[train])) < 2:
            raise ValueError("A cross-fitting fold lacks both edge classes.")
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
        model.fit(x[train], labels[train])
        output[target] = model.predict_proba(x[target])[:, 1]
        models += 1
    if not np.isfinite(output).all():
        raise ValueError("Some edges were not scored.")
    return np.clip(output, 1e-6, 1 - 1e-6), models


def calibration_summary(p, labels, *, band_mask=None, bins=10):
    rows = []
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        rows.append({"bin_low": float(lo), "bin_high": float(hi), "edges": int(m.sum()),
                     "mean_predicted": float(p[m].mean()) if m.any() else None,
                     "observed_same_rate": float(labels[m].mean()) if m.any() else None})
    summary = {"edges": len(p), "positive_edges": int(labels.sum()), "auc": risk_auc(p, labels),
               "brier": float(np.mean((p - labels) ** 2)), "reliability": rows}
    if band_mask is not None:
        summary["ambiguous_band_edges"] = int(band_mask.sum())
        summary["ambiguous_band_auc"] = risk_auc(p[band_mask], labels[band_mask]) if band_mask.any() else None
    return summary


# --- Grouping (label-free given edge probabilities) ----------------------------------------

def _logit(p):
    return np.log(p / (1 - p))


def group_threshold(n, pairs, p, tau):
    """G1: union-find over edges with p >= tau."""
    parent = np.arange(n)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for (a, b), q in zip(pairs, p):
        if q >= tau:
            ra, rb = root(a), root(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    return np.asarray([root(i) for i in range(n)])


def group_average_linkage(n, pairs, p, tau, support):
    """G2: merge groups while the mean log-odds over all their candidate cross-edges is >= logit(tau)
    and at least min(support, |A|*|B|) of those edges have p >= tau. One strong bridge cannot
    outvote the rest, and two single images can still merge on their one edge."""
    _positive_int(support, "support")
    cut = _logit(tau)
    w = _logit(np.asarray(p, float))
    neighbours = defaultdict(dict)  # cluster -> {cluster: [sum, count, strong]}
    for (a, b), weight, q in zip(pairs, w, p):
        for x, y in ((a, b), (b, a)):
            entry = neighbours[x].setdefault(y, [0.0, 0, 0])
            entry[0] += weight
            entry[1] += 1
            entry[2] += int(q >= tau)
    size = defaultdict(lambda: 1)
    needed = lambda x, y: min(support, size[x] * size[y])
    heap = [(-s / c, a, b) for a, nb in neighbours.items() for b, (s, c, strong) in nb.items()
            if a < b and strong >= needed(a, b) and s / c >= cut]
    heapq.heapify(heap)
    alive = set(range(n))
    label = np.arange(n)
    members = {i: [i] for i in range(n)}
    while heap:
        negative, a, b = heapq.heappop(heap)
        if a not in alive or b not in alive or b not in neighbours[a]:
            continue
        s, c, strong = neighbours[a][b]
        if abs(-negative - s / c) > 1e-12 or strong < needed(a, b) or s / c < cut:
            continue  # Stale entry; a fresh one was pushed when the pair changed.
        # Merge b into a.
        alive.discard(b)
        size[a] += size.pop(b)
        members[a].extend(members.pop(b))
        del neighbours[a][b]
        del neighbours[b][a]
        for other, (s2, c2, st2) in neighbours.pop(b).items():
            del neighbours[other][b]
            entry = neighbours[a].setdefault(other, [0.0, 0, 0])
            entry[0] += s2
            entry[1] += c2
            entry[2] += st2
            neighbours[other][a] = entry
        for other, (s3, c3, st3) in neighbours[a].items():
            if st3 >= needed(a, other) and s3 / c3 >= cut:
                heapq.heappush(heap, (-s3 / c3, min(a, other), max(a, other)))
    for root, group in members.items():
        label[group] = root
    return label


def group_louvain(n, pairs, p, resolution, seed=42):
    """G3: weighted Louvain on positive log-odds edges."""
    import networkx as nx
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    w = _logit(np.asarray(p, float))
    graph.add_weighted_edges_from((int(a), int(b), float(x)) for (a, b), x in zip(pairs, w) if x > 0)
    label = np.arange(n)
    for i, community in enumerate(nx.community.louvain_communities(
            graph, weight="weight", resolution=resolution, seed=seed)):
        label[list(community)] = n + i
    return label


def grouping_profiles():
    profiles = [{"algorithm": "G1_threshold", "tau": t} for t in THRESHOLDS]
    profiles += [{"algorithm": "G2_average_linkage", "tau": t, "support": s} for t in THRESHOLDS for s in SUPPORT]
    profiles += [{"algorithm": "G3_louvain", "resolution": r} for r in RESOLUTIONS]
    return profiles


def run_grouping(profile, n, pairs, p, seed=42):
    if profile["algorithm"] == "G1_threshold":
        raw = group_threshold(n, pairs, p, profile["tau"])
    elif profile["algorithm"] == "G2_average_linkage":
        raw = group_average_linkage(n, pairs, p, profile["tau"], profile["support"])
    elif profile["algorithm"] == "G3_louvain":
        raw = group_louvain(n, pairs, p, profile["resolution"], seed)
    else:
        raise ValueError("Unknown grouping algorithm.")
    _, labels = np.unique(raw, return_inverse=True)
    return labels


def evaluate_grouping(ids, labels, frame, valid, appearance):
    """Truth enters here only. Zero-vector images stay unresolved (-1), as in the controls."""
    labels = np.where(valid, labels, -1)
    metrics, _ = evaluate_clustering(ClusteringResult(list(ids), labels, [], {}), frame, appearance=appearance)
    exact = sum(r["recovered_as_one_pure_component"] for r in metrics["per_identity"])
    return {"groups": metrics["provisional_component_count"], "multi_image_groups": metrics["multi_image_groups"],
            "singleton_groups": metrics["singleton_proposals"], "pairwise_precision": metrics["pairwise_precision"],
            "pairwise_recall": metrics["pairwise_recall"], "pairwise_f1": metrics["pairwise_f1"],
            "adjusted_rand_index": metrics.get("adjusted_rand_index"),
            "merge_errors": metrics["merge_errors"], "split_errors": metrics["split_errors"],
            "corrections_needed": metrics["merge_errors"] + metrics["split_errors"],
            "exact_identity_recoveries": int(exact), "true_individuals": metrics["true_individuals"],
            "multi_image_group_count_error": metrics["multi_image_groups"] - metrics["true_individuals"]}


# --- Pilot sample for Stage B --------------------------------------------------------------

def pilot_sample(ids, pairs, features, edge_type, size=PILOT_EDGES, seed=42):
    """Label-free stratified sample: cosine deciles x edge type, equal allocation, seeded."""
    deciles = pd.qcut(features["cosine"].rank(method="first"), 10, labels=False)
    strata = pd.Series(list(zip(deciles, edge_type)))
    groups = {k: np.flatnonzero((strata == k).to_numpy()) for k in sorted(set(strata))}
    rng = np.random.default_rng(seed)
    per = max(1, size // len(groups))
    chosen = []
    for key, members in groups.items():
        take = min(per, len(members))
        chosen.extend(rng.choice(members, take, replace=False).tolist())
    remaining = np.setdiff1d(np.arange(len(pairs)), chosen)
    if len(chosen) < size and len(remaining):
        chosen.extend(rng.choice(remaining, min(size - len(chosen), len(remaining)), replace=False).tolist())
    chosen = np.sort(np.asarray(chosen, dtype=int))
    return pd.DataFrame({"image_id_a": [ids[pairs[i, 0]] for i in chosen],
                         "image_id_b": [ids[pairs[i, 1]] for i in chosen],
                         "cosine": features["cosine"].to_numpy()[chosen],
                         "edge_type": np.asarray(edge_type)[chosen]})


# --- Report ---------------------------------------------------------------------------------

def _fmt(value):
    return f"{value:.3f}" if isinstance(value, float) else "" if value is None else str(value)


def _table_html(rows, keys):
    body = "<table><tr>" + "".join(f"<th>{html.escape(k)}</th>" for k in keys) + "</tr>"
    for row in rows:
        body += "<tr>" + "".join(f"<td>{html.escape(_fmt(row.get(k)))}</td>" for k in keys) + "</tr>"
    return body + "</table>"


def plot_reliability(path, models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], color="#999", lw=1)
    for name, summary in models.items():
        rows = [r for r in summary["reliability"] if r["edges"]]
        ax.plot([r["mean_predicted"] for r in rows], [r["observed_same_rate"] for r in rows], marker="o",
                label=f"{name} (AUC {summary['auc']:.3f})")
    ax.set_xlabel("Cross-fitted P(same animal)")
    ax.set_ylabel("Observed same-animal rate")
    ax.legend(fontsize=7)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(out, results):
    out = Path(out)
    write_json(out / "summary.json", {**{k: v for k, v in results.items() if k != "pilot"}, "notes": NOTES})
    _write_table(out / "grouping_profiles.csv", results["groupings"], list(results["groupings"][0]))
    _write_table(out / "encounter_links.csv", results["encounter_links"], list(results["encounter_links"][0]))
    model_rows = [{"model": name, **{k: v for k, v in s.items() if k != "reliability"}}
                  for name, s in results["edge_models"].items()]
    _write_table(out / "edge_models.csv", model_rows, list(model_rows[0]))
    plot_reliability(out / "reliability.png", {k: v for k, v in results["edge_models"].items() if k.endswith("k150")})
    if "pilot" in results:
        results["pilot"].to_csv(out / "pilot_edges.csv", index=False)
    best = sorted(results["groupings"], key=lambda r: (-(r["exact_identity_recoveries"] or 0),
                                                       -(r.get("pairwise_f1") or 0)))
    body = ("<p>Calibration development evidence. Edges, features and groupings are label-free; the edge "
            "model is identity-cross-fitted. Group counts are not park abundance.</p>")
    body += "<h2>Declared rules</h2><pre>" + html.escape(str(results["declared_rules"])) + "</pre>"
    body += "<h2>Edge models (out-of-fold)</h2><img src='reliability.png' alt='Reliability'>" + _table_html(
        model_rows, ["model", "edges", "positive_edges", "auc", "ambiguous_band_auc", "brier", "fits"])
    body += "<h2>Metadata link rules</h2>" + _table_html(results["encounter_links"], list(results["encounter_links"][0]))
    body += "<h2>Grouping profiles (best first)</h2>" + _table_html(best, [
        "arm", "knn_k", "profile", "groups", "multi_image_groups", "pairwise_precision", "pairwise_recall",
        "exact_identity_recoveries", "merge_errors", "split_errors", "corrections_needed"])
    body += "<h2>Best existing controls</h2>" + _table_html(results["existing_controls"], list(results["existing_controls"][0]))
    body += "<h2>Limitations</h2><ul>" + "".join(f"<li>{html.escape(n)}</li>" for n in NOTES) + "</ul>"
    (out / "index.html").write_text("<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>CzechLynx edge grouping</title><style>body{font:15px system-ui;max-width:1300px;margin:30px auto;"
        "padding:16px}table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #bbb;padding:4px 7px}"
        "pre{white-space:pre-wrap}</style><h1>CzechLynx calibrated edge grouping (Stage A)</h1>" + body + "</html>")
    return {str(p.relative_to(out)): file_digest(p) for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "manifest.json"}


def existing_control_rows(calibration_dir, limit=5):
    """Best saved controls by pairwise F1, for side-by-side comparison."""
    table = pd.read_csv(Path(calibration_dir) / "profiles.csv")
    p, r = table.pairwise_precision.fillna(0), table.pairwise_recall.fillna(0)
    table = table.assign(pairwise_f1=np.where(p + r > 0, 2 * p * r / (p + r).replace(0, 1), 0))
    cols = ["profile_id", "provisional_component_count", "pairwise_precision", "pairwise_recall",
            "pairwise_f1", "merge_errors", "split_errors"]
    rows = table.sort_values("pairwise_f1", ascending=False)[cols].head(limit).to_dict("records")
    return [{k: None if pd.isna(v) else float(v) if isinstance(v, np.floating) else int(v)
             if isinstance(v, np.integer) else v for k, v in row.items()} for row in rows]


def declared_rules(results):
    groupings, models = results["groupings"], results["edge_models"]
    derived_auc = {k: v["auc"] for k, v in models.items() if k.startswith("full_derived")}
    progress = [g for g in groupings if (g["pairwise_precision"] or 0) >= .95 and (g["pairwise_recall"] or 0) >= .70
                and g["exact_identity_recoveries"] >= 15]
    count = [g for g in progress if abs(g["multi_image_group_count_error"]) <= 0.1 * g["true_individuals"]
             and g["merge_errors"] <= 2]

    def best(arm):
        rows = [g for g in groupings if g["arm"] == arm]
        precise = [g for g in rows if (g["pairwise_precision"] or 0) >= .95] or rows
        return max((g["exact_identity_recoveries"] for g in precise), default=0)
    annotated, derived = best("annotated"), best("derived")
    return {
        "edge_model_works": {"derived_auc": derived_auc, "met": any((a or 0) >= .90 for a in derived_auc.values())},
        "grouping_progress": {"profiles": [f"{g['arm']}/k{g['knn_k']}/{g['profile']}" for g in progress],
                              "met": bool(progress)},
        "count_candidate": {"profiles": [f"{g['arm']}/k{g['knn_k']}/{g['profile']}" for g in count], "met": bool(count)},
        "day_level_encounters_deployable": {"best_annotated_exact": annotated, "best_derived_exact": derived,
                                            "met": annotated > 0 and derived >= .8 * annotated},
        "policy": "Mechanical checks of the 2026-10-18 proposal thresholds; not conclusions.",
    }


def profile_name(profile):
    return "_".join(f"{k}{v}" if k != "algorithm" else v for k, v in profile.items())


def labels_for(pairs, identity_codes):
    return (identity_codes[pairs[:, 0]] == identity_codes[pairs[:, 1]]).astype(int)


def summarise_types(in_knn, features, arm):
    link = (features["same_annotated_encounter"].to_numpy() > 0 if arm == "annotated"
            else (features["same_trap_day"].to_numpy() > 0) | (features["same_location_day"].to_numpy() > 0))
    return np.where(in_knn & link, "knn+link", np.where(in_knn, "knn", "link_only"))


def count_by(values):
    return dict(Counter(values))
