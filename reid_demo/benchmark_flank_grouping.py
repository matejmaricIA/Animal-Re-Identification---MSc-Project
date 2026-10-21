"""Calibration-only flank-aware grouping.

Photographs are split by the direction the animal faces in the frame. Each side is
grouped by appearance, and left/right clusters are joined one-to-one on
day-level metadata co-occurrence. The count is reported as a bilateral interval.
Facing comes from an identity-cross-fitted classifier on frozen vectors, trained
on pose-keypoint geometry (an annotation, not identity). Truth enters only in
evaluation. Results are development evidence, not counts or park abundance.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reid_demo.benchmark import file_digest, write_json
from reid_demo.benchmark_clustering import ROOT, ClusteringResult, evaluate_clustering
from reid_demo.benchmark_cluster_diagnostics import _write_table
from reid_demo.benchmark_coverage_diagnostics import risk_auc
from reid_demo import benchmark_edge_grouping as eg


VERSION = "calibration_flank_grouping_v1"
SIDES = (0, 1)  # 0 = faces_left, 1 = faces_right (in the image frame)
SIDE_NAMES = {0: "faces_left", 1: "faces_right"}
FACING_RATIO = 0.5
FACING_SCALE = 10.0
FACING_C = 1.0
AMBIGUOUS = (0.2, 0.8)
BUDGETS = (50, 100, 150)
WITHIN_PROFILES = ({"algorithm": "G2_average_linkage", "tau": 0.5, "support": 1},
                   {"algorithm": "G2_average_linkage", "tau": 0.7, "support": 1},
                   {"algorithm": "G3_louvain", "resolution": 1.0},
                   {"algorithm": "G3_louvain", "resolution": 2.0})
MIN_COOCCURRENCE = (1, 2, 3)
AMBIGUITY_POLICIES = ("assign", "unresolved")
MIN_CLUSTER_IMAGES = 2
HEAD_KEYPOINTS = ("nose", "left-eye", "right-eye", "left-earbase", "right-earbase")
NOTES = [
    "Calibration development evidence only; no selection or official test observations are evaluated.",
    "Facing is predicted for every image by an identity-cross-fitted classifier; pose keypoints only train and validate it.",
    "'faces_left/right' describe the image frame, not anatomical flank names.",
    "Within-facing edges and cross-facing bridges are label-free; truth enters only in evaluation.",
    "Count bounds use flank clusters with at least two images; singletons are reported, not counted.",
    "Day-level co-occurrence can join animals travelling together; one-to-one matching limits but cannot exclude this.",
    "CzechLynx dates are day-level; capture designs elsewhere may not photograph both flanks on one day.",
]


def implementation_binding():
    names = ("reid_demo/benchmark_flank_grouping.py", "tools/evaluate_czechlynx_flank_grouping.py",
             "reid_demo/benchmark_edge_grouping.py")
    return {"version": VERSION, "code": {name: file_digest(ROOT / name) for name in names},
            "facing_ratio": FACING_RATIO, "facing_scale": FACING_SCALE, "facing_c": FACING_C,
            "ambiguous": list(AMBIGUOUS), "budgets": list(BUDGETS), "within_profiles": list(WITHIN_PROFILES),
            "min_cooccurrence": list(MIN_COOCCURRENCE), "ambiguity_policies": list(AMBIGUITY_POLICIES),
            "min_cluster_images": MIN_CLUSTER_IMAGES}


# --- Facing ---------------------------------------------------------------------------------

def pose_facing(pose, ratio=FACING_RATIO):
    """1 if the head is right of the tail base by >= ratio x head-tail distance, 0 if left, else NaN."""
    if not isinstance(pose, str) or not pose.strip():
        return np.nan
    points = json.loads(pose)
    head = [points[k] for k in HEAD_KEYPOINTS if k in points]
    if not head or "tailbase" not in points:
        return np.nan
    hx, hy = np.mean(np.asarray(head, float), axis=0)
    tx, ty = map(float, points["tailbase"])
    length = float(np.hypot(hx - tx, hy - ty))
    if length == 0:
        return np.nan
    offset = (hx - tx) / length
    return 1.0 if offset >= ratio else 0.0 if offset <= -ratio else np.nan


def load_pose_facing(raw_metadata, frame):
    """Pose facing aligned to frame rows by source_path; unmatched or unusable rows are NaN."""
    raw = pd.read_csv(raw_metadata, usecols=["path", "pose"])
    pose = dict(zip(raw.path.astype(str), raw.pose))
    if "source_path" not in frame:
        raise ValueError("Frame needs source_path to align pose keypoints.")
    return np.asarray([pose_facing(pose.get(str(p))) for p in frame.source_path], dtype=float)


def facing_probabilities(vectors, labels, folds, *, scale=FACING_SCALE, c=FACING_C):
    """p(faces_right) for every image from a model that never saw that image's animal."""
    from sklearn.linear_model import LogisticRegression
    x = np.asarray(vectors, float) * scale
    labelled = ~np.isnan(labels)
    output = np.full(len(x), np.nan)
    for fold in np.unique(folds):
        train = labelled & (folds != fold)
        if len(np.unique(labels[train])) < 2:
            raise ValueError("A facing fold lacks both facing classes.")
        model = LogisticRegression(max_iter=5000, C=c).fit(x[train], labels[train])
        output[folds == fold] = model.predict_proba(x[folds == fold])[:, 1]
    return output


def facing_validation(p, pose_labels, ids, reviews=None):
    """Agreement with pose labels and (optionally) the human audit, whose 'right' means faces right."""
    labelled = ~np.isnan(pose_labels)
    confident = (p < AMBIGUOUS[0]) | (p > AMBIGUOUS[1])
    result = {"pose_labelled_images": int(labelled.sum()),
              "pose_auc": risk_auc(p[labelled], pose_labels[labelled] == 1) if labelled.any() else None,
              "pose_accuracy": float(((p[labelled] > .5) == (pose_labels[labelled] == 1)).mean()) if labelled.any() else None,
              "confident_share_all": float(confident.mean()),
              "confident_share_unlabelled": float(confident[~labelled].mean()) if (~labelled).any() else None,
              "predicted_faces_right_share": float((p > .5).mean())}
    if reviews is not None:
        position = {rid: i for i, rid in enumerate(ids)}
        audit = reviews[reviews.image_id.isin(position) & reviews.flank.isin(["left", "right"])]
        index = np.asarray([position[r] for r in audit.image_id], dtype=int)
        truth = (audit.flank == "right").to_numpy()
        no_pose = ~labelled[index] if len(index) else np.zeros(0, bool)
        result.update(audit_images=len(index), audit_convention="audit 'right' == faces right in frame (pose-verified 56/56)",
                      audit_auc=risk_auc(p[index], truth) if len(index) else None,
                      audit_accuracy=float(((p[index] > .5) == truth).mean()) if len(index) else None,
                      audit_images_without_pose=int(no_pose.sum()),
                      audit_auc_without_pose=risk_auc(p[index][no_pose], truth[no_pose]) if no_pose.any() else None)
    return result


# --- Within-facing grouping -----------------------------------------------------------------

def side_subproblem(side, inputs, scores, facing, keep):
    """Indices (into the full collection) of valid kept images predicted to face `side`."""
    return np.flatnonzero(inputs.valid & keep & ((facing > .5) == bool(side)))


def within_side_edges(index, inputs, scores, risk, p_facing, arm, budget, folds, seed=42):
    """Candidate graph, cross-fitted edge probabilities and labels for one facing side (local indices)."""
    sub_scores = scores[np.ix_(index, index)]
    valid = np.ones(len(index), bool)
    frame = inputs.frame.iloc[index].reset_index(drop=True)
    keys = eg.metadata_keys(frame)
    pairs, _ = eg.candidate_graph(sub_scores, valid, keys, arm, budget)
    ranks = eg.rank_matrix(sub_scores, valid)
    features = eg.edge_features(pairs, sub_scores, ranks, valid, keys, inputs.appearance_codes[index],
                                risk[index], annotated=arm == "annotated")
    confidence = np.abs(p_facing[index] - .5) * 2
    features["facing_confidence_min"] = np.minimum(confidence[pairs[:, 0]], confidence[pairs[:, 1]])
    codes = inputs.identity_codes[index]
    labels = eg.labels_for(pairs, codes)
    columns = list(eg.GRAPH_FEATURES) + ["facing_confidence_min"] + (
        ["same_annotated_encounter"] if arm == "annotated" else [])
    p, fits = eg.cross_fit_probabilities(features, labels, pairs, folds[index], columns=columns, seed=seed)
    return pairs, p, labels, fits


def group_side(index, pairs, p, profile, seed=42):
    """Cluster labels (local) for one side."""
    return eg.run_grouping(profile, len(index), pairs, p, seed=seed)


# --- Cross-flank bridging and counts --------------------------------------------------------

def link_keys(frame, arm):
    keys = eg.metadata_keys(frame)
    if arm == "annotated":
        return [keys["annotated"]]
    return [keys["trap_day"], keys["location_day"]]


def cooccurrence(cluster_of, side_of, key_lists):
    """Count opposite-facing image pairs sharing any link key, per (left cluster, right cluster)."""
    counts = Counter()
    for keys in key_lists:
        groups = defaultdict(lambda: ([], []))
        for i, key in enumerate(keys):
            if key is not None and cluster_of[i] >= 0:
                groups[key][side_of[i]].append(cluster_of[i])
        for left, right in groups.values():
            for a in left:
                for b in right:
                    counts[(a, b)] += 1
    return counts


def match_one_to_one(counts, minimum):
    """Greedy one-to-one matching on co-occurrence counts (ties by cluster ids)."""
    if int(minimum) < 1:
        raise ValueError("Minimum co-occurrence must be positive.")
    used_left, used_right, pairs = set(), set(), []
    for (a, b), n in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if n < minimum:
            break
        if a not in used_left and b not in used_right:
            used_left.add(a)
            used_right.add(b)
            pairs.append((a, b, n))
    return pairs


def bilateral_bounds(left_sizes, right_sizes, matched, min_images=MIN_CLUSTER_IMAGES):
    """Count interval over flank clusters with >= min_images photos."""
    left = {c for c, n in left_sizes.items() if n >= min_images}
    right = {c for c, n in right_sizes.items() if n >= min_images}
    paired = [(a, b) for a, b, _ in matched if a in left and b in right]
    un_left = len(left - {a for a, _ in paired})
    un_right = len(right - {b for _, b in paired})
    return {"left_clusters": len(left), "right_clusters": len(right), "paired": len(paired),
            "unmatched_left": un_left, "unmatched_right": un_right,
            "lower": len(paired) + max(un_left, un_right), "upper": len(paired) + un_left + un_right,
            "point_unmatched_distinct": len(paired) + un_left + un_right}


def combine_labels(n, side_of, cluster_of, matched):
    """Animal labels: matched left/right clusters share one label; cluster_of -1 stays unresolved."""
    partner = {}
    for a, b, _ in matched:
        partner[(0, a)] = partner[(1, b)] = ("pair", a, b)
    names, labels = {}, np.full(n, -1)
    for i in range(n):
        if cluster_of[i] >= 0:
            key = partner.get((side_of[i], cluster_of[i]), ("solo", side_of[i], cluster_of[i]))
            labels[i] = names.setdefault(key, len(names))
    return labels


# --- Evaluation (truth used only here) ------------------------------------------------------

def evaluate_animals(ids, labels, frame, appearance):
    metrics, _ = evaluate_clustering(ClusteringResult(list(ids), labels, [], {}), frame, appearance=appearance)
    exact = sum(r["recovered_as_one_pure_component"] for r in metrics["per_identity"])
    return {"groups": metrics["provisional_component_count"], "multi_image_groups": metrics["multi_image_groups"],
            "unresolved_images": metrics["unresolved_images"],
            "pairwise_precision": metrics["pairwise_precision"], "pairwise_recall": metrics["pairwise_recall"],
            "pairwise_f1": metrics["pairwise_f1"], "merge_errors": metrics["merge_errors"],
            "split_errors": metrics["split_errors"], "exact_identity_recoveries": int(exact),
            "true_individuals": metrics["true_individuals"]}


def evaluate_flank_units(ids, cluster_labels, side_of, frame, appearance):
    """Flank-level metrics; truth unit = animal x predicted facing (an approximation)."""
    unit_frame = frame.copy()
    unit_frame["identity"] = [f"{i}|{SIDE_NAMES[s]}" for i, s in zip(frame.identity.astype(str), side_of)]
    labels = np.where(cluster_labels >= 0, cluster_labels * 2 + side_of, -1)
    metrics, _ = evaluate_clustering(ClusteringResult(list(ids), labels, [], {}), unit_frame, appearance=appearance)
    exact = sum(r["recovered_as_one_pure_component"] for r in metrics["per_identity"])
    return {"flank_units": metrics["true_individuals"], "flank_exact_recoveries": int(exact),
            "flank_pairwise_precision": metrics["pairwise_precision"],
            "flank_pairwise_recall": metrics["pairwise_recall"], "flank_merge_errors": metrics["merge_errors"],
            "flank_split_errors": metrics["split_errors"]}


def declared_rules(results):
    facing = results["facing"]
    rows = results["profiles"]
    facing_ok = ((facing.get("audit_auc") or 0) >= .97 and (facing.get("audit_accuracy") or 0) >= .93
                 and facing["confident_share_all"] >= .70)
    flank = [r for r in rows if r["flank_exact_recoveries"] >= 30 and (r["flank_pairwise_precision"] or 0) >= .90]
    animal = [r for r in rows if (r["pairwise_f1"] or 0) >= .70 and r["exact_identity_recoveries"] >= 10
              and r["merge_errors"] <= 10]
    count = [r for r in animal if r["count_lower"] <= r["true_individuals"] <= r["count_upper"]
             and r["count_upper"] - r["count_lower"] <= 0.2 * r["true_individuals"]]
    name = lambda r: r["profile_id"]
    return {"facing_classifier_usable": {"met": bool(facing_ok)},
            "flank_level_progress": {"profiles": [name(r) for r in flank][:20], "count": len(flank), "met": bool(flank)},
            "animal_level_progress": {"profiles": [name(r) for r in animal][:20], "count": len(animal), "met": bool(animal)},
            "count_candidate": {"profiles": [name(r) for r in count][:20], "count": len(count), "met": bool(count)},
            "policy": "Mechanical checks of the 2026-10-18 flank proposal thresholds; not conclusions."}


# --- Report ---------------------------------------------------------------------------------

def _fmt(value):
    return f"{value:.3f}" if isinstance(value, float) else "" if value is None else str(value)


def _table(rows, keys):
    body = "<table><tr>" + "".join(f"<th>{html.escape(k)}</th>" for k in keys) + "</tr>"
    for row in rows:
        body += "<tr>" + "".join(f"<td>{html.escape(_fmt(row.get(k)))}</td>" for k in keys) + "</tr>"
    return body + "</table>"


def write_report(out, results, facing_rows):
    out = Path(out)
    write_json(out / "summary.json", {**{k: v for k, v in results.items() if k != "profiles"},
                                      "profile_count": len(results["profiles"]), "notes": NOTES})
    _write_table(out / "profiles.csv", results["profiles"], list(results["profiles"][0]))
    _write_table(out / "facing.csv", facing_rows, list(facing_rows[0]))
    _write_table(out / "side_edge_models.csv", results["side_edge_models"], list(results["side_edge_models"][0]))
    best = sorted(results["profiles"], key=lambda r: (-r["exact_identity_recoveries"], -(r["pairwise_f1"] or 0)))[:25]
    body = ("<p>Calibration development evidence. Facing is predicted per image (identity-cross-fitted); "
            "within-facing grouping and cross-facing bridging are label-free; counts are intervals, not abundance.</p>")
    body += "<h2>Declared rules</h2><pre>" + html.escape(json.dumps(results["declared_rules"], indent=1)) + "</pre>"
    body += "<h2>Facing classifier</h2><pre>" + html.escape(json.dumps(results["facing"], indent=1)) + "</pre>"
    body += "<h2>Per-side edge models</h2>" + _table(results["side_edge_models"], list(results["side_edge_models"][0]))
    body += "<h2>Best profiles</h2>" + _table(best, [
        "profile_id", "pairwise_precision", "pairwise_recall", "pairwise_f1", "exact_identity_recoveries",
        "merge_errors", "split_errors", "unresolved_images", "flank_exact_recoveries", "flank_pairwise_precision",
        "paired", "count_lower", "count_upper", "true_individuals"])
    body += "<h2>Limitations</h2><ul>" + "".join(f"<li>{html.escape(n)}</li>" for n in NOTES) + "</ul>"
    (out / "index.html").write_text("<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>CzechLynx flank grouping</title><style>body{font:15px system-ui;max-width:1300px;margin:30px auto;"
        "padding:16px}table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #bbb;padding:4px 7px}"
        "pre{white-space:pre-wrap}</style><h1>CzechLynx flank-aware grouping</h1>" + body + "</html>")
    return {str(p.relative_to(out)): file_digest(p) for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "manifest.json"}
