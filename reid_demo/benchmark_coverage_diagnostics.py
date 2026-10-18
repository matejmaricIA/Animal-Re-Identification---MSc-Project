"""Calibration-only candidate-coverage and identifiability diagnostics.

Asks why correct links are missing from the label-free candidate graph: are the
disconnected photographs unidentifiable (H1), split encounters (H3), outside a
too-narrow neighbour list (H4) or opposite flanks (H2)? Every edge set and the
identifiability score are built without identities; reference truth enters only
in the evaluators below. Oracle fragments are diagnostics, never counts.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from reid_demo.benchmark import file_digest, fingerprint, write_json
from reid_demo.benchmark_clustering import ROOT, SCORE_DECIMALS, _camera, _positive_int, _value
from reid_demo.benchmark_cluster_diagnostics import _ratio, _write_table


VERSION = "calibration_coverage_identifiability_v1"
PRIMARY_BUDGET = 50
BUDGETS = (15, 50, 100)
ABSTENTION_FRACTIONS = (0.0, 0.05, 0.10, 0.20, 0.30)
RANDOM_REPEATS = 20
QE_NEIGHBOURS = (3, 5)
QE_ALPHA = 3
AUDIT_FIELDS = ("flank", "usable_markings", "animal_visibility", "crop_mask", "illumination")
RISK_COMPONENTS = ("small_animal", "low_contrast", "fragmented_mask", "truncated")
NOTES = [
    "Calibration diagnostics only; no selection or official test observations are evaluated.",
    "Candidate graphs, query expansion, encounter links and the identifiability score are label-free.",
    "Oracle connectivity uses reference truth to remove wrong edges; it is not achieved accuracy or a count.",
    "The identifiability risk is a fixed formula declared before this run, not fitted to calibration truth.",
    "Abstention removes images before candidate construction, as a deployed quality gate would.",
    "Audited images are purposive plus uniform cohorts; their cross-tabs are descriptive, not collection rates.",
    "Flank labels come from the human audit; unknown flanks are reported separately, never imputed.",
]


def implementation_binding():
    names = ("reid_demo/benchmark_coverage_diagnostics.py", "tools/diagnose_czechlynx_coverage.py")
    return {"version": VERSION, "code": {name: file_digest(ROOT / name) for name in names},
            "budgets": list(BUDGETS), "primary_budget": PRIMARY_BUDGET,
            "abstention_fractions": list(ABSTENTION_FRACTIONS), "random_repeats": RANDOM_REPEATS,
            "qe_neighbours": list(QE_NEIGHBOURS), "qe_alpha": QE_ALPHA,
            "risk_components": list(RISK_COMPONENTS), "cosine_rounding_decimals": SCORE_DECIMALS}


# --- Label-free image evidence -------------------------------------------------------------

def image_quality_features(frame):
    """Model-free measurements of each crop and its saved mask (whole crop for bbox inputs)."""
    from PIL import Image
    rows = []
    for row in frame.to_dict("records"):
        with Image.open(row["path"]) as image:
            grey = np.asarray(image.convert("L"), dtype=np.float64)
        mask_path = _value(row.get("mask_path"))
        if mask_path:
            expected = _value(row.get("mask_sha256"))
            if expected and file_digest(mask_path) != expected:
                raise ValueError(f"Mask changed since preparation: {row['image_id']}")
            with Image.open(mask_path) as image:
                mask = np.asarray(image.convert("L")) > 127
            if mask.shape != grey.shape:
                raise ValueError(f"Mask and crop shapes differ: {row['image_id']}")
        else:
            mask = np.ones(grey.shape, dtype=bool)
        pixels = int(mask.sum())
        labels, components = ndimage.label(mask)
        largest = int(np.bincount(labels.ravel())[1:].max()) if components else 0
        border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
        inside = grey[mask]
        rows.append({"image_id": str(row["image_id"]), "mask_available": bool(mask_path),
                     "crop_width": int(grey.shape[1]), "crop_height": int(grey.shape[0]),
                     "mask_pixels": pixels, "mask_area_fraction": pixels / mask.size,
                     "mask_components": int(components),
                     "largest_component_share": largest / pixels if pixels else 0.0,
                     "border_mask_fraction": float(border.mean()),
                     "luminance_mean": float(inside.mean()) if pixels else 0.0,
                     "luminance_std": float(inside.std()) if pixels else 0.0})
    return pd.DataFrame(rows)


def identifiability_risk(features):
    """Mean percentile rank of four declared risk signals; higher means less identifiable."""
    signals = pd.DataFrame({
        "small_animal": -np.sqrt(features.mask_pixels.astype(float)),
        "low_contrast": -features.luminance_std.astype(float),
        "fragmented_mask": 1 - features.largest_component_share.astype(float),
        "truncated": features.border_mask_fraction.astype(float),
    })
    ranks = signals.rank(method="average", pct=True)
    return ranks.mean(axis=1).to_numpy(), ranks


# --- Label-free candidate graphs -----------------------------------------------------------

def score_matrix(vectors):
    """Rounded cosine matrix with the same rounding as the calibration candidate builder."""
    x = np.asarray(vectors, dtype=np.float64)
    out = np.empty((len(x), len(x)))
    for start in range(0, len(x), 64):  # Same row blocks as the calibration builder.
        out[start:start + 64] = np.clip(np.round(x[start:start + 64] @ x.T, SCORE_DECIMALS), -1, 1)
    return out


def _directional_topk(scores, keep, k):
    """Rows/cols/ranks of each kept image's top-k kept neighbours, ties to the lower index."""
    index = np.flatnonzero(keep)
    k = min(k, len(index) - 1)
    rows, cols, ranks = [], [], []
    if k < 1:
        return np.empty(0, int), np.empty(0, int), np.empty(0, int)
    for position, i in enumerate(index):
        row = scores[i, index].copy()
        row[position] = -np.inf
        part = np.argpartition(-row, k - 1)[:k]
        kth = row[part].min()
        greater = np.flatnonzero(row > kth)
        equal = np.flatnonzero(row == kth)[:k - len(greater)]
        chosen = np.concatenate([greater, equal])
        chosen = chosen[np.lexsort((chosen, -row[chosen]))]
        rows.append(np.full(len(chosen), i))
        cols.append(index[chosen])
        ranks.append(np.arange(len(chosen)))
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(ranks)


def knn_edges(scores, keep, k, *, mutual=False):
    """Union (or mutual intersection) of directional top-k lists as sorted (a, b, score) edges."""
    _positive_int(k, "candidate budget")
    rows, cols, _ = _directional_topk(scores, keep, k)
    a, b = np.minimum(rows, cols), np.maximum(rows, cols)
    pairs, counts = np.unique(np.stack([a, b], axis=1), axis=0, return_counts=True)
    if mutual:
        pairs = pairs[counts == 2]
    values = scores[pairs[:, 0], pairs[:, 1]] if len(pairs) else np.empty(0)
    order = np.lexsort((pairs[:, 1], pairs[:, 0], -values)) if len(pairs) else np.empty(0, int)
    return [(int(pairs[i, 0]), int(pairs[i, 1]), float(values[i])) for i in order]


def query_expanded_vectors(scores, vectors, valid, neighbours, alpha=QE_ALPHA):
    """Alpha query expansion: each vector plus its top neighbours weighted by score**alpha."""
    _positive_int(neighbours, "query expansion neighbours")
    x = np.asarray(vectors, dtype=np.float64)
    rows, cols, _ = _directional_topk(scores, valid, neighbours)
    weights = np.maximum(scores[rows, cols], 0) ** alpha
    expanded = x.copy()
    np.add.at(expanded, rows, weights[:, None] * x[cols])
    norms = np.linalg.norm(expanded, axis=1)
    expanded[norms > 0] /= norms[norms > 0, None]
    expanded[~valid] = 0
    return expanded


def encounter_edges(frame):
    """Chain links inside each metadata encounter (no identities)."""
    groups = defaultdict(list)
    for i, value in enumerate(frame.encounter_id.map(_value)):
        if value is not None:
            groups[value].append(i)
    return [(members[j], members[j + 1]) for members in groups.values() for j in range(len(members) - 1)]


# --- Truth-based evaluators (after every edge set is fixed) --------------------------------

def oracle_components(identity_codes, keep, edges):
    """Truth-filtered components restricted to kept images; returns labels, degree, summary."""
    n = len(identity_codes)
    pairs = np.asarray([(a, b) for a, b, *_ in edges], dtype=int).reshape(-1, 2)
    if len(pairs):
        pairs = pairs[keep[pairs[:, 0]] & keep[pairs[:, 1]]
                      & (identity_codes[pairs[:, 0]] == identity_codes[pairs[:, 1]])]
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    degree = np.bincount(pairs.ravel(), minlength=n) if len(pairs) else np.zeros(n, int)
    labels = np.where(keep, labels, -1)
    fragments = Counter()
    kept_by_identity = Counter(identity_codes[keep].tolist())
    for code in kept_by_identity:
        fragments[code] = len(set(labels[keep & (identity_codes == code)].tolist()))
    identities = len(set(identity_codes.tolist()))
    multi = [c for c, m in kept_by_identity.items() if m > 1]
    summary = {"retained_images": int(keep.sum()), "retained_identities": len(kept_by_identity),
               "identities_without_retained_images": identities - len(kept_by_identity),
               "fully_connected_identities": sum(v == 1 for v in fragments.values()),
               "retained_multi_image_identities": len(multi),
               "fully_connected_multi_image_identities": sum(fragments[c] == 1 for c in multi),
               "oracle_fragments": int(sum(fragments.values())),
               "retained_images_without_correct_neighbour": int(np.sum(keep & (degree == 0)
                   & np.isin(identity_codes, multi)))}
    return labels, degree, summary


def edge_precision(identity_codes, edges):
    same = sum(identity_codes[a] == identity_codes[b] for a, b, *_ in edges)
    return {"edges": len(edges), "same_identity_edges": int(same),
            "different_identity_edges": len(edges) - int(same), "direct_edge_precision": _ratio(int(same), len(edges))}


def abstention_curve(scores, inputs, risk, *, budget=PRIMARY_BUDGET, fractions=ABSTENTION_FRACTIONS,
                     repeats=RANDOM_REPEATS, seed=42):
    """Drop highest-risk images vs random images, rebuilding the candidate graph each time."""
    valid = inputs.valid
    order = np.flatnonzero(valid)[np.lexsort((np.flatnonzero(valid), -risk[valid]))]
    rows = []
    for fraction in fractions:
        if not 0 <= fraction < 1:
            raise ValueError("Abstention fractions must be in [0, 1).")
        drop = int(round(fraction * len(order)))
        plans = [("risk", None, order[:drop])]
        if drop:
            plans += [("random", r, np.random.default_rng(seed + r).choice(order, drop, replace=False))
                      for r in range(repeats)]
        for policy, repeat, dropped in plans:
            keep = valid.copy()
            keep[dropped] = False
            edges = knn_edges(scores, keep, budget)
            _, _, summary = oracle_components(inputs.identity_codes, keep, edges)
            rows.append({"policy": policy, "repeat": repeat, "fraction": fraction, "dropped_images": int(drop),
                         "knn_k": budget, **summary, **edge_precision(inputs.identity_codes, edges)})
    return rows


def summarise_abstention(rows):
    out = []
    for fraction in sorted({r["fraction"] for r in rows}):
        risk = next(r for r in rows if r["fraction"] == fraction and r["policy"] == "risk")
        random = [r for r in rows if r["fraction"] == fraction and r["policy"] == "random"] or [risk]
        entry = {"fraction": fraction, "dropped_images": risk["dropped_images"]}
        for key in ("fully_connected_identities", "fully_connected_multi_image_identities",
                    "oracle_fragments", "identities_without_retained_images", "direct_edge_precision"):
            values = [r[key] for r in random]
            entry[f"risk_{key}"] = risk[key]
            entry[f"random_mean_{key}"] = float(np.mean(values))
            entry[f"random_min_{key}"] = min(values)
            entry[f"random_max_{key}"] = max(values)
        out.append(entry)
    return out


def encounter_analysis(scores, inputs, *, budget=PRIMARY_BUDGET):
    valid = inputs.valid
    base = knn_edges(scores, valid, budget)
    links = [(a, b) for a, b in encounter_edges(inputs.frame) if valid[a] and valid[b]]
    _, _, before = oracle_components(inputs.identity_codes, valid, base)
    _, _, after = oracle_components(inputs.identity_codes, valid, base + links)
    groups = defaultdict(list)
    for i, value in enumerate(inputs.frame.encounter_id.map(_value)):
        if value is not None:
            groups[value].append(i)
    pairs = mixed_pairs = 0
    mixed_encounters = []
    for value, members in groups.items():
        codes = inputs.identity_codes[members]
        n = len(members)
        pairs += n * (n - 1) // 2
        counts = Counter(codes.tolist())
        mixed_pairs += n * (n - 1) // 2 - sum(c * (c - 1) // 2 for c in counts.values())
        if len(counts) > 1:
            mixed_encounters.append(value)
    multi = sum(len(m) > 1 for m in groups.values())
    return {"knn_k": budget, "encounters": len(groups), "multi_image_encounters": multi,
            "images_missing_encounter": int(inputs.frame.encounter_id.map(_value).isna().sum()),
            "within_encounter_pairs": pairs, "within_encounter_pairs_joining_different_identities": mixed_pairs,
            "fraction_encounter_pairs_joining_different_identities": _ratio(mixed_pairs, pairs),
            "mixed_identity_encounters": len(mixed_encounters),
            "mixed_identity_encounter_ids": sorted(mixed_encounters)[:50],
            "chain_links_added": len(links), "candidates_only": before, "candidates_plus_encounters": after,
            "fully_connected_identity_gain": after["fully_connected_identities"] - before["fully_connected_identities"]}


def candidate_variants(scores, inputs, *, budgets=BUDGETS, qe_neighbours=QE_NEIGHBOURS):
    valid = inputs.valid
    variants = [(f"knn_k{k}", lambda k=k: knn_edges(scores, valid, k)) for k in budgets]
    wide = [k for k in budgets if k >= min(PRIMARY_BUDGET, max(budgets))]
    variants += [(f"mutual_knn_k{k}", lambda k=k: knn_edges(scores, valid, k, mutual=True)) for k in wide]
    expanded = {}

    def expanded_edges(m, k):
        if m not in expanded:
            expanded.clear()  # Hold one extra score matrix at a time.
            expanded[m] = score_matrix(query_expanded_vectors(scores, inputs.vectors, valid, m))
        return knn_edges(expanded[m], valid, k)
    variants += [(f"query_expansion_m{m}_knn_k{k}", lambda m=m, k=k: expanded_edges(m, k))
                 for m in qe_neighbours for k in wide]
    rows, primary = [], None
    for name, make in variants:
        edges = make()
        labels, degree, summary = oracle_components(inputs.identity_codes, valid, edges)
        rows.append({"variant": name, **summary, **edge_precision(inputs.identity_codes, edges)})
        if name == f"knn_k{PRIMARY_BUDGET}":
            primary = (edges, labels, degree)
    return rows, primary


# --- Per-image and audit tables ------------------------------------------------------------

def fragment_tables(inputs, labels, degree, features, risk):
    frame = inputs.frame.reset_index(drop=True)
    sizes = Counter(labels[labels >= 0].tolist())
    image_rows, by_identity = [], defaultdict(list)
    for i, row in enumerate(frame.to_dict("records")):
        by_identity[inputs.identity_codes[i]].append(i)
    rank_of = {}
    for code, members in by_identity.items():
        fragments = sorted({labels[i] for i in members if labels[i] >= 0},
                           key=lambda f: (-sizes[f], min(j for j in members if labels[j] == f)))
        rank_of.update({f: r for r, f in enumerate(fragments)})
    feature_rows = features.set_index("image_id")
    for i, row in enumerate(frame.to_dict("records")):
        rid = inputs.image_ids[i]
        image_rows.append({"image_id": rid, "identity": str(row["identity"]),
            "fragment_rank": rank_of.get(labels[i]), "fragment_size": sizes.get(labels[i], 0),
            "in_largest_fragment": rank_of.get(labels[i]) == 0,
            "has_correct_candidate_neighbour": bool(degree[i] > 0),
            "identifiability_risk": float(risk[i]), "encounter_id": _value(row.get("encounter_id")),
            "camera": _camera(row), "timestamp": _value(row.get("timestamp")),
            **{k: v for k, v in feature_rows.loc[rid].to_dict().items()}})
    images = pd.DataFrame(image_rows)
    fragment_rows = []
    for (identity, rank), group in images[images.fragment_rank.notna()].groupby(["identity", "fragment_rank"]):
        fragment_rows.append({"identity": identity, "fragment_rank": int(rank), "images": len(group),
            "cameras": group.camera.nunique(), "encounters": group.encounter_id.nunique(),
            "first_date": group.timestamp.min(), "last_date": group.timestamp.max(),
            "mean_identifiability_risk": float(group.identifiability_risk.mean())})
    return images, pd.DataFrame(fragment_rows)


def fragment_risk_summary(images, *, top_fraction=0.2):
    """Do photographs outside each identity's largest fragment carry higher identifiability risk?"""
    valid = images[images.fragment_rank.notna()]
    cutoff = valid.identifiability_risk.quantile(1 - top_fraction)
    rows = {}
    for name, group in (("largest_fragment", valid[valid.in_largest_fragment]),
                        ("other_fragments", valid[~valid.in_largest_fragment]),
                        ("singleton_fragments", valid[valid.fragment_size == 1])):
        rows[name] = {"images": len(group),
                      "mean_identifiability_risk": float(group.identifiability_risk.mean()) if len(group) else None,
                      f"fraction_in_top_{int(top_fraction * 100)}pct_risk":
                          float((group.identifiability_risk >= cutoff).mean()) if len(group) else None}
    rows["auc_risk_predicts_outside_largest_fragment"] = risk_auc(valid.identifiability_risk,
                                                                  ~valid.in_largest_fragment.astype(bool))
    return rows


def risk_auc(risk, positive):
    """Probability a positive image outranks a negative one (Mann-Whitney), ties count half."""
    risk, positive = np.asarray(risk, float), np.asarray(positive, bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return None
    ranks = pd.Series(risk).rank(method="average").to_numpy()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def load_audit(reviews_path):
    """Read the human audit and check it against its saved receipt."""
    reviews_path = Path(reviews_path).resolve()
    receipt = json.loads((reviews_path.parent / "receipt.json").read_text())
    if receipt.get("status") != "complete" or receipt.get("artifacts", {}).get(reviews_path.name) != file_digest(reviews_path):
        raise ValueError("Audit reviews do not match their completed receipt.")
    reviews = pd.DataFrame(json.loads(reviews_path.read_text()))
    if reviews.image_id.duplicated().any():
        raise ValueError("Audit reviews contain duplicate images.")
    cohorts = {s["image_id"]: "+".join(sorted(s["cohorts"])) for s in receipt.get("sample", [])}
    reviews["cohorts"] = reviews.image_id.map(cohorts).fillna("unknown")
    binding = {"reviews_sha256": file_digest(reviews_path),
               "receipt_sha256": file_digest(reviews_path.parent / "receipt.json"),
               "receipt_fingerprint": receipt.get("fingerprint")}
    return reviews, binding


def audit_crosstabs(images, reviews, scores, inputs, *, budget=PRIMARY_BUDGET):
    position = {rid: i for i, rid in enumerate(inputs.image_ids)}
    audited = reviews[reviews.image_id.isin(position)].copy()
    merged = audited.merge(images, on="image_id", how="left")
    rows = []
    for field in (*AUDIT_FIELDS, "cohorts"):
        for value, group in merged.groupby(field, dropna=False):
            rows.append({"field": field, "value": str(value), "images": len(group),
                "has_correct_candidate_neighbour_rate": float(group.has_correct_candidate_neighbour.mean()),
                "in_largest_fragment_rate": float(group.in_largest_fragment.mean()),
                "mean_identifiability_risk": float(group.identifiability_risk.mean())})
    aucs = {}
    for field, bad in (("usable_markings", {"no"}), ("crop_mask", {"issue"}),
                       ("animal_visibility", {"partial", "no_clear_animal"})):
        known = merged[merged[field].isin(bad | {"yes", "correct", "clear"})]
        aucs[field] = {"positive_values": sorted(bad), "images": len(known),
                       "auc": risk_auc(known.identifiability_risk, known[field].isin(bad))}
    # Same-identity audited pairs by human flank relation, with a k-NN membership check.
    rows_k, cols_k, _ = _directional_topk(scores, inputs.valid, budget)
    linked = set(zip(np.minimum(rows_k, cols_k).tolist(), np.maximum(rows_k, cols_k).tolist()))
    pair_rows = []
    flank = dict(zip(merged.image_id, merged.flank))
    for identity, group in merged.groupby("identity"):
        ids = sorted(group.image_id)
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                a, b = sorted((position[ids[x]], position[ids[y]]))
                fa, fb = flank[ids[x]], flank[ids[y]]
                relation = ("unknown" if "unknown" in (fa, fb) or not {fa, fb} <= {"left", "right"}
                            else "same_flank" if fa == fb else "opposite_flank")
                pair_rows.append({"identity": identity, "image_id_a": ids[x], "image_id_b": ids[y],
                                  "flank_a": fa, "flank_b": fb, "relation": relation,
                                  "cosine_similarity": float(scores[a, b]),
                                  f"in_knn_k{budget}": (a, b) in linked})
    pairs = pd.DataFrame(pair_rows)
    flank_summary = []
    for relation in ("same_flank", "opposite_flank", "unknown"):
        group = pairs[pairs.relation == relation] if len(pairs) else pairs
        flank_summary.append({"relation": relation, "pairs": len(group),
            "identities": int(group.identity.nunique()) if len(group) else 0,
            "median_cosine": float(group.cosine_similarity.median()) if len(group) else None,
            "mean_cosine": float(group.cosine_similarity.mean()) if len(group) else None,
            f"in_knn_k{budget}_rate": float(group[f"in_knn_k{budget}"].mean()) if len(group) else None})
    return {"audited_calibration_images": len(merged), "audit_images_outside_calibration": len(reviews) - len(merged),
            "crosstabs": rows, "risk_auc": aucs, "flank_pairs": pairs, "flank_summary": flank_summary}


# --- Report ---------------------------------------------------------------------------------

def _plot_abstention(path, summary_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [r["fraction"] * 100 for r in summary_rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in ((axes[0], "fully_connected_multi_image_identities",
                            "Fully connected multi-image identities"),
                           (axes[1], "oracle_fragments", "Oracle fragments")):
        ax.plot(x, [r[f"risk_{key}"] for r in summary_rows], marker="o", label="drop highest risk")
        ax.plot(x, [r[f"random_mean_{key}"] for r in summary_rows], marker="o", label="drop random (mean)")
        ax.fill_between(x, [r[f"random_min_{key}"] for r in summary_rows],
                        [r[f"random_max_{key}"] for r in summary_rows], alpha=.2)
        ax.set_xlabel("Images abstained (%)")
        ax.set_title(title)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"Truth-filtered k={PRIMARY_BUDGET} connectivity after abstention (diagnostic, not a count)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _table_html(rows, keys):
    body = "<table><tr>" + "".join(f"<th>{html.escape(k)}</th>" for k in keys) + "</tr>"
    for row in rows:
        body += "<tr>" + "".join(f"<td>{html.escape(_fmt(row.get(k)))}</td>" for k in keys) + "</tr>"
    return body + "</table>"


def _fmt(value):
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def write_coverage_report(out, inputs, results, *, worst_identities=8, fragments_per_identity=8):
    from PIL import Image
    out = Path(out)
    images, fragments = results["images"], results["fragments"]
    write_json(out / "summary.json", {k: v for k, v in results.items()
                                      if k not in ("images", "fragments", "abstention_rows", "audit")}
               | {"audit": {k: v for k, v in results["audit"].items() if k != "flank_pairs"}, "notes": NOTES})
    _write_table(out / "image_fragments.csv", images.to_dict("records"), list(images.columns))
    _write_table(out / "fragments.csv", fragments.to_dict("records"), list(fragments.columns))
    _write_table(out / "abstention_runs.csv", results["abstention_rows"], list(results["abstention_rows"][0]))
    _write_table(out / "abstention_curve.csv", results["abstention"], list(results["abstention"][0]))
    _write_table(out / "candidate_variants.csv", results["candidate_variants"], list(results["candidate_variants"][0]))
    _write_table(out / "audit_crosstabs.csv", results["audit"]["crosstabs"], list(results["audit"]["crosstabs"][0]))
    pairs = results["audit"]["flank_pairs"]
    _write_table(out / "audit_flank_pairs.csv", pairs.to_dict("records"), list(pairs.columns) or ["identity"])
    _plot_abstention(out / "abstention_curve.png", results["abstention"])
    body = ("<p>Calibration only. Every edge set and the risk score are label-free; truth is used only to "
            "evaluate. Oracle fragments are diagnostics, never animal counts.</p>")
    body += "<h2>H1: abstention on highest-risk images vs random</h2><img src='abstention_curve.png' alt='Abstention curve'>"
    body += _table_html(results["abstention"], ["fraction", "dropped_images", "risk_fully_connected_identities",
        "random_mean_fully_connected_identities", "risk_fully_connected_multi_image_identities",
        "random_mean_fully_connected_multi_image_identities", "risk_identities_without_retained_images",
        "risk_direct_edge_precision", "random_mean_direct_edge_precision"])
    body += "<h2>H3: encounter links</h2>" + _table_html([{k: v for k, v in results["encounters"].items()
        if not isinstance(v, (dict, list))}], [k for k, v in results["encounters"].items() if not isinstance(v, (dict, list))])
    body += "<h2>H4: candidate variants</h2>" + _table_html(results["candidate_variants"], ["variant", "edges",
        "direct_edge_precision", "fully_connected_identities", "fully_connected_multi_image_identities",
        "oracle_fragments", "retained_images_without_correct_neighbour"])
    body += "<h2>H1/H2: audited calibration images</h2>" + _table_html(results["audit"]["crosstabs"],
        ["field", "value", "images", "has_correct_candidate_neighbour_rate", "in_largest_fragment_rate",
         "mean_identifiability_risk"])
    body += "<h3>Risk score vs human labels (AUC)</h3>" + _table_html(
        [{"field": k, **v} for k, v in results["audit"]["risk_auc"].items()], ["field", "positive_values", "images", "auc"])
    body += "<h3>Same-identity audited pairs by flank</h3>" + _table_html(results["audit"]["flank_summary"],
        list(results["audit"]["flank_summary"][0]))
    body += "<h2>Most fragmented identities (one lowest-risk image per fragment)</h2>"
    thumbs = out / "thumbs"
    thumbs.mkdir()
    counts = fragments.groupby("identity").size().sort_values(ascending=False, kind="stable")
    path_of = dict(zip(inputs.image_ids, inputs.frame.path))
    sha_of = dict(zip(inputs.image_ids, inputs.frame.image_sha256)) if "image_sha256" in inputs.frame else {}
    for identity in counts.index[:worst_identities]:
        group = images[(images.identity == identity) & images.fragment_rank.notna()]
        body += f"<h3>{html.escape(identity)}: {int(counts[identity])} fragments</h3><div class='row'>"
        for rank in sorted(group.fragment_rank.unique())[:fragments_per_identity]:
            members = group[group.fragment_rank == rank].sort_values(["identifiability_risk", "image_id"])
            rid = members.image_id.iloc[0]
            if sha_of.get(rid) and file_digest(path_of[rid]) != sha_of[rid]:
                raise ValueError(f"Review image changed since vector validation: {rid}")
            name = fingerprint(rid)[:24] + ".jpg"
            with Image.open(path_of[rid]) as original:
                tile = original.convert("RGB")
                tile.thumbnail((220, 220))
                tile.save(thumbs / name, quality=88)
            caption = (f"fragment {int(rank)}: {len(members)} img; risk {members.identifiability_risk.iloc[0]:.2f}; "
                       f"{members.camera.iloc[0]}")
            body += (f"<figure><img src='thumbs/{name}' alt='{html.escape(caption, quote=True)}'>"
                     f"<figcaption>{html.escape(caption)}</figcaption></figure>")
        body += "</div>"
    body += "<h2>Limitations</h2><ul>" + "".join(f"<li>{html.escape(n)}</li>" for n in NOTES) + "</ul>"
    (out / "index.html").write_text("<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>CzechLynx coverage diagnostics</title><style>"
        "body{font:15px system-ui;max-width:1300px;margin:30px auto;padding:16px}"
        "table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #bbb;padding:5px 8px}"
        ".row{display:flex;flex-wrap:wrap;gap:10px}.row figure{margin:4px;width:200px}img{max-width:100%}"
        "figcaption{font-size:12px}</style><h1>CzechLynx candidate-coverage and identifiability diagnostics</h1>"
        + body + "</html>")
    return {str(p.relative_to(out)): file_digest(p) for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "manifest.json"}
