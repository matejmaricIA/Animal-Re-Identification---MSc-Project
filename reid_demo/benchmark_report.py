"""Portable, read-only benchmark reports with deterministic failure examples."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from reid_demo.benchmark import write_json


def global_model_name(spec):
    names = {"pretrained_megadescriptor_l_384": "MegaDescriptor-L-384",
             "pretrained_miewid_msv3": "MiewID-msv3", "scratch_checkpoint": "MegaDescriptor checkpoint"}
    return names.get(spec.get("kind"), spec.get("kind", "Global model"))


def select_examples(predictions, *, limit=24, seed=42):
    """Sample successes/errors/unknowns and changes relative to the global baseline."""
    rng = np.random.default_rng(seed)
    pools = {}
    base = {r["query_image_id"]: r for r in predictions.get("global", [])}
    for signal, rows in predictions.items():
        pools[signal + ":correct"] = [r["query_image_id"] for r in rows if r["known"] and r["top1_correct"]]
        pools[signal + ":wrong"] = [r["query_image_id"] for r in rows if r["known"] and not r["top1_correct"]]
        pools[signal + ":unknown"] = [r["query_image_id"] for r in rows if not r["known"]]
        if signal != "global":
            for name, before, after in (("improved", False, True), ("worsened", True, False)):
                pools[signal + ":" + name] = [
                    r["query_image_id"] for r in rows
                    if r["known"] and r["query_image_id"] in base
                    and base[r["query_image_id"]]["top1_correct"] == before
                    and r["top1_correct"] == after]
    chosen = {}
    for key in sorted(pools, key=lambda k: (not k.endswith(("improved", "worsened")), k)):
        ids = np.asarray(sorted(set(pools[key])), dtype=object)
        rng.shuffle(ids)
        for qid in ids[:3]:
            if len(chosen) < limit or qid in chosen:
                chosen.setdefault(str(qid), []).append(key)
    return [{"image_id": k, "categories": v} for k, v in chosen.items()]


def _page(title, body):
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>{html.escape(title)}</title><meta name="viewport" content="width=device-width">
<style>
body{{font:16px system-ui,sans-serif;max-width:1300px;margin:32px auto;padding:0 20px;color:#182820}}
table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}
.strip{{display:flex;gap:10px;flex-wrap:wrap}}figure{{margin:4px;width:170px}}img{{width:160px;height:150px;object-fit:contain;background:#eee}}
.ok img{{border:3px solid #237b48}}.bad img{{border:3px solid #b64736}}figcaption{{font-size:13px;overflow-wrap:anywhere}}
article{{border-top:1px solid #bbb;padding:22px 0}}small{{color:#566}}a{{color:#176843}}
</style><body><h1>{html.escape(title)}</h1>{body}</body></html>"""


def write_report(out_dir, manifest, metrics, predictions, metadata, *, clusters=None, limit=24):
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "manifest.json", manifest)
    write_json(root / "metrics.json", metrics)
    flat = []
    for signal, values in metrics.items():
        flat.append({"signal": signal, **{k: v for k, v in values.items()
                                        if not isinstance(v, (dict, list))}})
    pd.DataFrame(flat).to_csv(root / "comparison.csv", index=False)
    rows = [{"signal": signal, **r} for signal, group in predictions.items() for r in group]
    csv_rows = [{k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()}
                for r in rows]
    pd.DataFrame(csv_rows).to_csv(root / "predictions.csv", index=False)
    write_json(root / "predictions.json", predictions)
    examples = select_examples(predictions, limit=limit, seed=manifest.get("seed", 42))
    write_json(root / "examples.json", examples)
    indexed = metadata.set_index("image_id")
    thumbs = root / "thumbs"
    thumbs.mkdir(exist_ok=True)

    def tile(image_id, label, correctness=None, prefix=""):
        row = indexed.loc[image_id]
        filename = str(image_id) + ".jpg"
        path = thumbs / filename
        if not path.exists():
            with Image.open(row["path"]) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((320, 300))
                im.save(path, quality=85)
        css = "ok" if correctness is True else ("bad" if correctness is False else "")
        return (f'<figure class="{css}"><img loading="lazy" src="{prefix}thumbs/{html.escape(filename)}" '
                f'alt="{html.escape(str(label))}"><figcaption>{html.escape(str(label))}</figcaption></figure>')

    def number(value, percent=False):
        if value is None:
            return "—"
        return f"{100 * value:.1f}%" if percent else f"{value:.3f}"

    table = "<table><tr><th>Method</th><th>Top 1</th><th>Top 5</th><th>mAP@K</th><th>Known / unseen queries</th><th>GV attempts</th></tr>"
    for signal, values in metrics.items():
        table += (f"<tr><td>{html.escape(signal)}</td>"
                  f"<td>{number(values.get('top1'), True)}</td>"
                  f"<td>{number(values.get('top5'), True)}</td>"
                  f"<td>{number(values.get('map_at_k'), True)}</td>"
                  f"<td>{values.get('known_queries', 0)} / {values.get('unknown_queries', 0)}</td>"
                  f"<td>{values.get('gv_attempted_pairs', 0)}</td></tr>")
    table += "</table>"
    annotation = {
        "bbox": "Provided animal masks determine the crops.",
        "masked": ("Provided segmentation masks replace background pixels with grey. "
                   "Global and local features use these segmented images; local keypoints outside the animal are discarded. "
                   "This is annotation-assisted evaluation, not a test of automatic segmentation."),
    }.get(manifest.get("input_mode"), "Original camera-trap frames.")
    if manifest.get("gv_foreground_mask"):
        annotation += (" Global embeddings and Fisher retrieval use unchanged bbox images and features. "
                       "Provided masks restrict keypoint centres only for geometric verification; "
                       "descriptor patches can still include surrounding background. "
                       "This is annotation-assisted evaluation.")
    description = (f"<p>{html.escape(manifest.get('stage', ''))} evaluation. {annotation} "
                   "Identity labels are used for scoring. Coloured candidate borders show match correctness.</p>"
                   "<p>Top-1/top-5 and mAP use queries with a gallery match. Unseen animals are reported separately. "
                   "Groups in the catalogue are proposals, not verified animal counts.</p>")
    partition = manifest.get("partition")
    if partition:
        label = ("Selection validation: thresholds are frozen from calibration; this remains development data."
                 if partition == "selection" else "Calibration: rejection rates are in-sample calibration figures.")
        description += "<p><strong>" + html.escape(label) + "</strong></p>"
        description += ("<p>Fitting, calibration and selection use different identities. "
                        "Official test data remains reserved. Upstream model training exposure is unverified.</p>")
        description += "<h2>Development partitions</h2><table><tr><th>Pool</th><th>Images</th><th>Identities</th><th>Gallery</th><th>Known / unknown queries</th><th>Excluded</th></tr>"
        for role, pool in manifest.get("protocol_inventory", {}).items():
            inventory = pool["inventory"]
            description += (f"<tr><td>{html.escape(role)}</td><td>{inventory['images']}</td>"
                            f"<td>{inventory['identities']}</td><td>{pool.get('gallery', {}).get('images', '—')}</td>"
                            f"<td>{pool.get('known_queries', '—')} / {pool.get('unknown_queries', '—')}</td>"
                            f"<td>{len(pool.get('excluded', []))}</td></tr>")
        description += "</table><p>Protocol: " + html.escape(manifest["protocol"]["fingerprint"]) + "</p>"
        if manifest.get("threshold_provenance"):
            description += "<p>Calibration artifact SHA-256: " + html.escape(manifest["threshold_provenance"]["sha256"]) + "</p>"
    global_spec = manifest.get("global_model")
    if global_spec:
        details = f"Global embeddings: {global_model_name(global_spec)}. "
        if global_spec.get("dimension"):
            details += f"{global_spec['dimension']} dimensions. "
        details += global_spec.get("transform", "")
        if global_spec.get("revision"):
            details += f". Model revision: {global_spec['revision']}"
        description += "<p>" + html.escape(details) + "</p>"
    local = manifest.get("local_features")
    if local:
        names = {"disk": "DISK", "superpoint": "SuperPoint", "aliked": "ALIKED"}
        methods = " + ".join(names.get(method, method) for method in local["methods"])
        weights = ", ".join(f"{weight:.3f}" for weight in local["weights"])
        gv = names.get(local.get("gv_features"), local.get("gv_features"))
        settings = (f"Fisher: {methods}. Weights: {weights}. "
                    f"PCA: {local['pca_dim']} dimensions per extractor. "
                    f"GMM: {local['gmm_components_per_extractor']} components per extractor. "
                    f"Keypoints: up to {local['max_keypoints']} per image per extractor. ")
        if len(local["methods"]) > 1:
            settings += "Fisher vectors are combined by weighted sum and L2 normalisation. "
        settings += f"Geometric verification: {gv} + LightGlue." if gv else "Geometric verification disabled."
        description += "<p>" + html.escape(settings) + "</p>"
        counts = local.get("gv_mask_filter_counts", local.get("mask_filter_counts", {}))
        if local.get("gv_mask_filter_counts"):
            description += "<p>Foreground retention for geometric verification only (Fisher features unchanged):</p>"
        if counts:
            description += "<table><tr><th>Extractor</th><th>Detected points</th><th>Inside mask</th><th>Images with no retained points</th></tr>"
            for method, count in counts.items():
                description += (f"<tr><td>{html.escape(names.get(method, method))}</td>"
                                f"<td>{count['raw_keypoints']}</td><td>{count['foreground_keypoints']}</td>"
                                f"<td>{count['empty_images']} / {count['images']}</td></tr>")
            description += "</table>"
    body = description + table + '<p><a href="comparison.csv">Download comparison</a> · <a href="predictions.csv">Predictions</a>'
    if clusters is not None and not clusters.empty:
        body += ' · <a href="catalogue/index.html">Proposed groups</a>'
    body += "</p>"
    comparison = manifest.get("global_comparison")
    if comparison:
        write_json(root / "global_model_comparison.json", comparison)
        body += ("<h2>Global model comparison</h2><p>Same prepared images, gallery and queries. "
                 "Accuracy uses known queries before rejection. Recall@K means a correct identity appears "
                 "among the first K candidates.</p>")
        body += "<table><tr><th>Model</th><th>Top 1</th><th>Top 5</th><th>mAP@K</th><th>Recall@10</th><th>Recall@20</th><th>Recall@50</th></tr>"
        for spec, values in ((comparison["reference"]["global_model"], comparison["reference_metrics"]),
                             (comparison["current_model"], comparison["current_metrics"])):
            body += "<tr><td>" + html.escape(global_model_name(spec)) + "</td>"
            for key in ("top1", "top5", "map_at_k", "recall_at_10", "recall_at_20", "recall_at_50"):
                body += f"<td>{number(values.get(key), True)}</td>"
            body += "</tr>"
        body += (f"</table><p>Corrected {comparison['fixed_queries']} known queries; "
                 f"worsened {comparison['worsened_queries']}. "
                 '<a href="global_model_comparison.json">Comparison details and query IDs</a></p>')

    if any("threshold" in values for values in metrics.values()):
        rejection_note = (
            "Selection uses frozen calibration thresholds; no thresholds were fitted on these queries."
            if partition == "selection" else
            "Thresholds are calibrated on development queries. Development rejection rates are calibration figures; test rates use the frozen thresholds.")
        body += "<h2>Unseen-animal rejection</h2><p>" + rejection_note + "</p>"
        body += "<table><tr><th>Method</th><th>Threshold</th><th>Unseen falsely accepted</th><th>Known falsely rejected</th><th>Known correctly accepted</th></tr>"
        for signal, values in metrics.items():
            body += (f"<tr><td>{html.escape(signal)}</td><td>{number(values.get('threshold'))}</td>"
                     f"<td>{number(values.get('unknown_false_accept_rate'), True)}</td>"
                     f"<td>{number(values.get('known_false_reject_rate'), True)}</td>"
                     f"<td>{number(values.get('known_correct_accept_rate'), True)}</td></tr>")
        body += "</table>"
    diagnostics = manifest.get("diagnostics")
    if diagnostics:
        if not (root / "diagnostics.json").exists():
            write_json(root / "diagnostics.json", diagnostics)
        body += ('<h2>Appearance, camera and source diagnostics</h2>'
                 '<p>Near-monochrome is a colour proxy, not a verified night/infrared label. '
                 'Camera groups describe available correct references, not camera-disjoint training. '
                 '<a href="diagnostics.json">Download diagnostics and measurements</a></p>')
        for signal, groups in diagnostics["stages"].items():
            body += "<h3>" + html.escape(signal) + "</h3>"
            body += ("<table><tr><th>Group</th><th>Known / unknown queries</th><th>Identities / encounters</th>"
                     "<th>Top-1</th><th>Top-5</th><th>mAP</th><th>Recall 10 / 20 / 50</th>"
                     "<th>Union recall / size</th><th>Accepted precision / count</th><th>Correct known coverage</th>"
                     "<th>Unknown false accepts</th></tr>")
            for group, values in groups.items():
                recall = " / ".join(number(values["recall_at"].get(str(k)), True) for k in (10, 20, 50))
                body += (f"<tr><td>{html.escape(group)}</td>"
                         f"<td>{values['known_queries']} / {values['unknown_queries']}</td>"
                         f"<td>{values['identities']} / {values['encounters']}</td>"
                         f"<td>{number(values['top1'], True)}</td><td>{number(values['top5'], True)}</td>"
                         f"<td>{number(values['map_at_k'])}</td><td>{recall}</td>"
                         f"<td>{number(values['shortlist_union_recall'], True)} / "
                         f"{values['shortlist_size_min']}–{values['shortlist_size_max']}</td>"
                         f"<td>{number(values['accepted_precision'], True)} / {values['accepted']}</td>"
                         f"<td>{number(values['correct_known_coverage'], True)}</td>"
                         f"<td>{values['unknown_false_accepts']}</td></tr>")
            body += "</table>"
        body += "<h3>Foreground measurements</h3><table><tr><th>Measurement</th><th>Measured / missing queries</th><th>P10 / median / P90</th></tr>"
        for name, values in diagnostics["continuous_query_summaries"].items():
            body += (f"<tr><td>{html.escape(name)}</td>"
                     f"<td>{values['measured_queries']} / {values['missing_queries']}</td>"
                     f"<td>{number(values['p10'])} / {number(values['median'])} / {number(values['p90'])}</td></tr>")
        body += "</table>"
    if any("cluster_found_clusters" in values for values in metrics.values()):
        body += "<h2>Grouping the query sample</h2><p>The sample is bounded to control memory. No reference individual count is supplied to clustering. Flank is unknown in this metadata and is not used to separate animals.</p>"
        body += "<table><tr><th>Method / backend</th><th>Images</th><th>Proposed groups / reference individuals</th><th>Unassigned</th><th>Pair precision</th><th>Pair recall</th><th>ARI</th><th>Merges / splits</th></tr>"
        for signal, values in metrics.items():
            body += (f"<tr><td>{html.escape(signal)} / {html.escape(str(values.get('cluster_backend', '')))}</td>"
                     f"<td>{values.get('cluster_images', '—')}</td>"
                     f"<td>{values.get('cluster_found_clusters', '—')} / {values.get('cluster_true_individuals', '—')}</td>"
                     f"<td>{values.get('cluster_unassigned_images', '—')}</td>"
                     f"<td>{number(values.get('cluster_pairwise_precision'), True)}</td>"
                     f"<td>{number(values.get('cluster_pairwise_recall'), True)}</td>"
                     f"<td>{number(values.get('cluster_adjusted_rand_index'))}</td>"
                     f"<td>{values.get('cluster_merge_errors', '—')} / {values.get('cluster_split_errors', '—')}</td></tr>")
        body += "</table>"
    by_signal = {s: {r["query_image_id"]: r for r in group} for s, group in predictions.items()}
    for example in examples:
        qid = example["image_id"]
        identity = str(indexed.loc[qid, "identity"])
        body += "<article><p>" + html.escape(", ".join(example["categories"])) + "</p>"
        body += '<div class="strip">' + tile(qid, "Query • " + identity) + "</div>"
        for signal, lookup in by_signal.items():
            if qid not in lookup:
                continue
            row = lookup[qid]
            body += f"<p>{html.escape(signal)}</p><div class=\"strip\">"
            for rank, cid in enumerate(row["candidate_ids"][:5], 1):
                label = str(indexed.loc[cid, "identity"])
                body += tile(cid, f"{rank}. {label}", label == identity)
            body += "</div>"
        body += "</article>"
    (root / "index.html").write_text(_page("CzechLynx identification benchmark", body))
    if clusters is not None and not clusters.empty:
        clusters.to_csv(root / "clusters.csv", index=False)
        catalogue = root / "catalogue"
        catalogue.mkdir(exist_ok=True)
        cat_body = '<p><a href="../index.html">Benchmark results</a></p><p>Proposed groups; negative labels are unresolved.</p>'
        for (signal, cluster_id), group in clusters.groupby(["signal", "cluster_id"], sort=True):
            if cluster_id < 0:
                continue
            cat_body += (f"<article><h2>{html.escape(signal)} · group {int(cluster_id)}</h2>"
                         f"<p>{len(group)} photos · {group.identity.nunique()} reference identities</p><div class=\"strip\">")
            for row in group.sort_values("image_id").head(16).to_dict("records"):
                cat_body += tile(row["image_id"], row["identity"], prefix="../")
            cat_body += "</div></article>"
        (catalogue / "index.html").write_text(_page("Proposed lynx groups", cat_body))
    return root / "index.html"
