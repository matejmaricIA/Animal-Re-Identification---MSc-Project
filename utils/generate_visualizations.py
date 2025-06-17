#!/usr/bin/env python3
"""Generate comparison plots and summary table for multiple experiment folders."""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Root directory that contains the experiment folders (one folder per run)
EVAL_ROOT = "../evaluations"
VIS_ROOT = os.path.join(EVAL_ROOT, "visualizations")

def _format_method(method: str | None) -> str:
    if method is None:
        return "—"
    # Break long method strings by replacing '_' with('\n') => multi‑line cell
    return method.replace("_", "\n")


def select_tags(root: str) -> List[str]:
    """Interactively let the user select *multiple* folders under *root*.

    The user can enter the number of a folder to toggle its selection. Press 0
    to stop; the function returns the list of selected tag names (sorted)."""

    tags = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    tags.sort()
    if not tags:
        raise RuntimeError(f"No tag folders found in {root}")

    selected: set[str] = set()

    while True:
        os.system("clear" if os.name == "posix" else "cls")
        print("Select experiment folders (0 = done):\n")
        for i, t in enumerate(tags, 1):
            mark = "*" if t in selected else " "
            print(f" {i:2d}. [{mark}] {t}")
        try:
            idx = int(input("\nEnter number: "))
        except ValueError:
            continue

        if idx == 0:
            if not selected:
                print("Nothing selected – exiting.")
                sys.exit(0)
            return sorted(selected)
        if 1 <= idx <= len(tags):
            tag = tags[idx - 1]
            if tag in selected:
                selected.remove(tag)
            else:
                selected.add(tag)

# JSON extraction

_JSON_GLOB = re.compile(r"_evaluation\.json$")
_CONFIG_GMM = re.compile(r"GMM_(\d+)")
_CONFIG_GV = re.compile(r"gv_(True|False)")
_CONFIG_LG = re.compile(r"lg_(True|False)")
_CONFIG_METHOD = re.compile(r"closed_([^_]+(?:_[^_]+)*)_PCA")


def _parse_folder_config(tag: str) -> dict:
    """Extract config flags from *tag* using simple regexes."""
    gmm_match = _CONFIG_GMM.search(tag)
    gv_match = _CONFIG_GV.search(tag)
    lg_match = _CONFIG_LG.search(tag)
    method_match = _CONFIG_METHOD.search(tag)
    return {
        "gmm_components": int(gmm_match.group(1)) if gmm_match else None,
        "gv_used": (gv_match.group(1) == "True") if gv_match else None,
        "lg_used": (lg_match.group(1) == "True") if lg_match else None,
        "method": method_match.group(1) if method_match else None,
    }


def load_evaluations(tag: str, root: str = EVAL_ROOT) -> Tuple[dict, dict]:
    """Return (results, stats) dictionaries for every *_evaluation.json* in *tag*."""

    directory = os.path.join(root, tag)
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Folder '{directory}' not found")

    cfg = _parse_folder_config(tag)
    results: Dict[str, dict] = {}
    stats: Dict[str, dict] = {}

    for fname in os.listdir(directory):
        if not _JSON_GLOB.search(fname):
            continue

        dataset = fname.replace("_evaluation.json", "")
        with open(os.path.join(directory, fname), "r") as f:
            data = json.load(f)

        # ---------- metrics ---------- #
        cm = data.get("classification_metrics", {})
        f1_score = None
        if isinstance(cm, dict):
            weighted = cm.get("weighted avg", {})
            f1_score = weighted.get("f1-score")

        results[dataset] = {
            "accuracy": data.get("accuracy"),
            "top_n_accuracy": data.get("top_n_accuracy"),
            "f1_score": f1_score,
            "train_time": data.get("eval_runtime_sec"),
            **cfg,
        }

        # ---------- stats ---------- #
        classes = [k for k in cm.keys() if k not in ("accuracy", "macro avg", "weighted avg")]
        stats[dataset] = {
            "num_classes": len(classes),
            "num_samples": int(sum(cm[c]["support"] for c in classes if isinstance(cm[c], dict))),
        }

    return results, stats

# aggregation logic


def aggregate_best(tags: List[str]) -> Tuple[dict, dict]:
    """Return best‑accuracy metrics/stats for every dataset across *tags*."""

    best_results: Dict[str, dict] = {}
    best_stats: Dict[str, dict] = {}

    for tag in tags:
        res, sta = load_evaluations(tag)
        for ds, m in res.items():
            current = best_results.get(ds)
            if current is None or (m["accuracy"] is not None and m["accuracy"] > (current["accuracy"] or 0)):
                best_results[ds] = m | {"tag": tag}  # keep origin for debugging
                best_stats[ds] = sta.get(ds, {})

    if not best_results:
        raise RuntimeError("No evaluation results found in the selected folders.")

    return best_results, best_stats

# visualisation

def plot_comparison(results: dict, out_path: str) -> None:
    ds = list(results.keys())
    acc  = [results[d]["accuracy"]        for d in ds]
    topn = [results[d]["top_n_accuracy"]   for d in ds]
    f1   = [results[d]["f1_score"]         for d in ds]

    x = np.arange(len(ds))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w, acc,  w, label="Accuracy")
    ax.bar(x,     topn, w, label="Top‑N Acc")
    ax.bar(x + w, f1,   w, label="Weighted F1")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x, ds, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Best evaluation metrics by dataset (selected folders)")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dataset_statistics(stats: dict, out_path: str) -> None:
    ds = list(stats.keys())
    samples = [stats[d]["num_samples"]  for d in ds]
    classes = [stats[d]["num_classes"] for d in ds]
    x = np.arange(len(ds))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.bar(x, samples, alpha=0.7)
    ax1.set_ylabel("Images")
    ax1.set_title("Dataset size (best‑accuracy runs)")
    ax1.set_xticks(x, ds, rotation=45, ha="right")

    ax2.bar(x, classes, alpha=0.7)
    ax2.set_ylabel("Classes")
    ax2.set_title("Class count")
    ax2.set_xticks(x, ds, rotation=45, ha="right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_table_image(results: dict, stats: dict, out_path: str) -> None:
    rows = []
    for ds, m in results.items():
        rows.append({
            "Dataset"      : ds,
            "Accuracy"     : f"{m['accuracy']:.3f}"        if m["accuracy"] is not None else "—",
            "Top‑N Acc."   : f"{m['top_n_accuracy']:.3f}" if m["top_n_accuracy"] is not None else "—",
            "F1 Score"     : f"{m['f1_score']:.3f}"       if m["f1_score"] is not None else "—",
            "Samples"      : stats[ds]["num_samples"],
            "Classes"      : stats[ds]["num_classes"],
            "Train time [s]": f"{m['train_time']:.1f}"      if m["train_time"] is not None else "—",
            "Method"        : _format_method(m["method"]),
            "GMM"          : m["gmm_components"],
            "GV": m["gv_used"],
            "LightGlue"    : m["lg_used"],
        })

    df = pd.DataFrame(rows)
    
    df = df.sort_values(by="Accuracy", ascending=False, na_position="last")
    
    #fig, ax = plt.subplots(figsize=(12, len(df) * 0.35 + 1))
    #ax.axis("off")
    
    nrows = len(df)
    fig_width = max(9, 1.0 + 0.6 * len(df.columns))  # heuristic to keep within bounds
    fig, ax = plt.subplots(figsize=(fig_width, nrows * 0.33 + 1))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.5)
    
    dataset:col = df.columns.get_loc("Dataset")
    for (row, col), cell in tbl.get_celld().items():
        if col == dataset:                      
            cell.set_width(cell.get_width() * 1.4)
    

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)



if __name__ == "__main__":
    tags = select_tags(EVAL_ROOT)
    print(f"\nSelected folders: {', '.join(tags)}")

    results, stats = aggregate_best(tags)

    os.makedirs(VIS_ROOT, exist_ok=True)

    plot_comparison(results,        os.path.join(VIS_ROOT, "evaluation_comparison.png"))
    plot_dataset_statistics(stats,  os.path.join(VIS_ROOT, "dataset_statistics.png"))
    generate_table_image(results,   stats, os.path.join(VIS_ROOT, "results_table.png"))

    print(f"\nVisualisations saved to {VIS_ROOT}\n")
