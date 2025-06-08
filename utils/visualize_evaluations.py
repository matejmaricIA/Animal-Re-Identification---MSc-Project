import os
import sys
import json
from typing import Dict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import subprocess

import pandas as pd
import seaborn as sns


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

EVAL_ROOT = "../evaluations"

def select_tag(root):
    tags = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    tags.sort()
    if not tags:
        raise RuntimeError(f"No tag folders found in {root}")

    print("What tag do you wish to visualize?")
    for i, t in enumerate(tags):
        print(f"{i}: {t}")

    while True:
        try:
            idx = int(input("Enter number: "))
            if 0 <= idx < len(tags):
                return tags[idx]
        except ValueError:
            pass
        print("Invalid choice, try again.")

def load_evaluations(tag, root=EVAL_ROOT):
    """Load evaluation metrics from all *_evaluation.json files within ``directory``.

    The search is performed recursively to allow evaluation files to be organised
    in subfolders (e.g. configuration tags).
    """

    directory = os.path.join(root, tag)
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Folder '{directory}' not found")

    results = {}
    for fname in os.listdir(directory):
        if fname.endswith("_evaluation.json"):
            dataset = fname.replace("_evaluation.json", "")
            with open(os.path.join(directory, fname), "r") as f:
                data = json.load(f)

            cm = data.get("classification_metrics", {})
            f1_score = None
            if isinstance(cm, dict):
                weighted = cm.get("weighted avg", {})
                f1_score = weighted.get("f1-score")

            results[dataset] = {
                "accuracy": data.get("accuracy"),
                "top_n_accuracy": data.get("top_n_accuracy"),
                "f1_score": f1_score,
            }
    return results

def load_dataset_statistics(tag, root=EVAL_ROOT):
    directory = os.path.join(root, tag)
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Folder '{directory}' not found")

    stats = {}
    for fname in os.listdir(directory):
        if fname.endswith("_evaluation.json"):
            dataset = fname.replace("_evaluation.json", "")
            with open(os.path.join(directory, fname), "r") as f:
                data = json.load(f)
            cm = data.get("classification_metrics", {})
            classes = [k for k in cm.keys() if k not in ("accuracy", "macro avg", "weighted avg")]
            stats[dataset] = {
                "num_classes": len(classes),
                "num_samples": int(sum(cm[c]["support"] for c in classes if isinstance(cm[c], dict))),
            }
    return stats

def plot_comparison(results, out_path):
    if not results:
        raise ValueError("No evaluation results found")

    ds = list(results.keys())
    acc = [results[d]["accuracy"] for d in ds]
    topn = [results[d]["top_n_accuracy"] for d in ds]
    f1  = [results[d]["f1_score"] for d in ds]

    x = np.arange(len(ds))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w, acc,  w, label="Accuracy")
    ax.bar(x,     topn, w, label="Top-N Acc")
    ax.bar(x + w, f1,  w, label="Weighted F1")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x, ds, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Evaluation metrics by dataset")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

def plot_dataset_statistics(stats, out_path):
    if not stats:
        raise ValueError("No dataset stats found")

    ds = list(stats.keys())
    samples = [stats[d]["num_samples"] for d in ds]
    classes = [stats[d]["num_classes"] for d in ds]
    x = np.arange(len(ds))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.bar(x, samples, alpha=0.7)
    ax1.set_ylabel("Images")
    ax1.set_title("Dataset size")
    ax1.set_xticks(x, ds, rotation=45, ha="right")

    ax2.bar(x, classes, alpha=0.7, color="tab:orange")
    ax2.set_ylabel("Classes")
    ax2.set_title("Class count")
    ax2.set_xticks(x, ds, rotation=45, ha="right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

def generate_table_image(results, stats, out_path):
    rows = []
    for ds, m in results.items():
        rows.append({
            "Dataset"    : ds,
            "Accuracy"   : f"{m['accuracy']:.3f}" if m["accuracy"] else "—",
            "Top-N Acc." : f"{m['top_n_accuracy']:.3f}" if m["top_n_accuracy"] else "—",
            "F1 Score"   : f"{m['f1_score']:.3f}" if m["f1_score"] else "—",
            "Samples"    : stats[ds]["num_samples"],
            "Classes"    : stats[ds]["num_classes"],
        })

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.3 + 1))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    tag = select_tag(EVAL_ROOT)

    results = load_evaluations(tag)
    stats   = load_dataset_statistics(tag)

    vis_dir = os.path.join(EVAL_ROOT, tag, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    plot_comparison(results, os.path.join(vis_dir, "evaluation_comparison.png"))
    plot_dataset_statistics(stats, os.path.join(vis_dir, "dataset_statistics.png"))
    generate_table_image(results, stats, os.path.join(vis_dir, "results_table.png"))

    print(f"Visualisations saved in {vis_dir}")