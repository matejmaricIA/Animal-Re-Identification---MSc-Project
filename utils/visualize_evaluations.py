import os
import sys
import json
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constants import EVALUATION_DIR


def load_evaluations(directory: str = "../evaluations") -> Dict[str, Dict[str, float]]:
    """Load evaluation metrics from all *_evaluation.json files in the directory."""
    results = {}
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Evaluation directory '{directory}' not found")

    for fname in os.listdir(directory):
        if fname.endswith("_evaluation.json"):
            dataset = fname.replace("_evaluation.json", "")
            path = os.path.join(directory, fname)
            with open(path, "r") as f:
                data = json.load(f)
            # Extract relevant metrics
            accuracy = data.get("accuracy")
            top_n_accuracy = data.get("top_n_accuracy")
            f1_score = None
            cm = data.get("classification_metrics", {})
            if isinstance(cm, dict):
                weighted = cm.get("weighted avg", {})
                f1_score = weighted.get("f1-score")
            results[dataset] = {
                "accuracy": accuracy,
                "top_n_accuracy": top_n_accuracy,
                "f1_score": f1_score,
            }
    return results

def load_dataset_statistics(directory: str = "../evaluations") -> Dict[str, Dict[str, int]]:
    """Extract dataset size (samples) and class count from evaluation files."""
    stats = {}
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Evaluation directory '{directory}' not found")

    for fname in os.listdir(directory):
        if fname.endswith("_evaluation.json"):
            dataset = fname.replace("_evaluation.json", "")
            path = os.path.join(directory, fname)
            with open(path, "r") as f:
                data = json.load(f)
            cm = data.get("classification_metrics", {})
            classes = [k for k in cm.keys() if k not in ("accuracy", "macro avg", "weighted avg")]
            num_classes = len(classes)
            num_samples = int(sum(cm[c]["support"] for c in classes if isinstance(cm[c], dict)))
            stats[dataset] = {
                "num_classes": num_classes,
                "num_samples": num_samples,
            }
    return stats

def plot_comparison(results: Dict[str, Dict[str, float]], output_path: str = None, dataset_stats) -> str:
    """Create a bar chart comparing metrics across datasets."""
    if not results:
        raise ValueError("No evaluation results found")

    datasets = list(results.keys())
    accuracies = [results[d]["accuracy"] for d in datasets]
    top_n = [results[d]["top_n_accuracy"] for d in datasets]
    f1_scores = [results[d]["f1_score"] for d in datasets]

    x = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, accuracies, width, label="Accuracy")
    ax.bar(x, top_n, width, label="Top-N Accuracy")
    ax.bar(x + width, f1_scores, width, label="Weighted F1")

    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Evaluation Metrics by Dataset")
    ax.legend()
    if dataset_stats:
        for idx, ds in enumerate(datasets):
            n_cls = dataset_stats.get(ds, {}).get("num_classes")
            if n_cls is not None:
                ax.text(
                    idx,
                    1.02,
                    f"n_classes = {n_cls}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join("../evaluations", "evaluation_comparison.png")
    fig.savefig(output_path)
    plt.close(fig)
    return output_path

def plot_dataset_statistics(stats: Dict[str, Dict[str, int]], output_path: str = None) -> str:
    """Create bar charts for dataset size and class count."""
    if not stats:
        raise ValueError("No dataset statistics available")

    datasets = list(stats.keys())
    num_samples = [stats[d]["num_samples"] for d in datasets]
    num_classes = [stats[d]["num_classes"] for d in datasets]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].bar(datasets, num_samples)
    axes[0].set_ylabel("Images")
    axes[0].set_title("Dataset Size")
    axes[0].set_xticklabels(datasets, rotation=45, ha="right")

    axes[1].bar(datasets, num_classes, color="orange")
    axes[1].set_ylabel("Classes")
    axes[1].set_title("Number of Classes")
    axes[1].set_xticklabels(datasets, rotation=45, ha="right")

    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join("../evaluations", "dataset_statistics.png")
    fig.savefig(output_path)
    plt.close(fig)
    return output_path



if __name__ == "__main__":
    data = load_evaluations()
    out_file = plot_comparison(data)
    print(f"Saved comparison plot to {out_file}")
    
    stats = load_dataset_statistics()
    stats_file = plot_dataset_statistics(stats)
    print(f"Saved dataset statistics plot to {stats_file}")

    