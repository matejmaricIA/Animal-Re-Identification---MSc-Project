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


def plot_comparison(results: Dict[str, Dict[str, float]], output_path: str = None) -> str:
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

    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Evaluation Metrics by Dataset")
    ax.legend()
    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join("../evaluations", "evaluation_comparison.png")
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    data = load_evaluations()
    out_file = plot_comparison(data)
    print(f"Saved comparison plot to {out_file}")