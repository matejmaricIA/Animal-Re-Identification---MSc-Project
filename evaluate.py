from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import json
import os
from constants import EVALUATION_DIR


def evaluate_predictions(predictions, ground_truth, top_n=5):

    y_true = []
    y_pred = []
    top_n_hits = 0  # Count correct predictions within top-N

    for image_id, result in predictions.items():
        true_class = ground_truth[image_id]  # True class for this test image
        predicted_class = result['predicted_class']  # Predicted class
        top_n_matches = result['top_n']  # Top-N matches

        y_true.append(true_class)
        y_pred.append(predicted_class)

        # Check if true class is in the top-N matches
        if any(class_label == true_class for _, class_label in top_n_matches[:top_n]):
            top_n_hits += 1

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    classification_metrics = classification_report(y_true, y_pred, output_dict=True)

    # Compute top-N accuracy
    top_n_accuracy = top_n_hits / len(y_true)

    # Results summary
    results = {
        'accuracy': accuracy,
        'top_n_accuracy': top_n_accuracy,
        'classification_metrics': classification_metrics
    }

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Top-{top_n} Accuracy: {top_n_accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    return results

def save_evaluation_results( results, dataset_name: str, tag: str | None = None, output_dir: str = EVALUATION_DIR) -> str:
    """Save evaluation metrics to a JSON file under data/evaluations.

    Parameters
    ----------
    results : dict
        Evaluation metrics dictionary.
    dataset_name : str
        Name of the dataset/species being evaluated.
    tag : str | None, optional
        Configuration tag (e.g. version and preprocessing settings). When
        provided, results are saved under ``evaluations/<tag>/``.
    output_dir : str, optional
        Base output directory, by default ``EVALUATION_DIR``.

    Returns
    -------
    str
        Path to the saved JSON file.
    """

    if tag:
        output_dir = os.path.join(output_dir, tag)

    # Ensure base and visualization directories exist
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    path = os.path.join(output_dir, f"{dataset_name}_evaluation.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=4, default=str)
        
    print(f"Saved evaluation results to {path}")
    return path