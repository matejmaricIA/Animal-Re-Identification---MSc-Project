import argparse
import itertools
import json
import os
import subprocess
from typing import Dict, List, Optional

import pandas as pd

from constants import (
    N_COMPONENTS_GMM,
    N_COMPONENTS_PCA,
    EVALUATION_DIR,
)

# Configuration spaces
BACKGROUND_OPTIONS = [False, True]
EMBEDDING_MODELS = ["resnet50", "megadescriptor-l-384"]
WEIGHT_COMBINATIONS = [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0)]
GV_METHODS = [None, "RANSAC", "MAGSAC"]


def generate_configs() -> List[Dict[str, Optional[float]]]:
    """Generate all model configuration combinations."""
    configs: List[Dict[str, Optional[float]]] = []

    # Global embedding only
    for model in EMBEDDING_MODELS:
        configs.append(
            {
                "use_global_embedding": True,
                "embedding_model": model,
                "use_fisher": False,
                "use_geometric_verification": False,
            }
        )

    # Fisher vectors only
    configs.append({"use_fisher": True, "use_geometric_verification": False})
    for gv in GV_METHODS[1:]:  # skip None
        configs.append(
            {
                "use_fisher": True,
                "use_geometric_verification": True,
                "gv_method": gv,
            }
        )

    # Fisher + Global with weight combinations
    for model in EMBEDDING_MODELS:
        for w_fisher, w_global in WEIGHT_COMBINATIONS:
            # without GV
            configs.append(
                {
                    "use_fisher": True,
                    "use_global_embedding": True,
                    "embedding_model": model,
                    "w_fisher": w_fisher,
                    "w_global": w_global,
                    "use_geometric_verification": False,
                }
            )
            # with GV methods
            for gv in GV_METHODS[1:]:
                configs.append(
                    {
                        "use_fisher": True,
                        "use_global_embedding": True,
                        "embedding_model": model,
                        "w_fisher": w_fisher,
                        "w_global": w_global,
                        "use_geometric_verification": True,
                        "gv_method": gv,
                    }
                )
    return configs


def build_cmd(
    dataset: str, remove_background: bool, config: Dict[str, Optional[float]]
) -> List[str]:
    cmd = ["python", "main.py", "--train", "--ds", dataset]
    if remove_background:
        cmd.append("--remove_background")

    # Fisher usage
    if not config.get("use_fisher", True):
        cmd.append("--no-use_fisher")
    else:
        cmd.append("--use_fisher")

    # Global embedding usage
    if config.get("use_global_embedding"):
        cmd.append("--use_global_embedding")
        cmd.extend(["--embedding_model", config["embedding_model"]])
    # Geometric verification
    if config.get("use_geometric_verification"):
        cmd.append("--use_geometric_verification")
        gv_method = config.get("gv_method", "RANSAC")
        cmd.extend(["--gv_method", gv_method])
    else:
        cmd.append("--no-use_geometric_verification")

    # Weights
    if config.get("w_fisher") is not None:
        cmd.extend(["--w_fisher", str(config["w_fisher"])])
    if config.get("w_global") is not None:
        cmd.extend(["--w_global", str(config["w_global"])])
    return cmd


def eval_json_path(dataset: str, remove_background: bool, use_gv: bool) -> str:
    tag = (
        f"rmbkg_{remove_background}_tm_{False}_disk"
        f"_PCA_{N_COMPONENTS_PCA}_GMM_{N_COMPONENTS_GMM}"
        f"_gv_{use_gv}_lg_{True}_v1"
    )
    return os.path.join(EVALUATION_DIR, tag, f"{dataset}_evaluation.json")


def extract_worst_classes(metrics: Dict, k: int = 5) -> str:
    cls_metrics = metrics.get("classification_metrics", {})
    per_class = {
        k: v
        for k, v in cls_metrics.items()
        if k not in {"accuracy", "macro avg", "weighted avg"}
    }
    sorted_classes = sorted(per_class.items(), key=lambda kv: kv[1]["f1-score"])[:k]
    return "; ".join(
        [f"{cls}:{vals['f1-score']:.3f}" for cls, vals in sorted_classes]
    )


def run_experiment(dataset: str, remove_background: bool, config: Dict) -> Dict:
    cmd = build_cmd(dataset, remove_background, config)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    path = eval_json_path(
        dataset, remove_background, config.get("use_geometric_verification", False)
    )
    with open(path, "r") as f:
        metrics = json.load(f)

    row = {
        "Dataset": dataset,
        "Remove Background": remove_background,
        "Use Global": config.get("use_global_embedding", False),
        "Embedding Model": config.get("embedding_model"),
        "Use Fisher": config.get("use_fisher", True),
        "Use GV": config.get("use_geometric_verification", False),
        "GV Method": config.get("gv_method"),
        "w_fisher": config.get("w_fisher"),
        "w_global": config.get("w_global"),
        "Accuracy": metrics["accuracy"],
        "Top-5 Accuracy": metrics["top_n_accuracy"],
        "F1": metrics["classification_metrics"]["weighted avg"]["f1-score"],
        "Worst Classes": extract_worst_classes(metrics),
    }
    return row


def grid_search_dataset(dataset: str, output_dir: str) -> None:
    results: List[Dict] = []
    configs = generate_configs()
    for remove_background in BACKGROUND_OPTIONS:
        for cfg in configs:
            try:
                row = run_experiment(dataset, remove_background, cfg)
                results.append(row)
            except subprocess.CalledProcessError:
                print("Experiment failed for", dataset, cfg)

    df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    excel_path = os.path.join(output_dir, f"{dataset}_results.xlsx")
    best = df.loc[df["Accuracy"].idxmax()].to_frame().T
    with pd.ExcelWriter(excel_path) as writer:
        df.to_excel(writer, index=False, sheet_name="results")
        best.to_excel(writer, index=False, sheet_name="summary")
    print(f"Saved results to {excel_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search over main.py options")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ATRW"],
        help="Datasets to evaluate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluations/hyperparameter_search",
        help="Directory to store result spreadsheets",
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        grid_search_dataset(dataset, args.output_dir)


if __name__ == "__main__":
    main()
