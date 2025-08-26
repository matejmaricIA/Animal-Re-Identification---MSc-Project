import argparse
import subprocess
import json
from pathlib import Path
import pandas as pd
from itertools import product

from constants import (
    N_COMPONENTS_GMM,
    N_COMPONENTS_PCA,
    EVAL_RESULTS_XLSX,
    EVALUATION_DIR,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

METHODS = ["ensamble", "keynet_hardnet", "disk"]

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def run_main(dataset: str, cfg: dict, version: int, method: str) -> dict:
    """Run main.py with a specific configuration and method."""
    cmd = [
        "python",
        "main.py",
        "--train",
        "--ds",
        dataset,
        "--method",
        method,
        "--version",
        str(version),
        "--save_eval",
    ]

    if cfg.get("remove_background", False):
        cmd.append("--remove_background")

    if cfg.get("use_global_embedding", False):
        cmd.append("--use_global_embedding")
        cmd.extend(["--embedding_model", cfg["embedding_model"]])
        cmd.extend(["--w_global", str(cfg["w_global"])])

    if cfg.get("use_fisher", True):
        cmd.append("--use_fisher")
        cmd.extend(["--w_fisher", str(cfg["w_fisher"])])
    else:
        cmd.append("--no-use_fisher")

    if cfg.get("use_geometric_verification", False):
        cmd.append("--use_geometric_verification")

    subprocess.run(cmd, check=True)

    # Compose tag used by your evaluation writer; includes method now
    tag = (
        f"rmbkg_{cfg.get('remove_background', False)}_tm_False_{method}"
        f"_PCA_{N_COMPONENTS_PCA}_GMM_{N_COMPONENTS_GMM}"
        f"_gv_{cfg.get('use_geometric_verification', False)}_lg_True"
        f"_v{version}"
    )
    eval_path = Path(EVALUATION_DIR) / tag / f"{dataset}_evaluation.json"
    if eval_path.exists():
        with open(eval_path) as f:
            return json.load(f)
    return {}


def collect_problematic_classes(metrics: dict, top_k: int = 5) -> pd.DataFrame:
    """Return dataframe of classes with lowest F1 scores."""
    cm = metrics.get("classification_metrics", {})
    rows = []
    for label, stats in cm.items():
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        rows.append(
            {
                "class": label,
                "precision": stats.get("precision", 0.0),
                "recall": stats.get("recall", 0.0),
                "f1": stats.get("f1-score", 0.0),
                "support": stats.get("support", 0),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.sort_values("f1", inplace=True)
    return df.head(top_k)


def build_configurations():
    """Generate all experiment configurations."""
    configs = []
    backgrounds = [False, True]

    # Global embedding only
    for remove_bg, model in product(backgrounds, ["resnet50", "megadescriptor-l-384"]):
        configs.append(
            {
                "remove_background": remove_bg,
                "use_global_embedding": True,
                "embedding_model": model,
                "use_fisher": False,
                "use_geometric_verification": False,
                "w_global": 1.0,
                "w_fisher": 0.0,
            }
        )

    # Fisher vectors only
    for remove_bg, use_gv in product(backgrounds, [False, True]):
        if use_gv:
                configs.append(
                    {
                        "remove_background": remove_bg,
                        "use_global_embedding": False,
                        "use_fisher": True,
                        "use_geometric_verification": True,
                        "w_fisher": 1.0,
                        "w_global": 0.0,
                    }
                )
        else:
            configs.append(
                {
                    "remove_background": remove_bg,
                    "use_global_embedding": False,
                    "use_fisher": True,
                    "use_geometric_verification": False,
                    "w_fisher": 1.0,
                    "w_global": 0.0,
                }
            )

    # Both embeddings with weights
    weight_combos = [
        (1.0, 1.0), (2.0, 1.0), (1.0, 2.0),
        (3.0, 1.0), (1.0, 3.0), (0.5, 3.0),
        (3.0, 0.5),
    ]
    for remove_bg, (wf, wg) in product(backgrounds, weight_combos):
        for model in ["resnet50", "megadescriptor-l-384"]:
            for use_gv in [False, True]:
                if use_gv:
                        configs.append(
                            {
                                "remove_background": remove_bg,
                                "use_global_embedding": True,
                                "embedding_model": model,
                                "use_fisher": True,
                                "use_geometric_verification": True,
                                "w_fisher": wf,
                                "w_global": wg,
                            }
                        )
                else:
                    configs.append(
                        {
                            "remove_background": remove_bg,
                            "use_global_embedding": True,
                            "embedding_model": model,
                            "use_fisher": True,
                            "use_geometric_verification": False,
                            "w_fisher": wf,
                            "w_global": wg,
                        }
                    )

    return configs


def run_dataset(dataset: str):
    """Run all configurations for a single dataset and summarise results across methods."""
    configs = build_configurations()
    results = []

    # Global version counter across methods × configs
    version_counter = 0
    for method in METHODS:
        for idx, cfg in enumerate(configs):
            version_counter += 1
            print(f"\nRunning {dataset} method {method} configuration {idx+1}/{len(configs)} (v{version_counter}): {cfg}")
            metrics = run_main(dataset, cfg, version_counter, method)
            results.append({"version": version_counter, "method": method, "config": cfg, "metrics": metrics})

    # Read global results Excel to summarise best accuracy (if you aggregate methods there)
    try:
        global_df = pd.read_excel(EVAL_RESULTS_XLSX)
        ds_df = global_df[global_df["Dataset"] == dataset]
    except FileNotFoundError:
        ds_df = pd.DataFrame()

    if not ds_df.empty:
        best_row = ds_df.sort_values("Accuracy", ascending=False).iloc[0]
        print(f"Best configuration for {dataset} from XLSX: accuracy={best_row['Accuracy']}")
    else:
        best_row = None
        print(f"No XLSX results found for {dataset}.")

    # Determine best run using in-run metrics
    scored = [r for r in results if r["metrics"]]
    if scored:
        best = max(scored, key=lambda x: x["metrics"].get("accuracy", 0))
        problematic = collect_problematic_classes(best["metrics"])
    else:
        best = None
        problematic = pd.DataFrame()

    summary_path = Path(EVALUATION_DIR) / f"{dataset}_hyperparam_summary.xlsx"
    with pd.ExcelWriter(summary_path) as writer:
        df_runs = pd.DataFrame(
            [
                {
                    **r["config"],
                    "version": r["version"],
                    "method": r["method"],
                    "accuracy": r["metrics"].get("accuracy"),
                    "top5_accuracy": r["metrics"].get("top_n_accuracy"),
                }
                for r in results
            ]
        )
        df_runs.to_excel(writer, sheet_name="all_runs", index=False)
        if best_row is not None:
            best_row.to_frame().T.to_excel(writer, sheet_name="best_from_xlsx", index=False)
        if not problematic.empty:
            problematic.to_excel(writer, sheet_name="problematic_classes", index=False)

    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run grid search experiments over datasets and methods")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["NyalaData", "SealID", "HyenaID2022", "StripeSpotter", "Giraffes", "ATRW", "CowDataset", "IPanda50"],
        help="Datasets to evaluate",
    )
    args = parser.parse_args()
    for ds in args.datasets:
        try:
            run_dataset(ds)
        except Exception as e:
            print(f"Error running {ds}: {e}")
            continue
