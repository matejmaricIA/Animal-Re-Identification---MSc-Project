import argparse
import os
import subprocess
from itertools import product
from typing import List, Dict

import pandas as pd

from constants import COUNT_RESULTS_XLSX

# Datasets to evaluate
datasets = ["Giraffes", "NyalaData", "ATRW", "HyenaID2022", "StripeSpotter"]

# Parameter grid for count mode experiments
PARAM_GRID: Dict[str, List] = {
    "num_vertices": [50, 200],
    "num_neighbors": [50, 200, 500],
    "use_fisher": [True],
    "use_global_embedding": [True],
    "w_fisher": [3.0],
    "w_global": [1.0],
    "gv_threshold": [0.95],
    "use_geometric_verification": [True],
    "embedding_model": ["megadescriptor-l-384"],
    "remove_background": [False, True],
}

METHOD = "ensamble"


def run_main(dataset: str, cfg: Dict, seed: int) -> None:
    """Execute main.py in count mode with the provided configuration."""
    cmd = [
        "python", "main.py", "--count", "--save_count",
        "--ds", dataset, "--method", METHOD,
        "--num_vertices", str(cfg["num_vertices"]),
        "--num_neighbors", str(cfg["num_neighbors"]),
        "--gv_threshold", str(cfg["gv_threshold"]),
        "--seed", str(seed),
    ]

    if cfg.get("remove_background", False):
        cmd.append("--remove_background")

    if cfg.get("use_geometric_verification", False):
        cmd.append("--use_geometric_verification")
        cmd.append("--use_lightglue")
        cmd.append("--automated_mode")

    if cfg.get("use_fisher", True):
        cmd.append("--use_fisher")
        cmd.extend(["--w_fisher", str(cfg["w_fisher"])])
    else:
        cmd.append("--no-use_fisher")

    if cfg.get("use_global_embedding", False):
        cmd.append("--use_global_embedding")
        cmd.extend(["--embedding_model", cfg["embedding_model"]])
        cmd.extend(["--w_global", str(cfg["w_global"])])

    subprocess.run(cmd, check=True)


def build_configs() -> List[Dict]:
    keys = list(PARAM_GRID.keys())
    values = [PARAM_GRID[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def aggregate_runs(start_idx: int, n_runs: int) -> pd.DataFrame:
    df = pd.read_excel(COUNT_RESULTS_XLSX)
    return df.iloc[start_idx:start_idx + n_runs]


def main(n_runs: int, base_seed: int) -> None:
    configs = build_configs()
    summary_rows = []

    for dataset in datasets:
        for cfg in configs:
            start_idx = len(pd.read_excel(COUNT_RESULTS_XLSX)) if os.path.exists(COUNT_RESULTS_XLSX) else 0

            for i in range(n_runs):
                seed = base_seed + i
                run_main(dataset, cfg, seed)

            new_rows = aggregate_runs(start_idx, n_runs)
            est_mean = new_rows["Result"].mean()
            est_std = new_rows["Result"].std(ddof=1)
            se_mean = new_rows["Std Error"].mean()
            gt = new_rows["Ground Truth"].iloc[0] if not new_rows.empty else float("nan")

            summary = {
                "Dataset": dataset,
                **cfg,
                "Estimate Mean": est_mean,
                "Estimate Std": est_std,
                "Std Error Mean": se_mean,
                "Ground Truth": gt,
            }
            summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    os.makedirs("evaluations/count", exist_ok=True)
    out_path = "evaluations/count/hyperparameter_count_summary.xlsx"
    summary_df.to_excel(out_path, index=False)
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid search for population counting hyperparameters")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of runs per configuration")
    parser.add_argument("--base_seed", type=int, default=0, help="Base seed; increments for each run")
    args = parser.parse_args()
    main(args.n_runs, args.base_seed)
