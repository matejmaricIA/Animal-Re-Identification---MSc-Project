#!/usr/bin/env python3
import argparse
import ast
import csv
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from random import randint


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT_DIR / "evaluations" / "final_evals" / "count_evals_k_0_raw.csv"

# Requested setup:
# - global embeddings + fisher vectors
# - method=ensamble for all except cowdataset -> superpoint
# - num_vertices=100 for all
# - num_neighbors=1000 for all except elpephants -> 1500
# - preprocessing flags per dataset as requested
DATASET_CONFIGS = [
    
    {
        "label": "cowdataset",
        "cli_ds": "cowdataset",
        "method": "superpoint",
        "num_vertices": 100,
        "num_neighbors": 500,
        "remove_background": True,
        "use_mantiuk": True,
    },
    {
        "label": "chicks4free",
        "cli_ds": "chicks4freeid",
        "method": "ensamble",
        "num_vertices": 100,
        "num_neighbors": 500,
        "remove_background": False,
        "use_mantiuk": False,
    },
    {
        "label": "czoo",
        "cli_ds": "czoo",
        "method": "ensamble",
        "num_vertices": 100,
        "num_neighbors": 500,
        "remove_background": False,
        "use_mantiuk": False,
    },
    {
        "label": "elpephants",
        "cli_ds": "elpephants",
        "method": "ensamble",
        "num_vertices": 250,
        "num_neighbors": 2000,
        "remove_background": True,
        "use_mantiuk": True,
    },
    {
        "label": "sealid",
        "cli_ds": "sealid",
        "method": "ensamble",
        "num_vertices": 100,
        "num_neighbors": 500,
        "remove_background": False,
        "use_mantiuk": False,
    },
    {
        "label": "seastarreid2023",
        "cli_ds": "seastarreid2023",
        "method": "ensamble",
        "num_vertices": 100,
        "num_neighbors": 500,
        "remove_background": True,
        "use_mantiuk": False,
    },
    {
        "label": "atrw",
        "cli_ds": "atrw",
        "method": "ensamble",
        "num_vertices": 100,
        "num_neighbors": 500,
        "remove_background": True,
        "use_mantiuk": True,
    },
]

ESTIMATE_RE = re.compile(
    r"Estimated individuals:\s*([0-9eE+\-.]+)\s*[±]\s*([0-9eE+\-.]+)\s*\(95% CI \[([0-9eE+\-.]+),\s*([0-9eE+\-.]+)\]\)"
)


def parse_error_rates(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def build_cmd(
    python_exec: str,
    dataset_cfg: dict,
    proposal_mode: str,
    error_rate: float,
    seed: int,
    count_confirm_same_votes: int,
) -> list[str]:
    if proposal_mode not in {"calibrated", "power", "raw"}:
        raise ValueError(f"Unknown proposal_mode: {proposal_mode}")

    cli_proposal_mode = "calibrated" if proposal_mode == "raw" else proposal_mode

    cmd = [
        python_exec,
        "main.py",
        "--count",
        "--ds",
        dataset_cfg["cli_ds"],
        "--use_global_embedding",
        "--use_fisher",
        "--method",
        dataset_cfg["method"],
        "--count_proposal_mode",
        cli_proposal_mode,
        "--num_vertices",
        str(dataset_cfg["num_vertices"]),
        "--num_neighbors",
        str(dataset_cfg["num_neighbors"]),
        "--label_error_rate",
        str(error_rate),
        "--count_confirm_same_votes",
        str(count_confirm_same_votes),
        "--seed",
        str(seed),
    ]
    if proposal_mode == "raw":
        cmd.append("--count_skip_calibration")
    if dataset_cfg["remove_background"]:
        cmd.append("--remove_background")
    if dataset_cfg["use_mantiuk"]:
        cmd.append("--use_mantiuk")
    return cmd


def parse_stdout(stdout_text: str) -> tuple[float | None, float | None, float | None, float | None, dict | None]:
    estimate = None
    stderr = None
    ci_low = None
    ci_high = None
    stats = None

    for line in stdout_text.splitlines():
        match = ESTIMATE_RE.search(line)
        if match:
            estimate = float(match.group(1))
            stderr = float(match.group(2))
            ci_low = float(match.group(3))
            ci_high = float(match.group(4))

    for line in reversed(stdout_text.splitlines()):
        line = line.strip()
        if line.startswith("{") and "oracle_calls" in line:
            try:
                parsed = ast.literal_eval(line)
                if isinstance(parsed, dict):
                    stats = parsed
                    break
            except Exception:
                pass

    return estimate, stderr, ci_low, ci_high, stats


def aggregate(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, float, int, int, int], list[dict]] = {}
    for row in records:
        key = (
            row["dataset"],
            row["proposal_mode"],
            row["label_error_rate"],
            row["num_vertices"],
            row["num_neighbors"],
            row["count_confirm_same_votes"],
        )
        grouped.setdefault(key, []).append(row)

    out_rows: list[dict] = []
    for (dataset, mode, error_rate, num_vertices, num_neighbors, count_confirm_same_votes), items in sorted(
        grouped.items()
    ):
        ok_rows = [item for item in items if item["status"] == "ok"]
        estimates = [item["estimate"] for item in ok_rows if item["estimate"] is not None]
        stderrs = [item["stderr"] for item in ok_rows if item["stderr"] is not None]
        ci_widths = [
            item["ci_high"] - item["ci_low"]
            for item in ok_rows
            if item["ci_high"] is not None and item["ci_low"] is not None
        ]
        runtimes = [item["runtime_sec"] for item in ok_rows]
        oracle_calls = [
            item["oracle_calls"] for item in ok_rows if item["oracle_calls"] is not None
        ]
        confirm_extra_votes = [
            item["confirm_extra_votes"]
            for item in ok_rows
            if item.get("confirm_extra_votes") is not None
        ]

        mean_est = statistics.mean(estimates) if estimates else None
        std_est = statistics.stdev(estimates) if len(estimates) > 1 else 0.0 if estimates else None
        mean_se = statistics.mean(stderrs) if stderrs else None
        mean_ci_width = statistics.mean(ci_widths) if ci_widths else None
        mean_runtime = statistics.mean(runtimes) if runtimes else None
        mean_oracle = statistics.mean(oracle_calls) if oracle_calls else None
        mean_confirm_extra_votes = statistics.mean(confirm_extra_votes) if confirm_extra_votes else None

        out_rows.append(
            {
                "dataset": dataset,
                "proposal_mode": mode,
                "label_error_rate": error_rate,
                "num_vertices": int(num_vertices),
                "num_neighbors": int(num_neighbors),
                "count_confirm_same_votes": int(count_confirm_same_votes),
                "runs_total": len(items),
                "runs_ok": len(ok_rows),
                "runs_failed": len(items) - len(ok_rows),
                "mean_estimate": mean_est,
                "std_estimate": std_est,
                "mean_stderr": mean_se,
                "mean_ci_width": mean_ci_width,
                "mean_oracle_calls": mean_oracle,
                "mean_confirm_extra_votes": mean_confirm_extra_votes,
                "mean_runtime_sec": mean_runtime,
            }
        )
    return out_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run count-mode error sweeps across datasets/modes/seeds and save averaged CSV."
    )
    parser.add_argument(
        "--error_rates",
        type=str,
        default="0.00,0.02,0.05,0.10,0.15,0.30",
        help="Comma-separated label error rates.",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=10,
        help="How many seeds per dataset/mode/error combination.",
    )
    parser.add_argument(
        "--seed_start",
        type=int,
        default=randint(0, 1000000),
        help="Starting seed (inclusive).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--python_exec",
        type=str,
        default=sys.executable,
        help="Python executable used to invoke main.py.",
    )
    parser.add_argument(
        "--count_confirm_same_votes",
        type=int,
        default=1,
        help="Pass-through for main.py --count_confirm_same_votes.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately when a run fails.",
    )
    args = parser.parse_args()

    #modes = ["calibrated", "power", "raw"]
    modes = ["raw"]
    error_rates = parse_error_rates(args.error_rates)
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "proposal_mode",
        "label_error_rate",
        "num_vertices",
        "num_neighbors",
        "count_confirm_same_votes",
        "runs_total",
        "runs_ok",
        "runs_failed",
        "mean_estimate",
        "std_estimate",
        "mean_stderr",
        "mean_ci_width",
        "mean_oracle_calls",
        "mean_confirm_extra_votes",
        "mean_runtime_sec",
    ]

    # Initialize CSV with header
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    total_runs = len(DATASET_CONFIGS) * len(modes) * len(error_rates) * len(seeds)
    run_index = 0
    total_attempts = 0

    for dataset_cfg in DATASET_CONFIGS:
        for mode in modes:
            for error_rate in error_rates:
                batch_records: list[dict] = []
                for seed in seeds:
                    run_index += 1
                    total_attempts += 1
                    cmd = build_cmd(
                        python_exec=args.python_exec,
                        dataset_cfg=dataset_cfg,
                        proposal_mode=mode,
                        error_rate=error_rate,
                        seed=seed,
                        count_confirm_same_votes=int(args.count_confirm_same_votes),
                    )
                    print(
                        f"[{run_index}/{total_runs}] ds={dataset_cfg['label']} mode={mode} "
                        f"err={error_rate:.2f} seed={seed}"
                    )

                    start = time.time()
                    proc = subprocess.run(
                        cmd,
                        cwd=str(ROOT_DIR),
                        text=True,
                        capture_output=True,
                    )
                    runtime = time.time() - start

                    estimate, stderr, ci_low, ci_high, stats = parse_stdout(proc.stdout)
                    oracle_calls = None
                    confirm_extra_votes = None
                    if isinstance(stats, dict):
                        value = stats.get("oracle_calls")
                        if value is not None:
                            oracle_calls = float(value)
                        value = stats.get("confirm_extra_votes")
                        if value is not None:
                            confirm_extra_votes = float(value)

                    row = {
                        "dataset": dataset_cfg["label"],
                        "cli_dataset": dataset_cfg["cli_ds"],
                        "proposal_mode": mode,
                        "label_error_rate": float(error_rate),
                        "num_vertices": int(dataset_cfg["num_vertices"]),
                        "num_neighbors": int(dataset_cfg["num_neighbors"]),
                        "count_confirm_same_votes": int(args.count_confirm_same_votes),
                        "seed": int(seed),
                        "status": "ok" if proc.returncode == 0 else "failed",
                        "return_code": int(proc.returncode),
                        "estimate": estimate,
                        "stderr": stderr,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "oracle_calls": oracle_calls,
                        "confirm_extra_votes": confirm_extra_votes,
                        "runtime_sec": float(runtime),
                    }
                    batch_records.append(row)

                    if proc.returncode != 0:
                        print("[ERROR] Command failed:")
                        print(" ".join(cmd))
                        if proc.stderr:
                            print(proc.stderr.strip())
                        if args.stop_on_error:
                            break

                # Write batch results
                if batch_records:
                    aggregated_rows = aggregate(batch_records)
                    with output_path.open("a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        for agg_row in aggregated_rows:
                            writer.writerow(agg_row)

                if args.stop_on_error and batch_records and batch_records[-1]["status"] == "failed":
                    break
            if args.stop_on_error and batch_records and batch_records[-1]["status"] == "failed":
                break
        if args.stop_on_error and batch_records and batch_records[-1]["status"] == "failed":
            break

    print(f"[DONE] Wrote results to: {output_path}")
    print(f"[DONE] Total run attempts: {total_attempts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
