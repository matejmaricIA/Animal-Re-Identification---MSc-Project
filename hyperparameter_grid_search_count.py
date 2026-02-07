import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
import threading
from itertools import product
from pathlib import Path
from typing import List, Dict

import pandas as pd

from constants import COUNT_RESULTS_XLSX

DEFAULT_DATASETS = [
    #"atrw",
    #"cowdataset",
    #"elpephants",
    #"ctai",
    #"czoo",
    "chicks4freeid",
    #"sealid",
    #"seastarreid2023",
]


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


# Dataset-specific preprocessing rules (per your instruction).
# - segmentation => --remove_background
# - mantiuk => --use_mantiuk
DATASET_PREPROCESSING = {
    "atrw": {"remove_background": True, "use_mantiuk": True},
    "cowdataset": {"remove_background": True, "use_mantiuk": True},
    "elpephants": {"remove_background": True, "use_mantiuk": True},
    "seastarreid2023": {"remove_background": True, "use_mantiuk": False},
}

# Parameter grid for count mode experiments
PARAM_GRID: Dict[str, List] = {
    # NOTE: Keep the default grid reasonably small; override by editing PARAM_GRID.
    "count_local_evidence": ["conf_matches", "inliers"],
    "num_vertices": [250, 350],
    "num_neighbors": [30, 50],
    "count_proposal_mode": ["calibrated"],
    "count_local_mu": [0.5, 0.7],
    "count_shortlist_B": [150, 300],
    "count_mix_alpha": [0.6, 0.8],
    "count_cal_pairs": [1000, 2000],
    "count_cal_shortlist": [150, 300],
    "count_cal_negs_per_query": [350, 450],
}

METHOD = "ensamble"
USE_FISHER = True
USE_GLOBAL_EMBEDDING = True
EMBEDDING_MODEL = "megadescriptor-l-384"

EVAL_COUNT_DIR = Path("evaluations/count")
LOG_ROOT = EVAL_COUNT_DIR / "grid_search_logs"
RUNS_JSONL = EVAL_COUNT_DIR / "grid_search_runs.jsonl"
SUMMARY_XLSX = EVAL_COUNT_DIR / "hyperparameter_count_summary.xlsx"
SUMMARY_CSV = EVAL_COUNT_DIR / "hyperparameter_count_summary.csv"


def _now_tag() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _append_jsonl(path: Path, record: Dict) -> None:
    _ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _atomic_write_excel(df: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_excel(tmp, index=False)
    os.replace(tmp, path)


def _get_preprocessing_flags(dataset: str) -> Dict[str, bool]:
    key = _normalize_name(dataset)
    return DATASET_PREPROCESSING.get(key, {"remove_background": False, "use_mantiuk": False})


def _build_command(dataset: str, cfg: Dict, seed: int) -> List[str]:
    preprocess_flags = _get_preprocessing_flags(dataset)
    cmd: List[str] = [
        sys.executable,
        "main.py",
        "--count",
        "--save_count",
        "--ds",
        dataset,
        "--method",
        METHOD,
        "--num_vertices", str(cfg["num_vertices"]),
        "--num_neighbors", str(cfg["num_neighbors"]),
        "--count_proposal_mode", str(cfg["count_proposal_mode"]),
        "--count_local_evidence", str(cfg["count_local_evidence"]),
        "--count_local_mu", str(cfg["count_local_mu"]),
        "--count_shortlist_B", str(cfg["count_shortlist_B"]),
        "--count_mix_alpha", str(cfg["count_mix_alpha"]),
        "--count_cal_pairs", str(cfg["count_cal_pairs"]),
        "--count_cal_shortlist", str(cfg["count_cal_shortlist"]),
        "--count_cal_negs_per_query", str(cfg["count_cal_negs_per_query"]),
        "--seed", str(seed),
    ]

    if preprocess_flags.get("remove_background", False):
        cmd.append("--remove_background")

    if preprocess_flags.get("use_mantiuk", False):
        cmd.append("--use_mantiuk")

    # Local evidence uses descriptor matching; LightGlue is the default matcher.
    cmd.append("--use_lightglue")
    cmd.extend(["--gv_matcher", "lightglue"])

    if USE_FISHER:
        cmd.append("--use_fisher")

    if USE_GLOBAL_EMBEDDING:
        cmd.append("--use_global_embedding")
        cmd.extend(["--embedding_model", EMBEDDING_MODEL])

    return cmd


def run_main(
    dataset: str,
    cfg: Dict,
    seed: int,
    *,
    dry_run: bool = False,
    timeout_minutes: float | None = None,
    run_tag: str | None = None,
    live_output: bool = True,
) -> Dict:
    """Execute main.py in count mode with the provided configuration.

    Returns a dict describing the run attempt (status, return code, runtime, log path).
    """
    cmd = _build_command(dataset, cfg, seed)
    cmd_str = " ".join(cmd)

    record: Dict = {
        "ts": _now_tag(),
        "dataset": str(dataset),
        "seed": int(seed),
        "cfg": dict(cfg),
        "cmd": cmd_str,
        "dry_run": bool(dry_run),
        "timeout_minutes": float(timeout_minutes) if timeout_minutes is not None else None,
        "run_tag": str(run_tag) if run_tag is not None else None,
        "live_output": bool(live_output),
    }

    if dry_run:
        print(cmd_str)
        record.update(
            {
                "status": "dry_run",
                "returncode": 0,
                "runtime_sec": 0.0,
                "log_path": None,
            }
        )
        return record

    ds_log_dir = LOG_ROOT / _normalize_name(dataset)
    _ensure_dir(ds_log_dir)
    tag = run_tag if run_tag else f"seed{seed}"
    log_path = ds_log_dir / f"{tag}_{_now_tag()}.log"

    # Guardrails: keep BLAS / sklearn from spawning many threads and spiking memory.
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )

    start = time.time()
    status = "ok"
    returncode: int | None = None
    error: str | None = None

    try:
        with log_path.open("w", encoding="utf-8") as log_f:
            log_f.write(f"[RUN] {record['ts']} dataset={dataset} seed={seed}\n")
            log_f.write(f"[CMD] {cmd_str}\n")
            log_f.write(f"[CFG] {json.dumps(cfg, sort_keys=True)}\n\n")
            log_f.flush()

            timeout_sec = None
            if timeout_minutes is not None:
                timeout_sec = float(timeout_minutes) * 60.0

            if live_output:
                header = f"\n[GRID] dataset={dataset} seed={seed} log={log_path}\n"
                sys.stdout.write(header)
                sys.stdout.flush()

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    bufsize=1,
                )

                assert proc.stdout is not None

                def _drain() -> None:
                    try:
                        for line in proc.stdout:
                            log_f.write(line)
                            log_f.flush()
                            sys.stdout.write(line)
                            sys.stdout.flush()
                    except Exception:
                        # Avoid crashing the wrapper on output streaming issues.
                        log_f.write("\n[WARN] Output streaming interrupted.\n")
                        log_f.flush()
                    finally:
                        try:
                            proc.stdout.close()
                        except Exception:
                            pass

                t = threading.Thread(target=_drain, daemon=True)
                t.start()
                try:
                    proc.wait(timeout=timeout_sec)
                except subprocess.TimeoutExpired:
                    status = "timeout"
                    error = "TimeoutExpired"
                    proc.kill()
                    proc.wait()
                returncode = int(proc.returncode or 0)
                # Ensure the drain thread gets a chance to flush remaining output.
                t.join(timeout=5.0)
            else:
                proc = subprocess.run(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    timeout=timeout_sec,
                    check=False,
                )
                returncode = int(proc.returncode)

            if status == "ok" and returncode != 0:
                status = "failed"
    except subprocess.TimeoutExpired:
        status = "timeout"
        returncode = -1
        error = "TimeoutExpired"
        with log_path.open("a", encoding="utf-8") as log_f:
            log_f.write("\n[ERROR] TimeoutExpired\n")
    except Exception as e:
        status = "error"
        returncode = -1
        error = f"{type(e).__name__}: {e}"
        with log_path.open("a", encoding="utf-8") as log_f:
            log_f.write("\n[ERROR] Wrapper exception\n")
            log_f.write(traceback.format_exc())

    runtime_sec = float(time.time() - start)
    record.update(
        {
            "status": status,
            "returncode": returncode,
            "runtime_sec": runtime_sec,
            "log_path": str(log_path),
            "error": error,
        }
    )
    return record

def build_configs() -> List[Dict]:
    keys = list(PARAM_GRID.keys())
    values = [PARAM_GRID[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def _objective_key(objective: str) -> str:
    key = str(objective).strip().lower()
    if key in {"rel", "rel_error", "relative", "relative_error"}:
        return "Rel Error"
    if key in {"abs", "abs_error", "absolute", "absolute_error"}:
        return "Abs Error"
    raise ValueError(f"Unknown objective: {objective}")


def _objective_value(summary: Dict, objective: str) -> float:
    key = _objective_key(objective)
    value = summary.get(key, None)
    try:
        value_f = float(value)
    except Exception:
        return float("inf")
    if not math.isfinite(value_f):
        return float("inf")
    return value_f


def _suggest_param(trial, name: str, values: List):
    values = list(values)
    if len(values) == 1:
        return values[0]

    # Categorical for strings / mixed types.
    if any(isinstance(v, str) for v in values):
        return trial.suggest_categorical(name, values)

    # bool is a subclass of int; handle explicitly.
    if all(isinstance(v, bool) for v in values):
        return trial.suggest_categorical(name, values)

    is_int = all(isinstance(v, int) and not isinstance(v, bool) for v in values)
    is_float = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)

    if is_int:
        low = int(min(values))
        high = int(max(values))
        if low == high:
            return low
        return int(trial.suggest_int(name, low, high))

    if is_float:
        low_f = float(min(values))
        high_f = float(max(values))
        if low_f == high_f:
            return low_f
        return float(trial.suggest_float(name, low_f, high_f))

    return trial.suggest_categorical(name, values)


def _build_cfg_optuna(trial) -> Dict:
    cfg: Dict[str, object] = {}
    for key, values in PARAM_GRID.items():
        cfg[key] = _suggest_param(trial, str(key), list(values))
    return cfg


def _read_count_results() -> pd.DataFrame:
    if not os.path.exists(COUNT_RESULTS_XLSX):
        return pd.DataFrame()
    try:
        return pd.read_excel(COUNT_RESULTS_XLSX)
    except Exception as e:
        # If the XLSX becomes corrupted (e.g. interrupted write), move it aside so
        # the grid search can continue, while preserving the broken file.
        ts = _now_tag()
        src = Path(COUNT_RESULTS_XLSX)
        dst = src.with_name(f"{src.stem}_corrupt_{ts}{src.suffix}")
        try:
            os.replace(src, dst)
            print(f"[WARN] Could not read {src}; moved to {dst}. Error: {e}")
        except Exception as move_err:
            print(f"[WARN] Could not read {src} and could not move it aside: {move_err}. Error: {e}")
        return pd.DataFrame()


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col in {
            "Dataset",
            "Embedding Model",
            "Fisher Method",
            "Local Feature Method",
            "GV Matcher",
            "Calibration Method",
            "Dataset Type",
        }:
            continue
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            df[col] = series.astype(float)
            continue
        if pd.api.types.is_numeric_dtype(series):
            continue

        converted = pd.to_numeric(series, errors="coerce")
        non_null = int(series.notna().sum())
        if non_null == 0:
            continue
        frac_numeric = float(converted.notna().sum()) / float(non_null)
        # Only coerce object columns that are effectively numeric.
        if frac_numeric >= 0.9:
            df[col] = converted
    return df


def _aggregate_main_row_stats(new_rows: pd.DataFrame) -> Dict:
    """Aggregate main.py output rows into a single dict using means."""
    if new_rows.empty:
        return {}

    rows = _coerce_numeric(new_rows)
    out: Dict[str, object] = {}
    for col in rows.columns:
        if col in out:
            continue
        series = rows[col]
        if pd.api.types.is_numeric_dtype(series):
            out[col] = float(pd.to_numeric(series, errors="coerce").mean())
        else:
            values = [v for v in series.dropna().astype(str).unique().tolist() if v != ""]
            if not values:
                out[col] = ""
            elif len(values) == 1:
                out[col] = values[0]
            else:
                out[col] = ";".join(values[:5])
    return out


def _summarize_new_rows(dataset: str, cfg: Dict, new_rows: pd.DataFrame, run_reports: List[Dict]) -> Dict:
    preprocess_flags = _get_preprocessing_flags(dataset)

    summary: Dict = {
        "Dataset": dataset,
        "Remove Background": bool(preprocess_flags.get("remove_background", False)),
        "Use Mantiuk": bool(preprocess_flags.get("use_mantiuk", False)),
        "Method": METHOD,
        "Use Fisher": bool(USE_FISHER),
        "Use Global Embedding": bool(USE_GLOBAL_EMBEDDING),
        "Embedding Model": EMBEDDING_MODEL if USE_GLOBAL_EMBEDDING else "None",
        **cfg,
        "Runs Requested": int(len(run_reports)),
        "Runs Succeeded": int(sum(r.get("status") == "ok" and r.get("returncode") == 0 for r in run_reports)),
        "Runs Failed": int(sum(r.get("status") not in {"ok", "dry_run"} for r in run_reports)),
        "Last Log Path": str(run_reports[-1].get("log_path")) if run_reports else "",
        "Status": "ok",
    }

    if new_rows.empty:
        summary["Status"] = "no_rows_written"
        return summary

    new_rows = _coerce_numeric(new_rows)
    aggregated = _aggregate_main_row_stats(new_rows)
    summary.update(aggregated)

    est_mean = float(pd.to_numeric(new_rows.get("Result", pd.Series(dtype=float)), errors="coerce").mean())
    est_std = float(pd.to_numeric(new_rows.get("Result", pd.Series(dtype=float)), errors="coerce").std(ddof=1)) if len(new_rows) > 1 else 0.0
    se_mean = float(pd.to_numeric(new_rows.get("Std Error", pd.Series(dtype=float)), errors="coerce").mean())
    gt = float(pd.to_numeric(new_rows.get("Ground Truth", pd.Series([float("nan")])), errors="coerce").iloc[0])
    abs_error = abs(est_mean - gt) if pd.notna(gt) else float("nan")
    rel_error = abs_error / gt if gt and gt > 0 else float("nan")

    coverage = float(
        (
            (pd.to_numeric(new_rows.get("CI Low (95%)", pd.Series(dtype=float)), errors="coerce") <= gt)
            & (gt <= pd.to_numeric(new_rows.get("CI High (95%)", pd.Series(dtype=float)), errors="coerce"))
        ).mean()
    ) if pd.notna(gt) and "CI Low (95%)" in new_rows.columns and "CI High (95%)" in new_rows.columns else float("nan")

    summary.update(
        {
            "Estimate Mean": est_mean,
            "Estimate Std": est_std,
            "Std Error Mean": se_mean,
            "Ground Truth": gt,
            "Abs Error": float(abs_error),
            "Rel Error": float(rel_error),
            "CI Coverage (runs)": coverage,
        }
    )
    return summary


def _write_summary(summary_rows: List[Dict]) -> None:
    df = pd.DataFrame(summary_rows)
    _atomic_write_csv(df, SUMMARY_CSV)
    _atomic_write_excel(df, SUMMARY_XLSX)


def main(
    datasets: List[str],
    n_runs: int,
    base_seed: int,
    *,
    max_configs: int | None = None,
    search: str = "grid",
    objective: str = "rel_error",
    n_trials: int | None = None,
    dry_run: bool = False,
    timeout_minutes: float | None = None,
    live_output: bool = True,
) -> None:
    summary_rows = []

    search_mode = str(search).strip().lower()
    objective_key = _objective_key(objective)

    if search_mode not in {"grid", "optuna"}:
        raise ValueError(f"Unknown --search mode: {search}")

    if search_mode == "grid":
        configs = build_configs()
        for dataset in datasets:
            print(f"Running dataset: {dataset}")
            for cfg_idx, cfg in enumerate(configs):
                if max_configs is not None and cfg_idx >= int(max_configs):
                    break

                before = _read_count_results()
                start_idx = int(len(before))

                run_reports: List[Dict] = []
                cfg_hash = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()[:8]
                for i in range(n_runs):
                    seed = base_seed + i
                    run_tag = f"cfg{cfg_idx:05d}_{cfg_hash}_run{i:02d}_seed{seed}"
                    print(f"Running run: {run_tag}")
                    report = run_main(
                        dataset,
                        cfg,
                        seed,
                        dry_run=dry_run,
                        timeout_minutes=timeout_minutes,
                        run_tag=run_tag,
                        live_output=live_output,
                    )
                    run_reports.append(report)
                    try:
                        _append_jsonl(RUNS_JSONL, report)
                    except Exception as e:
                        print(f"[WARN] Failed to append to run manifest {RUNS_JSONL}: {e}")

                after = _read_count_results() if not dry_run else before
                if after.empty:
                    new_rows = pd.DataFrame()
                else:
                    new_rows = after.iloc[start_idx:]
                    if len(new_rows) > n_runs:
                        new_rows = new_rows.iloc[:n_runs]

                summary = _summarize_new_rows(dataset, cfg, new_rows, run_reports)
                summary["Search Mode"] = "grid"
                summary_rows.append(summary)
                try:
                    _write_summary(summary_rows)
                except Exception as e:
                    print(f"[WARN] Failed to write summary files: {e}")

        print(f"Saved summary to {SUMMARY_XLSX}")
        return

    # --- Optuna mode ---
    if dry_run:
        raise ValueError("Optuna search does not support --dry_run (it needs real results).")

    try:
        import optuna
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Optuna is not available in this environment. Install it in your venv "
            "or run in grid mode."
        ) from e

    trials = n_trials if n_trials is not None else max_configs
    if trials is None or int(trials) <= 0:
        raise ValueError("Optuna mode requires --n_trials (or --max_configs) > 0.")
    trials = int(trials)

    for dataset in datasets:
        print(f"Running dataset (optuna): {dataset}")

        sampler = optuna.samplers.TPESampler(seed=int(base_seed))
        study_name = f"count_{_normalize_name(dataset)}_{_now_tag()}"
        study = optuna.create_study(direction="minimize", sampler=sampler, study_name=study_name)
        seen_cfg: Dict[str, float] = {}

        def _objective(trial) -> float:
            cfg = _build_cfg_optuna(trial)
            cfg_key = json.dumps(cfg, sort_keys=True)
            if cfg_key in seen_cfg:
                trial.set_user_attr("duplicate", True)
                return float(seen_cfg[cfg_key])

            before = _read_count_results()
            start_idx = int(len(before))

            run_reports: List[Dict] = []
            cfg_hash = hashlib.sha1(cfg_key.encode("utf-8")).hexdigest()[:8]
            trial_seed_base = int(base_seed) + int(trial.number) * int(n_runs)
            for i in range(int(n_runs)):
                seed = trial_seed_base + i
                run_tag = f"optuna_t{trial.number:04d}_{cfg_hash}_run{i:02d}_seed{seed}"
                print(f"Running run: {run_tag}")
                report = run_main(
                    dataset,
                    cfg,
                    seed,
                    dry_run=False,
                    timeout_minutes=timeout_minutes,
                    run_tag=run_tag,
                    live_output=live_output,
                )
                run_reports.append(report)
                try:
                    _append_jsonl(RUNS_JSONL, report)
                except Exception as e:
                    print(f"[WARN] Failed to append to run manifest {RUNS_JSONL}: {e}")

            after = _read_count_results()
            if after.empty:
                new_rows = pd.DataFrame()
            else:
                new_rows = after.iloc[start_idx:]
                if len(new_rows) > int(n_runs):
                    new_rows = new_rows.iloc[: int(n_runs)]

            summary = _summarize_new_rows(dataset, cfg, new_rows, run_reports)
            summary["Search Mode"] = "optuna"
            summary["Trial"] = int(trial.number)
            summary["Study"] = str(study_name)
            summary_rows.append(summary)
            try:
                _write_summary(summary_rows)
            except Exception as e:
                print(f"[WARN] Failed to write summary files: {e}")

            succeeded = int(summary.get("Runs Succeeded", 0))
            value = float("inf") if succeeded <= 0 else _objective_value(summary, objective)

            trial.set_user_attr("cfg", cfg)
            trial.set_user_attr("objective", objective_key)
            trial.set_user_attr("value", value)
            seen_cfg[cfg_key] = float(value)
            return float(value)

        study.optimize(_objective, n_trials=trials, n_jobs=1)

        best_cfg = study.best_trial.user_attrs.get("cfg", study.best_trial.params)
        print(f"[OPTUNA] Best {objective_key}: {study.best_value}")
        print(f"[OPTUNA] Best cfg: {json.dumps(best_cfg, indent=2, sort_keys=True)}")

    print(f"Saved summary to {SUMMARY_XLSX}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid search for population counting hyperparameters")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to evaluate (default: requested list).",
    )
    parser.add_argument("--n_runs", type=int, default=1, help="Number of runs per configuration")
    parser.add_argument("--base_seed", type=int, default=0, help="Base seed; increments for each run")
    parser.add_argument(
        "--max_configs",
        type=int,
        default=None,
        help="Optional cap on number of configs per dataset.",
    )
    parser.add_argument(
        "--search",
        type=str,
        default="grid",
        choices=["grid", "optuna"],
        help="Search strategy (default: grid).",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="rel_error",
        choices=["rel_error", "abs_error"],
        help="Optimization target used by optuna (default: rel_error).",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=None,
        help="Optuna trials per dataset. Defaults to --max_configs when provided.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--timeout_minutes",
        type=float,
        default=None,
        help="Optional per-run timeout (minutes).",
    )
    parser.add_argument(
        "--live_output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream main.py output to console (still logged).",
    )
    args = parser.parse_args()
    main(
        datasets=[str(d) for d in args.datasets],
        n_runs=int(args.n_runs),
        base_seed=int(args.base_seed),
        max_configs=args.max_configs,
        search=str(args.search),
        objective=str(args.objective),
        n_trials=args.n_trials,
        dry_run=bool(args.dry_run),
        timeout_minutes=args.timeout_minutes,
        live_output=bool(args.live_output),
    )
