#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path


# Standard datasets used in the thesis (docs/thesis.pdf), normalized to pipeline tags.
DEFAULT_DATASETS = [
    "atrw",
    "cowdataset",
    "elpephants",
    "czoo",
    "chicks4freeid",
    "sealid",
    "seastarreid2023",
]

DEFAULT_PCA_DIMS_BY_METHOD = {
    "superpoint": [64, 128, 256],
    # DISK/ALIKED are 128-D in this pipeline; sweep smaller PCA outputs.
    "disk": [64, 96, 128],
    "aliked": [64, 96, 128],
}


def dataset_main_flags(dataset: str) -> list[str]:
    """Mirror dataset_main_flags() from run_final_comparisons.sh."""
    ds = str(dataset).strip().lower()
    if ds in {"atrw", "cowdataset", "elpephants"}:
        return ["--use_mantiuk", "--remove_background"]
    if ds == "sealid":
        return ["--use_mantiuk"]
    if ds == "seastarreid2023":
        return ["--remove_background"]
    return []


def _tail_text(path: Path, n_lines: int = 30) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    return "\n".join(lines[-n_lines:])


def _prepare_output_csv(path: Path, header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)


def _ensure_output_csv(path: Path, header: list[str], *, overwrite: bool) -> None:
    """Create output CSV with header, or append if header matches."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        if path.exists() and path.stat().st_size > 0:
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = path.with_name(f"{path.stem}_backup_{ts}{path.suffix}")
            backup.write_bytes(path.read_bytes())
            print(f"Backed up existing CSV to: {backup}")
        _prepare_output_csv(path, header)
        return

    if not path.exists() or path.stat().st_size == 0:
        _prepare_output_csv(path, header)
        return

    # Validate the existing header; if it doesn't match, back it up and start fresh.
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            first = next(reader, None)
    except Exception:
        first = None

    if list(first or []) != list(header):
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_name(f"{path.stem}_backup_{ts}{path.suffix}")
        backup.write_bytes(path.read_bytes())
        print(f"[WARN] Existing CSV header mismatch. Backed up to: {backup}")
        _prepare_output_csv(path, header)


def run_one(
    *,
    repo_root: Path,
    dataset: str,
    method: str,
    pca_dim: int,
    calib_ids: int,
    extra_flags: list[str],
    log_dir: Path,
    debug: bool,
    dry_run: bool,
) -> dict:
    ds = str(dataset).strip().lower()
    method = str(method).strip().lower()
    pca_dim = int(pca_dim)
    version = f"ablate_{method}_fisher_{ds}_pca{pca_dim}"

    cmd = [
        sys.executable,
        str(repo_root / "main.py"),
        "--train",
        "--ds",
        ds,
        "--save_eval",
        "--version",
        version,
        "--calib_ids",
        str(int(calib_ids)),
        "--use_fisher",
        "--method",
        method,
        "--fusion_signals",
        "fisher",
        "--pca_dim",
        str(pca_dim),
        *dataset_main_flags(ds),
        *extra_flags,
    ]
    if debug:
        cmd.insert(cmd.index("--version"), "--debug")

    cmd_str = " ".join(cmd)
    log_path = log_dir / f"main_{method}_{ds}_pca{pca_dim}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print()
    print(f"[RUN] dataset={ds} method={method} pca_dim={pca_dim}")
    print(f"[RUN] command: {cmd_str}")
    print(f"[RUN] log: {log_path}")

    if dry_run:
        return {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "dataset": ds,
            "method": method,
            "pca_dim": pca_dim,
            "status": "dry_run",
            "accuracy": "",
            "top5_accuracy": "",
            "f1_score": "",
            "runtime_minutes": "",
            "eval_json": "",
            "error": "",
            "command": cmd_str,
        }

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if proc.returncode != 0:
        err_tail = _tail_text(log_path, n_lines=40).replace("\n", " | ")
        return {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "dataset": ds,
            "method": method,
            "pca_dim": pca_dim,
            "status": "error",
            "accuracy": "",
            "top5_accuracy": "",
            "f1_score": "",
            "runtime_minutes": "",
            "eval_json": "",
            "error": f"exit={proc.returncode}; {err_tail}",
            "command": cmd_str,
        }

    # Match what run_final_comparisons.sh does: locate the evaluation JSON by version+dataset.
    candidates = sorted(
        (repo_root / "evaluations" / "full_evals").glob(f"*_v{version}/{ds}_evaluation.json")
    )
    if not candidates:
        return {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "dataset": ds,
            "method": method,
            "pca_dim": pca_dim,
            "status": "error",
            "accuracy": "",
            "top5_accuracy": "",
            "f1_score": "",
            "runtime_minutes": "",
            "eval_json": "",
            "error": f"Missing evaluation JSON for version={version}, dataset={ds}",
            "command": cmd_str,
        }

    eval_json = candidates[-1]
    try:
        metrics = json.loads(eval_json.read_text(encoding="utf-8"))
    except Exception as e:
        return {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "dataset": ds,
            "method": method,
            "pca_dim": pca_dim,
            "status": "error",
            "accuracy": "",
            "top5_accuracy": "",
            "f1_score": "",
            "runtime_minutes": "",
            "eval_json": str(eval_json),
            "error": f"Failed to parse evaluation JSON: {e}",
            "command": cmd_str,
        }

    acc = metrics.get("accuracy", "")
    top5 = metrics.get("top_n_accuracy", "")
    f1 = (((metrics.get("classification_metrics") or {}).get("weighted avg") or {}).get("f1-score", ""))
    eval_sec = metrics.get("eval_runtime_sec", "")
    rt_min = ""
    if isinstance(eval_sec, (int, float)):
        rt_min = float(eval_sec) / 60.0

    return {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset": ds,
        "method": method,
        "pca_dim": pca_dim,
        "status": "ok",
        "accuracy": acc,
        "top5_accuracy": top5,
        "f1_score": f1,
        "runtime_minutes": rt_min,
        "eval_json": str(eval_json),
        "error": "",
        "command": cmd_str,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Fisher-only classification across thesis datasets, "
            "sweeping PCA output dimension for a chosen local descriptor method."
        )
    )
    parser.add_argument(
        "--method",
        type=str,
        default="superpoint",
        choices=["superpoint", "disk", "aliked"],
        help="Local descriptor method used for Fisher vectors.",
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument(
        "--pca_dims",
        nargs="+",
        type=int,
        default=None,
        help="Override PCA dims; default depends on --method.",
    )
    parser.add_argument("--calib_ids", type=int, default=10)
    parser.add_argument(
        "--out_csv",
        type=str,
        default="evaluations/classifications/superpoint_fisher_pca_ablation.csv",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="evaluations/classifications/logs_pca_ablation",
    )
    parser.add_argument("--no_debug", action="store_true", help="Disable --debug for main.py runs.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running.")
    parser.add_argument(
        "--overwrite_csv",
        action="store_true",
        help="Overwrite (with backup) instead of appending to --out_csv.",
    )
    parser.add_argument(
        "--extra_flags",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra flags forwarded to main.py (prefix with --extra_flags -- ...).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_csv = repo_root / str(args.out_csv)
    log_dir = repo_root / str(args.log_dir)

    header = [
        "timestamp",
        "dataset",
        "method",
        "pca_dim",
        "status",
        "accuracy",
        "top5_accuracy",
        "f1_score",
        "runtime_minutes",
        "eval_json",
        "error",
        "command",
    ]
    print(f"Writing results to: {out_csv}")
    _ensure_output_csv(out_csv, header, overwrite=bool(args.overwrite_csv))

    method = str(args.method).strip().lower()
    pca_dims = args.pca_dims
    if not pca_dims:
        pca_dims = DEFAULT_PCA_DIMS_BY_METHOD.get(method, [64, 128])

    rows: list[dict] = []
    for ds in args.datasets:
        for d in pca_dims:
            row = run_one(
                repo_root=repo_root,
                dataset=ds,
                method=method,
                pca_dim=int(d),
                calib_ids=int(args.calib_ids),
                extra_flags=list(args.extra_flags or []),
                log_dir=log_dir,
                debug=not bool(args.no_debug),
                dry_run=bool(args.dry_run),
            )
            rows.append(row)
            with out_csv.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([row.get(h, "") for h in header])

    ok = sum(1 for r in rows if r.get("status") == "ok")
    err = sum(1 for r in rows if r.get("status") == "error")
    print()
    print(f"Done. ok={ok} error={err} total={len(rows)}")
    return 0 if err == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
