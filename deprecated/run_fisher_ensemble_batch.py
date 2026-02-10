import argparse
import os
import shlex
import subprocess
from datetime import datetime
from typing import List


FAILS_LOG_PATH = "data/logs/fails.txt"


def _build_command(dataset: str) -> List[str]:
    cmd = [
        "python",
        "main.py",
        "--train",
        "--use_fisher",
        "--method",
        "ensamble",
        "--use_lightglue",
        "--use_global_embedding",
        #"--embedding_model",
        #"megadescriptor-l-384",
        "--ds",
        dataset,
    ]
    return cmd


def _log_failure(
    dataset: str,
    cmd: List[str],
    return_code: int,
    stdout: str,
    stderr: str,
    log_path: str = FAILS_LOG_PATH,
) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    signal = f"signal {-return_code}" if return_code < 0 else ""
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    stdout_tail = (stdout or "")[-2000:]
    stderr_tail = (stderr or "")[-2000:]

    lines = [
        f"[{timestamp}] dataset={dataset} exit_code={return_code} {signal}".strip(),
        f"cmd: {cmd_str}",
    ]
    if stderr_tail:
        lines.append("stderr (tail):")
        lines.append(stderr_tail)
    if stdout_tail:
        lines.append("stdout (tail):")
        lines.append(stdout_tail)
    lines.append("-" * 80)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Fisher-vector ensemble training across datasets with robust failure logging."
    )
    parser.add_argument(
        "--ds",
        nargs="+",
        required=True,
        help="Datasets to run, e.g. --ds ATRW SealID",
    )
    args = parser.parse_args()

    datasets = [str(ds).strip() for ds in args.ds if str(ds).strip()]
    if not datasets:
        print("No datasets provided.")
        return 1

    for idx, dataset in enumerate(datasets, start=1):
        print(f"[{idx}/{len(datasets)}] Running dataset: {dataset}")
        cmd = _build_command(dataset)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  -> Failed (exit code {result.returncode}). Logging to {FAILS_LOG_PATH}")
            _log_failure(
                dataset=dataset,
                cmd=cmd,
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
            continue
        print("  -> Success.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
