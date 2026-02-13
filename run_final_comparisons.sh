#!/usr/bin/env bash

# Robust batch runner for final classification comparisons.
# - No CLI arguments (edit dataset lists below directly).
# - Continues after per-dataset/per-config failures.
# - Aggregates both main-pipeline and WildFusion baseline results into:
#     evaluations/classifications/final_comparisons.csv

set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

OUT_CSV="evaluations/classifications/final_comparisons_fewshot.csv"
LOG_DIR="evaluations/classifications/logs"
TMP_WF_CSV="/tmp/wildfusion_final_comparisons_tmp.csv"
CSV_HEADER="timestamp,source,dataset,config,status,accuracy,top5_accuracy,f1_score,runtime_minutes,error,n_train,n_test,n_id_train,n_id_test,seconds,command"
CALIB_IDS=10

mkdir -p "$(dirname "$OUT_CSV")" "$LOG_DIR"
rm -f "$TMP_WF_CSV"

prepare_output_csv() {
  if [ -s "$OUT_CSV" ]; then
    local ts
    local backup_path
    ts="$(date +%Y%m%d_%H%M%S)"
    backup_path="${OUT_CSV%.csv}_backup_${ts}.csv"
    cp "$OUT_CSV" "$backup_path"
    echo "Backed up existing CSV to: $backup_path"
  fi

  if [ ! -s "$OUT_CSV" ]; then
    echo "$CSV_HEADER" > "$OUT_CSV"
    return
  fi

  local first_line
  first_line="$(head -n 1 "$OUT_CSV" 2>/dev/null || true)"
  if [ "$first_line" != "$CSV_HEADER" ]; then
    local tmp_file
    tmp_file="$(mktemp)"
    {
      echo "$CSV_HEADER"
      cat "$OUT_CSV"
    } > "$tmp_file"
    mv "$tmp_file" "$OUT_CSV"
    echo "[WARN] Existing CSV had no expected header. Header was prepended."
  fi
}

# Edit these lists directly to run any combination you want.
MAIN_DATASETS=(
  #sealid_fewshot
  atrw_fewshot
  cowdataset_fewshot
  elpephants_fewshot
  czoo_fewshot
  seastarreid2023_fewshot
  chicks4freeid_fewshot
  
)

# WildFusion list is intentionally separate and hardcoded.
WILDFUSION_DATASETS=(
  chicks4freeid_fewshot
  sealid_fewshot
  seastarreid2023_fewshot
  cowdataset_fewshot
  elpephants_fewshot
  czoo_fewshot
  atrw_fewshot
)

CONFIGS=(
  fisher_only
  fisher_gv_power
  global_fisher
  fisher_disk
  fisher_aliked
  fisher_superpoint
  global_only
  global_fisher_gv_power
  #global_gv
)

prepare_output_csv

append_main_row() {
  local dataset="$1"
  local config="$2"
  local status="$3"
  local version="$4"
  local error_msg="$5"
  local cmd_str="$6"

  OUTPUT_CSV="$OUT_CSV" \
  ROW_DATASET="$dataset" \
  ROW_CONFIG="$config" \
  ROW_STATUS="$status" \
  ROW_VERSION="$version" \
  ROW_ERROR="$error_msg" \
  ROW_COMMAND="$cmd_str" \
  python3 - <<'PY'
import csv
import datetime as dt
import json
import os
from pathlib import Path

out_csv = os.environ["OUTPUT_CSV"]
dataset = os.environ["ROW_DATASET"]
config = os.environ["ROW_CONFIG"]
status = os.environ["ROW_STATUS"]
version = os.environ["ROW_VERSION"]
error_msg = os.environ.get("ROW_ERROR", "")
cmd_str = os.environ.get("ROW_COMMAND", "")

acc = ""
top5 = ""
f1 = ""
rt_min = ""
seconds = ""
n_train = ""
n_test = ""
n_id_train = ""
n_id_test = ""

if status == "ok":
    candidates = sorted(Path("evaluations/full_evals").glob(f"*_v{version}/{dataset}_evaluation.json"))
    if candidates:
        with open(candidates[-1], "r", encoding="utf-8") as f:
            metrics = json.load(f)
        acc = metrics.get("accuracy", "")
        top5 = metrics.get("top_n_accuracy", "")
        f1 = (((metrics.get("classification_metrics") or {}).get("weighted avg") or {}).get("f1-score", ""))
        eval_sec = metrics.get("eval_runtime_sec", "")
        if isinstance(eval_sec, (int, float)):
            seconds = eval_sec
            rt_min = eval_sec / 60.0
    else:
        status = "error"
        error_msg = f"Missing evaluation JSON for version={version}, dataset={dataset}"

row = [
    dt.datetime.now().isoformat(timespec="seconds"),
    "main",
    dataset,
    config,
    status,
    acc,
    top5,
    f1,
    rt_min,
    error_msg,
    n_train,
    n_test,
    n_id_train,
    n_id_test,
    seconds,
    cmd_str,
]

with open(out_csv, "a", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(row)
PY
}

append_wildfusion_last_row() {
  local dataset="$1"
  local wf_status_override="$2"
  local wf_error_override="$3"
  local cmd_str="$4"

  OUTPUT_CSV="$OUT_CSV" \
  TMP_WF_CSV_PATH="$TMP_WF_CSV" \
  ROW_DATASET="$dataset" \
  WF_STATUS_OVERRIDE="$wf_status_override" \
  WF_ERROR_OVERRIDE="$wf_error_override" \
  ROW_COMMAND="$cmd_str" \
  python3 - <<'PY'
import csv
import datetime as dt
import os
from pathlib import Path

out_csv = os.environ["OUTPUT_CSV"]
tmp_wf_csv = Path(os.environ["TMP_WF_CSV_PATH"])
dataset = os.environ["ROW_DATASET"]
status_override = os.environ.get("WF_STATUS_OVERRIDE", "")
error_override = os.environ.get("WF_ERROR_OVERRIDE", "")
cmd_str = os.environ.get("ROW_COMMAND", "")

status = "error"
error = ""
top1 = ""
top5 = ""
f1 = ""
seconds = ""
n_train = ""
n_test = ""
n_id_train = ""
n_id_test = ""
runtime_minutes = ""

if status_override:
    status = status_override
    error = error_override
else:
    if tmp_wf_csv.exists() and tmp_wf_csv.stat().st_size > 0:
        with open(tmp_wf_csv, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            if str(last.get("dataset", "")).strip().lower() == dataset.lower():
                status = str(last.get("status", "error"))
                error = str(last.get("error", ""))
                top1 = last.get("top1_acc", "")
                top5 = last.get("top5_acc", "")
                f1 = last.get("f1_score", "")
                seconds = last.get("seconds", "")
                n_train = last.get("n_train", "")
                n_test = last.get("n_test", "")
                n_id_train = last.get("n_id_train", "")
                n_id_test = last.get("n_id_test", "")
                try:
                    runtime_minutes = float(seconds) / 60.0
                except Exception:
                    runtime_minutes = ""
            else:
                error = "WildFusion CSV last row does not match requested dataset."
        else:
            error = "WildFusion CSV is empty."
    else:
        error = "WildFusion CSV was not produced."

row = [
    dt.datetime.now().isoformat(timespec="seconds"),
    "wildfusion",
    dataset,
    "wildfusion_baseline",
    status,
    top1,
    top5,
    f1,
    runtime_minutes,
    error,
    n_train,
    n_test,
    n_id_train,
    n_id_test,
    seconds,
    cmd_str,
]

with open(out_csv, "a", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(row)
PY
}

dataset_main_flags() {
  local ds="$1"
  MAIN_FLAGS=()
  case "${ds,,}" in
    atrw|atrw_fewshot)
      MAIN_FLAGS+=(--use_mantiuk --remove_background)
      ;;
    cowdataset|cowdataset_fewshot)
      MAIN_FLAGS+=(--use_mantiuk --remove_background)
      ;;
    elpephants|elpephant|elpephants_fewshot)
      MAIN_FLAGS+=(--use_mantiuk --remove_background)
      ;;
    czoo|czoo_fewshot)
      ;;
    chicks4freeid|chicks4freeid_fewshot)
    MAIN_FLAGS+=()
      ;;
    sealid|sealid_fewshot)
    MAIN_FLAGS+=(--use_mantiuk)
      ;;
    seastarreid2023|seastarreid2023_fewshot)
      MAIN_FLAGS+=(--remove_background)
      ;;
  esac
}

dataset_wildfusion_flags() {
  local ds="$1"
  WF_FLAGS=()
  case "${ds,,}" in
    atrw_fewshot)
      WF_FLAGS+=(--segmented)
      ;;
    cowdataset_fewshot)
      WF_FLAGS+=(--segmented)
      ;;
    elpephants|elpephant|elpephants_fewshot)
      WF_FLAGS+=(--segmented)
      ;;
    czoo|czoo_fewshot)
      ;;
    chicks4freeid|chicks4freeid_fewshot)
      ;;
    sealid|sealid_fewshot)
      ;;
    seastarreid2023|seastarreid2023_fewshot)
      WF_FLAGS+=(--segmented)
      ;;
  esac
}

config_flags() {
  local cfg="$1"
  CFG_FLAGS=()
  case "$cfg" in
    fisher_only)
      CFG_FLAGS+=(--use_fisher --method ensamble --fusion_signals fisher)
      ;;
    fisher_gv_power)
      CFG_FLAGS+=(--use_fisher --method ensamble --use_lightglue --fusion_signals fisher gv)
      ;;
    global_fisher)
      CFG_FLAGS+=(--use_global_embedding --use_fisher --method ensamble --fusion_signals global fisher)
      ;;
    fisher_disk)
      CFG_FLAGS+=(--use_fisher --method disk --fusion_signals fisher)
      ;;
    fisher_aliked)
      CFG_FLAGS+=(--use_fisher --method aliked --fusion_signals fisher)
      ;;
    fisher_superpoint)
      CFG_FLAGS+=(--use_fisher --method superpoint --fusion_signals fisher)
      ;;
    # Supported if you add them to CONFIGS:
    global_fisher_gv_power|fisher_global_gv)
      CFG_FLAGS+=(--use_global_embedding --use_fisher --method ensamble --use_lightglue --fusion_signals global fisher gv)
      ;;
    global_only)
      CFG_FLAGS+=(--use_global_embedding --fusion_signals global)
      ;;
    global_gv)
      CFG_FLAGS+=(--use_global_embedding --use_lightglue --fusion_signals global gv)
      ;;
    *)
      echo "[WARN] Unknown config: $cfg"
      ;;
  esac
}

echo "Writing combined results to: $OUT_CSV"
echo "Starting main pipeline runs..."

for ds in "${MAIN_DATASETS[@]}"; do
  dataset_main_flags "$ds"
  for cfg in "${CONFIGS[@]}"; do
    config_flags "$cfg"

    # Override for cowdataset: force superpoint instead of ensemble
    if [[ "${ds,,}" == "cowdataset" ]]; then
      for i in "${!CFG_FLAGS[@]}"; do
        if [[ "${CFG_FLAGS[i]}" == "ensamble" ]]; then
          CFG_FLAGS[i]="superpoint"
        fi
      done
      # Also ensure GV uses superpoint if GV is active
      CFG_FLAGS+=(--gv_features superpoint)
    fi

    version="final_${ds}_${cfg}"
    log_path="$LOG_DIR/main_${ds}_${cfg}.log"

    cmd=(
      python main.py
      --train
      --ds "$ds"
      --save_eval
      --debug
      --version "$version"
      --calib_ids "$CALIB_IDS"
      "${MAIN_FLAGS[@]}"
      "${CFG_FLAGS[@]}"
    )

    cmd_str="${cmd[*]}"
    echo
    echo "[MAIN] Dataset=$ds Config=$cfg"
    echo "[MAIN] Command: $cmd_str"
    PYTHONUNBUFFERED=1 "${cmd[@]}" 2>&1 | tee "$log_path"
    rc=$?

    if [ "$rc" -ne 0 ]; then
      err_tail="$(tail -n 20 "$log_path" | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g')"
      echo "[MAIN] FAILED (exit=$rc). Continuing. Log: $log_path"
      append_main_row "$ds" "$cfg" "error" "$version" "exit=$rc; $err_tail" "$cmd_str"
      continue
    fi

    echo "[MAIN] OK. Log: $log_path"
    append_main_row "$ds" "$cfg" "ok" "$version" "" "$cmd_str"
  done
done

echo
echo "Starting WildFusion baseline runs..."

if [ ! -x "./venv_wildlife_tools/bin/python" ]; then
  echo "[WF] Missing executable: ./venv_wildlife_tools/bin/python"
  for ds in "${WILDFUSION_DATASETS[@]}"; do
    append_wildfusion_last_row "$ds" "error" "Missing venv_wildlife_tools python executable." "N/A"
  done
else
  for ds in "${WILDFUSION_DATASETS[@]}"; do
    dataset_wildfusion_flags "$ds"
    log_path="$LOG_DIR/wildfusion_${ds}.log"
    wf_cmd=(
      ./venv_wildlife_tools/bin/python
      test-scripts/run_wildfusion_paper_baseline.py
      --ds "$ds"
      --results-csv "$TMP_WF_CSV"
      --calib-ids "$CALIB_IDS"
      "${WF_FLAGS[@]}"
    )
    wf_cmd_str="HF_HUB_OFFLINE=1 ${wf_cmd[*]}"
    echo
    echo "[WF] Dataset=$ds"
    echo "[WF] Command: $wf_cmd_str"
    HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 "${wf_cmd[@]}" 2>&1 | tee "$log_path"
    rc=$?

    if [ "$rc" -ne 0 ]; then
      err_tail="$(tail -n 20 "$log_path" | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g')"
      echo "[WF] FAILED (exit=$rc). Continuing. Log: $log_path"
      append_wildfusion_last_row "$ds" "error" "exit=$rc; $err_tail" "$wf_cmd_str"
      continue
    fi

    echo "[WF] OK. Log: $log_path"
    append_wildfusion_last_row "$ds" "" "" "$wf_cmd_str"
  done
fi

echo
echo "Done. Combined CSV written to: $OUT_CSV"
