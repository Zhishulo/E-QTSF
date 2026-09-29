#!/usr/bin/env bash
# 全量实验流水线：主模型 + 4 消融 + 基线 + 评估图表
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
LOG=data/output
mkdir -p "$LOG" runs

run() {  # run <name> <cmd...>
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] START $name"
  "$@" > "$LOG/run_$name.log" 2>&1
  local code=$?
  echo "[$(date +%H:%M:%S)] EXIT=$code $name"
  echo "$name EXIT=$code" >> "$LOG/pipeline_status.txt"
}

rm -f "$LOG/pipeline_status.txt"
run full        $PY model/train.py --epochs 40
run no_event    $PY model/train.py --ablation no_event --epochs 40
run no_timestep $PY model/train.py --ablation no_timestep --epochs 40
run no_contrast $PY model/train.py --ablation no_contrast --epochs 40
run no_multiscale $PY model/train.py --ablation no_multiscale --epochs 40
run baselines   $PY model/baselines.py --epochs 30
run evaluate    $PY model/evaluate.py
echo "PIPELINE_DONE" >> "$LOG/pipeline_status.txt"
