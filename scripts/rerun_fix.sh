#!/usr/bin/env bash
# 修正版 Phase B/C/D：train.py 的 --run_dir 直接作为产物目录，
# 必须传到 eqtsf_* 子目录级；同时清理旧协议残留 checkpoint。
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
LOG=data/output
mkdir -p "$LOG"

run() {
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] START $name"
  "$@" > "$LOG/rerunfix_$name.log" 2>&1
  local code=$?
  echo "[$(date +%H:%M:%S)] EXIT=$code $name"
  echo "$name EXIT=$code" >> "$LOG/rerunfix_status.txt"
}

rm -f "$LOG/rerunfix_status.txt"
rm -rf runs_shock/eqtsf_full runs_shock/eqtsf_no_event runs_shock/eqtsf_no_timestep
rm -f runs_shock/best.pt runs_shock/log.json
rm -f runs_m5/best.pt runs_m5/log.json
rm -rf runs_seed2/eqtsf_full runs_seed3/eqtsf_full
rm -f runs_seed2/best.pt runs_seed2/log.json runs_seed3/best.pt runs_seed3/log.json

# ---- Phase B'：shock 尖锐事件制度（新协议、正确产物目录）----
run shock_full        $PY model/train.py --data_dir data/output_shock --run_dir runs_shock/eqtsf_full --epochs 40
run shock_no_event    $PY model/train.py --ablation no_event --data_dir data/output_shock --run_dir runs_shock/eqtsf_no_event --epochs 40
run shock_no_timestep $PY model/train.py --ablation no_timestep --data_dir data/output_shock --run_dir runs_shock/eqtsf_no_timestep --epochs 40
run shock_evaluate    $PY model/evaluate.py --runs_dir runs_shock --data_dir data/output_shock

# ---- Phase C'：M5 真实数据（新协议、正确产物目录）----
run m5_full        $PY model/train.py --data_dir data/output_m5 --run_dir runs_m5/eqtsf_full --stride 7 --epochs 40
run m5_no_event    $PY model/train.py --ablation no_event --data_dir data/output_m5 --run_dir runs_m5/eqtsf_no_event --stride 7 --epochs 40
run m5_no_timestep $PY model/train.py --ablation no_timestep --data_dir data/output_m5 --run_dir runs_m5/eqtsf_no_timestep --stride 7 --epochs 40
run m5_evaluate    $PY model/evaluate.py --runs_dir runs_m5 --data_dir data/output_m5

# ---- Phase D'：多种子（新协议）----
run seed2      $PY model/train.py --seed 2025 --run_dir runs_seed2/eqtsf_full --epochs 40
run seed3      $PY model/train.py --seed 2026 --run_dir runs_seed3/eqtsf_full --epochs 40
run seed2_eval $PY model/evaluate.py --runs_dir runs_seed2
run seed3_eval $PY model/evaluate.py --runs_dir runs_seed3

echo "RERUNFIX_DONE" >> "$LOG/rerunfix_status.txt"
