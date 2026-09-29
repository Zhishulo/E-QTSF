#!/usr/bin/env bash
# 协议修复后的全量重跑：
#   Phase A 模拟数据主套件（5 训练+4 基线+评估）
#   Phase B shock 机制三组 + 跨制度迁移评估
#   Phase C M5 真实数据三组 + 基线 + 评估（真实数据验证）
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
LOG=data/output
mkdir -p "$LOG" runs runs_shock runs_m5 runs_m5/baselines

run() {
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] START $name"
  "$@" > "$LOG/rerun_$name.log" 2>&1
  local code=$?
  echo "[$(date +%H:%M:%S)] EXIT=$code $name"
  echo "$name EXIT=$code" >> "$LOG/rerun_status.txt"
}

rm -f "$LOG/rerun_status.txt"

# ---- Phase A：模拟数据主套件（修复 split 泄漏后的协议）----
run full          $PY model/train.py --epochs 40
run no_event      $PY model/train.py --ablation no_event --epochs 40
run no_timestep   $PY model/train.py --ablation no_timestep --epochs 40
run no_contrast   $PY model/train.py --ablation no_contrast --epochs 40
run no_multiscale $PY model/train.py --ablation no_multiscale --epochs 40
run baselines     $PY model/baselines.py --epochs 30
run evaluate      $PY model/evaluate.py
cp runs/results.json runs/results_smooth_sim.json

# ---- Phase B：shock 尖锐事件制度（机制验证 + 迁移）----
run shock_full        $PY model/train.py --data_dir data/output_shock --run_dir runs_shock --epochs 40
run shock_no_event    $PY model/train.py --ablation no_event --data_dir data/output_shock --run_dir runs_shock --epochs 40
run shock_no_timestep $PY model/train.py --ablation no_timestep --data_dir data/output_shock --run_dir runs_shock --epochs 40
run shock_evaluate    $PY model/evaluate.py --runs_dir runs_shock --data_dir data/output_shock
# 跨制度迁移：平滑制度训练的模型直接测 shock 测试集（结果另存，随后恢复主结果）
run transfer_evaluate $PY model/evaluate.py --runs_dir runs --data_dir data/output_shock
mv runs/results.json runs/results_transfer_shock.json
cp runs/results_smooth_sim.json runs/results.json

# ---- Phase C：M5 真实数据（Walmart 销量 + 真实事件日历）----
run m5_full        $PY model/train.py --data_dir data/output_m5 --run_dir runs_m5 --stride 7 --epochs 40
run m5_no_event    $PY model/train.py --ablation no_event --data_dir data/output_m5 --run_dir runs_m5 --stride 7 --epochs 40
run m5_no_timestep $PY model/train.py --ablation no_timestep --data_dir data/output_m5 --run_dir runs_m5 --stride 7 --epochs 40
run m5_baselines   $PY scripts/run_baselines_m5.py --epochs 30
run m5_evaluate    $PY model/evaluate.py --runs_dir runs_m5 --data_dir data/output_m5

echo "RERUN_PIPELINE_DONE" >> "$LOG/rerun_status.txt"
