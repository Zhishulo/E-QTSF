#!/usr/bin/env bash
# 断点续跑：自动跳过已完成的训练（以 runs/<run>/log.json 是否存在为准）
# 重启电脑后直接运行本脚本即可继续剩余实验。
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
LOG=data/output
mkdir -p "$LOG" runs

done_run() { [ -f "runs/$1/log.json" ]; }

run() {
  local name="$1"; shift
  if done_run "$name"; then
    echo "[$(date +%H:%M:%S)] SKIP $name (已完成)"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] START $name"
  "$@" > "$LOG/run_$name.log" 2>&1
  local code=$?
  echo "[$(date +%H:%M:%S)] EXIT=$code $name"
  echo "$name EXIT=$code" >> "$LOG/pipeline_status.txt"
}

# 模型训练（已完成的自动跳过）
for ab in none no_event no_timestep no_contrast no_multiscale; do
  rn=$([ "$ab" = none ] && echo eqtsf_full || echo "eqtsf_$ab")
  if done_run "$rn"; then
    echo "[$(date +%H:%M:%S)] SKIP $rn (已完成)"
  else
    if [ "$ab" = none ]; then
      run "$rn" $PY model/train.py --epochs 40
    else
      run "$rn" $PY model/train.py --ablation "$ab" --epochs 40
    fi
  fi
done

# 基线（baselines.json 存在则跳过）
if [ -f "runs/baselines/baselines.json" ]; then
  echo "SKIP baselines (已完成)"
else
  run baselines $PY model/baselines.py --epochs 30
fi

# 评估（results.json 存在则跳过）
if [ -f "runs/results.json" ]; then
  echo "SKIP evaluate (已完成)"
else
  run evaluate $PY model/evaluate.py
fi

# 生成实验报告
$PY scripts/make_report.py > "$LOG/report.log" 2>&1
echo "[$(date +%H:%M:%S)] 报告: $(tail -1 "$LOG/report.log")"
echo "RESUME_DONE" >> "$LOG/pipeline_status.txt"
echo "全部完成。报告: docs/实验报告.md"
