# -*- coding: utf-8 -*-
"""在 M5 真实数据目录上运行基线（复用 model/baselines.py，重绑目录全局量）。

运行方式：项目根目录下 .venv/Scripts/python.exe scripts/run_baselines_m5.py
产物：runs_m5/baselines/baselines.json（与 evaluate.py --runs_dir runs_m5 对接）
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import model.baselines as B  # noqa: E402

B.DATA_DIR = "data/output_m5"
B.RUNS_DIR = "runs_m5/baselines"

if __name__ == "__main__":
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(B.RUNS_DIR, exist_ok=True)
    results = []
    results.append(B.train_baseline("seasonal_naive"))
    for name in ["dlinear", "itransformer", "patchtst"]:
        print("=== baseline", name, "===")
        results.append(B.train_baseline(name, epochs=args.epochs))
    out_path = B.RUNS_DIR + "/baselines.json"
    Path(out_path).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BASELINES_DONE")
    for r in results:
        print(r)
