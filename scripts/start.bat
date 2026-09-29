@echo off
rem 机械加工质量管理系统 · 一键启动（Windows）
rem 用法：双击运行，或命令行执行 scripts\start.bat
chcp 65001 >nul
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
  echo [!] 未找到虚拟环境，请先运行: python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
  pause & exit /b 1
)

if not exist "data\output\qms.db" (
  echo [*] 数据库不存在，先生成模拟数据...
  .venv\Scripts\python.exe data\generator.py
)

if not exist "runs\eqtsf_full\best.pt" (
  echo [!] 模型未训练，预测功能不可用（可运行 scripts\run_all.sh 训练，约 1 小时）
)

echo [*] 启动服务: http://127.0.0.1:8000  (账号 admin / 123456)
start "" http://127.0.0.1:8000
.venv\Scripts\python.exe -m uvicorn server.main:app --host 127.0.0.1 --port 8000
