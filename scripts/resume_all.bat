@echo off
rem 断点续跑（Windows 双击版）：自动跳过已完成的训练，继续剩余实验并生成报告
chcp 65001 >nul
cd /d "%~dp0.."
echo [*] 检查已完成训练并继续剩余实验（完整模型/消融/基线/评估/报告）...
bash scripts\resume_all.sh
echo.
echo [完成] 实验报告: docs\实验报告.md  图表: docs\figures\
pause
