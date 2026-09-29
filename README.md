# E-QTSF：事件感知质量时序预测模型 + 机械加工质量管理系统

以 **模型创新为核心**（自研 E-QTSF，参考 EVENTTSF 与 EventTime 两篇论文并做质量域扩展），
以《机械加工企业质量管理系统的设计与实现》的业务模型为系统底座，
实现「**数据 → 事件 → 概率预测 → 预警 → NCR 闭环**」的完整落地系统。

## 一、快速开始（Windows）

```bash
cd eqtsf-system
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
# GPU（RTX 4060，CUDA 12.4）：
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu124

# 1) 生成模拟数据（10 产品 × 730 天 + 事件注入 + SQLite/CSV）
.venv\Scripts\python.exe data\generator.py

# 2) 训练 + 基线 + 消融 + 图表（约 40 分钟，GPU）
bash scripts\run_all.sh          # 或逐条运行 model\train.py / baselines.py / evaluate.py

#    训练中断（断电/中断）后可断点续跑：自动跳过已完成的 run，继续剩余实验并生成报告
scripts\resume_all.bat           # 或 bash scripts\resume_all.sh

# 3) 启动系统（浏览器自动打开）
scripts\start.bat                # http://127.0.0.1:8000  账号 admin / 123456
```

账号：`admin/123456`（管理员）、`insp/123456`（检验员）、`eng/123456`（工程师）。

## 二、系统功能

| 模块 | 内容 |
|---|---|
| 质量看板 | 合格率/PPM 趋势、缺陷 Top5、部门分布、产品排行 |
| 基础数据 | 产品 / 批次 / 质量记录 / 缺陷代码（自动编号 PRD-/BATCH-/NCR-） |
| 事件管理 | 六大类质量事件（设备/刀具/物料/工艺/人员/质量措施）+ 严重度四档 |
| NCR 闭环 | 待评审 → 处理中 → 已关闭/驳回（状态机 + 日志 + 非法流转拒绝） |
| SPC | X̄-R 控制图（n=5 实测值）、P 图、PPM、失控点统计 |
| **预测预警** | **E-QTSF 28 天概率预测 + 超标概率 + 事件归因 → 一键生成 NCR** |
| 用户管理 | JWT + PBKDF2 + RBAC（管理员/检验员/工程师） |

## 三、模型（核心创新）

`model/` 目录：

```
dataset.py     窗口构建（84天上下文→28天预测）+ 多尺度视图 + 事件段对齐 + RevIN
modules.py     事件编码器 / 多尺度编码器 / translate-locate-imprint 事件融合 /
               事件感知时间步 / 流匹配去噪器
beta_icdf.py   可微 Beta CDF/ICDF（连分式 + logit 牛顿，PyTorch 未实现 icdf）
contrastive.py 动态对比学习（启发式↔学习式相似度渐进切换，解决高严重度事件稀疏）
model.py       E-QTSF 整体：自回归潜状态 + 采样 + 事件消融归因
metrics.py     CRPS / WQL / MAE / MSE / RMSE + 预警命中率/提前期/误报率
train.py       训练（含 4 组消融开关）  baselines.py  4 个基线
evaluate.py    评估 + 论文图表
```

**相对 EVENTTSF 的创新点：**
1. **质量事件体系 + 严重度条件化时间步**：六类事件本体（类型/子类型/严重度），
   Beta(α,β) 时间步重参数化的 α/β 除条件于事件嵌入外，额外注入严重度权重
   （致命事件 → 更精细的自适应去噪调度）；
2. **多尺度质量上下文**（EventTime）：日/周/月聚合视图 + 短期窗口，
   跨视图交叉注意力，捕捉设备劣化周期与季节规律；
3. **动态对比学习**（EventTime）：启发式（时序统计+事件属性）与学习式相似度
   动态加权（λ: 0.3→0.9），top-k 正对 InfoNCE + 相似度正则，
   缓解高严重度事件（占比仅 ~8%）样本稀疏；
4. **可解释归因 + 预警闭环**：translate-locate-imprint 注意力定位 + 逐事件消融
   |Δ| 量化每个未来事件对预测的影响，超标概率联动阈值生成红/橙/黄预警并可一键转 NCR。

## 四、实验结果（详见 docs/实验报告.md，2026-09-29 修复训练协议后全量重跑）

**模拟数据**（10 产品 × 730 天，~310 注入事件，测试集 = 最后 15% 窗口）：

| 模型 | 测试 CRPS↓ | 说明 |
|---|---|---|
| **E-QTSF 完整** | **0.7596** | 优于最强基线 iTransformer-lite（0.7888）3.7% |
| w/o 多尺度编码 | 0.8017 | 多尺度贡献 +0.042 |
| w/o 事件感知时间步 | 0.7993 | 时间步贡献 +0.040 |
| w/o 事件条件化 | 0.7813 | 事件通路贡献 +0.022 |
| w/o 动态对比学习 | 0.7602 | 近中性，价值在表征结构与可解释归因 |

- 3 种子 std 0.0152 < 消融间距；PICP-80 = 0.791（名义 0.80）
- 跨制度迁移（平滑训练 → shock 测试不重训）：事件增益 +0.022 → +0.146，学到的是推理而非查表
- 动态阈值预警：命中率 52.1%，平均提前 6.1 天（全局固定阈值仅 5.9%）

**真实数据验证（M5，Walmart 真实零售 + 官方事件日历，154 个真实事件日）**：
事件条件化贡献 +5.61、事件感知时间步贡献 +5.71 CRPS（去掉任一组件显著变差），
PICP-80 = 0.774，事件窗优于平静窗——模型创新的机制价值获得真实数据支持；
整体 CRPS 58.15 落后最强基线 PatchTST（54.64）约 6.4%，差距集中在非事件时段（诚实记录）。

> 注：早期版本（2026-09-28/29 前）的训练协议存在「训练窗口覆盖测试期」的时间泄漏，
> 相关数字已作废并在报告第 0 节更正。

## 五、真实数据（M5）复现

```bash
# M5 原始数据（calendar.csv + sales_train_validation.csv，约 120MB）
# 从 HuggingFace 镜像下载后放入 data/real/（国内可达）：
curl -L -o data/real/calendar.csv "https://hf-mirror.com/datasets/denephew/M5_Forecasting/resolve/main/calendar.csv"
curl -L -o data/real/sales_train_validation.csv "https://hf-mirror.com/datasets/denephew/M5_Forecasting/resolve/main/sales_train_validation.csv"

# 适配（Top-10 商品聚合 + 事件映射到质量本体）→ data/output_m5/
.venv/Scripts/python.exe scripts/build_m5_dataset.py

# 训练 + 基线 + 评估（产物在 runs_m5/）
.venv/Scripts/python.exe model/train.py --data_dir data/output_m5 --run_dir runs_m5/eqtsf_full --stride 7 --epochs 40
.venv/Scripts/python.exe scripts/run_baselines_m5.py
.venv/Scripts/python.exe model/evaluate.py --runs_dir runs_m5 --data_dir data/output_m5
```

## 六、目录结构

```
eqtsf-system/
├── data/            schema.sql + generator.py + output/(qms.db, series_*.csv, ...)
├── model/           E-QTSF 模型 + 训练/评估/基线
├── server/          FastAPI 服务端（单文件 main.py：CRUD/SPC/推理/认证）
├── web/             Vue3 前端（vendor 本地化，离线可用）
├── docs/            设计文档 + 实验报告 + figures/
├── scripts/         start.bat 启动 / run_all.sh 训练流水线
└── runs/            训练产物（results.json/log.json 全量 + 两个主模型 best.pt；
                     其余权重可由 run_all.sh 复现）
```

## 七、License

MIT License（见 [LICENSE](LICENSE)）。M5 数据集版权归其发布方（Kaggle M5 competition）所有，
本仓库仅含经适配的派生小文件，原始数据请从官方渠道获取。
