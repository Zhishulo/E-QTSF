# -*- coding: utf-8 -*-
"""E-QTSF 模型配置（全部超参数集中管理，含消融开关）。"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EQTSFConfig:
    # ---------- 数据 ----------
    data_dir: str = "data/output"
    channels: List[str] = field(default_factory=lambda: [
        "pass_rate", "ppm", "dim_rate", "app_rate", "volume"])
    context_len: int = 84          # 自回归历史段窗口（12 周）
    ctx_multiscale_len: int = 180  # 多尺度编码器回看长度
    seg_len: int = 7               # 段长度（周段，对齐 EVENTTSF 固定窗口分段）
    forecast_segs: int = 4         # 自回归预测段数（4 段 × 7 天 = 28 天）
    stride: int = 7                # 训练窗口滑动步长
    min_history: int = 200         # 样本要求的最小历史长度

    # ---------- 事件 ----------
    n_event_types: int = 6         # equipment/tooling/material/process/personnel/measure
    n_event_subtypes: int = 13
    n_severity: int = 4            # 致命/严重/一般/轻微
    event_emb_dim: int = 64

    # ---------- 编码器 ----------
    d_model: int = 96
    n_heads: int = 4
    ms_layers: int = 2             # 多尺度分支层数
    ms_dropout: float = 0.1

    # ---------- 去噪器（流匹配） ----------
    denoise_layers: int = 4
    d_ff: int = 256
    dropout: float = 0.1

    # ---------- 事件感知时间步 ----------
    use_event_timestep: bool = True
    beta_eps: float = 0.02         # Beta 参数下限（防 0/1 退化）

    # ---------- 对比学习 ----------
    use_contrastive: bool = True
    contrast_weight: float = 0.3   # λ
    similarity_weight: float = 0.5  # γ
    contrast_temp: float = 0.2     # τ
    pos_threshold: float = 0.75    # ρ
    neg_threshold: float = 0.45    # ρ'
    topk_pos: int = 4
    lambda_start: float = 0.3      # 启发式权重起点
    lambda_end: float = 0.9        # 收敛到启发式权重

    # ---------- 消融开关 ----------
    use_events: bool = True        # False: 事件条件置空（w/o Event）
    use_multiscale: bool = True    # False: 仅日视图（w/o Multi-scale）

    # ---------- 训练 ----------
    batch_size: int = 64
    epochs: int = 40
    lr: float = 2e-3
    weight_decay: float = 1e-4
    n_samples: int = 100           # 概率预测采样轨迹数
    seed: int = 2024
    device: str = "cuda"
    amp: bool = True

    # ---------- 输出 ----------
    run_dir: str = "runs"
    model_name: str = "eqtsf"

    @property
    def horizon(self) -> int:
        return self.seg_len * self.forecast_segs

    @property
    def n_channels(self) -> int:
        return len(self.channels)
