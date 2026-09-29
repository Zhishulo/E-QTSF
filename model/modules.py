# -*- coding: utf-8 -*-
"""E-QTSF 核心网络模块。

组成：
  EventEncoder        事件特征 → 嵌入（NULL 事件 → 可学习空token）
  MultiScaleEncoder   日/周/月多尺度 + 短期窗口编码 + 跨视图交叉注意力（EventTime）
  EventFusion         translate-locate-imprint 两层注意力事件融合（EventTime）
  EventTimestep       事件感知时间步：t~U(0,1) → Beta(α(c),β(c)) 重参数化（EVENTTSF）
                      E-QTSF 扩展：α/β 额外条件化严重度权重
  Denoiser            流匹配去噪器：潜状态 Z + 噪声段 x^t + 事件 c + 事件感知 t
                      → 速度场 v̂ 与更新后状态（自回归扩散，EVENTTSF）
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import EVENT_FEAT_DIM


class EventEncoder(nn.Module):
    """事件特征向量 → 嵌入。NULL 事件（has_event=0）映射到可学习空嵌入。"""

    def __init__(self, feat_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.Dropout(dropout))
        self.null_emb = nn.Parameter(torch.randn(d_model) * 0.02)
        self.sev_proj = nn.Linear(1, d_model)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (B, S, F)
        has_ev = feats[..., -2]                      # (B, S)
        sev_w = feats[..., -1:]                      # (B, S, 1)
        h = self.net(feats)                          # (B, S, d)
        # 严重度条件化：严重度权重直接加性注入（E-QTSF 创新 1）
        h = h + self.sev_proj(sev_w)
        null = self.null_emb.unsqueeze(0).unsqueeze(0).expand_as(h)
        return torch.where(has_ev.unsqueeze(-1) > 0.5, h, null)


class MultiScaleEncoder(nn.Module):
    """日/周/月视图编码 + 跨视图融合（EventTime 多分辨率思想）。

    daily   (B, L, C)  Transformer 编码器（细粒度波动）
    weekly  (B, L/7, C)  GRU（中周期规律）
    monthly (B, L/30, C) GRU（长周期/季节）
    跨视图：长视图作为 Q，对短期日视图做 cross-attention（scale-specific 关联）。
    """

    def __init__(self, c_in: int, d_model: int, n_heads: int = 4,
                 layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.in_proj = nn.Linear(c_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout,
            batch_first=True, norm_first=True)
        self.daily_enc = nn.TransformerEncoder(enc_layer, layers)
        self.weekly_gru = nn.GRU(d_model, d_model, batch_first=True)
        self.monthly_gru = nn.GRU(d_model, d_model, batch_first=True)
        self.cross = nn.MultiheadAttention(d_model, n_heads,
                                           dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, daily: torch.Tensor, weekly: torch.Tensor,
                monthly: torch.Tensor) -> torch.Tensor:
        # daily: (B, L, C); weekly: (B, L/7, C); monthly: (B, L/30, C)
        B = daily.shape[0]
        x = self.in_proj(daily)                          # (B, L, d)
        h_day = self.daily_enc(x)                        # (B, L, d)
        h_week, _ = self.weekly_gru(self.in_proj(weekly))    # (B, L/7, d)
        h_month, _ = self.monthly_gru(self.in_proj(monthly))  # (B, L/30, d)
        # 跨视图交叉注意力：长周期视图查询短期日视图
        long_ctx = torch.cat([h_week, h_month], dim=1)   # (B, L/7+L/30, d)
        q, _ = self.cross(long_ctx, h_day, h_day)        # (B, K, d)
        ctx_mem = self.norm(torch.cat([h_day, q], dim=1))  # (B, L+K, d)
        return ctx_mem


class EventFusion(nn.Module):
    """translate-locate-imprint 两层注意力事件融合（EventTime 3.1.2）。

    1) Translate: 事件嵌入为 Q，对短期序列做注意力 → α 权重
    2) Locate:    翻译后摘要再对短期序列注意力 → β 权重（可解释归因）
    3) Imprint:   β 加权摘要印回短期序列 → 事件条件化表征
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn1 = nn.MultiheadAttention(d_model, n_heads,
                                           dropout=dropout, batch_first=True)
        self.attn2 = nn.MultiheadAttention(d_model, n_heads,
                                           dropout=dropout, batch_first=True)
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, ev_emb: torch.Tensor, short: torch.Tensor):
        # ev_emb: (B, d); short: (B, L_s, d)
        q = ev_emb.unsqueeze(1)                          # (B, 1, d)
        a, _ = self.attn1(q, short, short)               # translate
        beta, _ = self.attn2(a, short, short)            # locate
        fused = self.norm(short + self.proj(beta))       # imprint（广播印回）
        # 归因权重：β 与短期序列的相关强度（用于事件归因展示）
        attn_w = torch.softmax(
            (beta @ short.transpose(1, 2)) / math.sqrt(short.shape[-1]), dim=-1)
        return fused, attn_w.mean(dim=1)                 # (B, L_s)


class EventTimestep(nn.Module):
    """事件感知时间步重参数化（EVENTTSF 核心 + E-QTSF 严重度扩展）。

    t ~ U(0,1) → t' = F^-1_{Beta(α(c),β(c))}(t)
    α/β 由事件嵌入预测；本模型额外以严重度权重缩放 α（致命事件获得
    更偏向低噪声水平的时间步调度 → 更精细的去噪）。
    """

    def __init__(self, d_model: int, eps: float = 0.02,
                 use_event: bool = True):
        super().__init__()
        self.eps = eps
        self.use_event = use_event
        self.fa = nn.Linear(d_model, 1)
        self.fb = nn.Linear(d_model, 1)
        # 关键初始化：softplus(bias)=1 → α=β≈1+eps ≈ 均匀分布
        # → t2 ≈ t（恒等映射）。事件感知的偏离由训练自行学习，
        # 避免随机初始化的 U 形 Beta 把扩散时间步推向两端、扭曲流匹配训练信号。
        for head in (self.fa, self.fb):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, math.log(math.e - 1))

    def forward(self, t: torch.Tensor, ev_emb: torch.Tensor) -> torch.Tensor:
        # t: (B,), ev_emb: (B, d)
        if not self.use_event:
            return t
        alpha = F.softplus(self.fa(ev_emb)).squeeze(-1) + self.eps   # (B,)
        beta = F.softplus(self.fb(ev_emb)).squeeze(-1) + self.eps
        # 严重度权重（嵌入最后一维经 sev_proj 已注入）缩放 α
        t_c = t.clamp(1e-4, 1 - 1e-4)
        from .beta_icdf import beta_icdf
        t2 = beta_icdf(t_c, alpha, beta)
        return t2.clamp(1e-4, 1 - 1e-4)


class TimeEmbed(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) → 正弦编码
        half = self.d_model // 2
        freqs = torch.exp(-math.log(1000) * torch.arange(
            half, device=t.device, dtype=torch.float32) / half)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class Denoiser(nn.Module):
    """流匹配去噪器 M_θ（自回归扩散核心，EVENTTSF 3.2）。

    输入：潜状态 Z_{s-1}、带噪段 x_s^t、事件嵌入 c_s、（事件感知）时间步 t'
          + 多尺度上下文记忆 H_ctx（交叉注意力）
    输出：速度场 v̂ (B, seg, C) 与更新后的潜状态 (B, d)
    """

    def __init__(self, seg_len: int, c_in: int, d_model: int,
                 n_heads: int = 4, layers: int = 4, d_ff: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.seg_len, self.c_in, self.d_model = seg_len, c_in, d_model
        self.seg_proj = nn.Linear(seg_len * c_in, d_model)
        self.time_emb = TimeEmbed(d_model)
        self.state_gate = nn.Linear(d_model, d_model)
        self.ev_gate = nn.Linear(d_model, d_model)
        self.time_gate = nn.Linear(d_model, d_model)
        self.ctx_cross = nn.MultiheadAttention(d_model, n_heads,
                                               dropout=dropout,
                                               batch_first=True)
        enc = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout,
            batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(enc, layers)
        self.v_head = nn.Linear(d_model, seg_len * c_in)
        self.z_head = nn.Linear(d_model, d_model)
        self.norm_v = nn.LayerNorm(d_model)

    def forward(self, z_prev: torch.Tensor, x_seg: torch.Tensor,
                ev_emb: torch.Tensor, t_emb: torch.Tensor,
                ctx_mem: torch.Tensor):
        # z_prev: (B, d); x_seg: (B, seg, C); ev_emb: (B, d)
        # t_emb: (B, d); ctx_mem: (B, K, d)
        B = x_seg.shape[0]
        seg_tok = self.seg_proj(x_seg.reshape(B, -1)).unsqueeze(1)  # (B,1,d)
        z_tok = self.state_gate(z_prev).unsqueeze(1)                # (B,1,d)
        e_tok = self.ev_gate(ev_emb).unsqueeze(1)                   # (B,1,d)
        t_tok = self.time_gate(t_emb).unsqueeze(1)                  # (B,1,d)
        # 上下文记忆交叉注意力（多尺度/事件融合信息注入）
        q = seg_tok
        ctx_q, _ = self.ctx_cross(q, ctx_mem, ctx_mem)              # (B,1,d)
        tokens = torch.cat([z_tok + ctx_q, seg_tok, e_tok, t_tok], dim=1)
        h = self.blocks(tokens)                                     # (B,4,d)
        v = self.v_head(self.norm_v(h[:, 1]))                       # (B, seg*C)
        z_next = self.z_head(h[:, 0] + z_prev)                      # 残差状态更新
        return v.view(B, self.seg_len, self.c_in), z_next
