# -*- coding: utf-8 -*-
"""E-QTSF：事件感知质量时序预测模型（自回归流匹配）。

架构总览（EVENTTSF 自回归扩散 + EventTime 多尺度/对比学习 + 质量域扩展）：
  1. 多尺度上下文编码器（日/周/月 + 短期窗口）→ 上下文记忆
  2. translate-locate-imprint 事件融合 → 事件条件化短期表征 + 可解释归因
  3. 自回归流匹配去噪器（潜状态 Z 逐段演化）
  4. 事件感知时间步重参数化 Beta(α(c),β(c))，α 受严重度权重缩放
  5. 动态对比学习（启发式+学习式相似度动态加权）
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contrastive import DynamicContrastive
from .dataset import EVENT_FEAT_DIM, FORE_DAYS, SEG
from .modules import (Denoiser, EventEncoder, EventFusion, EventTimestep,
                      MultiScaleEncoder, TimeEmbed)


class EQTSF(nn.Module):
    def __init__(self, c_in: int = 5, d_model: int = 96, n_heads: int = 4,
                 seg_len: int = SEG, n_seg_hist: int = 12, n_seg_fut: int = 4,
                 layers: int = 4, dropout: float = 0.15,
                 use_events: bool = True,
                 use_event_timestep: bool = True,
                 use_multiscale: bool = True, use_contrastive: bool = True):
        super().__init__()
        self.c_in, self.d_model = c_in, d_model
        self.seg_len, self.n_seg_fut = seg_len, n_seg_fut
        self.n_seg_hist = n_seg_hist
        self.use_events = use_events
        self.use_event_timestep = use_event_timestep
        self.use_multiscale = use_multiscale
        self.use_contrastive = use_contrastive

        self.ev_enc = EventEncoder(EVENT_FEAT_DIM, d_model, dropout)
        self.ms_enc = MultiScaleEncoder(c_in, d_model, n_heads, layers=2,
                                        dropout=dropout)
        self.ev_fuse = EventFusion(d_model, n_heads, dropout)
        self.timestep = EventTimestep(d_model, use_event=use_event_timestep)
        self.denoiser = Denoiser(seg_len, c_in, d_model, n_heads, layers,
                                 dropout=dropout)
        self.time_emb = TimeEmbed(d_model)
        self.short_proj = nn.Linear(c_in, d_model)
        self.short_enc = nn.GRU(d_model, d_model, batch_first=True)
        self.fuse_rep = nn.Linear(d_model * 2, d_model)
        self.z0 = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.contrastive = DynamicContrastive(d_model)

    # -------------------- 上下文 --------------------
    def build_context(self, daily, weekly, monthly, short, hist_events,
                      fut_events):
        """多尺度上下文记忆 + 事件融合表征 + 归因权重。"""
        B = daily.shape[0]
        if self.use_multiscale:
            ctx_mem = self.ms_enc(daily, weekly, monthly)      # (B, L+K, d)
        else:  # 消融：仅日视图
            ctx_mem = self.ms_enc.daily_enc(self.ms_enc.in_proj(daily))
        hist_e = self.ev_enc(hist_events)                      # (B, 12, d)
        # 短期窗口编码 + 与首个未来事件融合（translate-locate-imprint）
        short_tok = self.short_enc(self.short_proj(short))[0]  # (B, 28, d)
        ev0 = self.ev_enc(fut_events[:, :1])                   # (B, 1, d)
        if not self.use_events:
            null = self.ev_enc.null_emb.unsqueeze(0).unsqueeze(0)
            ev0 = null.expand(B, 1, -1)
            hist_e = null.expand_as(hist_e)
        fused_short, attn_w = self.ev_fuse(ev0.squeeze(1), short_tok)
        rep = self.fuse_rep(torch.cat([
            fused_short.mean(dim=1), ctx_mem.mean(dim=1)], dim=-1))
        return ctx_mem, rep, attn_w, hist_e

    def init_state(self, ctx, hist_events, ctx_mem):
        """由干净历史段自回归构建初始潜状态（推理用，t=1）。"""
        B = ctx.shape[0]
        Z = self.z0.expand(B, -1)
        t_emb = self.time_emb(torch.ones(B, device=ctx.device))
        for s in range(self.n_seg_hist):
            x_s = ctx[:, s * self.seg_len:(s + 1) * self.seg_len]
            c_s = hist_events[:, s]                            # 已是嵌入
            _, Z = self.denoiser(Z, x_s, c_s, t_emb, ctx_mem)
        return Z

    # -------------------- 训练前向 --------------------
    def forward(self, batch: Dict[str, torch.Tensor], epoch: float = 0.0,
                total_epochs: float = 1.0) -> Dict[str, torch.Tensor]:
        dev = batch["ctx"].device
        ctx_mem, rep, attn_w, hist_e = self.build_context(
            batch["daily"], batch["weekly"], batch["monthly"],
            batch["short"], batch["hist_events"], batch["fut_events"])
        B = ctx_mem.shape[0]
        x = batch["target"]                                    # (B, 28, C)
        fut_e = self.ev_enc(batch["fut_events"])               # (B, 4, d)
        Z = self.z0.expand(B, -1)
        # 一次性批量计算全部段的事件感知时间步（1 次 ICDF 调用替代 4 次，
        # 消除 Python 循环派发开销——这是完整模型训练的主要瓶颈）
        S = self.n_seg_fut
        t_all = torch.rand(B * S, device=dev)
        if self.use_events:
            c_all = fut_e.reshape(B * S, -1)
        else:
            c_all = self.ev_enc.null_emb.unsqueeze(0).expand(B * S, -1)
        t2_all = self.timestep(t_all, c_all)
        l_fm = x.new_zeros(())
        for s in range(S):
            c_s = fut_e[:, s]
            if not self.use_events:
                c_s = self.ev_enc.null_emb.unsqueeze(0).expand(B, -1)
            t2 = t2_all[s * B:(s + 1) * B]
            x_s = x[:, s * self.seg_len:(s + 1) * self.seg_len]
            eps = torch.randn_like(x_s)
            # OT 路径：t=0 数据 → t=1 噪声（Eq.1 of EVENTTSF）
            x_t = ((1 - t2).view(B, 1, 1) * x_s
                   + t2.view(B, 1, 1) * eps)
            v_tgt = eps - x_s
            t_emb = self.time_emb(t2)
            v_hat, Z = self.denoiser(Z, x_t, c_s, t_emb, ctx_mem)
            l_fm = l_fm + F.mse_loss(v_hat, v_tgt)
        l_fm = l_fm / S
        out = {"loss_fm": l_fm, "attn": attn_w}
        if self.use_contrastive and rep.shape[0] >= 4:
            lam = self.contrastive.lambda_at(epoch, total_epochs)
            ev_agg = batch["fut_events"][:, 0]                 # (B, F)
            l_con, l_sim, _ = self.contrastive(rep, batch["heur"], ev_agg, lam)
            out["loss_con"] = l_con
            out["loss_sim"] = l_sim
        else:
            out["loss_con"] = x.new_zeros(())
            out["loss_sim"] = x.new_zeros(())
        return out

    # -------------------- 采样（推理） --------------------
    @torch.no_grad()
    def sample(self, batch: Dict[str, torch.Tensor], n_samples: int = 50,
               n_steps: int = 10) -> torch.Tensor:
        """概率预测：返回 (B, 28, C, n_samples)。"""
        dev = batch["ctx"].device
        ctx_mem, rep, attn_w, hist_e = self.build_context(
            batch["daily"], batch["weekly"], batch["monthly"],
            batch["short"], batch["hist_events"], batch["fut_events"])
        B = ctx_mem.shape[0]
        Z = self.init_state(batch["ctx"], hist_e, ctx_mem)
        fut_e = self.ev_enc(batch["fut_events"])
        K = n_samples
        Zk = Z.repeat_interleave(K, dim=0)                     # (B*K, d)
        ctx_k = ctx_mem.repeat_interleave(K, dim=0)
        # 预计算事件感知时间步表（含端点 t=1→0，n_steps+1 个节点）
        # 与相邻节点的真实差分 Δt2 —— Euler 积分步长必须沿调度参数 t2，
        # 否则学习到的非恒等调度会引入积分误差。
        t2_table, dt_table = [], []
        ts_grid = torch.tensor(
            [1.0 - i / n_steps for i in range(n_steps + 1)], device=dev)
        for s in range(self.n_seg_fut):
            c_s = fut_e[:, s]
            if not self.use_events:
                c_s = self.ev_enc.null_emb.unsqueeze(0).expand(B, -1)
            t_flat = ts_grid.repeat_interleave(B)              # ((N+1)*B,)
            c_flat = c_s.repeat(n_steps + 1, 1)
            t2_seq = self.timestep(t_flat, c_flat).view(n_steps + 1, B)
            t2_table.append(t2_seq)
            dt_table.append((t2_seq[1:] - t2_seq[:-1]))        # (N, B) <0
        outs = []
        for s in range(self.n_seg_fut):
            c_s = fut_e[:, s]
            if not self.use_events:
                c_s = self.ev_enc.null_emb.unsqueeze(0).expand(B, -1)
            c_k = c_s.repeat_interleave(K, dim=0)
            n = torch.randn(B * K, self.seg_len, self.c_in, device=dev)
            # ODE 积分 t2: 1 → 0（Euler，步长 = 调度真实差分）
            for i in range(n_steps):
                t2 = t2_table[s][i].repeat_interleave(K)       # (B*K,)
                dt = dt_table[s][i].repeat_interleave(K).view(-1, 1, 1)
                t_emb = self.time_emb(t2)
                v_hat, _ = self.denoiser(Zk, n, c_k, t_emb, ctx_k)
                n = n + v_hat * dt
            x_hat = n                                           # (B*K, seg, C)
            t1 = torch.ones(B * K, device=dev)
            _, Zk = self.denoiser(Zk, x_hat, c_k,
                                  self.time_emb(t1), ctx_k)
            outs.append(x_hat.view(B, K, self.seg_len, self.c_in))
        # (B, K, 4, seg, C) → (B, 28, C, K)
        x = torch.stack(outs, dim=2).permute(0, 2, 3, 4, 1).reshape(
            B, self.n_seg_fut * self.seg_len, self.c_in, K)
        return x

    @torch.no_grad()
    def point_forecast(self, batch, n_samples: int = 50,
                       n_steps: int = 10) -> torch.Tensor:
        x = self.sample(batch, n_samples, n_steps)             # (B,28,C,K)
        return x.median(dim=-1).values

    # -------------------- 归因 --------------------
    @torch.no_grad()
    def event_ablation(self, batch, n_samples: int = 20,
                       n_steps: int = 10) -> torch.Tensor:
        """事件消融归因：有事件 vs 全 NULL 预测的均值差（|Δ| 越大越关键）。"""
        saved = self.use_events
        try:
            self.use_events = True
            x_with = self.sample(batch, n_samples, n_steps).mean(dim=-1)
            self.use_events = False
            x_without = self.sample(batch, n_samples, n_steps).mean(dim=-1)
        finally:
            self.use_events = saved
        return (x_with - x_without).abs().mean(dim=(1, 2))     # (B,)
