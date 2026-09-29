# -*- coding: utf-8 -*-
"""E-QTSF 动态对比学习（EventTime 3.2 思想，质量域适配）。

解决高严重度质量事件稀疏问题：
  - 启发式相似度：时序统计量 cos + 事件属性 cos 动态加权
  - 学习式相似度：融合表征 cos
  - 动态加权 S = S_h^(λ) · S_l^(1-λ)，λ 随训练轮次递增
  - top-k 正对（S>ρ）/ 负对（S<ρ'）+ InfoNCE + 相似度正则
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """归一化余弦相似度 → [0,1]（EventTime 式）。"""
    cos = F.cosine_similarity(a, b, dim=-1)
    return (1 + cos) / 2


def heuristic_similarity(stats: torch.Tensor, ev_feat: torch.Tensor,
                         alpha: float = 0.5) -> torch.Tensor:
    """批内启发式相似度矩阵。

    stats:   (B, 7)   时序启发式统计量（dataset._heur_stats）
    ev_feat: (B, F)   未来段事件聚合特征（严重度加权 multi-hot）
    alpha:   时序统计量占比
    """
    s1 = norm_cos(stats.unsqueeze(1), stats.unsqueeze(0))        # (B, B)
    # 事件属性相似度：multi-hot 加权 Jaccard 风格
    ev = ev_feat[..., :-2]                                       # 去 has/sev 列
    inter = ev.unsqueeze(1) * ev.unsqueeze(0)                    # (B,B,F)
    union = ev.unsqueeze(1).sum(-1) + ev.unsqueeze(0).sum(-1) - inter.sum(-1)
    s2 = inter.sum(-1) / (union + 1e-6)
    # 严重度权重接近度
    w = ev_feat[..., -1]
    s3 = 1.0 - (w.unsqueeze(1) - w.unsqueeze(0)).abs()
    return alpha * s1 + (1 - alpha) * (0.7 * s2 + 0.3 * s3)


class DynamicContrastive(nn.Module):
    """动态对比学习目标（启发式 → 学习式渐进切换）。"""

    def __init__(self, d_model: int, temp: float = 0.2,
                 pos_thr: float = 0.75, neg_thr: float = 0.45,
                 topk: int = 4, lambda_start: float = 0.3,
                 lambda_end: float = 0.9):
        super().__init__()
        self.tau = temp
        self.pos_thr = pos_thr
        self.neg_thr = neg_thr
        self.topk = topk
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model))

    def lambda_at(self, epoch: float, total: float) -> float:
        r = min(max(epoch / max(total, 1), 0.0), 1.0)
        return self.lambda_start + (self.lambda_end - self.lambda_start) * r

    def forward(self, rep: torch.Tensor, stats: torch.Tensor,
                ev_feat: torch.Tensor, lam: float
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (L_contrast, L_similarity, S_learn)。

        rep:    (B, d)  融合表征（投影前）
        stats:  (B, 7)  启发式时序统计量
        ev_feat:(B, F)  事件聚合特征
        lam:    当前动态权重（启发式占比）
        """
        B = rep.shape[0]
        if B < 4:
            zero = rep.sum() * 0.0
            return zero, zero, norm_cos(rep, rep)
        h = self.proj(rep)
        s_learn = norm_cos(h.unsqueeze(1), h.unsqueeze(0))           # (B,B)
        s_heur = heuristic_similarity(stats, ev_feat)                # (B,B)
        # 动态加权相似度
        s_dyn = (s_heur.clamp_min(1e-6) ** lam) * (s_learn.clamp_min(1e-6) ** (1 - lam))
        # 正/负对选择（对角排除）
        eye = torch.eye(B, device=rep.device, dtype=torch.bool)
        pos_mask = (s_dyn > self.pos_thr) & ~eye
        neg_mask = (s_dyn < self.neg_thr) | eye
        # top-k 正对
        if pos_mask.sum() == 0:
            k = min(self.topk, B - 1)
            s_off = s_dyn.masked_fill(eye, -1.0)
            idx = s_off.topk(k, dim=1).indices                       # (B,k)
            pos_mask = torch.zeros_like(s_dyn, dtype=torch.bool)
            pos_mask.scatter_(1, idx, True)
        # InfoNCE：锚点 i 的正对集合 P_i、负对集合 Q_i
        logits = s_learn / self.tau
        logits = logits.masked_fill(neg_mask, -1e9)
        log_prob = logits - torch.logsumexp(
            logits.masked_fill(neg_mask & ~pos_mask, -1e9).clamp_min(-1e9)
            if False else logits, dim=1, keepdim=True)
        n_pos = pos_mask.sum(dim=1).clamp_min(1)
        l_con = -(log_prob * pos_mask).sum(dim=1) / n_pos
        l_con = l_con.mean()
        # 相似度正则：学习式向启发式靠拢（EventTime L_similarity）
        l_sim = ((s_learn - s_heur) ** 2).mean()
        return l_con, l_sim, s_learn
