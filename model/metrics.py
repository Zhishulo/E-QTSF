# -*- coding: utf-8 -*-
"""评估指标：概率（CRPS/WQL）+ 点预测（MAE/MSE/RMSE）+ 业务预警指标。"""
from typing import Tuple

import numpy as np


def crps_sample(y: np.ndarray, samples: np.ndarray,
                max_pairs: int = 20) -> float:
    """样本 CRPS = E|X-y| - 0.5 E|X-X'|。

    y: (N,) 真值; samples: (N, K) 预测样本。
    E|X-X'| 用等间隔子采样近似（K 大时高效且无偏）。
    """
    n, k = samples.shape
    t1 = np.abs(samples - y[:, None]).mean()
    step = max(1, k // max_pairs)
    sub = samples[:, ::step]
    m = sub.shape[1]
    t2 = 0.0
    for i in range(m):
        t2 += np.abs(sub - sub[:, i:i + 1]).mean()
    return float(t1 - 0.5 * t2 / m)


def wql(y: np.ndarray, samples: np.ndarray,
        quantiles=(0.1, 0.25, 0.5, 0.75, 0.9)) -> float:
    """Weighted Quantile Loss（标准分位损失协议）。"""
    qs = np.quantile(samples, quantiles, axis=1)            # (Q, N)
    denom = np.sum(np.abs(y)) + 1e-8
    total = 0.0
    for i, q in enumerate(quantiles):
        qhat = qs[i]
        total += 2 * np.sum(np.maximum(q * (y - qhat),
                                       (q - 1) * (y - qhat)))
    return float(total / denom)


def mae_mse_rmse(y: np.ndarray, yhat: np.ndarray) -> Tuple[float, float, float]:
    e = yhat - y
    return (float(np.abs(e).mean()), float((e ** 2).mean()),
            float(np.sqrt((e ** 2).mean())))


def alert_metrics(y_true: np.ndarray, samples: np.ndarray,
                  threshold: float, prob_thr: float = 0.5) -> dict:
    """业务预警指标（质量域）。

    y_true:   (N, H) 未来窗口真实合格率
    samples:  (N, H, K) 预测样本
    threshold: 合格率预警阈值（如 SPC 控制下限）
    返回：命中率 / 平均提前期 / 误报率
    """
    p_breach = (samples < threshold).any(axis=1).mean(axis=1)   # (N,)
    alert = p_breach >= prob_thr
    actual = (y_true < threshold).any(axis=1)
    hits, leads, miss = 0, [], 0
    for i in range(len(y_true)):
        if alert[i] and actual[i]:
            hits += 1
            leads.append(int(np.argmax(y_true[i] < threshold)))
        elif alert[i] != actual[i]:
            miss += 1
    fp = int(np.sum(alert & ~actual))
    tn = int(np.sum(~alert & ~actual))
    return {"hit_rate": hits / max(hits + miss, 1),
            "avg_lead_days": float(np.mean(leads)) if leads else 0.0,
            "false_alarm_rate": fp / max(fp + tn, 1),
            "n_alerts": int(alert.sum()), "n_actual": int(actual.sum())}
