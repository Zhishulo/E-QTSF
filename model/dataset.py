# -*- coding: utf-8 -*-
"""E-QTSF 数据集：窗口构建 + 多尺度视图 + 事件对齐 + RevIN 归一化。

样本定义（对齐 EVENTTSF Definition 2）：
  - 序列按 7 天为段离散化；每段对齐段内事件（无事件为 NULL 向量）
  - 上下文 84 天（12 段）+ 预测 28 天（4 段）
  - 多尺度视图：日 / 周 / 月聚合 + 短期窗口（末 28 天）
"""
import csv
import json
import os
from typing import Dict, List

import numpy as np

EVENT_TYPES = ["equipment", "tooling", "material", "process",
               "personnel", "measure"]
EVENT_SUBTYPES = ["repair", "maintain", "tool_change", "batch_change",
                  "supplier_change", "param_adjust", "fixture_adjust",
                  "operator_change", "new_operator", "capa",
                  "strict_inspect", "poka_yoke", "other"]
SEVERITIES = ["致命", "严重", "一般", "轻微"]
SEVERITY_W = {"致命": 1.6, "严重": 1.2, "一般": 1.0, "轻微": 0.6}
NULL_SEV_W = 0.0

EVENT_FEAT_DIM = (len(EVENT_TYPES) + len(EVENT_SUBTYPES)
                  + len(SEVERITIES) + 2)  # +has_event +max_sev_w


def event_feature(event: dict) -> np.ndarray:
    """单个事件 → 特征向量（类型/子类型/严重度 multi-hot + 权重）。"""
    f = np.zeros(EVENT_FEAT_DIM, dtype=np.float32)
    f[EVENT_TYPES.index(event["event_type"])] = 1.0
    off = len(EVENT_TYPES)
    sub = event.get("event_subtype", "other")
    f[off + EVENT_SUBTYPES.index(sub if sub in EVENT_SUBTYPES else "other")] = 1.0
    off2 = off + len(EVENT_SUBTYPES)
    sev = event["severity"]
    f[off2 + SEVERITIES.index(sev)] = 1.0
    f[-2] = 1.0                       # has_event
    f[-1] = SEVERITY_W.get(sev, 1.0)  # 严重度权重
    return f


def null_event() -> np.ndarray:
    f = np.zeros(EVENT_FEAT_DIM, dtype=np.float32)
    f[-1] = NULL_SEV_W
    return f


def segment_events(day_lo: int, day_hi: int,
                   events: List[dict]) -> np.ndarray:
    """聚合 [day_lo, day_hi) 内事件：严重度最高者代表该段；无事件为 NULL。"""
    inside = [e for e in events if day_lo <= e["day"] < day_hi]
    if not inside:
        return null_event()
    best = max(inside, key=lambda e: SEVERITY_W.get(e["severity"], 1.0))
    return event_feature(best)


CHANNELS = ["pass_rate", "ppm", "dim_rate", "app_rate", "volume"]
CTX_DAYS = 84      # 12 段 × 7 天
FORE_DAYS = 28     # 4 段 × 7 天
SEG = 7
SHORT_DAYS = 28


class QTSFDataset:
    """逐产品滑窗样本集。split: train/val/test（按窗口终点时间划分）。"""

    def __init__(self, data_dir: str, split: str = "train",
                 products: List[str] = None, stride: int = 7,
                 norm_channels: List[int] = (0, 1)):
        self.dir = data_dir
        self.split = split
        self.samples: List[dict] = []
        with open(os.path.join(data_dir, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        pcodes = products or [p["code"] for p in meta["products"]]
        self.pcodes = pcodes
        self.norm_ch = list(norm_channels)  # 参与归一化的目标通道
        for pc in pcodes:
            arr = self._read_series(os.path.join(data_dir, f"series_{pc}.csv"))
            evs = self._read_events(os.path.join(data_dir, f"events_{pc}.csv"))
            n = arr.shape[0]
            lo, hi = self._split_range(n, split)
            # 窗口终点必须落在当前 split 区间内，避免训练目标与验证/测试期重叠
            for t_end in range(max(lo, CTX_DAYS),
                               min(hi, n - FORE_DAYS + 1), stride):
                self.samples.append(self._make(arr, evs, t_end, pc))

    # ---------------- IO ----------------
    @staticmethod
    def _read_series(path: str) -> np.ndarray:
        with open(path, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        arr = np.zeros((len(rows), len(CHANNELS)), dtype=np.float64)
        for i, r in enumerate(rows):
            for j, c in enumerate(CHANNELS):
                arr[i, j] = float(r[c])
        return arr

    @staticmethod
    def _read_events(path: str) -> List[dict]:
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        evs = []
        for r in rows:
            evs.append({
                "day": int(r["date"][11:13]) if False else
                       (int(r["_didx"]) if "_didx" in r else
                        QTSFDataset._day_index(r["date"])),
                "event_type": r["event_type"],
                "event_subtype": r.get("event_subtype", "other"),
                "severity": r["severity"],
            })
        return evs

    @staticmethod
    def _day_index(date_iso: str) -> int:
        from datetime import date
        d0 = date(2024, 1, 1)
        y, m, d = map(int, date_iso.split("-"))
        return (date(y, m, d) - d0).days

    @staticmethod
    def _split_range(n: int, split: str):
        t_train = int(n * 0.70)
        t_val = int(n * 0.85)
        if split == "train":
            return 0, t_train
        if split == "val":
            return t_train, t_val
        return t_val, n

    # ---------------- 样本构建 ----------------
    def _make(self, arr: np.ndarray, evs: List[dict], t_end: int,
              pcode: str) -> dict:
        ctx = arr[t_end - CTX_DAYS:t_end].copy()          # (84, C)
        tgt = arr[t_end:t_end + FORE_DAYS].copy()         # (28, C)
        # RevIN：全部通道按上下文统计量归一化（评估时按通道 0 还原合格率）
        mu = ctx.mean(axis=0)                              # (C,)
        sd = ctx.std(axis=0) + 1e-5
        ctx = (ctx - mu) / sd
        tgt = (tgt - mu) / sd
        # 训练集输入抖动（正则化，让事件通路有更长的成熟窗口）
        if self.split == "train":
            ctx = ctx + np.random.randn(*ctx.shape).astype(np.float64) * 0.02
        # 多尺度视图（归一化后的 ctx）
        daily = ctx                                        # (84, C)
        weekly = ctx.reshape(12, SEG, -1).mean(axis=1)     # (12, C)
        monthly = ctx.reshape(3, 28, -1).mean(axis=1)      # (3, C)
        short = ctx[-SHORT_DAYS:]                          # (28, C)
        # 事件对齐
        hist_ev = np.stack([
            segment_events(t_end - CTX_DAYS + s * SEG,
                           t_end - CTX_DAYS + (s + 1) * SEG, evs)
            for s in range(CTX_DAYS // SEG)])              # (12, F)
        fut_ev = np.stack([
            segment_events(t_end + s * SEG, t_end + (s + 1) * SEG, evs)
            for s in range(FORE_DAYS // SEG)])             # (4, F)
        # 对比学习启发式统计量（未归一化 pass_rate 末 28 天）
        raw28 = arr[t_end - SHORT_DAYS:t_end, 0]
        stats = self._heur_stats(raw28)
        return {
            "ctx": ctx.astype(np.float32),                 # (84, C)
            "target": tgt.astype(np.float32),              # (28, C)
            "daily": daily.astype(np.float32),
            "weekly": weekly.astype(np.float32),
            "monthly": monthly.astype(np.float32),
            "short": short.astype(np.float32),
            "hist_events": hist_ev.astype(np.float32),     # (12, F)
            "fut_events": fut_ev.astype(np.float32),       # (4, F)
            "heur": stats.astype(np.float32),              # (7,)
            "mu": mu.astype(np.float32),                   # (2,)
            "sd": sd.astype(np.float32),
            "pcode": pcode, "t_end": t_end,
        }

    @staticmethod
    def _heur_stats(x: np.ndarray) -> np.ndarray:
        """EventTime 式启发式统计量：均值/方差/极值/趋势/波动变化/近期冲击。"""
        m = x.mean()
        s = x.std() + 1e-8
        mn, mx = x.min(), x.max()
        k = np.polyfit(np.arange(len(x)), x, 1)[0]          # 趋势斜率
        vol_chg = x[-7:].std() / (x[-28:-7].std() + 1e-8)   # 波动比
        shock = abs(x[-1] - x[-7]) / s                      # 近期冲击
        return np.array([(m - 90.0) / 5.0, s / 2.0, (mn - 90.0) / 5.0,
                         (mx - 90.0) / 5.0, k, vol_chg, shock],
                        dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

    def channels_raw(self, i: int) -> np.ndarray:
        """还原归一化前的目标段（评估用）。"""
        s = self.samples[i]
        return s["target"] * s["sd"] + s["mu"]

    def event_vocab(self) -> dict:
        return {"types": EVENT_TYPES, "subtypes": EVENT_SUBTYPES,
                "severities": SEVERITIES, "feat_dim": EVENT_FEAT_DIM}


def collate(batch: List[dict]) -> Dict[str, "torch.Tensor"]:
    import torch
    out = {}
    for k in ["ctx", "target", "daily", "weekly", "monthly", "short",
              "hist_events", "fut_events", "heur", "mu", "sd", "t_end"]:
        out[k] = torch.from_numpy(
            np.stack([b[k] for b in batch]).astype(np.float32))
    out["pcode"] = [b["pcode"] for b in batch]
    return out
