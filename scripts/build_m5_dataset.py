# -*- coding: utf-8 -*-
"""M5 真实零售数据 → E-QTSF 训练格式适配器（真实数据验证用）。

数据源：M5 Forecasting Accuracy 竞赛公共数据集（Walmart 真实零售数据），
本地缓存在 data/real/（calendar.csv + sales_train_validation.csv）。
- 序列：sales_train_validation.csv 中销量 Top-10 商品（跨 10 家门店聚合），
  1913 天（2011-01-29 起），日列 d_1 至 d_1913 映射为日索引 0 至 1912
- 通道映射（列名沿用质量数据协议，通道 0 为预测目标）：
    pass_rate = 日销量（件）      ppm      = SNAP 日比例（三州均值 0~1）
    dim_rate  = 日历事件标志      app_rate = 周末标志
    volume    = 星期几归一化（0~1）
- 日历事件映射到质量事件本体（类别槽位借用，严重度按事件类别先验固定）：
    National→process/严重   Religious→personnel/一般
    Cultural→measure/一般   Sporting→measure/轻微
  SNAP 不作事件（已作通道），保持事件流稀疏以支撑事件分层评估。
- 输出 data/output_m5/：meta.json + series_{pc}.csv + events_{pc}.csv
  运行方式：在项目根目录 eqtsf-system 下执行
  .venv/Scripts/python.exe scripts/build_m5_dataset.py
"""
import json
import os
from pathlib import Path

import pandas as pd

SRC = "data/real"
OUT = "data/output_m5"

TYPE_MAP = {
    "NATIONAL": ("process", "严重"),
    "RELIGIOUS": ("personnel", "一般"),
    "CULTURAL": ("measure", "一般"),
    "SPORTING": ("measure", "轻微"),
}
SEV_RANK = {"致命": 1.6, "严重": 1.2, "一般": 1.0, "轻微": 0.6}
TOP_N = 10


def main():
    cal = pd.read_csv(SRC + "/calendar.csv")
    dtype = {("d_" + str(i)): "int32" for i in range(1, 1914)}
    sales = pd.read_csv(SRC + "/sales_train_validation.csv", dtype=dtype)
    dcols = [c for c in sales.columns if c.startswith("d_")]
    n_days = len(dcols)
    agg = sales.groupby("item_id", as_index=False)[dcols].sum()
    agg["total"] = agg[dcols].sum(axis=1)
    top = agg.sort_values("total", ascending=False).head(TOP_N).reset_index(
        drop=True)
    print("Top items:", list(top["item_id"]))

    # 事件表：日历命名事件 → 质量本体（按天去重取严重度最高）
    best = {}
    for _, r in cal.iterrows():
        didx = int(str(r["d"])[2:]) - 1
        if didx >= n_days:
            continue
        for ncol, tcol in (("event_name_1", "event_type_1"),
                           ("event_name_2", "event_type_2")):
            name, typ = r[ncol], r[tcol]
            if not isinstance(name, str) or not isinstance(typ, str):
                continue
            if typ.upper() not in TYPE_MAP:
                continue
            etype, sev = TYPE_MAP[typ.upper()]
            if didx not in best or SEV_RANK[sev] > SEV_RANK[best[didx][2]]:
                best[didx] = (etype, typ.lower(), sev, name)
    ev_rows = []
    for didx in sorted(best):
        etype, sub, sev, name = best[didx]
        ev_rows.append({"date": cal["date"][didx], "_didx": didx,
                        "event_type": etype, "event_subtype": "other",
                        "severity": sev, "event_name": name})
    print("events:", len(ev_rows), "distinct days:",
          len(set(r["_didx"] for r in ev_rows)), "/", n_days)

    # 通道：SNAP 比例 / 事件标志 / 周末 / 星期几
    snap = cal[["snap_CA", "snap_TX", "snap_WI"]].mean(axis=1).to_numpy()
    snap = snap[:n_days]
    ev_flag = cal["event_name_1"].notna() | cal["event_name_2"].notna()
    ev_flag = ev_flag.to_numpy()[:n_days].astype(float)
    wday = cal["wday"].to_numpy()[:n_days].astype(float)
    weekend = ((wday == 1) | (wday == 2)).astype(float)
    dow_norm = (wday - 1.0) / 6.0
    dates = cal["date"].tolist()[:n_days]

    os.makedirs(OUT, exist_ok=True)
    products = []
    for k in range(len(top)):
        row = top.iloc[k]
        code = "M5_P" + str(k + 1).zfill(2)
        products.append({"code": code, "name": str(row["item_id"])})
        y = row[dcols].to_numpy(dtype=float)
        df = pd.DataFrame({"date": dates, "pass_rate": y, "ppm": snap,
                           "dim_rate": ev_flag, "app_rate": weekend,
                           "volume": dow_norm})
        df.to_csv("%s/series_%s.csv" % (OUT, code), index=False,
                  encoding="utf-8-sig")
        pd.DataFrame(ev_rows).to_csv("%s/events_%s.csv" % (OUT, code),
                                     index=False, encoding="utf-8-sig")
    meta = {
        "products": products,
        "start": dates[0], "days": n_days,
        "event_types": ["equipment", "tooling", "material", "process",
                        "personnel", "measure"],
        "event_subtypes": {"other": "其他"},
        "severities": ["致命", "严重", "一般", "轻微"],
        "channels": ["pass_rate", "ppm", "dim_rate", "app_rate", "volume"],
        "source": "M5 Forecasting Accuracy (Walmart public dataset)",
        "note": "真实数据适配：通道与事件映射见 scripts/build_m5_dataset.py",
    }
    Path(OUT, "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("WROTE", OUT, "products:", len(products), "days:", n_days)


if __name__ == "__main__":
    main()
