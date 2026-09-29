# -*- coding: utf-8 -*-
"""
模拟数据生成器：机械加工企业质量数据 + 质量事件注入（无状态纯函数版）
=====================================================================
生成内容（对齐《机械加工企业质量管理系统的设计与实现》业务模型）：
  1. 产品/批次/质量记录/缺陷明细（继承原论文表主线）
  2. 检验实测值（SPC 数据源）
  3. 六大类质量事件（E-QTSF 模型核心输入）：
     equipment 设备 / tooling 刀具 / material 物料 /
     process   工艺 / personnel 人员 / measure 质量措施
  4. NCR 不合格品报告（闭环）
  5. 面向模型的 CSV + meta.json

设计要点：
  - 每个事件的随机因子（方向/坏批次等）在采样时一次性确定并写入事件 dict，
    event_effect 为纯函数（无跨调用状态），保证可复现、可审计。
  - 严重度（致命/严重/一般/轻微）缩放效应强度。
"""
import json
import math
import os
import sqlite3
import traceback
from datetime import date, timedelta

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "output")
SCHEMA = os.path.join(BASE_DIR, "schema.sql")

START_DATE = date(2024, 1, 1)
DAYS = 730

EVENT_TYPES = {
    "equipment": "设备类", "tooling": "刀具类", "material": "物料类",
    "process": "工艺类", "personnel": "人员类", "measure": "质量措施类",
}

# 子类型: (大类, 效应profile, 基础效应强度pp, 持续天数范围)
EVENT_SUBTYPES = {
    "设备故障维修": ("equipment", "repair", 4.0, (3, 7)),
    "设备保养": ("equipment", "maintain", 1.5, (15, 30)),
    "刀具磨损更换": ("tooling", "tool_change", 2.5, (20, 35)),
    "刀具批次更换": ("tooling", "tool_change", 2.0, (15, 25)),
    "来料批次变更": ("material", "batch_change", 3.0, (5, 10)),
    "供应商变更": ("material", "supplier_change", 3.5, (10, 20)),
    "工艺参数调整": ("process", "param_adjust", 2.5, (7, 14)),
    "工装夹具调整": ("process", "fixture_adjust", 1.5, (10, 20)),
    "操作员变更": ("personnel", "operator_change", 1.5, (3, 7)),
    "新员工上岗": ("personnel", "new_operator", 2.0, (15, 30)),
    "CAPA实施": ("measure", "capa", 1.5, (20, 40)),
    "检验标准加严": ("measure", "strict_inspect", 2.0, (10, 20)),
    "防错装置导入": ("measure", "poka_yoke", 1.0, (30, 60)),
}
SEVERITY_SCALE = {"致命": 1.6, "严重": 1.2, "一般": 1.0, "轻微": 0.6}
SEVERITY_WEIGHTS = [0.08, 0.27, 0.45, 0.20]

DEFECT_CATEGORIES = ["尺寸", "外观", "性能", "包装", "其他"]
DEFECT_CODES = [
    ("DEF-001", "直径超差", "尺寸", "严重", "设备部"),
    ("DEF-002", "平面度超差", "尺寸", "严重", "设备部"),
    ("DEF-003", "孔径超差", "尺寸", "致命", "设备部"),
    ("DEF-004", "划伤", "外观", "一般", "生产部"),
    ("DEF-005", "毛刺", "外观", "一般", "生产部"),
    ("DEF-006", "表面粗糙度差", "外观", "严重", "生产部"),
    ("DEF-007", "硬度不足", "性能", "致命", "工艺部"),
    ("DEF-008", "强度不达标", "性能", "致命", "工艺部"),
    ("DEF-009", "包装破损", "包装", "轻微", "仓储部"),
    ("DEF-010", "标识错误", "包装", "轻微", "仓储部"),
    ("DEF-011", "其他", "其他", "一般", "质量部"),
]

PRODUCTS = [
    ("PRD-001", "传动轴", "轴类", "φ20×250", "45钢"),
    ("PRD-002", "齿轮坯", "盘类", "φ80×20", "20CrMnTi"),
    ("PRD-003", "法兰盘", "盘类", "φ120×15", "Q235"),
    ("PRD-004", "液压缸体", "箱体类", "100×80×60", "HT250"),
    ("PRD-005", "衬套", "套类", "φ30×45", "锡青铜"),
    ("PRD-006", "丝杠", "轴类", "φ16×600", "40Cr"),
    ("PRD-007", "轴承座", "箱体类", "φ90×50", "HT200"),
    ("PRD-008", "键条", "杂类", "10×8×200", "45钢"),
    ("PRD-009", "阀芯", "套类", "φ12×30", "不锈钢304"),
    ("PRD-010", "连杆", "杂类", "150×40", "40Cr"),
]

DEPTS = ["质量部", "生产部", "工艺部", "设备部", "采购部", "仓储部", "技术部"]
EVENT_DEPT = {
    "equipment": "设备部", "tooling": "生产部", "material": "采购部",
    "process": "工艺部", "personnel": "生产部", "measure": "质量部",
}


def sample_events(rng):
    """采样事件列表；每个事件的随机因子在此一次确定。"""
    events = []
    d = START_DATE + timedelta(days=int(rng.integers(10, 26)))
    idx = 0
    while d < START_DATE + timedelta(days=DAYS - 30):
        subtype = str(rng.choice(list(EVENT_SUBTYPES.keys())))
        etype, profile, intensity, dur_rng = EVENT_SUBTYPES[subtype]
        sev = str(rng.choice(list(SEVERITY_SCALE.keys()), p=SEVERITY_WEIGHTS))
        dur = int(rng.integers(dur_rng[0], dur_rng[1] + 1))
        ev = {
            "idx": idx, "date": d, "type": etype, "subtype": subtype,
            "severity": sev, "profile": profile,
            "intensity": intensity * SEVERITY_SCALE[sev],
            "duration": dur,
            # 一次性确定的随机因子（纯函数化）
            "bad_batch": bool(rng.random() < 0.30),       # batch_change 用
            "dir": 1 if rng.random() < 0.55 else -1,      # supplier_change 用
            "good_adjust": bool(rng.random() < 0.70),     # param_adjust 用
        }
        events.append(ev)
        d += timedelta(days=int(rng.integers(12, 35)))
        idx += 1
    return events


def event_effect(ev, day_offset):
    """纯函数：返回事件后第 day_offset 天的 (Δ均值pp, 方差乘子)。

    所有方差乘子恒 >= 1.0；超出持续期(dur)后改善类效应按 taper 线性消退，
    恢复类效应 clamp 在恢复完成状态，杜绝外推出负乘子。
    """
    prof = ev["profile"]
    k = ev["intensity"]
    dur = ev["duration"]
    if day_offset < 0 or day_offset > dur + 14:
        return 0.0, 1.0
    taper = 1.0 - (day_offset - dur) / 14.0 if day_offset > dur else 1.0
    if prof == "repair":            # 设备故障：先跌后修复回升（rec clamp<=1）
        half = max(dur // 2, 1)
        if day_offset <= half:
            return -k * (0.5 + day_offset / half * 0.5), 1.4
        rec = min(1.0, (day_offset - half) / max(dur - half, 1))
        return -k * (1 - rec) * 0.8 * taper, 1.0 + 0.3 * (1 - rec)
    if prof == "maintain":          # 保养：当天小跌，之后改善并重置劣化
        if day_offset == 0:
            return -0.3 * k, 1.1
        return k * 0.6 * min(1.0, day_offset / 5) * taper, 1.0
    if prof == "tool_change":       # 换刀：磨合期方差大，之后改善
        if day_offset <= 1:
            return -0.2 * k, 2.0
        return k * 0.5 * min(1.0, day_offset / 4) * taper, 1.0
    if prof == "batch_change":      # 来料批次：方差增大；坏批次均值跌
        vm = 1.0 + 0.8 * max(0.0, 1.0 - day_offset / dur)
        if ev["bad_batch"] and day_offset <= dur:
            return -k * 0.7 * max(0.2, 1.0 - day_offset / dur), vm
        return 0.0, vm
    if prof == "supplier_change":   # 供应商变更：方向随机，持续中段
        ramp = min(1.0, day_offset / max(dur * 0.3, 1.0))
        return ev["dir"] * k * 0.6 * ramp * taper, 1.0 + 0.4 * (1.0 - ramp)
    if prof == "param_adjust":      # 工艺调整：多数改善，少数误调先跌后缓慢恢复
        if ev["good_adjust"]:
            return k * 0.6 * min(1.0, day_offset / 3) * taper, 1.0
        drop = -k * max(0.0, 0.8 - 0.6 * day_offset / dur) * taper
        return drop, 1.0 + 0.2 * max(0.0, 1.0 - day_offset / dur)
    if prof == "fixture_adjust":    # 夹具调整：方差先大后小，小幅改善
        return (k * 0.3 * min(1.0, day_offset / 4) * taper,
                1.0 + 0.6 * max(0.0, 1.0 - day_offset / dur))
    if prof == "operator_change":   # 人员变更：方差放大
        vm = 1.0 + 0.9 * max(0.0, 1.0 - day_offset / dur)
        return -0.15 * k * max(0.0, 1.0 - day_offset / dur), vm
    if prof == "new_operator":      # 新员工：学习曲线
        vm = 1.0 + 0.9 * max(0.0, 1.0 - day_offset / dur)
        return 0.0, vm
    if prof == "capa":              # CAPA：渐进持续改善
        return k * 0.7 * min(1.0, day_offset / 10) * taper, 1.0
    if prof == "strict_inspect":    # 检验加严：合格率先降后回升
        return -k * 0.8 * max(0.0, 1.0 - day_offset / dur), 1.0
    if prof == "poka_yoke":         # 防错导入：持续改善
        return k * 0.8 * min(1.0, day_offset / 7) * taper, 1.0
    if prof == "shock":             # 瞬时冲击：1-2 天深谷 + 方差脉冲
        vm = 1.0 + 1.6 * max(0.0, 1.0 - day_offset / max(dur, 1))
        return -k * (1.0 - 0.5 * day_offset / max(dur, 1)), vm
    return 0.0, 1.0


def simulate(rng, product_idx, events):
    """仿真单产品日粒度质量序列。"""
    n = DAYS
    base_pass = float(rng.uniform(88.0, 96.5))
    w = rng.dirichlet(np.ones(5) * 3.0)
    defect_profile = w / w.sum()
    nominal = float(rng.choice([10.0, 20.0, 25.0, 50.0]))
    sigma0 = float(rng.uniform(0.004, 0.010))

    delta = np.zeros(n)
    vmul = np.ones(n)
    resets = set()
    for ev in events:
        di = (ev["date"] - START_DATE).days
        if ev["profile"] in ("maintain", "tool_change"):
            resets.add(di)
        for off in range(0, ev["duration"] + 15):
            j = di + off
            if j >= n:
                break
            dm, vm = event_effect(ev, off)
            delta[j] += dm
            vmul[j] *= vm
    if not np.all(np.isfinite(delta)) or not np.all(vmul >= 0.9):
        bad = np.where(~np.isfinite(delta) | (vmul < 0.9))[0]
        raise ValueError(f"event effect invalid at days {bad[:5]}")

    rows = []
    tool_age = 0
    for t in range(n):
        d = START_DATE + timedelta(days=t)
        if t in resets:
            tool_age = 0
        tool_age += 1
        wear = -min(tool_age, 60) * 0.010
        ann = 0.25 * math.sin(2 * math.pi * t / 365)
        dow = d.weekday()
        wk = -0.30 if dow == 0 else (-0.20 if dow == 5 else 0.0)
        mu = base_pass + wear + ann + wk + delta[t]
        sigma = vmul[t]
        pass_rate = float(np.clip(mu + rng.normal(0, 0.55 * sigma), 82.0, 99.4))
        vol = int(rng.integers(280, 620))
        if dow == 6:
            vol = int(vol * 0.4)
        defect_rate = 1 - pass_rate / 100
        n_def = max(0, int(round(vol * defect_rate + rng.normal(0, 1.5))))
        wc = defect_profile.copy()
        wc[0] *= (1 + tool_age / 80)
        wc[1] *= vmul[t]
        wc = wc / wc.sum()
        cats = rng.multinomial(n_def, wc) if n_def > 0 else np.zeros(5, dtype=int)
        shift = (base_pass - pass_rate) / 100 * 1.2 * nominal * 0.02
        sig = sigma0 * vmul[t] * (1 + tool_age / 120)
        samples = [float(rng.normal(nominal + shift, sig)) for _ in range(5)]
        rows.append({
            "date": d.isoformat(), "pass_rate": round(pass_rate, 2),
            "volume": vol, "defect_qty": n_def,
            "dim_rate": round(cats[0] / max(vol, 1) * 1e6, 1),
            "app_rate": round(cats[1] / max(vol, 1) * 1e6, 1),
            "perf_rate": round(cats[2] / max(vol, 1) * 1e6, 1),
            "pack_rate": round(cats[3] / max(vol, 1) * 1e6, 1),
            "other_rate": round(cats[4] / max(vol, 1) * 1e6, 1),
            "ppm": round(n_def / max(vol, 1) * 1e6, 1),
            "spc_mean": round(float(np.mean(samples)), 4),
            "spc_std": round(float(np.std(samples, ddof=1)), 4),
            "spc_samples": [round(s, 4) for s in samples],
        })
    df = pd.DataFrame(rows)
    return df, events, nominal


def build_database(df_all, events_all, nominals):
    os.makedirs(OUT_DIR, exist_ok=True)
    db_path = os.path.join(OUT_DIR, "qms.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    with open(SCHEMA, encoding="utf-8") as f:
        conn.executescript(f.read())
    cur = conn.cursor()
    for i, name in enumerate(DEPTS):
        cur.execute("INSERT INTO department(dept_name,sort_order) VALUES(?,?)", (name, i))
    dept_id = {n: i + 1 for i, n in enumerate(DEPTS)}
    users = [("admin", "ADMIN", "质量部"), ("insp", "INSPECTOR", "质量部"),
             ("eng", "ENGINEER", "质量部"), ("wang", "INSPECTOR", "生产部"),
             ("li", "ENGINEER", "工艺部")]
    real_names = {"admin": "系统管理员", "insp": "王检验", "eng": "李工程师",
                  "wang": "赵检验", "li": "钱工程师"}
    for i, (u, r, dn) in enumerate(users):
        cur.execute(
            "INSERT INTO sys_user(username,password_hash,real_name,role,dept_id)"
            " VALUES(?,?,?,?,?)",
            (u, "PBKDF2$pending", real_names.get(u, u), r, dept_id[dn]))
    user_id = {u: i + 1 for i, (u, _, _) in enumerate(users)}
    for code, name, cat, sev, dept in DEFECT_CODES:
        cur.execute(
            "INSERT INTO defect_code(defect_code,defect_name,category,severity,"
            "description,responsible_dept_id) VALUES(?,?,?,?,?,?)",
            (code, name, cat, sev, f"{cat}类缺陷：{name}", dept_id[dept]))

    evt_no, ncr_no = 1, 1
    for pi, (pcode, pname, ptype, spec, mat) in enumerate(PRODUCTS):
        pid = pi + 1
        cur.execute(
            "INSERT INTO product(product_code,product_name,product_type,"
            "specification,material,unit) VALUES(?,?,?,?,?,?)",
            (pcode, pname, ptype, spec, mat, "件"))
        df, evs, nominal = df_all[pid], events_all[pid], nominals[pid]
        dts = pd.to_datetime(df["date"])
        df = df.assign(year=dts.dt.year, week=dts.dt.isocalendar().week.astype(int))
        batch_seq = 0
        for (_, _), g in df.groupby(["year", "week"], sort=True):
            batch_seq += 1
            yr = int(g["year"].iloc[0])
            bno = f"BATCH-{pcode[4:]}-{yr}-{batch_seq:03d}"
            total = int(g["volume"].sum())
            defect = int(g["defect_qty"].sum())
            passed = total - defect
            cur.execute(
                "INSERT INTO product_batch(product_id,batch_no,production_date,"
                "total_quantity) VALUES(?,?,?,?)",
                (pid, bno, g["date"].iloc[0], total))
            bid = cur.lastrowid
            cur.execute(
                "INSERT INTO quality_record(product_id,batch_id,record_date,"
                "total_quantity,defect_quantity,pass_quantity,pass_rate,ppm)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (pid, bid, g["date"].iloc[-1], total, defect, passed,
                 round(passed / max(total, 1) * 100, 2),
                 round(defect / max(total, 1) * 1e6, 1)))
            qrid = cur.lastrowid
            for ci, cat in enumerate(DEFECT_CATEGORIES):
                col = ["dim_rate", "app_rate", "perf_rate", "pack_rate", "other_rate"][ci]
                qty = int(round(g[col].sum() / 1e6 * total))
                if qty <= 0:
                    continue
                cur.execute("SELECT id FROM defect_code WHERE category=? LIMIT 1", (cat,))
                row = cur.fetchone()
                if row:
                    cur.execute(
                        "INSERT INTO defect_detail(quality_record_id,defect_code_id,"
                        "defect_quantity) VALUES(?,?,?)", (qrid, row[0], qty))
        tol = 0.025
        for _, r in df.iterrows():
            for si, s in enumerate(r["spc_samples"]):
                cur.execute(
                    "INSERT INTO inspection_value(product_id,inspection_date,"
                    "characteristic,nominal,lsl,usl,sample_no,measured_value)"
                    " VALUES(?,?,?,?,?,?,?,?)",
                    (pid, r["date"], "关键直径", nominal,
                     nominal - tol, nominal + tol, si + 1, s))
        for ev in evs:
            no = f"EVT-{ev['date'].year}-{evt_no:05d}"
            evt_no += 1
            cur.execute(
                "INSERT INTO quality_event(event_no,product_id,event_date,event_type,"
                "event_subtype,severity,description,responsible_dept_id)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (no, pid, ev["date"].isoformat(), ev["type"], ev["subtype"],
                 ev["severity"],
                 f"[{EVENT_TYPES[ev['type']]}] {ev['subtype']}（{ev['severity']}）："
                 f"影响约 {ev['duration']} 天",
                 dept_id[EVENT_DEPT[ev["type"]]]))
        arr = df["pass_rate"].to_numpy()
        last_ncr_t = -99
        for t in range(30, len(arr) - 5):
            if t - last_ncr_t < 20:
                continue
            win = arr[max(0, t - 30):t]
            mu, sd = win.mean(), max(win.std(), 0.3)
            if arr[t] < mu - 2.5 * sd:
                last_ncr_t = t
                no = f"NCR-{df['date'].iloc[t][:4]}-{ncr_no:04d}"
                ncr_no += 1
                cur.execute(
                    "INSERT INTO nonconforming_report(report_no,product_id,batch_no,"
                    "defect_code_id,defect_quantity,severity,discovery_date,"
                    "responsible_dept_id,reporter_id,description,disposition,status,"
                    "close_date) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (no, pid, "", 1, int(df["defect_qty"].iloc[t]), "严重",
                     df["date"].iloc[t], dept_id["质量部"], user_id["insp"],
                     f"合格率 {arr[t]:.2f}% 低于控制下限，触发不合格品报告",
                     "返工", "已关闭",
                     df["date"].iloc[min(t + 5, len(arr) - 1)]))
                ncr_id = cur.lastrowid
                cur.execute(
                    "INSERT INTO ncr_status_log(ncr_id,from_status,to_status,"
                    "operator_id,opinion) VALUES(?,?,?,?,?)",
                    (ncr_id, "待评审", "处理中", user_id["eng"], "评审通过，转责任部门处理"))
                cur.execute(
                    "INSERT INTO ncr_status_log(ncr_id,from_status,to_status,"
                    "operator_id,opinion) VALUES(?,?,?,?,?)",
                    (ncr_id, "处理中", "已关闭", user_id["eng"], "复检合格，关闭"))
    conn.commit()
    conn.close()
    return db_path


def export_model_csv(df_all, events_all):
    for pid, df in df_all.items():
        pcode = PRODUCTS[pid - 1][0]
        cols = ["date", "pass_rate", "ppm", "dim_rate", "app_rate", "volume"]
        df[cols].to_csv(os.path.join(OUT_DIR, f"series_{pcode}.csv"),
                        index=False, encoding="utf-8-sig")
        ev_rows = [{
            "date": ev["date"].isoformat(), "event_type": ev["type"],
            "event_subtype": ev["subtype"], "severity": ev["severity"],
            "description": f"[{EVENT_TYPES[ev['type']]}] {ev['subtype']}（{ev['severity']}）",
        } for ev in events_all[pid]]
        pd.DataFrame(ev_rows).to_csv(
            os.path.join(OUT_DIR, f"events_{pcode}.csv"),
            index=False, encoding="utf-8-sig")
        df[["date", "spc_mean", "spc_std"]].to_csv(
            os.path.join(OUT_DIR, f"spc_{pcode}.csv"),
            index=False, encoding="utf-8-sig")
    meta = {
        "products": [{"code": c, "name": n} for c, n, *_ in PRODUCTS],
        "start": START_DATE.isoformat(), "days": DAYS,
        "event_types": EVENT_TYPES,
        "event_subtypes": {k: v[0] for k, v in EVENT_SUBTYPES.items()},
        "severities": list(SEVERITY_SCALE.keys()),
        "channels": ["pass_rate", "ppm", "dim_rate", "app_rate", "volume"],
    }
    from pathlib import Path
    Path(OUT_DIR, "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main(shock: bool = False, out_dir: str = None):
    global OUT_DIR
    if out_dir:
        OUT_DIR = out_dir
    scale = 1.4 if shock else 1.0
    df_all, events_all, nominals = {}, {}, {}
    for pi in range(1, len(PRODUCTS) + 1):
        rng = np.random.default_rng(2024 + pi * 17)
        events = sample_events(rng)
        if shock:
            # 尖锐化：强度放大 + 注入瞬时冲击事件（1-2 天深谷）
            for ev in events:
                ev["intensity"] *= scale
                if ev["profile"] == "batch_change":
                    ev["bad_batch"] = rng.random() < 0.50
            n_shock = max(2, len(events) // 6)
            for k in range(n_shock):
                d = START_DATE + timedelta(
                    days=int(rng.integers(40, DAYS - 40)))
                events.append({
                    "idx": 1000 + k, "date": d, "type": "equipment",
                    "subtype": "repair", "severity": "致命",
                    "profile": "shock", "intensity": 6.0,
                    "duration": int(rng.integers(1, 3)),
                    "bad_batch": False, "dir": 1, "good_adjust": True,
                })
            events.sort(key=lambda e: e["date"])
        df, _, nominal = simulate(rng, pi, events)
        df_all[pi] = df
        events_all[pi] = events
        nominals[pi] = nominal
        print(f"{PRODUCTS[pi-1][0]} {PRODUCTS[pi-1][1]}: {len(df)} 天, "
              f"{len(events)} 个事件, 平均合格率 {df['pass_rate'].mean():.2f}%")
    db_path = build_database(df_all, events_all, nominals)
    export_model_csv(df_all, events_all)
    print(f"数据库: {db_path}")
    print(f"CSV 输出: {OUT_DIR}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shock", action="store_true",
                    help="尖锐事件效应制度（机制验证实验）")
    ap.add_argument("--out", default=None, help="输出目录覆盖")
    args = ap.parse_args()
    try:
        main(shock=args.shock, out_dir=args.out)
    except Exception:
        os.makedirs(OUT_DIR, exist_ok=True)
        from pathlib import Path
        Path(OUT_DIR, "error.log").write_text(traceback.format_exc(),
                                              encoding="utf-8")
        raise
