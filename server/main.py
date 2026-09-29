# -*- coding: utf-8 -*-
"""机械加工质量管理系统 · FastAPI 服务端（自包含单文件）。

包含：SQLite 访问 / JWT 认证 / 业务 CRUD / NCR 闭环 / 看板 / SPC /
      E-QTSF 推理服务（预测 + 超标概率 + 事件归因）/ 静态前端托管。

启动：.venv/Scripts/python.exe -m uvicorn server.main:app --port 8000
"""
import csv
import hashlib
import json
import os
import secrets
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import jwt
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE, "data", "output", "qms.db")
WEB_DIR = os.path.join(BASE, "web")
MODEL_DIR = os.path.join(BASE, "runs")
DATA_DIR = os.path.join(BASE, "data", "output")
JWT_SECRET = os.environ.get("JWT_SECRET", "eqtsf-dev-secret-2024")
JWT_EXPIRE_H = 12

app = FastAPI(title="机械加工质量管理系统（E-QTSF）", version="1.0")

_local = threading.local()


# ================== DB ==================
def conn() -> sqlite3.Connection:
    c = getattr(_local, "conn", None)
    if c is None:
        if not os.path.exists(DB_PATH):
            raise RuntimeError(f"数据库不存在：{DB_PATH}（先运行 data/generator.py）")
        c = sqlite3.connect(DB_PATH, check_same_thread=False)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        _local.conn = c
    return c


def q(sql: str, params: Tuple = ()) -> List[Dict[str, Any]]:
    return [dict(r) for r in conn().execute(sql, params).fetchall()]


def q1(sql: str, params: Tuple = ()) -> Optional[Dict[str, Any]]:
    r = conn().execute(sql, params).fetchone()
    return dict(r) if r else None


def ex(sql: str, params: Tuple = ()) -> int:
    cur = conn().execute(sql, params)
    conn().commit()
    return cur.lastrowid


def next_no(table: str, col: str, prefix: str) -> str:
    """自动编号：PREFIX-YYYY-NNN（事务内查最大值 +1）。"""
    yr = datetime.now().year
    row = q1(f"SELECT {col} AS no FROM {table} WHERE {col} LIKE ? "
             f"ORDER BY {col} DESC LIMIT 1", (f"{prefix}-{yr}-%",))
    seq = int(row["no"].rsplit("-", 1)[1]) + 1 if row else 1
    return f"{prefix}-{yr}-{seq:03d}"


# ================== 认证 ==================
def hash_pwd(pwd: str, salt: str = None) -> str:
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", pwd.encode(), salt.encode(), 100_000)
    return f"PBKDF2${salt}${dk.hex()}"


def verify_pwd(pwd: str, stored: str) -> bool:
    if stored == "PBKDF2$pending":          # 生成器占位 → 默认密码 123456
        stored = DEFAULT_HASH
    try:
        _, salt, hexd = stored.split("$")
    except ValueError:
        return False
    return hash_pwd(pwd, salt) == stored


DEFAULT_HASH = hash_pwd("123456")


class LoginIn(BaseModel):
    username: str
    password: str


@app.post("/api/auth/login")
def login(body: LoginIn):
    u = q1("SELECT * FROM sys_user WHERE username=? AND deleted=0",
           (body.username,))
    if not u or not verify_pwd(body.password, u["password_hash"]):
        raise HTTPException(401, "用户名或密码错误")
    if u["password_hash"] == "PBKDF2$pending":
        ex("UPDATE sys_user SET password_hash=? WHERE id=?",
           (DEFAULT_HASH, u["id"]))
    token = jwt.encode(
        {"sub": u["username"], "uid": u["id"], "role": u["role"],
         "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRE_H)},
        JWT_SECRET, algorithm="HS256")
    return {"token": token, "user": {"id": u["id"], "username": u["username"],
                                     "real_name": u["real_name"],
                                     "role": u["role"]}}


def auth(request: Request) -> Dict[str, Any]:
    h = request.headers.get("Authorization", "")
    if not h.startswith("Bearer "):
        raise HTTPException(401, "未登录")
    try:
        return jwt.decode(h[7:], JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(401, "登录已过期")


def require_admin(user: Dict = Depends(auth)):
    if user.get("role") != "ADMIN":
        raise HTTPException(403, "需要管理员权限")
    return user


# ================== NCR 闭环 ==================
NCR_FLOW = {
    "待评审": {"submit": "处理中", "reject": "已驳回"},
    "处理中": {"verify_pass": "已关闭", "verify_fail": "待评审"},
    "已驳回": {"close": "已关闭"},
}
NCR_ACTION_NAME = {"submit": "提交评审", "reject": "评审驳回",
                   "verify_pass": "验证通过", "verify_fail": "验证不通过",
                   "close": "关闭"}


class NcrIn(BaseModel):
    product_id: int
    defect_code_id: Optional[int] = None
    defect_quantity: int = 0
    severity: str = "一般"
    description: str = ""
    disposition: str = "返工"


class NcrActionIn(BaseModel):
    action: str
    opinion: str = ""


@app.get("/api/ncr")
def list_ncr(user: Dict = Depends(auth)):
    return q("SELECT n.*, p.product_name, d.dept_name, c.defect_name "
             "FROM nonconforming_report n JOIN product p ON p.id=n.product_id "
             "LEFT JOIN department d ON d.id=n.responsible_dept_id "
             "LEFT JOIN defect_code c ON c.id=n.defect_code_id "
             "WHERE n.deleted=0 ORDER BY n.id DESC LIMIT 300")


@app.post("/api/ncr")
def create_ncr(body: NcrIn, user: Dict = Depends(auth)):
    no = next_no("nonconforming_report", "report_no", "NCR")
    p = q1("SELECT product_name FROM product WHERE id=?", (body.product_id,))
    nid = ex(
        "INSERT INTO nonconforming_report(report_no,product_id,batch_no,"
        "defect_code_id,defect_quantity,severity,discovery_date,"
        "responsible_dept_id,reporter_id,description,disposition,status)"
        " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (no, body.product_id, "", body.defect_code_id, body.defect_quantity,
         body.severity, datetime.now().strftime("%Y-%m-%d"),
         q1("SELECT id FROM department WHERE dept_name='质量部'")["id"],
         user["uid"], body.description, body.disposition, "待评审"))
    ex("INSERT INTO ncr_status_log(ncr_id,from_status,to_status,operator_id,"
       "opinion) VALUES(?,?,?,?,?)", (nid, "", "待评审", user["uid"], "创建"))
    return {"id": nid, "report_no": no, "product": p and p["product_name"]}


@app.post("/api/ncr/{nid}/action")
def ncr_action(nid: int, body: NcrActionIn, user: Dict = Depends(auth)):
    n = q1("SELECT * FROM nonconforming_report WHERE id=? AND deleted=0", (nid,))
    if not n:
        raise HTTPException(404, "NCR 不存在")
    flow = NCR_FLOW.get(n["status"], {})
    if body.action not in flow:
        raise HTTPException(400, f"状态 {n['status']} 不允许动作 {body.action}")
    to = flow[body.action]
    ex("UPDATE nonconforming_report SET status=?, close_date=? WHERE id=?",
       (to, datetime.now().strftime("%Y-%m-%d") if to == "已关闭" else None, nid))
    ex("INSERT INTO ncr_status_log(ncr_id,from_status,to_status,operator_id,"
       "opinion) VALUES(?,?,?,?,?)",
       (nid, n["status"], to, user["uid"], body.opinion or NCR_ACTION_NAME[body.action]))
    return {"ok": True, "status": to}


@app.get("/api/ncr/{nid}/logs")
def ncr_logs(nid: int, user: Dict = Depends(auth)):
    return q("SELECT l.*, u.real_name FROM ncr_status_log l "
             "LEFT JOIN sys_user u ON u.id=l.operator_id "
             "WHERE l.ncr_id=? ORDER BY l.id", (nid,))


# ================== 看板 ==================
@app.get("/api/dashboard")
def dashboard(user: Dict = Depends(auth)):
    trend = q("SELECT r.record_date AS date, AVG(r.pass_rate) AS pass_rate, "
              "AVG(r.ppm) AS ppm FROM quality_record r WHERE r.deleted=0 "
              "GROUP BY r.record_date ORDER BY r.record_date DESC LIMIT 90")
    trend.reverse()
    top_defects = q("SELECT c.defect_name, SUM(dt.defect_quantity) AS qty "
                    "FROM defect_detail dt JOIN defect_code c ON c.id=dt.defect_code_id "
                    "GROUP BY c.id ORDER BY qty DESC LIMIT 5")
    dept_stats = q("SELECT d.dept_name, SUM(dt.defect_quantity) AS qty "
                   "FROM defect_detail dt JOIN defect_code c ON c.id=dt.defect_code_id "
                   "JOIN department d ON d.id=c.responsible_dept_id "
                   "GROUP BY d.id ORDER BY qty DESC")
    product_rank = q("SELECT p.product_name, AVG(r.pass_rate) AS pass_rate "
                     "FROM quality_record r JOIN product p ON p.id=r.product_id "
                     "WHERE r.deleted=0 GROUP BY p.id ORDER BY pass_rate")
    summary = q1("SELECT COUNT(*) AS n_products FROM product WHERE deleted=0")
    ncr_open = q1("SELECT COUNT(*) AS n FROM nonconforming_report "
                  "WHERE status IN ('待评审','处理中') AND deleted=0")["n"]
    ev_month = q1("SELECT COUNT(*) AS n FROM quality_event "
                  "WHERE event_date >= ?",
                  ((datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),))["n"]
    alerts = q1("SELECT COUNT(*) AS n FROM quality_alert WHERE status='未处理'")["n"]
    return {"trend": trend, "top_defects": top_defects,
            "dept_stats": dept_stats, "product_rank": product_rank,
            "summary": {"products": summary["n_products"],
                        "ncr_open": ncr_open, "events_30d": ev_month,
                        "alerts": alerts}}


# ================== SPC ==================
@app.get("/api/spc/{product_id}")
def spc(product_id: int, days: int = 90, user: Dict = Depends(auth)):
    """X̄-R 图 + P 图 + PPM。"""
    p = q1("SELECT * FROM product WHERE id=?", (product_id,))
    if not p:
        raise HTTPException(404, "产品不存在")
    # --- X̄-R：inspection_value 每日 5 样本 ---
    rows = q("SELECT inspection_date AS d, measured_value AS v FROM "
             "inspection_value WHERE product_id=? ORDER BY inspection_date "
             "DESC LIMIT ?", (product_id, days * 5))
    by_day: Dict[str, List[float]] = {}
    for r in rows:
        by_day.setdefault(r["d"], []).append(r["v"])
    days_sorted = sorted(by_day)[-days:]
    xbar = [float(np_mean(by_day[d])) for d in days_sorted]
    rng = [max(by_day[d]) - min(by_day[d]) for d in days_sorted]
    xbb = np_mean(xbar) if xbar else 0.0
    rbar = np_mean(rng) if rng else 0.0
    a2, d3, d4 = 0.577, 0.0, 2.114          # n=5 系数
    spc_x = {"ucl": xbb + a2 * rbar, "cl": xbb, "lcl": xbb - a2 * rbar}
    spc_r = {"ucl": d4 * rbar, "cl": rbar, "lcl": d3 * rbar}
    # --- P 图 + PPM：quality_record ---
    qr = q("SELECT record_date AS d, total_quantity AS n, defect_quantity AS dft "
           "FROM quality_record WHERE product_id=? ORDER BY record_date DESC "
           "LIMIT ?", (product_id, days))
    qr.reverse()
    pbar = (sum(r["dft"] for r in qr) /
            max(sum(r["n"] for r in qr), 1)) if qr else 0.0
    p_chart = []
    for r in qr:
        n = max(r["n"], 1)
        pr = r["dft"] / n
        s = (pbar * (1 - pbar) / n) ** 0.5
        p_chart.append({"date": r["d"], "p": round(pr * 100, 3),
                        "ucl": round((pbar + 3 * s) * 100, 3),
                        "lcl": round(max(pbar - 3 * s, 0) * 100, 3)})
    ppm = round(pbar * 1e6, 1)
    # OOC 点统计
    ooc_x = sum(1 for v in xbar
                if v > spc_x["ucl"] or v < spc_x["lcl"]) if xbar else 0
    return {"product": p, "dates": days_sorted,
            "xbar": [round(v, 4) for v in xbar], "range": [round(v, 4) for v in rng],
            "spc_x": spc_x, "spc_r": spc_r, "p_chart": p_chart,
            "ppm": ppm, "pbar_pct": round(pbar * 100, 3), "ooc_xbar": ooc_x}


def np_mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# ================== E-QTSF 推理服务 ==================
_MODEL = {"obj": None, "dev": None}


def load_model():
    import torch
    from model.model import EQTSF
    if _MODEL["obj"] is not None:
        return _MODEL["obj"], _MODEL["dev"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    path = os.path.join(MODEL_DIR, "eqtsf_full", "best.pt")
    if not os.path.exists(path):
        raise HTTPException(503, "模型未训练（先运行 scripts/run_all.sh）")
    ck = torch.load(path, map_location=dev)
    m = EQTSF(c_in=5, d_model=ck["d_model"], layers=ck["layers"],
              **ck["flags"]).to(dev).eval()
    _MODEL["obj"], _MODEL["dev"] = m, dev
    return m, dev


def read_series(pcode: str):
    path = os.path.join(DATA_DIR, f"series_{pcode}.csv")
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return rows


def read_events(pcode: str):
    path = os.path.join(DATA_DIR, f"events_{pcode}.csv")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def day_index(date_iso: str) -> int:
    from datetime import date
    y, m, d = map(int, date_iso.split("-"))
    return (date(y, m, d) - date(2024, 1, 1)).days


def build_infer_batch(pcode: str, extra_events: List[Dict] = None):
    """构建最新窗口的推理 batch（与训练协议一致的 RevIN/事件对齐）。

    extra_events：系统内登记的计划事件（DB），与 CSV 历史事件合并。
    """
    import numpy as np
    import torch
    from model.dataset import (CHANNELS, CTX_DAYS, FORE_DAYS, SEG,
                               segment_events)
    rows = read_series(pcode)
    arr = np.array([[float(r[c]) for c in CHANNELS] for r in rows])
    evs = [{"day": day_index(r["date"]), "event_type": r["event_type"],
            "event_subtype": r["event_subtype"], "severity": r["severity"]}
           for r in read_events(pcode)]
    for e in (extra_events or []):
        evs.append({"day": day_index(e["event_date"]), "date": e["event_date"],
                    "event_type": e["event_type"],
                    "event_subtype": e["event_subtype"],
                    "severity": e["severity"]})
    n = arr.shape[0]
    t_end = n                                            # 预测从最后一天之后
    ctx = arr[t_end - CTX_DAYS:t_end].copy()
    tgt = np.zeros((FORE_DAYS, len(CHANNELS)), dtype=np.float64)
    mu = ctx.mean(axis=0)
    sd = ctx.std(axis=0) + 1e-5
    ctx_n = (ctx - mu) / sd
    tgt_n = (tgt - mu) / sd
    daily, weekly, monthly = ctx_n, \
        ctx_n.reshape(12, SEG, -1).mean(axis=1), \
        ctx_n.reshape(3, 28, -1).mean(axis=1)
    short = ctx_n[-28:]
    hist_ev = np.stack([segment_events(t_end - CTX_DAYS + s * SEG,
                                       t_end - CTX_DAYS + (s + 1) * SEG, evs)
                        for s in range(CTX_DAYS // SEG)])
    fut_ev = np.stack([segment_events(t_end + s * SEG,
                                      t_end + (s + 1) * SEG, evs)
                       for s in range(FORE_DAYS // SEG)])
    # 启发式统计量（通道 0 未归一化末 28 天）
    from model.dataset import QTSFDataset
    stats = QTSFDataset._heur_stats(arr[t_end - 28:t_end, 0])
    b = {k: torch.from_numpy(v.astype(np.float32)).unsqueeze(0)
         for k, v in [("ctx", ctx_n), ("daily", daily), ("weekly", weekly),
                      ("monthly", monthly), ("short", short),
                      ("hist_events", hist_ev), ("fut_events", fut_ev),
                      ("heur", stats), ("mu", mu), ("sd", sd)]}
    b["target"] = torch.from_numpy(tgt_n.astype(np.float32)).unsqueeze(0)
    b["pcode"] = [pcode]
    return b, rows, evs, mu, sd


@app.post("/api/forecast/run/{product_id}")
def forecast_run(product_id: int, user: Dict = Depends(auth)):
    """运行 E-QTSF 预测：28 天合格率概率预测 + 超标概率 + 事件归因。"""
    import numpy as np
    import torch
    p = q1("SELECT * FROM product WHERE id=?", (product_id,))
    if not p:
        raise HTTPException(404, "产品不存在")
    pcode = p["product_code"]
    model, dev = load_model()
    last_date = read_series(pcode)[-1]["date"]
    plan_evs = q("SELECT event_date, event_type, event_subtype, severity "
                 "FROM quality_event WHERE product_id=? AND deleted=0 "
                 "AND event_date > ?", (product_id, last_date))
    b, rows, evs, mu, sd = build_infer_batch(pcode, extra_events=plan_evs)
    b = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v)
         for k, v in b.items()}
    with torch.no_grad():
        s = model.sample(b, n_samples=100, n_steps=12)   # (1,28,C,K)
    s_pr = s[0, :, 0].cpu().numpy() * sd[0] + mu[0]      # (28, K) 合格率
    med = np.median(s_pr, axis=1)
    q10 = np.quantile(s_pr, 0.10, axis=1)
    q90 = np.quantile(s_pr, 0.90, axis=1)
    # 阈值：近 60 天均值 - 2σ
    recent = [float(r["pass_rate"]) for r in rows[-60:]]
    thr = float(np.mean(recent) - 2 * np.std(recent))
    p_breach = (s_pr < thr).mean(axis=1)                 # (28,)
    # 事件归因：逐事件消融
    from model.dataset import (EVENT_SUBTYPES, EVENT_TYPES, SEVERITIES,
                               event_feature, segment_events)
    t_end = len(rows)
    fut = [e for e in evs if t_end <= e["day"] < t_end + 28]
    attributions = []
    if fut:
        base_ev = b["fut_events"].clone()
        null = np.stack([segment_events(1, 0, [])] * 4)
        with torch.no_grad():
            b["fut_events"] = torch.from_numpy(
                null.astype(np.float32)).unsqueeze(0).to(dev)
            s0 = model.sample(b, 30, 10)[0, :, 0].cpu().numpy() * sd[0] + mu[0]
            for e in fut:
                seg = min((e["day"] - t_end) // 7, 3)
                one = null.copy()
                one[seg] = event_feature(e)
                b["fut_events"] = torch.from_numpy(
                    one.astype(np.float32)).unsqueeze(0).to(dev)
                s1 = model.sample(b, 30, 10)[0, :, 0].cpu().numpy() * sd[0] + mu[0]
                delta = float(np.mean(np.abs(s1 - s0)))
                attributions.append({
                    "event": f"{e['event_type']}/{e['event_subtype']}"
                             f"（{e['severity']}）",
                    "date": e.get("date", ""), "delta": round(delta, 4)})
        b["fut_events"] = base_ev
    attributions.sort(key=lambda x: -x["delta"])
    # 落库：forecast_result
    fdate = rows[-1]["date"]
    y, m, d = map(int, fdate.split("-"))
    from datetime import date as _date
    ex("DELETE FROM forecast_result WHERE product_id=? AND forecast_date=?",
       (product_id, fdate))
    for i in range(28):
        td = (_date(y, m, d) + timedelta(days=i + 1)).isoformat()
        ex("INSERT INTO forecast_result(product_id,forecast_date,target_date,"
           "horizon,y_pred,y_lower,y_upper,exceed_prob,model_version)"
           " VALUES(?,?,?,?,?,?,?,?,?)",
           (product_id, fdate, td, i + 1, round(float(med[i]), 3),
            round(float(q10[i]), 3), round(float(q90[i]), 3),
            round(float(p_breach[i]), 4), "eqtsf_full"))
    # 预警判定
    worst7 = float(p_breach[:7].max())
    worst14 = float(p_breach[:14].max())
    level, msg = None, ""
    if worst7 >= 0.8:
        level, msg = "红", f"未来 7 天跌破阈值({thr:.2f}%)概率 {worst7:.0%}"
    elif worst7 >= 0.6:
        level, msg = "橙", f"未来 7 天跌破阈值概率 {worst7:.0%}"
    elif worst14 >= 0.5:
        level, msg = "黄", f"未来 14 天跌破阈值概率 {worst14:.0%}"
    alert_id = None
    if level:
        ex("DELETE FROM quality_alert WHERE product_id=? AND alert_date=? "
           "AND alert_type='FORECAST'", (product_id, fdate))
        alert_id = ex(
            "INSERT INTO quality_alert(product_id,alert_date,alert_type,"
            "level,message,attribution,related_event_ids,status)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (product_id, fdate, "FORECAST", level, msg,
             json.dumps(attributions[:3], ensure_ascii=False),
             ",".join(str(a) for a in []), "未处理"))
    return {"product": p, "forecast_date": fdate, "threshold": round(thr, 3),
            "dates": [r["date"] for r in []] or
                     [(_date(y, m, d) + timedelta(days=i + 1)).isoformat()
                      for i in range(28)],
            "median": [round(float(v), 3) for v in med],
            "q10": [round(float(v), 3) for v in q10],
            "q90": [round(float(v), 3) for v in q90],
            "exceed_prob": [round(float(v), 4) for v in p_breach],
            "level": level, "message": msg,
            "attribution": attributions[:5],
            "history": [{"date": r["date"], "pass_rate": r["pass_rate"]}
                        for r in rows[-42:]]}


@app.get("/api/forecast/latest/{product_id}")
def forecast_latest(product_id: int, user: Dict = Depends(auth)):
    rows = q("SELECT * FROM forecast_result WHERE product_id=? "
             "ORDER BY forecast_date DESC, horizon LIMIT 28", (product_id,))
    if not rows:
        return {"empty": True}
    return {"empty": False, "rows": rows}


@app.get("/api/alerts")
def alerts(user: Dict = Depends(auth)):
    return q("SELECT a.*, p.product_name FROM quality_alert a "
             "JOIN product p ON p.id=a.product_id ORDER BY a.id DESC LIMIT 100")


class AlertHandleIn(BaseModel):
    action: str        # confirm / create_ncr / ignore
    opinion: str = ""


@app.post("/api/alerts/{aid}/handle")
def alert_handle(aid: int, body: AlertHandleIn, user: Dict = Depends(auth)):
    a = q1("SELECT * FROM quality_alert WHERE id=?", (aid,))
    if not a:
        raise HTTPException(404, "预警不存在")
    if body.action == "create_ncr":
        no = next_no("nonconforming_report", "report_no", "NCR")
        nid = ex("INSERT INTO nonconforming_report(report_no,product_id,"
                 "defect_quantity,severity,discovery_date,reporter_id,"
                 "description,disposition,status) VALUES(?,?,?,?,?,?,?,?,?)",
                 (no, a["product_id"], 0, "严重",
                  datetime.now().strftime("%Y-%m-%d"), user["uid"],
                  f"预测预警自动创建：{a['message']}", "返工", "待评审"))
        ex("UPDATE quality_alert SET status='已生成NCR', ncr_id=? WHERE id=?",
           (nid, aid))
        return {"ok": True, "ncr_id": nid, "report_no": no}
    status = {"confirm": "已确认", "ignore": "已忽略"}.get(body.action)
    if not status:
        raise HTTPException(400, body.action)
    ex("UPDATE quality_alert SET status=? WHERE id=?", (status, aid))
    return {"ok": True}


# ================== 用户管理 ==================
@app.get("/api/users")
def users(user: Dict = Depends(require_admin)):
    return q("SELECT u.id,u.username,u.real_name,u.role,d.dept_name "
             "FROM sys_user u LEFT JOIN department d ON d.id=u.dept_id "
             "WHERE u.deleted=0")


class UserIn(BaseModel):
    username: str
    password: str = ""
    real_name: str = ""
    role: str = "INSPECTOR"
    dept_name: str = ""


@app.post("/api/users")
def add_user(body: UserIn, user: Dict = Depends(require_admin)):
    if q1("SELECT id FROM sys_user WHERE username=?", (body.username,)):
        raise HTTPException(400, "用户名已存在")
    did = q1("SELECT id FROM department WHERE dept_name=?",
             (body.dept_name or "质量部",))
    ex("INSERT INTO sys_user(username,password_hash,real_name,role,dept_id)"
       " VALUES(?,?,?,?,?)",
       (body.username, hash_pwd(body.password or "123456"),
        body.real_name or body.username, body.role,
        did["id"] if did else None))
    return {"ok": True}


# ================== 通用 CRUD ==================
TABLES = {
    "products": ("product", ["product_code", "product_name", "product_type",
                             "specification", "material", "unit", "status"]),
    "defect-codes": ("defect_code", ["defect_code", "defect_name", "category",
                                     "severity", "description"]),
    "departments": ("department", ["dept_name", "parent_id", "sort_order"]),
    "events": ("quality_event", ["product_id", "event_date", "event_type",
                                 "event_subtype", "severity", "description"]),
}
AUTO_NO = {"products": ("product", "product_code", "PRD"),
           "defect-codes": ("defect_code", "defect_code", "DEF"),
           "events": ("quality_event", "event_no", "EVT")}


@app.get("/api/{table}")
def list_table(table: str, user: Dict = Depends(auth)):
    if table == "products":
        return q("SELECT * FROM product WHERE deleted=0 ORDER BY id")
    if table == "defect-codes":
        return q("SELECT * FROM defect_code WHERE deleted=0 ORDER BY id")
    if table == "departments":
        return q("SELECT * FROM department WHERE deleted=0 ORDER BY sort_order")
    if table == "batches":
        return q("SELECT b.*, p.product_name, p.product_code FROM product_batch b "
                 "JOIN product p ON p.id=b.product_id WHERE b.deleted=0 "
                 "ORDER BY b.id DESC LIMIT 300")
    if table == "quality-records":
        return q("SELECT r.*, p.product_name, p.product_code, b.batch_no "
                 "FROM quality_record r JOIN product p ON p.id=r.product_id "
                 "JOIN product_batch b ON b.id=r.batch_id WHERE r.deleted=0 "
                 "ORDER BY r.record_date DESC, r.id DESC LIMIT 300")
    if table == "events":
        return q("SELECT e.*, p.product_name, p.product_code, d.dept_name "
                 "FROM quality_event e JOIN product p ON p.id=e.product_id "
                 "LEFT JOIN department d ON d.id=e.responsible_dept_id "
                 "WHERE e.deleted=0 ORDER BY e.event_date DESC, e.id DESC LIMIT 400")
    raise HTTPException(404, table)


class ItemIn(BaseModel):
    fields: Dict[str, Any]


@app.post("/api/{table}")
def create_item(table: str, body: ItemIn, user: Dict = Depends(auth)):
    if table not in TABLES:
        raise HTTPException(404, table)
    t, cols = TABLES[table]
    if table in AUTO_NO:
        tname, col, prefix = AUTO_NO[table]
        body.fields[col] = next_no(tname, col, prefix)
        if col not in cols:
            cols = cols + [col]
    if table == "events" and "description" in body.fields \
            and not body.fields["description"]:
        body.fields["description"] = (
            f"[计划] {body.fields.get('event_subtype', '')}"
            f"（{body.fields.get('severity', '一般')}）")
    use = [c for c in cols if c in body.fields]
    if not use:
        raise HTTPException(400, "无有效字段")
    vals = [body.fields[c] for c in use]
    pid = ex(f"INSERT INTO {t}({','.join(use)}) VALUES({','.join('?' * len(use))})",
             tuple(vals))
    return {"id": pid}


@app.put("/api/{table}/{item_id}")
def update_item(table: str, item_id: int, body: ItemIn,
                user: Dict = Depends(auth)):
    if table not in TABLES:
        raise HTTPException(404, table)
    t, cols = TABLES[table]
    use = [c for c in cols if c in body.fields]
    if not use:
        raise HTTPException(400, "无有效字段")
    sets = ",".join(f"{c}=?" for c in use)
    ex(f"UPDATE {t} SET {sets} WHERE id=?", 
       tuple(body.fields[c] for c in use) + (item_id,))
    return {"ok": True}


@app.delete("/api/{table}/{item_id}")
def delete_item(table: str, item_id: int, user: Dict = Depends(require_admin)):
    if table not in TABLES:
        raise HTTPException(404, table)
    ex(f"UPDATE {TABLES[table][0]} SET deleted=1 WHERE id=?", (item_id,))
    return {"ok": True}




# ================== 静态前端 ==================
@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


if os.path.isdir(os.path.join(WEB_DIR, "vendor")):
    app.mount("/vendor", StaticFiles(directory=os.path.join(WEB_DIR, "vendor")),
              name="vendor")
if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
