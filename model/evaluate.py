# -*- coding: utf-8 -*-
"""最终评估 + 论文图表生成。

用法：python evaluate.py            # 全量评估 + 图表
      python evaluate.py --smoke   # 冒烟（少量样本）
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dataset import QTSFDataset, collate  # noqa: E402
from model.metrics import (alert_metrics, crps_sample, mae_mse_rmse,  # noqa
                           wql)
from model.model import EQTSF  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "output")
RUNS = os.path.join(ROOT, "runs")
FIGS = os.path.join(ROOT, "docs", "figures")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_model(run_name, dev):
    ck = torch.load(os.path.join(RUNS, run_name, "best.pt"),
                    map_location=dev)
    model = EQTSF(c_in=5, d_model=ck["d_model"], layers=ck["layers"],
                  **ck["flags"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


@torch.no_grad()
def collect(model, loader, dev, n_samples=50, n_steps=10):
    """返回 (y, samples, median, meta) —— pass_rate 通道还原后。"""
    ys, ss, md, meta, ev, ctxs = [], [], [], [], [], []
    for batch in loader:
        b = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v)
             for k, v in batch.items()}
        s = model.sample(b, n_samples, n_steps)             # (B,28,C,K)
        mu = b["mu"][:, 0].view(-1, 1, 1)
        sd = b["sd"][:, 0].view(-1, 1, 1)
        s_pr = s[:, :, 0] * sd + mu                          # (B,28,K)
        y_pr = b["target"][:, :, 0] * sd.squeeze(-1) + mu.squeeze(-1)
        ys.append(y_pr.cpu().numpy())
        ss.append(s_pr.cpu().numpy())
        md.append(s_pr.median(dim=-1).values.cpu().numpy())
        meta += list(zip(batch["pcode"], batch["t_end"].tolist()))
        ev.append(batch["fut_events"][..., -2].any(dim=1)
                  .cpu().numpy().astype(bool))
        mu_c = batch["mu"][:, 0].cpu().view(-1, 1)
        sd_c = batch["sd"][:, 0].cpu().view(-1, 1)
        ctx0 = batch["ctx"][:, :, 0].cpu() * sd_c + mu_c
        ctxs.append(ctx0.numpy())
    return (np.concatenate(ys), np.concatenate(ss),
            np.concatenate(md), meta, np.concatenate(ev),
            np.concatenate(ctxs))


def eval_model_metrics(model, loader, dev, smoke=False):
    y, s, m, meta, ev, ctxs = collect(model, loader, dev,
                                      n_samples=20 if smoke else 50)
    flat_y = y.reshape(-1)
    flat_s = s.reshape(-1, s.shape[-1])
    flat_m = m.reshape(-1)
    mae, mse, rmse = mae_mse_rmse(flat_y, flat_m)
    # 事件分层 CRPS：预测窗含未来事件的窗口 vs 平静窗口
    crps_i = np.array([crps_sample(y[i], s[i]) for i in range(len(y))])
    ev_mask = np.asarray(ev, dtype=bool)
    q10 = np.quantile(s, 0.10, axis=2)
    q90 = np.quantile(s, 0.90, axis=2)
    picp = float(((y >= q10) & (y <= q90)).mean())
    width = float((q90 - q10).mean())
    horizons = {}
    for lo, hi, hname in [(0, 7, "1-7d"), (7, 14, "8-14d"),
                          (14, 21, "15-21d"), (21, 28, "22-28d")]:
        horizons[hname] = float(np.mean(
            [crps_sample(y[i, lo:hi], s[i, lo:hi]) for i in range(len(y))]))
    thr = ctxs.mean(axis=1) - 2 * ctxs.std(axis=1)
    am = alert_metrics(y, s, threshold=float(np.mean(thr)))
    out = {"crps": crps_sample(flat_y, flat_s), "wql": wql(flat_y, flat_s),
           "mae": mae, "mse": mse, "rmse": rmse,
           "crps_event": float(crps_i[ev_mask].mean()) if ev_mask.any() else None,
           "crps_calm": float(crps_i[~ev_mask].mean()) if (~ev_mask).any() else None,
           "n_event_windows": int(ev_mask.sum()),
           "n_calm_windows": int((~ev_mask).sum()),
           "picp80": picp, "interval_width": width,
           "crps_by_horizon": horizons,
           "alert_dyn": {k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in am.items()}}
    return out


def fig_training(run="eqtsf_full"):
    log_path = os.path.join(RUNS, run, "log.json")
    if not os.path.exists(log_path):
        return
    with open(log_path, encoding="utf-8") as f:
        log = json.load(f)
    eps = [e["epoch"] for e in log["epochs"] if "crps" in e]
    fm = [e["loss_fm"] for e in log["epochs"] if "crps" in e]
    crps = [e["crps"] for e in log["epochs"] if "crps" in e]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(eps, fm, "o-", color="#4062BB", label="Flow Matching Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("FM Loss", color="#4062BB")
    ax2 = ax1.twinx()
    ax2.plot(eps, crps, "s--", color="#B5442D", label="Val CRPS")
    ax2.set_ylabel("Val CRPS", color="#B5442D")
    ax1.set_title(f"E-QTSF Training Curve ({run})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_training.png"), dpi=150)
    plt.close(fig)


def fig_forecast(model, loader, dev, data_dir):
    """单产品测试窗预测区间图（含事件标记）。"""
    ds = loader.dataset
    smp = ds.samples[len(ds.samples) // 2]
    pc = smp["pcode"]
    # 找到该样本所在 batch
    b = collate([smp])
    b = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v)
         for k, v in b.items()}
    s = model.sample(b, 100, 10)[0, :, :, :].cpu().numpy()  # (28,C,K)
    q10 = np.quantile(s[:, 0], 0.10, axis=1)
    q90 = np.quantile(s[:, 0], 0.90, axis=1)
    med = np.median(s[:, 0], axis=1)
    y = smp["target"][:, 0] * smp["sd"][0] + smp["mu"][0]
    # 还原 med
    med = med * smp["sd"][0] + smp["mu"][0]
    q10 = q10 * smp["sd"][0] + smp["mu"][0]
    q90 = q90 * smp["sd"][0] + smp["mu"][0]
    # 历史末 42 天（还原）
    hist = smp["ctx"][-42:, 0] * smp["sd"][0] + smp["mu"][0]
    # 未来 28 天事件
    evs = []
    t_end = smp["t_end"]
    ev_path = os.path.join(data_dir, f"events_{pc}.csv")
    if os.path.exists(ev_path):
        import csv
        with open(ev_path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                from model.dataset import QTSFDataset
                di = QTSFDataset._day_index(r["date"])
                if t_end <= di < t_end + 28:
                    evs.append((di - t_end, r["event_subtype"], r["severity"]))
    xs = np.arange(28)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    hx = np.arange(-42, 0)
    ax.plot(hx, hist, color="#5A5A5A", lw=1.2, label="历史合格率")
    ax.plot(xs, y, "o-", color="#1F77B4", ms=3, lw=1.4, label="真实合格率")
    ax.plot(xs, med, "s--", color="#D62728", ms=3, lw=1.4, label="E-QTSF 中位数预测")
    ax.fill_between(xs, q10, q90, color="#D62728", alpha=0.15, label="80% 预测区间")
    for dx, sub, sev in evs:
        color = {"致命": "#8B0000", "严重": "#E4572E",
                 "一般": "#F3A712", "轻微": "#76B5C5"}.get(sev, "#999999")
        ax.axvline(dx, color=color, ls=":", lw=1.2)
        ax.annotate(sub, (dx, ax.get_ylim()[1]), rotation=90,
                    fontsize=6.5, color=color, va="top")
    ax.axvline(0, color="k", lw=0.8, ls="-", alpha=0.5)
    ax.set_xlabel("相对预测起点的天数")
    ax.set_ylabel("合格率 (%)")
    ax.set_title(f"E-QTSF 概率预测（{pc}，测试窗，事件以竖线标注）")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_forecast.png"), dpi=150)
    plt.close(fig)


def fig_ablation(results):
    names = [r["run"] for r in results]
    labels = {"eqtsf_full": "E-QTSF 完整", "eqtsf_no_event": "去事件条件化",
              "eqtsf_no_timestep": "去事件感知时间步",
              "eqtsf_no_contrast": "去对比学习",
              "eqtsf_no_multiscale": "去多尺度编码"}
    crps = [r["crps"] for r in results]
    mae = [r["mae"] for r in results]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(x, crps, color="#4062BB", width=0.6)
    axes[0].set_title("CRPS（越低越好）")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([labels.get(n, n) for n in names],
                            rotation=20, fontsize=8)
    axes[1].bar(x, mae, color="#B5442D", width=0.6)
    axes[1].set_title("MAE（越低越好）")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([labels.get(n, n) for n in names],
                            rotation=20, fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("消融实验")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_ablation.png"), dpi=150)
    plt.close(fig)


def fig_baselines(all_results):
    base = [r for r in all_results if r.get("model") in
            ("seasonal_naive", "dlinear", "itransformer", "patchtst")]
    if not base:
        return
    names = [r["model"] for r in base]
    labels = {"seasonal_naive": "SeasonalNaive", "dlinear": "DLinear",
              "itransformer": "iTransformer-lite", "patchtst": "PatchTST-lite"}
    mae = [r["mae"] for r in base]
    crps = [r.get("crps") for r in base]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.18, mae, 0.36, label="MAE", color="#4062BB")
    if all(c is not None for c in crps):
        ax2 = ax.twinx()
        ax2.bar(x + 0.18, crps, 0.36, label="CRPS", color="#B5442D")
        ax2.set_ylabel("CRPS", color="#B5442D")
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(n, n) for n in names], fontsize=9)
    ax.set_ylabel("MAE", color="#4062BB")
    ax.set_title("基线对比（测试集，pass_rate 通道）")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_baselines.png"), dpi=150)
    plt.close(fig)


def fig_attribution(model, loader, dev):
    """事件归因示例：注意力权重 × 事件消融 Δ。"""
    batch = collate([loader.dataset.samples[5], loader.dataset.samples[17]])
    b = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v)
         for k, v in batch.items()}
    with torch.no_grad():
        attn = model.build_context(b["daily"], b["weekly"], b["monthly"],
                                   b["short"], b["hist_events"],
                                   b["fut_events"])[2]    # (B, 28)
    delta = model.event_ablation(b)                        # (B,)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    for i in range(2):
        axes[i].bar(np.arange(28), attn[i].cpu().numpy(),
                    color="#4C721D", alpha=0.75)
        axes[i].set_ylabel("事件注意力权重")
        axes[i].set_title(f"样本 {i+1}：事件消融 |Δ|={delta[i]:.3f}")
    axes[1].set_xlabel("未来天")
    fig.suptitle("translate-locate-imprint 事件归因（哪些天受事件影响）")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_attribution.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--runs_dir", default="",
                    help="覆盖检查点根目录（空=默认 runs）")
    ap.add_argument("--data_dir", default="",
                    help="覆盖数据目录（空=默认 data/output）")
    args = ap.parse_args()
    global RUNS, DATA_DIR
    if args.runs_dir:
        RUNS = args.runs_dir
    if args.data_dir:
        DATA_DIR = args.data_dir
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    test = QTSFDataset(DATA_DIR, "test")
    if args.smoke:
        test.samples = test.samples[:24]
    loader = DataLoader(test, batch_size=32, collate_fn=collate)

    all_results = []
    # 基线
    bl_path = os.path.join(RUNS, "baselines", "baselines.json")
    if os.path.exists(bl_path):
        with open(bl_path, encoding="utf-8") as f:
            all_results += json.load(f)
    # 主模型 + 消融
    model_results = []
    for run in ["eqtsf_full", "eqtsf_no_event", "eqtsf_no_timestep",
                "eqtsf_no_contrast", "eqtsf_no_multiscale"]:
        p = os.path.join(RUNS, run, "best.pt")
        if not os.path.exists(p):
            print(f"skip {run} (no checkpoint)")
            continue
        model = load_model(run, dev)
        m = eval_model_metrics(model, loader, dev, smoke=args.smoke)
        m["run"] = run
        model_results.append(m)
        all_results.append({"model": run, **m})
        print(run, m)
    # 预警业务指标（完整模型 vs 去事件）
    alerts = {}
    for run in ["eqtsf_full", "eqtsf_no_event"]:
        p = os.path.join(RUNS, run, "best.pt")
        if not os.path.exists(p):
            continue
        model = load_model(run, dev)
        y, s, m, meta, ev, ctxs = collect(model, loader, dev, 50, 10)
        # 阈值：每产品历史合格率均值 - 2σ（简化：全局 88.0）
        am = alert_metrics(y.reshape(len(y), -1), s, threshold=88.0)
        alerts[run] = am
        print(run, "alert:", am)
    out = {"models": model_results, "baselines":
           [r for r in all_results if "run" not in r], "alerts": alerts}
    from pathlib import Path
    Path(RUNS, "results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    # 图表（仅在未指定 --runs_dir/--data_dir 时生成，避免种子/shock 运行覆盖）
    if not args.runs_dir and not args.data_dir:
        fig_training()
        if model_results:
            fig_ablation(model_results)
            fig_baselines(all_results)
            model = load_model("eqtsf_full", dev)
            fig_forecast(model, loader, dev, DATA_DIR)
            fig_attribution(model, loader, dev)
    print("EVAL_DONE")


if __name__ == "__main__":
    main()
