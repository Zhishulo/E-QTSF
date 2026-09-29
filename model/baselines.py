# -*- coding: utf-8 -*-
"""基线模型：SeasonalNaive / DLinear / iTransformer-lite / PatchTST-lite。

统一协议：84 天上下文 → 28 天预测（与 E-QTSF 窗口一致）。
概率指标（CRPS/WQL）：点预测 + 验证集残差 std 的高斯近似分布。
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dataset import QTSFDataset, collate  # noqa: E402
from model.metrics import crps_sample, mae_mse_rmse  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "output")
RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "runs", "baselines")
CTX, FORE, C = 84, 28, 5


# ---------------- 基线网络 ----------------
class SeasonalNaive:
    """上周重复 ×4（无参数）。"""

    @staticmethod
    def predict(ctx: np.ndarray) -> np.ndarray:
        return np.tile(ctx[:, -7:, :], (1, FORE // 7, 1))


class DLinear(nn.Module):
    """序列线性 + 移动平均季节分解。"""

    def __init__(self):
        super().__init__()
        self.trend = nn.Linear(CTX, FORE)
        self.season = nn.Linear(CTX, FORE)

    def forward(self, x):                     # (B, CTX, C)
        y = x.permute(0, 2, 1)                # (B, C, CTX)
        ma = nn.functional.avg_pool1d(y, 7, 1, 0)
        ma = nn.functional.pad(ma, (0, CTX - ma.shape[2])).repeat_interleave(7, dim=2)
        ma = ma[..., :CTX]
        out = self.trend(y - ma) + self.season(ma)
        return out.permute(0, 2, 1)           # (B, FORE, C)


# 产物路径用字符串拼接构造（等价 os.path.join，保持模块内一致）


class ITransformerLite(nn.Module):
    """iTransformer 思想：通道为 token 的注意力。"""

    def __init__(self, d=64, heads=2, layers=1):
        super().__init__()
        self.inp = nn.Linear(CTX, d)
        enc = nn.TransformerEncoderLayer(
            d, heads, d * 2, 0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.out = nn.Linear(d, FORE)

    def forward(self, x):                     # (B, CTX, C)
        h = self.inp(x.permute(0, 2, 1))      # (B, C, d)
        h = self.enc(h)
        return self.out(h).permute(0, 2, 1)   # (B, FORE, C)


class PatchTSTLite(nn.Module):
    """PatchTST 思想：7 天 patch + Transformer（通道独立）。"""

    def __init__(self, d=64, heads=2, layers=1):
        super().__init__()
        self.p = 7
        self.n_patch = CTX // self.p          # 12
        self.inp = nn.Linear(self.p, d)
        half = d // 2
        freqs = torch.exp(-np.log(1000) * torch.arange(half, dtype=torch.float32)
                          / half)
        args = torch.arange(self.n_patch, dtype=torch.float32).unsqueeze(1) * freqs
        pos = torch.zeros(self.n_patch, d)
        pos[:, 0::2] = torch.sin(args)
        pos[:, 1::2] = torch.cos(args[:d - half])
        self.register_buffer("pos", pos)
        enc = nn.TransformerEncoderLayer(
            d, heads, d * 2, 0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.out = nn.Linear(self.n_patch * d, FORE)

    def forward(self, x):                     # (B, CTX, C)
        B, _, Cc = x.shape
        y = x.permute(0, 2, 1).reshape(B * Cc, self.n_patch, self.p)
        h = self.inp(y) + self.pos.unsqueeze(0)
        h = self.enc(h).reshape(B * Cc, -1)
        out = self.out(h).view(B, Cc, FORE)
        return out.permute(0, 2, 1)           # (B, FORE, C)


# ---------------- 反归一化 ----------------
def _denorm_ch0(batch, x_ch0):
    """用样本 RevIN 统计量还原通道 0（pass_rate）。

    x_ch0: (B, FORE) 或 (B, FORE, K)；mu/sd: (B, C)。
    """
    import torch
    mu = batch["mu"][:, 0].view(-1, 1, 1)
    sd = batch["sd"][:, 0].view(-1, 1, 1)
    return x_ch0 * sd + mu


# ---------------- 训练/评估 ----------------
def train_baseline(name, epochs=30, batch_size=64, lr=1e-3, smoke=False,
                   dev=None):
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2024)
    os.makedirs(RUNS_DIR, exist_ok=True)
    train = QTSFDataset(DATA_DIR, "train")
    val = QTSFDataset(DATA_DIR, "val")
    test = QTSFDataset(DATA_DIR, "test")
    if smoke:
        train.samples = train.samples[:64]
        val.samples = val.samples[:32]
        test.samples = test.samples[:32]
    tl = DataLoader(train, batch_size=batch_size, shuffle=True,
                    collate_fn=collate)
    vl = DataLoader(val, batch_size=batch_size, collate_fn=collate)
    sl = DataLoader(test, batch_size=batch_size, collate_fn=collate)

    if name == "seasonal_naive":
        return eval_baseline(name, SeasonalNaive(), sl, None)
    model = {"dlinear": DLinear, "itransformer": ITransformerLite,
             "patchtst": PatchTSTLite}[name]().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best, best_sd = float("inf"), 1.0
    for ep in range(1, epochs + 1):
        model.train()
        for batch in tl:
            x = batch["ctx"].to(dev)
            y = batch["target"].to(dev)
            pred = model(x)
            loss = nn.functional.l1_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        errs, res = [], []
        with torch.no_grad():
            for batch in vl:
                x = batch["ctx"].to(dev)
                y = batch["target"].to(dev)
                pred0 = model(x)[:, :, 0].cpu()
                pred_raw = _denorm_ch0(batch, pred0)
                y_raw = _denorm_ch0(batch, batch["target"][:, :, 0])
                e = pred_raw - y_raw
                errs.append(float(e.abs().mean()))
                res.append(e.reshape(-1).cpu().numpy())
        mae = float(np.mean(errs))
        sd = float(np.std(np.concatenate(res))) + 1e-6
        if mae < best:
            best = mae
            best_sd = sd
            torch.save(model.state_dict(),
                       RUNS_DIR + "/" + name + ".pt")
        if ep % 5 == 0 or ep == 1:
            print(f"[{name}] ep{ep} val MAE={mae:.3f}")
    model.load_state_dict(torch.load(
        RUNS_DIR + "/" + name + ".pt", map_location=dev))
    return eval_baseline(name, model, sl, best_sd)


def eval_baseline(name, model, loader, resid_sd):
    """测试评估：点预测指标 + 高斯近似 CRPS/WQL。"""
    ys, ps, ss = [], [], []
    is_nn = hasattr(model, "forward") and not isinstance(model, type)
    for batch in loader:
        y = batch["target"]
        if is_nn:
            dev = next(model.parameters()).device
            x = batch["ctx"].to(dev)
            with torch.no_grad():
                pred0 = model(x)[:, :, 0].cpu()
            pred_raw = _denorm_ch0(batch, pred0).numpy()
        else:
            pred_raw = _denorm_ch0(batch, torch.from_numpy(
                model.predict(batch["ctx"].numpy())[:, :, 0])).numpy()
        y_raw = _denorm_ch0(batch, y[:, :, 0]).numpy()
        ys.append(y_raw.reshape(-1))
        ps.append(pred_raw.reshape(-1))
        if resid_sd:
            k = 30
            ss.append(np.repeat(pred_raw.reshape(-1, 1), k, axis=1)
                      + resid_sd * np.random.randn(pred_raw.size, k))
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    mae, mse, rmse = mae_mse_rmse(y, p)
    out = {"model": name, "mae": mae, "mse": mse, "rmse": rmse}
    if resid_sd:
        s = np.concatenate(ss)
        out["crps"] = crps_sample(y, s)
    return out


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    results = []
    results.append(train_baseline("seasonal_naive", smoke=args.smoke))
    for name in ["dlinear", "itransformer", "patchtst"]:
        print(f"=== baseline {name} ===")
        results.append(train_baseline(name, epochs=args.epochs,
                                      smoke=args.smoke))
    os.makedirs(RUNS_DIR, exist_ok=True)
    out_path = RUNS_DIR + "/baselines.json"
    from pathlib import Path
    Path(out_path).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BASELINES_DONE")
    for r in results:
        print(r)
