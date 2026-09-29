# -*- coding: utf-8 -*-
"""E-QTSF 训练脚本（含消融开关）。

用法：
  python train.py --epochs 40                    # 完整模型
  python train.py --ablation no_event            # 消融：去事件条件化
  python train.py --ablation no_timestep         # 消融：去事件感知时间步
  python train.py --ablation no_contrast         # 消融：去对比学习
  python train.py --ablation no_multiscale       # 消融：去多尺度编码
  python train.py --smoke                        # 冒烟测试（CPU 2 epoch）
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dataset import QTSFDataset, collate  # noqa: E402
from model.metrics import crps_sample, mae_mse_rmse  # noqa: E402
from model.model import EQTSF  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "output")
RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "runs")

ABLATIONS = {
    "none": dict(use_events=True, use_event_timestep=True,
                 use_contrastive=True, use_multiscale=True),
    "no_event": dict(use_events=False, use_event_timestep=False,
                     use_contrastive=True, use_multiscale=True),
    "no_timestep": dict(use_events=True, use_event_timestep=False,
                        use_contrastive=True, use_multiscale=True),
    "no_contrast": dict(use_events=True, use_event_timestep=True,
                        use_contrastive=False, use_multiscale=True),
    "no_multiscale": dict(use_events=True, use_event_timestep=True,
                          use_contrastive=True, use_multiscale=False),
}


def make_loaders(batch_size, smoke=False, data_dir=None, stride=3):
    data_dir = data_dir or DATA_DIR
    train = QTSFDataset(data_dir, "train", stride=stride)  # 密集滑窗抗过拟合
    val = QTSFDataset(data_dir, "val")
    test = QTSFDataset(data_dir, "test")
    if smoke:
        train.samples = train.samples[:64]
        val.samples = val.samples[:32]
        test.samples = test.samples[:32]
    return (DataLoader(train, batch_size=batch_size, shuffle=True,
                       collate_fn=collate, num_workers=0),
            DataLoader(val, batch_size=batch_size, shuffle=False,
                       collate_fn=collate, num_workers=0),
            DataLoader(test, batch_size=batch_size, shuffle=False,
                       collate_fn=collate, num_workers=0))


def to_device(batch, dev):
    return {k: (v.to(dev) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


@torch.no_grad()
def evaluate_model(model, loader, dev, n_samples=30, n_steps=10):
    """验证/测试：CRPS + MAE（目标通道 pass_rate）。"""
    model.eval()
    ys, ss, ms = [], [], []
    for batch in loader:
        batch = to_device(batch, dev)
        samp = model.sample(batch, n_samples, n_steps)      # (B,28,C,K)
        # 还原归一化（通道 0,1 被 RevIN）
        mu = batch["mu"].unsqueeze(1).unsqueeze(-1)         # (B,1,1,K? ) no
        mu = batch["mu"][:, 0].view(-1, 1, 1)               # pass_rate mu
        sd = batch["sd"][:, 0].view(-1, 1, 1)
        samp_pr = samp[:, :, 0] * sd + mu                   # (B,28,K)
        tgt_pr = batch["target"][:, :, 0] * sd.squeeze(-1) + mu.squeeze(-1)
        med = samp_pr.median(dim=-1).values
        ys.append(tgt_pr.cpu().numpy().reshape(-1))
        ss.append(samp_pr.cpu().numpy().reshape(-1, samp_pr.shape[-1]))
        ms.append(med.cpu().numpy().reshape(-1))
    y = np.concatenate(ys)
    s = np.concatenate(ss)
    m = np.concatenate(ms)
    crps = crps_sample(y, s)
    mae, mse, rmse = mae_mse_rmse(y, m)
    return {"crps": crps, "mae": mae, "mse": mse, "rmse": rmse}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", default="none", choices=list(ABLATIONS))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d_model", type=int, default=96)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--data_dir", default=None, help="数据目录覆盖（如 shock 数据集）")
    ap.add_argument("--run_dir", default=None, help="训练产物目录覆盖")
    ap.add_argument("--stride", type=int, default=3,
                    help="训练滑窗步长（数据量大时用 7）")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    flags = ABLATIONS[args.ablation]
    run_name = (f"eqtsf_{args.ablation}" if args.ablation != "none"
                else "eqtsf_full")
    run_dir = args.run_dir or os.path.join(RUNS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    train_loader, val_loader, test_loader = make_loaders(
        args.batch_size, smoke=args.smoke, data_dir=args.data_dir,
        stride=args.stride)
    model = EQTSF(c_in=5, d_model=args.d_model, layers=args.layers,
                  dropout=args.dropout, **flags).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.epochs
    warmup = max(2, total // 10)
    log = {"run": run_name, "flags": flags, "device": dev,
           "params": n_params, "epochs": []}
    print(f"[{run_name}] device={dev} params={n_params:,} "
          f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
          f"test={len(test_loader.dataset)}", flush=True)
    best_crps = float("inf")
    t0 = time.time()
    for ep in range(1, total + 1):
        model.train()
        lr = args.lr * min(ep / warmup, 1.0) * (
            0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(total - warmup, 1)))
            if ep > warmup else 1.0)
        for g in opt.param_groups:
            g["lr"] = lr
        sums = {"loss_fm": 0.0, "loss_con": 0.0, "loss_sim": 0.0}
        nb = 0
        for batch in train_loader:
            batch = to_device(batch, dev)
            out = model(batch, epoch=ep, total_epochs=total)
            loss = (out["loss_fm"]
                    + 0.15 * out.get("loss_con", 0)
                    + 0.05 * out.get("loss_sim", 0))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k in sums:
                sums[k] += float(out[k])
            nb += 1
        msg = {k: v / nb for k, v in sums.items()}
        if ep % 4 == 0 or ep == total or args.smoke:
            vm = evaluate_model(model, val_loader, dev)
            log["epochs"].append({"epoch": ep, **msg, **vm})
            star = ""
            if vm["crps"] < best_crps:
                best_crps = vm["crps"]
                torch.save({"model": model.state_dict(), "flags": flags,
                            "d_model": args.d_model, "layers": args.layers},
                           os.path.join(run_dir, "best.pt"))
                star = " *"
            print(f"ep{ep:03d} fm={msg['loss_fm']:.4f} con={msg['loss_con']:.4f} "
                  f"| val CRPS={vm['crps']:.4f} MAE={vm['mae']:.3f}{star}",
                  flush=True)
        else:
            log["epochs"].append({"epoch": ep, **msg})
    # 测试
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location=dev)
    model.load_state_dict(ck["model"])
    tm = evaluate_model(model, test_loader, dev, n_samples=50)
    tm["train_minutes"] = (time.time() - t0) / 60
    log["test"] = tm
    from pathlib import Path
    Path(run_dir, "log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{run_name}] TEST CRPS={tm['crps']:.4f} MAE={tm['mae']:.3f} "
          f"({tm['train_minutes']:.1f} min)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
