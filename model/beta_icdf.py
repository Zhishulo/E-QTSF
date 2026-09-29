# -*- coding: utf-8 -*-
"""可微 Beta 分布 CDF / ICDF（PyTorch 未实现 icdf，自行实现）。

- betainc_reg: 正则化不完全 Beta 函数 I_x(a,b)，连分式（Lentz）实现，
  全部由可微原语构成 → 对 a, b, x 均可反传。
- beta_icdf:   logit 空间牛顿迭代求分位数（数值条件良好，α<1 时稳定）。
"""
import torch

_TINY = 1e-30
_LO = -9.2   # logit(1e-4)
_HI = 9.2    # logit(1-1e-4)


def _betacf(a: torch.Tensor, b: torch.Tensor, x: torch.Tensor,
            iters: int = 40) -> torch.Tensor:
    """连分式 betacf（Numerical Recipes 6.4），向量化 + 可微。"""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = torch.ones_like(x)
    d = 1.0 - qab * x / qap
    d = torch.where(d.abs() < _TINY, torch.full_like(d, _TINY), d)
    d = 1.0 / d
    h = d
    for m in range(1, iters + 1):
        m2 = 2.0 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = torch.where(d.abs() < _TINY, torch.full_like(d, _TINY), d)
        d = 1.0 / d
        c = 1.0 + aa / c
        c = torch.where(c.abs() < _TINY, torch.full_like(c, _TINY), c)
        h = h * d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = torch.where(d.abs() < _TINY, torch.full_like(d, _TINY), d)
        d = 1.0 / d
        c = 1.0 + aa / c
        c = torch.where(c.abs() < _TINY, torch.full_like(c, _TINY), c)
        h = h * d * c
    return h


def betainc_reg(a: torch.Tensor, b: torch.Tensor, x: torch.Tensor,
                iters: int = 80) -> torch.Tensor:
    """正则化不完全 Beta I_x(a,b)（连续、可微）。"""
    x = x.clamp(1e-6, 1 - 1e-6)
    # ln B(a,b) = lgamma(a) + lgamma(b) - lgamma(a+b)
    log_beta_fn = (torch.lgamma(a) + torch.lgamma(b)
                   - torch.lgamma(a + b))
    front = torch.exp(a * torch.log(x) + b * torch.log1p(-x) - log_beta_fn)
    # 对称技巧：x > (a+1)/(a+b+2) 时用 1 - I_{1-x}(b,a)
    swap = x > (a + 1.0) / (a + b + 2.0)
    xs = torch.where(swap, 1.0 - x, x)
    aa = torch.where(swap, b, a)
    bb = torch.where(swap, a, b)
    cf = _betacf(aa, bb, xs, iters)
    # NR: I = front·betacf(a,b,x)/a；swap 分支 I = 1 − front·betacf(b,a,1−x)/b
    # 两个分支都除以 betacf 的第一个参数（aa）
    i_lo = front * cf / aa
    i_hi = 1.0 - front * cf / aa
    return torch.where(swap, i_hi, i_lo).clamp(0.0, 1.0)


def beta_pdf(a: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    log_beta_fn = (torch.lgamma(a + b) - torch.lgamma(a) - torch.lgamma(b))
    return torch.exp((a - 1) * torch.log(x.clamp(1e-6, 1 - 1e-6))
                     + (b - 1) * torch.log1p(-(x.clamp(1e-6, 1 - 1e-6)))
                     - log_beta_fn)


def beta_icdf(t: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
              newton_steps: int = 6) -> torch.Tensor:
    """Beta 分位数函数（logit 空间牛顿迭代，对 a/b/x 可微）。

    t: (N,) 均匀样本; a, b: (N,) Beta 参数（>0）。
    """
    t = t.clamp(1e-4, 1 - 1e-4)
    a = a.clamp_min(1e-3)
    b = b.clamp_min(1e-3)
    # 初始化：logit(均值)，尾部分位数时向对应侧偏置
    u0 = torch.log(a / b)
    u = u0.clamp(_LO, _HI)
    for _ in range(newton_steps):
        x = torch.sigmoid(u)
        Fx = betainc_reg(a, b, x)
        # logit 空间导数：pdf(x)·x(1-x) = exp(a ln x + b ln(1-x) - lnB)
        log_beta_fn = (torch.lgamma(a) + torch.lgamma(b)
                       - torch.lgamma(a + b))
        dF_du = torch.exp(a * torch.log(x) + b * torch.log1p(-x)
                          - log_beta_fn).clamp_min(1e-8)
        u = u - (Fx - t) / dF_du
        u = u.clamp(_LO, _HI)
    return torch.sigmoid(u)
