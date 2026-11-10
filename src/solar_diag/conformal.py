"""保形预测（conformal prediction）纯函数：全局 CQR 与分组（Mondrian）校准。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cqr_scores(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """CQR 非一致性分数：>0 表示真值出界，≤0 表示在界内（越负区间越富余）。"""
    y = np.asarray(y, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    return np.maximum(lo - y, y - hi)


def conformal_quantile(scores: np.ndarray, nominal: float) -> float:
    """有限样本修正的分位数：level = ceil((n+1)·nominal)/n。"""
    s = np.asarray(scores, dtype=np.float64)
    n = len(s)
    if n == 0:
        return 0.0
    level = min(1.0, np.ceil((n + 1) * nominal) / n)
    return float(np.quantile(s, level))


def group_quantiles(
    y: np.ndarray, lo: np.ndarray, hi: np.ndarray,
    groups: pd.Series | np.ndarray, nominal: float,
) -> dict[str, float]:
    """Mondrian 分组校准：每组独立取 q̂。返回 {组名: q̂}。"""
    g = pd.Series(np.asarray(groups)).astype(str).reset_index(drop=True)
    scores = cqr_scores(y, lo, hi)
    out: dict[str, float] = {}
    for name, idx in g.groupby(g).groups.items():
        out[str(name)] = conformal_quantile(scores[np.asarray(list(idx))], nominal)
    return out


def apply_offsets(
    lo: np.ndarray, hi: np.ndarray,
    groups: pd.Series | np.ndarray, offsets: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """按组施加对称偏移：lo−q̂, hi+q̂（q̂<0 时收缩）。"""
    lo_a = np.asarray(lo, dtype=np.float64).copy()
    hi_a = np.asarray(hi, dtype=np.float64).copy()
    g = pd.Series(np.asarray(groups)).astype(str).reset_index(drop=True)
    for name, q in offsets.items():
        m = (g == str(name)).to_numpy()
        lo_a[m] = lo_a[m] - q
        hi_a[m] = hi_a[m] + q
    return lo_a, hi_a


def group_coverage(
    y: np.ndarray, lo: np.ndarray, hi: np.ndarray,
    groups: pd.Series | np.ndarray,
) -> dict[str, dict]:
    """按组统计覆盖率与平均宽度。"""
    y = np.asarray(y, dtype=np.float64)
    lo_a = np.asarray(lo, dtype=np.float64)
    hi_a = np.asarray(hi, dtype=np.float64)
    g = pd.Series(np.asarray(groups)).astype(str).reset_index(drop=True)
    out: dict[str, dict] = {}
    for name, idx in g.groupby(g).groups.items():
        i = np.asarray(list(idx))
        cov = float(np.mean((y[i] >= lo_a[i]) & (y[i] <= hi_a[i])))
        out[str(name)] = {"coverage": cov, "n": int(len(i)),
                          "width": float(np.mean(hi_a[i] - lo_a[i]))}
    return out


def coverage_spread(by_group: dict[str, dict]) -> float:
    """组间覆盖极差（max−min）：均匀化目标是把它压到 0。"""
    covs = [v["coverage"] for v in by_group.values()]
    return float(max(covs) - min(covs)) if covs else 0.0
