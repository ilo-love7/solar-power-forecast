"""pinball + 共形二段缩放：把"选分位对"与"校准因子"解耦。

v1.1/1.2 的 CQR 是加成式校准（lo−q̂, hi+q̂），本数据集 q̂=0 是空操作；
且加法偏移对所有样本一视同仁，无法利用分位数模型自带的异方差宽度
（§3.8 显示云跳变处 raw 区间已自动加宽到平稳段 2.2–2.8 倍）。

本模块的两段：
  第一段（pinball）：在若干候选分位对中，用验证集 pinball 锐度挑选最锐的一对；
  第二段（共形）：用归一化残差 r=(y−c)/h 的共形分位数求乘性因子 λ，
                 区间 = c ± λ·h，保留逐点宽度结构。

对照口径（脚本层组装）：raw / add(CQR) / mult(全局 λ) / mult(分桶 λ) /
two_stage(pinball 选对 + λ) / two_stage_group(选对 + 分桶 λ)。

所有函数为纯函数，输入 numpy 数组。
"""
from __future__ import annotations

import numpy as np

from .conformal import conformal_quantile, cqr_scores
from .evaluation import pinball_loss


def normalized_residuals(y: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                         eps: float = 1e-8) -> np.ndarray:
    """逐点归一化残差 r=(y−c)/h_side：c 为区间中点，按 y 落侧取对应半宽。"""
    y = np.asarray(y, dtype=np.float64).ravel()
    lo = np.asarray(lo, dtype=np.float64).ravel()
    hi = np.asarray(hi, dtype=np.float64).ravel()
    c = (lo + hi) / 2.0
    h = np.where(y >= c, hi - c, c - lo)
    return (y - c) / np.maximum(h, eps)


def scale_factor(y_cal: np.ndarray, lo_cal: np.ndarray, hi_cal: np.ndarray,
                 nominal: float, mask: np.ndarray | None = None) -> float:
    """全局/分组乘性因子 λ = 共形分位数(|r|)。λ<1 表示原始区间偏保守需收缩。"""
    r = normalized_residuals(y_cal, lo_cal, hi_cal)
    if mask is not None:
        r = r[np.asarray(mask, dtype=bool)]
    return conformal_quantile(np.abs(r), nominal)


def group_scale_factors(y_cal: np.ndarray, lo_cal: np.ndarray, hi_cal: np.ndarray,
                        groups_cal: np.ndarray, nominal: float,
                        n_min: int = 30) -> dict[str, float]:
    """分桶 λ：小样本桶回退全局 λ（呼应 smoothing.min_n 的最小样本量约束）。"""
    r = np.abs(normalized_residuals(y_cal, lo_cal, hi_cal))
    g = np.asarray(groups_cal).astype(str).ravel()
    lam_global = conformal_quantile(r, nominal)
    out: dict[str, float] = {}
    for name in np.unique(g):
        m = g == name
        out[str(name)] = (conformal_quantile(r[m], nominal)
                          if int(m.sum()) >= n_min else lam_global)
    return out


def apply_scale(lo: np.ndarray, hi: np.ndarray, lam: float,
                groups: np.ndarray | None = None,
                lam_map: dict[str, float] | None = None,
                eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """按 λ（全局标量或分组查表）围绕中点缩放区间。

    注：半宽用 `max(..., eps)` 防除零，会把独立训练的 q_lo>q_hi（分位数交叉）
    样本的负宽度夹到 0；λ=1 时覆盖率不变，但平均宽度会有极小上移。
    """
    lo = np.asarray(lo, dtype=np.float64).ravel()
    hi = np.asarray(hi, dtype=np.float64).ravel()
    if lam_map is not None:
        if groups is None:
            raise ValueError("分组 λ 需要 groups")
        g = np.asarray(groups).astype(str).ravel()
        fallback = float(np.mean(list(lam_map.values()))) if lam_map else float(lam)
        lam_vec = np.array([lam_map.get(k, fallback) for k in g], dtype=np.float64)
    else:
        lam_vec = np.full(lo.size, float(lam), dtype=np.float64)
    c = (lo + hi) / 2.0
    h_lo = np.maximum(c - lo, eps)
    h_hi = np.maximum(hi - c, eps)
    return c - lam_vec * h_lo, c + lam_vec * h_hi


def pinball_sharpness(y: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                      q_lo: float, q_hi: float) -> float:
    """分位对的 pinball 锐度（越小越好）：两个分位头的损失之和。"""
    return pinball_loss(y, lo, q_lo) + pinball_loss(y, hi, q_hi)


def select_quantile_pair(candidates: dict[str, tuple[np.ndarray, np.ndarray, float, float]],
                         y_cal: np.ndarray, nominal: float,
                         require_coverage: bool = False) -> tuple[str, dict]:
    """第一段：按验证集 pinball 锐度选最锐的分位对。

    candidates: {名称: (lo_cal, hi_cal, q_lo, q_hi)}
    require_coverage=True 时只在原始覆盖 ≥ nominal 的候选里选（更保守）。
    返回 (选中名, {候选: {sharpness, coverage}})。
    """
    y = np.asarray(y_cal, dtype=np.float64).ravel()
    table: dict[str, dict] = {}
    for name, (lo, hi, q_lo, q_hi) in candidates.items():
        lo = np.asarray(lo, dtype=np.float64).ravel()
        hi = np.asarray(hi, dtype=np.float64).ravel()
        cov = float(np.mean((y >= lo) & (y <= hi)))
        table[name] = {"sharpness": pinball_sharpness(y, lo, hi, q_lo, q_hi),
                       "coverage": cov}
    pool = [k for k, v in table.items() if v["coverage"] >= nominal] if require_coverage else list(table)
    if not pool:
        pool = list(table)
    best = min(pool, key=lambda k: table[k]["sharpness"])
    return best, table


def two_stage_interval(
    y_cal: np.ndarray,
    candidates: dict[str, tuple[np.ndarray, np.ndarray, float, float]],
    lo_te_by_name: dict[str, np.ndarray],
    hi_te_by_name: dict[str, np.ndarray],
    nominal: float,
    groups_cal: np.ndarray | None = None,
    groups_te: np.ndarray | None = None,
    grouped: bool = False,
    require_coverage: bool = False,
    n_min: int = 30,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """二段式完整流程：pinball 选分位对 → 共形乘性缩放。

    返回 (lo, hi, params)；params 含选中对、pinball 表、λ（或分组 λ）。
    """
    name, table = select_quantile_pair(candidates, y_cal, nominal, require_coverage)
    lo_cal, hi_cal, _, _ = candidates[name]
    lo_te = np.asarray(lo_te_by_name[name], dtype=np.float64)
    hi_te = np.asarray(hi_te_by_name[name], dtype=np.float64)
    params: dict = {"selected": name, "pinball_table": table}
    if grouped:
        if groups_cal is None or groups_te is None:
            raise ValueError("grouped=True 需要 groups_cal / groups_te")
        lam_map = group_scale_factors(y_cal, lo_cal, hi_cal, groups_cal, nominal, n_min)
        lo_out, hi_out = apply_scale(lo_te, hi_te, 1.0, groups_te, lam_map)
        params["lambda_map"] = {k: float(v) for k, v in lam_map.items()}
        params["lambda"] = float(np.mean(list(lam_map.values()))) if lam_map else 1.0
    else:
        lam = scale_factor(y_cal, lo_cal, hi_cal, nominal)
        lo_out, hi_out = apply_scale(lo_te, hi_te, lam)
        params["lambda"] = lam
    return lo_out, hi_out, params


def cqr_add_interval(y_cal: np.ndarray, lo_cal: np.ndarray, hi_cal: np.ndarray,
                     lo_te: np.ndarray, hi_te: np.ndarray,
                     nominal: float) -> tuple[np.ndarray, np.ndarray, float]:
    """对照：加成式 CQR（v1.1/1.2 口径）。"""
    q = conformal_quantile(cqr_scores(y_cal, lo_cal, hi_cal), nominal)
    return (np.asarray(lo_te, dtype=np.float64) - q,
            np.asarray(hi_te, dtype=np.float64) + q, q)
