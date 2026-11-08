"""误差指标与分桶评估。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true, y_pred) -> float:
    yt, yp = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true, y_pred) -> float:
    yt, yp = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(yt - yp)))


def r2(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def nrmse_pct(y_true, y_pred, capacity: float) -> float:
    """RMSE / 额定容量（%）。"""
    return 100.0 * rmse(y_true, y_pred) / capacity


def metric_pack(y_true, y_pred, capacity: float) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "nrmse_pct": nrmse_pct(y_true, y_pred, capacity),
        "n": int(len(y_true)),
    }


def bucket_by_irradiation(irr: pd.Series | np.ndarray, bins: list[float]) -> pd.Series:
    """按目标时刻辐照分桶（天气条件代理）。"""
    irr = pd.Series(np.asarray(irr, dtype=np.float64))
    labels = [f"[{bins[i]:g},{bins[i+1]:g})" for i in range(len(bins) - 1)]
    return pd.cut(irr, bins=bins, labels=labels, right=False, include_lowest=True)


def bucket_by_daypart(hour: pd.Series | np.ndarray) -> pd.Series:
    """按目标时刻小时分桶：清晨/上午/正午/下午/傍晚/夜间。"""
    h = pd.Series(np.asarray(hour))
    cuts = [0, 6, 9, 12, 15, 18, 24]
    labels = ["night(0-6)", "early(6-9)", "morning(9-12)", "noon(12-15)", "afternoon(15-18)", "evening(18-24)"]
    return pd.cut(h, bins=cuts, labels=labels, right=False, include_lowest=True)


def grouped_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    groups: pd.Series,
    capacity: float,
) -> pd.DataFrame:
    """按组输出 RMSE/MAE/nRMSE/样本数。"""
    yt = pd.Series(np.asarray(y_true, dtype=np.float64))
    yp = pd.Series(np.asarray(y_pred, dtype=np.float64))
    rows = []
    for g, idx in groups.groupby(groups, observed=True).groups.items():
        rows.append({"bucket": str(g), **metric_pack(yt.loc[idx], yp.loc[idx], capacity)})
    return pd.DataFrame(rows).sort_values("bucket").reset_index(drop=True)


# ---------- 分位数 / 区间指标 ----------

def pinball_loss(y_true, y_pred, q: float) -> float:
    """分位数 q 的 pinball（quantile）损失。"""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    d = yt - yp
    return float(np.mean(np.maximum(q * d, (q - 1.0) * d)))


def interval_coverage(y_true, lo, hi) -> float:
    """经验覆盖率：y ∈ [lo, hi] 的比例。"""
    yt = np.asarray(y_true, dtype=np.float64)
    lo_a = np.asarray(lo, dtype=np.float64)
    hi_a = np.asarray(hi, dtype=np.float64)
    return float(np.mean((yt >= lo_a) & (yt <= hi_a)))


def interval_width(lo, hi) -> float:
    """平均区间宽度。"""
    return float(np.mean(np.asarray(hi, dtype=np.float64) - np.asarray(lo, dtype=np.float64)))
