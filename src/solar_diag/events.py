"""云跳变（ramp）事件检测与晴空包络：v1.2 共用工具。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def detect_ramps(
    dts: pd.Series | np.ndarray,
    power: np.ndarray,
    capacity: float,
    frac: float = 0.2,
) -> pd.DataFrame:
    """相邻 15min 槽功率跳变检测。

    只在相邻时间戳恰为 15min 的相邻样本上计算 ΔP；|ΔP| ≥ frac·capacity
    判为一次 ramp，事件归属到**后一个**槽。返回事件表（按 |ΔP| 降序）。
    """
    dt = pd.Series(pd.to_datetime(np.asarray(dts))).reset_index(drop=True)
    p = np.asarray(power, dtype=np.float64)
    d = dt.diff().dt.total_seconds().to_numpy()
    dp = np.diff(p)
    ok = np.isclose(d[1:], 900.0)  # diff 对齐到后一个样本
    thr = frac * capacity
    hit = ok & (np.abs(dp) >= thr)
    idx = np.where(hit)[0] + 1  # 事件槽 = 后一个样本的索引
    ev = pd.DataFrame({
        "index": idx,
        "dt": dt.iloc[idx].to_numpy(),
        "delta_w": dp[hit],
        "abs_delta_w": np.abs(dp[hit]),
    })
    return ev.sort_values("abs_delta_w", ascending=False).reset_index(drop=True)


def clear_sky_envelope(
    dts: pd.Series | np.ndarray,
    irr: np.ndarray,
    smooth: int = 3,
) -> pd.Series:
    """逐 15min 槽的晴空辐照包络：训练期各槽 95 分位 + 邻槽中值平滑。

    返回以「槽序号 0..95」为索引的 Series（W/m²）。
    """
    dt = pd.Series(pd.to_datetime(np.asarray(dts))).reset_index(drop=True)
    g = pd.DataFrame({
        "slot": dt.dt.hour * 4 + dt.dt.minute // 15,
        "irr": np.asarray(irr, dtype=np.float64),
    })
    env = g.groupby("slot")["irr"].quantile(0.95)
    env = env.reindex(range(96))
    env = env.interpolate(limit_direction="both")
    if smooth and smooth > 1:
        env = env.rolling(smooth, center=True, min_periods=1).median()
    return env


def clearness_index(
    dts: pd.Series | np.ndarray,
    irr: np.ndarray,
    envelope: pd.Series,
    cap: float = 1.2,
) -> np.ndarray:
    """晴空指数 k_t = irr / 包络(槽)；包络近零时置 0。"""
    dt = pd.Series(pd.to_datetime(np.asarray(dts))).reset_index(drop=True)
    slot = (dt.dt.hour * 4 + dt.dt.minute // 15).to_numpy()
    env = np.asarray(envelope.reindex(slot).to_numpy(), dtype=np.float64)
    irr_a = np.asarray(irr, dtype=np.float64)
    kt = np.where(env > 1e-6, irr_a / np.maximum(env, 1e-6), 0.0)
    return np.clip(kt, 0.0, cap)
