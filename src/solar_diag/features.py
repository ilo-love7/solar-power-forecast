"""特征工程：周期时间编码、功率滞后、天气特征、监督样本框。"""
from __future__ import annotations

import numpy as np
import pandas as pd

TIME_NAMES = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
NOWCAST_FEATS = TIME_NAMES + ["ac_now", "ac_lag1", "ac_lag4", "ac_lag8", "ac_lag96", "ac_roll_mean4"]
WEATHER_FEATS = NOWCAST_FEATS + [
    "irradiation", "ambient_temp", "module_temp", "temp_diff", "irr_lag1", "irr_lag4",
]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """周期时间编码：小时与年内日的 sin/cos。"""
    out = df.copy()
    dt = out["dt"]
    minute_of_day = dt.dt.hour + dt.dt.minute / 60.0
    doy = dt.dt.day_of_year + minute_of_day / 24.0
    out["hour_sin"] = np.sin(2 * np.pi * minute_of_day / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * minute_of_day / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """功率/辐照滞后与滚动特征（按时间排序后 shift，禁止打乱）。"""
    out = df.sort_values("dt").reset_index(drop=True).copy()
    out["ac_now"] = out["ac_power"]  # t 时刻功率已知，可作特征（持久性基线用）
    for k in (1, 4, 8, 96):
        out[f"ac_lag{k}"] = out["ac_power"].shift(k)
    out["ac_roll_mean4"] = out["ac_power"].shift(1).rolling(4, min_periods=1).mean()
    out["irr_lag1"] = out["irradiation"].shift(1)
    out["irr_lag4"] = out["irradiation"].shift(4)
    out["temp_diff"] = out["module_temp"] - out["ambient_temp"]
    return out


def build_supervised(
    df: pd.DataFrame,
    horizon: int = 4,
    regime: str = "nowcast",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """构造 X, y 与元信息（dt / 目标时刻天气桶所需列）。

    regime='nowcast'：时间编码 + 功率滞后（不依赖天气预报）；
    regime='weather'：在 nowcast 基础上加 t 时刻天气（近似天气预报情形）。
    y = ac_power(t+horizon)，特征只用 t 及以前 → 无目标泄漏。
    """
    if regime not in ("nowcast", "weather"):
        raise ValueError(f"未知 regime: {regime}")
    d = add_time_features(df)
    d = add_lag_features(d)
    feat_names = NOWCAST_FEATS if regime == "nowcast" else WEATHER_FEATS
    d["target"] = d["ac_power"].shift(-horizon)
    # 元信息取目标时刻天气（仅用于误差归因，不进特征）
    d["target_dt"] = d["dt"].shift(-horizon)
    d["target_irr"] = d["irradiation"].shift(-horizon)
    d["target_hour"] = d["target_dt"].dt.hour
    valid = d.dropna(subset=["target"] + feat_names).reset_index(drop=True)
    x = valid[feat_names]
    y = valid["target"]
    meta = valid[["dt", "target_dt", "target_irr", "target_hour"]].copy()
    return x, y, meta


def sequence_arrays(
    x: pd.DataFrame,
    y: pd.Series,
    history: int,
) -> tuple[np.ndarray, np.ndarray]:
    """把监督样本折成 LSTM 用的 (n, history, f) 序列。

    约定：调用前 df 已按时间排序且行间等间隔（缺槽已剔除，间隔处序列视为连续窗口，
    在缺槽密集时会缩短有效样本——数据完整率 >97%，影响可忽略）。
    """
    xa = x.to_numpy(dtype=np.float32)
    ya = y.to_numpy(dtype=np.float32)
    if len(xa) < history:
        raise ValueError(f"样本数 {len(xa)} < history {history}")
    n = len(xa) - history + 1
    xs = np.stack([xa[i:i + history] for i in range(n)])
    ys = ya[history - 1:]
    return xs, ys


def split_supervised_by_day(
    x: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    train_frac: float,
    val_frac: float,
) -> dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]]:
    """按 target_dt 所在天切 train/val/test（特征已只用 t 及以前，按目标时刻切即无泄漏）。"""
    days = np.array(sorted(meta["target_dt"].dt.normalize().unique()))
    n = len(days)
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac)))
    if n_train + n_val >= n:
        raise ValueError(f"天数过少无法切分: n={n}")
    bounds = {
        "train": (days[0], days[n_train - 1]),
        "val": (days[n_train], days[n_train + n_val - 1]),
        "test": (days[n_train + n_val], days[-1]),
    }
    out = {}
    tdays = meta["target_dt"].dt.normalize()
    for k, (lo, hi) in bounds.items():
        mask = (tdays >= lo) & (tdays <= hi)
        out[k] = (x.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True), meta.loc[mask].reset_index(drop=True))
    return out
