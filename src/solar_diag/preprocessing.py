"""对齐、清洗与按时间切分。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io import PlantData

FREQ = "15min"


def plant_level(plant: PlantData) -> pd.DataFrame:
    """逆变器级求和 → 站级 15min 序列，并合并站级气象（外连接后前向填气象）。

    返回列：dt, dc_power, ac_power, ambient_temp, module_temp, irradiation, n_inv。
    """
    g = plant.gen.groupby("dt").agg(
        dc_power=("DC_POWER", "sum"),
        ac_power=("AC_POWER", "sum"),
        n_inv=("SOURCE_KEY", "nunique"),
    )
    w = plant.weather.set_index("dt")[
        ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
    ].rename(columns={
        "AMBIENT_TEMPERATURE": "ambient_temp",
        "MODULE_TEMPERATURE": "module_temp",
        "IRRADIATION": "irradiation",
    })
    df = g.join(w, how="outer").sort_index()
    df.index.name = "dt"
    # 气象少量缺槽：前向填充；功率缺槽保持 NaN（后续丢弃）
    df[["ambient_temp", "module_temp", "irradiation"]] = df[
        ["ambient_temp", "module_temp", "irradiation"]
    ].ffill()
    return df.reset_index()


def inverter_level(plant: PlantData) -> pd.DataFrame:
    """逆变器级长表（诊断用），带 dt 与基础列。"""
    df = plant.gen[["dt", "SOURCE_KEY", "DC_POWER", "AC_POWER", "DAILY_YIELD", "TOTAL_YIELD"]].copy()
    return df.sort_values(["SOURCE_KEY", "dt"]).reset_index(drop=True)


def drop_missing_power(df: pd.DataFrame) -> pd.DataFrame:
    """去掉功率缺失的时间片（气象可缺、功率必须有）。"""
    return df.dropna(subset=["ac_power", "dc_power"]).reset_index(drop=True)


def split_days(df: pd.DataFrame, train_frac: float, val_frac: float) -> dict[str, pd.DataFrame]:
    """按时间顺序切天（train/val/test），返回按 target 时间过滤的三个子集。

    防泄漏：以天为单位整体划分，同一天不跨集合；
    特征里的历史 lags 允许来自更早集合（真实预测场景如此）。
    """
    days = np.array(sorted(df["dt"].dt.normalize().unique()))
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
    for k, (lo, hi) in bounds.items():
        mask = (df["dt"].dt.normalize() >= lo) & (df["dt"].dt.normalize() <= hi)
        out[k] = df.loc[mask].reset_index(drop=True)
    return out


def capacity_of(df: pd.DataFrame, q: float = 0.99) -> float:
    """额定容量代理：站级 AC 功率 q 分位数（W）。"""
    return float(df["ac_power"].quantile(q))
