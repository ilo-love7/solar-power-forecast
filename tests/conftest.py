import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_plant_df():
    """7 天 × 96 槽的合成站级数据（含昼夜与噪声）。"""
    rng = np.random.default_rng(0)
    dt = pd.date_range("2020-05-15 00:00", periods=7 * 96, freq="15min")
    hour = np.asarray(dt.hour + dt.minute / 60.0)
    irr = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None) * (0.8 + 0.2 * rng.random(len(dt)))
    irr[~((hour >= 6) & (hour <= 18))] = 0.0
    ac = 30000 * irr + rng.normal(0, 100, len(dt))
    ac = np.clip(ac, 0, None)
    df = pd.DataFrame({
        "dt": dt,
        "dc_power": ac / 0.97,
        "ac_power": ac,
        "ambient_temp": 25 + 5 * irr,
        "module_temp": 30 + 15 * irr,
        "irradiation": irr,
        "n_inv": 22,
    })
    return df


@pytest.fixture
def tiny_gen_inv():
    """2 台正常 + 1 台故障逆变器（2 天）。

    invBad：第一天 10 点后「行在但功率 0」（转换失效→停机事件），
           第二天 0-6 点整行缺失（通讯中断→完整率下降）。
    """
    dt = pd.date_range("2020-05-15 00:00", periods=2 * 96, freq="15min")
    rows = []
    for inv, cap, bad in [("invA", 1000.0, False), ("invB", 1000.0, False), ("invBad", 1000.0, True)]:
        for t in dt:
            h = t.hour + t.minute / 60.0
            if bad and t.day == 16 and h < 6:
                continue  # 第二天凌晨通讯中断：整行缺失
            dc = max(0.0, cap * np.sin(np.pi * (h - 6) / 12)) if 6 <= h <= 18 else 0.0
            if bad and h > 10 and t.day == 15:
                dc = 0.0  # 第一天午后转换失效：行在、功率 0
            rows.append({"dt": t, "SOURCE_KEY": inv, "DC_POWER": dc, "AC_POWER": dc * 0.975,
                         "DAILY_YIELD": 0.0, "TOTAL_YIELD": 0.0})
    return pd.DataFrame(rows)
