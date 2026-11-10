"""v1.2 纯函数单测：Mondrian 共形、ramp 检测、晴空包络。"""
import numpy as np
import pandas as pd

from solar_diag.conformal import (
    apply_offsets,
    conformal_quantile,
    coverage_spread,
    cqr_scores,
    group_coverage,
    group_quantiles,
)
from solar_diag.events import clear_sky_envelope, clearness_index, detect_ramps


# ---------- conformal ----------

def test_cqr_scores_sign():
    y = np.array([5.0, 5.0, 5.0])
    lo = np.array([0.0, 6.0, 4.0])
    hi = np.array([10.0, 9.0, 4.5])
    s = cqr_scores(y, lo, hi)
    assert s[0] < 0          # 界内富余
    assert s[1] == 1.0       # 真值低于下界
    assert s[2] == 0.5       # 真值高于上界


def test_conformal_quantile_level_correction():
    # n=9, nominal=0.8 → level = ceil(10*0.8)/9 = 8/9
    s = np.arange(9, dtype=float)  # 0..8
    q = conformal_quantile(s, 0.8)
    assert abs(q - np.quantile(s, 8 / 9)) < 1e-12


def test_conformal_quantile_empty():
    assert conformal_quantile(np.array([]), 0.9) == 0.0


def test_mondrian_offsets_per_group():
    # 组 A 全在界外（需要正偏移），组 B 全在界内（偏移 ≤ 0）
    y = np.array([10.0, 10.0, 5.0, 5.0])
    lo = np.array([8.0, 8.0, 0.0, 0.0])
    hi = np.array([9.0, 9.0, 10.0, 10.0])
    g = np.array(["A", "A", "B", "B"])
    q = group_quantiles(y, lo, hi, g, nominal=0.5)
    assert q["A"] > 0
    assert q["B"] <= 0


def test_mondrian_uniformizes_coverage():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.normal(0, 1, n)
    pred = np.zeros(n)
    # 组 A 区间过窄（±0.5，覆盖 ~0.38），组 B 过宽（±5，覆盖 ~1.0）
    g = np.where(np.arange(n) < 200, "A", "B")
    half = np.where(np.arange(n) < 200, 0.5, 5.0)
    lo, hi = pred - half, pred + half
    q = group_quantiles(y, lo, hi, g, nominal=0.8)
    assert q["A"] > 0 and q["B"] < 0
    lo2, hi2 = apply_offsets(lo, hi, g, q)
    cov = group_coverage(y, lo2, hi2, g)
    assert abs(cov["A"]["coverage"] - 0.8) <= 0.1
    assert abs(cov["B"]["coverage"] - 0.8) <= 0.1
    assert coverage_spread(cov) <= 0.1


def test_apply_offsets_unknown_group_untouched():
    lo = np.array([0.0, 0.0])
    hi = np.array([2.0, 2.0])
    g = np.array(["A", "B"])
    lo2, hi2 = apply_offsets(lo, hi, g, {"A": 1.0})
    assert lo2[0] == -1.0 and hi2[0] == 3.0
    assert lo2[1] == 0.0 and hi2[1] == 2.0


# ---------- events ----------

def test_detect_ramps_step():
    dt = pd.date_range("2020-05-15 08:00", periods=12, freq="15min")
    p = np.full(12, 1000.0)
    p[6:] = 300.0  # 阶跃 −700 W，capacity=1000 → frac=0.2 命中一次
    ev = detect_ramps(dt, p, capacity=1000.0, frac=0.2)
    assert len(ev) == 1
    assert ev.loc[0, "index"] == 6
    assert abs(ev.loc[0, "delta_w"] + 700.0) < 1e-9


def test_detect_ramps_two_way():
    dt = pd.date_range("2020-05-15 08:00", periods=12, freq="15min")
    p = np.full(12, 1000.0)
    p[6] = 300.0  # 单槽凹陷：下坡 + 上坡各一次
    ev = detect_ramps(dt, p, capacity=1000.0, frac=0.2)
    assert set(ev["index"]) == {6, 7}


def test_detect_ramps_respects_time_gap():
    dt = list(pd.date_range("2020-05-15 08:00", periods=3, freq="15min"))
    dt += list(pd.date_range("2020-05-16 08:00", periods=3, freq="15min"))
    p = np.array([1000.0, 1000.0, 200.0, 200.0, 1000.0, 1000.0])
    # 跨天缺口处也有 800/800 的跳变，但 Δt≠15min 不应计
    ev = detect_ramps(pd.Series(dt), p, capacity=1000.0, frac=0.2)
    assert set(ev["index"]) == {2, 4}


def test_clear_sky_envelope_shape():
    dt = pd.date_range("2020-05-01", periods=5 * 96, freq="15min")
    hour = dt.hour + dt.minute / 60.0
    irr = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None) * 1000.0
    env = clear_sky_envelope(dt, irr)
    assert len(env) == 96
    assert env.loc[48] > env.loc[30] > env.loc[6]  # 正午 > 上午 > 夜间
    assert env.min() >= 0


def test_clearness_index_range():
    dt = pd.date_range("2020-05-01", periods=96, freq="15min")
    hour = dt.hour + dt.minute / 60.0
    irr = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None) * 800.0
    env = clear_sky_envelope(dt, irr * 1.0)  # 包络 ~800
    kt = clearness_index(dt, irr, env, cap=1.2)
    assert np.all(kt >= 0) and np.all(kt <= 1.2)
    day = (hour >= 8) & (hour <= 16)
    assert np.all(np.abs(kt[day] - 1.0) < 0.15)  # 与包络同形 → ≈1
