import numpy as np
import pandas as pd

from solar_diag.features import (
    NOWCAST_FEATS, WEATHER_FEATS, build_supervised, sequence_arrays, split_supervised_by_day,
)


def test_time_features_range(tiny_plant_df):
    x, y, meta = build_supervised(tiny_plant_df, horizon=4, regime="nowcast")
    for c in ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]:
        assert x[c].between(-1.0001, 1.0001).all()


def test_supervised_no_target_leak(tiny_plant_df):
    x, y, meta = build_supervised(tiny_plant_df, horizon=4, regime="weather")
    assert "ac_power" not in x.columns
    assert "target" not in x.columns
    # 特征里 ac_now 是 t 时刻，target 是 t+4，不应相等（除非常数序列）
    assert not np.allclose(x["ac_now"].to_numpy(), y.to_numpy())


def test_target_is_shifted(tiny_plant_df):
    horizon = 4
    x, y, meta = build_supervised(tiny_plant_df, horizon=horizon, regime="nowcast")
    # 第一行 target 应等于原始 ac_power 在 target_dt 时刻的值
    first_dt = meta["target_dt"].iloc[0]
    expected = tiny_plant_df.loc[tiny_plant_df["dt"] == first_dt, "ac_power"].iloc[0]
    assert np.isclose(y.iloc[0], expected)


def test_regime_feature_sets(tiny_plant_df):
    x_now, _, _ = build_supervised(tiny_plant_df, 4, "nowcast")
    x_wx, _, _ = build_supervised(tiny_plant_df, 4, "weather")
    assert set(x_now.columns) == set(NOWCAST_FEATS)
    assert set(x_wx.columns) == set(WEATHER_FEATS)
    assert set(NOWCAST_FEATS) < set(WEATHER_FEATS)


def test_split_supervised_by_day(tiny_plant_df):
    x, y, meta = build_supervised(tiny_plant_df, 4, "nowcast")
    splits = split_supervised_by_day(x, y, meta, 0.7, 0.15)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    assert tr[2]["target_dt"].max() < va[2]["target_dt"].min()
    assert va[2]["target_dt"].max() < te[2]["target_dt"].min()


def test_sequence_shapes(tiny_plant_df):
    x, y, _ = build_supervised(tiny_plant_df, 4, "nowcast")
    xs, ys = sequence_arrays(x, y, history=8)
    assert xs.shape == (len(x) - 8 + 1, 8, x.shape[1])
    assert ys.shape == (len(x) - 8 + 1,)
    assert np.isclose(ys[0], y.iloc[7])
