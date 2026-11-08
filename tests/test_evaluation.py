import numpy as np
import pandas as pd

from solar_diag.evaluation import (
    bucket_by_daypart, bucket_by_irradiation, grouped_metrics, mae, metric_pack, nrmse_pct, r2, rmse,
)


def test_rmse_mae_known():
    y = np.array([0.0, 0.0, 0.0])
    p = np.array([3.0, 4.0, 0.0])
    assert np.isclose(rmse(y, p), np.sqrt((9 + 16) / 3))
    assert np.isclose(mae(y, p), 7 / 3)


def test_r2_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert np.isclose(r2(y, y), 1.0)


def test_nrmse():
    y = np.array([0.0, 100.0])
    p = np.array([0.0, 0.0])
    assert np.isclose(nrmse_pct(y, p, capacity=100.0), 100.0 * np.sqrt((0 + 100**2) / 2) / 100.0)


def test_bucket_irradiation():
    b = bucket_by_irradiation([0.0, 0.3, 0.6, 0.9], [0.0, 0.2, 0.5, 0.8, 999.0])
    assert b.iloc[0] == "[0,0.2)"
    assert b.iloc[3] == "[0.8,999)"


def test_bucket_daypart():
    b = bucket_by_daypart([3, 7, 13, 20])
    assert b.iloc[0] == "night(0-6)"
    assert b.iloc[2] == "noon(12-15)"


def test_grouped_metrics():
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    p = pd.Series([1.0, 2.0, 3.0, 4.0])
    g = pd.Series(["a", "a", "b", "b"])
    df = grouped_metrics(y, p, g, capacity=4.0)
    assert len(df) == 2
    assert (df["rmse"] == 0).all()
    assert df["n"].tolist() == [2, 2]


def test_metric_pack_keys():
    m = metric_pack([1.0, 2.0], [1.0, 2.0], 2.0)
    assert set(m) == {"rmse", "mae", "r2", "nrmse_pct", "n"}
