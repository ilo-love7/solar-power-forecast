import numpy as np
import pytest

from solar_diag.models import (
    LSTMRegressor, predict_lstm, predict_persistence, predict_slot_mean,
    train_lgbm, train_lstm, train_ridge, train_xgboost,
)
import pandas as pd


def test_persistence():
    s = pd.Series([1.0, 2.0, 3.0])
    assert np.allclose(predict_persistence(s, 4), [1, 2, 3])


def test_slot_mean():
    train = pd.DataFrame({
        "dt": pd.to_datetime(["2020-05-15 08:00", "2020-05-16 08:00", "2020-05-15 09:00"]),
        "ac_power": [100.0, 200.0, 50.0],
    })
    pred = predict_slot_mean(train, pd.Series(pd.to_datetime(["2020-05-17 08:00", "2020-05-17 09:00"])))
    assert np.allclose(pred, [150.0, 50.0])


def _lin_data(n=200, f=5, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, f))
    w = rng.normal(size=f)
    y = x @ w + 0.1 * rng.normal(size=n)
    return x.astype(np.float64), y.astype(np.float64)


def test_ridge_fit():
    x, y = _lin_data()
    m = train_ridge(pd.DataFrame(x), pd.Series(y), alpha=0.1)
    pred = m.predict(pd.DataFrame(x))
    assert np.sqrt(np.mean((pred - y) ** 2)) < 0.5


def test_lgbm_fit():
    lgb = pytest.importorskip("lightgbm")
    x, y = _lin_data()
    m = train_lgbm(pd.DataFrame(x), pd.Series(y))
    pred = m.predict(pd.DataFrame(x))
    assert np.sqrt(np.mean((pred - y) ** 2)) < 0.5


def test_xgboost_fit():
    pytest.importorskip("xgboost")
    x, y = _lin_data()
    m = train_xgboost(pd.DataFrame(x), pd.Series(y))
    pred = m.predict(pd.DataFrame(x))
    assert np.sqrt(np.mean((pred - y) ** 2)) < 0.5


def test_lstm_overfit_small():
    rng = np.random.default_rng(0)
    n, t, f = 64, 8, 3
    x = rng.normal(size=(n, t, f)).astype(np.float32)
    y = x[:, -1, 0]
    model, hist = train_lstm(x[:48], y[:48], x[48:], y[48:], hidden=32, lr=5e-3,
                             epochs=150, patience=150, batch=16, seed=0)
    assert hist[-1]["val_rmse"] < hist[0]["val_rmse"]
    pred = predict_lstm(model, x[:48])
    # 应显著优于预测均值（std(y)≈1）
    assert np.sqrt(np.mean((pred - y[:48]) ** 2)) < 0.6


def test_lstm_module_shape():
    m = LSTMRegressor(n_feat=4, hidden=8)
    import torch
    out = m(torch.zeros(2, 5, 4))
    assert out.shape == (2,)
