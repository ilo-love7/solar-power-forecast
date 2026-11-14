"""v1.3 E31 单测：Transformer 结构与同协议训练器。"""
import numpy as np
import pytest
import torch

from solar_diag.models import LSTMRegressor, train_lstm
from solar_diag.transformer import (
    PositionalEncoding,
    TransformerRegressor,
    build_sequence_model,
    count_parameters,
    predict_sequence,
    train_sequence_model,
)


def test_positional_encoding_shape_and_determinism():
    pe = PositionalEncoding(16, max_len=32)
    x = torch.zeros(2, 10, 16)
    a, b = pe(x), pe(x)
    assert a.shape == (2, 10, 16)
    torch.testing.assert_close(a, b)
    assert not torch.allclose(a[0, 0], a[0, 5])   # 不同位置编码不同


def test_positional_encoding_odd_d_model():
    pe = PositionalEncoding(15, max_len=8)
    assert pe(torch.zeros(1, 4, 15)).shape == (1, 4, 15)


def test_transformer_forward_shape():
    m = TransformerRegressor(16, d_model=48, nhead=4, layers=1)
    m.eval()
    with torch.no_grad():
        assert m(torch.randn(7, 16, 16)).shape == (7,)


def test_transformer_pooling_mean():
    m = TransformerRegressor(16, d_model=32, nhead=4, layers=1, pooling="mean")
    m.eval()
    with torch.no_grad():
        assert m(torch.randn(3, 8, 16)).shape == (3,)


def test_transformer_nhead_mismatch_raises():
    with pytest.raises(ValueError, match="整除"):
        TransformerRegressor(16, d_model=46, nhead=4)


def test_parameter_count_ordering_and_parity():
    lstm = build_sequence_model("lstm", 16)
    small = build_sequence_model("tr_small", 16)
    deep = build_sequence_model("tr_deep", 16)
    n_l, n_s, n_d = map(count_parameters, (lstm, small, deep))
    assert n_l > 0 and n_s > 0
    assert n_d > n_s
    assert 0.5 < n_s / n_l < 2.0     # 小 Transformer 与 LSTM 参数量刻意配平


def test_build_sequence_model_unknown():
    with pytest.raises(ValueError, match="未知模型"):
        build_sequence_model("mlp", 16)


def _toy_sequences(n=240, hist=8, feat=3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, hist, feat)).astype(np.float32)
    y = (2.0 * x[:, -1, 0] + 0.5 * x[:, -2, 1]).astype(np.float32)
    return x[:180], y[:180], x[180:], y[180:]


def test_trainer_reduces_val_rmse():
    xtr, ytr, xva, yva = _toy_sequences()
    torch.manual_seed(0)
    m = TransformerRegressor(3, d_model=16, nhead=4, layers=1, dim_ff=32)
    _, hist, elapsed = train_sequence_model(
        m, xtr, ytr, xva, yva, epochs=25, patience=10, seed=0)
    assert len(hist) >= 2
    assert hist[-1]["val_rmse"] < hist[0]["val_rmse"]
    assert elapsed > 0.0


def test_trainer_early_stops():
    xtr, ytr, xva, yva = _toy_sequences()
    m = TransformerRegressor(3, d_model=16, nhead=4, layers=1, dim_ff=32)
    _, hist, _ = train_sequence_model(
        m, xtr, ytr, xva, yva, epochs=60, patience=3, seed=0)
    assert len(hist) <= 60
    assert hist[-1]["val_rmse"] >= 0


def test_trainer_matches_train_lstm_protocol():
    """同种子、同模型类型下，通用训练器与 v1.0 的 train_lstm 逐 epoch 一致。"""
    xtr, ytr, xva, yva = _toy_sequences()
    torch.manual_seed(7)
    np.random.seed(7)
    m1 = LSTMRegressor(3, hidden=16, layers=1, dropout=0.1)
    _, h1, _ = train_sequence_model(
        m1, xtr, ytr, xva, yva, epochs=6, patience=6, seed=7)
    m2, h2 = train_lstm(
        xtr, ytr, xva, yva, hidden=16, layers=1, dropout=0.1,
        epochs=6, patience=6, seed=7)
    assert len(h1) == len(h2)
    for a, b in zip(h1, h2):
        assert a["val_rmse"] == pytest.approx(b["val_rmse"], rel=1e-5, abs=1e-6)


def test_predict_sequence_length_and_finite():
    _, _, xva, _ = _toy_sequences()
    m = TransformerRegressor(3, d_model=16, nhead=4, layers=1, dim_ff=32)
    p = predict_sequence(m, xva)
    assert p.shape == (len(xva),)
    assert np.isfinite(p).all()


def test_predict_sequence_empty():
    m = TransformerRegressor(3, d_model=16, nhead=4, layers=1, dim_ff=32)
    assert predict_sequence(m, np.empty((0, 8, 3), dtype=np.float32)).size == 0
