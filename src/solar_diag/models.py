"""模型：基线、Ridge、LightGBM、XGBoost、LSTM。"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:  # 可选依赖，测试环境缺失时跳过对应模型
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None
try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None

import torch
from torch import nn
from sklearn.linear_model import Ridge


# ---------- 基线 ----------

def predict_persistence(ac: pd.Series, horizon: int) -> np.ndarray:
    """持久性：y(t+h) = y(t)（与监督框对齐：返回每个 t 的预测）。"""
    return ac.to_numpy(dtype=np.float64)


def predict_slot_mean(train: pd.DataFrame, target_dts: pd.Series) -> np.ndarray:
    """槽均值基线：按目标时刻的「当日分钟槽」取训练集均值。"""
    slots = train["dt"].dt.hour * 60 + train["dt"].dt.minute
    mean_by_slot = train.groupby(slots)["ac_power"].mean()
    t_slots = pd.to_datetime(target_dts).dt.hour * 60 + pd.to_datetime(target_dts).dt.minute
    return t_slots.map(mean_by_slot).to_numpy(dtype=np.float64)


# ---------- GBDT / 线性 ----------

def train_ridge(x_tr, y_tr, alpha: float = 1.0) -> Ridge:
    m = Ridge(alpha=alpha)
    m.fit(x_tr, y_tr)
    return m


def train_lgbm(x_tr, y_tr, x_va=None, y_va=None, params: dict | None = None):
    if lgb is None:
        raise ImportError("lightgbm 未安装")
    p = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20, verbose=-1)
    p.update(params or {})
    m = lgb.LGBMRegressor(**p)
    fit_kw = {}
    if x_va is not None:
        fit_kw["eval_set"] = [(x_va, y_va)]
        fit_kw["callbacks"] = [lgb.early_stopping(50, verbose=False)]
    m.fit(x_tr, y_tr, **fit_kw)
    return m


def train_xgboost(x_tr, y_tr, x_va=None, y_va=None, params: dict | None = None):
    if xgb is None:
        raise ImportError("xgboost 未安装")
    p = dict(n_estimators=400, learning_rate=0.05, max_depth=6, verbosity=0)
    p.update(params or {})
    fit_kw = {}
    if x_va is not None:
        p["early_stopping_rounds"] = 50  # xgboost>=2：构造器参数
        fit_kw["eval_set"] = [(x_va, y_va)]
        fit_kw["verbose"] = False
    m = xgb.XGBRegressor(**p)
    m.fit(x_tr, y_tr, **fit_kw)
    return m


# ---------- LSTM ----------

class LSTMRegressor(nn.Module):
    def __init__(self, n_feat: int, hidden: int = 64, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            n_feat, hidden, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):  # x: (b, t, f)
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def train_lstm(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    hidden: int = 64,
    layers: int = 1,
    dropout: float = 0.1,
    lr: float = 1e-3,
    batch: int = 64,
    epochs: int = 30,
    patience: int = 5,
    seed: int = 0,
) -> tuple[LSTMRegressor, list[dict]]:
    """CPU 小 LSTM，早停以验证 RMSE 为准。返回 (model, history)。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = LSTMRegressor(x_tr.shape[-1], hidden, layers, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    xt = torch.from_numpy(x_tr.astype(np.float32))
    yt = torch.from_numpy(y_tr.astype(np.float32))
    xv = torch.from_numpy(x_va.astype(np.float32))
    yv = torch.from_numpy(y_va.astype(np.float32))
    ds = torch.utils.data.TensorDataset(xt, yt)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, generator=torch.Generator().manual_seed(seed))
    best = float("inf")
    best_state = None
    bad = 0
    history: list[dict] = []
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va_rmse = float(torch.sqrt(loss_fn(model(xv), yv)))
        history.append({"epoch": ep, "val_rmse": va_rmse})
        if va_rmse < best - 1e-6:
            best = va_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


def predict_lstm(model: LSTMRegressor, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        xt = torch.from_numpy(x.astype(np.float32))
        preds = []
        for i in range(0, len(xt), 1024):
            preds.append(model(xt[i:i + 1024]).numpy())
    return np.concatenate(preds) if preds else np.array([])
