"""轻量 Transformer 编码器回归 + 与 LSTM 完全同协议的通用序列训练器。

回应 N3 报告 §6 未做项"Transformer"：34 天 × 96 槽 ≈ 3.2k 点/站属小样本，
v1.0 的 LSTM（3 种子集成 10.64%/12.50%）未超过 LightGBM（9.98%/11.87%）。
本模块用同一个 train_sequence_model 训练 LSTM 与 Transformer，保证
标准化、早停、优化器、种子、batch 完全一致，唯一变量是网络结构。

参数量对照（weather 口径 n_feat=16）：
  LSTM hidden=64 ≈ 21.1k；Transformer d_model=48/layers=1 ≈ 20k（刻意配平）。
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from .models import LSTMRegressor


class PositionalEncoding(nn.Module):
    """固定正弦位置编码（不引入可训练参数，便于参数量公平对比）。"""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerRegressor(nn.Module):
    """批量优先的 TransformerEncoder + 末时刻池化 + 线性头。"""

    def __init__(self, n_feat: int, d_model: int = 48, nhead: int = 4,
                 layers: int = 1, dim_ff: int = 96, dropout: float = 0.1,
                 pooling: str = "last", max_len: int = 512):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model 必须能被 nhead 整除")
        self.input_proj = nn.Linear(n_feat, d_model)
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=layers, enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d_model)
        self.pooling = pooling
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pos(self.input_proj(x))
        h = self.norm(self.encoder(h))
        z = h[:, -1] if self.pooling == "last" else h.mean(dim=1)
        return self.head(z).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def train_sequence_model(
    model: nn.Module,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    lr: float = 1e-3,
    batch: int = 64,
    epochs: int = 30,
    patience: int = 5,
    seed: int = 0,
) -> tuple[nn.Module, list[dict], float]:
    """与 train_lstm 逐行等价的通用训练器；额外返回墙钟训练耗时（秒）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    xt = torch.from_numpy(np.asarray(x_tr, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(y_tr, dtype=np.float32))
    xv = torch.from_numpy(np.asarray(x_va, dtype=np.float32))
    yv = torch.from_numpy(np.asarray(y_va, dtype=np.float32))
    ds = torch.utils.data.TensorDataset(xt, yt)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best, best_state, bad = float("inf"), None, 0
    history: list[dict] = []
    t0 = time.perf_counter()
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
    elapsed = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history, elapsed


def predict_sequence(model: nn.Module, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
        preds = [model(xt[i:i + 1024]).numpy() for i in range(0, len(xt), 1024)]
    return np.concatenate(preds) if preds else np.array([])


def build_sequence_model(kind: str, n_feat: int) -> nn.Module:
    """按名称构造模型：lstm / tr_small（参数配平）/ tr_deep（容量加倍）。"""
    if kind == "lstm":
        return LSTMRegressor(n_feat, hidden=64, layers=1, dropout=0.1)
    if kind == "tr_small":
        return TransformerRegressor(n_feat, d_model=48, nhead=4, layers=1,
                                    dim_ff=96, dropout=0.1)
    if kind == "tr_deep":
        return TransformerRegressor(n_feat, d_model=64, nhead=4, layers=2,
                                    dim_ff=128, dropout=0.1)
    raise ValueError(f"未知模型: {kind}")
