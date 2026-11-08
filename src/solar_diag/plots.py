"""绘图（全部落盘 figures/，无交互后端依赖）。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 120, "font.size": 9,
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
})


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_daily_profile(df: pd.DataFrame, plant: str, out_dir: Path) -> str:
    d = df.copy()
    d["slot"] = d["dt"].dt.hour + d["dt"].dt.minute / 60.0
    g = d.groupby("slot")["ac_power"]
    mean, lo, hi = g.mean(), g.quantile(0.1), g.quantile(0.9)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.fill_between(mean.index, lo / 1000, hi / 1000, alpha=0.25, label="P10-P90")
    ax.plot(mean.index, mean / 1000, color="C1", lw=1.8, label="均值")
    ax.set_xlabel("小时")
    ax.set_ylabel("AC 功率 (kW)")
    ax.set_title(f"{plant} 站级功率日内曲线（15min 聚合）")
    ax.legend()
    return _save(fig, out_dir / f"{plant}_daily_profile.png")


def plot_daily_energy(df: pd.DataFrame, plant: str, out_dir: Path) -> str:
    d = df.copy()
    d["date"] = d["dt"].dt.normalize()
    daily = d.groupby("date")["ac_power"].sum() * 0.25 / 1000  # kWh
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(range(len(daily)), daily.values, color="C2", width=0.8)
    ticks = list(range(0, len(daily), 5))
    ax.set_xticks(ticks)
    ax.set_xticklabels([daily.index[i].strftime("%m-%d") for i in ticks], rotation=45, ha="right")
    ax.set_ylabel("日电量 (kWh)")
    ax.set_title(f"{plant} 逐日发电量")
    return _save(fig, out_dir / f"{plant}_daily_energy.png")


def plot_irradiation_scatter(df: pd.DataFrame, plant: str, out_dir: Path) -> str:
    d = df.dropna(subset=["irradiation", "ac_power"])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    hb = ax.hexbin(d["irradiation"], d["ac_power"] / 1000, gridsize=40, cmap="viridis", mincnt=1)
    fig.colorbar(hb, ax=ax, label="计数")
    ax.set_xlabel("辐照度 (kW/m²)")
    ax.set_ylabel("AC 功率 (kW)")
    ax.set_title(f"{plant} 辐照-功率关系")
    return _save(fig, out_dir / f"{plant}_irr_scatter.png")


def plot_weather_corr(df: pd.DataFrame, plant: str, out_dir: Path) -> str:
    cols = ["ac_power", "dc_power", "irradiation", "ambient_temp", "module_temp"]
    corr = df[cols].dropna().corr()
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=40, ha="right")
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"{plant} 变量相关性")
    return _save(fig, out_dir / f"{plant}_corr.png")


def plot_missing_heatmap(gen_inv: pd.DataFrame, plant: str, out_dir: Path) -> str:
    """逆变器 × 日期 的缺槽数热图。"""
    g = gen_inv.copy()
    g["date"] = g["dt"].dt.normalize()
    full = pd.date_range(g["dt"].min().normalize(), g["dt"].max().normalize() + pd.Timedelta(days=1) - pd.Timedelta(minutes=15), freq="15min")
    exp_per_day = 96
    pivot = g.groupby(["SOURCE_KEY", "date"])["dt"].nunique().unstack(fill_value=0)
    pivot = exp_per_day - pivot  # 缺槽数
    fig, ax = plt.subplots(figsize=(9, 0.28 * len(pivot) + 1.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="OrRd", vmin=0, vmax=max(10, pivot.values.max()))
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=6)
    n_dates = pivot.shape[1]
    ax.set_xticks(range(n_dates))
    ax.set_xticklabels([d.strftime("%m-%d") for d in pivot.columns], rotation=90, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8, label="缺槽数")
    ax.set_title(f"{plant} 逆变器缺槽热图（每格=1 天）")
    return _save(fig, out_dir / f"{plant}_missing_heatmap.png")


def plot_pred_vs_true(y_true, y_pred, model: str, plant: str, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(4.6, 4))
    yt, yp = np.asarray(y_true) / 1000, np.asarray(y_pred) / 1000
    ax.scatter(yt, yp, s=6, alpha=0.4)
    lim = max(yt.max(), yp.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlabel("真实 AC 功率 (kW)")
    ax.set_ylabel("预测 (kW)")
    ax.set_title(f"{plant} {model} 预测 vs 真实（测试集）")
    return _save(fig, out_dir / f"{plant}_{model}_pred_vs_true.png")


def plot_error_by_bucket(buckets_df: pd.DataFrame, model: str, plant: str, kind: str, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar(buckets_df["bucket"].astype(str), buckets_df["rmse"] / 1000, color="C3", alpha=0.85, label="RMSE")
    ax2 = ax.twinx()
    ax2.plot(buckets_df["bucket"].astype(str), buckets_df["n"], "ko-", lw=1, ms=4, label="样本数")
    ax2.set_ylabel("样本数")
    ax.set_ylabel("RMSE (kW)")
    ax.set_title(f"{plant} {model} 测试误差按{kind}分桶")
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, out_dir / f"{plant}_{model}_err_by_{kind}.png")


def plot_efficiency_scatter(diag: pd.DataFrame, plant: str, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(4.8, 4))
    colors = np.where(diag["flag"] > 0, "C3", "C0")
    ax.scatter(diag["slope"], diag["intercept"], c=colors, s=40)
    for _, r in diag.iterrows():
        if r["flag"] > 0:
            ax.annotate(r["SOURCE_KEY"][:8], (r["slope"], r["intercept"]), fontsize=6)
    ax.set_xlabel("转换增益 a（AC = a·DC + b）")
    ax.set_ylabel("静态损耗 b (W)")
    ax.set_title(f"{plant} 逆变器增益-静态损耗（红=告警）")
    return _save(fig, out_dir / f"{plant}_eff_scatter.png")


def plot_completeness_bars(diag: pd.DataFrame, plant: str, out_dir: Path) -> str:
    d = diag.sort_values("completeness")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    colors = np.where(d["flag"] > 0, "C3", "C0")
    ax.barh(d["SOURCE_KEY"], d["completeness"], color=colors)
    ax.axvline(0.9, color="k", ls="--", lw=1, label="告警线 0.90")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("槽位完整率")
    ax.set_title(f"{plant} 逆变器数据完整率（红=告警）")
    ax.legend()
    return _save(fig, out_dir / f"{plant}_completeness.png")


def plot_train_curves(history: list[dict], plant: str, regime: str, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(5, 3))
    ep = [h["epoch"] for h in history]
    ax.plot(ep, [h["val_rmse"] for h in history] if "val_rmse" in history[0] else [h["train_loss"] for h in history],
            "o-", ms=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val RMSE")
    ax.set_title(f"{plant}/{regime} LSTM 训练曲线")
    return _save(fig, out_dir / f"{plant}_{regime}_lstm_curve.png")
