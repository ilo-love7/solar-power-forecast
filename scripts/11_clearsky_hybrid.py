"""11 晴空物理混合：智慧持久性基线 + ML 残差学习 + 残差分解。

物理侧（无坐标依赖）：训练期逐 15min 槽 95 分位辐照包络 Ĝ(slot) →
晴空指数 k_t(t) = irr(t)/Ĝ(slot(t)) → 智慧持久性预测
    P_phys(t+h) = β·cap·k_t(t)·Ĝ(slot(t+h))/1000
（β 由训练集最小二乘拟合；夜间包络≈0 自动出 0）。

混合侧：LightGBM 在 weather 特征上学习残差 P − P_phys，P_hyb = P_phys + r̂。

残差分解（测试集，对最优模型）：MSE 按 k_t 桶（阴/多变/晴）与
ramp/平稳 拆解；白天偏差按辐照桶拆解——区分系统性偏差与随机噪声。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar_diag import evaluation as ev
from solar_diag.config import figures_dir, load_config, results_dir
from solar_diag.events import (
    clear_sky_envelope,
    clearness_index,
    detect_ramps,
)
from solar_diag.io import load_all
from solar_diag.models import train_lgbm
from solar_diag.pipeline import make_frames, prepare_plant

RAMP_FRAC = 0.2


def fit_beta(irr_slot: np.ndarray, power: np.ndarray) -> float:
    """P ≈ β·G_eff：最小二乘过原点，β = <P·G>/<G·G>。"""
    g = np.asarray(irr_slot, dtype=np.float64)
    p = np.asarray(power, dtype=np.float64)
    denom = float(np.dot(g, g))
    return float(np.dot(p, g) / denom) if denom > 1e-9 else 0.0


def main() -> int:
    cfg = load_config()
    data_root = Path(cfg["data_root"]) / cfg["raw_subdir"]
    plants = load_all(data_root)
    res_dir, fig_dir = results_dir(), figures_dir()
    out: dict = {}

    for plant in cfg["plants"]:
        pl, cap = prepare_plant(plants[plant], cfg)
        splits = make_frames(pl, cfg, "weather")
        x_tr, y_tr, m_tr = splits["train"]
        x_va, y_va, m_va = splits["val"]
        x_te, y_te, m_te = splits["test"]
        y_tr_a, y_va_a, y_te_a = y_tr.to_numpy(), y_va.to_numpy(), y_te.to_numpy()

        # 晴空包络只用训练集时间戳/辐照
        env = clear_sky_envelope(m_tr["target_dt"], m_tr["target_irr"])

        # 物理输入：特征时刻辐照（nowcast 已知）+ 目标槽的晴空形状（气候学已知）。
        # 注意不能用 target_irr（目标时刻辐照，泄漏）。pl 的 dt 是列不是索引。
        by_dt = pl.set_index("dt")
        irr_t_tr = by_dt["irradiation"].reindex(m_tr["dt"].to_numpy()).to_numpy()
        irr_t_va = by_dt["irradiation"].reindex(m_va["dt"].to_numpy()).to_numpy()
        irr_t_te = by_dt["irradiation"].reindex(m_te["dt"].to_numpy()).to_numpy()
        kt_tr = clearness_index(m_tr["dt"], irr_t_tr, env)
        kt_va = clearness_index(m_va["dt"], irr_t_va, env)
        kt_te = clearness_index(m_te["dt"], irr_t_te, env)

        env_target_tr = env.reindex(
            (pd.to_datetime(m_tr["target_dt"]).dt.hour * 4
             + pd.to_datetime(m_tr["target_dt"]).dt.minute // 15).to_numpy()
        ).to_numpy()
        env_target_va = env.reindex(
            (pd.to_datetime(m_va["target_dt"]).dt.hour * 4
             + pd.to_datetime(m_va["target_dt"]).dt.minute // 15).to_numpy()
        ).to_numpy()
        env_target_te = env.reindex(
            (pd.to_datetime(m_te["target_dt"]).dt.hour * 4
             + pd.to_datetime(m_te["target_dt"]).dt.minute // 15).to_numpy()
        ).to_numpy()

        g_eff_tr = kt_tr * np.nan_to_num(env_target_tr)
        g_eff_va = kt_va * np.nan_to_num(env_target_va)
        g_eff_te = kt_te * np.nan_to_num(env_target_te)

        beta = fit_beta(g_eff_tr, y_tr_a)
        phys_tr = beta * g_eff_tr
        phys_va = beta * g_eff_va
        phys_te = beta * g_eff_te

        # 持久性 / 智慧持久性 / 纯 ML / 混合
        pers_te = by_dt["ac_power"].reindex(m_te["dt"].to_numpy()).to_numpy()
        m_ml = train_lgbm(x_tr, y_tr, x_va, y_va, cfg["lgbm"])
        ml_te = m_ml.predict(x_te)

        resid_tr = y_tr_a - phys_tr
        m_hy = train_lgbm(x_tr, pd.Series(resid_tr), x_va,
                          pd.Series(y_va_a - phys_va), cfg["lgbm"])
        hy_te = phys_te + m_hy.predict(x_te)

        methods = {
            "persistence": pers_te,
            "smart_persistence": phys_te,
            "lgbm_weather": ml_te,
            "hybrid": hy_te,
        }
        pres = {"capacity_w": cap, "beta": beta, "methods": {}}
        for name, p in methods.items():
            pres["methods"][name] = ev.metric_pack(y_te_a, p, cap)
            print(f"{plant} {name:18s} nRMSE={pres['methods'][name]['nrmse_pct']:.2f}% "
                  f"RMSE={pres['methods'][name]['rmse']:.0f}W")

        best_name = min(pres["methods"], key=lambda k: pres["methods"][k]["rmse"])
        best_pred = methods[best_name]
        resid = best_pred - y_te_a  # 有符号残差（正=高估）

        # 残差分解
        kt_full = kt_te
        kt_bins = pd.cut(kt_full, [-0.01, 0.4, 0.8, 1.21],
                         labels=["overcast(<0.4)", "variable(0.4-0.8)", "clear(>0.8)"])
        ev_r = detect_ramps(m_te["target_dt"], y_te_a, cap, frac=RAMP_FRAC)
        is_ramp = np.zeros(len(y_te_a), dtype=bool)
        is_ramp[ev_r["index"].to_numpy()] = True

        def decomp(mask, label):
            r = resid[mask]
            mse = float(np.mean(r ** 2)) if mask.sum() else 0.0
            return {"label": label, "n": int(mask.sum()), "mse_w2": mse,
                    "bias_w": float(np.mean(r)) if mask.sum() else 0.0,
                    "share_mse": 0.0}

        parts = []
        for cat in ["overcast(<0.4)", "variable(0.4-0.8)", "clear(>0.8)"]:
            m = np.asarray(kt_bins.astype(str) == cat)
            parts.append(decomp(m, f"kt_{cat}"))
        parts.append(decomp(is_ramp, "ramp_event"))
        parts.append(decomp(~is_ramp, "smooth"))
        total_mse = float(np.mean(resid ** 2))
        for p in parts:
            p["share_mse"] = p["mse_w2"] * p["n"] / max(total_mse * len(resid), 1e-9)

        irr_bins = pd.cut(np.nan_to_num(m_te["target_irr"]),
                          [-0.01, 0.2, 0.5, 0.8, 999],
                          labels=["[0,0.2)", "[0.2,0.5)", "[0.5,0.8)", "[0.8,inf)"])
        bias_by_irr = {}
        for cat in ["[0,0.2)", "[0.2,0.5)", "[0.5,0.8)", "[0.8,inf)"]:
            m = np.asarray(irr_bins.astype(str) == cat)
            bias_by_irr[cat] = {
                "n": int(m.sum()),
                "bias_w": float(np.mean(resid[m])) if m.sum() else 0.0,
                "rmse_w": ev.rmse(y_te_a[m], best_pred[m]) if m.sum() else 0.0,
            }

        pres["best_method"] = best_name
        pres["residual_decomposition"] = {
            "total_mse_w2": total_mse,
            "parts": parts,
            "bias_by_irr": bias_by_irr,
        }
        out[plant] = pres

    with open(res_dir / "clearsky_hybrid.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # 图：方法对比 + 残差分解
    import matplotlib
    matplotlib.use("Agg")
    import solar_diag.plots  # noqa: F401  字体 rcParams 副作用
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    names = ["persistence", "smart_persistence", "lgbm_weather", "hybrid"]
    xs = np.arange(len(names))
    for k, plant in enumerate(cfg["plants"]):
        vals = [out[plant]["methods"][n]["nrmse_pct"] for n in names]
        axes[0].bar(xs + (k - 0.5) * 0.36, vals, 0.36, label=plant)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(["持久性", "智慧持久性", "LightGBM", "混合"], fontsize=8)
    axes[0].set_ylabel("nRMSE (%容量)")
    axes[0].set_title("物理 / ML / 混合")
    axes[0].legend(fontsize=8)

    plant = cfg["plants"][0]
    parts = out[plant]["residual_decomposition"]["parts"]
    labels = [p["label"] for p in parts]
    shares = [100 * p["share_mse"] for p in parts]
    axes[1].barh(np.arange(len(labels)), shares, color="#1f77b4")
    axes[1].set_yticks(np.arange(len(labels)))
    axes[1].set_yticklabels(labels, fontsize=7)
    axes[1].set_xlabel("MSE 份额 (%)")
    axes[1].set_title(f"{plant}：最优模型残差分解")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig23_clearsky_hybrid.png", dpi=160, bbox_inches="tight")
    print("图 →", fig_dir / "fig23_clearsky_hybrid.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
