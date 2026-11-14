"""14（v1.3 E31）Transformer 对比：小样本时序建模的边界。

N3 报告 §6 未做项之一。v1.0 的 LSTM×3 集成（10.64%/12.50%）未超过
LightGBM（9.98%/11.87%），本脚本用**同一个训练器**（train_sequence_model）
训练 LSTM 与两种 Transformer，保证标准化、优化器、早停、batch、种子完全一致，
唯一变量是网络结构；另加 LightGBM 参照与"打乱时序"消融。

  lstm           hidden=64，≈21k 参数（v1.0 同款）
  tr_small       d_model=48/1 层，≈20k 参数（刻意与 LSTM 配平）
  tr_deep        d_model=64/2 层（检验是否只是容量不足）
  tr_small_shuf  同一网络，但训练/推理时把 16 步窗口固定置换打乱
                 ——若性能不降，说明注意力没用上时序排列（特征里已有 lag+时间编码）
  LightGBM       非序列参照（v1.0 最优）

两站 × 提前量 {1h, 4h} × 3 种子，weather 口径。
产物：results/v13_transformer.json、figures/fig26_transformer.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar_diag import evaluation as ev
from solar_diag import plots  # noqa: F401  字体 rcParams 副作用
from solar_diag.config import figures_dir, load_config, results_dir
from solar_diag.features import sequence_arrays
from solar_diag.io import load_all
from solar_diag.models import train_lgbm
from solar_diag.pipeline import make_frames, prepare_plant
from solar_diag.transformer import (
    build_sequence_model,
    count_parameters,
    predict_sequence,
    train_sequence_model,
)

HORIZONS = [4, 16]  # 1h / 4h
SEQ_MODELS = ["lstm", "tr_small", "tr_deep", "tr_small_shuf"]
ALL_MODELS = ["LightGBM"] + SEQ_MODELS


def fit_scaler(x_tr, y_tr):
    mu = x_tr.mean(axis=(0, 1))
    sd = x_tr.std(axis=(0, 1)) + 1e-8
    return mu, sd, float(y_tr.mean()), float(y_tr.std()) + 1e-8


def apply(x, mu, sd):
    return (x - mu) / sd


def run_one(name, plant, cfg, horizon, seed_shuffle) -> dict:
    pl, cap = prepare_plant(plant, cfg)
    cfg_h = dict(cfg)
    cfg_h["horizon_steps"] = horizon
    hist = cfg["history"]
    splits = make_frames(pl, cfg_h, "weather")
    x_tr, y_tr, _ = splits["train"]
    x_va, y_va, _ = splits["val"]
    x_te, y_te, _ = splits["test"]
    arrays = {k: sequence_arrays(x, y, hist) for k, (x, y, m) in splits.items()}
    (xa_tr, ya_tr), (xa_va, ya_va), (xa_te, ya_te) = (
        arrays["train"], arrays["val"], arrays["test"])

    mu, sd, y_mu, y_sd = fit_scaler(xa_tr, ya_tr)
    xa_tr, xa_va, xa_te = apply(xa_tr, mu, sd), apply(xa_va, mu, sd), apply(xa_te, mu, sd)
    ya_tr_n, ya_va_n = (ya_tr - y_mu) / y_sd, (ya_va - y_mu) / y_sd

    res = {"capacity_w": cap, "horizon_steps": horizon,
           "horizon_min": 15 * horizon, "n_train": int(len(ya_tr)),
           "n_test": int(len(ya_te)), "models": {}}

    # LightGBM 参照（表格特征，非序列）
    m = train_lgbm(x_tr, y_tr, x_va, y_va, cfg["lgbm"])
    yp = m.predict(x_te)
    res["models"]["LightGBM"] = {
        "all": ev.metric_pack(y_te.to_numpy(), yp, cap),
        "seeds": [], "seed_rmse_std": 0.0, "n_params": None,
        "train_s_per_seed": 0.0, "mean_epochs": None,
    }

    for kind in SEQ_MODELS:
        shuffled = kind.endswith("_shuf")
        base = "tr_small" if shuffled else kind
        xt_tr, xt_va, xt_te = (xa_tr, xa_va, xa_te)
        if shuffled:
            perm = np.random.default_rng(seed_shuffle).permutation(hist)
            xt_tr, xt_va, xt_te = xa_tr[:, perm], xa_va[:, perm], xa_te[:, perm]

        seed_metrics, seed_preds, times, epochs = [], [], [], []
        n_params = None
        for seed in cfg["seeds"]:
            # 所有结构在同一 seed 下用同一 RNG 流初始化，保证"唯一变量是网络结构"；
            # 同时与 v1.0 的 train_lstm（seed 后建模）逐项对齐。
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = build_sequence_model(base, xt_tr.shape[-1])
            if n_params is None:
                n_params = count_parameters(model)
            model, history, elapsed = train_sequence_model(
                model, xt_tr, ya_tr_n, xt_va, ya_va_n,
                lr=cfg["lstm"]["lr"], batch=cfg["lstm"]["batch"],
                epochs=cfg["lstm"]["epochs"], patience=cfg["lstm"]["patience"],
                seed=seed,
            )
            yp = predict_sequence(model, xt_te) * y_sd + y_mu
            seed_preds.append(yp)
            seed_metrics.append(ev.metric_pack(ya_te, yp, cap))
            times.append(elapsed)
            epochs.append(len(history))
        ens = np.mean(seed_preds, axis=0)
        res["models"][kind] = {
            "all": ev.metric_pack(ya_te, ens, cap),
            "seeds": seed_metrics,
            "seed_rmse_std": float(np.std([s["rmse"] for s in seed_metrics])),
            "n_params": n_params,
            "train_s_per_seed": float(np.mean(times)),
            "mean_epochs": float(np.mean(epochs)),
        }
        print(f"  [{name}/h={15*horizon}min/{kind:14s}] "
              f"nRMSE={res['models'][kind]['all']['nrmse_pct']:.2f}% "
              f"seed_std={res['models'][kind]['seed_rmse_std']:.0f}W "
              f"params={n_params} t={res['models'][kind]['train_s_per_seed']:.1f}s")
    return res


def draw_figure(out: dict, cfg, fig_dir) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#888888", "#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.4))
    for row, plant in enumerate(cfg["plants"]):
        r = out[plant]
        ax = axes[row, 0]
        xs = np.arange(len(ALL_MODELS))
        for j, h in enumerate(HORIZONS):
            vals = [r["horizons"][str(h)]["models"][m]["all"]["nrmse_pct"]
                    for m in ALL_MODELS]
            ax.bar(xs + (j - 0.5) * 0.36, vals, 0.36,
                   label=f"{15*h}min", color=["#1f77b4", "#ff7f0e"][j])
        ax.set_xticks(xs)
        ax.set_xticklabels(ALL_MODELS, rotation=18, fontsize=8)
        ax.set_ylabel("nRMSE (%)")
        ax.set_title(f"{plant} 测试集 nRMSE（3 种子集成）", fontsize=10)
        if row == 0:
            ax.legend(fontsize=8)

        ax = axes[row, 1]
        h = HORIZONS[0]
        mm = r["horizons"][str(h)]["models"]
        vals = [mm[k]["train_s_per_seed"] for k in ALL_MODELS]
        ax.bar(np.arange(len(ALL_MODELS)), vals, 0.6, color=colors)
        for i, k in enumerate(ALL_MODELS):
            np_ = mm[k]["n_params"]
            ax.annotate("—" if np_ is None else f"{np_/1000:.1f}k",
                        (i, vals[i]), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7.5)
        ax.set_xticks(np.arange(len(ALL_MODELS)))
        ax.set_xticklabels(ALL_MODELS, rotation=18, fontsize=8)
        ax.set_ylabel("单种子训练耗时 (s)")
        ax.set_title(f"{plant} 训练成本（柱上标注参数量，1h 口径）", fontsize=10)
        ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    p = fig_dir / "fig26_transformer.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("图 →", p)


def main() -> int:
    cfg = load_config()
    plants = load_all(Path(cfg["data_root"]) / cfg["raw_subdir"])
    res_dir, fig_dir = results_dir(), figures_dir()
    out: dict = {}
    for name in cfg["plants"]:
        print(f"== {name} ==")
        out[name] = {"horizons": {}}
        for h in HORIZONS:
            print(f"  -- horizon {15*h}min --")
            out[name]["horizons"][str(h)] = run_one(name, plants[name], cfg, h, 2026)
    with open(res_dir / "v13_transformer.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    draw_figure(out, cfg, fig_dir)
    print("结果 →", res_dir / "v13_transformer.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
