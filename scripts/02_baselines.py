"""02 基线 + Ridge + LightGBM + XGBoost（nowcast / weather 两个特征口径）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar_diag import evaluation as ev
from solar_diag import plots
from solar_diag.config import figures_dir, load_config, results_dir
from solar_diag.features import NOWCAST_FEATS
from solar_diag.io import load_all
from solar_diag.models import predict_persistence, predict_slot_mean, train_lgbm, train_ridge, train_xgboost
from solar_diag.pipeline import daytime_mask, make_frames, prepare_plant


def run_plant(plant, cfg, fig_dir, res_dir):
    pl, cap = prepare_plant(plant, cfg)
    out = {"capacity_w": cap, "regimes": {}}
    for regime in ("nowcast", "weather"):
        splits = make_frames(pl, cfg, regime)
        x_tr, y_tr, m_tr = splits["train"]
        x_va, y_va, m_va = splits["val"]
        x_te, y_te, m_te = splits["test"]
        y_tr_a, y_va_a, y_te_a = y_tr.to_numpy(), y_va.to_numpy(), y_te.to_numpy()
        reg = {}
        preds = {}

        # 持久性基线：y(t+h) = y(t)
        p_pers = predict_persistence(x_te["ac_now"], cfg["horizon_steps"])
        preds["Persistence"] = p_pers
        # 槽均值基线（仅用训练期统计，防测试泄漏）
        train_pl = pl[pl["dt"].dt.normalize() <= m_tr["target_dt"].dt.normalize().max()]
        p_slot = predict_slot_mean(train_pl, m_te["target_dt"])
        preds["SlotMean"] = p_slot

        ridge = train_ridge(x_tr, y_tr, cfg["ridge_alpha"])
        preds["Ridge"] = ridge.predict(x_te)

        lgbm = train_lgbm(x_tr, y_tr, x_va, y_va, cfg["lgbm"])
        preds["LightGBM"] = lgbm.predict(x_te)

        xgbm = train_xgboost(x_tr, y_tr, x_va, y_va, cfg["xgboost"])
        preds["XGBoost"] = xgbm.predict(x_te)

        day_mask = daytime_mask(m_te, cfg["daytime_irr_min"]).to_numpy()
        for name, yp in preds.items():
            m_all = ev.metric_pack(y_te_a, yp, cap)
            m_day = ev.metric_pack(y_te_a[day_mask], yp[day_mask], cap) if day_mask.any() else {}
            reg[name] = {"all": m_all, "daytime": m_day}
            np.savez_compressed(
                res_dir / f"pred_{plant.name}_{name}_{regime}.npz",
                y_true=y_te_a, y_pred=np.asarray(yp),
                target_dt=m_te["target_dt"].to_numpy(),
                target_irr=m_te["target_irr"].to_numpy(),
                target_hour=m_te["target_hour"].to_numpy(),
            )
            print(f"  [{plant.name}/{regime}/{name}] RMSE={m_all['rmse']:.1f}W "
                  f"nRMSE={m_all['nrmse_pct']:.2f}% R2={m_all['r2']:.3f}")

        # 图：LightGBM 预测散点 + 分桶（详细归因在 05）
        plots.plot_pred_vs_true(y_te_a, preds["LightGBM"], f"LightGBM_{regime}", plant.name, fig_dir)
        buckets = ev.bucket_by_irradiation(m_te["target_irr"], cfg["irr_bins"])
        bdf = ev.grouped_metrics(y_te_a, preds["LightGBM"], buckets, cap)
        plots.plot_error_by_bucket(bdf, f"LightGBM_{regime}", plant.name, "天气", fig_dir)
        out["regimes"][regime] = reg
    return out


def main() -> None:
    cfg = load_config()
    raw = Path(cfg["data_root"]) / cfg["raw_subdir"]
    plants = load_all(raw, cfg["plants"])
    fig_dir, res_dir = figures_dir(), results_dir()
    report = {}
    for name, p in plants.items():
        print(f"== {name} ==")
        report[name] = run_plant(p, cfg, fig_dir, res_dir)
    out = res_dir / "forecast_metrics.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
