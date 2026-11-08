"""03 LSTM 时序回归（nowcast / weather 两口径，3 种子集成）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar_diag import evaluation as ev
from solar_diag import plots
from solar_diag.config import figures_dir, load_config, results_dir
from solar_diag.features import sequence_arrays
from solar_diag.io import load_all
from solar_diag.models import predict_lstm, train_lstm
from solar_diag.pipeline import daytime_mask, make_frames, prepare_plant


def _fit_scaler(x_tr: np.ndarray, y_tr: np.ndarray):
    """按 train 集统计标准化特征与目标（LSTM 对量纲敏感，必须归一）。"""
    mu = x_tr.mean(axis=(0, 1))
    sd = x_tr.std(axis=(0, 1)) + 1e-8
    y_mu, y_sd = float(y_tr.mean()), float(y_tr.std()) + 1e-8
    return mu, sd, y_mu, y_sd


def _apply(x: np.ndarray, mu, sd) -> np.ndarray:
    return (x - mu) / sd


def run_plant(plant, cfg, fig_dir, res_dir):
    pl, cap = prepare_plant(plant, cfg)
    hist = cfg["history"]
    out = {"capacity_w": cap, "regimes": {}}
    for regime in ("nowcast", "weather"):
        splits = make_frames(pl, cfg, regime)
        arrays = {k: sequence_arrays(x, y, hist) for k, (x, y, m) in splits.items()}
        metas = {k: m for k, (x, y, m) in splits.items()}
        (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = arrays["train"], arrays["val"], arrays["test"]
        m_te = metas["test"].iloc[hist - 1:].reset_index(drop=True)
        mu, sd, y_mu, y_sd = _fit_scaler(x_tr, y_tr)
        x_tr_n, x_va_n, x_te_n = _apply(x_tr, mu, sd), _apply(x_va, mu, sd), _apply(x_te, mu, sd)
        y_tr_n, y_va_n = (y_tr - y_mu) / y_sd, (y_va - y_mu) / y_sd
        seed_metrics, seed_preds = [], []
        last_hist = None
        for seed in cfg["seeds"]:
            model, last_hist = train_lstm(
                x_tr_n, y_tr_n, x_va_n, y_va_n,
                hidden=cfg["lstm"]["hidden"], layers=cfg["lstm"]["layers"],
                dropout=cfg["lstm"]["dropout"], lr=cfg["lstm"]["lr"],
                batch=cfg["lstm"]["batch"], epochs=cfg["lstm"]["epochs"],
                patience=cfg["lstm"]["patience"], seed=seed,
            )
            yp = predict_lstm(model, x_te_n) * y_sd + y_mu
            seed_preds.append(yp)
            seed_metrics.append(ev.metric_pack(y_te, yp, cap))
            print(f"  [{plant.name}/{regime}/LSTM seed{seed}] RMSE={seed_metrics[-1]['rmse']:.1f}W")
        ens = np.mean(seed_preds, axis=0)
        m_ens = ev.metric_pack(y_te, ens, cap)
        day_mask = daytime_mask(m_te, cfg["daytime_irr_min"]).to_numpy()
        m_day = ev.metric_pack(y_te[day_mask], ens[day_mask], cap) if day_mask.any() else {}
        # 历史 val_rmse 为标准化单位，换算回 W 便于阅读
        hist_w = [{"epoch": h["epoch"], "val_rmse": h["val_rmse"] * y_sd} for h in last_hist]
        plots.plot_train_curves(hist_w, plant.name, regime, fig_dir)
        plots.plot_pred_vs_true(y_te, ens, f"LSTM_{regime}", plant.name, fig_dir)
        buckets = ev.bucket_by_irradiation(m_te["target_irr"], cfg["irr_bins"])
        bdf = ev.grouped_metrics(y_te, ens, buckets, cap)
        plots.plot_error_by_bucket(bdf, f"LSTM_{regime}", plant.name, "天气", fig_dir)
        np.savez_compressed(
            res_dir / f"pred_{plant.name}_LSTM_{regime}.npz",
            y_true=y_te, y_pred=ens,
            target_dt=m_te["target_dt"].to_numpy(),
            target_irr=m_te["target_irr"].to_numpy(),
            target_hour=m_te["target_hour"].to_numpy(),
        )
        out["regimes"][regime] = {
            "seeds": seed_metrics,
            "ensemble": {"all": m_ens, "daytime": m_day},
            "ensemble_seed_rmse_std": float(np.std([s["rmse"] for s in seed_metrics])),
        }
        print(f"  [{plant.name}/{regime}/LSTM ens] RMSE={m_ens['rmse']:.1f}W nRMSE={m_ens['nrmse_pct']:.2f}%")
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
    out = res_dir / "lstm_metrics.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
