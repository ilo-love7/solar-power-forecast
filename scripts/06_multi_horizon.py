"""06 多提前量预测退化：15min→4h，Persistence vs LightGBM（nowcast/weather）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar_diag import evaluation as ev
from solar_diag import plots
from solar_diag.config import figures_dir, load_config, results_dir
from solar_diag.io import load_all
from solar_diag.models import predict_persistence, train_lgbm
from solar_diag.pipeline import daytime_mask, make_frames, prepare_plant

HORIZONS = [1, 2, 4, 8, 16]  # 15min / 30min / 1h / 2h / 4h


def run_plant(plant, cfg, fig_dir, res_dir):
    pl, cap = prepare_plant(plant, cfg)
    rows = []
    for h in HORIZONS:
        cfg_h = dict(cfg)
        cfg_h["horizon_steps"] = h
        for regime in ("nowcast", "weather"):
            splits = make_frames(pl, cfg_h, regime)
            x_tr, y_tr, _ = splits["train"]
            x_va, y_va, _ = splits["val"]
            x_te, y_te, m_te = splits["test"]
            y_te_a = y_te.to_numpy()

            if regime == "nowcast":  # 持久性与 regime 无关，只记一次
                p_pers = predict_persistence(x_te["ac_now"], h)
                m = ev.metric_pack(y_te_a, p_pers, cap)
                rows.append({"horizon_steps": h, "horizon_min": 15 * h, "model": "Persistence",
                             "regime": "-", "all": m})
                print(f"  [{plant.name}/h={15*h}min/Persistence] nRMSE={m['nrmse_pct']:.2f}%")

            lgbm = train_lgbm(x_tr, y_tr, x_va, y_va, cfg["lgbm"])
            yp = lgbm.predict(x_te)
            m_all = ev.metric_pack(y_te_a, yp, cap)
            day_mask = daytime_mask(m_te, cfg["daytime_irr_min"]).to_numpy()
            m_day = ev.metric_pack(y_te_a[day_mask], yp[day_mask], cap) if day_mask.any() else {}
            rows.append({"horizon_steps": h, "horizon_min": 15 * h, "model": "LightGBM",
                         "regime": regime, "all": m_all, "daytime": m_day})
            print(f"  [{plant.name}/h={15*h}min/LightGBM-{regime}] nRMSE={m_all['nrmse_pct']:.2f}% "
                  f"day={m_day.get('nrmse_pct', float('nan')):.2f}%")

    df = pd.DataFrame([{
        "horizon_min": r["horizon_min"], "model": r["model"], "regime": r["regime"],
        "nrmse_pct": r["all"]["nrmse_pct"], "rmse": r["all"]["rmse"], "r2": r["all"]["r2"],
    } for r in rows])
    plots.plot_horizon_curves(df, plant.name, fig_dir)
    return {"capacity_w": cap, "rows": rows}


def main() -> None:
    cfg = load_config()
    raw = Path(cfg["data_root"]) / cfg["raw_subdir"]
    plants = load_all(raw, cfg["plants"])
    fig_dir, res_dir = figures_dir(), results_dir()
    report = {}
    for name, p in plants.items():
        print(f"== {name} ==")
        report[name] = run_plant(p, cfg, fig_dir, res_dir)
    out = res_dir / "multi_horizon_metrics.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
