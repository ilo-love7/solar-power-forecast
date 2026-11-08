"""07 跨站迁移：P1→P2 / P2→P1 零样本、容量归一化、少样本适配 vs 域内上限。"""
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
from solar_diag.features import WEATHER_FEATS
from solar_diag.io import load_all
from solar_diag.models import predict_persistence, train_lgbm
from solar_diag.pipeline import make_frames, prepare_plant

POWER_COLS = ["ac_now", "ac_lag1", "ac_lag4", "ac_lag8", "ac_lag96", "ac_roll_mean4"]
FEW_SHOT_DAYS = [1, 3, 7]


def _scale_xy(x: pd.DataFrame, y: pd.Series, cap: float, normalize: bool):
    """normalize=True 时功率特征/目标按容量归一（跨站可比），天气列保持物理单位。"""
    if not normalize:
        return x.copy(), y.copy()
    xn = x.copy()
    for c in POWER_COLS:
        if c in xn.columns:
            xn[c] = xn[c] / cap
    return xn, y / cap


def _frames(pl, cfg):
    splits = make_frames(pl, cfg, "weather")
    return {k: (x[WEATHER_FEATS], y, m) for k, (x, y, m) in splits.items()}


def run_direction(src_plant, tgt_plant, cfg, cap_src, cap_tgt):
    """src 训练 → tgt 测试；返回各变体指标（全部换算回目标站 W）。"""
    src = _frames(src_plant, cfg)
    tgt = _frames(tgt_plant, cfg)
    x_ts, y_ts, _ = tgt["test"]
    y_ts_a = y_ts.to_numpy()
    out = {}

    # 参照：目标站持久性
    pers = predict_persistence(x_ts["ac_now"], cfg["horizon_steps"])
    out["Persistence(tgt)"] = ev.metric_pack(y_ts_a, pers, cap_tgt)

    for norm in (False, True):
        xs_tr, ys_tr = _scale_xy(*src["train"][:2], cap_src, norm)
        xs_va, ys_va = _scale_xy(*src["val"][:2], cap_src, norm)
        xt_te, _ = _scale_xy(x_ts, y_ts, cap_tgt, norm)
        m = train_lgbm(xs_tr, ys_tr, xs_va, ys_va, cfg["lgbm"])
        yp = m.predict(xt_te) * (cap_tgt if norm else 1.0)
        out["ZeroShot_norm" if norm else "ZeroShot_raw"] = ev.metric_pack(y_ts_a, yp, cap_tgt)

    # 少样本适配：目标站前 k 个训练天并入源站（归一化口径）
    xt_tr, yt_tr, mt_tr = _frames(tgt_plant, cfg)["train"]
    days = np.array(sorted(mt_tr["target_dt"].dt.normalize().unique()))
    for k in FEW_SHOT_DAYS:
        sel_days = days[:k]
        mask = mt_tr["target_dt"].dt.normalize().isin(sel_days)
        xs_tr, ys_tr = _scale_xy(*src["train"][:2], cap_src, True)
        xk, yk = _scale_xy(xt_tr.loc[mask].reset_index(drop=True), yt_tr.loc[mask].reset_index(drop=True), cap_tgt, True)
        xs_all = pd.concat([xs_tr, xk], ignore_index=True)
        ys_all = pd.concat([ys_tr, yk], ignore_index=True)
        xs_va, ys_va = _scale_xy(*src["val"][:2], cap_src, True)
        xt_te, _ = _scale_xy(x_ts, y_ts, cap_tgt, True)
        m = train_lgbm(xs_all, ys_all, xs_va, ys_va, cfg["lgbm"])
        yp = m.predict(xt_te) * cap_tgt
        out[f"FewShot_{k}d"] = ev.metric_pack(y_ts_a, yp, cap_tgt)

    # 域内上限：目标站自己训练
    xd_tr, yd_tr = _scale_xy(xt_tr, yt_tr, cap_tgt, True)
    xd_va, yd_va = _scale_xy(*tgt["val"][:2], cap_tgt, True)
    xd_te, _ = _scale_xy(x_ts, y_ts, cap_tgt, True)
    m = train_lgbm(xd_tr, yd_tr, xd_va, yd_va, cfg["lgbm"])
    yp = m.predict(xd_te) * cap_tgt
    out["InDomain"] = ev.metric_pack(y_ts_a, yp, cap_tgt)
    return out


def main() -> None:
    cfg = load_config()
    raw = Path(cfg["data_root"]) / cfg["raw_subdir"]
    plants = load_all(raw, cfg["plants"])
    fig_dir, res_dir = figures_dir(), results_dir()
    prepared = {}
    for name, p in plants.items():
        pl, cap = prepare_plant(p, cfg)
        prepared[name] = (pl, cap)

    report, bar_rows = {}, []
    for src_name, tgt_name in [("Plant_1", "Plant_2"), ("Plant_2", "Plant_1")]:
        print(f"== {src_name} -> {tgt_name} ==")
        pl_s, cap_s = prepared[src_name]
        pl_t, cap_t = prepared[tgt_name]
        res = run_direction(pl_s, pl_t, cfg, cap_s, cap_t)
        report[f"{src_name}->{tgt_name}"] = res
        for variant, m in res.items():
            print(f"  {variant:16s} RMSE={m['rmse']:.0f}W nRMSE={m['nrmse_pct']:.2f}% R2={m['r2']:.3f}")
            bar_rows.append({"direction": f"{src_name}->{tgt_name}", "variant": variant,
                             "nrmse_pct": m["nrmse_pct"]})
    plots.plot_transfer_bars(pd.DataFrame(bar_rows), fig_dir)
    out = res_dir / "cross_plant_metrics.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
