"""08 分位数概率预测：LightGBM q10/q50/q90 + 验证集共形校准（CQR）。"""
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
from solar_diag.models import train_lgbm
from solar_diag.pipeline import make_frames, prepare_plant

QUANTILES = {"q10": 0.1, "q50": 0.5, "q90": 0.9}
NOMINAL = 0.80  # q10–q90 名义覆盖


def conformal_adjust(y_va, lo_va, hi_va, lo_te, hi_te, nominal: float = NOMINAL):
    """CQR：验证集残差分位扩张/收缩区间，使经验覆盖贴近名义值。"""
    scores = np.maximum(lo_va - y_va, y_va - hi_va)  # >0 表示出界
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * nominal) / n)
    q_hat = float(np.quantile(scores, level))
    return lo_te - q_hat, hi_te + q_hat, q_hat


def run_plant(plant, cfg, fig_dir, res_dir):
    pl, cap = prepare_plant(plant, cfg)
    splits = make_frames(pl, cfg, "weather")
    x_tr, y_tr, _ = splits["train"]
    x_va, y_va, _ = splits["val"]
    x_te, y_te, m_te = splits["test"]
    y_va_a, y_te_a = y_va.to_numpy(), y_te.to_numpy()

    models, preds, preds_va = {}, {}, {}
    for name, q in QUANTILES.items():
        m = train_lgbm(x_tr, y_tr, x_va, y_va,
                       {**cfg["lgbm"], "objective": "quantile", "alpha": q})
        models[name] = m
        preds[name] = m.predict(x_te)
        preds_va[name] = m.predict(x_va)

    lo_raw, hi_raw = preds["q10"], preds["q90"]
    lo_cal, hi_cal, q_hat = conformal_adjust(
        y_va_a, preds_va["q10"], preds_va["q90"], lo_raw, hi_raw,
    )

    cov_raw = ev.interval_coverage(y_te_a, lo_raw, hi_raw)
    cov_cal = ev.interval_coverage(y_te_a, lo_cal, hi_cal)
    res = {
        "capacity_w": cap,
        "conformal_q_hat_w": q_hat,
        "raw": {"coverage": cov_raw, "width_w": ev.interval_width(lo_raw, hi_raw),
                "width_pct_cap": 100 * ev.interval_width(lo_raw, hi_raw) / cap},
        "calibrated": {"coverage": cov_cal, "width_w": ev.interval_width(lo_cal, hi_cal),
                       "width_pct_cap": 100 * ev.interval_width(lo_cal, hi_cal) / cap},
        "pinball": {name: ev.pinball_loss(y_te_a, preds[name], q) for name, q in QUANTILES.items()},
        "median_metrics": ev.metric_pack(y_te_a, preds["q50"], cap),
    }

    # 覆盖率按辐照分桶（校准后）
    buckets = ev.bucket_by_irradiation(m_te["target_irr"], cfg["irr_bins"])
    bdf = pd.DataFrame({"bucket": buckets, "inside": (y_te_a >= lo_cal) & (y_te_a <= hi_cal)})
    cov_b = bdf.groupby("bucket", observed=True)["inside"].agg(["mean", "count"]).reset_index()
    cov_b.columns = ["bucket", "coverage", "n"]
    res["coverage_by_bucket"] = cov_b.to_dict(orient="records")

    # 夜间/白天覆盖
    day = (m_te["target_irr"] > cfg["daytime_irr_min"]).to_numpy()
    res["coverage_daytime"] = ev.interval_coverage(y_te_a[day], lo_cal[day], hi_cal[day]) if day.any() else None
    res["coverage_night"] = ev.interval_coverage(y_te_a[~day], lo_cal[~day], hi_cal[~day]) if (~day).any() else None

    plots.plot_quantile_bands(y_te_a, lo_cal, hi_cal, m_te["target_dt"].to_numpy(), plant.name, fig_dir)
    plots.plot_coverage_by_bucket(cov_b, plant.name, fig_dir)
    print(f"  [{plant.name}] raw_cov={cov_raw:.3f} cal_cov={cov_cal:.3f} "
          f"width={res['calibrated']['width_pct_cap']:.1f}%cap q_hat={q_hat:.0f}W")
    return res


def main() -> None:
    cfg = load_config()
    raw = Path(cfg["data_root"]) / cfg["raw_subdir"]
    plants = load_all(raw, cfg["plants"])
    fig_dir, res_dir = figures_dir(), results_dir()
    report = {}
    for name, p in plants.items():
        print(f"== {name} ==")
        report[name] = run_plant(p, cfg, fig_dir, res_dir)
    out = res_dir / "quantile_metrics.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
