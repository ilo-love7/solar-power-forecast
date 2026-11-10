"""10 云跳变（ramp）极端事件：检测、误差剖析、区间可靠性、案例图。

v1.0 §4 指出"预测价值最高的正午强光时段最难，改进方向是云况突变建模"。
本脚本把"云跳变"显式化：测试集相邻 15min 槽 |ΔP| ≥ 20% 容量判为 ramp 事件，
对比事件/非事件窗口的点误差与区间覆盖，给出最坏案例与时间序列案例图。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar_diag import conformal as cf
from solar_diag import evaluation as ev
from solar_diag.config import figures_dir, load_config, results_dir
from solar_diag.events import detect_ramps
from solar_diag.io import load_all
from solar_diag.models import train_lgbm
from solar_diag.pipeline import make_frames, prepare_plant

NOMINAL = 0.80
RAMP_FRAC = 0.2


def main() -> int:
    cfg = load_config()
    data_root = Path(cfg["data_root"]) / cfg["raw_subdir"]
    plants = load_all(data_root)
    res_dir, fig_dir = results_dir(), figures_dir()
    out: dict = {}

    for plant in cfg["plants"]:
        pl, cap = prepare_plant(plants[plant], cfg)
        splits = make_frames(pl, cfg, "weather")
        x_tr, y_tr, _ = splits["train"]
        x_va, y_va, _ = splits["val"]
        x_te, y_te, m_te = splits["test"]
        y_te_a = y_te.to_numpy()

        # 点预测：LightGBM_weather（复用 v1.0 最优口径）
        m_pt = train_lgbm(x_tr, y_tr, x_va, y_va, cfg["lgbm"])
        pred = m_pt.predict(x_te)

        # 区间：q10/q90 + 全局 CQR
        preds_q = {}
        preds_va = {}
        for name, q in [("q10", 0.1), ("q90", 0.9)]:
            m = train_lgbm(x_tr, y_tr, x_va, y_va,
                           {**cfg["lgbm"], "objective": "quantile", "alpha": q})
            preds_q[name] = m.predict(x_te)
            preds_va[name] = m.predict(x_va)
        q_hat = cf.conformal_quantile(
            cf.cqr_scores(y_va.to_numpy(), preds_va["q10"], preds_va["q90"]), NOMINAL)
        lo, hi = preds_q["q10"] - q_hat, preds_q["q90"] + q_hat

        ev_r = detect_ramps(m_te["target_dt"], y_te_a, cap, frac=RAMP_FRAC)
        is_event = np.zeros(len(y_te_a), dtype=bool)
        is_event[ev_r["index"].to_numpy()] = True

        def pack(mask):
            return {
                "n": int(mask.sum()),
                "rmse_w": ev.rmse(y_te_a[mask], pred[mask]),
                "mae_w": ev.mae(y_te_a[mask], pred[mask]),
                "bias_w": float(np.mean(pred[mask] - y_te_a[mask])),
                "coverage": ev.interval_coverage(y_te_a[mask], lo[mask], hi[mask]),
                "width_w": ev.interval_width(lo[mask], hi[mask]),
            }

        ev_mask, quiet_mask = is_event, ~is_event
        # ramp 事件也看「跳变方向」：下坡（云遮）常伴随高估
        down = ev_r[ev_r["delta_w"] < 0]
        up = ev_r[ev_r["delta_w"] > 0]

        res = {
            "capacity_w": cap,
            "ramp_frac": RAMP_FRAC,
            "n_events": int(len(ev_r)),
            "event_share": float(is_event.mean()),
            "conformal_q_hat_w": q_hat,
            "event": pack(ev_mask),
            "quiet": pack(quiet_mask),
            "down_ramp": {
                "n": int(len(down)),
                "bias_w": (float(np.mean(
                    pred[down["index"].to_numpy()] - y_te_a[down["index"].to_numpy()]))
                    if len(down) else 0.0),
            },
            "up_ramp": {
                "n": int(len(up)),
                "bias_w": (float(np.mean(
                    pred[up["index"].to_numpy()] - y_te_a[up["index"].to_numpy()]))
                    if len(up) else 0.0),
            },
            "worst": [],
        }
        top = ev_r.head(5)
        for _, r in top.iterrows():
            i = int(r["index"])
            res["worst"].append({
                "dt": str(r["dt"]), "delta_w": float(r["delta_w"]),
                "y_true_w": float(y_te_a[i]), "y_pred_w": float(pred[i]),
                "error_w": float(pred[i] - y_te_a[i]),
            })
        out[plant] = res
        print(f"{plant}: events={res['n_events']} ({100*res['event_share']:.1f}%) "
              f"RMSE event={res['event']['rmse_w']:.0f} vs quiet={res['quiet']['rmse_w']:.0f} "
              f"cov event={res['event']['coverage']:.3f} vs quiet={res['quiet']['coverage']:.3f} "
              f"down_bias={res['down_ramp']['bias_w']:+.0f}W")

        # 案例图：最坏 3 个事件 ±8 槽
        import matplotlib
        matplotlib.use("Agg")
        import solar_diag.plots  # noqa: F401  字体 rcParams 副作用
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
        for ax, (_, r) in zip(axes, ev_r.head(3).iterrows()):
            i = int(r["index"])
            lo_i, hi_i = max(0, i - 8), min(len(y_te_a), i + 9)
            xs = np.arange(lo_i, hi_i)
            ax.plot(xs, y_te_a[lo_i:hi_i], "k-", lw=1.2, label="实际")
            ax.plot(xs, pred[lo_i:hi_i], color="#1f77b4", lw=1.0, label="预测")
            ax.fill_between(xs, lo[lo_i:hi_i], hi[lo_i:hi_i],
                            color="#1f77b4", alpha=0.18, label="CQR 区间")
            ax.axvline(i, color="#d62728", ls="--", lw=0.9)
            ax.set_title(str(r["dt"])[:16], fontsize=8)
            ax.set_xlabel("测试样本序号", fontsize=7)
        axes[0].set_ylabel("AC 功率 (W)")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / f"fig22_ramp_cases_{plant}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    with open(res_dir / "ramp_events.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("图 →", fig_dir / "fig22_ramp_cases_Plant_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
