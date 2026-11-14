"""13（v1.3 E30）pinball + 共形二段缩放。

v1.1/1.2 的校准都是加成式 CQR（lo−q̂, hi+q̂），本数据集 q̂=0 是空操作；
§3.8 又发现云跳变处 raw 区间已自动加宽到平稳段 2.2–2.8 倍，加成偏移会
把这种逐点异方差结构压平。本脚本把"选分位对"与"校准因子"解耦：

  第一段（pinball）：从 4 对候选分位数 [q05,q95]/[q10,q90]/[q15,q85]/[q20,q80]
                     中按验证集 pinball 锐度选最锐的一对；
  第二段（共形）：  用归一化残差 r=(y−c)/h 求乘性因子 λ，区间 = c ± λ·h。

比较 8 种口径：raw / add(CQR) / add_group(v1.2 Mondrian) / mult /
mult_group / two_stage / two_stage_group / two_stage_shrink(分桶 λ 再软收缩)。
评价除总覆盖、宽度、分桶极差外，重点看 §3.8 的 ramp 事件覆盖与
"宽度自适应比"（事件宽度 / 平稳宽度）——这是乘性缩放相对加成式的机制差异。

产物：results/v13_two_stage.json、figures/fig25_two_stage.png
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
from solar_diag import plots  # noqa: F401  字体 rcParams 副作用
from solar_diag.config import figures_dir, load_config, results_dir
from solar_diag.events import detect_ramps
from solar_diag.io import load_all
from solar_diag.models import train_lgbm
from solar_diag.pipeline import make_frames, prepare_plant
from solar_diag.scaled_intervals import (
    apply_scale,
    cqr_add_interval,
    group_scale_factors,
    normalized_residuals,
    scale_factor,
    two_stage_interval,
)
from solar_diag.smoothing import shrink_quantiles, group_sizes

NOMINAL = 0.80
RAMP_FRAC = 0.2
PAIRS = {"p05_95": (0.05, 0.95), "p10_90": (0.10, 0.90),
         "p15_85": (0.15, 0.85), "p20_80": (0.20, 0.80)}
IRR_BINS = [0.0, 0.2, 0.5, 0.8, 999.0]
N_MIN, K = 30, 20.0
METHODS = ["raw", "add", "add_group", "mult", "mult_group",
           "two_stage", "two_stage_group", "two_stage_shrink"]


def irr_groups(meta: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    g = np.asarray(ev.bucket_by_irradiation(meta["target_irr"], IRR_BINS)).astype(str)
    order = [f"[{IRR_BINS[i]:g},{IRR_BINS[i+1]:g})" for i in range(len(IRR_BINS) - 1)]
    return g, order


def assess(y, lo, hi, groups, order, ev_mask, cap) -> dict:
    e, q = ev_mask, ~ev_mask
    by = cf.group_coverage(y, lo, hi, groups)
    w_e = ev.interval_width(lo[e], hi[e]) if e.any() else 0.0
    w_q = ev.interval_width(lo[q], hi[q]) if q.any() else 0.0
    return {
        "coverage": ev.interval_coverage(y, lo, hi),
        "width_w": ev.interval_width(lo, hi),
        "width_pct_cap": 100.0 * ev.interval_width(lo, hi) / cap,
        "spread_irr": cf.coverage_spread(by),
        "by_irr": {k: v for k, v in by.items()},
        "coverage_ramp": ev.interval_coverage(y[e], lo[e], hi[e]) if e.any() else None,
        "coverage_quiet": ev.interval_coverage(y[q], lo[q], hi[q]) if q.any() else None,
        "width_ramp": w_e,
        "width_quiet": w_q,
        "width_adapt_ratio": float(w_e / w_q) if w_q > 0 else None,
        "coverage_highirr": by.get(order[-1], {}).get("coverage"),
    }


def run_plant(name, plant, cfg) -> dict:
    pl, cap = prepare_plant(plant, cfg)
    splits = make_frames(pl, cfg, "weather")
    x_tr, y_tr, _ = splits["train"]
    x_va, y_va, m_va = splits["val"]
    x_te, y_te, m_te = splits["test"]
    y_va_a, y_te_a = y_va.to_numpy(), y_te.to_numpy()

    preds_va, preds_te = {}, {}
    for pname, (qlo, qhi) in PAIRS.items():
        for tag, q in (("lo", qlo), ("hi", qhi)):
            m = train_lgbm(x_tr, y_tr, x_va, y_va,
                           {**cfg["lgbm"], "objective": "quantile", "alpha": q})
            preds_va[(pname, tag)] = m.predict(x_va)
            preds_te[(pname, tag)] = m.predict(x_te)
    print(f"  [{name}] 8 个分位数模型训练完成")

    g_va, order = irr_groups(m_va)
    g_te, _ = irr_groups(m_te)
    ramp = detect_ramps(m_te["target_dt"], y_te_a, cap, frac=RAMP_FRAC)
    ev_mask = np.zeros(len(y_te_a), dtype=bool)
    ev_mask[ramp["index"].to_numpy()] = True

    cal = {p: (preds_va[(p, "lo")], preds_va[(p, "hi")], ql, qh)
           for p, (ql, qh) in PAIRS.items()}
    lo_by = {p: preds_te[(p, "lo")] for p in PAIRS}
    hi_by = {p: preds_te[(p, "hi")] for p in PAIRS}

    lo0, hi0 = lo_by["p10_90"], hi_by["p10_90"]
    lo0v, hi0v = preds_va[("p10_90", "lo")], preds_va[("p10_90", "hi")]

    intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    params: dict = {}

    intervals["raw"] = (lo0, hi0)

    lo_a, hi_a, q_hat = cqr_add_interval(y_va_a, lo0v, hi0v, lo0, hi0, NOMINAL)
    intervals["add"] = (lo_a, hi_a)
    params["add"] = {"q_hat_w": q_hat}

    qg = cf.group_quantiles(y_va_a, lo0v, hi0v, g_va, NOMINAL)
    intervals["add_group"] = cf.apply_offsets(lo0, hi0, g_te, qg)
    params["add_group"] = {"q_map": {k: float(v) for k, v in qg.items()}}

    lam = scale_factor(y_va_a, lo0v, hi0v, NOMINAL)
    intervals["mult"] = apply_scale(lo0, hi0, lam)
    params["mult"] = {"lambda": lam}

    lam_map = group_scale_factors(y_va_a, lo0v, hi0v, g_va, NOMINAL, n_min=N_MIN)
    intervals["mult_group"] = apply_scale(lo0, hi0, 1.0, g_te, lam_map)
    params["mult_group"] = {"lambda_map": {k: float(v) for k, v in lam_map.items()}}

    lo_ts, hi_ts, p_ts = two_stage_interval(y_va_a, cal, lo_by, hi_by, NOMINAL)
    intervals["two_stage"] = (lo_ts, hi_ts)
    params["two_stage"] = {
        "selected": p_ts["selected"], "lambda": p_ts["lambda"],
        "pinball_table": {k: {"sharpness": float(v["sharpness"]),
                              "coverage": float(v["coverage"])}
                          for k, v in p_ts["pinball_table"].items()},
    }

    lo_tsg, hi_tsg, p_tsg = two_stage_interval(
        y_va_a, cal, lo_by, hi_by, NOMINAL,
        groups_cal=g_va, groups_te=g_te, grouped=True, n_min=N_MIN)
    intervals["two_stage_group"] = (lo_tsg, hi_tsg)
    params["two_stage_group"] = {
        "selected": p_tsg["selected"],
        "lambda_map": p_tsg["lambda_map"],
    }

    # 两实验联动：分桶 λ 也按最小样本量软收缩（E29 的 shrink 思路）
    n_map = group_sizes(g_va)
    sel = p_tsg["selected"]
    lam_global_for_pair = scale_factor(y_va_a, cal[sel][0], cal[sel][1], NOMINAL)
    lam_smooth = shrink_quantiles(p_tsg["lambda_map"], n_map,
                                  lam_global_for_pair, k=K)
    intervals["two_stage_shrink"] = apply_scale(
        lo_by[sel], hi_by[sel], 1.0, g_te, lam_smooth)
    params["two_stage_shrink"] = {
        "selected": sel, "lambda_global": lam_global_for_pair,
        "lambda_map": {k: float(v) for k, v in lam_smooth.items()},
    }

    # ---- 机制诊断：夜间零功率样本在 |r|=1 / CQR=0 处的点质量是否钉死全局校准 ----
    r_abs = np.abs(normalized_residuals(y_va_a, lo0v, hi0v))
    s_cqr = cf.cqr_scores(y_va_a, lo0v, hi0v)
    day_va = (m_va["target_irr"] > cfg["daytime_irr_min"]).to_numpy()
    day_te = (m_te["target_irr"] > cfg["daytime_irr_min"]).to_numpy()
    lam_day = scale_factor(y_va_a[day_va], lo0v[day_va], hi0v[day_va], NOMINAL)
    q_day = cf.conformal_quantile(s_cqr[day_va], NOMINAL)
    lo_md, hi_md = apply_scale(lo0, hi0, lam_day)
    lo_ad, hi_ad = lo0 - q_day, hi0 + q_day
    diag = {
        "frac_abs_r_eq_1": float(np.mean(np.isclose(r_abs, 1.0, atol=1e-9))),
        "frac_cqr_score_eq_0": float(np.mean(np.isclose(s_cqr, 0.0, atol=1e-9))),
        "frac_night_cal": float(np.mean(~day_va)),
        "frac_night_test": float(np.mean(~day_te)),
        "lambda_day_only": lam_day,
        "q_hat_day_only_w": q_day,
        "daytime_coverage_raw": ev.interval_coverage(y_te_a[day_te], lo0[day_te], hi0[day_te]),
        "daytime_coverage_mult_day": ev.interval_coverage(y_te_a[day_te], lo_md[day_te], hi_md[day_te]),
        "daytime_coverage_add_day": ev.interval_coverage(y_te_a[day_te], lo_ad[day_te], hi_ad[day_te]),
        "daytime_width_raw": ev.interval_width(lo0[day_te], hi0[day_te]),
        "daytime_width_mult_day": ev.interval_width(lo_md[day_te], hi_md[day_te]),
    }

    res = {
        "capacity_w": cap, "nominal": NOMINAL, "n_cal": int(len(y_va_a)),
        "n_test": int(len(y_te_a)), "n_ramp": int(ev_mask.sum()),
        "ramp_share": float(ev_mask.mean()), "ramp_frac": RAMP_FRAC,
        "order": order, "params": params, "diagnostics": diag, "methods": {},
    }
    for mname, (lo, hi) in intervals.items():
        e = assess(y_te_a, lo, hi, g_te, order, ev_mask, cap)
        res["methods"][mname] = e
        print(f"  [{name}/{mname:17s}] cov={e['coverage']:.3f} width={e['width_w']:.0f}W "
              f"spread={e['spread_irr']:.3f} ramp_cov={e['coverage_ramp']:.3f} "
              f"adapt={e['width_adapt_ratio']:.2f}")
    return res


def draw_figure(out: dict, cfg, fig_dir) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#888888", "#1f77b4", "#9467bd", "#ff7f0e", "#8c564b",
              "#2ca02c", "#d62728", "#17becf"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.4))
    for row, plant in enumerate(cfg["plants"]):
        r = out[plant]
        ax = axes[row, 0]
        xs = np.arange(len(METHODS))
        ev_cov = [r["methods"][m]["coverage_ramp"] for m in METHODS]
        qt_cov = [r["methods"][m]["coverage_quiet"] for m in METHODS]
        ax.bar(xs - 0.19, ev_cov, 0.38, label="云跳变事件", color="#d62728")
        ax.bar(xs + 0.19, qt_cov, 0.38, label="平稳段", color="#1f77b4")
        ax.axhline(NOMINAL, color="k", ls="--", lw=0.9)
        ax.set_xticks(xs)
        ax.set_xticklabels(METHODS, rotation=30, ha="right", fontsize=7)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("覆盖率")
        ax.set_title(f"{plant} 事件/平稳覆盖（{r['n_ramp']} 个 ramp）", fontsize=10)
        if row == 0:
            ax.legend(fontsize=8)

        ax = axes[row, 1]
        for j, m in enumerate(METHODS):
            e = r["methods"][m]
            ax.scatter(e["width_w"], e["coverage"], s=55, color=colors[j], label=m)
            ax.annotate(m, (e["width_w"], e["coverage"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=6.5)
        ax.axhline(NOMINAL, color="k", ls="--", lw=0.9)
        ax.set_xlabel("平均区间宽度 (W)")
        ax.set_ylabel("总覆盖率")
        ax.set_title(f"{plant} 宽度—覆盖权衡", fontsize=10)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    p = fig_dir / "fig25_two_stage.png"
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
        out[name] = run_plant(name, plants[name], cfg)
    with open(res_dir / "v13_two_stage.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    draw_figure(out, cfg, fig_dir)
    print("结果 →", res_dir / "v13_two_stage.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
