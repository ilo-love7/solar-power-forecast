"""09 条件保形：全局 CQR vs Mondrian 分桶校准，压平分桶覆盖极差。

v1.1（08）的全局 CQR 总覆盖 0.88 但桶间极差大（强光桶 0.455 / 夜间 0.965）。
本脚本对同一组 q10/q50/q90 分位数模型比较五种区间口径：
  raw              原始 q10–q90
  global           全局 CQR（= v1.1）
  mondrian_irr     按辐照桶独立校准
  mondrian_day     按白天/夜间独立校准
  mondrian_combo   按辐照桶×昼夜独立校准
评价：总覆盖、平均宽度、分桶覆盖、极差（均匀化目标）。
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
from solar_diag.io import load_all
from solar_diag.models import train_lgbm
from solar_diag.pipeline import make_frames, prepare_plant

QUANTILES = {"q10": 0.1, "q50": 0.5, "q90": 0.9}
NOMINAL = 0.80
IRR_BINS = [0.0, 0.2, 0.5, 0.8, 999.0]


def build_groups(meta: pd.DataFrame) -> dict[str, pd.Series]:
    irr_g = ev.bucket_by_irradiation(meta["target_irr"], IRR_BINS).astype(str)
    day_g = pd.Series(
        np.where((meta["target_hour"] >= 6) & (meta["target_hour"] < 18), "day", "night"),
        index=meta.index,
    )
    combo_g = (irr_g + "|" + day_g).astype(str)
    return {"irr": irr_g, "day": day_g, "combo": combo_g}


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
        y_va_a, y_te_a = y_va.to_numpy(), y_te.to_numpy()

        preds, preds_va = {}, {}
        for name, q in QUANTILES.items():
            m = train_lgbm(x_tr, y_tr, x_va, y_va,
                           {**cfg["lgbm"], "objective": "quantile", "alpha": q})
            preds[name] = m.predict(x_te)
            preds_va[name] = m.predict(x_va)

        lo_raw, hi_raw = preds["q10"], preds["q90"]
        lo_va, hi_va = preds_va["q10"], preds_va["q90"]

        groups_te = build_groups(m_te)
        groups_va = build_groups(splits["val"][2])

        methods: dict[str, tuple[np.ndarray, np.ndarray]] = {"raw": (lo_raw, hi_raw)}

        q_glob = cf.conformal_quantile(cf.cqr_scores(y_va_a, lo_va, hi_va), NOMINAL)
        methods["global"] = (lo_raw - q_glob, hi_raw + q_glob)

        for mname, gname in [("mondrian_irr", "irr"), ("mondrian_day", "day"),
                             ("mondrian_combo", "combo")]:
            qg = cf.group_quantiles(y_va_a, lo_va, hi_va, groups_va[gname], NOMINAL)
            methods[mname] = cf.apply_offsets(lo_raw, hi_raw, groups_te[gname], qg)

        pres: dict = {"capacity_w": cap, "nominal": NOMINAL,
                      "global_q_hat_w": q_glob, "methods": {}}
        for mname, (lo_m, hi_m) in methods.items():
            entry = {
                "coverage": ev.interval_coverage(y_te_a, lo_m, hi_m),
                "width_w": ev.interval_width(lo_m, hi_m),
                "by_irr": cf.group_coverage(y_te_a, lo_m, hi_m, groups_te["irr"]),
                "by_day": cf.group_coverage(y_te_a, lo_m, hi_m, groups_te["day"]),
            }
            entry["spread_irr"] = cf.coverage_spread(entry["by_irr"])
            entry["spread_day"] = cf.coverage_spread(entry["by_day"])
            pres["methods"][mname] = entry
            print(f"{plant} {mname:16s} cov={entry['coverage']:.3f} "
                  f"width={entry['width_w']:.0f}W "
                  f"spread_irr={entry['spread_irr']:.3f} spread_day={entry['spread_day']:.3f}")
        out[plant] = pres

    with open(res_dir / "conditional_conformal.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # 图：分辐照桶覆盖（各方法）
    import matplotlib
    matplotlib.use("Agg")
    import solar_diag.plots  # noqa: F401  字体 rcParams 副作用
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    methods_show = ["raw", "global", "mondrian_irr", "mondrian_combo"]
    colors = ["#888", "#1f77b4", "#d62728", "#2ca02c"]
    for ax, plant in zip(axes, cfg["plants"]):
        buckets = list(out[plant]["methods"]["raw"]["by_irr"].keys())
        xs = np.arange(len(buckets))
        for k, mname in enumerate(methods_show):
            covs = [out[plant]["methods"][mname]["by_irr"][b]["coverage"] for b in buckets]
            ax.bar(xs + (k - 1.5) * 0.2, covs, 0.2, label=mname, color=colors[k])
        ax.axhline(NOMINAL, color="k", ls="--", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(buckets, rotation=20, fontsize=8)
        ax.set_title(plant)
        ax.set_ylabel("覆盖率")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig21_conditional_conformal.png", dpi=160, bbox_inches="tight")
    print("图 →", fig_dir / "fig21_conditional_conformal.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
