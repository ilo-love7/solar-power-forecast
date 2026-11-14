"""12（v1.3 E29）分组共形的最小样本量平滑。

v1.2 §3.7 结论 3：Mondrian 组合分组最平，但稀疏桶（P1 强光桶校准 n≈22）
的 q̂ 噪声大、覆盖过冲到 1.0，"极端桶宜向邻桶借强度（本次未做）"。

本脚本在**同一组** q10/q90 分位数模型上比较 6 种 q̂ 生成方式：
  global      全组用全局 q̂（放弃条件覆盖）
  raw         v1.2 的组内独立校准（基线）
  min_n       n_g < n_min 的组回退全局 q̂（硬阈值）
  shrink      w=n/(n+k) 软收缩到全局 q̂
  pool        有序相邻桶的校准分数按高斯核加权池化（借样本）
  pool_shrink 池化后再按有效样本量收缩

评价不只看单次测试覆盖，还做两重稳健性诊断：
  A. 校准集 bootstrap（B=120）→ 总覆盖/稀疏桶覆盖的均值与标准差；
  B. val 与 test 的组内 q̂ 差异 → 区分"抽样噪声"与"分布偏移"。
产物：results/v13_group_smoothing.json、figures/fig24_group_smoothing.png
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
from solar_diag.io import load_all
from solar_diag.models import train_lgbm
from solar_diag.pipeline import make_frames, prepare_plant
from solar_diag.smoothing import (
    bootstrap_q_variance,
    oracle_group_quantiles,
    smoothed_quantiles,
)

NOMINAL = 0.80
QUANTILES = {"q10": 0.1, "q90": 0.9}
IRR_BINS = [0.0, 0.2, 0.5, 0.8, 999.0]
MODES = ["global", "raw", "min_n", "shrink", "pool", "pool_shrink"]
BOOT_MODES = ["raw", "min_n", "shrink", "pool", "pool_shrink"]
N_MIN, K, BANDWIDTH = 30, 20.0, 1.0
N_BOOT = 120


def irr_order() -> list[str]:
    return [f"[{IRR_BINS[i]:g},{IRR_BINS[i+1]:g})" for i in range(len(IRR_BINS) - 1)]


def make_groups(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    irr = ev.bucket_by_irradiation(meta["target_irr"], IRR_BINS)
    irr_s = np.asarray(irr).astype(str)
    day = np.where((meta["target_hour"] >= 6) & (meta["target_hour"] < 18), "day", "night")
    return {"irr": irr_s, "combo": np.char.add(np.char.add(irr_s, "|"), day.astype(str))}


def combo_order() -> list[str]:
    return [f"{b}|{d}" for b in irr_order() for d in ("day", "night")]


def evaluate(y, lo, hi, groups, order, sparse_names) -> dict:
    by = cf.group_coverage(y, lo, hi, groups)
    sparse_cov = [by[n]["coverage"] for n in sparse_names if n in by]
    return {
        "coverage": ev.interval_coverage(y, lo, hi),
        "width_w": ev.interval_width(lo, hi),
        "by_group": {k: v for k, v in by.items()},
        "spread": cf.coverage_spread({k: by[k] for k in order if k in by}),
        "sparse_abs_bias": float(np.mean(np.abs(np.asarray(sparse_cov) - NOMINAL))) if sparse_cov else 0.0,
        "sparse_coverage": float(np.mean(sparse_cov)) if sparse_cov else None,
    }


def run_plant(name, plant, cfg, fig_dir, res_dir) -> dict:
    pl, cap = prepare_plant(plant, cfg)
    splits = make_frames(pl, cfg, "weather")
    x_tr, y_tr, _ = splits["train"]
    x_va, y_va, m_va = splits["val"]
    x_te, y_te, m_te = splits["test"]
    y_va_a, y_te_a = y_va.to_numpy(), y_te.to_numpy()

    preds, preds_va = {}, {}
    for qname, q in QUANTILES.items():
        m = train_lgbm(x_tr, y_tr, x_va, y_va,
                       {**cfg["lgbm"], "objective": "quantile", "alpha": q})
        preds[qname] = m.predict(x_te)
        preds_va[qname] = m.predict(x_va)
    lo_raw, hi_raw = preds["q10"], preds["q90"]
    lo_va, hi_va = preds_va["q10"], preds_va["q90"]
    s_va = cf.cqr_scores(y_va_a, lo_va, hi_va)

    g_va, g_te = make_groups(m_va), make_groups(m_te)
    order = irr_order()
    n_map = {k: int((g_va["irr"] == k).sum()) for k in order}
    sparse = [k for k in order if n_map[k] < N_MIN]

    res: dict = {
        "capacity_w": cap, "nominal": NOMINAL, "n_cal": int(len(s_va)),
        "n_test": int(len(y_te_a)), "n_min": N_MIN, "k": K,
        "bandwidth": BANDWIDTH, "order": order, "n_map": n_map,
        "sparse_groups": sparse, "methods": {},
    }

    kw = dict(n_min=N_MIN, k=K, bandwidth=BANDWIDTH)
    for mode in MODES:
        q_map, diag = smoothed_quantiles(s_va, g_va["irr"], NOMINAL,
                                         order=order, mode=mode, **kw)
        lo_m, hi_m = cf.apply_offsets(lo_raw, hi_raw, g_te["irr"], q_map)
        entry = evaluate(y_te_a, lo_m, hi_m, g_te["irr"], order, sparse)
        entry["q_map"] = {k: float(v) for k, v in q_map.items()}
        res["methods"][mode] = entry
        print(f"  [{name}/{mode:11s}] cov={entry['coverage']:.3f} "
              f"width={entry['width_w']:.0f}W spread={entry['spread']:.3f} "
              f"sparse_bias={entry['sparse_abs_bias']:.3f}")

    # ---- 诊断 A：q̂ 的 bootstrap 抽样标准差 ----
    res["q_bootstrap"] = {}
    for mode in BOOT_MODES:
        res["q_bootstrap"][mode] = bootstrap_q_variance(
            s_va, g_va["irr"], NOMINAL, mode=mode, order=order,
            n_boot=300, seed=0, **kw)

    # ---- 诊断 B：校准集 bootstrap 下的测试覆盖分布（部署视角） ----
    rng = np.random.default_rng(1234)
    n = len(s_va)
    boot_cov: dict[str, dict[str, list[float]]] = {
        m: {"total": [], "sparse": []} for m in BOOT_MODES
    }
    for _ in range(N_BOOT):
        pick = rng.choice(n, size=n, replace=True)
        sb, gb = s_va[pick], g_va["irr"][pick]
        for mode in BOOT_MODES:
            q_map, _ = smoothed_quantiles(sb, gb, NOMINAL, order=order, mode=mode, **kw)
            lo_b, hi_b = cf.apply_offsets(lo_raw, hi_raw, g_te["irr"], q_map)
            boot_cov[mode]["total"].append(ev.interval_coverage(y_te_a, lo_b, hi_b))
            by = cf.group_coverage(y_te_a, lo_b, hi_b, g_te["irr"])
            vals = [by[k]["coverage"] for k in sparse if k in by]
            boot_cov[mode]["sparse"].append(float(np.mean(vals)) if vals else 0.0)
    res["coverage_bootstrap"] = {
        m: {target: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for target, v in d.items()}
        for m, d in boot_cov.items()
    }

    # ---- 诊断 C：val vs test 组内 q̂（分布偏移证据） ----
    q_val, _ = smoothed_quantiles(s_va, g_va["irr"], NOMINAL,
                                  order=order, mode="raw", **kw)
    q_test = oracle_group_quantiles(y_te_a, lo_raw, hi_raw, g_te["irr"], NOMINAL, order)
    res["val_test_q_gap"] = {
        k: {"q_val": float(q_val[k]), "q_test": float(q_test[k]),
            "abs_gap_w": float(abs(q_val[k] - q_test[k])),
            "n_test": int((g_te["irr"] == k).sum())}
        for k in order
    }

    # ---- 诊断 D：超参敏感性（证明结论不依赖 k / n_min / bandwidth 的挑选）----
    sens: dict = {"shrink_k": {}, "pool_bandwidth": {}, "min_n": {}}
    for k in (5.0, 10.0, 20.0, 40.0, 80.0):
        q_map, _ = smoothed_quantiles(s_va, g_va["irr"], NOMINAL, order=order,
                                      mode="shrink", n_min=N_MIN, k=k,
                                      bandwidth=BANDWIDTH)
        lo_s, hi_s = cf.apply_offsets(lo_raw, hi_raw, g_te["irr"], q_map)
        e = evaluate(y_te_a, lo_s, hi_s, g_te["irr"], order, sparse)
        sens["shrink_k"][str(k)] = {
            "sparse_coverage": e["sparse_coverage"],
            "sparse_abs_bias": e["sparse_abs_bias"],
            "spread": e["spread"], "width_w": e["width_w"],
        }
    for bw in (0.5, 0.7, 1.0, 1.5, 2.0):
        q_map, _ = smoothed_quantiles(s_va, g_va["irr"], NOMINAL, order=order,
                                      mode="pool", n_min=N_MIN, k=K,
                                      bandwidth=bw)
        lo_s, hi_s = cf.apply_offsets(lo_raw, hi_raw, g_te["irr"], q_map)
        e = evaluate(y_te_a, lo_s, hi_s, g_te["irr"], order, sparse)
        sens["pool_bandwidth"][str(bw)] = {
            "sparse_coverage": e["sparse_coverage"],
            "sparse_abs_bias": e["sparse_abs_bias"],
            "spread": e["spread"],
        }
    for nm in (10, 20, 30, 50, 80):
        q_map, _ = smoothed_quantiles(s_va, g_va["irr"], NOMINAL, order=order,
                                      mode="min_n", n_min=nm, k=K,
                                      bandwidth=BANDWIDTH)
        lo_s, hi_s = cf.apply_offsets(lo_raw, hi_raw, g_te["irr"], q_map)
        e = evaluate(y_te_a, lo_s, hi_s, g_te["irr"], order, sparse)
        sens["min_n"][str(nm)] = {
            "sparse_coverage": e["sparse_coverage"],
            "sparse_abs_bias": e["sparse_abs_bias"],
            "spread": e["spread"],
        }
    res["sensitivity"] = sens

    # ---- 附加：组合分组（更细粒度、稀疏桶更多）----
    corder = combo_order()
    cn_map = {k: int((g_va["combo"] == k).sum()) for k in corder}
    csparse = [k for k in corder if cn_map[k] < N_MIN]
    res["combo"] = {"n_map": cn_map, "sparse_groups": csparse, "methods": {}}
    for mode in ("raw", "pool_shrink"):
        q_map, _ = smoothed_quantiles(s_va, g_va["combo"], NOMINAL,
                                      order=corder, mode=mode, **kw)
        lo_c, hi_c = cf.apply_offsets(lo_raw, hi_raw, g_te["combo"], q_map)
        res["combo"]["methods"][mode] = evaluate(
            y_te_a, lo_c, hi_c, g_te["combo"], corder, csparse)
    return res


def draw_figure(out: dict, cfg, fig_dir) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plants = cfg["plants"]
    order = irr_order()
    show = ["raw", "min_n", "shrink", "pool", "pool_shrink"]
    colors = ["#888888", "#ff7f0e", "#1f77b4", "#2ca02c", "#d62728"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.2))
    for row, plant in enumerate(plants):
        r = out[plant]
        ax = axes[row, 0]
        xs = np.arange(len(order))
        for j, mode in enumerate(show):
            covs = [r["methods"][mode]["by_group"].get(b, {}).get("coverage", np.nan)
                    for b in order]
            ax.bar(xs + (j - 2) * 0.16, covs, 0.16, label=mode, color=colors[j])
        ax.axhline(NOMINAL, color="k", ls="--", lw=0.9)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{b}\nn={r['n_map'][b]}" for b in order], fontsize=7)
        ax.set_ylabel("测试覆盖率")
        ax.set_title(f"{plant} 分桶覆盖（名义 {NOMINAL:.0%}）", fontsize=10)
        ax.set_ylim(0, 1.15)
        if row == 0:
            ax.legend(fontsize=7, ncol=3)

        ax = axes[row, 1]
        means = [r["coverage_bootstrap"][m]["sparse"]["mean"] for m in BOOT_MODES]
        stds = [r["coverage_bootstrap"][m]["sparse"]["std"] for m in BOOT_MODES]
        ax.bar(np.arange(len(BOOT_MODES)), means, 0.6, yerr=stds,
               capsize=4, color=[colors[show.index(m)] if m in show else "#888"
                                 for m in BOOT_MODES])
        ax.axhline(NOMINAL, color="k", ls="--", lw=0.9)
        ax.set_xticks(np.arange(len(BOOT_MODES)))
        ax.set_xticklabels(BOOT_MODES, rotation=15, fontsize=8)
        ax.set_ylabel("稀疏桶覆盖（120 次重抽）")
        ax.set_title(f"{plant} 校准集重抽下的稀疏桶覆盖", fontsize=10)
        ax.set_ylim(0, 1.15)
    fig.tight_layout()
    p = fig_dir / "fig24_group_smoothing.png"
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
        out[name] = run_plant(name, plants[name], cfg, fig_dir, res_dir)
    with open(res_dir / "v13_group_smoothing.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    draw_figure(out, cfg, fig_dir)
    print("结果 →", res_dir / "v13_group_smoothing.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
