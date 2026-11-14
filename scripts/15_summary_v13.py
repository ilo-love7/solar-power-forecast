"""15（v1.3）汇总三个实验 JSON → results/v13_summary_tables.md。

报告 §3.10–§3.12 与摘要直接引用本文件生成的表格，避免手抄数字出错。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar_diag.config import load_config, results_dir

MODES = ["global", "raw", "min_n", "shrink", "pool", "pool_shrink"]
E30_METHODS = ["raw", "add", "add_group", "mult", "mult_group",
               "two_stage", "two_stage_group", "two_stage_shrink"]
MODELS = ["LightGBM", "lstm", "tr_small", "tr_deep", "tr_small_shuf"]


def f(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def main() -> int:
    cfg = load_config()
    rd = results_dir()
    gs = json.loads((rd / "v13_group_smoothing.json").read_text(encoding="utf-8"))
    ts = json.loads((rd / "v13_two_stage.json").read_text(encoding="utf-8"))
    tr = json.loads((rd / "v13_transformer.json").read_text(encoding="utf-8"))
    plants = cfg["plants"]
    lines: list[str] = ["# N3 v1.3 汇总表（脚本 15 自动生成，勿手改）", ""]

    # ---------- E29 ----------
    lines += ["## 表 A｜E29 分组共形平滑（名义覆盖 0.80）", ""]
    lines += ["| 站 | 方法 | 总覆盖 | 宽度 W | 桶间极差 | 稀疏桶覆盖 | 稀疏桶偏差 | q̂ bootstrap std (W) | 重抽稀疏桶覆盖 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for p in plants:
        r = gs[p]
        sparse = r["sparse_groups"][0] if r["sparse_groups"] else r["order"][-1]
        for m in MODES:
            e = r["methods"][m]
            qb = r["q_bootstrap"].get(m, {}).get(sparse, {}).get("std")
            cb = r["coverage_bootstrap"].get(m, {}).get("sparse", {})
            cbtxt = (f"{cb['mean']:.3f}±{cb['std']:.3f}" if cb else "—")
            lines.append(f"| {p} | {m} | {f(e['coverage'])} | {f(e['width_w'],0)} | "
                         f"{f(e['spread'])} | {f(e['sparse_coverage'])} | "
                         f"{f(e['sparse_abs_bias'])} | {f(qb,0)} | {cbtxt} |")
    lines += ["", f"稀疏桶 = 校准 n<{gs[plants[0]]['n_min']} 的辐照桶（P1/P2 均为强光桶 "
              f"n={gs[plants[0]]['n_map'][gs[plants[0]]['order'][-1]]}/"
              f"{gs[plants[1]]['n_map'][gs[plants[1]]['order'][-1]]}）。", ""]

    lines += ["## 表 B｜E29 超参敏感性", "", "| 站 | 超参 | 取值 | 稀疏桶覆盖 | 偏差 | 桶间极差 |",
              "|---|---|---|---|---|---|"]
    for p in plants:
        for group, keys in (("shrink_k", ["5.0", "10.0", "20.0", "40.0", "80.0"]),
                            ("pool_bandwidth", ["0.5", "0.7", "1.0", "1.5", "2.0"]),
                            ("min_n", ["10", "20", "30", "50", "80"])):
            for k in keys:
                v = gs[p]["sensitivity"][group][k]
                lines.append(f"| {p} | {group} | {k} | {f(v['sparse_coverage'])} | "
                             f"{f(v['sparse_abs_bias'])} | {f(v['spread'])} |")
    lines += ["", "## 表 B2｜E29 val/test 组内 q̂ 差异（分布偏移证据）", "",
              "| 站 | 辐照桶 | q̂_val (W) | q̂_test (W) | 绝对差 (W) | 测试 n |",
              "|---|---|---|---|---|---|"]
    for p in plants:
        for k, v in gs[p]["val_test_q_gap"].items():
            lines.append(f"| {p} | {k} | {f(v['q_val'],0)} | {f(v['q_test'],0)} | "
                         f"{f(v['abs_gap_w'],0)} | {v['n_test']} |")
    lines.append("")

    # ---------- E30 ----------
    lines += ["## 表 C｜E30 二段缩放（8 口径）", ""]
    lines += ["| 站 | 方法 | 总覆盖 | 宽度 W | 桶间极差 | ramp 覆盖 | 平稳覆盖 | 宽度自适应比 | 强光桶覆盖 | 事件覆盖效率 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for p in plants:
        r = ts[p]
        raw = r["methods"]["raw"]
        for m in E30_METHODS:
            e = r["methods"][m]
            dw_pct = 100 * (e["width_w"] - raw["width_w"]) / raw["width_w"]
            dcov_pp = 100 * (e["coverage_ramp"] - raw["coverage_ramp"])
            eff = dcov_pp / dw_pct if abs(dw_pct) > 0.05 else None
            lines.append(f"| {p} | {m} | {f(e['coverage'])} | {f(e['width_w'],0)} | "
                         f"{f(e['spread_irr'])} | {f(e['coverage_ramp'])} | "
                         f"{f(e['coverage_quiet'])} | {f(e['width_adapt_ratio'],2)} | "
                         f"{f(e['coverage_highirr'])} | {f(eff,2)} |")
    lines += ["", "事件覆盖效率 = Δramp覆盖(百分点) / Δ平均宽度(%)，越大越划算。", "",
              "## 表 D｜E30 机制诊断：夜间点质量与白天专用校准", "",
              "| 站 | \\|r\\|=1 占比 | CQR=0 占比 | 校准夜间占比 | 全局 λ | 白天专用 λ | 白天 q̂ (W) | 白天覆盖 raw→乘性/加法 |",
              "|---|---|---|---|---|---|---|---|"]
    for p in plants:
        d = ts[p]["diagnostics"]
        lines.append(f"| {p} | {f(d['frac_abs_r_eq_1'])} | {f(d['frac_cqr_score_eq_0'])} | "
                     f"{f(d['frac_night_cal'])} | {f(ts[p]['params']['mult']['lambda'],3)} | "
                     f"{f(d['lambda_day_only'],3)} | {f(d['q_hat_day_only_w'],0)} | "
                     f"{f(d['daytime_coverage_raw'])}→{f(d['daytime_coverage_mult_day'])}/"
                     f"{f(d['daytime_coverage_add_day'])} |")
    lines += ["", "## 表 D2｜pinball 第一段候选评分（验证集）", "",
              "| 站 | 候选分位对 | pinball 锐度 | 原始覆盖 |", "|---|---|---|---|"]
    for p in plants:
        for k, v in ts[p]["params"]["two_stage"]["pinball_table"].items():
            lines.append(f"| {p} | {k} | {f(v['sharpness'],1)} | {f(v['coverage'])} |")
    lines.append("")

    # ---------- E31 ----------
    lines += ["## 表 E｜E31 模型对比（weather 口径，3 种子集成）", ""]
    lines += ["| 站 | 提前量 | 模型 | nRMSE % | RMSE W | R² | 种子 RMSE std W | 参数量 | 单种子训练 s | 平均 epoch |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for p in plants:
        for h in ("4", "16"):
            r = tr[p]["horizons"][h]
            for m in MODELS:
                e = r["models"][m]
                a = e["all"]
                np_txt = "—" if e["n_params"] is None else f"{e['n_params']/1000:.1f}k"
                lines.append(f"| {p} | {15*int(h)}min | {m} | {f(a['nrmse_pct'],2)} | "
                             f"{f(a['rmse'],0)} | {f(a['r2'])} | {f(e['seed_rmse_std'],0)} | "
                             f"{np_txt} | {f(e['train_s_per_seed'],1)} | "
                             f"{f(e['mean_epochs'],1)} |")
    lines += ["", "## 表 F｜E31 关键差值（相对 LightGBM 与 LSTM）", "",
              "| 站 | 提前量 | 模型 | ΔnRMSE vs LightGBM (pp) | ΔnRMSE vs LSTM (pp) | 时序打乱 Δ (pp) |",
              "|---|---|---|---|---|---|"]
    for p in plants:
        for h in ("4", "16"):
            r = tr[p]["horizons"][h]["models"]
            base, lstm = r["LightGBM"]["all"]["nrmse_pct"], r["lstm"]["all"]["nrmse_pct"]
            for m in MODELS[1:]:
                v = r[m]["all"]["nrmse_pct"]
                shuf = (v - r["tr_small"]["all"]["nrmse_pct"]) if m == "tr_small_shuf" else None
                lines.append(f"| {p} | {15*int(h)}min | {m} | {f(v-base,2)} | {f(v-lstm,2)} | "
                             f"{f(shuf,2)} |")
    lines.append("")

    out = rd / "v13_summary_tables.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
