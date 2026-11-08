"""逆变器损失诊断：完整性、白天停机、转换效率、静态损耗、相对出力。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def robust_z(values: pd.Series, k: float = 1.4826) -> pd.Series:
    """MAD 稳健 z 分数（对离群比均值/方差稳定）。"""
    med = values.median()
    mad = (values - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return pd.Series(0.0, index=values.index)
    return (values - med) / (k * mad)


def completeness(gen_inv: pd.DataFrame, slots_expected: int) -> pd.DataFrame:
    """每台逆变器槽位完整率与最长连续缺失（槽）。"""
    rows = []
    for inv, d in gen_inv.groupby("SOURCE_KEY"):
        n = d["dt"].nunique()
        full = pd.date_range(d["dt"].min(), d["dt"].max(), freq="15min")
        have = pd.DatetimeIndex(d["dt"].unique())
        missing = full.difference(have)
        max_gap = 0
        if len(missing):
            s = pd.Series(1, index=missing.sort_values())
            grp = (s.index.to_series().diff().dt.total_seconds().fillna(0) > 900).cumsum()
            max_gap = int(s.groupby(grp).size().max())
        rows.append({
            "SOURCE_KEY": inv,
            "n_slots": n,
            "completeness": n / slots_expected,
            "max_gap_slots": max_gap,
        })
    return pd.DataFrame(rows)


def daytime_outages(
    gen_inv: pd.DataFrame,
    irradiation: pd.Series,
    irr_min: float = 0.5,
    fleet_min: float = 200.0,
    self_frac: float = 0.05,
) -> pd.DataFrame:
    """白天相对停机：辐照充足、全队正常出力，但自身 < self_frac × 全队同刻中位。

    连续多个 15min 槽合并为一次停机事件（episode），返回事件数/总时长/最长事件。
    """
    g = gen_inv.copy()
    g["irradiation"] = g["dt"].map(irradiation)
    daytime = g[g["irradiation"] > irr_min]
    slot_med = daytime.groupby("dt")["AC_POWER"].median()
    # 同刻全队记 0 比例（≥0.5 视为站级事件/记录伪影，不算单机停机）
    zero_frac = daytime.assign(z=(daytime["AC_POWER"] == 0)).groupby("dt")["z"].mean()
    g["slot_med"] = g["dt"].map(slot_med)
    g["zero_frac"] = g["dt"].map(zero_frac)
    ev = g[(g["irradiation"] > irr_min) & (g["slot_med"] > fleet_min)
           & (g["AC_POWER"] < self_frac * g["slot_med"]) & (g["zero_frac"] < 0.5)]
    rows = []
    for inv, d in ev.groupby("SOURCE_KEY"):
        ts = d["dt"].sort_values()
        # 间隔 > 30min 视为新事件
        grp = ts.diff().dt.total_seconds().fillna(0).gt(1800).cumsum()
        sizes = ts.groupby(grp).size()
        rows.append({
            "SOURCE_KEY": inv,
            "outage_events": int(len(sizes)),
            "outage_slots": int(sizes.sum()),
            "outage_max_episode": int(sizes.max()),
        })
    return pd.DataFrame(rows)


def efficiency_table(gen_inv: pd.DataFrame, dc_day_min: float = 100.0) -> pd.DataFrame:
    """转换效率 AC/DC（白天 DC>dc_day_min）与线性静态损耗拟合 AC = a·DC + b。"""
    rows = []
    for inv, d in gen_inv.groupby("SOURCE_KEY"):
        day = d[d["DC_POWER"] > dc_day_min]
        if len(day) < 50:
            rows.append({"SOURCE_KEY": inv, "n_day": len(day), "eff_med": np.nan,
                         "eff_std": np.nan, "slope": np.nan, "intercept": np.nan})
            continue
        eff = day["AC_POWER"] / day["DC_POWER"]
        slope, intercept = np.polyfit(day["DC_POWER"], day["AC_POWER"], 1)
        rows.append({
            "SOURCE_KEY": inv,
            "n_day": len(day),
            "eff_med": float(eff.median()),
            "eff_std": float(eff.std()),
            "slope": float(slope),
            "intercept": float(intercept),
        })
    return pd.DataFrame(rows)


def relative_yield(gen_inv: pd.DataFrame, irradiation: pd.Series) -> pd.DataFrame:
    """相对日发电量：daily AC / (自身 AC p95 × 当日辐照均值)，再看相对全队的稳健 z。"""
    g = gen_inv.copy()
    g["irradiation"] = g["dt"].map(irradiation)
    g["date"] = g["dt"].dt.normalize()
    cap = g.groupby("SOURCE_KEY")["AC_POWER"].quantile(0.95).rename("cap")
    daily = g.groupby(["SOURCE_KEY", "date"]).agg(e=("AC_POWER", "sum"), irr=("irradiation", "mean")).reset_index()
    daily = daily.merge(cap, on="SOURCE_KEY")
    daily = daily[daily["irr"] > 0.3]  # 只看有效日照日
    daily["rel"] = daily["e"] / (daily["cap"] * daily["irr"])
    per = daily.groupby("SOURCE_KEY")["rel"].median().rename("rel_yield")
    return per.reset_index()


def diagnose_inverters(
    gen_inv: pd.DataFrame,
    irradiation: pd.Series,
    slots_expected: int = 3264,
    dc_day_min: float = 100.0,
    irr_min: float = 0.5,
    fleet_min: float = 200.0,
    self_frac: float = 0.05,
    mad_z_flag: float = 3.5,
    completeness_min: float = 0.90,
    outage_slots_min: int = 24,
) -> pd.DataFrame:
    """汇总各诊断维度 → 每台逆变器一行，含 flag 与 reasons。

    告警规则：完整率 < completeness_min；停机总槽位 >= outage_slots_min（约 6h）；
    效率 / 静态损耗 / 相对出力的 |MAD z| > mad_z_flag。
    """
    comp = completeness(gen_inv, slots_expected)
    outg = daytime_outages(gen_inv, irradiation, irr_min, fleet_min, self_frac)
    eff = efficiency_table(gen_inv, dc_day_min)
    rel = relative_yield(gen_inv, irradiation)
    df = comp.merge(outg, on="SOURCE_KEY", how="left").merge(eff, on="SOURCE_KEY").merge(rel, on="SOURCE_KEY", how="left")
    for c in ("outage_events", "outage_slots", "outage_max_episode"):
        df[c] = df[c].fillna(0).astype(int)
    df["rel_yield_z"] = robust_z(df["rel_yield"])
    df["intercept_z"] = robust_z(df["intercept"])
    df["eff_z"] = robust_z(df["eff_med"])
    reasons, flags, severities = [], [], []
    for _, r in df.iterrows():
        why = []
        sev = 0
        zmax = max(abs(r["eff_z"]), abs(r["intercept_z"]), abs(r["rel_yield_z"]))
        if r["completeness"] < 0.85 or r["outage_slots"] >= 150 or zmax >= 4.5:
            sev = 2
        if r["completeness"] < completeness_min or r["outage_slots"] >= outage_slots_min or zmax >= mad_z_flag:
            sev = max(sev, 1)
        if r["completeness"] < completeness_min:
            why.append(f"completeness={r['completeness']:.2f}")
        if r["outage_slots"] >= outage_slots_min:
            why.append(f"outage={int(r['outage_slots'])}slots/{int(r['outage_events'])}次")
        if abs(r["eff_z"]) > mad_z_flag:
            why.append(f"eff_z={r['eff_z']:.1f}")
        if abs(r["intercept_z"]) > mad_z_flag:
            why.append(f"static_loss_z={r['intercept_z']:.1f}")
        if abs(r["rel_yield_z"]) > mad_z_flag:
            why.append(f"rel_yield_z={r['rel_yield_z']:.1f}")
        flags.append(1 if sev > 0 else 0)
        severities.append(sev)
        reasons.append(";".join(why))
    df["flag"] = flags
    df["severity"] = severities  # 0 正常 / 1 告警 / 2 重点
    df["reasons"] = reasons
    return df.sort_values(["severity", "outage_slots"], ascending=[False, False]).reset_index(drop=True)


def plant_scale_anomaly(eff_df: pd.DataFrame, low: float = 0.5, high: float = 1.05) -> str | None:
    """站级 DC/AC 量纲异常判定（如 DC 通道整体 ×10）。"""
    med = eff_df["eff_med"].median()
    if np.isnan(med):
        return None
    if med < low:
        return f"站级 DC/AC 中位效率 {med:.4f} ≪ 1，疑似 DC 通道量纲偏大（×{1/med:.1f}）"
    if med > high:
        return f"站级 DC/AC 中位效率 {med:.4f} > 1，数据异常"
    return None
