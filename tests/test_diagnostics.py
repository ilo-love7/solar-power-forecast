import numpy as np
import pandas as pd

from solar_diag.diagnostics import (
    completeness, daytime_outages, diagnose_inverters, efficiency_table,
    plant_scale_anomaly, relative_yield, robust_z,
)


def test_robust_z_outlier():
    s = pd.Series([1.0, 1.1, 0.9, 1.05, 100.0])
    z = robust_z(s)
    assert abs(z.iloc[-1]) > 3.5
    assert abs(z.iloc[0]) < 3.5


def test_completeness_missing(tiny_gen_inv):
    # invBad 第一天午后缺行 → completeness 下降
    comp = completeness(tiny_gen_inv, slots_expected=192)
    bad = comp.set_index("SOURCE_KEY").loc["invBad"]
    ok = comp.set_index("SOURCE_KEY").loc["invA"]
    assert bad["completeness"] < ok["completeness"]
    assert ok["completeness"] == 1.0
    assert bad["n_slots"] < 192


def test_daytime_outages_detects(tiny_gen_inv):
    dt_index = pd.DatetimeIndex(sorted(tiny_gen_inv["dt"].unique()))
    irr = pd.Series(0.8, index=dt_index)
    ev = daytime_outages(tiny_gen_inv, irradiation=irr, irr_min=0.5, fleet_min=200.0, self_frac=0.05)
    assert "invBad" in ev["SOURCE_KEY"].values
    row = ev.set_index("SOURCE_KEY").loc["invBad"]
    assert row["outage_slots"] > 10
    assert row["outage_events"] == 1  # 连续午后停机合并为一次
    assert row["outage_max_episode"] == row["outage_slots"]
    assert "invA" not in set(ev["SOURCE_KEY"])


def test_efficiency_table(tiny_gen_inv):
    eff = efficiency_table(tiny_gen_inv, dc_day_min=100.0).set_index("SOURCE_KEY")
    assert np.isclose(eff.loc["invA", "eff_med"], 0.975, atol=0.01)
    assert np.isclose(eff.loc["invA", "slope"], 0.975, atol=0.02)


def test_relative_yield_columns(tiny_gen_inv):
    dt_index = pd.DatetimeIndex(sorted(tiny_gen_inv["dt"].unique()))
    irr = pd.Series(0.8, index=dt_index)
    rel = relative_yield(tiny_gen_inv, irr)
    assert set(rel.columns) == {"SOURCE_KEY", "rel_yield"}
    assert (rel["rel_yield"] > 0).all()


def test_diagnose_flags_bad(tiny_gen_inv):
    dt_index = pd.DatetimeIndex(sorted(tiny_gen_inv["dt"].unique()))
    irr = pd.Series(0.8, index=dt_index)
    diag = diagnose_inverters(tiny_gen_inv, irradiation=irr, slots_expected=192,
                              completeness_min=0.90).set_index("SOURCE_KEY")
    assert diag.loc["invBad", "flag"] == 1
    assert diag.loc["invA", "flag"] == 0
    assert diag.loc["invB", "flag"] == 0


def test_scale_anomaly():
    eff = pd.DataFrame({"eff_med": [0.0978, 0.0979, 0.0978]})
    note = plant_scale_anomaly(eff)
    assert note is not None and "量纲" in note
    eff2 = pd.DataFrame({"eff_med": [0.97, 0.98, 0.975]})
    assert plant_scale_anomaly(eff2) is None
