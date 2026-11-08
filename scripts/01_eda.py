"""01 EDA：昼夜模式、辐照-功率、相关性、缺槽热图、逐日电量。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar_diag import plots
from solar_diag.config import figures_dir, load_config, results_dir
from solar_diag.io import load_all
from solar_diag.preprocessing import drop_missing_power, inverter_level, plant_level


def main() -> None:
    cfg = load_config()
    raw = Path(cfg["data_root"]) / cfg["raw_subdir"]
    plants = load_all(raw, cfg["plants"])
    fig_dir, res_dir = figures_dir(), results_dir()
    summary = {}
    for name, p in plants.items():
        pl = drop_missing_power(plant_level(p))
        inv = inverter_level(p)
        figs = [
            plots.plot_daily_profile(pl, name, fig_dir),
            plots.plot_daily_energy(pl, name, fig_dir),
            plots.plot_irradiation_scatter(pl, name, fig_dir),
            plots.plot_weather_corr(pl, name, fig_dir),
            plots.plot_missing_heatmap(inv, name, fig_dir),
        ]
        summary[name] = {
            "n_plant_slots": int(len(pl)),
            "time_range": [str(pl["dt"].min()), str(pl["dt"].max())],
            "ac_max_w": float(pl["ac_power"].max()),
            "ac_mean_day_w": float(pl.loc[pl["irradiation"] > 0.1, "ac_power"].mean()),
            "irr_max": float(pl["irradiation"].max()),
            "daily_energy_kwh_mean": float(pl.groupby(pl["dt"].dt.normalize())["ac_power"].sum().mean() * 0.25 / 1000),
            "ac_irr_corr": float(pl[["ac_power", "irradiation"]].dropna().corr().iloc[0, 1]),
            "figures": figs,
        }
        print(f"[{name}] slots={len(pl)} ac_max={summary[name]['ac_max_w']/1000:.1f}kW "
              f"corr(ac,irr)={summary[name]['ac_irr_corr']:.3f} figs={len(figs)}")
    out = res_dir / "eda_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
