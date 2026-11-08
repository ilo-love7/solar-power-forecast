"""04 逆变器 DC/AC 效率与静态损耗诊断 + 完整性/停机/相对出力。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar_diag import plots
from solar_diag.config import figures_dir, load_config, results_dir
from solar_diag.diagnostics import diagnose_inverters, efficiency_table, plant_scale_anomaly
from solar_diag.io import load_all
from solar_diag.preprocessing import inverter_level


def run_plant(plant, cfg, fig_dir, res_dir):
    inv = inverter_level(plant)
    irr = plant.weather.set_index("dt")["IRRADIATION"]
    diag = diagnose_inverters(
        inv, irradiation=irr,
        slots_expected=cfg["slots_expected"],
        dc_day_min=cfg["dc_day_min"],
        irr_min=cfg["outage_irr_min"],
        fleet_min=cfg["outage_fleet_min"],
        self_frac=cfg["outage_self_frac"],
        mad_z_flag=cfg["mad_z_flag"],
        completeness_min=cfg["completeness_min"],
        outage_slots_min=cfg["outage_slots_min"],
    )
    eff = efficiency_table(inv, cfg["dc_day_min"])
    scale_note = plant_scale_anomaly(eff)
    csv_path = res_dir / f"inverter_diagnosis_{plant.name}.csv"
    diag.to_csv(csv_path, index=False, encoding="utf-8-sig")
    plots.plot_efficiency_scatter(diag, plant.name, fig_dir)
    plots.plot_completeness_bars(diag, plant.name, fig_dir)
    flagged = diag[diag["flag"] > 0][["SOURCE_KEY", "severity", "reasons", "completeness", "outage_events", "outage_slots", "eff_med", "slope", "intercept"]]
    summary = {
        "n_inverters": int(len(diag)),
        "n_flagged": int((diag["flag"] > 0).sum()),
        "n_critical": int((diag["severity"] == 2).sum()),
        "n_warning": int((diag["severity"] == 1).sum()),
        "scale_anomaly": scale_note,
        "eff_median_fleet": float(eff["eff_med"].median()),
        "flagged": flagged.to_dict(orient="records"),
        "csv": str(csv_path),
    }
    print(f"[{plant.name}] inverters={len(diag)} critical={summary['n_critical']} warning={summary['n_warning']} scale={scale_note}")
    for _, r in flagged.iterrows():
        tag = "!!" if r["severity"] == 2 else " !"
        print(f"  {tag} {r['SOURCE_KEY']}: {r['reasons']}")
    return summary, diag


def main() -> None:
    cfg = load_config()
    raw = Path(cfg["data_root"]) / cfg["raw_subdir"]
    plants = load_all(raw, cfg["plants"])
    fig_dir, res_dir = figures_dir(), results_dir()
    report = {}
    for name, p in plants.items():
        summary, _ = run_plant(p, cfg, fig_dir, res_dir)
        report[name] = summary
    out = res_dir / "inverter_diagnosis.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
