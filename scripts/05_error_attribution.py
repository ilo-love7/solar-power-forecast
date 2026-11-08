"""05 预测误差归因：按目标时刻天气（辐照）与时段分桶。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar_diag import evaluation as ev
from solar_diag import plots
from solar_diag.config import figures_dir, load_config, results_dir


def pick_best_pred(res_dir: Path, plant: str) -> tuple[str, np.lib.npyio.NpzFile]:
    """在已保存的测试预测里选全量 RMSE 最小的模型（含 LSTM）。"""
    best, best_rmse = None, float("inf")
    for f in sorted(res_dir.glob(f"pred_{plant}_*.npz")):
        d = np.load(f, allow_pickle=True)
        r = ev.rmse(d["y_true"], d["y_pred"])
        if r < best_rmse:
            best, best_rmse = f, r
    if best is None:
        raise FileNotFoundError(f"{plant} 无预测文件，请先跑 02/03")
    name = best.stem.replace(f"pred_{plant}_", "")
    return name, np.load(best, allow_pickle=True)


def main() -> None:
    cfg = load_config()
    fig_dir, res_dir = figures_dir(), results_dir()
    report = {}
    for plant in cfg["plants"]:
        name, d = pick_best_pred(res_dir, plant)
        y_true, y_pred = d["y_true"], d["y_pred"]
        cap = float(np.nanquantile(y_true, 0.99)) if len(y_true) else 1.0
        # 用 02 记录的容量更准
        fm = json.loads((res_dir / "forecast_metrics.json").read_text(encoding="utf-8"))
        cap = fm.get(plant, {}).get("capacity_w", cap)
        irr = d["target_irr"]
        hour = d["target_hour"]
        wb = ev.bucket_by_irradiation(irr, cfg["irr_bins"])
        db = ev.bucket_by_daypart(hour)
        wdf = ev.grouped_metrics(y_true, y_pred, wb, cap)
        ddf = ev.grouped_metrics(y_true, y_pred, db, cap)
        wdf.to_csv(res_dir / f"err_by_weather_{plant}.csv", index=False, encoding="utf-8-sig")
        ddf.to_csv(res_dir / f"err_by_daypart_{plant}.csv", index=False, encoding="utf-8-sig")
        plots.plot_error_by_bucket(wdf, name, plant, "天气", fig_dir)
        plots.plot_error_by_bucket(ddf, name, plant, "时段", fig_dir)
        report[plant] = {
            "best_model": name,
            "capacity_w": cap,
            "overall": ev.metric_pack(y_true, y_pred, cap),
            "by_weather": wdf.to_dict(orient="records"),
            "by_daypart": ddf.to_dict(orient="records"),
        }
        print(f"[{plant}] best={name} overall RMSE={report[plant]['overall']['rmse']:.1f}W")
        print(wdf.to_string(index=False))
    out = res_dir / "error_attribution.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
