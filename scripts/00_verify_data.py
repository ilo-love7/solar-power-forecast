"""00 数据校验：schema、行数、日期范围、逆变器数、时间戳可解析性。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar_diag.config import load_config, results_dir
from solar_diag.io import GEN_COLS, WTHR_COLS, load_all

EXPECTED = {
    "Plant_1": {"gen_rows": (60000, 80000), "wthr_rows": (3100, 3300), "inverters": 22},
    "Plant_2": {"gen_rows": (60000, 80000), "wthr_rows": (3100, 3300), "inverters": 22},
}


def main() -> None:
    cfg = load_config()
    raw = Path(cfg["data_root"]) / cfg["raw_subdir"]
    plants = load_all(raw, cfg["plants"])
    report = {"raw_dir": str(raw), "plants": {}}
    ok = True
    for name, p in plants.items():
        exp = EXPECTED.get(name, {})
        g, w = p.gen, p.weather
        checks = {
            "gen_cols_ok": all(c in g.columns for c in GEN_COLS),
            "wthr_cols_ok": all(c in w.columns for c in WTHR_COLS),
            "gen_rows": int(len(g)),
            "wthr_rows": int(len(w)),
            "inverters": int(g["SOURCE_KEY"].nunique()),
            "gen_dt_min": str(g["dt"].min()),
            "gen_dt_max": str(g["dt"].max()),
            "gen_dup_rows": int(g.duplicated(subset=["dt", "SOURCE_KEY"]).sum()),
            "gen_nulls": int(g[GEN_COLS[:-2]].isna().sum().sum()),
            "dc_zero_frac": float((g["DC_POWER"] == 0).mean()),
        }
        checks["gen_rows_ok"] = exp.get("gen_rows", (0, 10**9))[0] <= checks["gen_rows"] <= exp.get("gen_rows", (0, 10**9))[1]
        checks["wthr_rows_ok"] = exp.get("wthr_rows", (0, 10**9))[0] <= checks["wthr_rows"] <= exp.get("wthr_rows", (0, 10**9))[1]
        checks["inverters_ok"] = checks["inverters"] == exp.get("inverters", checks["inverters"])
        checks["dup_ok"] = checks["gen_dup_rows"] == 0
        passed = all(v for k, v in checks.items() if k.endswith("_ok"))
        checks["passed"] = passed
        ok = ok and passed
        report["plants"][name] = checks
        print(f"[{name}] rows={checks['gen_rows']}/{checks['wthr_rows']} inv={checks['inverters']} "
              f"{checks['gen_dt_min'][:10]}~{checks['gen_dt_max'][:10]} passed={passed}")
    report["all_passed"] = ok
    out = results_dir() / "data_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
