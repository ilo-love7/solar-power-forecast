"""光伏双电站 CSV 加载。

Kaggle Solar Power Generation Data：每电站一对文件（逆变器发电量 + 站级气象）。
已知坑：4 个文件 DATE_TIME 格式不统一（dd-mm-yyyy HH:MM 与 ISO 两种），
禁止 dayfirst=True 混解析（ISO 中 day<=12 的日期会被翻转），按首行显式探测格式。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

GEN_COLS = ["DATE_TIME", "PLANT_ID", "SOURCE_KEY", "DC_POWER", "AC_POWER", "DAILY_YIELD", "TOTAL_YIELD"]
WTHR_COLS = ["DATE_TIME", "PLANT_ID", "SOURCE_KEY", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]

# 显式候选格式（禁止 dayfirst 混解析）
_DT_FORMATS = ["%d-%m-%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%m/%d/%Y %H:%M"]


def parse_dt(s: pd.Series) -> pd.Series:
    """按首个非空值探测格式并整列解析，失败抛 ValueError。"""
    first = s.dropna().astype(str).iloc[0]
    for fmt in _DT_FORMATS:
        try:
            parsed = pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError):
            continue
        if parsed.notna().all():
            return parsed
    raise ValueError(f"无法解析 DATE_TIME，首值={first!r}")


@dataclass
class PlantData:
    """一个电站的发电量（逆变器级）与气象（站级）。"""

    name: str
    gen: pd.DataFrame    # 列含 gen cols + dt
    weather: pd.DataFrame  # 列含 weather cols + dt

    @property
    def inverters(self) -> list[str]:
        return sorted(self.gen["SOURCE_KEY"].unique().tolist())

    @property
    def time_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.gen["dt"].min(), self.gen["dt"].max()


def _read_csv(path: Path, need_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} 缺列: {missing}（实际 {list(df.columns)}）")
    df = df.copy()
    df["dt"] = parse_dt(df["DATE_TIME"])
    return df


def load_plant(raw_dir: str | Path, name: str) -> PlantData:
    """加载 Plant_1 / Plant_2 之一。"""
    raw = Path(raw_dir)
    gen = _read_csv(raw / f"{name}_Generation_Data.csv", GEN_COLS)
    weather = _read_csv(raw / f"{name}_Weather_Sensor_Data.csv", WTHR_COLS)
    return PlantData(name=name, gen=gen, weather=weather)


def load_all(raw_dir: str | Path, names=("Plant_1", "Plant_2")) -> dict[str, PlantData]:
    return {n: load_plant(raw_dir, n) for n in names}
