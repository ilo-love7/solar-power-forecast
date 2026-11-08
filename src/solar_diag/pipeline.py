"""脚本共用流程：站级对齐 → 监督框 → 按天切分。"""
from __future__ import annotations

import pandas as pd

from .features import build_supervised, split_supervised_by_day
from .io import PlantData
from .preprocessing import capacity_of, drop_missing_power, plant_level


def prepare_plant(plant: PlantData, cfg: dict) -> tuple[pd.DataFrame, float]:
    pl = drop_missing_power(plant_level(plant))
    cap = capacity_of(pl, cfg.get("capacity_q", 0.99))
    return pl, cap


def make_frames(
    pl: pd.DataFrame,
    cfg: dict,
    regime: str,
) -> dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]]:
    x, y, meta = build_supervised(pl, horizon=cfg["horizon_steps"], regime=regime)
    return split_supervised_by_day(x, y, meta, cfg["train_frac"], cfg["val_frac"])


def daytime_mask(meta: pd.DataFrame, irr_min: float) -> pd.Series:
    return meta["target_irr"] > irr_min
