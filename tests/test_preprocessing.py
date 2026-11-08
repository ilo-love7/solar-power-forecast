import numpy as np
import pandas as pd

from solar_diag.preprocessing import capacity_of, plant_level, split_days
from solar_diag.io import PlantData


def _mk_plant(base_df, n_inv=1):
    """由合成站级数据造 PlantData（gen 为逆变器级大写列）。"""
    frames = []
    for i in range(n_inv):
        g = pd.DataFrame({
            "dt": base_df["dt"],
            "SOURCE_KEY": f"inv{i}",
            "DC_POWER": base_df["dc_power"] / n_inv,
            "AC_POWER": base_df["ac_power"] / n_inv,
            "DAILY_YIELD": 0.0,
            "TOTAL_YIELD": 0.0,
        })
        frames.append(g)
    gen = pd.concat(frames, ignore_index=True)
    w = pd.DataFrame({
        "dt": base_df["dt"],
        "AMBIENT_TEMPERATURE": base_df["ambient_temp"],
        "MODULE_TEMPERATURE": base_df["module_temp"],
        "IRRADIATION": base_df["irradiation"],
        "SOURCE_KEY": "sensor",
        "PLANT_ID": 1,
    })
    return PlantData(name="T", gen=gen, weather=w)


def test_plant_level_sum(tiny_plant_df):
    plant = _mk_plant(tiny_plant_df.head(96), n_inv=2)
    pl = plant_level(plant)
    assert np.isclose(pl["ac_power"].iloc[0], tiny_plant_df["ac_power"].iloc[0])
    assert pl["n_inv"].iloc[0] == 2


def test_split_days_ordered(tiny_plant_df):
    splits = split_days(tiny_plant_df, 0.7, 0.15)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    assert tr["dt"].max() < va["dt"].min()
    assert va["dt"].max() < te["dt"].min()
    assert len(tr) + len(va) + len(te) == len(tiny_plant_df)


def test_capacity(tiny_plant_df):
    cap = capacity_of(tiny_plant_df, 0.99)
    assert cap <= tiny_plant_df["ac_power"].max() * 1.001
    assert cap >= tiny_plant_df["ac_power"].quantile(0.5)
