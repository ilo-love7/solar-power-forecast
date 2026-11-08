import pandas as pd
import pytest

from solar_diag.io import parse_dt, GEN_COLS, WTHR_COLS, _read_csv
from pathlib import Path


def test_parse_dt_iso():
    s = pd.Series(["2020-05-15 00:00:00", "2020-06-01 23:45:00"])
    out = parse_dt(s)
    assert out.iloc[0] == pd.Timestamp("2020-05-15 00:00")
    assert out.iloc[1] == pd.Timestamp("2020-06-01 23:45")  # dayfirst 翻转回归


def test_parse_dt_ddmm():
    s = pd.Series(["15-05-2020 00:00", "01-06-2020 23:45"])
    out = parse_dt(s)
    assert out.iloc[0] == pd.Timestamp("2020-05-15")
    assert out.iloc[1] == pd.Timestamp("2020-06-01 23:45")


def test_parse_dt_bad():
    with pytest.raises(ValueError):
        parse_dt(pd.Series(["not-a-date", "xxx"]))


def test_read_csv_missing_col(tmp_path):
    p = tmp_path / "Plant_X_Generation_Data.csv"
    pd.DataFrame({"DATE_TIME": ["2020-05-15 00:00:00"], "DC_POWER": [1.0]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        _read_csv(p, GEN_COLS)


def test_gen_cols_documented():
    assert "AC_POWER" in GEN_COLS and "IRRADIATION" in WTHR_COLS
