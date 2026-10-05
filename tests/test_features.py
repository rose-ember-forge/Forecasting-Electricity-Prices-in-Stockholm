"""The features must only use information available before the day-ahead auction."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from se3forecast.build import complete_grid
from se3forecast.features import CALENDAR_FEATURES, DAY_LAG_COLUMNS, PRICE_FEATURES, make_features, swedish_holidays


def synthetic_hourly(days: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    idx = pd.MultiIndex.from_product([dates, range(24)], names=["date", "hour"])
    n = len(idx)
    df = pd.DataFrame(index=idx).reset_index()
    for col in ["se1", "se2", "se3", "se4"]:
        df[col] = rng.normal(60, 20, n)
    df["temp_sthlm"] = rng.normal(0, 5, n)
    df["wind_index"] = rng.gamma(2, 2, n)
    df["temp_sthlm_fc"] = df["temp_sthlm"] + rng.normal(0, 1, n)
    df["wind_index_fc"] = df["wind_index"] + rng.normal(0, 0.5, n)
    return df


def test_price_features_do_not_see_the_target_day():
    hourly = synthetic_hourly()
    target = pd.Timestamp("2025-01-20")
    before = make_features(hourly).set_index(["date", "hour"])
    changed = hourly.copy()
    on_target = changed["date"] >= target
    for col in ["se1", "se2", "se3", "se4"]:
        changed.loc[on_target, col] += 1000  # wildly different prices on and after the target day
    after = make_features(changed).set_index(["date", "hour"])
    cols = PRICE_FEATURES + DAY_LAG_COLUMNS + [c for c in CALENDAR_FEATURES if c != "hour"]
    pd.testing.assert_frame_equal(before.loc[target, cols], after.loc[target, cols])
    assert np.allclose(after.loc[target, "y"] - before.loc[target, "y"], 1000)


def test_lags_point_at_the_right_days():
    hourly = synthetic_hourly()
    f = make_features(hourly).set_index(["date", "hour"])
    p = hourly.set_index(["date", "hour"])["se3"]
    t = pd.Timestamp("2025-01-15")
    assert f.loc[(t, 18), "p_lag1"] == p.loc[(t - pd.Timedelta(days=1), 18)]
    assert f.loc[(t, 18), "p_lag7"] == p.loc[(t - pd.Timedelta(days=7), 18)]
    assert f.loc[(t, 3), "d1_last"] == p.loc[(t - pd.Timedelta(days=1), 23)]
    assert f.loc[(t, 3), "p2_h07"] == p.loc[(t - pd.Timedelta(days=2), 7)]


def test_forecast_mode_uses_forecast_weather():
    hourly = synthetic_hourly()
    obs = make_features(hourly, "observed")
    fc = make_features(hourly, "forecast")
    assert np.allclose(obs["temp"], hourly["temp_sthlm"])
    assert np.allclose(fc["temp"], hourly["temp_sthlm_fc"])


def test_swedish_holidays_include_the_eves_but_not_sundays():
    days = swedish_holidays(range(2026, 2027))
    assert date(2026, 6, 19) in days  # Midsummer Eve (Friday)
    assert date(2026, 12, 24) in days
    assert date(2026, 5, 1) in days
    assert date(2026, 10, 4) not in days  # an ordinary Sunday


@pytest.mark.parametrize("year,expected", [(2025, date(2025, 6, 20)), (2027, date(2027, 6, 25))])
def test_midsummer_eve_is_a_friday(year, expected):
    assert expected in swedish_holidays(range(year, year + 1))


def test_complete_grid_fills_the_missing_spring_dst_hour():
    df = pd.DataFrame({"date": ["2026-03-29"] * 23, "hour": [h for h in range(24) if h != 2], "x": [float(h) for h in range(24) if h != 2]})
    out = complete_grid(df, ["x"], limit=2)
    assert len(out) == 24
    assert out.loc[out["hour"] == 2, "x"].item() == pytest.approx(2.0)
