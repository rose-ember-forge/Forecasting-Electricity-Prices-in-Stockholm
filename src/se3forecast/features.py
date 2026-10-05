"""Features for forecasting tomorrow's 24 hourly SE3 prices.

The forecast for delivery day T is made on the morning of T-1, before bids close
at 12:00. At that point all prices up to the end of T-1 are known (they were set
in the auction on T-2), so every price feature is lagged by at least one day.
Weather for day T comes either from observations (training) or from forecasts
issued before the deadline (backtest and live forecasts).
"""

from __future__ import annotations

from datetime import date, timedelta

import holidays
import numpy as np
import pandas as pd

PRICE_LAG_DAYS = [1, 2, 3, 7]


def swedish_holidays(years: range) -> set[date]:
    """Public holidays plus the eves when most workplaces close (not plain Sundays)."""
    days = set(holidays.Sweden(years=years, include_sundays=False))
    for year in years:
        days |= {date(year, 12, 24), date(year, 12, 31)}
        midsummer = date(year, 6, 19)  # Midsummer Eve is the Friday between 19 and 25 June
        days.add(midsummer + timedelta(days=(4 - midsummer.weekday()) % 7))
    return days


def _wide(hourly: pd.DataFrame, col: str) -> pd.DataFrame:
    return hourly.pivot(index="date", columns="hour", values=col).asfreq("D")


def _long(wide: pd.DataFrame, name: str) -> pd.Series:
    return wide.stack(future_stack=True).rename(name)


def make_features(hourly: pd.DataFrame, weather: str = "observed") -> pd.DataFrame:
    """One row per delivery date and hour with the target ``y`` (SE3, EUR/MWh) and features.

    ``weather`` is ``"observed"`` (SMHI observations) or ``"forecast"`` (forecasts
    available before the auction).
    """
    suffix = "" if weather == "observed" else "_fc"
    p = _wide(hourly, "se3")
    index = p.stack(future_stack=True).index
    f = pd.DataFrame(index=index)
    f["y"] = _long(p, "y")

    # Same hour on earlier days, and how this hour usually looks over the last week.
    for k in PRICE_LAG_DAYS:
        f[f"p_lag{k}"] = _long(p.shift(k), "x")
    f["p_hour_mean7"] = _long(p.shift(1).rolling(7).mean(), "x")
    f["p_hour_std7"] = _long(p.shift(1).rolling(7).std(), "x")

    # Yesterday's level and shape, broadcast to every hour of the target day.
    daily = pd.DataFrame(index=p.index)
    y1 = p.shift(1)
    daily["d1_mean"] = y1.mean(axis=1)
    daily["d1_min"] = y1.min(axis=1)
    daily["d1_max"] = y1.max(axis=1)
    daily["d1_std"] = y1.std(axis=1)
    daily["d1_last"] = y1[23]
    daily["d7_mean"] = p.mean(axis=1).shift(1).rolling(7).mean()
    for area in ["se1", "se2", "se4"]:
        daily[f"{area}_d1_mean"] = _wide(hourly, area).shift(1).mean(axis=1)
    daily["spread_se3_se1_d1"] = daily["d1_mean"] - daily["se1_d1_mean"]
    daily["spread_se4_se3_d1"] = daily["se4_d1_mean"] - daily["d1_mean"]
    # All 24 prices of earlier days as separate columns (the inputs of the LEAR benchmark).
    day_lags = {f"p{k}_h{h:02d}": p.shift(k)[h] for k in PRICE_LAG_DAYS for h in range(24)}
    daily = pd.concat([daily, pd.DataFrame(day_lags)], axis=1)

    # Weather on the target day.
    temp = _wide(hourly, f"temp_sthlm{suffix}")
    wind = _wide(hourly, f"wind_index{suffix}")
    f["temp"] = _long(temp, "x")
    f["wind"] = _long(wind, "x")
    daily["temp_mean"] = temp.mean(axis=1)
    daily["wind_mean"] = wind.mean(axis=1)
    daily["wind_max"] = wind.max(axis=1)
    daily["temp_mean_change"] = daily["temp_mean"] - temp.shift(1).mean(axis=1)
    daily["wind_mean_change"] = daily["wind_mean"] - wind.shift(1).mean(axis=1)

    # Calendar.
    dates = p.index
    hol = swedish_holidays(range(dates.min().year, dates.max().year + 1))
    daily["dow"] = dates.dayofweek
    daily["month"] = dates.month
    daily["doy_sin"] = np.sin(2 * np.pi * dates.dayofyear / 365.25)
    daily["doy_cos"] = np.cos(2 * np.pi * dates.dayofyear / 365.25)
    daily["holiday"] = [d.date() in hol for d in dates]
    daily["holiday_d1"] = [(d - pd.Timedelta(days=1)).date() in hol for d in dates]
    daily["holiday"] = daily["holiday"].astype(int)
    daily["holiday_d1"] = daily["holiday_d1"].astype(int)

    f = f.join(daily, on="date")
    f = f.reset_index()
    f["hour"] = f["hour"].astype(int)
    return f


PRICE_FEATURES = (
    [f"p_lag{k}" for k in PRICE_LAG_DAYS]
    + ["p_hour_mean7", "p_hour_std7", "d1_mean", "d1_min", "d1_max", "d1_std", "d1_last", "d7_mean"]
    + ["se1_d1_mean", "se2_d1_mean", "se4_d1_mean", "spread_se3_se1_d1", "spread_se4_se3_d1"]
)
CALENDAR_FEATURES = ["hour", "dow", "month", "doy_sin", "doy_cos", "holiday", "holiday_d1"]
WEATHER_FEATURES = ["temp", "wind", "temp_mean", "wind_mean", "wind_max", "temp_mean_change", "wind_mean_change"]
DAY_LAG_COLUMNS = [f"p{k}_h{h:02d}" for k in PRICE_LAG_DAYS for h in range(24)]
