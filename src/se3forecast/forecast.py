"""Forecast the next day that has no published prices yet, and score earlier forecasts.

Run every morning before the noon auction (see ``.github/workflows/forecast.yml``):
it refreshes the data, retrains the final model on all history, forecasts the next
delivery day with the latest weather forecast, appends the result to
``forecasts/log.csv`` and scores every logged forecast whose prices are now known.

Usage::

    python -m se3forecast.forecast
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from . import PROCESSED, ROOT, TZ
from .backtest import INTERVAL, final_model
from .build import build
from .features import make_features
from .ingest import (
    TEMPERATURE_STATIONS,
    WIND_STATIONS,
    fetch_prices,
    fetch_weather,
    fetch_weather_forecasts,
    session,
    station_position,
)
from .models import GBM

FORECASTS = ROOT / "forecasts"
LIVE_URL = "https://api.open-meteo.com/v1/forecast"


def live_weather_forecast() -> pd.DataFrame:
    """The latest weather forecast for the coming days, per local date and hour."""
    frames = []
    for param, stations, om_var, variable in [
        (1, TEMPERATURE_STATIONS, "temperature_2m", "temp_sthlm_fc"),
        (4, WIND_STATIONS, "wind_speed_10m", "wind_index_fc"),
    ]:
        for station in stations:
            lat, lon = station_position(param, station)
            params = {"latitude": lat, "longitude": lon, "hourly": om_var, "wind_speed_unit": "ms",
                      "timezone": TZ, "forecast_days": 4}
            r = session.get(LIVE_URL, params=params, timeout=60)
            r.raise_for_status()
            hourly = r.json()["hourly"]
            t = pd.to_datetime(hourly["time"])
            frames.append(pd.DataFrame({"date": t.normalize(), "hour": t.hour, "variable": variable, "value": hourly[om_var]}))
    df = pd.concat(frames)
    return df.pivot_table(index=["date", "hour"], columns="variable", values="value", aggfunc="mean").reset_index()


def forecast_next_day(refresh: bool = True) -> pd.DataFrame:
    if refresh:
        fetch_prices()
        fetch_weather()
        fetch_weather_forecasts()
        build()
    hourly = pd.read_parquet(PROCESSED / "hourly.parquet")
    target = hourly["date"].max() + pd.Timedelta(days=1)

    # Add empty rows for the target day, then put the latest weather forecast on today onwards.
    blank = pd.DataFrame({"date": target, "hour": range(24)})
    hourly = pd.concat([hourly, blank], ignore_index=True)
    live = live_weather_forecast()
    today = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
    live = live[live["date"] >= today].set_index(["date", "hour"])
    hourly = hourly.set_index(["date", "hour"])
    hourly.update(live)
    hourly = hourly.reset_index()

    train = make_features(hourly, "observed")
    train = train[train["date"] < target].dropna()
    test = make_features(hourly, "forecast")
    test = test[test["date"] == target]
    feature_cols = [c for c in test.columns if c != "y"]
    if test[feature_cols].isna().any().any():
        missing = test[feature_cols].columns[test[feature_cols].isna().any()].tolist()
        raise RuntimeError(f"missing inputs for {target:%Y-%m-%d}: {missing}")

    model = final_model().fit(train)
    band = GBM(day_lags=True, interval=INTERVAL).fit(train).predict_interval(test)
    out = pd.DataFrame(
        {
            "date": target.date(),
            "hour": test["hour"].to_numpy(),
            "forecast_eur_mwh": model.predict(test).round(2),
            "low_80_eur_mwh": band["lo"].to_numpy().round(2),
            "high_80_eur_mwh": band["hi"].to_numpy().round(2),
            "made_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        }
    )
    FORECASTS.mkdir(exist_ok=True)
    out.to_csv(FORECASTS / "latest.csv", index=False)
    log_path = FORECASTS / "log.csv"
    log = pd.concat([pd.read_csv(log_path), out]) if log_path.exists() else out
    log["date"] = pd.to_datetime(log["date"]).dt.date
    log = log.drop_duplicates(["date", "hour"], keep="first")  # keep the forecast made before the auction
    log.to_csv(log_path, index=False)
    print(f"forecast for {target:%Y-%m-%d}: mean {out['forecast_eur_mwh'].mean():.1f} EUR/MWh, "
          f"min {out['forecast_eur_mwh'].min():.1f}, max {out['forecast_eur_mwh'].max():.1f}")
    return out


def score() -> pd.DataFrame:
    """Daily error of the logged live forecasts against the prices that came out."""
    log_path = FORECASTS / "log.csv"
    if not log_path.exists():
        return pd.DataFrame()
    log = pd.read_csv(log_path, parse_dates=["date"])
    actual = pd.read_parquet(PROCESSED / "hourly.parquet")[["date", "hour", "se3"]]
    df = log.merge(actual, on=["date", "hour"])
    if df.empty:
        return df
    df["abs_err"] = (df["forecast_eur_mwh"] - df["se3"]).abs()
    df["inside"] = df["se3"].between(df["low_80_eur_mwh"], df["high_80_eur_mwh"])
    daily = df.groupby("date").agg(mean_price=("se3", "mean"), mae=("abs_err", "mean"), interval_coverage=("inside", "mean"))
    daily.round(3).to_csv(FORECASTS / "live_scores.csv")
    print(f"live forecasts scored: {len(daily)} days, MAE {df['abs_err'].mean():.1f} EUR/MWh, "
          f"80% interval coverage {df['inside'].mean():.0%}")
    return daily


def main() -> None:
    forecast_next_day()
    score()


if __name__ == "__main__":
    main()
