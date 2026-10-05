"""Download day-ahead prices (elprisetjustnu.se), weather observations (SMHI)
and archived weather forecasts (Open-Meteo).

Prices come as one small JSON file per bidding zone and delivery day. They are
cached per month in ``data/raw/prices/`` so a rerun only fetches new days.
Weather is SMHI's quality-controlled archive plus the last four months of
unchecked observations, per station and parameter.

Usage::

    python -m se3forecast.ingest            # everything up to tomorrow
    python -m se3forecast.ingest --prices   # prices only
"""

from __future__ import annotations

import argparse
import io
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd
import requests

from . import AREAS, RAW

PRICE_URL = "https://www.elprisetjustnu.se/api/v1/prices/{d:%Y}/{d:%m-%d}_{area}.json"
FIRST_DAY = date(2021, 10, 1)  # the API has nothing earlier; the first days return 404

SMHI_URL = (
    "https://opendata-download-metobs.smhi.se/api/version/1.0/"
    "parameter/{param}/station/{station}/period/{period}/data.{ext}"
)
# Hourly air temperature (parameter 1) where most of SE3's demand is,
# and hourly mean wind speed (parameter 4) at stations near the main wind areas.
TEMPERATURE_STATIONS = {98230: "Stockholm"}
WIND_STATIONS = {
    71420: "Göteborg",
    62260: "Hallands Väderö",
    77210: "Ölands norra udde",
    107420: "Gävle",
    127310: "Sundsvall",
    134110: "Östersund",
}

session = requests.Session()
session.headers["User-Agent"] = "se3-price-forecast (portfolio project)"


def _get(url: str, retries: int = 4) -> requests.Response | None:
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    return None


def fetch_price_day(day: date, area: str) -> pd.DataFrame:
    r = _get(PRICE_URL.format(d=day, area=area))
    if r is None:
        return pd.DataFrame()
    df = pd.DataFrame(r.json())
    df["area"] = area
    return df[["area", "time_start", "time_end", "EUR_per_kWh", "SEK_per_kWh"]]


def fetch_prices(until: date | None = None) -> pd.DataFrame:
    """Fetch every zone and day, reusing complete cached months."""
    until = until or date.today() + timedelta(days=1)
    cache = RAW / "prices"
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    month = FIRST_DAY.replace(day=1)
    while month <= until:
        nxt = (month + timedelta(days=32)).replace(day=1)
        path = cache / f"{month:%Y-%m}.parquet"
        complete = nxt <= until - timedelta(days=1)  # tomorrow's file may not exist yet
        if path.exists() and complete:
            frames.append(pd.read_parquet(path))
        else:
            days = [month + timedelta(days=i) for i in range((min(nxt - timedelta(days=1), until) - month).days + 1)]
            jobs = [(d, a) for d in days for a in AREAS]
            with ThreadPoolExecutor(max_workers=4) as pool:
                parts = list(pool.map(lambda job: fetch_price_day(*job), jobs))
            df = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else pd.DataFrame()
            if len(df):
                df.to_parquet(path, index=False)
                frames.append(df)
            print(f"prices {month:%Y-%m}: {len(df):,} rows")
        month = nxt
    prices = pd.concat(frames, ignore_index=True).drop_duplicates(["area", "time_start"])
    prices.to_parquet(RAW / "prices.parquet", index=False)
    return prices


def _smhi_archive(param: int, station: int) -> pd.DataFrame:
    r = _get(SMHI_URL.format(param=param, station=station, period="corrected-archive", ext="csv"))
    if r is None:
        return pd.DataFrame(columns=["time_utc", "value"])
    lines = r.content.decode("utf-8-sig").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("Datum;Tid (UTC)"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])), sep=";", usecols=[0, 1, 2], header=0)
    df.columns = ["date", "time", "value"]
    df["time_utc"] = pd.to_datetime(df["date"] + " " + df["time"], utc=True)
    return df[["time_utc", "value"]]


def _smhi_recent(param: int, station: int) -> pd.DataFrame:
    r = _get(SMHI_URL.format(param=param, station=station, period="latest-months", ext="json"))
    if r is None or not r.json().get("value"):
        return pd.DataFrame(columns=["time_utc", "value"])
    df = pd.DataFrame(r.json()["value"])
    df["time_utc"] = pd.to_datetime(df["date"], unit="ms", utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["time_utc", "value"]]


def fetch_weather() -> pd.DataFrame:
    frames = []
    for param, variable, stations in [(1, "temperature", TEMPERATURE_STATIONS), (4, "wind_speed", WIND_STATIONS)]:
        for station, name in stations.items():
            archive = _smhi_archive(param, station)
            recent = _smhi_recent(param, station)
            df = pd.concat([archive, recent]).drop_duplicates("time_utc", keep="last")
            df = df[df["time_utc"] >= pd.Timestamp(FIRST_DAY - timedelta(days=14), tz="UTC")]
            df["variable"], df["station"], df["station_name"] = variable, station, name
            frames.append(df)
            print(f"weather {variable} {name}: {len(df):,} hours")
    weather = pd.concat(frames, ignore_index=True)
    weather.to_parquet(RAW / "weather.parquet", index=False)
    return weather


FORECAST_ARCHIVE_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST_ARCHIVE_START = date(2024, 1, 1)  # archived forecasts are complete from about February 2024


def station_position(param: int, station: int) -> tuple[float, float]:
    meta = _get(f"https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/{param}/station/{station}.json").json()
    pos = meta["position"][-1]
    return pos["latitude"], pos["longitude"]


def fetch_weather_forecasts(until: date | None = None) -> pd.DataFrame:
    """Archived weather forecasts at the same stations, as they looked one and two days ahead.

    Open-Meteo's previous-runs API keeps what its forecast said 24 h (``previous_day1``) and
    48 h (``previous_day2``) before each hour. The backtest uses these instead of the observed
    weather, so the model only sees weather information it could have had at bidding time.
    """
    until = until or date.today() + timedelta(days=1)
    stations = [(1, s, "temperature", n) for s, n in TEMPERATURE_STATIONS.items()]
    stations += [(4, s, "wind_speed", n) for s, n in WIND_STATIONS.items()]
    frames = []
    for param, station, variable, name in stations:
        lat, lon = station_position(param, station)
        om_var = "temperature_2m" if variable == "temperature" else "wind_speed_10m"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": f"{om_var}_previous_day1,{om_var}_previous_day2",
            "start_date": f"{FORECAST_ARCHIVE_START}",
            "end_date": f"{until}",
            "wind_speed_unit": "ms",
            "timezone": "GMT",
        }
        r = session.get(FORECAST_ARCHIVE_URL, params=params, timeout=60)
        r.raise_for_status()
        hourly = r.json()["hourly"]
        df = pd.DataFrame(
            {
                "time_utc": pd.to_datetime(hourly["time"], utc=True),
                "lead_1d": hourly[f"{om_var}_previous_day1"],
                "lead_2d": hourly[f"{om_var}_previous_day2"],
            }
        )
        df["variable"], df["station"], df["station_name"] = variable, station, name
        frames.append(df)
        print(f"forecast archive {variable} {name}: {df['lead_1d'].notna().sum():,} hours")
        time.sleep(1)
    forecasts = pd.concat(frames, ignore_index=True)
    forecasts.to_parquet(RAW / "weather_forecasts.parquet", index=False)
    return forecasts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prices", action="store_true", help="prices only")
    parser.add_argument("--weather", action="store_true", help="weather only")
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    both = not (args.prices or args.weather)
    if args.prices or both:
        fetch_prices()
    if args.weather or both:
        fetch_weather()
        fetch_weather_forecasts()


if __name__ == "__main__":
    main()
