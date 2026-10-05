"""Turn the raw downloads into one hourly table, ``data/processed/hourly.parquet``.

Each row is a delivery date and local hour (0-23) with the four zone prices in
EUR/MWh, the SE3 price in öre/kWh, Stockholm temperature and the wind index
(observed), and the same two weather values as forecast before the auction
(``_fc`` columns, from 2024).

Usage::

    python -m se3forecast.build
"""

from __future__ import annotations

import duckdb
import pandas as pd

from . import PROCESSED, RAW, ROOT

SQL = ROOT / "sql"


def _query(name: str, **params: str) -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    sql = (SQL / f"{name}.sql").read_text(encoding="utf-8")
    return con.execute(sql, {k: str(v) for k, v in params.items()}).df()


def complete_grid(df: pd.DataFrame, value_cols: list[str], limit: int) -> pd.DataFrame:
    """Reindex to every date x hour and fill short gaps (the spring DST hour) linearly."""
    df["date"] = pd.to_datetime(df["date"])
    dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    grid = pd.MultiIndex.from_product([dates, range(24)], names=["date", "hour"])
    out = df.set_index(["date", "hour"])[value_cols].reindex(grid)
    out[value_cols] = out[value_cols].interpolate(limit=limit, limit_area="inside")
    return out.reset_index()


def build() -> pd.DataFrame:
    prices = _query("hourly_prices", prices=RAW / "prices.parquet")
    wide = prices.pivot_table(index=["date", "hour"], columns="area", values="eur_mwh").reset_index()
    wide.columns = [c.lower() if c.startswith("SE") else c for c in wide.columns]
    ore = prices[prices["area"] == "SE3"][["date", "hour", "ore_kwh"]].rename(columns={"ore_kwh": "se3_ore_kwh"})
    wide = wide.merge(ore, on=["date", "hour"])
    price_cols = ["se1", "se2", "se3", "se4", "se3_ore_kwh"]
    hourly = complete_grid(wide, price_cols, limit=2)

    weather = _query("hourly_weather", weather=RAW / "weather.parquet")
    weather = complete_grid(weather, ["temp_sthlm", "wind_index"], limit=6)
    hourly = hourly.merge(weather, on=["date", "hour"], how="left")

    forecasts = _query("hourly_weather_forecasts", forecasts=RAW / "weather_forecasts.parquet")
    forecasts = complete_grid(forecasts, ["temp_sthlm_fc", "wind_index_fc"], limit=6)
    hourly = hourly.merge(forecasts, on=["date", "hour"], how="left")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(PROCESSED / "hourly.parquet", index=False)
    print(
        f"{len(hourly):,} hours from {hourly['date'].min():%Y-%m-%d} to {hourly['date'].max():%Y-%m-%d}; "
        f"missing SE3 {hourly['se3'].isna().sum()}, temperature {hourly['temp_sthlm'].isna().sum()}, "
        f"wind {hourly['wind_index'].isna().sum()}"
    )
    return hourly


if __name__ == "__main__":
    build()
