"""Rolling-origin backtest: retrain at the start of every month, forecast each day of it.

Model choices (training window, which models to keep) were made on the validation
year, October 2024 to September 2025. The test year, October 2025 to September 2026,
was run once with those choices fixed.

Models are trained on observed weather and evaluated with the weather forecasts that
existed before the auction, as they would be used in practice. The best model is also
run with observed weather to show what perfect weather forecasts would be worth.

Usage::

    python -m se3forecast.backtest               # test year, writes data/processed/backtest_*
    python -m se3forecast.backtest --validation  # model choices on the validation year
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from . import PROCESSED
from .features import make_features
from .models import GBM, LEAR, Ensemble, Naive, WeeklyNaive

VALIDATION = ("2024-10-01", "2025-09-30")
TEST = ("2025-10-01", "2026-09-30")
INTERVAL = (0.1, 0.9)  # an 80% prediction interval

# Chosen on the validation year: LEAR on the last two years, LightGBM on all history with
# all 96 hourly lags, and their average, which beat both on its own.
FINAL = Ensemble.name


def final_lear() -> LEAR:
    return LEAR(window_days=730)


def final_model() -> Ensemble:
    return Ensemble(final_lear(), GBM(day_lags=True))


def load_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    hourly = pd.read_parquet(PROCESSED / "hourly.parquet")
    return make_features(hourly, "observed"), make_features(hourly, "forecast")


def rolling_backtest(models: dict, train_feats: pd.DataFrame, test_feats: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Return one row per model, date and hour with the actual price and the forecast."""
    rows = []
    end_ts = pd.Timestamp(end)
    for month in pd.date_range(start, end, freq="MS"):
        month_end = min(month + pd.offsets.MonthEnd(0), end_ts)
        train = train_feats[train_feats["date"] < month].dropna()
        test = test_feats[test_feats["date"].between(month, month_end)].dropna()
        for name, make in models.items():
            model = make().fit(train)
            out = test[["date", "hour", "y"]].copy()
            out["model"] = name
            out["pred"] = model.predict(test)
            if getattr(model, "interval", None):
                out = out.join(model.predict_interval(test))
            rows.append(out)
        print(f"  {month:%Y-%m}: trained on {train['date'].nunique()} days, forecast {test['date'].nunique()} days")
    return pd.concat(rows, ignore_index=True)


def metrics(preds: pd.DataFrame, reference: str) -> pd.DataFrame:
    """MAE, RMSE, bias and MAE relative to the naive benchmark (rMAE < 1 beats it)."""
    err = preds["pred"] - preds["y"]
    out = (
        preds.assign(abs_err=err.abs(), sq_err=err**2, err=err)
        .groupby("model", sort=False)
        .agg(mae=("abs_err", "mean"), rmse=("sq_err", lambda s: np.sqrt(s.mean())), bias=("err", "mean"), hours=("y", "size"))
    )
    out["rmae"] = out["mae"] / out.loc[reference, "mae"]
    if "lo" in preds:
        has = preds.dropna(subset=["lo"])
        inside = lambda lo, hi: ((has["y"] >= has[lo]) & (has["y"] <= has[hi])).groupby(has["model"]).mean()
        out["coverage_raw"] = inside("lo_raw", "hi_raw")
        out["coverage"] = inside("lo", "hi")
    return out.round(3)


def validation() -> pd.DataFrame:
    train_feats, test_feats = load_features()
    models = {
        Naive.name: Naive,
        "LEAR, 1-year window": lambda: LEAR(window_days=365),
        "LEAR, 2-year window": lambda: LEAR(window_days=730),
        "LEAR, all history": lambda: LEAR(window_days=None),
        "LightGBM, 2-year window": lambda: GBM(window_days=730),
        "LightGBM, all history": lambda: GBM(window_days=None),
        "LightGBM, no weather": lambda: GBM(weather=False),
        "LightGBM, + all 96 hourly lags": lambda: GBM(day_lags=True, interval=INTERVAL),
        "LightGBM, raw prices (no asinh)": lambda: GBM(asinh=False),
        "Average of LEAR (2 years) and LightGBM": lambda: Ensemble(LEAR(window_days=730), GBM()),
    }
    preds = rolling_backtest(models, train_feats, test_feats, *VALIDATION)
    table = metrics(preds, Naive.name)
    table.to_csv(PROCESSED / "validation_metrics.csv")
    return table


def test() -> pd.DataFrame:
    train_feats, test_feats = load_features()
    models = {
        Naive.name: Naive,
        WeeklyNaive.name: WeeklyNaive,
        LEAR.name: final_lear,
        "LightGBM, prices + calendar": lambda: GBM(weather=False, day_lags=True),
        "LightGBM, prices + calendar + weather": lambda: GBM(day_lags=True, interval=INTERVAL),
        FINAL: final_model,
    }
    preds = rolling_backtest(models, train_feats, test_feats, *TEST)
    print("  with observed weather instead of forecasts:")
    oracle = rolling_backtest({FINAL + " (observed weather)": final_model}, train_feats, train_feats, *TEST)
    preds = pd.concat([preds, oracle], ignore_index=True)
    preds.to_parquet(PROCESSED / "backtest_predictions.parquet", index=False)
    table = metrics(preds, Naive.name)
    table.to_csv(PROCESSED / "backtest_metrics.csv")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validation", action="store_true", help="run the model choices on the validation year")
    args = parser.parse_args()
    started = time.time()
    table = validation() if args.validation else test()
    pd.set_option("display.width", 140)
    print(table)
    print(f"done in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
