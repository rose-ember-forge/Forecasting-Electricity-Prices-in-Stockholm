"""Forecasting models: two naive baselines, a LEAR benchmark and gradient boosting.

All models share one interface: ``fit(train)`` on a feature frame from
:func:`se3forecast.features.make_features`, then ``predict(test)`` returns a
price in EUR/MWh for every row.
"""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsIC
from sklearn.preprocessing import StandardScaler

from .features import CALENDAR_FEATURES, DAY_LAG_COLUMNS, PRICE_FEATURES, WEATHER_FEATURES


class AsinhScaler:
    """Variance-stabilising transform for prices: asinh of the robustly standardised value.

    Day-ahead prices have spikes of several hundred EUR/MWh and can go negative, so a log
    does not work. ``asinh`` behaves like a log for large values and is linear around zero.
    """

    def fit(self, prices: pd.Series | np.ndarray) -> AsinhScaler:
        prices = np.asarray(prices, dtype=float)
        self.median = np.nanmedian(prices)
        self.mad = np.nanmedian(np.abs(prices - self.median)) / 0.6745 or 1.0
        return self

    def transform(self, prices):
        return np.arcsinh((np.asarray(prices, dtype=float) - self.median) / self.mad)

    def inverse(self, z):
        return np.sinh(np.asarray(z, dtype=float)) * self.mad + self.median


class Identity:
    def transform(self, prices):
        return np.asarray(prices, dtype=float)

    inverse = transform


class Naive:
    """The standard naive benchmark: same hour yesterday, but same hour last week on Monday,
    Saturday and Sunday, when yesterday was a different kind of day (Lago et al., 2021)."""

    name = "Naive (yesterday / last week)"

    def fit(self, train: pd.DataFrame) -> Naive:
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        use_week = test["dow"].isin([0, 5, 6])
        return np.where(use_week, test["p_lag7"], test["p_lag1"])


class WeeklyNaive:
    name = "Same hour last week"

    def fit(self, train: pd.DataFrame) -> WeeklyNaive:
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return test["p_lag7"].to_numpy()


class LEAR:
    """Lasso-estimated autoregressive model, the linear benchmark in the price forecasting
    literature (Uniejewski, Nowotarski and Weron, 2016; Lago et al., 2021).

    One sparse linear model per hour on asinh-transformed prices. Its inputs are all 24
    prices of the days 1, 2, 3 and 7 days back, yesterday's mean prices in the other zones,
    the weather and the day of the week. The Lasso penalty is chosen by AIC.
    """

    name = "LEAR (Lasso, one model per hour)"

    def __init__(self, weather: bool = True, window_days: int | None = 730):
        self.weather = weather
        self.window_days = window_days

    def _design(self, df: pd.DataFrame) -> pd.DataFrame:
        price_cols = DAY_LAG_COLUMNS + ["se1_d1_mean", "se2_d1_mean", "se4_d1_mean"]
        parts = [pd.DataFrame(self.scaler.transform(df[price_cols]), index=df.index, columns=price_cols)]
        if self.weather:
            parts.append(df[WEATHER_FEATURES])
        parts.append(pd.get_dummies(df["dow"], prefix="dow").reindex(columns=[f"dow_{d}" for d in range(7)], fill_value=0))
        parts.append(df[["holiday"]])
        return pd.concat(parts, axis=1).astype(float)

    def fit(self, train: pd.DataFrame) -> LEAR:
        if self.window_days:
            train = train[train["date"] > train["date"].max() - pd.Timedelta(days=self.window_days)]
        self.scaler = AsinhScaler().fit(train["y"])
        self.models = {}
        for h, part in train.groupby("hour"):
            x = self._design(part)
            std = StandardScaler().fit(x)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = LassoLarsIC(criterion="aic").fit(std.transform(x), self.scaler.transform(part["y"]))
            self.models[h] = (std, model)
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        out = pd.Series(np.nan, index=test.index)
        for h, part in test.groupby("hour"):
            std, model = self.models[h]
            out[part.index] = model.predict(std.transform(self._design(part)))
        return self.scaler.inverse(out.to_numpy())


GBM_PARAMS = dict(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=1.0,
    verbose=-1,
)


class GBM:
    """LightGBM on the same-hour lags, yesterday's level and shape, calendar and weather.

    One model for all 24 hours (the hour is a feature). It is trained on the asinh scale so
    the 2022 crisis prices do not dominate the loss.

    With ``interval=(lo, hi)`` it also fits two quantile models for a prediction interval.
    Raw quantile models tend to give intervals that are too narrow, so the interval is
    widened by split conformal calibration (conformalised quantile regression, Romano et
    al., 2019): quantile models are fitted without the last ``calibration_days`` of the
    training data, the widening that would have made the interval cover ``hi - lo`` of those
    days is measured, and that widening is added to quantile models refitted on all data.
    """

    def __init__(
        self,
        weather: bool = True,
        interval: tuple[float, float] | None = None,
        calibration_days: int = 90,
        window_days: int | None = None,
        day_lags: bool = False,
        asinh: bool = True,
        seed: int = 0,
    ):
        self.weather = weather
        self.interval = interval
        self.calibration_days = calibration_days
        self.window_days = window_days
        self.asinh = asinh
        self.seed = seed
        self.features = PRICE_FEATURES + CALENDAR_FEATURES + (WEATHER_FEATURES if weather else [])
        if day_lags:
            self.features += DAY_LAG_COLUMNS
        self.name = "LightGBM, prices + calendar + weather" if weather else "LightGBM, prices + calendar"

    def fit(self, train: pd.DataFrame) -> GBM:
        if self.window_days:
            train = train[train["date"] > train["date"].max() - pd.Timedelta(days=self.window_days)]
        self.scaler = AsinhScaler().fit(train["y"]) if self.asinh else Identity()
        z = self.scaler.transform(train["y"])
        x = train[self.features]
        self.model = lgb.LGBMRegressor(**GBM_PARAMS, random_state=self.seed).fit(x, z)
        if self.interval:
            calib = (train["date"] > train["date"].max() - pd.Timedelta(days=self.calibration_days)).to_numpy()
            lo, hi = self._fit_quantiles(x[~calib], z[~calib])
            score = np.maximum(lo.predict(x[calib]) - z[calib], z[calib] - hi.predict(x[calib]))
            level = self.interval[1] - self.interval[0]
            n = len(score)
            self.widening = np.quantile(score, min(1.0, np.ceil((n + 1) * level) / n))
            self.quantile_models = self._fit_quantiles(x, z)
        return self

    def _fit_quantiles(self, x, z):
        return tuple(
            lgb.LGBMRegressor(**GBM_PARAMS, objective="quantile", alpha=q, random_state=self.seed).fit(x, z)
            for q in self.interval
        )

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return self.scaler.inverse(self.model.predict(test[self.features]))

    def predict_interval(self, test: pd.DataFrame) -> pd.DataFrame:
        """Raw quantile forecasts (``lo_raw``, ``hi_raw``) and the calibrated interval (``lo``, ``hi``)."""
        lo, hi = (m.predict(test[self.features]) for m in self.quantile_models)
        return pd.DataFrame(
            {
                "lo_raw": self.scaler.inverse(lo),
                "hi_raw": self.scaler.inverse(hi),
                "lo": self.scaler.inverse(lo - self.widening),
                "hi": self.scaler.inverse(hi + self.widening),
            },
            index=test.index,
        )


class Ensemble:
    """Plain average of the LEAR and LightGBM forecasts."""

    name = "Average of LEAR and LightGBM"

    def __init__(self, *members):
        self.members = members

    def fit(self, train: pd.DataFrame) -> Ensemble:
        for m in self.members:
            m.fit(train)
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict(test) for m in self.members], axis=0)
