"""Is the improvement real? Significance tests, interval scores and confidence intervals
for the test-year backtest in ``data/processed/backtest_predictions.parquet``.

- **Diebold-Mariano tests** between every pair of models, on daily losses (the mean
  absolute error over the 24 hours of each day, as in Lago et al., 2021), with the
  Harvey, Leybourne and Newbold (1997) small-sample correction.
- **Pinball loss and interval score** for the 80% prediction interval, compared with a
  naive interval: the naive forecast plus the 10th and 90th percentiles of its own past
  errors for that hour (historical simulation).
- **Block bootstrap confidence intervals** for each model's MAE and its improvement over
  the naive forecast, resampling whole weeks so that correlated days stay together.

Usage::

    python -m se3forecast.evaluation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import PROCESSED
from .backtest import INTERVAL, TEST, load_features
from .models import Naive

INTERVAL_MODEL = "LightGBM, prices + calendar + weather"


def daily_loss(preds: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute error per day (rows) and model (columns)."""
    return (
        preds.assign(abs_err=(preds["pred"] - preds["y"]).abs())
        .pivot_table(index="date", columns="model", values="abs_err", aggfunc="mean")
        .reindex(columns=list(dict.fromkeys(preds["model"])))
    )


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, horizon: int = 1) -> tuple[float, float]:
    """One-sided Diebold-Mariano test of H0: model A is not more accurate than model B.

    Returns the HLN-corrected statistic and its p-value from a t distribution with n-1
    degrees of freedom. A small p-value means A's losses are significantly lower.
    Day-ahead forecasts are one step ahead per day, so the default ``horizon`` of 1 uses
    the plain variance of the loss differential.
    """
    d = np.asarray(loss_b, dtype=float) - np.asarray(loss_a, dtype=float)
    n = len(d)
    mean = d.mean()
    centred = d - mean
    gamma = [centred @ centred / n] + [centred[k:] @ centred[:-k] / n for k in range(1, horizon)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return 0.0, 0.5
    dm = mean / np.sqrt(var)
    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    stat = dm * correction
    return float(stat), float(stats.t.sf(stat, df=n - 1))


def dm_matrix(loss: pd.DataFrame) -> pd.DataFrame:
    """p-value of 'row model is more accurate than column model' for every pair."""
    models = loss.columns
    out = pd.DataFrame(np.nan, index=models, columns=models)
    for a in models:
        for b in models:
            if a != b:
                out.loc[a, b] = dm_test(loss[a].to_numpy(), loss[b].to_numpy())[1]
    return out


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    """Mean pinball (quantile) loss: the proper score for a forecast of the tau quantile."""
    diff = np.asarray(y) - np.asarray(q)
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> float:
    """Winkler interval score: the width, plus 2/alpha times every miss. Lower is better."""
    y, lo, hi = map(np.asarray, (y, lo, hi))
    penalty = (2 / alpha) * ((lo - y) * (y < lo) + (y - hi) * (y > hi))
    return float(np.mean(hi - lo + penalty))


def naive_interval(start: str = TEST[0], end: str = TEST[1], history_days: int = 365) -> pd.DataFrame:
    """Naive forecast plus empirical quantiles of its errors over the previous year, per
    hour of the day, refreshed at the start of every month like the other models."""
    feats, _ = load_features()
    feats = feats.dropna(subset=["y", "p_lag1", "p_lag7"]).copy()
    feats["naive"] = Naive().predict(feats)
    feats["err"] = feats["y"] - feats["naive"]
    rows = []
    end_ts = pd.Timestamp(end)
    for month in pd.date_range(start, end, freq="MS"):
        month_end = min(month + pd.offsets.MonthEnd(0), end_ts)
        past = feats[feats["date"].between(month - pd.Timedelta(days=history_days), month - pd.Timedelta(days=1))]
        q = past.groupby("hour")["err"].quantile(list(INTERVAL)).unstack()
        cur = feats[feats["date"].between(month, month_end)][["date", "hour", "y", "naive"]].copy()
        cur["lo"] = cur["naive"] + cur["hour"].map(q[INTERVAL[0]])
        cur["hi"] = cur["naive"] + cur["hour"].map(q[INTERVAL[1]])
        rows.append(cur)
    return pd.concat(rows, ignore_index=True)


def interval_scores(preds: pd.DataFrame) -> pd.DataFrame:
    alpha = 1 - (INTERVAL[1] - INTERVAL[0])
    model = preds[preds["model"] == INTERVAL_MODEL]
    naive = naive_interval()
    rows = {}
    for name, df, lo, hi in [
        ("Naive + past errors (historical simulation)", naive, "lo", "hi"),
        ("LightGBM quantiles, raw", model, "lo_raw", "hi_raw"),
        ("LightGBM quantiles, conformal", model, "lo", "hi"),
    ]:
        y = df["y"].to_numpy()
        rows[name] = {
            "coverage": df["y"].between(df[lo], df[hi]).mean(),
            "mean_width": (df[hi] - df[lo]).mean(),
            "pinball_q10": pinball(y, df[lo], INTERVAL[0]),
            "pinball_q90": pinball(y, df[hi], INTERVAL[1]),
            "interval_score": interval_score(y, df[lo], df[hi], alpha),
        }
    return pd.DataFrame(rows).T.round(3)


def block_bootstrap(loss: pd.DataFrame, reference: str = Naive.name, block_days: int = 7,
                    n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """95% confidence intervals for each model's MAE and its improvement over the reference.

    Uses a moving block bootstrap over days: blocks of ``block_days`` consecutive days are
    drawn with replacement until the sample is as long as the test year.
    """
    rng = np.random.default_rng(seed)
    values = loss.to_numpy()
    n = len(values)
    starts = rng.integers(0, n - block_days + 1, size=(n_boot, int(np.ceil(n / block_days))))
    idx = (starts[:, :, None] + np.arange(block_days)).reshape(n_boot, -1)[:, :n]
    boot_mae = values[idx].mean(axis=1)  # n_boot x models
    ref = loss.columns.get_loc(reference)
    improvement = 1 - boot_mae / boot_mae[:, [ref]]
    out = pd.DataFrame(
        {
            "mae": values.mean(axis=0),
            "mae_low": np.percentile(boot_mae, 2.5, axis=0),
            "mae_high": np.percentile(boot_mae, 97.5, axis=0),
            "improvement": 1 - values.mean(axis=0) / values[:, ref].mean(),
            "improvement_low": np.percentile(improvement, 2.5, axis=0),
            "improvement_high": np.percentile(improvement, 97.5, axis=0),
        },
        index=loss.columns,
    )
    return out.round(3)


def main() -> None:
    preds = pd.read_parquet(PROCESSED / "backtest_predictions.parquet")
    loss = daily_loss(preds)
    pvals = dm_matrix(loss)
    pvals.to_csv(PROCESSED / "dm_pvalues.csv")
    ci = block_bootstrap(loss)
    ci.to_csv(PROCESSED / "bootstrap_ci.csv")
    scores = interval_scores(preds)
    scores.to_csv(PROCESSED / "interval_scores.csv")
    pd.set_option("display.width", 160)
    print("Diebold-Mariano p-values (row more accurate than column):")
    print(pvals.round(3))
    print("\nMAE with 95% block bootstrap intervals:")
    print(ci)
    print("\n80% prediction intervals:")
    print(scores)


if __name__ == "__main__":
    main()
