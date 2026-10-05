import numpy as np
import pandas as pd

from se3forecast.backtest import metrics
from se3forecast.models import AsinhScaler, Naive


def test_asinh_scaler_round_trips_negative_and_spike_prices():
    prices = np.array([-20.0, 0.0, 35.0, 80.0, 800.0])
    scaler = AsinhScaler().fit(prices)
    assert np.allclose(scaler.inverse(scaler.transform(prices)), prices)


def test_naive_uses_last_week_on_monday_and_weekend():
    test = pd.DataFrame({"dow": [0, 1, 4, 5, 6], "p_lag1": [1.0] * 5, "p_lag7": [7.0] * 5})
    assert Naive().predict(test).tolist() == [7.0, 1.0, 1.0, 7.0, 7.0]


def test_metrics_relative_to_reference():
    preds = pd.DataFrame(
        {
            "model": ["a", "a", "b", "b"],
            "y": [10.0, 20.0, 10.0, 20.0],
            "pred": [12.0, 18.0, 11.0, 19.0],
        }
    )
    m = metrics(preds, "a")
    assert m.loc["a", "mae"] == 2.0
    assert m.loc["b", "rmae"] == 0.5
