import numpy as np
import pandas as pd
import pytest

from se3forecast.evaluation import block_bootstrap, dm_test, interval_score, pinball


def test_dm_detects_a_clearly_better_model():
    rng = np.random.default_rng(1)
    worse = rng.gamma(4, 5, 365)
    better = worse * 0.8 + rng.normal(0, 1, 365)
    stat, p = dm_test(better, worse)
    assert stat > 0 and p < 0.001
    _, p_reverse = dm_test(worse, better)
    assert p_reverse > 0.999


def test_dm_is_not_significant_for_noise():
    rng = np.random.default_rng(2)
    a = rng.gamma(4, 5, 365)
    b = a + rng.normal(0, 1, 365)
    _, p = dm_test(a, b)
    assert 0.01 < p < 0.99


def test_dm_on_identical_losses_returns_one_half():
    loss = np.arange(10.0)
    assert dm_test(loss, loss) == (0.0, 0.5)


def test_pinball_loss_known_values():
    y = np.array([10.0, 10.0])
    # Forecast of the 10th percentile: under-forecasting costs 0.1 per unit, over-forecasting 0.9.
    assert pinball(y, np.array([8.0, 8.0]), 0.1) == pytest.approx(0.2)
    assert pinball(y, np.array([12.0, 12.0]), 0.1) == pytest.approx(1.8)


def test_interval_score_penalises_misses():
    y = np.array([5.0, 15.0])
    lo, hi = np.array([0.0, 0.0]), np.array([10.0, 10.0])
    # Width 10 for both, plus 2/0.2 * 5 for the miss on the second.
    assert interval_score(y, lo, hi, alpha=0.2) == pytest.approx(10 + 50 / 2)


def test_block_bootstrap_brackets_the_estimate():
    rng = np.random.default_rng(3)
    loss = pd.DataFrame({"naive": rng.gamma(4, 5, 365)})
    loss["model"] = loss["naive"] * 0.6
    ci = block_bootstrap(loss, reference="naive", n_boot=300)
    assert ci.loc["model", "mae_low"] < ci.loc["model", "mae"] < ci.loc["model", "mae_high"]
    assert ci.loc["model", "improvement"] == pytest.approx(0.4)
