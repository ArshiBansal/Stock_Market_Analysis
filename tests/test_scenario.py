"""
Zephyr – Unit tests for the Scenario Lab
(models/scenario.py)
"""

import numpy as np
import pandas as pd
import pytest

from models.scenario import (
    simulate_paths,
    fan_chart_stats,
    prob_up,
    prob_down,
    first_passage_prob,
    detect_vol_regimes,
    regime_persistence,
    shock_impact,
)


def _returns(n=252, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0005, 0.01, n))


def test_simulate_paths_shape_and_seed_reproducibility():
    r = _returns()
    p1 = simulate_paths(r, n_paths=50, horizon=10, method="bootstrap", seed=42)
    p2 = simulate_paths(r, n_paths=50, horizon=10, method="bootstrap", seed=42)
    assert p1.shape == (50, 10)
    assert np.allclose(p1, p2)


def test_simulate_paths_method_validation():
    r = _returns()
    with pytest.raises(ValueError):
        simulate_paths(r, method="other")


def test_simulate_paths_too_few_returns():
    with pytest.raises(ValueError):
        simulate_paths(pd.Series([0.01]), n_paths=10, horizon=5)


def test_fan_chart_monotonic_percentiles():
    r = _returns()
    paths = simulate_paths(r, n_paths=200, horizon=15, seed=1)
    stats = fan_chart_stats(paths)
    assert np.all(stats["p05"] <= stats["p25"])
    assert np.all(stats["p25"] <= stats["p50"])
    assert np.all(stats["p50"] <= stats["p75"])
    assert np.all(stats["p75"] <= stats["p95"])


def test_prob_up_and_down_sum_to_100():
    r = _returns()
    paths = simulate_paths(r, n_paths=500, horizon=10, seed=2)
    total = prob_up(paths) + prob_down(paths)
    assert total == pytest.approx(100.0, abs=1.0)


def test_first_passage_prob_range():
    r = _returns()
    paths = simulate_paths(r, n_paths=300, horizon=20, seed=3)
    p = first_passage_prob(paths, target=0.05, stop=-0.05)
    assert 0.0 <= p <= 100.0


def test_first_passage_validation():
    r = _returns()
    paths = simulate_paths(r, n_paths=50, horizon=10, seed=4)
    with pytest.raises(ValueError):
        first_passage_prob(paths, target=-0.05, stop=-0.05)
    with pytest.raises(ValueError):
        first_passage_prob(paths, target=0.05, stop=0.05)


def test_detect_vol_regimes_labels():
    r = _returns(n=300)
    regimes = detect_vol_regimes(r, window=20)
    valid = {"Low", "Normal", "High", "Warmup"}
    assert set(regimes.dropna().unique()).issubset(valid)
    assert len(regimes) == len(r)


def test_regime_persistence_shares_sum_to_100():
    r = _returns(n=400, seed=9)
    regimes = detect_vol_regimes(r, window=20)
    persist = regime_persistence(regimes)
    share = persist["share_pct"]
    total = sum(share.get(k, 0.0) for k in ["Low", "Normal", "High"])
    assert total == pytest.approx(100.0, abs=1.0)


def test_shock_impact_recovery_non_negative():
    idx = pd.date_range("2022-01-01", periods=300, freq="B")
    rng = np.random.default_rng(5)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, 300))
    prices = pd.Series(close, index=idx)

    # Use a shock magnitude that actually occurs in this history.
    impact = shock_impact(prices, shock_pct=-0.02)
    assert impact["shock_pct"] == pytest.approx(-0.02)
    assert impact["recovery_days_median"] >= 0
    assert impact["recovery_days_max"] >= impact["recovery_days_median"]


def test_shock_impact_validation():
    prices = pd.Series([100, 101, 102, 103])
    with pytest.raises(ValueError):
        shock_impact(prices, shock_pct=-1.5)