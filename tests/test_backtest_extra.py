"""
Zephyr – Unit tests for the backtest engine with generated signals
(models/backtest.py + models/signals.py)
"""

import numpy as np
import pandas as pd
import pytest

from models.backtest import run_backtest, BacktestConfig
from models.signals import signals_from_predictions


def _prices(n=60):
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    rng = np.random.default_rng(1)
    close = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, n))
    return pd.Series(close, index=idx)


def test_backtest_basic_fields():
    prices = _prices()
    preds = np.random.default_rng(2).normal(0, 0.01, len(prices))
    signals = signals_from_predictions(preds, threshold=0.0)

    result = run_backtest(prices, signals, BacktestConfig())

    assert result.trades >= 0
    assert 0.0 <= result.win_rate <= 1.0
    assert result.final_value > 0
    assert result.total_return >= -1.0
    assert len(result.equity_curve) == len(prices)


def test_backtest_signal_length_mismatch():
    prices = _prices(30)
    signals = np.array([1, 0, -1])
    with pytest.raises(ValueError):
        run_backtest(prices, signals)


def test_backtest_all_flat_no_trades():
    prices = _prices()
    signals = np.zeros(len(prices), dtype=int)
    result = run_backtest(prices, signals, BacktestConfig())
    assert result.trades == 0
    assert result.final_value == pytest.approx(BacktestConfig().initial_capital)