"""
Zephyr – Metric helpers
Lightweight wrappers around common regression / directional metrics.
"""

from __future__ import annotations

from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import RISK_FREE_RATE, TRADING_DAYS, CONFIDENCE_LEVEL


def mae(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def directional_accuracy(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> float:
    """
    Percentage of times the predicted direction matches the actual direction.
    Zero actual returns are ignored.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0

    correct = np.sign(y_true[mask]) == np.sign(y_pred[mask])
    return float(np.mean(correct) * 100)


def evaluate_forecast(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> Dict[str, float]:
    """
    Convenience function returning MAE, RMSE and Directional Accuracy.
    """
    return {
        "MAE": round(mae(y_true, y_pred), 6),
        "RMSE": round(rmse(y_true, y_pred), 6),
        "DirectionalAccuracy": round(directional_accuracy(y_true, y_pred), 2),
    }


def sharpe_ratio(returns: Union[np.ndarray, pd.Series], risk_free_rate: float = RISK_FREE_RATE) -> float:
    """Annualized Sharpe ratio of a return series."""
    r = np.asarray(returns).ravel()
    excess = r - risk_free_rate / TRADING_DAYS
    if excess.std() == 0 or np.isnan(excess.std()):
        return 0.0
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / excess.std())


def max_drawdown(returns: Union[np.ndarray, pd.Series]) -> float:
    """Maximum drawdown of a cumulative return series."""
    r = np.asarray(returns).ravel()
    cum = np.cumprod(1 + r)
    running_max = np.maximum.accumulate(cum)
    drawdown = (cum - running_max) / running_max
    return float(drawdown.min() * 100)


def value_at_risk(
    returns: Union[np.ndarray, pd.Series], confidence: float = CONFIDENCE_LEVEL
) -> float:
    """Value at Risk (historical) at the given confidence level."""
    r = np.asarray(returns).ravel()
    return float(np.percentile(r, (1 - confidence) * 100) * 100)