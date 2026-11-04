"""
Zephyr – Simple forecasting baselines
Naive last-close, last-return and optional moving-average baseline.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class NaiveLastClose(BaseEstimator, RegressorMixin):
    """
    Predicts a zero return, equivalent to carrying the last close forward.
    """

    def __init__(self):
        self.last_value_: float = 0.0

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        n = len(X) if hasattr(X, "__len__") else 1
        return np.full(n, self.last_value_)


class NaiveLastReturn(BaseEstimator, RegressorMixin):
    """
    Repeats the last observed one-day return over the selected horizon.
    """

    def __init__(self, horizon: int = 1):
        self.horizon = horizon
        self.last_return_: Optional[float] = None

    def fit(self, X, y=None):
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int) or self.horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if not isinstance(X, pd.DataFrame) or "return_1d" not in X.columns:
            raise ValueError("NaiveLastReturn requires a 'return_1d' feature column")
        self.last_return_ = float(X["return_1d"].iloc[-1])
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame) and "return_1d" in X.columns:
            daily_returns = X["return_1d"].to_numpy(dtype=float)
            return np.power(1 + daily_returns, self.horizon) - 1
        if self.last_return_ is None:
            raise ValueError("NaiveLastReturn must be fitted before prediction")
        n = len(X) if hasattr(X, "__len__") else 1
        return np.full(n, np.power(1 + self.last_return_, self.horizon) - 1)


class MovingAverageBaseline(BaseEstimator, RegressorMixin):
    """
    Predicts the next close as the simple moving average of the last `window` closes.
    """

    def __init__(self, window: int = 5):
        self.window = window
        self.ma_: Optional[float] = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            series = X["Close"] if "Close" in X.columns else X.iloc[:, 0]
        else:
            series = pd.Series(np.asarray(X).ravel())

        self.ma_ = float(series.tail(self.window).mean())
        return self

    def predict(self, X):
        n = len(X) if hasattr(X, "__len__") else 1
        return np.full(n, self.ma_)