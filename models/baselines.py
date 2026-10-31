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
    Predicts the next close as the last observed close.
    (Equivalent to predicting a zero return.)
    """

    def __init__(self):
        self.last_value_: Optional[float] = None

    def fit(self, X, y=None):
        # X is expected to be a DataFrame/Series of Close prices or features
        # We only need the last close value from the training set
        if isinstance(X, pd.DataFrame):
            if "Close" in X.columns:
                self.last_value_ = float(X["Close"].iloc[-1])
            else:
                self.last_value_ = float(X.iloc[-1, 0])
        else:
            self.last_value_ = float(np.asarray(X).ravel()[-1])
        return self

    def predict(self, X):
        n = len(X) if hasattr(X, "__len__") else 1
        return np.full(n, self.last_value_)


class NaiveLastReturn(BaseEstimator, RegressorMixin):
    """
    Predicts the next return as the last observed return.
    """

    def __init__(self):
        self.last_return_: Optional[float] = None

    def fit(self, X, y=None):
        if y is not None:
            y_arr = np.asarray(y).ravel()
            self.last_return_ = float(y_arr[-1]) if len(y_arr) > 0 else 0.0
        else:
            # Fallback: try to compute from X if it contains returns
            if isinstance(X, pd.DataFrame) and "return_1d" in X.columns:
                self.last_return_ = float(X["return_1d"].iloc[-1])
            else:
                self.last_return_ = 0.0
        return self

    def predict(self, X):
        n = len(X) if hasattr(X, "__len__") else 1
        return np.full(n, self.last_return_)


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