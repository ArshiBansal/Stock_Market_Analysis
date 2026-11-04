"""
Zephyr – Feature engineering
Creates technical / lag features that use ONLY information available
at prediction time (no future leakage).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from config import MA_WINDOWS, VOLATILITY_WINDOW, MOMENTUM_WINDOW

logger = logging.getLogger(__name__)


def _safe_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """Percentage change that avoids division by zero."""
    return series.pct_change(periods=periods).replace([np.inf, -np.inf], np.nan)


def add_technical_features(
    df: pd.DataFrame,
    ma_windows: Optional[List[int]] = None,
    volatility_window: int = VOLATILITY_WINDOW,
    momentum_window: int = MOMENTUM_WINDOW,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Add a rich set of technical and lag features.

    All features are computed with past information only
    (shift / rolling with default closed="right" behaviour).

    Parameters
    ----------
    df : OHLCV DataFrame (must contain Open, High, Low, Close, Volume)
    ma_windows : list of moving-average windows (default from config)
    volatility_window : window for rolling volatility
    momentum_window : window for momentum
    dropna : whether to drop rows that contain NaNs after feature creation

    Returns
    -------
    DataFrame with original columns + new feature columns
    """
    if df.empty:
        return df

    df = df.copy()
    ma_windows = ma_windows or MA_WINDOWS

    # ------------------------------------------------------------------
    # Returns
    # ------------------------------------------------------------------
    df["return_1d"] = _safe_pct_change(df["Close"], 1)
    df["log_return_1d"] = np.log(df["Close"] / df["Close"].shift(1))

    # Lagged returns (explicit lags)
    for lag in [1, 2, 3, 5]:
        df[f"return_lag_{lag}"] = df["return_1d"].shift(lag)

    # ------------------------------------------------------------------
    # Moving averages
    # ------------------------------------------------------------------
    for w in ma_windows:
        df[f"sma_{w}"] = df["Close"].rolling(window=w, min_periods=w).mean()
        df[f"ema_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()
        # Distance from MA (normalised)
        df[f"close_sma_{w}_ratio"] = df["Close"] / df[f"sma_{w}"] - 1

    # ------------------------------------------------------------------
    # Volatility
    # ------------------------------------------------------------------
    df["volatility"] = (
        df["return_1d"]
        .rolling(window=volatility_window, min_periods=volatility_window)
        .std()
    )
    df["volatility_annualised"] = df["volatility"] * np.sqrt(252)

    # ------------------------------------------------------------------
    # Momentum
    # ------------------------------------------------------------------
    df["momentum"] = df["Close"] / df["Close"].shift(momentum_window) - 1

    # ------------------------------------------------------------------
    # Price range & candle features
    # ------------------------------------------------------------------
    df["daily_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["body"] = (df["Close"] - df["Open"]) / df["Open"]
    df["upper_shadow"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / df["Close"]
    df["lower_shadow"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / df["Close"]

    # ------------------------------------------------------------------
    # Volume features
    # ------------------------------------------------------------------
    df["volume_change"] = _safe_pct_change(df["Volume"], 1)
    df["volume_sma_20"] = df["Volume"].rolling(20, min_periods=10).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_sma_20"]

    # ------------------------------------------------------------------
    # 52-week high / low (only when enough history exists)
    # ------------------------------------------------------------------
    window_252 = min(252, len(df))
    if window_252 >= 60:  # require at least ~3 months
        df["high_52w"] = df["High"].rolling(window=window_252, min_periods=60).max()
        df["low_52w"] = df["Low"].rolling(window=window_252, min_periods=60).min()
        df["dist_from_52w_high"] = df["Close"] / df["high_52w"] - 1
        df["dist_from_52w_low"] = df["Close"] / df["low_52w"] - 1

    # ------------------------------------------------------------------
    # Target (next-period return / close) – for supervised learning
    # IMPORTANT: this is the ONLY place a future value is introduced.
    # It must be shifted later when creating X/y.
    # ------------------------------------------------------------------
    df["target_return_1d"] = df["return_1d"].shift(-1)
    df["target_close"] = df["Close"].shift(-1)

    if dropna:
        # Keep the latest feature row available for inference even without a future target.
        feature_cols = get_feature_columns(df)
        df = df.dropna(subset=feature_cols)

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Return the list of columns that should be used as model features.
    Excludes OHLCV raw prices, target columns and metadata.
    """
    exclude = {
        "Open", "High", "Low", "Close", "Volume",
        "Ticker", "Region",
        "target_return_1d", "target_close",
        "high_52w", "low_52w",  # optional – can keep if desired
    }
    # Also exclude any remaining pure price columns if present
    candidates = [
        c for c in df.columns
        if c not in exclude
        and not c.startswith("target_")
        and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]
    return candidates


def create_supervised_dataset(
    df: pd.DataFrame,
    target_col: str = "target_return_1d",
    dropna: bool = True,
    horizon: int = 1,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split a feature-enhanced DataFrame into X (features) and y (target).

    For return targets, ``horizon`` is the cumulative close-to-close return over
    that many trading observations. Features remain aligned to the start date.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")

    feature_cols = get_feature_columns(df)

    X = df[feature_cols].copy()
    if target_col == "target_return_1d":
        y = df["Close"].shift(-horizon).div(df["Close"]).sub(1).rename(target_col)
    else:
        y = df[target_col].copy()

    if dropna:
        mask = X.notna().all(axis=1) & y.notna()
        X = X.loc[mask]
        y = y.loc[mask]

    return X, y