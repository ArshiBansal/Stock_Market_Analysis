"""
Zephyr – Data preprocessing
Cleaning, missing-value handling and basic quality checks.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning of a single-ticker OHLCV DataFrame.

    - Ensures correct dtypes
    - Removes rows with non-positive prices
    - Sorts by date
    - Drops exact duplicate timestamps
    """
    if df.empty:
        return df

    df = df.copy()

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Numeric conversion
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows with non-positive prices
    price_cols = ["Open", "High", "Low", "Close"]
    mask = (df[price_cols] > 0).all(axis=1)
    removed = (~mask).sum()
    if removed > 0:
        logger.debug(f"Removed {removed} rows with non-positive prices")
        df = df.loc[mask]

    return df


def handle_missing_values(
    df: pd.DataFrame,
    method: str = "ffill",
    max_ffill: int = 5,
) -> pd.DataFrame:
    """
    Handle missing values in OHLCV data.

    Parameters
    ----------
    method : "ffill" | "bfill" | "drop" | "none"
    max_ffill : maximum consecutive fills allowed
    """
    if df.empty:
        return df

    df = df.copy()

    if method == "none":
        return df

    if method == "drop":
        return df.dropna(subset=["Open", "High", "Low", "Close"])

    # Forward / backward fill with limit
    if method in ("ffill", "bfill"):
        df[["Open", "High", "Low", "Close"]] = (
            df[["Open", "High", "Low", "Close"]]
            .fillna(method=method, limit=max_ffill)
        )
        # Volume – fill with 0 after limited ffill
        df["Volume"] = df["Volume"].fillna(method=method, limit=max_ffill)
        df["Volume"] = df["Volume"].fillna(0)

    # Final safety drop of any remaining NaNs in price columns
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    return df


def quality_report(df: pd.DataFrame, ticker: str = "") -> Dict:
    """
    Generate a simple data-quality summary for a DataFrame.
    """
    if df.empty:
        return {
            "ticker": ticker,
            "rows": 0,
            "missing_pct": 100.0,
            "date_start": None,
            "date_end": None,
            "status": "EMPTY",
        }

    total_cells = df[["Open", "High", "Low", "Close", "Volume"]].size
    missing_cells = df[["Open", "High", "Low", "Close", "Volume"]].isna().sum().sum()
    missing_pct = (missing_cells / total_cells * 100) if total_cells else 0.0

    return {
        "ticker": ticker,
        "rows": len(df),
        "missing_pct": round(missing_pct, 2),
        "date_start": df.index.min().date() if len(df) else None,
        "date_end": df.index.max().date() if len(df) else None,
        "status": "OK" if missing_pct < 5 else "WARNING",
    }


def preprocess_pipeline(
    raw: Dict[str, pd.DataFrame],
    fill_method: str = "ffill",
) -> Dict[str, pd.DataFrame]:
    """
    Full preprocessing pipeline applied to a dict of ticker → DataFrame.
    """
    cleaned = {}
    for ticker, df in raw.items():
        df = clean_ohlcv(df)
        df = handle_missing_values(df, method=fill_method)
        if not df.empty:
            cleaned[ticker] = df
        else:
            logger.warning(f"{ticker}: became empty after preprocessing")
    return cleaned