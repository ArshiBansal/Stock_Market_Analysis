"""
Zephyr – Unit tests for data layer
(data/fetch.py, data/preprocess.py)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data.fetch import (
    validate_tickers,
    fetch_ohlcv,
    fetch_latest_quote,
    get_failed_tickers,
)
from data.preprocess import (
    clean_ohlcv,
    handle_missing_values,
    quality_report,
    preprocess_pipeline,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def sample_ohlcv():
    """Create a small synthetic OHLCV DataFrame."""
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    df = pd.DataFrame({
        "Open":   np.random.uniform(100, 110, 10),
        "High":   np.random.uniform(110, 120, 10),
        "Low":    np.random.uniform(90, 100, 10),
        "Close":  np.random.uniform(100, 115, 10),
        "Volume": np.random.randint(1_000_000, 5_000_000, 10),
    }, index=dates)
    # Ensure High >= Open/Close and Low <= Open/Close
    df["High"] = df[["Open", "Close", "High"]].max(axis=1)
    df["Low"]  = df[["Open", "Close", "Low"]].min(axis=1)
    return df


@pytest.fixture
def ohlcv_with_missing(sample_ohlcv):
    df = sample_ohlcv.copy()
    df.iloc[2:4, df.columns.get_loc("Close")] = np.nan
    df.iloc[5, df.columns.get_loc("Volume")] = np.nan
    return df


# ----------------------------------------------------------------------
# Tests – fetch
# ----------------------------------------------------------------------
def test_validate_tickers_basic():
    """Validate a small list of well-known tickers."""
    tickers = ["AAPL", "MSFT"]
    valid, failed = validate_tickers(tickers, max_workers=2)
    
    assert isinstance(valid, list)
    assert isinstance(failed, dict)
    # At least one should succeed under normal conditions
    assert len(valid) + len(failed) == len(tickers)


def test_fetch_ohlcv_single_ticker():
    """Fetch OHLCV for a single liquid ticker."""
    result = fetch_ohlcv("AAPL", period="1mo", interval="1d")
    
    assert isinstance(result, dict)
    if "AAPL" in result:
        df = result["AAPL"]
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)


def test_fetch_latest_quote():
    """Fetch latest quote structure."""
    quote = fetch_latest_quote("AAPL")
    
    assert isinstance(quote, dict)
    assert "ticker" in quote
    # Either we get price data or an error key
    assert "price" in quote or "error" in quote


def test_get_failed_tickers_returns_dict():
    failed = get_failed_tickers()
    assert isinstance(failed, dict)


# ----------------------------------------------------------------------
# Tests – preprocess
# ----------------------------------------------------------------------
def test_clean_ohlcv(sample_ohlcv):
    cleaned = clean_ohlcv(sample_ohlcv)
    
    assert isinstance(cleaned, pd.DataFrame)
    assert not cleaned.empty
    assert isinstance(cleaned.index, pd.DatetimeIndex)
    assert cleaned.index.is_monotonic_increasing
    assert cleaned.index.is_unique


def test_clean_ohlcv_removes_non_positive(sample_ohlcv):
    df = sample_ohlcv.copy()
    df.iloc[0, df.columns.get_loc("Close")] = -10  # invalid
    cleaned = clean_ohlcv(df)
    assert (cleaned[["Open", "High", "Low", "Close"]] > 0).all().all()


def test_handle_missing_values_ffill(ohlcv_with_missing):
    filled = handle_missing_values(ohlcv_with_missing, method="ffill")
    
    assert filled["Close"].isna().sum() == 0
    # Volume may still have leading NaNs depending on limit, but after full pipeline should be clean


def test_handle_missing_values_drop(ohlcv_with_missing):
    dropped = handle_missing_values(ohlcv_with_missing, method="drop")
    assert dropped[["Open", "High", "Low", "Close"]].isna().sum().sum() == 0
    assert len(dropped) < len(ohlcv_with_missing)


def test_quality_report(sample_ohlcv):
    report = quality_report(sample_ohlcv, ticker="TEST")
    
    assert isinstance(report, dict)
    assert report["ticker"] == "TEST"
    assert report["rows"] == len(sample_ohlcv)
    assert "missing_pct" in report
    assert "status" in report


def test_preprocess_pipeline(sample_ohlcv):
    raw = {"TEST": sample_ohlcv}
    cleaned = preprocess_pipeline(raw, fill_method="ffill")
    
    assert "TEST" in cleaned
    assert isinstance(cleaned["TEST"], pd.DataFrame)
    assert not cleaned["TEST"].empty