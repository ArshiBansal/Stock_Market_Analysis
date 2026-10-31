"""
Zephyr – Data fetching module
Responsible for downloading OHLCV data via yfinance,
validating tickers, handling failures and providing latest quotes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import ALL_TICKERS, DEFAULT_HISTORY_PERIOD, TICKER_TO_REGION

logger = logging.getLogger(__name__)

# Global registry of tickers that failed during the current session
_FAILED_TICKERS: Dict[str, str] = {}


def get_failed_tickers() -> Dict[str, str]:
    """Return a copy of the failed-ticker registry."""
    return _FAILED_TICKERS.copy()


def _clear_failed_tickers() -> None:
    _FAILED_TICKERS.clear()


def validate_tickers(
    tickers: Optional[List[str]] = None,
    max_workers: int = 8,
) -> Tuple[List[str], Dict[str, str]]:
    """
    Validate that tickers return usable data from Yahoo Finance.

    Returns
    -------
    valid : list of tickers that returned at least some data
    failed : dict {ticker: reason}
    """
    if tickers is None:
        tickers = ALL_TICKERS

    _clear_failed_tickers()
    valid: List[str] = []
    failed: Dict[str, str] = {}

    def _check(ticker: str) -> Tuple[str, bool, str]:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="5d", auto_adjust=True)
            if hist.empty:
                return ticker, False, "Empty history"
            return ticker, True, ""
        except Exception as e:
            return ticker, False, str(e)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_check, t): t for t in tickers}
        for future in as_completed(futures):
            ticker, ok, reason = future.result()
            if ok:
                valid.append(ticker)
            else:
                failed[ticker] = reason
                _FAILED_TICKERS[ticker] = reason
                logger.warning(f"Ticker validation failed for {ticker}: {reason}")

    logger.info(f"Validated {len(valid)}/{len(tickers)} tickers successfully.")
    return valid, failed


def fetch_ohlcv(
    tickers: List[str] | str,
    period: str = DEFAULT_HISTORY_PERIOD,
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
    auto_adjust: bool = True,
    progress: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Download OHLCV data for one or more tickers.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    results: Dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            if start or end:
                df = t.history(
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    actions=False,
                )
            else:
                df = t.history(
                    period=period,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    actions=False,
                )

            if df.empty:
                msg = "Empty DataFrame returned"
                _FAILED_TICKERS[ticker] = msg
                logger.warning(f"{ticker}: {msg}")
                continue

            # Standardise column names
            df = df.rename(columns=str.title)
            required = {"Open", "High", "Low", "Close", "Volume"}
            if not required.issubset(df.columns):
                msg = f"Missing columns: {required - set(df.columns)}"
                _FAILED_TICKERS[ticker] = msg
                logger.warning(f"{ticker}: {msg}")
                continue

            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index = pd.to_datetime(df.index)

            # Make index timezone-naive
            if df.index.tz is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)

            df["Ticker"] = ticker
            df["Region"] = TICKER_TO_REGION.get(ticker, "Unknown")

            results[ticker] = df
            _FAILED_TICKERS.pop(ticker, None)

        except Exception as e:
            msg = str(e)
            _FAILED_TICKERS[ticker] = msg
            logger.error(f"Failed to fetch {ticker}: {msg}")

    return results


def fetch_latest_quote(ticker: str) -> Dict:
    """
    Fetch the most recent available quote information for a single ticker.
    Fixed: handles timezone-aware vs timezone-naive datetime subtraction.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        hist = t.history(period="5d", interval="1d", auto_adjust=True)

        if hist.empty:
            return {
                "ticker": ticker,
                "error": "No data returned",
                "stale": True,
            }

        last = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else last

        price = float(last["Close"])
        prev_close = float(prev["Close"])
        change = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0.0

        # ---------- Fixed timestamp handling ----------
        last_ts = hist.index[-1]

        # Convert to timezone-naive UTC
        if getattr(last_ts, "tz", None) is not None or getattr(last_ts, "tzinfo", None) is not None:
            try:
                last_ts = last_ts.tz_convert("UTC").tz_localize(None)
            except Exception:
                last_ts = last_ts.tz_localize(None) if hasattr(last_ts, "tz_localize") else last_ts

        if hasattr(last_ts, "to_pydatetime"):
            last_ts = last_ts.to_pydatetime()

        # Ensure last_ts is naive
        if getattr(last_ts, "tzinfo", None) is not None:
            last_ts = last_ts.replace(tzinfo=None)

        now = datetime.utcnow()  # naive UTC
        age_hours = (now - last_ts).total_seconds() / 3600 if last_ts else 999.0
        stale = age_hours > 24

        return {
            "ticker": ticker,
            "price": round(price, 4),
            "change": round(change, 4),
            "change_pct": round(change_pct, 2),
            "volume": int(last["Volume"]),
            "high": round(float(last["High"]), 4),
            "low": round(float(last["Low"]), 4),
            "open": round(float(last["Open"]), 4),
            "prev_close": round(prev_close, 4),
            "timestamp": last_ts,
            "age_hours": round(age_hours, 1),
            "stale": stale,
            "currency": info.get("currency", "N/A"),
            "exchange": info.get("exchange", "N/A"),
            "region": TICKER_TO_REGION.get(ticker, "Unknown"),
            "name": info.get("shortName") or info.get("longName") or ticker,
        }

    except Exception as e:
        logger.error(f"Latest quote failed for {ticker}: {e}")
        return {
            "ticker": ticker,
            "error": str(e),
            "stale": True,
        }


def fetch_multi_latest_quotes(
    tickers: List[str],
    max_workers: int = 6,
) -> Dict[str, Dict]:
    """Fetch latest quotes for multiple tickers concurrently."""
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_latest_quote, t): t for t in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            results[ticker] = future.result()
    return results