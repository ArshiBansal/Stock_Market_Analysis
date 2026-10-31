"""
Zephyr – General helper utilities
Timestamps, staleness checks, safe math, cache TTL helpers, etc.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional, Union, Any

import pandas as pd
import numpy as np

from config import CACHE_TTL_SECONDS, REFRESH_OPTIONS


def format_timestamp(
    ts: Optional[Union[datetime, pd.Timestamp, str]],
    fmt: str = "%Y-%m-%d %H:%M",
) -> str:
    """
    Safely format a timestamp into a readable string.
    """
    if ts is None:
        return "N/A"
    try:
        if isinstance(ts, str):
            ts = pd.to_datetime(ts)
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()
        return ts.strftime(fmt)
    except Exception:
        return str(ts)


def is_stale(
    ts: Optional[Union[datetime, pd.Timestamp]],
    max_age_hours: float = 24.0,
) -> bool:
    """
    Return True if the observation is older than max_age_hours.
    """
    if ts is None:
        return True
    try:
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        age = (now - ts).total_seconds() / 3600
        return age > max_age_hours
    except Exception:
        return True


def safe_divide(
    numerator: Union[float, np.ndarray, pd.Series],
    denominator: Union[float, np.ndarray, pd.Series],
    fill: float = 0.0,
) -> Union[float, np.ndarray, pd.Series]:
    """
    Division that avoids division-by-zero and replaces inf/nan with `fill`.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(numerator, denominator)
    if isinstance(result, (np.ndarray, pd.Series)):
        result = np.where(np.isfinite(result), result, fill)
        return result
    return fill if not np.isfinite(result) else float(result)


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee that the DataFrame has a sorted, unique DatetimeIndex.
    """
    if df.empty:
        return df
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def get_cache_ttl(refresh_label: str) -> int:
    """
    Return the recommended cache TTL (in seconds) for a given refresh setting.
    Falls back to a safe default if the label is unknown.
    """
    return CACHE_TTL_SECONDS.get(refresh_label, 240)


def age_in_hours(ts: Optional[Union[datetime, pd.Timestamp]]) -> float:
    """
    Return the age of a timestamp in hours (or 999 if unknown).
    """
    if ts is None:
        return 999.0
    try:
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        return (now - ts).total_seconds() / 3600
    except Exception:
        return 999.0


def truncate_text(text: str, max_len: int = 120) -> str:
    """Simple helper to truncate long headlines for display."""
    if not text or len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."