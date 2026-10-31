"""
Zephyr – Shared sidebar filters
(News & Sentiment controls have been removed)
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict

import streamlit as st

from config import (
    ALL_TICKERS,
    REGIONS,
    TICKERS,
    REFRESH_OPTIONS,
    DEFAULT_REFRESH,
    HORIZON_OPTIONS,
    DEFAULT_HORIZON,
    ML_MODELS,
    DL_MODELS,
    BASELINE_MODELS,
    DEFAULT_ML_MODEL,
    PROJECT_TITLE,
    DISCLAIMER,
)


def render_sidebar() -> Dict[str, Any]:
    """
    Render the common sidebar and return a dictionary of selected filters.
    """
    st.sidebar.title("Zephyr Controls")
    st.sidebar.caption(PROJECT_TITLE)

    # ------------------------------------------------------------------
    # Region & Ticker
    # ------------------------------------------------------------------
    st.sidebar.subheader("Universe")

    selected_region = st.sidebar.selectbox(
        "Region",
        options=["All"] + REGIONS,
        index=0,
    )

    if selected_region == "All":
        available_tickers = ALL_TICKERS
    else:
        available_tickers = TICKERS.get(selected_region, [])

    selected_ticker = st.sidebar.selectbox(
        "Ticker",
        options=available_tickers,
        index=0 if available_tickers else None,
        help="Select a company from the high-profile universe",
    )

    # ------------------------------------------------------------------
    # Date range
    # ------------------------------------------------------------------
    st.sidebar.subheader("Date Range")

    today = date.today()
    default_start = today - timedelta(days=365 * 2)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start", value=default_start, max_value=today)
    with col2:
        end_date = st.date_input("End", value=today, max_value=today)

    if start_date > end_date:
        st.sidebar.error("Start date must be before end date")

    # ------------------------------------------------------------------
    # Forecast settings
    # ------------------------------------------------------------------
    st.sidebar.subheader("Forecasting")

    horizon = st.sidebar.selectbox(
        "Forecast Horizon (days)",
        options=HORIZON_OPTIONS,
        index=HORIZON_OPTIONS.index(DEFAULT_HORIZON),
    )

    all_models = BASELINE_MODELS + ML_MODELS + DL_MODELS
    selected_model = st.sidebar.selectbox(
        "Model",
        options=all_models,
        index=all_models.index(DEFAULT_ML_MODEL) if DEFAULT_ML_MODEL in all_models else 0,
    )

    # ------------------------------------------------------------------
    # Data refresh
    # ------------------------------------------------------------------
    st.sidebar.subheader("Data Refresh")

    refresh_label = st.sidebar.selectbox(
        "Refresh Interval",
        options=list(REFRESH_OPTIONS.keys()),
        index=list(REFRESH_OPTIONS.keys()).index(DEFAULT_REFRESH),
        help="How often the app checks for new data",
    )
    refresh_seconds = REFRESH_OPTIONS[refresh_label]

    # ------------------------------------------------------------------
    # Disclaimer
    # ------------------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.caption(DISCLAIMER)

    # ------------------------------------------------------------------
    # Return unified filter dictionary
    # ------------------------------------------------------------------
    return {
        "region": selected_region,
        "ticker": selected_ticker,
        "start_date": start_date,
        "end_date": end_date,
        "horizon": horizon,
        "model": selected_model,
        "refresh_label": refresh_label,
        "refresh_seconds": refresh_seconds,
    }