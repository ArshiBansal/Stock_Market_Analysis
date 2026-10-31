"""
Zephyr – Tab 2: Raw Data
OHLCV records, date range, missing-data checks and download status.
"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st
import pandas as pd

from data.fetch import fetch_ohlcv, get_failed_tickers
from data.preprocess import clean_ohlcv, handle_missing_values, quality_report
from config import DISCLAIMER


def render_raw_data(filters: Dict[str, Any]) -> None:
    st.header("2. Raw Data")
    st.caption("OHLCV history, quality checks and download status")

    ticker = filters.get("ticker")
    start_date = filters.get("start_date")
    end_date = filters.get("end_date")

    if not ticker:
        st.warning("Please select a ticker from the sidebar.")
        return

    # ------------------------------------------------------------------
    # Fetch data
    # ------------------------------------------------------------------
    with st.spinner(f"Downloading OHLCV data for {ticker}..."):
        raw = fetch_ohlcv(
            tickers=ticker,
            start=str(start_date) if start_date else None,
            end=str(end_date) if end_date else None,
            period="2y",
            interval="1d",
        )

    if ticker not in raw or raw[ticker].empty:
        st.error(f"No data returned for **{ticker}**.")
        failed = get_failed_tickers()
        if ticker in failed:
            st.write(f"Reason: {failed[ticker]}")
        return

    df = raw[ticker].copy()

    # ------------------------------------------------------------------
    # Quality report
    # ------------------------------------------------------------------
    st.subheader("Data Quality")
    report = quality_report(df, ticker=ticker)

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Rows", report["rows"])
    q2.metric("Missing %", f"{report['missing_pct']}%")
    q3.metric("Start", str(report["date_start"]))
    q4.metric("End", str(report["date_end"]))

    if report["status"] == "WARNING":
        st.warning("Data quality warning – significant missing values detected.")
    else:
        st.success("Data quality looks good.")

    # ------------------------------------------------------------------
    # Cleaning options
    # ------------------------------------------------------------------
    st.subheader("Preprocessing")
    fill_method = st.selectbox(
        "Missing value handling",
        options=["ffill", "drop", "none"],
        index=0,
        help="Forward-fill is recommended for price data",
    )

    df_clean = clean_ohlcv(df)
    df_clean = handle_missing_values(df_clean, method=fill_method)

    st.write(f"Rows after cleaning: **{len(df_clean)}** (removed {len(df) - len(df_clean)})")

    # ------------------------------------------------------------------
    # Display table
    # ------------------------------------------------------------------
    st.subheader("OHLCV Table")
    st.dataframe(
        df_clean.style.format({
            "Open": "{:.2f}",
            "High": "{:.2f}",
            "Low": "{:.2f}",
            "Close": "{:.2f}",
            "Volume": "{:,.0f}",
        }),
        use_container_width=True,
        height=420,
    )

    # Download button
    csv = df_clean.to_csv().encode("utf-8")
    st.download_button(
        label="Download cleaned CSV",
        data=csv,
        file_name=f"{ticker}_ohlcv.csv",
        mime="text/csv",
    )

    # Failed tickers expander
    failed = get_failed_tickers()
    if failed:
        with st.expander("Failed tickers this session"):
            st.json(failed)

    st.markdown("---")
    st.caption(DISCLAIMER)