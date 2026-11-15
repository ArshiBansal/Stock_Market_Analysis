"""
Zephyr – Tab 1: Overview
Live market snapshot plus the former "Raw Data" tooling (quality, cleaning,
OHLCV table and CSV download) in a collapsible section.
"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st
import pandas as pd

from data.fetch import fetch_latest_quote, fetch_ohlcv, get_failed_tickers
from data.preprocess import clean_ohlcv, handle_missing_values, quality_report
from config import DISCLAIMER


def render_overview(filters: Dict[str, Any]) -> None:
    st.header("1. Overview")
    st.caption("Live market snapshot for the selected ticker")

    ticker = filters.get("ticker")
    start_date = filters.get("start_date")
    end_date = filters.get("end_date")

    if not ticker:
        st.warning("Please select a ticker from the sidebar.")
        return

    with st.spinner(f"Fetching latest quote for {ticker}..."):
        quote = fetch_latest_quote(ticker)

    if "error" in quote:
        st.error(f"Could not retrieve data for **{ticker}**: {quote['error']}")
        failed = get_failed_tickers()
        if failed:
            with st.expander("Failed tickers this session"):
                st.json(failed)
        return

    # ------------------------------------------------------------------
    # Header metrics
    # ------------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    price = quote.get("price", 0)
    change = quote.get("change", 0)
    change_pct = quote.get("change_pct", 0)
    volume = quote.get("volume", 0)

    delta_color = "normal" if change >= 0 else "inverse"

    col1.metric(
        label=f"{quote.get('name', ticker)} ({ticker})",
        value=f"{price:,.2f} {quote.get('currency', '')}",
        delta=f"{change:+.2f} ({change_pct:+.2f}%)",
        delta_color=delta_color,
    )
    col2.metric("Volume", f"{volume:,.0f}")
    col3.metric("Day High", f"{quote.get('high', 0):,.2f}")
    col4.metric("Day Low", f"{quote.get('low', 0):,.2f}")

    # ------------------------------------------------------------------
    # Additional info
    # ------------------------------------------------------------------
    st.markdown("---")
    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.write("**Region**")
        st.info(quote.get("region", "Unknown"))

    with info_col2:
        st.write("**Exchange**")
        st.info(quote.get("exchange", "N/A"))

    with info_col3:
        st.write("**Last Observation**")
        ts = quote.get("timestamp")
        age = quote.get("age_hours", 0)
        stale = quote.get("stale", False)

        if ts:
            st.write(ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts))
        else:
            st.write("N/A")

        if stale:
            st.error(f"⚠️ Data appears stale ({age:.1f} hours old)")
        else:
            st.success(f"Data age: {age:.1f} hours")

    # ------------------------------------------------------------------
    # Quick stats table
    # ------------------------------------------------------------------
    st.subheader("Quote Details")
    details = {
        "Open": quote.get("open"),
        "Previous Close": quote.get("prev_close"),
        "Change": quote.get("change"),
        "Change %": quote.get("change_pct"),
        "Volume": quote.get("volume"),
        "Currency": quote.get("currency"),
    }
    st.table(pd.DataFrame([details]).T.rename(columns={0: "Value"}))

    # ------------------------------------------------------------------
    # Data & Features (formerly the Raw Data tab)
    # ------------------------------------------------------------------
    with st.expander("Data & Features", expanded=False):
        _render_data_and_features(ticker, start_date, end_date)

    st.markdown("---")
    st.caption(DISCLAIMER)


def _render_data_and_features(ticker: str, start_date, end_date) -> None:
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

    st.subheader("Preprocessing")
    fill_method = st.selectbox(
        "Missing value handling",
        options=["ffill", "drop", "none"],
        index=0,
        help="Forward-fill is recommended for price data",
        key="rawdata_fill",
    )

    df_clean = clean_ohlcv(df)
    df_clean = handle_missing_values(df_clean, method=fill_method)

    st.write(f"Rows after cleaning: **{len(df_clean)}** (removed {len(df) - len(df_clean)})")

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

    csv = df_clean.to_csv().encode("utf-8")
    st.download_button(
        label="Download cleaned CSV",
        data=csv,
        file_name=f"{ticker}_ohlcv.csv",
        mime="text/csv",
    )

    failed = get_failed_tickers()
    if failed:
        with st.expander("Failed tickers this session"):
            st.json(failed)