"""
Zephyr – Tab 1: Overview
Market snapshot, latest quote, daily change, volume, high/low, region & data freshness.
"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st
import pandas as pd

from data.fetch import fetch_latest_quote, get_failed_tickers
from config import DISCLAIMER


def render_overview(filters: Dict[str, Any]) -> None:
    st.header("1. Overview")
    st.caption("Live market snapshot for the selected ticker")

    ticker = filters.get("ticker")
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

    st.markdown("---")
    st.caption(DISCLAIMER)