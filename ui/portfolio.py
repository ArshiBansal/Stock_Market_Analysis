"""
Zephyr – Tab: Portfolio
Multi-ticker portfolio view, correlation matrix and fundamentals.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data.fetch import fetch_fundamentals, fetch_ohlcv
from data.preprocess import clean_ohlcv, handle_missing_values
from config import TICKERS


def _fetch_portfolio_data(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    raw = fetch_ohlcv(tickers=tickers, period="1y", interval="1d")
    cleaned: Dict[str, pd.DataFrame] = {}
    for ticker, df in raw.items():
        df = clean_ohlcv(df)
        df = handle_missing_values(df, method="ffill")
        if not df.empty:
            cleaned[ticker] = df
    return cleaned


def render_portfolio(filters: Dict[str, Any]) -> None:
    st.header("Portfolio")
    st.caption("Multi-ticker portfolio analysis, correlation and fundamentals")

    region = filters.get("region", "All")
    if region == "All":
        tickers = [t for region_tickers in TICKERS.values() for t in region_tickers]
    else:
        tickers = TICKERS.get(region, [])

    default_pick = tickers[:5] if tickers else []
    selected = st.multiselect("Select tickers", options=tickers, default=default_pick)

    if not selected:
        st.info("Select at least one ticker.")
        return

    col1, col2 = st.columns(2)
    with col1:
        run = st.button("Analyze Portfolio", type="primary")
    with col2:
        show_fundamentals = st.checkbox("Show fundamentals", value=True)

    if not run:
        st.info("Click **Analyze Portfolio** to compute correlation and performance.")
        return

    with st.spinner("Fetching portfolio data..."):
        data = _fetch_portfolio_data(selected)

    if not data:
        st.error("No data returned for selected tickers.")
        return

    # Alignment: build close-price dataframe on common dates
    closes = pd.DataFrame({t: df["Close"] for t, df in data.items()}).dropna()
    if closes.empty or len(closes) < 2:
        st.error("Insufficient overlapping history.")
        return

    # Correlation matrix
    corr = closes.pct_change().dropna().corr()
    st.subheader("Correlation Matrix")
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Return Correlation",
    )
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

    # Normalized performance
    st.subheader("Normalized Performance (1 = start)")
    norm = closes / closes.iloc[0]
    fig_perf = go.Figure()
    for t in norm.columns:
        fig_perf.add_trace(go.Scatter(x=norm.index, y=norm[t], name=t, mode="lines"))
    fig_perf.update_layout(height=400, template="plotly_white")
    st.plotly_chart(fig_perf, use_container_width=True)

    # Portfolio summary (equal weight)
    st.subheader("Equal-Weight Portfolio Summary")
    rets = closes.pct_change().dropna()
    port_rets = rets.mean(axis=1)
    from utils.metrics import sharpe_ratio, max_drawdown, value_at_risk

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sharpe", f"{sharpe_ratio(port_rets):.2f}")
    m2.metric("Max Drawdown", f"{max_drawdown(port_rets):.2f}%")
    m3.metric("VaR (95%)", f"{value_at_risk(port_rets):.2f}%")
    m4.metric("Cumulative Return", f"{((1 + port_rets).prod() - 1) * 100:.2f}%")

    # Fundamentals
    if show_fundamentals:
        st.subheader("Fundamentals")
        with st.spinner("Loading fundamentals..."):
            fund = {t: fetch_fundamentals(t) for t in selected}
        fund_df = pd.DataFrame(fund).T
        display_cols = [c for c in ["name", "sector", "market_cap", "pe_ratio", "dividend_yield", "beta"] if c in fund_df.columns]
        st.dataframe(fund_df[display_cols], use_container_width=True)
