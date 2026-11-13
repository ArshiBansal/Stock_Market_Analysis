"""
Zephyr – Intelligent Stock Market Forecasting

Main Streamlit entry point.
"""

from __future__ import annotations

import streamlit as st
from datetime import datetime

# Page configuration (must be first Streamlit command)
st.set_page_config(
    page_title="Zephyr – Stock Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Project imports
from dotenv import load_dotenv
from config import (
    PROJECT_TITLE,
    PROJECT_NAME,
    VERSION,
    DISCLAIMER,
    TAB_NAMES,
    ensure_directories,
)

load_dotenv()
from ui.sidebar import render_sidebar
from ui.overview import render_overview
from ui.raw_data import render_raw_data
from ui.time_series_ml import render_time_series_ml
from ui.neural_networks import render_neural_networks
from ui.performance import render_performance
from ui.ai_forecast import render_ai_forecast
from ui.portfolio import render_portfolio

from data.fetch import fetch_ohlcv
from data.preprocess import clean_ohlcv, handle_missing_values
from data.features import add_technical_features, create_supervised_dataset


# Main application
def get_cached_supervised(ticker: str, horizon: int, start: str, end: str):
    """Cache the engineered X/y dataset in session state to avoid recomputation across tabs."""
    key = f"supervised_{ticker}_{horizon}_{start}_{end}"
    if key not in st.session_state:
        raw = fetch_ohlcv(tickers=ticker, start=start, end=end, period="2y")
        if ticker not in raw or raw[ticker].empty:
            st.session_state[key] = None
        else:
            df = clean_ohlcv(raw[ticker])
            df = handle_missing_values(df, method="ffill")
            df = add_technical_features(df, dropna=True)
            X, y = create_supervised_dataset(df, target_col="target_return_1d", horizon=horizon)
            st.session_state[key] = (X, y, df)
    return st.session_state.get(key)


# Main application
def main() -> None:
    ensure_directories()

    # ---------- Header ----------
    st.title(f"📈 {PROJECT_NAME}")
    st.caption(f"{PROJECT_TITLE}  |  v{VERSION}")

    # ---------- Sidebar (shared filters) ----------
    filters = render_sidebar()

    # ---------- Tabs ----------
    tabs = st.tabs(TAB_NAMES)

    with tabs[0]:
        render_overview(filters)

    with tabs[1]:
        render_raw_data(filters)

    with tabs[2]:
        render_time_series_ml(filters)

    with tabs[3]:
        render_neural_networks(filters)

    with tabs[4]:
        render_performance(filters)

    with tabs[5]:
        render_ai_forecast(filters)

    with tabs[6]:
        render_portfolio(filters)

    # ---------- Footer (Centered) ----------
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: gray; font-size: 0.9rem;'>
            {DISCLAIMER}<br>
            Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
            Built by <a href='https://arshibansal.github.io/Personal_Portfolio/' target='_blank'>Arshi Bansal</a>
        </div>
        """,
        unsafe_allow_html=True
    )


# Entry point
if __name__ == "__main__":
    main()