"""
Zephyr – Intelligent Stock Market Forecasting & News Sentiment Decision Support System

Main Streamlit entry point.
(News & Sentiment tab has been removed)
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
from config import (
    PROJECT_TITLE,
    PROJECT_NAME,
    VERSION,
    DISCLAIMER,
    TAB_NAMES,
)
from ui.sidebar import render_sidebar
from ui.overview import render_overview
from ui.raw_data import render_raw_data
from ui.time_series_ml import render_time_series_ml
from ui.neural_networks import render_neural_networks
from ui.performance import render_performance
from ui.ai_forecast import render_ai_forecast


# Main application
def main() -> None:
    # ---------- Header ----------
    st.title(f"📈 {PROJECT_NAME}")
    st.caption(f"{PROJECT_TITLE}  |  v{VERSION}")

    # ---------- Sidebar (shared filters) ----------
    filters = render_sidebar()

    # ---------- Tabs (News tab removed) ----------
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