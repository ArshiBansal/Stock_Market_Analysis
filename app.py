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
from ui.models_evaluation import render_models_evaluation
from ui.ai_forecast import render_ai_forecast
from ui.portfolio import render_portfolio
from ui.research import render_research
from ui.scenario import render_scenario


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
        render_models_evaluation(filters)

    with tabs[2]:
        render_ai_forecast(filters)

    with tabs[3]:
        render_portfolio(filters)

    with tabs[4]:
        render_research(filters)

    with tabs[5]:
        render_scenario(filters)

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