"""
Zephyr – Tab 3: Time Series + ML
Historical chart, technical features, ML model selection and forecast comparison.
"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.fetch import fetch_ohlcv
from data.preprocess import clean_ohlcv, handle_missing_values
from data.features import add_technical_features, create_supervised_dataset, get_feature_columns
from models.train_eval import run_model_comparison, chronological_split
from models.ml_models import ML_MODEL_MAP
from config import ML_MODELS, BASELINE_MODELS, DISCLAIMER


def render_time_series_ml(filters: Dict[str, Any]) -> None:
    st.header("3. Time Series + ML")
    st.caption("Historical prices, technical features and classical ML forecasts")

    ticker = filters.get("ticker")
    start_date = filters.get("start_date")
    end_date = filters.get("end_date")
    horizon = filters.get("horizon", 1)
    selected_model = filters.get("model", "XGBoost")

    if not ticker:
        st.warning("Please select a ticker from the sidebar.")
        return

    # ------------------------------------------------------------------
    # Load & prepare data
    # ------------------------------------------------------------------
    with st.spinner("Loading and engineering features..."):
        raw = fetch_ohlcv(
            tickers=ticker,
            start=str(start_date) if start_date else None,
            end=str(end_date) if end_date else None,
            period="2y",
        )
        if ticker not in raw or raw[ticker].empty:
            st.error(f"No data available for {ticker}")
            return

        df = clean_ohlcv(raw[ticker])
        df = handle_missing_values(df, method="ffill")
        df = add_technical_features(df, dropna=True)

    if df.empty:
        st.error("No data left after feature engineering.")
        return

    # ------------------------------------------------------------------
    # Price chart
    # ------------------------------------------------------------------
    st.subheader("Price History")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="OHLC",
    ))
    fig.update_layout(
        title=f"{ticker} – Candlestick",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=450,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Feature preview
    # ------------------------------------------------------------------
    with st.expander("Technical / Lag Features (preview)"):
        feature_cols = get_feature_columns(df)
        st.write(f"Feature columns ({len(feature_cols)}):")
        st.code(", ".join(feature_cols[:20]) + (" ..." if len(feature_cols) > 20 else ""))
        st.dataframe(df[feature_cols].tail(10), use_container_width=True)

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------
    st.subheader(f"Model Comparison ({horizon}-Trading-Day Return)")

    ml_and_baselines = BASELINE_MODELS + [m for m in ML_MODELS if m in ML_MODEL_MAP]

    default_selection = [m for m in ["NaiveLastClose", "RandomForest", "XGBoost", "LightGBM"] if m in ml_and_baselines]
    models_to_run = st.multiselect(
        "Select models to compare",
        options=ml_and_baselines,
        default=default_selection,
    )

    if st.button("Run ML Comparison", type="primary"):
        if not models_to_run:
            st.warning("Select at least one model.")
        else:
            X, y = create_supervised_dataset(
                df, target_col="target_return_1d", horizon=horizon
            )

            with st.spinner("Training and evaluating models..."):
                results_df = run_model_comparison(
                    X, y,
                    models=models_to_run,
                    test_size=0.2,
                    horizon=horizon,
                )

            st.dataframe(
                results_df.style.format({
                    "MAE": "{:.6f}",
                    "RMSE": "{:.6f}",
                    "DirectionalAccuracy": "{:.1f}%",
                }),
                use_container_width=True,
            )

            # Simple bar chart of MAE
            if not results_df.empty and "MAE" in results_df.columns:
                fig_mae = go.Figure(go.Bar(
                    x=results_df["Model"],
                    y=results_df["MAE"],
                    marker_color="steelblue",
                ))
                fig_mae.update_layout(
                    title="MAE by Model (lower is better)",
                    xaxis_title="Model",
                    yaxis_title="MAE",
                    height=350,
                    template="plotly_white",
                )
                st.plotly_chart(fig_mae, use_container_width=True)

    st.markdown("---")
    st.caption(DISCLAIMER)