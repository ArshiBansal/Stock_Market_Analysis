"""
Zephyr – Tab 5: Performance Metrics
MAE, RMSE, Directional Accuracy, baseline comparison and per-ticker results.
"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data.fetch import fetch_ohlcv
from data.preprocess import clean_ohlcv, handle_missing_values
from data.features import add_technical_features, create_supervised_dataset
from models.train_eval import run_model_comparison, walk_forward_validation
from models.ml_models import ML_MODEL_MAP
from config import (
    ML_MODELS,
    DL_MODELS,
    BASELINE_MODELS,
    DISCLAIMER,
)


def render_performance(filters: Dict[str, Any]) -> None:
    st.header("5. Performance Metrics")
    st.caption("Model evaluation – MAE, RMSE, Directional Accuracy & baseline comparison")

    ticker = filters.get("ticker")
    start_date = filters.get("start_date")
    end_date = filters.get("end_date")

    if not ticker:
        st.warning("Please select a ticker from the sidebar.")
        return

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------
    all_models = BASELINE_MODELS + list(ML_MODEL_MAP.keys()) + DL_MODELS
    default_models = ["NaiveLastClose", "NaiveLastReturn", "RandomForest", "XGBoost", "LightGBM"]

    selected_models = st.multiselect(
        "Models to evaluate",
        options=all_models,
        default=[m for m in default_models if m in all_models],
        help="Select the models you want to compare on the same chronological split",
    )

    test_size = st.slider("Test set size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

    run_walk_forward = st.checkbox("Also run walk-forward validation (slower)", value=False)

    if not selected_models:
        st.info("Select at least one model to begin evaluation.")
        return

    # ------------------------------------------------------------------
    # Load & prepare data
    # ------------------------------------------------------------------
    with st.spinner(f"Preparing data for {ticker}..."):
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

    if df.empty or len(df) < 100:
        st.error("Insufficient data after feature engineering.")
        return

    X, y = create_supervised_dataset(df, target_col="target_return_1d")

    # ------------------------------------------------------------------
    # Run comparison
    # ------------------------------------------------------------------
    if st.button("Run Evaluation", type="primary"):
        with st.spinner("Training and evaluating models (this may take a while)..."):
            results_df = run_model_comparison(
                X, y,
                models=selected_models,
                test_size=test_size,
            )

        if results_df.empty:
            st.error("No results returned.")
            return

        st.subheader("Summary Table")
        st.dataframe(
            results_df.style.format({
                "MAE": "{:.6f}",
                "RMSE": "{:.6f}",
                "DirectionalAccuracy": "{:.1f}%",
            }),
            use_container_width=True,
        )

        # Ranking highlight
        if "MAE" in results_df.columns:
            best = results_df.loc[results_df["MAE"].idxmin()]
            st.success(
                f"Best model by MAE: **{best['Model']}** "
                f"(MAE={best['MAE']:.6f}, Dir.Acc={best.get('DirectionalAccuracy', 0):.1f}%)"
            )

        # MAE bar chart
        st.subheader("MAE Comparison")
        fig_mae = px.bar(
            results_df,
            x="Model",
            y="MAE",
            color="Type",
            title="Mean Absolute Error (lower is better)",
            template="plotly_white",
        )
        fig_mae.update_layout(height=380)
        st.plotly_chart(fig_mae, use_container_width=True)

        # Directional Accuracy
        if "DirectionalAccuracy" in results_df.columns:
            st.subheader("Directional Accuracy")
            fig_dir = px.bar(
                results_df,
                x="Model",
                y="DirectionalAccuracy",
                color="Type",
                title="Directional Accuracy % (higher is better)",
                template="plotly_white",
            )
            fig_dir.update_layout(height=380)
            st.plotly_chart(fig_dir, use_container_width=True)

        # Walk-forward (optional)
        if run_walk_forward:
            st.subheader("Walk-Forward Validation")
            wf_model = st.selectbox(
                "Model for walk-forward",
                options=[m for m in selected_models if m in ML_MODEL_MAP],
                index=0,
            )
            with st.spinner(f"Running walk-forward for {wf_model}..."):
                wf_result = walk_forward_validation(X, y, model_name=wf_model, n_splits=5)

            st.write("Average metrics across folds:")
            st.json(wf_result["average"])

            fold_df = pd.DataFrame(wf_result["fold_scores"])
            st.dataframe(fold_df.style.format("{:.4f}"), use_container_width=True)

    st.markdown("---")
    st.caption(DISCLAIMER)