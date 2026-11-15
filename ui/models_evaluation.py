"""
Zephyr – Tab: Models & Evaluation
Merged view of the former "Time Series + ML", "Neural Networks" and
"Performance Metrics" tabs, organized with a segmented sub-navigation.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.preprocessing import StandardScaler

from data.features import (
    create_supervised_dataset,
    get_feature_columns,
    load_featured_prices,
)
from models.train_eval import (
    chronological_split,
    evaluate_regression,
    run_model_comparison,
    walk_forward_validation,
)
from models.ml_models import ML_MODEL_MAP
from models.dl_models import (
    build_lstm,
    build_gru,
    create_sequences,
    train_dl_model,
    predict_dl,
)
from utils.plotting import plot_candlestick
from ui._common import render_segmented
from config import (
    MODEL_EVAL_SECTIONS,
    ML_MODELS,
    DL_MODELS,
    BASELINE_MODELS,
    DISCLAIMER,
)


def _load_featured(ticker: str, start_date, end_date) -> pd.DataFrame | None:
    return load_featured_prices(
        ticker,
        start=str(start_date) if start_date else None,
        end=str(end_date) if end_date else None,
    )


def render_models_evaluation(filters: Dict[str, Any]) -> None:
    st.header("2. Models & Evaluation")
    st.caption("Price history, classical ML, neural networks, and model evaluation")

    ticker = filters.get("ticker")
    start_date = filters.get("start_date")
    end_date = filters.get("end_date")
    horizon = filters.get("horizon", 1)
    selected_model = filters.get("model", "XGBoost")

    if not ticker:
        st.warning("Please select a ticker from the sidebar.")
        return

    with st.spinner("Loading and engineering features..."):
        df = _load_featured(ticker, start_date, end_date)

    if df is None:
        st.error(f"No data available for {ticker}")
        return
    if df.empty:
        st.error("No data left after feature engineering.")
        return

    section = render_segmented(MODEL_EVAL_SECTIONS, default="Classical ML")

    if section == "Classical ML":
        _render_classical(df, ticker, horizon, selected_model)
    elif section == "Neural Networks":
        _render_neural(df, ticker, horizon)
    else:
        _render_evaluation(df, ticker, horizon)

    st.markdown("---")
    st.caption(DISCLAIMER)


# ----------------------------------------------------------------------
# Classical ML (formerly ui/time_series_ml.py)
# ----------------------------------------------------------------------
def _render_classical(df: pd.DataFrame, ticker: str, horizon: int, selected_model: str) -> None:
    st.subheader("Price History")
    st.plotly_chart(
        plot_candlestick(df, title=f"{ticker} – Candlestick"),
        use_container_width=True,
    )

    with st.expander("Technical / Lag Features (preview)"):
        feature_cols = get_feature_columns(df)
        st.write(f"Feature columns ({len(feature_cols)}):")
        st.code(", ".join(feature_cols[:20]) + (" ..." if len(feature_cols) > 20 else ""))
        st.dataframe(df[feature_cols].tail(10), use_container_width=True)

    st.subheader(f"Model Comparison ({horizon}-Trading-Day Return)")

    ml_and_baselines = BASELINE_MODELS + [m for m in ML_MODELS if m in ML_MODEL_MAP]

    default_selection = [m for m in ["NaiveLastClose", "RandomForest", "XGBoost", "LightGBM"] if m in ml_and_baselines]
    models_to_run = st.multiselect(
        "Select models to compare",
        options=ml_and_baselines,
        default=default_selection,
        key="classical_models",
    )

    if st.button("Run ML Comparison", type="primary"):
        if not models_to_run:
            st.warning("Select at least one model.")
        else:
            X, y = create_supervised_dataset(df, target_col="target_return_1d", horizon=horizon)

            with st.spinner("Training and evaluating models..."):
                results_df = run_model_comparison(
                    X, y,
                    models=models_to_run,
                    test_size=0.2,
                    horizon=horizon,
                )

            _store_comparison(ticker, horizon, 0.2, tuple(models_to_run), results_df)
            _render_comparison(results_df)


def _render_comparison(results_df: pd.DataFrame, best_callout: bool = False) -> None:
    if results_df is None or results_df.empty:
        st.info("No comparison results to display yet.")
        return

    st.dataframe(
        results_df.style.format({
            "MAE": "{:.6f}",
            "RMSE": "{:.6f}",
            "DirectionalAccuracy": "{:.1f}%",
        }),
        use_container_width=True,
    )

    if "MAE" not in results_df.columns:
        return

    if best_callout:
        best = results_df.loc[results_df["MAE"].idxmin()]
        st.success(
            f"Best model by MAE: **{best['Model']}** "
            f"(MAE={best['MAE']:.6f}, Dir.Acc={best.get('DirectionalAccuracy', 0):.1f}%)"
        )

    fig_mae = px.bar(
        results_df,
        x="Model",
        y="MAE",
        color="Type" if "Type" in results_df.columns else None,
        title="Mean Absolute Error (lower is better)",
        template="plotly_white",
    )
    fig_mae.update_layout(height=380)
    st.plotly_chart(fig_mae, use_container_width=True)

    if "DirectionalAccuracy" in results_df.columns:
        fig_dir = px.bar(
            results_df,
            x="Model",
            y="DirectionalAccuracy",
            color="Type" if "Type" in results_df.columns else None,
            title="Directional Accuracy % (higher is better)",
            template="plotly_white",
        )
        fig_dir.update_layout(height=380)
        st.plotly_chart(fig_dir, use_container_width=True)


# ----------------------------------------------------------------------
# Neural Networks (formerly ui/neural_networks.py)
# ----------------------------------------------------------------------
def _render_neural(df: pd.DataFrame, ticker: str, horizon: int) -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        dl_model_name = st.selectbox("DL Model", options=DL_MODELS, index=0)
    with col2:
        seq_len = st.slider("Sequence Length", min_value=10, max_value=60, value=30, step=5)
    with col3:
        epochs = st.slider("Epochs", min_value=10, max_value=100, value=40, step=5)

    if len(df) < seq_len + 50:
        st.error(f"Not enough data for sequence length {seq_len}. Try a shorter sequence or longer history.")
        return

    X, y = create_supervised_dataset(df, target_col="target_return_1d", horizon=horizon)
    X_train, X_test, y_train, y_test = chronological_split(X, y, test_size=0.2, gap=horizon)

    if st.button(f"Train {dl_model_name}", type="primary"):
        with st.spinner(f"Training {dl_model_name} (this may take a minute)..."):
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            Xtr_seq, ytr_seq = create_sequences(X_train_s, y_train.values, seq_len)
            Xte_seq, yte_seq = create_sequences(X_test_s, y_test.values, seq_len)

            if len(Xtr_seq) < 20:
                st.error("Insufficient sequences after splitting. Reduce sequence length or use more history.")
                return

            n_features = Xtr_seq.shape[2]

            if dl_model_name == "LSTM":
                model = build_lstm(seq_len, n_features)
            else:
                model = build_gru(seq_len, n_features)

            val_idx = int(len(Xtr_seq) * 0.85)
            history = train_dl_model(
                model,
                Xtr_seq[:val_idx],
                ytr_seq[:val_idx],
                Xtr_seq[val_idx:],
                ytr_seq[val_idx:],
                epochs=epochs,
                verbose=0,
            )

            preds = predict_dl(model, Xte_seq)
            metrics = evaluate_regression(yte_seq, preds)

        st.subheader("Evaluation Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{metrics['MAE']:.6f}")
        m2.metric("RMSE", f"{metrics['RMSE']:.6f}")
        m3.metric("Directional Accuracy", f"{metrics['DirectionalAccuracy']:.1f}%")

        st.subheader("Training Curves")
        if history and history.history:
            hist = history.history
            fig_loss = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "MAE"))

            fig_loss.add_trace(go.Scatter(y=hist.get("loss", []), name="Train Loss"), row=1, col=1)
            if "val_loss" in hist:
                fig_loss.add_trace(go.Scatter(y=hist["val_loss"], name="Val Loss"), row=1, col=1)

            fig_loss.add_trace(go.Scatter(y=hist.get("mae", []), name="Train MAE"), row=1, col=2)
            if "val_mae" in hist:
                fig_loss.add_trace(go.Scatter(y=hist["val_mae"], name="Val MAE"), row=1, col=2)

            fig_loss.update_layout(height=380, template="plotly_white")
            st.plotly_chart(fig_loss, use_container_width=True)

        st.subheader("Forecast vs Actual (Test Set)")
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=yte_seq, mode="lines", name="Actual", line=dict(color="black")))
        fig_pred.add_trace(go.Scatter(y=preds, mode="lines", name="Predicted", line=dict(color="orange")))
        fig_pred.update_layout(
            title=f"{dl_model_name} – Test Predictions",
            xaxis_title="Test Sequence Index",
            yaxis_title="Return",
            height=400,
            template="plotly_white",
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        st.success(f"{dl_model_name} training completed.")
    else:
        st.info("Configure the parameters above and click **Train** to start the experiment.")


# ----------------------------------------------------------------------
# Evaluation (formerly ui/performance.py)
# ----------------------------------------------------------------------
def _render_evaluation(df: pd.DataFrame, ticker: str, horizon: int) -> None:
    st.caption(f"Evaluating cumulative returns over {horizon} trading day(s).")

    all_models = BASELINE_MODELS + list(ML_MODEL_MAP.keys()) + DL_MODELS
    default_models = ["NaiveLastClose", "NaiveLastReturn", "RandomForest", "XGBoost", "LightGBM"]

    selected_models = st.multiselect(
        "Models to evaluate",
        options=all_models,
        default=[m for m in default_models if m in all_models],
        help="Select the models you want to compare on the same chronological split",
        key="eval_models",
    )

    test_size = st.slider("Test set size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    run_walk_forward = st.checkbox("Also run walk-forward validation (slower)", value=False)

    if not selected_models:
        st.info("Select at least one model to begin evaluation.")
        return

    if df.empty or len(df) < 100:
        st.error("Insufficient data after feature engineering.")
        return

    X, y = create_supervised_dataset(df, target_col="target_return_1d", horizon=horizon)

    if st.button("Run Evaluation", type="primary"):
        with st.spinner("Training and evaluating models (this may take a while)..."):
            results_df = run_model_comparison(
                X, y,
                models=selected_models,
                test_size=test_size,
                horizon=horizon,
            )

        if results_df.empty:
            st.error("No results returned.")
            return

        _store_comparison(ticker, horizon, test_size, tuple(selected_models), results_df)

        st.subheader("Summary Table")
        _render_comparison(results_df, best_callout=True)

        if run_walk_forward:
            st.subheader("Walk-Forward Validation")
            wf_options = [m for m in selected_models if m in ML_MODEL_MAP]
            if not wf_options:
                st.info("Walk-forward validation is only available for classical ML models.")
            else:
                wf_model = st.selectbox(
                    "Model for walk-forward",
                    options=wf_options,
                    index=0,
                )
                with st.spinner(f"Running walk-forward for {wf_model}..."):
                    wf_result = walk_forward_validation(
                        X, y, model_name=wf_model, n_splits=5, horizon=horizon
                    )

                st.write("Average metrics across folds:")
                st.json(wf_result["average"])

                fold_df = pd.DataFrame(wf_result["fold_scores"])
                st.dataframe(fold_df.style.format("{:.4f}"), use_container_width=True)


# ----------------------------------------------------------------------
# Session-state cache for comparison results (shared across sub-sections)
# ----------------------------------------------------------------------
def _comparison_key(ticker: str, horizon: int, test_size: float, models: tuple) -> str:
    return f"cmp_{ticker}_{horizon}_{test_size}_{'_'.join(models)}"


def _store_comparison(ticker: str, horizon: int, test_size: float, models: tuple, results_df: pd.DataFrame) -> None:
    st.session_state[_comparison_key(ticker, horizon, test_size, models)] = results_df


def get_cached_comparison(ticker: str, horizon: int, test_size: float, models: tuple):
    """Return a previously cached comparison, or None."""
    return st.session_state.get(_comparison_key(ticker, horizon, test_size, models))