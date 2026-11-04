"""
Zephyr – Tab 4: Neural Networks
LSTM / GRU experiments, training curves and sequence forecasts.
"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.fetch import fetch_ohlcv
from data.preprocess import clean_ohlcv, handle_missing_values
from data.features import add_technical_features, create_supervised_dataset
from models.dl_models import (
    build_lstm,
    build_gru,
    create_sequences,
    train_dl_model,
    predict_dl,
)
from models.train_eval import chronological_split, evaluate_regression
from sklearn.preprocessing import StandardScaler
from config import DL_MODELS, DISCLAIMER


def render_neural_networks(filters: Dict[str, Any]) -> None:
    st.header("4. Neural Networks")
    st.caption("LSTM & GRU sequence models for short-horizon forecasting")

    ticker = filters.get("ticker")
    start_date = filters.get("start_date")
    end_date = filters.get("end_date")
    horizon = filters.get("horizon", 1)

    if not ticker:
        st.warning("Please select a ticker from the sidebar.")
        return

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        dl_model_name = st.selectbox("DL Model", options=DL_MODELS, index=0)
    with col2:
        seq_len = st.slider("Sequence Length", min_value=10, max_value=60, value=30, step=5)
    with col3:
        epochs = st.slider("Epochs", min_value=10, max_value=100, value=40, step=5)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    with st.spinner("Preparing sequence data..."):
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

    if len(df) < seq_len + 50:
        st.error(f"Not enough data for sequence length {seq_len}. Try a shorter sequence or longer history.")
        return

    X, y = create_supervised_dataset(
        df, target_col="target_return_1d", horizon=horizon
    )
    X_train, X_test, y_train, y_test = chronological_split(
        X, y, test_size=0.2, gap=horizon
    )

    # ------------------------------------------------------------------
    # Train button
    # ------------------------------------------------------------------
    if st.button(f"Train {dl_model_name}", type="primary"):
        with st.spinner(f"Training {dl_model_name} (this may take a minute)..."):
            # Scale
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # Sequences
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

            # Validation split from training sequences
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

        # ------------------------------------------------------------------
        # Results
        # ------------------------------------------------------------------
        st.subheader("Evaluation Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{metrics['MAE']:.6f}")
        m2.metric("RMSE", f"{metrics['RMSE']:.6f}")
        m3.metric("Directional Accuracy", f"{metrics['DirectionalAccuracy']:.1f}%")

        # Training curves
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

        # Forecast vs Actual
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

    st.markdown("---")
    st.caption(DISCLAIMER)