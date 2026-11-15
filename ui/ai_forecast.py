"""
Zephyr – Tab 6: AI Forecast
Forecast chart, horizon, selected model, latest observed timestamp and limitations.
"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st
import plotly.graph_objects as go

from data.fetch import fetch_ohlcv, fetch_latest_quote
from data.preprocess import clean_ohlcv, handle_missing_values
from data.features import (
    add_technical_features,
    create_supervised_dataset,
    get_feature_columns,
    get_cached_supervised_dataset,
)
from models.ml_models import get_ml_model, ML_MODEL_MAP
from models.baselines import NaiveLastClose, NaiveLastReturn
from models.train_eval import chronological_split, evaluate_regression
from models.persistence import save_model, load_model, model_exists
from models.uncertainty import (
    conformal_half_width,
    interval_coverage,
    prediction_interval,
)
from config import (
    DISCLAIMER,
    BASELINE_MODELS,
    FORECAST_INTERVAL_LEVELS,
    DEFAULT_INTERVAL_LEVEL,
)


def render_ai_forecast(filters: Dict[str, Any]) -> None:
    st.header("6. AI Forecast")
    st.caption("Single-model forecast for the selected trading-day horizon")

    ticker = filters.get("ticker")
    start_date = filters.get("start_date")
    end_date = filters.get("end_date")
    horizon = filters.get("horizon", 1)
    selected_model = filters.get("model", "XGBoost")

    if not ticker:
        st.warning("Please select a ticker from the sidebar.")
        return

    # ------------------------------------------------------------------
    # Latest quote for context
    # ------------------------------------------------------------------
    quote = fetch_latest_quote(ticker)
    last_price = quote.get("price")
    last_ts = quote.get("timestamp")
    stale = quote.get("stale", False)

    col1, col2, col3 = st.columns(3)
    col1.metric("Selected Model", selected_model)
    col2.metric("Forecast Horizon", f"{horizon} day(s)")
    col3.metric(
        "Last Observed Price",
        f"{last_price:,.2f}" if last_price else "N/A",
        help=str(last_ts) if last_ts else None,
    )

    interval_level = st.select_slider(
        "Prediction interval level",
        options=FORECAST_INTERVAL_LEVELS,
        value=DEFAULT_INTERVAL_LEVEL,
        format_func=lambda v: f"{int(v * 100)}%",
        help="Split-conformal interval around the point forecast (calibrated on the test set)",
    )

    if stale:
        st.warning("⚠️ Latest data appears stale. Forecasts are based on the most recent available observation.")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    with st.spinner("Preparing forecast data..."):
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
        st.error("No usable data after feature engineering.")
        return

    cached = get_cached_supervised_dataset(
        ticker, horizon=horizon, start=str(start_date), end=str(end_date)
    )
    if cached is not None:
        X, y, _ = cached
    else:
        X, y = create_supervised_dataset(df, target_col="target_return_1d", horizon=horizon)
    X_train, X_test, y_train, y_test = chronological_split(
        X, y, test_size=0.2, gap=horizon
    )
    latest_features = df[get_feature_columns(df)].iloc[[-1]]

    # ------------------------------------------------------------------
    # Train selected model
    # ------------------------------------------------------------------
    if st.button("Generate Forecast", type="primary"):
        with st.spinner(f"Training {selected_model} and generating forecast..."):
            try:
                if selected_model in BASELINE_MODELS:
                    if selected_model == "NaiveLastClose":
                        model = NaiveLastClose()
                    else:
                        model = NaiveLastReturn(horizon=horizon)
                    model.fit(X_train, y_train)
                    test_preds = model.predict(X_test)
                    point_forecast_return = float(model.predict(latest_features)[0])

                elif selected_model in ML_MODEL_MAP:
                    model_path_key = f"{ticker}_{selected_model}"
                    if model_exists(selected_model, ticker):
                        model = load_model(selected_model, ticker)
                    else:
                        model = get_ml_model(selected_model, scale=True)
                        model.fit(X_train, y_train)
                        save_model(model, selected_model, ticker)
                    test_preds = model.predict(X_test)
                    point_forecast_return = float(model.predict(latest_features)[0])

                else:
                    st.warning(
                        f"Deep-learning model **{selected_model}** is best explored in the "
                        "Neural Networks tab. Falling back to XGBoost for this view."
                    )
                    model_path_key = "ticker_XGBoost"
                    if model_exists("XGBoost", ticker):
                        model = load_model("XGBoost", ticker)
                    else:
                        model = get_ml_model("XGBoost", scale=True)
                        model.fit(X_train, y_train)
                        save_model(model, "XGBoost", ticker)
                    test_preds = model.predict(X_test)
                    point_forecast_return = float(model.predict(latest_features)[0])

                metrics = evaluate_regression(y_test, test_preds)

                # Split-conformal interval (calibrated on the holdout)
                q_half = conformal_half_width(y_test, test_preds, alpha=1 - interval_level)
                coverage = interval_coverage(y_test, test_preds, q_half)

            except Exception as e:
                st.error(f"Forecast generation failed: {e}")
                return

        # ------------------------------------------------------------------
        # Results
        # ------------------------------------------------------------------
        st.subheader("Test-Set Performance")
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{metrics['MAE']:.6f}")
        m2.metric("RMSE", f"{metrics['RMSE']:.6f}")
        m3.metric("Directional Accuracy", f"{metrics['DirectionalAccuracy']:.1f}%")

        # Interval calibration summary
        st.subheader("Prediction Interval Calibration")
        c1, c2 = st.columns(2)
        c1.metric(
            f"{int(interval_level * 100)}% Interval Half-Width",
            f"{q_half*100:.3f}%",
            help="Point forecast +/- this half-width forms the interval",
        )
        c2.metric(
            "Empirical Coverage",
            f"{coverage:.1f}%",
            help="Share of holdout points that fall inside the interval",
        )

        # Point forecast interpretation
        st.subheader(f"{horizon}-Trading-Day Forecast")
        lo_ret, hi_ret = prediction_interval(point_forecast_return, q_half)
        if last_price:
            lo_price = last_price * (1 + lo_ret)
            hi_price = last_price * (1 + hi_ret)
            expected_price = last_price * (1 + point_forecast_return)
            st.metric(
                "Expected Return",
                f"{point_forecast_return*100:+.3f}%",
                help=f"{int(interval_level * 100)}% interval: "
                f"[{lo_ret*100:+.3f}%, {hi_ret*100:+.3f}%]",
            )
            st.metric(
                "Implied Price (approx)",
                f"{expected_price:,.2f}",
                help=f"{int(interval_level * 100)}% interval: "
                f"[{lo_price:,.2f}, {hi_price:,.2f}] – illustrative only",
            )
        else:
            st.metric(
                "Predicted Return",
                f"{point_forecast_return*100:+.3f}%",
                help=f"{int(interval_level * 100)}% interval: "
                f"[{lo_ret*100:+.3f}%, {hi_ret*100:+.3f}%]",
            )

        # Forecast vs Actual chart (test period) with interval band
        st.subheader("Forecast vs Actual (Test Period)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_test.index,
            y=y_test.values,
            mode="lines",
            name="Actual Return",
            line=dict(color="black"),
        ))
        fig.add_trace(go.Scatter(
            x=y_test.index,
            y=test_preds + q_half,
            mode="lines",
            name=f"Upper ({int(interval_level*100)}%)",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=y_test.index,
            y=test_preds - q_half,
            mode="lines",
            name=f"Lower ({int(interval_level*100)}%)",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(255,165,0,0.18)",
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=y_test.index,
            y=test_preds,
            mode="lines",
            name="Predicted Return",
            line=dict(color="orange"),
        ))
        fig.update_layout(
            title=f"{selected_model} – {horizon}-Trading-Day Test Set",
            xaxis_title="Forecast origin",
            yaxis_title=f"{horizon}-day cumulative return",
            height=400,
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Limitations box
        st.info(
            """
            **Limitations & Notes**
            - Forecasts are for educational purposes only and are **not financial advice**.
            - Models use only information available at prediction time (no future leakage).
            - Short-horizon forecasts are noisy; directional accuracy is often close to chance.
            - Prediction intervals use split-conformal calibration on the holdout; coverage is
              approximate and can degrade under regime change or distribution shift.
            - Market regime changes, news events and liquidity can invalidate historical patterns.
            - Always combine quantitative signals with fundamental and risk analysis.
            """
        )

    else:
        st.info("Click **Generate Forecast** to train the selected model and produce a prediction.")

    st.markdown("---")
    st.caption(DISCLAIMER)