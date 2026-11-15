"""
Zephyr – Tab: Research
Explainability, hyperparameter tuning, backtesting, and the experiment registry.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data.features import create_supervised_dataset, load_featured_prices
from models.train_eval import chronological_split
from models.ml_models import ML_MODEL_MAP
from models.baselines import NaiveLastClose, NaiveLastReturn
from models.explainability import explain, local_contributions, _has_shap
from models.tuning import tune_model
from models.signals import signals_from_predictions, signals_from_predictions_strided
from models.backtest import run_backtest, BacktestConfig
from models.registry import (
    log_experiment,
    load_experiments,
    clear_experiments,
    summarize_experiments,
)
from ui._common import render_segmented
from config import (
    RESEARCH_SECTIONS,
    EXPLAIN_METHODS,
    PERMUTATION_REPEATS,
    TUNING_N_ITER,
    TUNING_CV_SPLITS,
    PARAM_GRIDS,
    BACKTEST_DEFAULTS,
    DEFAULT_SIGNAL_THRESHOLD,
    BASELINE_MODELS,
    DISCLAIMER,
)


def _load_featured(ticker: str, start_date, end_date):
    return load_featured_prices(
        ticker,
        start=str(start_date) if start_date else None,
        end=str(end_date) if end_date else None,
    )


def render_research(filters: Dict[str, Any]) -> None:
    st.header("5. Research")
    st.caption("Explainability, hyperparameter tuning, backtesting, and experiment registry")

    ticker = filters.get("ticker")
    start_date = filters.get("start_date")
    end_date = filters.get("end_date")
    horizon = filters.get("horizon", 1)

    section = render_segmented(RESEARCH_SECTIONS, default="Explainability")

    if section == "Explainability":
        _render_explainability(ticker, start_date, end_date, horizon)
    elif section == "Hyperparameter Tuning":
        _render_tuning(ticker, start_date, end_date, horizon)
    elif section == "Backtest":
        _render_backtest(ticker, start_date, end_date, horizon)
    else:
        _render_experiments()

    st.markdown("---")
    st.caption(DISCLAIMER)


# ----------------------------------------------------------------------
# Explainability
# ----------------------------------------------------------------------
def _render_explainability(ticker: str, start_date, end_date, horizon: int) -> None:
    if not ticker:
        st.warning("Please select a ticker from the sidebar.")
        return

    st.write("Global feature importance for a trained model.")
    if not _has_shap():
        st.info("SHAP is not installed; only permutation importance is available. Install with `pip install shap`.")

    c1, c2 = st.columns(2)
    with c1:
        model_name = st.selectbox(
            "Model",
            options=list(ML_MODEL_MAP.keys()),
            index=list(ML_MODEL_MAP.keys()).index("XGBoost") if "XGBoost" in ML_MODEL_MAP else 0,
        )
    with c2:
        method = st.selectbox("Method", options=["auto", *EXPLAIN_METHODS], index=0)

    if not st.button("Explain Model", type="primary"):
        st.info("Select a model and click **Explain Model**.")
        return

    with st.spinner(f"Training {model_name} and computing importance..."):
        df = _load_featured(ticker, start_date, end_date)
        if df is None or df.empty:
            st.error(f"No data available for {ticker}")
            return

        X, y = create_supervised_dataset(df, target_col="target_return_1d", horizon=horizon)
        X_train, X_test, y_train, y_test = chronological_split(X, y, test_size=0.2, gap=horizon)

        from models.ml_models import get_ml_model
        model = get_ml_model(model_name, scale=True)
        model.fit(X_train, y_train)

        result = explain(model, X_test, y_test, method=method, n_repeats=PERMUTATION_REPEATS)

    st.subheader(f"{result['method'].title()} Feature Importance")
    if result.get("note"):
        st.info(result["note"])

    importance = result["importance"]
    if importance is None or importance.empty:
        st.warning("No importance values were returned.")
        return

    top = importance.head(20)

    if result["method"] == "shap":
        fig = px.bar(
            top.iloc[::-1],
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            title="Mean |SHAP| (global importance)",
            template="plotly_white",
        )
        x_label = "Mean |SHAP|"
    else:
        fig = px.bar(
            top.iloc[::-1],
            x="mean",
            y="feature",
            orientation="h",
            title="Permutation importance (Δ MAE when shuffled)",
            template="plotly_white",
            error_x="std" if "std" in top.columns else None,
        )
        x_label = "Importance"

    fig.update_layout(height=520, xaxis_title=x_label)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(top, use_container_width=True)

    # Local contribution for the latest row (SHAP only)
    if result["method"] == "shap":
        try:
            latest = X_test.iloc[[-1]]
            contrib = local_contributions(model, latest)
            if contrib is not None and not contrib.empty:
                st.subheader("Latest Observation – Local Contributions")
                contrib_sorted = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
                fig_local = px.bar(
                    contrib_sorted,
                    title="Feature contribution to the latest prediction",
                    template="plotly_white",
                )
                fig_local.update_layout(height=420)
                st.plotly_chart(fig_local, use_container_width=True)
        except Exception as e:  # noqa: BLE001
            st.caption(f"Local explanation unavailable: {e}")


# ----------------------------------------------------------------------
# Hyperparameter tuning
# ----------------------------------------------------------------------
def _render_tuning(ticker: str, start_date, end_date, horizon: int) -> None:
    if not ticker:
        st.warning("Please select a ticker from the sidebar.")
        return

    tunable = [m for m in ML_MODEL_MAP.keys() if m in PARAM_GRIDS]
    if not tunable:
        st.info("No tunable models are available.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        model_name = st.selectbox("Model to tune", options=tunable, index=0)
    with c2:
        n_iter = st.slider("Search iterations", min_value=5, max_value=40, value=TUNING_N_ITER, step=5)
    with c3:
        cv_splits = st.slider("CV folds", min_value=2, max_value=8, value=TUNING_CV_SPLITS, step=1)

    if not st.button("Run Tuning", type="primary"):
        st.info("Configure the search and click **Run Tuning**.")
        return

    with st.spinner(f"Running randomized search for {model_name} (this may take a while)..."):
        df = _load_featured(ticker, start_date, end_date)
        if df is None or df.empty:
            st.error(f"No data available for {ticker}")
            return

        X, y = create_supervised_dataset(df, target_col="target_return_1d", horizon=horizon)

        try:
            result = tune_model(
                X, y,
                model_name=model_name,
                param_distributions=PARAM_GRIDS[model_name],
                n_iter=n_iter,
                cv_splits=cv_splits,
                horizon=horizon,
            )
        except ValueError as e:
            st.error(f"Tuning failed: {e}")
            return

    st.subheader("Best Parameters")
    st.json(result["best_params"])
    st.metric("Best CV MAE", f"{result['best_score']:.6f}")

    st.subheader("Top Search Results")
    cv_df = result["cv_results"].copy()
    if "mean_test_score" in cv_df.columns:
        cv_df = cv_df.sort_values("mean_test_score", ascending=False)
        cv_df["mean_test_score"] = -cv_df["mean_test_score"]  # show as MAE
        cv_df = cv_df.rename(columns={"mean_test_score": "MAE", "std_test_score": "MAE_std"})
    st.dataframe(cv_df.head(15), use_container_width=True)


# ----------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------
def _render_backtest(ticker: str, start_date, end_date, horizon: int) -> None:
    if not ticker:
        st.warning("Please select a ticker from the sidebar.")
        return

    st.caption(
        "Turn model return predictions into directional signals and simulate a "
        "trade with costs, slippage, and an optional stop-loss."
    )

    c1, c2 = st.columns(2)
    with c1:
        model_name = st.selectbox(
            "Signal model",
            options=BASELINE_MODELS + list(ML_MODEL_MAP.keys()),
            index=0,
        )
    with c2:
        signal_threshold = st.number_input(
            "Signal threshold (return)",
            min_value=0.0,
            max_value=0.05,
            value=DEFAULT_SIGNAL_THRESHOLD,
            step=0.001,
            format="%.3f",
            help="Predictions above +threshold -> long, below -threshold -> short, else flat",
        )

    c3, c4 = st.columns(2)
    with c3:
        stride_mode = st.checkbox(
            "Non-overlapping signals (stride = horizon)",
            value=False,
            help="Keeps every horizon-th signal so trades do not overlap in the return window.",
        )
    with c4:
        initial_capital = st.number_input(
            "Initial capital",
            min_value=1000.0,
            value=BACKTEST_DEFAULTS["initial_capital"],
            step=1000.0,
        )

    c5, c6, c7 = st.columns(3)
    with c5:
        transaction_cost_bps = st.number_input(
            "Transaction cost (bps/side)",
            min_value=0.0,
            value=BACKTEST_DEFAULTS["transaction_cost_bps"],
            step=0.5,
        )
    with c6:
        slippage_bps = st.number_input(
            "Slippage (bps/side)",
            min_value=0.0,
            value=BACKTEST_DEFAULTS["slippage_bps"],
            step=0.1,
        )
    with c7:
        stop_loss_pct = st.number_input(
            "Stop loss (%, 0 = off)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=1.0,
        )
    stop_loss = -(stop_loss_pct / 100.0) if stop_loss_pct > 0 else None

    if not st.button("Run Backtest", type="primary"):
        st.info("Configure the strategy and click **Run Backtest**.")
        return

    with st.spinner(f"Training {model_name} and running backtest..."):
        df = _load_featured(ticker, start_date, end_date)
        if df is None or df.empty:
            st.error(f"No data available for {ticker}")
            return

        X, y = create_supervised_dataset(df, target_col="target_return_1d", horizon=horizon)
        X_train, X_test, y_train, y_test = chronological_split(X, y, test_size=0.2, gap=horizon)

        from models.ml_models import get_ml_model

        if model_name in BASELINE_MODELS:
            model = NaiveLastClose() if model_name == "NaiveLastClose" else NaiveLastReturn(horizon=horizon)
            model.fit(X_train, y_train)
        else:
            model = get_ml_model(model_name, scale=True)
            model.fit(X_train, y_train)

        preds = model.predict(X_test)

        if stride_mode:
            signals = signals_from_predictions_strided(preds, threshold=signal_threshold, stride=horizon)
        else:
            signals = signals_from_predictions(preds, threshold=signal_threshold)

        prices = df.loc[X_test.index, "Close"]
        if len(signals) != len(prices):
            st.error("Signal/price alignment error.")
            return

        config = BacktestConfig(
            initial_capital=initial_capital,
            position_size=BACKTEST_DEFAULTS["position_size"],
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            stop_loss=stop_loss,
        )
        result = run_backtest(prices, signals, config)

    st.subheader("Backtest Summary")
    summary = result.to_dict()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Return", f"{summary['total_return_pct']:.2f}%")
    m2.metric("Final Value", f"{summary['final_value']:,.2f}")
    m3.metric("Trades", summary["trades"])
    m4.metric("Win Rate", f"{summary['win_rate_pct']:.1f}%")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Sharpe", f"{summary['sharpe']:.2f}")
    m6.metric("Max Drawdown", f"{summary['max_drawdown_pct']:.2f}%")
    m7.metric("VaR (95%)", f"{summary['var_pct']:.2f}%")
    m8.metric("Initial Capital", f"{summary['initial_capital']:,.2f}")

    st.caption(
        "Signals are produced per forecast origin over the selected horizon; the engine "
        "trades them as a daily series. Enable non-overlapping mode to reduce the "
        "overlapping-horizon approximation."
    )

    st.subheader("Equity Curve")
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=result.equity_curve.index,
        y=result.equity_curve.values,
        mode="lines",
        name="Equity",
        line=dict(color="steelblue"),
    ))
    fig_eq.add_hline(y=config.initial_capital, line_dash="dash", line_color="gray")
    fig_eq.update_layout(
        title="Portfolio Equity",
        xaxis_title="Date",
        yaxis_title="Equity",
        height=380,
        template="plotly_white",
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    st.subheader("Drawdown")
    equity = result.equity_curve
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max * 100
    fig_dd = go.Figure(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        mode="lines",
        name="Drawdown %",
        fill="tozeroy",
        line=dict(color="firebrick"),
    ))
    fig_dd.update_layout(
        title="Drawdown (%)",
        xaxis_title="Date",
        yaxis_title="Drawdown %",
        height=320,
        template="plotly_white",
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    record = st.checkbox("Record this run to the experiment registry", value=True)
    if record:
        run_id = log_experiment({
            "category": "backtest",
            "ticker": ticker,
            "horizon": horizon,
            "model": model_name,
            "signal_threshold": signal_threshold,
            "stride": horizon if stride_mode else 1,
            "MAE": None,
            "RMSE": None,
            "DirectionalAccuracy": None,
            "total_return_pct": summary["total_return_pct"],
            "sharpe": summary["sharpe"],
            "max_drawdown_pct": summary["max_drawdown_pct"],
            "trades": summary["trades"],
        })
        st.success(f"Recorded run `{run_id[:8]}`.")


# ----------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------
def _render_experiments() -> None:
    st.write("Browse and compare previously recorded experiment runs.")

    df = load_experiments()
    if df.empty:
        st.info("No experiments recorded yet. Run a backtest, tuning, or explainability job and record it.")
        return

    tickers = sorted([t for t in df["ticker"].dropna().unique()]) if "ticker" in df.columns else []
    models = sorted([m for m in df["model"].dropna().unique()]) if "model" in df.columns else []

    c1, c2 = st.columns(2)
    with c1:
        ticker_filter = st.selectbox("Filter by ticker", options=["All"] + tickers, index=0)
    with c2:
        model_filter = st.selectbox("Filter by model", options=["All"] + models, index=0)

    filtered = summarize_experiments(df, {
        "ticker": ticker_filter,
        "model": model_filter,
    })

    st.dataframe(filtered, use_container_width=True)

    # Compare MAE across runs when available
    if "MAE" in filtered.columns and filtered["MAE"].notna().any():
        mae_df = filtered.dropna(subset=["MAE"]).copy()
        if not mae_df.empty:
            mae_df["label"] = mae_df.get("model", pd.Series(range(len(mae_df)))).astype(str) + " (" + mae_df.get(
                "ticker", pd.Series([""] * len(mae_df))
            ).astype(str) + ")"
            fig = px.bar(
                mae_df.sort_values("MAE"),
                x="label",
                y="MAE",
                title="MAE by recorded run (lower is better)",
                template="plotly_white",
            )
            fig.update_layout(height=380, xaxis_title="Run")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    if st.button("Clear experiment registry", type="secondary"):
        removed = clear_experiments()
        st.success(f"Cleared {removed} record(s).")
        st.rerun()