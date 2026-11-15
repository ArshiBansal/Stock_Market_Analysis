"""
Zephyr – Tab: Scenario Lab
Model-free scenario tools: Monte Carlo fan chart, target/stop-touch probability,
volatility regime map, and interactive shock what-if.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.features import load_featured_prices
from models.scenario import (
    simulate_paths,
    fan_chart_stats,
    terminal_distribution,
    prob_up,
    prob_down,
    first_passage_prob,
    detect_vol_regimes,
    regime_persistence,
    shock_impact,
)
from ui._common import render_segmented
from config import (
    SCENARIO_SECTIONS,
    MC_PATHS,
    MC_HORIZON,
    MC_METHODS,
    VOL_REGIME_WINDOW,
    VOL_REGIME_QUANTILES,
    SCENARIO_SAMPLE_PATHS,
    RANDOM_SEED,
    DISCLAIMER,
)


def _load_prices(ticker: str, start_date, end_date) -> pd.DataFrame | None:
    return load_featured_prices(
        ticker,
        start=str(start_date) if start_date else None,
        end=str(end_date) if end_date else None,
    )


def render_scenario(filters: Dict[str, Any]) -> None:
    st.header("6. Scenario Lab")
    st.caption("Monte Carlo scenarios, first-passage probabilities, volatility regimes, and shocks")

    ticker = filters.get("ticker")
    start_date = filters.get("start_date")
    end_date = filters.get("end_date")

    section = render_segmented(SCENARIO_SECTIONS, default="Monte Carlo")

    if section == "Monte Carlo":
        _render_monte_carlo(ticker, start_date, end_date)
    elif section == "Target / Stop":
        _render_target_stop(ticker, start_date, end_date)
    elif section == "Volatility Regime":
        _render_regime(ticker, start_date, end_date)
    else:
        _render_shock(ticker, start_date, end_date)

    st.markdown("---")
    st.caption(DISCLAIMER)


def _require_ticker(ticker: str) -> bool:
    if not ticker:
        st.warning("Please select a ticker from the sidebar.")
        return False
    return True


def _get_returns(ticker: str, start_date, end_date):
    df = _load_prices(ticker, start_date, end_date)
    if df is None or df.empty:
        st.error(f"No data available for {ticker}")
        return None, None
    returns = df["Close"].pct_change().dropna()
    if len(returns) < 30:
        st.error("Not enough history to build scenarios (need >= 30 observations).")
        return None, None
    return df, returns


# ----------------------------------------------------------------------
# Monte Carlo fan chart
# ----------------------------------------------------------------------
def _render_monte_carlo(ticker: str, start_date, end_date) -> None:
    if not _require_ticker(ticker):
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        method = st.selectbox("Simulation method", options=MC_METHODS, index=0,
                              help="bootstrap resamples historical returns; gbm assumes log-normal")
    with c2:
        n_paths = st.slider("Simulated paths", min_value=200, max_value=5000, value=MC_PATHS, step=200)
    with c3:
        horizon = st.slider("Horizon (trading days)", min_value=5, max_value=60, value=MC_HORIZON, step=1)

    df, returns = _get_returns(ticker, start_date, end_date)
    if returns is None:
        return

    last_price = float(df["Close"].iloc[-1])
    paths = simulate_paths(returns, n_paths=n_paths, horizon=horizon, method=method, seed=RANDOM_SEED)
    stats = fan_chart_stats(paths)

    # Fan chart (multipliers -> price)
    x = list(range(1, horizon + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=stats["p05"] * last_price, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=stats["p95"] * last_price, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(31,119,180,0.12)", name="5–95%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=stats["p25"] * last_price, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=stats["p75"] * last_price, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(31,119,180,0.25)", name="25–75%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=stats["p50"] * last_price, mode="lines", line=dict(color="navy", width=2), name="Median"))
    fig.add_hline(y=last_price, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=f"{ticker} – {method.upper()} Fan Chart ({n_paths} paths)",
        xaxis_title="Trading days ahead",
        yaxis_title="Price",
        height=460,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("P(Up)", f"{prob_up(paths):.1f}%")
    m2.metric("P(Down)", f"{prob_down(paths):.1f}%")
    m3.metric("Median terminal", f"{(stats['p50'][-1]*last_price):,.2f}")
    m4.metric("90% range", f"[{(stats['p05'][-1]*last_price):,.2f}, {(stats['p95'][-1]*last_price):,.2f}]")

    # Terminal distribution
    terminal = terminal_distribution(paths) * last_price
    fig_hist = go.Figure(go.Histogram(x=terminal, nbinsx=50, marker_color="steelblue"))
    fig_hist.add_vline(x=last_price, line_dash="dash", line_color="gray")
    fig_hist.update_layout(
        title="Terminal Price Distribution",
        xaxis_title="Price",
        yaxis_title="Frequency",
        height=340,
        template="plotly_white",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.caption(
        "Monte Carlo resamples the historical return distribution; it is a model of the past, "
        "not a prediction. It cannot anticipate regime change, news, or structural breaks."
    )


# ----------------------------------------------------------------------
# Target / Stop first-passage
# ----------------------------------------------------------------------
def _render_target_stop(ticker: str, start_date, end_date) -> None:
    if not _require_ticker(ticker):
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        target_pct = st.slider("Target (+%)", min_value=1.0, max_value=30.0, value=5.0, step=0.5)
    with c2:
        stop_pct = st.slider("Stop (-%)", min_value=1.0, max_value=30.0, value=5.0, step=0.5)
    with c3:
        n_paths = st.slider("Simulated paths", min_value=200, max_value=5000, value=MC_PATHS, step=200)
    with c4:
        horizon = st.slider("Horizon (trading days)", min_value=5, max_value=60, value=MC_HORIZON, step=1)

    df, returns = _get_returns(ticker, start_date, end_date)
    if returns is None:
        return

    paths = simulate_paths(returns, n_paths=n_paths, horizon=horizon, method="bootstrap", seed=RANDOM_SEED)
    prob = first_passage_prob(paths, target=target_pct / 100.0, stop=-(stop_pct / 100.0))

    st.subheader("First-Passage Probability")
    st.metric(
        f"P(hit +{target_pct:.1f}% before -{stop_pct:.1f}%)",
        f"{prob:.1f}%",
        help="Estimated from simulated paths (bootstrap of historical returns).",
    )
    st.progress(min(max(prob / 100.0, 0.0), 1.0))

    st.caption(
        f"Over {horizon} trading days. A path that hits neither bound counts as neither. "
        "Probability is descriptive of the historical distribution, not a guarantee."
    )

    # Sample path overlays
    n_show = min(SCENARIO_SAMPLE_PATHS, paths.shape[0])
    x = list(range(1, horizon + 1))
    fig = go.Figure()
    for i in range(n_show):
        fig.add_trace(go.Scatter(
            x=x, y=paths[i] * 100 - 100,
            mode="lines",
            line=dict(width=1),
            opacity=0.5,
            showlegend=False,
            hoverinfo="skip",
        ))
    fig.add_hline(y=target_pct, line_dash="dash", line_color="green")
    fig.add_hline(y=-stop_pct, line_dash="dash", line_color="red")
    fig.update_layout(
        title=f"{n_show} sample paths (cumulative return %)",
        xaxis_title="Trading days ahead",
        yaxis_title="Return %",
        height=420,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------
# Volatility regime map
# ----------------------------------------------------------------------
def _render_regime(ticker: str, start_date, end_date) -> None:
    if not _require_ticker(ticker):
        return

    df = _load_prices(ticker, start_date, end_date)
    if df is None or df.empty:
        st.error(f"No data available for {ticker}")
        return

    returns = df["Close"].pct_change().dropna()
    regimes = detect_vol_regimes(
        returns,
        window=VOL_REGIME_WINDOW,
        quantiles=VOL_REGIME_QUANTILES,
    )

    # Align regimes to price index
    aligned = regimes.reindex(df.index)

    color_map = {"Low": "#2ca02c", "Normal": "#7f7f7f", "High": "#d62728", "Warmup": "#bcbd22"}

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        line=dict(color="black", width=1),
        name="Close",
    ))
    for label, color in color_map.items():
        mask = aligned == label
        if mask.any():
            fig.add_trace(go.Scatter(
                x=df.index[mask],
                y=df["Close"][mask],
                mode="markers",
                marker=dict(color=color, size=4),
                name=label,
            ))
    fig.update_layout(
        title=f"{ticker} – Volatility Regimes",
        xaxis_title="Date",
        yaxis_title="Close",
        height=440,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    persist = regime_persistence(aligned)
    if persist:
        share = persist["share_pct"]
        run_len = persist["mean_run_length"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Low-vol share", f"{share.get('Low', 0):.1f}%")
        m2.metric("Normal-vol share", f"{share.get('Normal', 0):.1f}%")
        m3.metric("High-vol share", f"{share.get('High', 0):.1f}%")

        rows = []
        for label in ["Low", "Normal", "High"]:
            rows.append({
                "Regime": label,
                "Share %": round(share.get(label, 0.0), 2),
                "Mean run length": round(run_len.get(label, 0.0), 2),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    current = aligned.dropna().iloc[-1] if aligned.notna().any() else "N/A"
    st.info(f"Current regime: **{current}** (rolling {VOL_REGIME_WINDOW}-day volatility)")

    st.caption(
        "Regimes are descriptive (rolling-volatility quantiles), not a predictive model. "
        "Persistence measures how long the market historically stayed in each regime."
    )


# ----------------------------------------------------------------------
# Shock what-if
# ----------------------------------------------------------------------
def _render_shock(ticker: str, start_date, end_date) -> None:
    if not _require_ticker(ticker):
        return

    df = _load_prices(ticker, start_date, end_date)
    if df is None or df.empty:
        st.error(f"No data available for {ticker}")
        return

    last_price = float(df["Close"].iloc[-1])
    st.metric("Last Close", f"{last_price:,.2f}")

    c1, c2 = st.columns(2)
    with c1:
        shock_pct = st.slider(
            "Instant shock (%)",
            min_value=-20.0,
            max_value=20.0,
            value=-5.0,
            step=0.5,
            help="Negative = drop; positive = gap up.",
        )
    with c2:
        window = st.slider(
            "Recovery history window (recent N shocks, 0 = all)",
            min_value=0,
            max_value=50,
            value=0,
            step=5,
        )

    impact = shock_impact(df["Close"], shock_pct=shock_pct / 100.0, window=(window or None))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Shocked Price", f"{impact['shocked_price']:,.2f}")
    m2.metric("Shock", f"{impact['shock_pct']*100:+.2f}%")
    m3.metric("Median Recovery", f"{impact['recovery_days_median']:.1f} days" if np.isfinite(impact['recovery_days_median']) else "N/A")
    m4.metric("Max Recovery", f"{impact['recovery_days_max']:.1f} days" if np.isfinite(impact['recovery_days_max']) else "N/A")

    st.metric("Historical VaR (95%)", f"{impact['var_pct']:.2f}%")

    st.caption(
        "Recovery time is estimated from historical days with a drop at least as large as the shock: "
        "median/max trading days for the price to return to the pre-shock level. It is descriptive, "
        "not a forecast, and future recoveries can be faster, slower, or never."
    )

    # Visualize the shock on the recent price history
    recent = df["Close"].tail(120)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent.index, y=recent.values, mode="lines", line=dict(color="steelblue"), name="Close"))
    fig.add_hline(y=impact["shocked_price"], line_dash="dash", line_color="red",
                  annotation_text=f"shock {impact['shock_pct']*100:+.1f}%")
    fig.update_layout(
        title=f"{ticker} – Shock Scenario",
        xaxis_title="Date",
        yaxis_title="Price",
        height=360,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)