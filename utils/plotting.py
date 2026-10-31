"""
Zephyr – Shared plotting helpers
Plotly-based charts used across multiple tabs.
"""

from __future__ import annotations

from typing import Optional, Dict, Any

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def plot_candlestick(
    df: pd.DataFrame,
    title: str = "OHLC Chart",
    height: int = 450,
) -> go.Figure:
    """
    Create a candlestick chart from an OHLCV DataFrame.
    Expects columns: Open, High, Low, Close (index = datetime).
    """
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="OHLC",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=height,
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_forecast_vs_actual(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    title: str = "Forecast vs Actual",
    height: int = 400,
) -> go.Figure:
    """
    Line chart comparing actual vs predicted values.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=np.asarray(y_true).ravel(),
            mode="lines",
            name="Actual",
            line=dict(color="black", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            y=np.asarray(y_pred).ravel(),
            mode="lines",
            name="Predicted",
            line=dict(color="orange", width=2),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Index",
        yaxis_title="Value",
        height=height,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_sentiment_distribution(
    agg: Dict[str, Dict[str, float]],
    title: str = "Sentiment Distribution",
    height: int = 320,
) -> go.Figure:
    """
    Horizontal bar or pie-friendly chart from the aggregate_sentiment output.
    """
    labels = list(agg.keys())
    counts = [agg[k]["count"] for k in labels]
    colors = {"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"}

    fig = go.Figure(
        go.Bar(
            x=counts,
            y=labels,
            orientation="h",
            marker_color=[colors.get(l, "#3498db") for l in labels],
            text=counts,
            textposition="auto",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Number of Headlines",
        height=height,
        template="plotly_white",
        margin=dict(l=80, r=40, t=50, b=40),
    )
    return fig


def plot_training_curves(
    history: Dict[str, list],
    height: int = 380,
) -> go.Figure:
    """
    Plot loss and MAE curves from a Keras History object (history.history).
    """
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "MAE"))

    if "loss" in history:
        fig.add_trace(
            go.Scatter(y=history["loss"], name="Train Loss", line=dict(color="steelblue")),
            row=1, col=1,
        )
    if "val_loss" in history:
        fig.add_trace(
            go.Scatter(y=history["val_loss"], name="Val Loss", line=dict(color="firebrick")),
            row=1, col=1,
        )

    if "mae" in history:
        fig.add_trace(
            go.Scatter(y=history["mae"], name="Train MAE", line=dict(color="steelblue")),
            row=1, col=2,
        )
    if "val_mae" in history:
        fig.add_trace(
            go.Scatter(y=history["val_mae"], name="Val MAE", line=dict(color="firebrick")),
            row=1, col=2,
        )

    fig.update_layout(
        height=height,
        template="plotly_white",
        showlegend=True,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig