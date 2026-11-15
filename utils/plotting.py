"""
Zephyr – Shared plotting helpers
Plotly-based charts used across multiple tabs.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


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