"""
Zephyr – Utility helpers
Metrics, plotting and general helper functions.
"""

from .metrics import (
    mae,
    rmse,
    directional_accuracy,
    evaluate_forecast,
    sharpe_ratio,
    max_drawdown,
    value_at_risk,
)
from .plotting import (
    plot_candlestick,
    plot_forecast_vs_actual,
    plot_training_curves,
)
from .helpers import (
    format_timestamp,
    is_stale,
    safe_divide,
    ensure_datetime_index,
    get_cache_ttl,
)

__all__ = [
    # Metrics
    "mae",
    "rmse",
    "directional_accuracy",
    "evaluate_forecast",
    "sharpe_ratio",
    "max_drawdown",
    "value_at_risk",
    # Plotting
    "plot_candlestick",
    "plot_forecast_vs_actual",
    "plot_training_curves",
    # Helpers
    "format_timestamp",
    "is_stale",
    "safe_divide",
    "ensure_datetime_index",
    "get_cache_ttl",
]