"""
Zephyr – Uncertainty quantification
Split-conformal prediction intervals for point forecasts.

Model-agnostic: works with any fitted regressor (baselines, sklearn pipelines,
or any object exposing ``predict``). Uses the absolute-residual nonconformity
score on a calibration set (the holdout) to produce a symmetric interval.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _as_1d(a) -> np.ndarray:
    """Coerce an array-like into a 1-D float array."""
    return np.asarray(a, dtype=float).ravel()


def conformal_half_width(y_true, y_pred, alpha: float = 0.1) -> float:
    """
    Split-conformal interval half-width using absolute residuals.

    Parameters
    ----------
    y_true : array-like of actual values (calibration set)
    y_pred : array-like of predictions on the same rows
    alpha : miscoverage level (0 < alpha < 1); e.g. 0.1 -> ~90% interval

    Returns
    -------
    half_width : non-negative float; ``point +/- half_width`` is the interval.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    yt = _as_1d(y_true)
    yp = _as_1d(y_pred)
    if yt.size == 0 or yp.size == 0:
        raise ValueError("y_true and y_pred must be non-empty")
    if yt.size != yp.size:
        raise ValueError("y_true and y_pred must have the same length")

    scores = np.abs(yt - yp)
    n = scores.size

    # Finite-sample corrected quantile level.
    q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, q_level))


def interval_coverage(y_true, y_pred, half_width: float) -> float:
    """
    Empirical coverage (%) of ``point +/- half_width`` on the given rows.
    """
    yt = _as_1d(y_true)
    yp = _as_1d(y_pred)
    if yt.size == 0:
        return 0.0
    if yt.size != yp.size:
        raise ValueError("y_true and y_pred must have the same length")
    if half_width < 0:
        raise ValueError("half_width must be non-negative")

    inside = np.abs(yt - yp) <= half_width
    return float(np.mean(inside) * 100.0)


def prediction_interval(point: float, half_width: float) -> Tuple[float, float]:
    """
    Symmetric prediction interval around ``point``.

    Returns
    -------
    (lo, hi) with ``lo <= point <= hi``.
    """
    if half_width < 0:
        raise ValueError("half_width must be non-negative")
    return float(point - half_width), float(point + half_width)