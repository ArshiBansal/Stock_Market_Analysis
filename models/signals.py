"""
Zephyr – Signal generation
Convert model return predictions into directional trading signals for the backtest engine.
"""

from __future__ import annotations

import numpy as np


def _validate_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("threshold must be a number")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be a finite, non-negative number")
    return float(threshold)


def signals_from_predictions(
    preds,
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Map return predictions to directional signals.

    ``+1`` when pred > +threshold, ``-1`` when pred < -threshold, else ``0`` (flat).
    """
    threshold = _validate_threshold(threshold)
    p = np.asarray(preds, dtype=float).ravel()
    signals = np.zeros_like(p, dtype=int)
    signals[p > threshold] = 1
    signals[p < -threshold] = -1
    return signals


def signals_from_predictions_strided(
    preds,
    threshold: float = 0.0,
    stride: int = 1,
) -> np.ndarray:
    """
    Non-overlapping variant.

    Keeps every ``stride``-th prediction (others set to ``0``/flat) so that, when
    ``stride`` equals the forecast horizon, consecutive traded signals do not
    overlap in the return window. Reduces the overlapping-horizon approximation
    inherent in ``signals_from_predictions``.
    """
    threshold = _validate_threshold(threshold)
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")

    p = np.asarray(preds, dtype=float).ravel()
    base = signals_from_predictions(p, threshold=threshold)

    strided = np.zeros_like(base, dtype=int)
    strided[::stride] = base[::stride]
    return strided