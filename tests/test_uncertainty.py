"""
Zephyr – Unit tests for conformal prediction intervals
(models/uncertainty.py)
"""

import numpy as np
import pytest

from models.uncertainty import (
    conformal_half_width,
    interval_coverage,
    prediction_interval,
)


def test_conformal_width_non_negative():
    y_true = np.array([0.01, -0.02, 0.015, -0.005, 0.03])
    y_pred = np.array([0.012, -0.015, 0.01, 0.002, 0.025])
    assert conformal_half_width(y_true, y_pred, alpha=0.1) >= 0.0


def test_conformal_alpha_validation():
    y = np.array([0.01, -0.02, 0.015])
    with pytest.raises(ValueError):
        conformal_half_width(y, y, alpha=0.0)
    with pytest.raises(ValueError):
        conformal_half_width(y, y, alpha=1.0)


def test_conformal_length_mismatch():
    with pytest.raises(ValueError):
        conformal_half_width(np.array([1.0, 2.0]), np.array([1.0]))


def test_interval_bounds_ordering():
    point = 0.02
    half = 0.01
    lo, hi = prediction_interval(point, half)
    assert lo <= point <= hi
    assert lo == pytest.approx(0.01)
    assert hi == pytest.approx(0.03)


def test_interval_negative_half_raises():
    with pytest.raises(ValueError):
        prediction_interval(0.0, -1.0)


def test_coverage_increases_with_width():
    rng = np.random.default_rng(0)
    y_true = rng.normal(0, 0.02, 200)
    y_pred = y_true + rng.normal(0, 0.005, 200)
    narrow = conformal_half_width(y_true, y_pred, alpha=0.5)
    wide = conformal_half_width(y_true, y_pred, alpha=0.05)
    assert interval_coverage(y_true, y_pred, wide) >= interval_coverage(y_true, y_pred, narrow)


def test_coverage_range():
    y_true = np.array([0.01, -0.02, 0.015, -0.005, 0.03])
    y_pred = np.array([0.012, -0.015, 0.01, 0.002, 0.025])
    cov = interval_coverage(y_true, y_pred, 1.0)
    assert 0.0 <= cov <= 100.0