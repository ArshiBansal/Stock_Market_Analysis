"""
Zephyr – Unit tests for signal generation
(models/signals.py)
"""

import numpy as np
import pytest

from models.signals import signals_from_predictions, signals_from_predictions_strided


def test_signs_and_threshold():
    preds = np.array([0.02, -0.03, 0.005, -0.004, 0.0])
    sig = signals_from_predictions(preds, threshold=0.01)
    assert sig.tolist() == [1, -1, 0, 0, 0]


def test_threshold_zero():
    preds = np.array([0.01, -0.01, 0.0])
    sig = signals_from_predictions(preds, threshold=0.0)
    assert sig.tolist() == [1, -1, 0]


def test_threshold_validation():
    with pytest.raises(ValueError):
        signals_from_predictions([0.1], threshold=-1.0)
    with pytest.raises(ValueError):
        signals_from_predictions([0.1], threshold=float("nan"))


def test_strided_keeps_every_nth():
    preds = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
    sig = signals_from_predictions_strided(preds, threshold=0.01, stride=2)
    # Only indices 0, 2, 4 kept
    assert sig.tolist() == [1, 0, 1, 0, 1]


def test_strided_stride_validation():
    with pytest.raises(ValueError):
        signals_from_predictions_strided([0.1], stride=0)


def test_length_preserved():
    preds = np.linspace(-0.02, 0.02, 11)
    assert len(signals_from_predictions(preds)) == 11
    assert len(signals_from_predictions_strided(preds, stride=3)) == 11