"""
Zephyr – Unit tests for metrics
(utils/metrics.py and models/train_eval.py evaluate_regression)
"""

import pytest
import numpy as np
import pandas as pd

from utils.metrics import mae, rmse, directional_accuracy, evaluate_forecast
from models.train_eval import evaluate_regression, directional_accuracy as dir_acc_train_eval


@pytest.fixture
def perfect_predictions():
    y_true = np.array([0.01, -0.02, 0.015, -0.005, 0.03])
    y_pred = y_true.copy()
    return y_true, y_pred


@pytest.fixture
def imperfect_predictions():
    y_true = np.array([0.01, -0.02, 0.015, -0.005, 0.03, 0.0, -0.01])
    y_pred = np.array([0.012, -0.015, 0.01, 0.002, 0.025, 0.001, -0.008])
    return y_true, y_pred


# ----------------------------------------------------------------------
# utils.metrics
# ----------------------------------------------------------------------
def test_mae_perfect(perfect_predictions):
    y_true, y_pred = perfect_predictions
    assert mae(y_true, y_pred) == pytest.approx(0.0)


def test_rmse_perfect(perfect_predictions):
    y_true, y_pred = perfect_predictions
    assert rmse(y_true, y_pred) == pytest.approx(0.0)


def test_mae_imperfect(imperfect_predictions):
    y_true, y_pred = imperfect_predictions
    result = mae(y_true, y_pred)
    assert result > 0
    assert isinstance(result, float)


def test_rmse_imperfect(imperfect_predictions):
    y_true, y_pred = imperfect_predictions
    result = rmse(y_true, y_pred)
    assert result > 0
    assert result >= mae(y_true, y_pred)  # RMSE >= MAE


def test_directional_accuracy_perfect(perfect_predictions):
    y_true, y_pred = perfect_predictions
    acc = directional_accuracy(y_true, y_pred)
    assert acc == pytest.approx(100.0)


def test_directional_accuracy_range(imperfect_predictions):
    y_true, y_pred = imperfect_predictions
    acc = directional_accuracy(y_true, y_pred)
    assert 0.0 <= acc <= 100.0


def test_directional_accuracy_ignores_zero_actual():
    y_true = np.array([0.0, 0.01, -0.02, 0.0, 0.03])
    y_pred = np.array([0.1, 0.012, -0.015, -0.5, 0.02])
    acc = directional_accuracy(y_true, y_pred)
    # Only three non-zero actuals, all correctly signed
    assert acc == pytest.approx(100.0)


def test_evaluate_forecast(imperfect_predictions):
    y_true, y_pred = imperfect_predictions
    result = evaluate_forecast(y_true, y_pred)
    
    assert isinstance(result, dict)
    assert "MAE" in result
    assert "RMSE" in result
    assert "DirectionalAccuracy" in result
    assert result["MAE"] > 0
    assert result["RMSE"] > 0


# ----------------------------------------------------------------------
# models.train_eval.evaluate_regression (the one used in the app)
# ----------------------------------------------------------------------
def test_evaluate_regression_structure(imperfect_predictions):
    y_true, y_pred = imperfect_predictions
    result = evaluate_regression(y_true, y_pred)
    
    assert set(result.keys()) == {"MAE", "RMSE", "DirectionalAccuracy"}
    assert isinstance(result["MAE"], float)
    assert isinstance(result["RMSE"], float)
    assert isinstance(result["DirectionalAccuracy"], float)


def test_evaluate_regression_perfect(perfect_predictions):
    y_true, y_pred = perfect_predictions
    result = evaluate_regression(y_true, y_pred)
    
    assert result["MAE"] == pytest.approx(0.0)
    assert result["RMSE"] == pytest.approx(0.0)
    assert result["DirectionalAccuracy"] == pytest.approx(100.0)


def test_directional_accuracy_consistency():
    """Both implementations should give the same result."""
    y_true = np.array([0.01, -0.02, 0.015, -0.005, 0.03])
    y_pred = np.array([0.012, -0.018, 0.01, 0.001, 0.028])
    
    acc1 = directional_accuracy(y_true, y_pred)
    acc2 = dir_acc_train_eval(y_true, y_pred)
    assert acc1 == pytest.approx(acc2)