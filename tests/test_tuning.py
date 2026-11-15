"""
Zephyr – Unit tests for hyperparameter tuning
(models/tuning.py)
"""

import numpy as np
import pandas as pd
import pytest

from models.tuning import tune_model


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "c": rng.normal(size=n),
    })
    y = pd.Series(2.0 * X["a"] - X["b"] + rng.normal(0, 0.1, n))
    return X, y


def test_tune_model_returns_expected_keys():
    X, y = _data()
    grid = {
        "model__n_estimators": [10, 20],
        "model__max_depth": [2, 4],
    }
    result = tune_model(
        X, y, model_name="RandomForest",
        param_distributions=grid,
        n_iter=2, cv_splits=3, horizon=1,
    )
    assert "best_params" in result
    assert "best_score" in result
    assert "cv_results" in result
    assert isinstance(result["cv_results"], pd.DataFrame)
    assert result["best_score"] >= 0.0


def test_tune_model_rejects_prophet():
    X, y = _data()
    with pytest.raises(ValueError):
        tune_model(X, y, model_name="Prophet", param_distributions={"x": [1]}, n_iter=1, cv_splits=2)


def test_tune_model_cv_splits_validation():
    X, y = _data()
    with pytest.raises(ValueError):
        tune_model(X, y, model_name="RandomForest", param_distributions={"model__n_estimators": [10]}, cv_splits=1)


def test_tune_model_insufficient_data():
    X, y = _data(n=15)
    with pytest.raises(ValueError):
        tune_model(X, y, model_name="RandomForest", param_distributions={"model__n_estimators": [10]}, cv_splits=5)