"""
Zephyr – Unit tests for explainability
(models/explainability.py)
"""

import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import RandomForestRegressor

import models.explainability as expl
from models.explainability import permutation_importance_df, explain


def _data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "noise": rng.normal(size=n),
    })
    y = 3.0 * X["a"] - 2.0 * X["b"] + rng.normal(0, 0.1, n)
    return X, y


def test_permutation_importance_shape_and_order():
    X, y = _data()
    model = RandomForestRegressor(n_estimators=20, random_state=0).fit(X, y)
    imp = permutation_importance_df(model, X, y, n_repeats=3)
    assert set(imp.columns) >= {"feature", "mean"}
    assert len(imp) == X.shape[1]
    # Sorted by |mean| descending
    mags = imp["mean"].abs().tolist()
    assert mags == sorted(mags, reverse=True)


def test_permutation_ranks_informative_features_high():
    X, y = _data()
    model = RandomForestRegressor(n_estimators=30, random_state=0).fit(X, y)
    imp = permutation_importance_df(model, X, y, n_repeats=5)
    ranked = imp.set_index("feature")["mean"].abs()
    assert ranked["a"] > ranked["noise"]
    assert ranked["b"] > ranked["noise"]


def test_explain_auto_falls_back_when_shap_missing(monkeypatch):
    X, y = _data()
    model = RandomForestRegressor(n_estimators=20, random_state=0).fit(X, y)

    monkeypatch.setattr(expl, "_has_shap", lambda: False)
    result = explain(model, X, y, method="auto")
    assert result["method"] == "permutation"
    assert result["available"] is True
    assert not result["importance"].empty


def test_explain_shap_without_shap_raises(monkeypatch):
    X, y = _data()
    model = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
    monkeypatch.setattr(expl, "_has_shap", lambda: False)
    with pytest.raises(ImportError):
        explain(model, X, y, method="shap")