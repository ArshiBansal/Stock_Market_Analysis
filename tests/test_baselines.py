import numpy as np
import pandas as pd
import pytest

from models.baselines import NaiveLastClose, NaiveLastReturn
from models.train_eval import run_model_comparison


def test_naive_last_close_predicts_zero_return_for_feature_input():
    X = pd.DataFrame({"return_1d": [0.03, -0.02], "momentum": [0.1, 0.2]})

    predictions = NaiveLastClose().fit(X, pd.Series([0.04, 0.01])).predict(X)

    np.testing.assert_array_equal(predictions, np.zeros(len(X)))


def test_naive_last_return_predicts_each_row_observed_return():
    X = pd.DataFrame({"return_1d": [0.03, -0.02, 0.01]})
    model = NaiveLastReturn().fit(X.iloc[:2], pd.Series([0.5, 0.6]))

    np.testing.assert_allclose(model.predict(X.iloc[2:]), np.array([0.01]))


def test_naive_last_return_compounds_observed_return_over_horizon():
    X = pd.DataFrame({"return_1d": [0.03]})
    model = NaiveLastReturn(horizon=3).fit(X)

    assert model.predict(X)[0] == pytest.approx((1.03 ** 3) - 1)


def test_naive_last_return_requires_observed_return_feature():
    with pytest.raises(ValueError, match="return_1d"):
        NaiveLastReturn().fit(pd.DataFrame({"momentum": [0.1]}))


def test_model_comparison_uses_horizon_for_return_baseline():
    horizon = 3
    expected_return = (1.1 ** horizon) - 1
    X = pd.DataFrame({"return_1d": np.full(20, 0.1)})
    y = pd.Series(np.full(20, expected_return))

    results = run_model_comparison(
        X,
        y,
        models=["NaiveLastReturn", "NaiveLastClose"],
        test_size=0.25,
        horizon=horizon,
    ).set_index("Model")

    assert results.loc["NaiveLastReturn", "MAE"] == pytest.approx(0.0)
    assert results.loc["NaiveLastClose", "MAE"] == pytest.approx(expected_return)
