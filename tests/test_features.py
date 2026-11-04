"""
Zephyr – Unit tests for feature engineering
(data/features.py)
"""

import pytest
import pandas as pd
import numpy as np

from data.features import (
    add_technical_features,
    get_feature_columns,
    create_supervised_dataset,
)


@pytest.fixture
def sample_ohlcv():
    """Synthetic OHLCV with enough rows for rolling features."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=120, freq="B")
    close = 100 + np.cumsum(np.random.randn(120) * 0.5)
    
    df = pd.DataFrame({
        "Open": close + np.random.randn(120) * 0.3,
        "High": close + np.abs(np.random.randn(120)),
        "Low":  close - np.abs(np.random.randn(120)),
        "Close": close,
        "Volume": np.random.randint(1_000_000, 8_000_000, 120),
    }, index=dates)
    
    df["High"] = df[["Open", "Close", "High"]].max(axis=1)
    df["Low"]  = df[["Open", "Close", "Low"]].min(axis=1)
    return df


def test_add_technical_features_shape(sample_ohlcv):
    df_feat = add_technical_features(sample_ohlcv, dropna=False)
    
    assert isinstance(df_feat, pd.DataFrame)
    assert len(df_feat) == len(sample_ohlcv)
    # New columns should have been added
    assert df_feat.shape[1] > sample_ohlcv.shape[1]


def test_add_technical_features_no_future_leakage(sample_ohlcv):
    """
    Basic leakage check: target columns should be shifted versions
    of return/close, and feature columns should not contain pure future info.
    """
    df_feat = add_technical_features(sample_ohlcv, dropna=False)
    
    assert "target_return_1d" in df_feat.columns
    assert "target_close" in df_feat.columns
    
    # target_return_1d should be return_1d shifted by -1
    expected = df_feat["return_1d"].shift(-1)
    # Allow small floating point differences
    diff = (df_feat["target_return_1d"] - expected).abs().dropna()
    assert (diff < 1e-8).all()


def test_required_feature_columns_exist(sample_ohlcv):
    df_feat = add_technical_features(sample_ohlcv, dropna=True)
    
    expected_some = [
        "return_1d", "sma_5", "sma_10", "sma_20",
        "volatility", "momentum", "volume_change"
    ]
    for col in expected_some:
        assert col in df_feat.columns, f"Missing expected feature: {col}"


def test_get_feature_columns(sample_ohlcv):
    df_feat = add_technical_features(sample_ohlcv, dropna=True)
    features = get_feature_columns(df_feat)
    
    assert isinstance(features, list)
    assert len(features) > 0
    
    # Should not contain raw price or target columns
    forbidden = {"Open", "High", "Low", "Close", "Volume", "target_return_1d", "target_close"}
    assert forbidden.isdisjoint(set(features))


def test_create_supervised_dataset(sample_ohlcv):
    df_feat = add_technical_features(sample_ohlcv, dropna=True)
    X, y = create_supervised_dataset(df_feat, target_col="target_return_1d")
    
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    assert X.isna().sum().sum() == 0
    assert y.isna().sum() == 0
    assert len(X) > 0


def test_features_with_dropna(sample_ohlcv):
    df_feat = add_technical_features(sample_ohlcv, dropna=True)
    # Keep the newest feature row for inference; its future target is unknown.
    feature_cols = get_feature_columns(df_feat)
    assert df_feat[feature_cols].isna().sum().sum() == 0
    assert df_feat.index[-1] == sample_ohlcv.index[-1]
    assert pd.isna(df_feat["target_return_1d"].iloc[-1])


@pytest.mark.parametrize("horizon", [1, 3, 5])
def test_create_supervised_dataset_horizon(sample_ohlcv, horizon):
    df_feat = add_technical_features(sample_ohlcv, dropna=True)
    X, y = create_supervised_dataset(df_feat, horizon=horizon)

    assert X.index.equals(y.index)
    assert y.index[-1] == df_feat.index[-horizon - 1]
    expected = (
        df_feat.loc[y.index[-1] + pd.offsets.BDay(horizon), "Close"]
        / df_feat.loc[y.index[-1], "Close"]
        - 1
    )
    assert y.iloc[-1] == pytest.approx(expected)


@pytest.mark.parametrize("horizon", [0, -1, 1.5, True])
def test_create_supervised_dataset_rejects_invalid_horizon(sample_ohlcv, horizon):
    df_feat = add_technical_features(sample_ohlcv, dropna=True)
    with pytest.raises(ValueError, match="positive integer"):
        create_supervised_dataset(df_feat, horizon=horizon)