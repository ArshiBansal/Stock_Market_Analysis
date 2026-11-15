"""
Zephyr – Unit tests for the experiment registry
(models/registry.py)
"""

import pandas as pd
import pytest

import models.registry as registry


@pytest.fixture
def tmp_registry(tmp_path, monkeypatch):
    """Point the registry at a temporary directory."""
    monkeypatch.setattr(registry, "EXPERIMENTS_DIR", tmp_path)
    monkeypatch.setattr(registry, "EXPERIMENTS_FILE", tmp_path / "experiments.jsonl")
    return tmp_path


def test_log_and_load_roundtrip(tmp_registry):
    run_id = registry.log_experiment({"category": "backtest", "ticker": "AAPL", "model": "XGBoost"})
    assert isinstance(run_id, str) and len(run_id) > 0

    df = registry.load_experiments()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["category"] == "backtest"
    assert row["ticker"] == "AAPL"
    assert "timestamp" in row and "run_id" in row


def test_load_empty_when_no_file(tmp_registry):
    df = registry.load_experiments()
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_clear_returns_count(tmp_registry):
    registry.log_experiment({"ticker": "MSFT"})
    registry.log_experiment({"ticker": "TSLA"})
    removed = registry.clear_experiments()
    assert removed == 2
    assert registry.load_experiments().empty


def test_summarize_filters(tmp_registry):
    registry.log_experiment({"ticker": "AAPL", "model": "XGBoost"})
    registry.log_experiment({"ticker": "MSFT", "model": "RandomForest"})
    df = registry.load_experiments()
    filtered = registry.summarize_experiments(df, {"ticker": "AAPL"})
    assert len(filtered) == 1
    assert filtered.iloc[0]["ticker"] == "AAPL"


def test_log_injects_run_id_and_timestamp(tmp_registry):
    registry.log_experiment({"ticker": "NVDA"})
    df = registry.load_experiments()
    assert df.iloc[0]["run_id"]
    assert df.iloc[0]["timestamp"]