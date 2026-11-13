"""
Zephyr – Model persistence
Save/load trained models with joblib into artifacts/models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from config import MODEL_ARTIFACTS_DIR


def save_model(model: Any, name: str, ticker: str) -> Path:
    """Persist a fitted model to disk. Returns the artifact path."""
    path = MODEL_ARTIFACTS_DIR / f"{ticker}_{name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(name: str, ticker: str) -> Any:
    """Load a persisted model from disk."""
    path = MODEL_ARTIFACTS_DIR / f"{ticker}_{name}.joblib"
    return joblib.load(path)


def model_exists(name: str, ticker: str) -> bool:
    """Check whether a persisted model artifact exists."""
    return (MODEL_ARTIFACTS_DIR / f"{ticker}_{name}.joblib").exists()
