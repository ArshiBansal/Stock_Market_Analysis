"""
Zephyr – Classical Machine Learning models
RandomForest, ExtraTrees, XGBoost, LightGBM, HistGradientBoosting
"""

from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import RANDOM_SEED

# Optional imports – fail gracefully if not installed
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False


def _make_rf(**kwargs) -> RandomForestRegressor:
    defaults = dict(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    defaults.update(kwargs)
    return RandomForestRegressor(**defaults)


def _make_et(**kwargs) -> ExtraTreesRegressor:
    defaults = dict(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    defaults.update(kwargs)
    return ExtraTreesRegressor(**defaults)


def _make_hgb(**kwargs) -> HistGradientBoostingRegressor:
    defaults = dict(
        max_iter=200,
        max_depth=8,
        learning_rate=0.05,
        random_state=RANDOM_SEED,
    )
    defaults.update(kwargs)
    return HistGradientBoostingRegressor(**defaults)


def _make_xgb(**kwargs):
    if not _HAS_XGB:
        raise ImportError("xgboost is not installed")
    defaults = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
    )
    defaults.update(kwargs)
    return XGBRegressor(**defaults)


def _make_lgbm(**kwargs):
    if not _HAS_LGBM:
        raise ImportError("lightgbm is not installed")
    defaults = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    defaults.update(kwargs)
    return LGBMRegressor(**defaults)


ML_MODEL_MAP = {
    "RandomForest": _make_rf,
    "ExtraTrees": _make_et,
    "HistGradientBoosting": _make_hgb,
    "XGBoost": _make_xgb,
    "LightGBM": _make_lgbm,
}


def get_ml_model(
    name: str,
    scale: bool = True,
    **kwargs,
) -> Any:
    """
    Factory that returns a ready-to-fit model (optionally wrapped in a scaling pipeline).

    Parameters
    ----------
    name : one of the keys in ML_MODEL_MAP
    scale : whether to wrap with StandardScaler (recommended for most models)
    **kwargs : extra hyperparameters passed to the underlying model
    """
    if name not in ML_MODEL_MAP:
        raise ValueError(f"Unknown ML model: {name}. Available: {list(ML_MODEL_MAP)}")

    model = ML_MODEL_MAP[name](**kwargs)

    if scale:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])
    return model