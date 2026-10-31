"""
Zephyr – Models layer
Baselines, classical ML, deep learning and training/evaluation utilities.
"""

from .baselines import NaiveLastClose, NaiveLastReturn, MovingAverageBaseline
from .ml_models import get_ml_model, ML_MODEL_MAP
from .dl_models import build_lstm, build_gru, train_dl_model, predict_dl
from .train_eval import (
    chronological_split,
    evaluate_regression,
    directional_accuracy,
    run_model_comparison,
    walk_forward_validation,
)

__all__ = [
    # Baselines
    "NaiveLastClose",
    "NaiveLastReturn",
    "MovingAverageBaseline",
    # ML
    "get_ml_model",
    "ML_MODEL_MAP",
    # DL
    "build_lstm",
    "build_gru",
    "train_dl_model",
    "predict_dl",
    # Train / Eval
    "chronological_split",
    "evaluate_regression",
    "directional_accuracy",
    "run_model_comparison",
    "walk_forward_validation",
]