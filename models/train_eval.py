"""
Zephyr – Training & Evaluation utilities
Chronological splitting, metrics, model comparison and walk-forward validation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from config import RANDOM_SEED
from .baselines import NaiveLastClose, NaiveLastReturn, MovingAverageBaseline
from .ml_models import get_ml_model, ML_MODEL_MAP
from .dl_models import (
    build_lstm,
    build_gru,
    create_sequences,
    train_dl_model,
    predict_dl,
)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Percentage of times the predicted direction matches the actual direction.
    Assumes both are returns (or differences). Zero actuals are ignored.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0

    correct = np.sign(y_true[mask]) == np.sign(y_pred[mask])
    return float(np.mean(correct) * 100)


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Return MAE, RMSE and Directional Accuracy."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mae = mean_absolute_error(y_true, y_pred)

    # Compatible with all recent scikit-learn versions
    # (the 'squared' argument was removed)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    dir_acc = directional_accuracy(y_true, y_pred)

    return {
        "MAE": round(mae, 6),
        "RMSE": round(rmse, 6),
        "DirectionalAccuracy": round(dir_acc, 2),
    }


# ----------------------------------------------------------------------
# Splitting
# ----------------------------------------------------------------------
def chronological_split(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    test_size: float = 0.2,
) -> Tuple[Any, Any, Any, Any]:
    """
    Simple chronological train/test split (no shuffling).
    """
    n = len(X)
    split_idx = int(n * (1 - test_size))

    if isinstance(X, pd.DataFrame):
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    else:
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test


# ----------------------------------------------------------------------
# Model runners
# ----------------------------------------------------------------------
def _run_baseline(
    name: str,
    X_train, y_train, X_test, y_test,
) -> Dict[str, Any]:
    if name == "NaiveLastClose":
        model = NaiveLastClose()
    elif name == "NaiveLastReturn":
        model = NaiveLastReturn()
    elif name == "MovingAverage":
        model = MovingAverageBaseline(window=5)
    else:
        raise ValueError(f"Unknown baseline: {name}")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = evaluate_regression(y_test, preds)
    return {"model": name, "type": "baseline", "metrics": metrics, "predictions": preds}


def _run_ml(
    name: str,
    X_train, y_train, X_test, y_test,
    **kwargs,
) -> Dict[str, Any]:
    model = get_ml_model(name, scale=True, **kwargs)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = evaluate_regression(y_test, preds)
    return {"model": name, "type": "ml", "metrics": metrics, "predictions": preds}


def _run_dl(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seq_len: int = 30,
    epochs: int = 40,
) -> Dict[str, Any]:
    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Create sequences
    Xtr_seq, ytr_seq = create_sequences(X_train_s, y_train.values, seq_len)
    Xte_seq, yte_seq = create_sequences(X_test_s, y_test.values, seq_len)

    if len(Xtr_seq) < 20 or len(Xte_seq) < 5:
        return {
            "model": name,
            "type": "dl",
            "metrics": {"MAE": np.nan, "RMSE": np.nan, "DirectionalAccuracy": np.nan},
            "predictions": np.array([]),
            "error": "Insufficient data for sequences",
        }

    n_features = Xtr_seq.shape[2]

    if name == "LSTM":
        model = build_lstm(seq_len, n_features)
    elif name == "GRU":
        model = build_gru(seq_len, n_features)
    else:
        raise ValueError(f"Unknown DL model: {name}")

    # Small validation split from the end of training sequences
    val_split = int(len(Xtr_seq) * 0.85)
    history = train_dl_model(
        model,
        Xtr_seq[:val_split],
        ytr_seq[:val_split],
        Xtr_seq[val_split:],
        ytr_seq[val_split:],
        epochs=epochs,
        verbose=0,
    )

    preds = predict_dl(model, Xte_seq)
    metrics = evaluate_regression(yte_seq, preds)

    return {
        "model": name,
        "type": "dl",
        "metrics": metrics,
        "predictions": preds,
        "history": history.history if history else None,
    }


# ----------------------------------------------------------------------
# High-level comparison
# ----------------------------------------------------------------------
def run_model_comparison(
    X: pd.DataFrame,
    y: pd.Series,
    models: Optional[List[str]] = None,
    test_size: float = 0.2,
    seq_len: int = 30,
) -> pd.DataFrame:
    """
    Run a set of models on the same chronological split and return a summary table.
    """
    if models is None:
        models = [
            "NaiveLastClose",
            "NaiveLastReturn",
            "RandomForest",
            "XGBoost",
            "LightGBM",
            "LSTM",
        ]

    X_train, X_test, y_train, y_test = chronological_split(X, y, test_size=test_size)

    results = []

    for name in models:
        try:
            if name in ("NaiveLastClose", "NaiveLastReturn", "MovingAverage"):
                res = _run_baseline(name, X_train, y_train, X_test, y_test)
            elif name in ML_MODEL_MAP:
                res = _run_ml(name, X_train, y_train, X_test, y_test)
            elif name in ("LSTM", "GRU"):
                res = _run_dl(name, X_train, y_train, X_test, y_test, seq_len=seq_len)
            else:
                continue

            row = {"Model": res["model"], "Type": res["type"]}
            row.update(res["metrics"])
            results.append(row)

        except Exception as e:
            results.append({
                "Model": name,
                "Type": "error",
                "MAE": np.nan,
                "RMSE": np.nan,
                "DirectionalAccuracy": np.nan,
                "Error": str(e),
            })

    df = pd.DataFrame(results)
    if not df.empty and "MAE" in df.columns:
        df = df.sort_values("MAE")
    return df.reset_index(drop=True)


def walk_forward_validation(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "XGBoost",
    n_splits: int = 5,
    **model_kwargs,
) -> Dict[str, Any]:
    """
    Simple walk-forward (expanding window) validation using TimeSeriesSplit.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        if model_name in ML_MODEL_MAP:
            model = get_ml_model(model_name, **model_kwargs)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
        else:
            # Fallback to baseline for simplicity
            model = NaiveLastReturn()
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)

        metrics = evaluate_regression(y_te, preds)
        scores.append(metrics)

    avg = {
        "MAE": round(np.mean([s["MAE"] for s in scores]), 6),
        "RMSE": round(np.mean([s["RMSE"] for s in scores]), 6),
        "DirectionalAccuracy": round(np.mean([s["DirectionalAccuracy"] for s in scores]), 2),
    }
    return {"model": model_name, "n_splits": n_splits, "fold_scores": scores, "average": avg}