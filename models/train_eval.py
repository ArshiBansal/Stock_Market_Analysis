"""
Zephyr – Deep Learning models (LSTM & GRU)
Uses TensorFlow / Keras.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import RANDOM_SEED

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential  # type: ignore[reportMissingModuleSource]
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout  # type: ignore[reportMissingModuleSource]
    from tensorflow.keras.callbacks import EarlyStopping  # type: ignore[reportMissingModuleSource]

    tf.random.set_seed(RANDOM_SEED)
    _HAS_TF = True
except ImportError:
    _HAS_TF = False


def _check_tf():
    if not _HAS_TF:
        raise ImportError(
            "TensorFlow is required for LSTM/GRU models. "
            "Install with: pip install tensorflow"
        )


def create_sequences(
    data: np.ndarray,
    target: np.ndarray,
    seq_len: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 2-D feature matrix into 3-D sequences for RNN models.
    data shape: (n_samples, n_features)
    Returns X of shape (n_sequences, seq_len, n_features)
    """
    Xs, ys = [], []
    for i in range(seq_len, len(data)):
        Xs.append(data[i - seq_len : i])
        ys.append(target[i])
    return np.array(Xs), np.array(ys)


def build_lstm(
    seq_len: int,
    n_features: int,
    units: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> "tf.keras.Model":
    _check_tf()
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(seq_len, n_features)),
        Dropout(dropout),
        LSTM(units // 2),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_gru(
    seq_len: int,
    n_features: int,
    units: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> "tf.keras.Model":
    _check_tf()
    model = Sequential([
        GRU(units, return_sequences=True, input_shape=(seq_len, n_features)),
        Dropout(dropout),
        GRU(units // 2),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_dl_model(
    model: "tf.keras.Model",
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 8,
    verbose: int = 0,
) -> "tf.keras.callbacks.History":
    """
    Train an LSTM/GRU model with early stopping.
    """
    _check_tf()
    callbacks = [
        EarlyStopping(
            monitor="val_loss" if X_val is not None else "loss",
            patience=patience,
            restore_best_weights=True,
            verbose=0,
        )
    ]

    validation_data = (X_val, y_val) if X_val is not None else None

    history = model.fit(
        X_train,
        y_train,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )
    return history


def predict_dl(model: "tf.keras.Model", X: np.ndarray) -> np.ndarray:
    """Generate predictions from a trained DL model."""
    _check_tf()
    preds = model.predict(X, verbose=0)
    return preds.ravel()