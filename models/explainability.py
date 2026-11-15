"""
Zephyr – Model explainability
Global feature importance via permutation importance (always available) and,
when installed, SHAP values for tree/ML models.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from config import RANDOM_SEED, SHAP_SAMPLE_SIZE


def permutation_importance_df(
    model: Any,
    X: pd.DataFrame,
    y,
    n_repeats: int = 10,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Permutation importance (model-agnostic) on a MAE-based scoring.

    Returns a DataFrame with columns [feature, mean, std], sorted by mean desc.
    """
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    features = list(X.columns) if isinstance(X, pd.DataFrame) else [f"f{i}" for i in range(len(result.importances_mean))]
    df = pd.DataFrame({
        "feature": features,
        "mean": result.importances_mean,
        "std": result.importances_std,
    })
    # Higher (less negative) is more important; sort by magnitude descending.
    return df.reindex(df["mean"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def _has_shap() -> bool:
    try:
        import shap  # noqa: F401
        return True
    except ImportError:
        return False


def shap_importance(
    model: Any,
    X_sample: pd.DataFrame,
    max_samples: int = SHAP_SAMPLE_SIZE,
) -> pd.DataFrame:
    """
    SHAP global mean-|SHAP| importance.

    Raises ImportError with guidance if shap is unavailable, and a ValueError if
    the sample is too small.
    """
    if not _has_shap():
        raise ImportError(
            "shap is not installed. Install with: pip install shap "
            "(or use method='permutation')."
        )

    import shap

    Xs = X_sample
    if isinstance(Xs, pd.DataFrame) and len(Xs) > max_samples:
        Xs = Xs.iloc[:max_samples]

    try:
        explainer = shap.Explainer(model.predict, Xs)
    except Exception:
        # Fallback for plain estimators without a masker-compatible predict
        explainer = shap.Explainer(model, Xs)

    shap_values = explainer(Xs)

    # shap_values.values shape: (n_samples, n_features) or (n_samples, n_features, n_classes)
    values = np.asarray(shap_values.values)
    if values.ndim == 3:
        values = values[..., 0]
    mean_abs = np.abs(values).mean(axis=0)

    features = list(Xs.columns) if isinstance(Xs, pd.DataFrame) else [f"f{i}" for i in range(mean_abs.shape[0])]
    df = pd.DataFrame({"feature": features, "mean_abs_shap": mean_abs})
    return df.reindex(df["mean_abs_shap"].sort_values(ascending=False).index).reset_index(drop=True)


def explain(
    model: Any,
    X: pd.DataFrame,
    y,
    method: str = "auto",
    n_repeats: int = 10,
    sample_size: int = SHAP_SAMPLE_SIZE,
) -> dict:
    """
    Explain a fitted model.

    ``method`` in {"auto", "permutation", "shap"}.
    ``"auto"`` prefers SHAP and falls back to permutation if SHAP is unavailable.

    Returns
    -------
    dict with keys: method, importance (DataFrame), available (bool), note (str|None)
    """
    method = (method or "auto").lower()
    if method not in {"auto", "permutation", "shap"}:
        raise ValueError("method must be 'auto', 'permutation', or 'shap'")

    if method in {"shap", "auto"} and _has_shap():
        try:
            importance = shap_importance(model, X, max_samples=sample_size)
            return {
                "method": "shap",
                "importance": importance,
                "available": True,
                "note": None,
            }
        except Exception as e:  # noqa: BLE001 - fall back gracefully
            if method == "shap":
                raise
            # auto -> fall through to permutation
            return {
                "method": "permutation",
                "importance": permutation_importance_df(model, X, y, n_repeats=n_repeats),
                "available": True,
                "note": f"SHAP unavailable, used permutation instead: {e}",
            }

    if method == "shap" and not _has_shap():
        raise ImportError("shap is not installed.")

    return {
        "method": "permutation",
        "importance": permutation_importance_df(model, X, y, n_repeats=n_repeats),
        "available": True,
        "note": None if method == "permutation" else "SHAP not installed; used permutation.",
    }


def local_contributions(model: Any, X_row: pd.DataFrame) -> Optional[pd.Series]:
    """
    Per-feature SHAP contribution for a single row (e.g. the latest observation).

    Returns a Series indexed by feature name, or None if SHAP is unavailable.
    """
    if not _has_shap():
        return None

    import shap

    try:
        explainer = shap.Explainer(model.predict, X_row)
    except Exception:
        explainer = shap.Explainer(model, X_row)

    sv = explainer(X_row)
    values = np.asarray(sv.values)
    if values.ndim == 3:
        values = values[..., 0]
    contrib = values.ravel() if values.size == X_row.shape[1] else values[0]

    features = list(X_row.columns)
    return pd.Series(contrib, index=features, name="contribution")