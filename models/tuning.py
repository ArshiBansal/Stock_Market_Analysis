"""
Zephyr – Hyperparameter tuning
Randomized search with time-series-aware cross-validation for classical ML models.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from config import RANDOM_SEED
from .ml_models import get_ml_model, ML_MODEL_MAP


def _check_model(name: str) -> None:
    if name not in ML_MODEL_MAP:
        raise ValueError(
            f"Unknown model: {name}. Available: {list(ML_MODEL_MAP)}"
        )
    if name == "Prophet":
        raise ValueError("Hyperparameter tuning is not supported for Prophet.")


def tune_model(
    X: pd.DataFrame,
    y,
    model_name: str,
    param_distributions: Dict[str, Any],
    n_iter: int = 20,
    cv_splits: int = 5,
    horizon: int = 1,
    random_state: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Randomized search over hyperparameters with expanding-window time-series CV.

    Parameters
    ----------
    X, y : training data
    model_name : key in ``ML_MODEL_MAP`` (Prophet excluded)
    param_distributions : dict of param -> list of values (scikit-learn search space)
    n_iter : number of parameter settings sampled
    cv_splits : number of TimeSeriesSplit folds
    horizon : gap between train and test folds (matches forecast horizon)

    Returns
    -------
    dict with best_params, best_score, cv_results (DataFrame), best_estimator
    """
    _check_model(model_name)

    if isinstance(cv_splits, bool) or not isinstance(cv_splits, int) or cv_splits < 2:
        raise ValueError("cv_splits must be an integer >= 2")
    if len(X) < cv_splits * 10:
        raise ValueError(
            f"Not enough rows for {cv_splits} CV folds (need >= {cv_splits * 10})."
        )

    estimator = get_ml_model(model_name, scale=True)
    cv = TimeSeriesSplit(n_splits=cv_splits, gap=horizon)

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_absolute_error",
        random_state=random_state,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    search.fit(X, y)

    cv_df = pd.DataFrame(search.cv_results_)
    # Keep a compact, readable subset if available.
    keep = [c for c in [
        "param_model__n_estimators", "param_model__max_depth",
        "param_model__learning_rate", "param_model__min_samples_leaf",
        "param_model__max_iter", "param_model__subsample",
        "mean_test_score", "std_test_score", "rank_test_score",
    ] if c in cv_df.columns]
    if keep:
        cv_df = cv_df[keep]

    return {
        "model": model_name,
        "best_params": search.best_params_,
        "best_score": -float(search.best_score_),  # convert back to MAE
        "cv_results": cv_df,
        "best_estimator": search.best_estimator_,
    }