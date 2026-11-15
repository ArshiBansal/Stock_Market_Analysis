"""
Zephyr – Scenario Lab
Model-free scenario tools: Monte Carlo paths, first-passage probabilities,
volatility regimes, and shock what-if analysis.

All stochastic functions accept a ``seed`` and use ``config.RANDOM_SEED`` by default
so experiments are reproducible.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    RANDOM_SEED,
    MC_PATHS,
    MC_HORIZON,
    VOL_REGIME_WINDOW,
    VOL_REGIME_QUANTILES,
)
from utils.metrics import value_at_risk


# ----------------------------------------------------------------------
# Monte Carlo path simulation
# ----------------------------------------------------------------------
def simulate_paths(
    returns: pd.Series | np.ndarray,
    n_paths: int = MC_PATHS,
    horizon: int = MC_HORIZON,
    method: str = "bootstrap",
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """
    Simulate future cumulative price multipliers.

    Parameters
    ----------
    returns : historical simple returns (Series or array)
    n_paths : number of simulated paths
    horizon : number of steps per path
    method : "bootstrap" (resample historical returns) or "gbm" (geometric Brownian motion)
    seed : RNG seed

    Returns
    -------
    paths : ndarray of shape (n_paths, horizon) of cumulative price multipliers
            (1.0 == unchanged price).
    """
    if isinstance(returns, pd.Series):
        r = returns.dropna().to_numpy(dtype=float)
    else:
        r = np.asarray(returns, dtype=float).ravel()
        r = r[np.isfinite(r)]

    if r.size < 2:
        raise ValueError("Need at least 2 return observations to simulate.")

    if isinstance(n_paths, bool) or not isinstance(n_paths, int) or n_paths < 1:
        raise ValueError("n_paths must be a positive integer")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")

    method = (method or "bootstrap").lower()
    if method not in {"bootstrap", "gbm"}:
        raise ValueError("method must be 'bootstrap' or 'gbm'")

    rng = np.random.default_rng(seed)

    if method == "bootstrap":
        idx = rng.integers(0, r.size, size=(n_paths, horizon))
        steps = r[idx]
        paths = np.cumprod(1.0 + steps, axis=1)
    else:  # gbm
        mu = float(np.mean(r))
        sigma = float(np.std(r, ddof=1))
        z = rng.standard_normal(size=(n_paths, horizon))
        steps = (mu - 0.5 * sigma**2) + sigma * z
        paths = np.exp(np.cumsum(steps, axis=1))

    return paths


def fan_chart_stats(paths: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Per-step percentile bands for a fan chart.

    Returns dict of {p05, p25, p50, p75, p95} arrays of length ``horizon``.
    """
    p = np.asarray(paths, dtype=float)
    if p.ndim != 2:
        raise ValueError("paths must be 2-D (n_paths, horizon)")
    return {
        "p05": np.percentile(p, 5, axis=0),
        "p25": np.percentile(p, 25, axis=0),
        "p50": np.percentile(p, 50, axis=0),
        "p75": np.percentile(p, 75, axis=0),
        "p95": np.percentile(p, 95, axis=0),
    }


def terminal_distribution(paths: np.ndarray) -> np.ndarray:
    """Terminal (final-step) cumulative multipliers across paths."""
    p = np.asarray(paths, dtype=float)
    if p.ndim != 2:
        raise ValueError("paths must be 2-D (n_paths, horizon)")
    return p[:, -1]


def prob_up(paths: np.ndarray) -> float:
    """Fraction of paths ending above 1.0 (a gain)."""
    t = terminal_distribution(paths)
    if t.size == 0:
        return 0.0
    return float(np.mean(t > 1.0) * 100.0)


def prob_down(paths: np.ndarray) -> float:
    """Fraction of paths ending below 1.0 (a loss)."""
    t = terminal_distribution(paths)
    if t.size == 0:
        return 0.0
    return float(np.mean(t < 1.0) * 100.0)


# ----------------------------------------------------------------------
# First-passage probability
# ----------------------------------------------------------------------
def first_passage_prob(
    paths: np.ndarray,
    target: float,
    stop: float,
) -> float:
    """
    Fraction of paths that hit an upper ``target`` (cumulative return, e.g. 0.05)
    before hitting a lower ``stop`` (e.g. -0.05).

    Returns a percentage in [0, 100]. A path that hits neither counts as neither.
    """
    p = np.asarray(paths, dtype=float)
    if p.ndim != 2:
        raise ValueError("paths must be 2-D (n_paths, horizon)")
    if target <= 0:
        raise ValueError("target must be positive")
    if stop >= 0:
        raise ValueError("stop must be negative")
    if p.size == 0:
        return 0.0

    # Running cumulative max/min along each path.
    run_max = np.maximum.accumulate(p, axis=1)
    run_min = np.minimum.accumulate(p, axis=1)

    # First index where each bound is crossed (or horizon+1 if never).
    n_steps = p.shape[1]
    hit_target = np.argmax(run_max >= (1.0 + target), axis=1)
    hit_stop = np.argmax(run_min <= (1.0 + stop), axis=1)

    ever_target = (run_max >= (1.0 + target)).any(axis=1)
    ever_stop = (run_min <= (1.0 + stop)).any(axis=1)

    target_first = ever_target & (~ever_stop | (hit_target < hit_stop))
    return float(target_first.mean() * 100.0)


# ----------------------------------------------------------------------
# Volatility regimes
# ----------------------------------------------------------------------
def detect_vol_regimes(
    returns: pd.Series | np.ndarray,
    window: int = VOL_REGIME_WINDOW,
    quantiles: Tuple[float, float] = VOL_REGIME_QUANTILES,
) -> pd.Series:
    """
    Label each observation's volatility regime as Low / Normal / High.

    Regime is determined by the rolling standard deviation of returns, compared to
    the empirical ``quantiles`` (lower, upper) of that rolling-vol series.

    Returns a Series of labels indexed like the input.
    """
    if isinstance(returns, pd.Series):
        r = returns.copy()
    else:
        r = pd.Series(np.asarray(returns, dtype=float).ravel())

    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("window must be an integer >= 2")

    rolling_vol = r.rolling(window=window, min_periods=window).std()

    lo_q, hi_q = quantiles
    if not (0 <= lo_q < hi_q <= 1):
        raise ValueError("quantiles must satisfy 0 <= lo_q < hi_q <= 1")

    lo = rolling_vol.quantile(lo_q)
    hi = rolling_vol.quantile(hi_q)

    labels = pd.Series(index=r.index, dtype="object")
    labels[rolling_vol.isna()] = "Warmup"
    low_mask = rolling_vol <= lo
    high_mask = rolling_vol > hi
    labels[low_mask] = "Low"
    labels[high_mask] = "High"
    labels[rolling_vol.notna() & ~low_mask & ~high_mask] = "Normal"

    return labels


def regime_persistence(regimes: pd.Series) -> Dict[str, float]:
    """
    Share of time (%) spent in each regime and mean run length (in observations).
    """
    if regimes is None or len(regimes) == 0:
        return {}

    r = regimes.dropna()
    r = r[r != "Warmup"]
    if r.empty:
        return {}

    share = r.value_counts(normalize=True) * 100.0

    # Mean run length per regime.
    runs: Dict[str, list] = {}
    current = None
    length = 0
    for label in r.tolist():
        if label == current:
            length += 1
        else:
            if current is not None:
                runs.setdefault(current, []).append(length)
            current = label
            length = 1
    if current is not None:
        runs.setdefault(current, []).append(length)

    mean_len = {k: float(np.mean(v)) for k, v in runs.items()}

    return {
        "share_pct": {k: float(v) for k, v in share.items()},
        "mean_run_length": mean_len,
    }


# ----------------------------------------------------------------------
# Shock what-if
# ----------------------------------------------------------------------
def shock_impact(
    prices: pd.Series,
    shock_pct: float,
    window: Optional[int] = None,
) -> Dict[str, float]:
    """
    Apply an instantaneous shock of ``shock_pct`` (e.g. -0.05) to the last price
    and estimate the historical recovery time from similar shocks.

    Returns a dict with:
      - last_price, shocked_price, shock_pct
      - recovery_days_median : median days for the price to recover to the
        pre-shock level historically (NaN if no recovery found)
      - recovery_days_max
      - var_pct : historical 95% VaR of daily returns (%)
    """
    if isinstance(prices, pd.Series):
        p = prices.dropna().astype(float)
    else:
        p = pd.Series(np.asarray(prices, dtype=float).ravel()).dropna()

    if p.empty:
        raise ValueError("prices must be non-empty")
    if not np.isfinite(shock_pct) or shock_pct <= -1:
        raise ValueError("shock_pct must be finite and > -1 (e.g. -0.05 for -5%)")

    last_price = float(p.iloc[-1])
    shocked_price = last_price * (1.0 + shock_pct)

    # Estimate recovery time from historical shocks of similar magnitude.
    rets = p.pct_change().dropna()
    if rets.empty:
        return {
            "last_price": last_price,
            "shocked_price": shocked_price,
            "shock_pct": float(shock_pct),
            "recovery_days_median": float("nan"),
            "recovery_days_max": float("nan"),
            "var_pct": float("nan"),
        }

    # Find historical days with a drop at least as large as the shock (negative).
    threshold = shock_pct  # negative for a drop
    drop_idx = rets.index[rets <= threshold]

    recovery_days = []
    price_arr = p.to_numpy()
    idx_map = {ts: i for i, ts in enumerate(p.index)}
    for ts in drop_idx:
        i = idx_map[ts]
        pre_price = price_arr[i - 1] if i > 0 else price_arr[i]
        # Look forward for recovery to >= pre_price
        future = price_arr[i + 1:]
        recovered = np.where(future >= pre_price)[0]
        if recovered.size > 0:
            recovery_days.append(int(recovered[0] + 1))
        else:
            recovery_days.append(len(future))  # censored at end of history

    if window is not None and len(recovery_days) > window:
        recovery_days = recovery_days[-window:]

    return {
        "last_price": last_price,
        "shocked_price": shocked_price,
        "shock_pct": float(shock_pct),
        "recovery_days_median": float(np.median(recovery_days)) if recovery_days else float("nan"),
        "recovery_days_max": float(np.max(recovery_days)) if recovery_days else float("nan"),
        "var_pct": float(value_at_risk(rets)),
    }