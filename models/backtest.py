"""
Zephyr – Backtesting engine
Simple event-driven backtest with transaction costs, slippage and position sizing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from utils.metrics import max_drawdown, sharpe_ratio, value_at_risk


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    position_size: float = 1.0  # fraction of capital per trade
    transaction_cost_bps: float = 1.0  # bps per side
    slippage_bps: float = 0.5  # bps per side
    stop_loss: Optional[float] = None  # e.g. -0.05 for 5%


@dataclass
class BacktestResult:
    config: BacktestConfig
    equity_curve: pd.Series
    returns: pd.Series
    trades: int
    win_rate: float
    total_return: float
    sharpe: float
    max_dd: float
    var: float
    final_value: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_capital": self.config.initial_capital,
            "final_value": round(self.final_value, 2),
            "total_return_pct": round(self.total_return * 100, 2),
            "trades": self.trades,
            "win_rate_pct": round(self.win_rate * 100, 2),
            "sharpe": round(self.sharpe, 3),
            "max_drawdown_pct": round(self.max_dd, 2),
            "var_pct": round(self.var, 2),
        }


def run_backtest(
    prices: pd.Series,
    signals: np.ndarray,
    config: Optional[BacktestConfig] = None,
) -> BacktestResult:
    """
    Backtest a directional signal series on price data.

    signals: array aligned with prices; >0 = long, <0 = short, 0 = flat.
    position_size: fraction of capital deployed per trade.
    """
    if config is None:
        config = BacktestConfig()

    prices = pd.Series(prices).astype(float)
    n = len(prices)
    if len(signals) != n:
        raise ValueError("signals must align with prices")

    capital = config.initial_capital
    position = 0.0  # shares held
    entry_price = 0.0
    equity = np.zeros(n)
    trades = 0
    wins = 0
    prev_signal = 0.0

    for i in range(n):
        price = float(prices.iloc[i])
        signal = float(signals[i])

        # Exit on stop loss
        if position != 0 and config.stop_loss is not None:
            ret = (price - entry_price) / entry_price if position > 0 else (entry_price - price) / entry_price
            if ret <= config.stop_loss:
                # close position
                proceeds = position * price * (1 - config.transaction_cost_bps / 1e4 - config.slippage_bps / 1e4)
                capital = proceeds
                if (position > 0 and price > entry_price) or (position < 0 and price < entry_price):
                    wins += 1
                trades += 1
                position = 0.0
                entry_price = 0.0

        # Enter / flip position when signal changes
        if signal != 0 and signal != prev_signal:
            # close existing
            if position != 0:
                proceeds = position * price * (1 - config.transaction_cost_bps / 1e4 - config.slippage_bps / 1e4)
                pnl = proceeds - position * entry_price
                capital += proceeds
                if pnl > 0:
                    wins += 1
                trades += 1
            # open new
            trade_capital = capital * config.position_size
            shares = (trade_capital / price) * (1 if signal > 0 else -1)
            cost = trade_capital * (config.transaction_cost_bps / 1e4 + config.slippage_bps / 1e4)
            capital -= cost
            position = shares
            entry_price = price

        # mark-to-market equity
        equity[i] = capital + position * price
        prev_signal = signal

    # close final position
    if position != 0:
        price = float(prices.iloc[-1])
        proceeds = position * price * (1 - config.transaction_cost_bps / 1e4 - config.slippage_bps / 1e4)
        capital += proceeds
        if (position > 0 and price > entry_price) or (position < 0 and price < entry_price):
            wins += 1
        trades += 1

    equity_series = pd.Series(equity, index=prices.index)
    returns_series = equity_series.pct_change().fillna(0)

    total_return = equity_series.iloc[-1] / config.initial_capital - 1
    return BacktestResult(
        config=config,
        equity_curve=equity_series,
        returns=returns_series,
        trades=trades,
        win_rate=(wins / trades) if trades else 0.0,
        total_return=total_return,
        sharpe=sharpe_ratio(returns_series),
        max_dd=max_drawdown(returns_series),
        var=value_at_risk(returns_series),
        final_value=float(equity_series.iloc[-1]),
    )
