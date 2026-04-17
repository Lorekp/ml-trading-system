"""
Base strategy interface for all vol-based strategies.

Every strategy:
1. Takes vol predictions + OHLCV → generates signals (direction + size)
2. Applies 1-bar lag (no look-ahead)
3. Evaluates with net-of-cost metrics
4. Backtests with PurgedKFold or chronological split
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging

import numpy as np
import pandas as pd
from scipy.stats import binom

logger = logging.getLogger(__name__)


class BaseVolStrategy(ABC):
    """Abstract base class for all vol-based strategies."""

    name: str = "base"

    def __init__(self, cost_per_trade: float = 0.0015):
        self.cost = cost_per_trade

    @abstractmethod
    def generate_signals(self, ohlcv: pd.DataFrame, predicted_vol: pd.Series) -> pd.DataFrame:
        """
        Generate trading signals.

        CRITICAL: All signals must be lagged 1 bar. Signal at bar t
        uses data up to bar t, but the TRADE happens at bar t+1.

        Args:
            ohlcv: OHLCV DataFrame.
            predicted_vol: Forward vol predictions from vol model.

        Returns:
            DataFrame with columns:
                direction: +1 (long), -1 (short), 0 (no trade)
                size: position size multiplier [0, 1]
                confidence: model confidence [0, 1]
        """
        pass

    def backtest(
        self,
        ohlcv: pd.DataFrame,
        predicted_vol: pd.Series,
        test_pct: float = 0.2,
    ) -> dict:
        """
        Backtest strategy with proper lag and cost accounting.

        Uses chronological train/test split (last test_pct as test).
        """
        signals = self.generate_signals(ohlcv, predicted_vol)
        returns = ohlcv["close"].pct_change()

        # Align: signal at t, return at t+1 (the signal is already lagged internally)
        # Strategy return = signal_direction * signal_size * next_bar_return
        strat_returns = (signals["direction"] * signals["size"] * returns).dropna()

        # Test set only
        split = int(len(strat_returns) * (1 - test_pct))
        test_ret = strat_returns.iloc[split:]
        test_signals = signals.iloc[split:]

        if len(test_ret) < 20:
            return {"error": "Too few test observations", "n_test": len(test_ret)}

        # Cost: charged on every position change
        position_changes = test_signals["direction"].diff().abs().fillna(0) > 0
        costs = position_changes.astype(float) * self.cost
        net_ret = test_ret - costs.reindex(test_ret.index, fill_value=0)

        # Metrics
        n_trades = int(position_changes.sum())
        n_test = len(test_ret)

        # Direction accuracy (only on bars where we trade)
        trading_bars = test_signals["direction"] != 0
        if trading_bars.sum() > 0:
            actual_dir = np.sign(returns.reindex(test_signals.index))
            correct = (test_signals.loc[trading_bars, "direction"] == actual_dir.loc[trading_bars])
            win_rate = correct.mean()
            n_trades_dir = int(trading_bars.sum())
        else:
            win_rate = 0.0
            n_trades_dir = 0

        # Sharpe (annualized for 15m: 4*24*365 = 35040 bars/year)
        bars_per_year = 35040  # 15m default
        gross_sharpe = (test_ret.mean() / test_ret.std() * np.sqrt(bars_per_year)
                        if test_ret.std() > 0 else 0)
        net_sharpe = (net_ret.mean() / net_ret.std() * np.sqrt(bars_per_year)
                      if net_ret.std() > 0 else 0)

        # Max drawdown
        cum = net_ret.cumsum()
        dd = (cum.cummax() - cum)
        max_dd = dd.max() if len(dd) > 0 else 0

        # Win rate significance
        if n_trades_dir > 0:
            k = int(win_rate * n_trades_dir)
            p_val = 1 - binom.cdf(k - 1, n_trades_dir, 0.5) if k > n_trades_dir / 2 else 1.0
        else:
            p_val = 1.0

        # HHI concentration
        positive_trades = test_ret[test_ret > 0]
        if len(positive_trades) > 0:
            weights = positive_trades / positive_trades.sum()
            hhi = (weights ** 2).sum()
        else:
            hhi = 1.0

        return {
            "strategy": self.name,
            "n_test": n_test,
            "n_trades": n_trades,
            "n_trades_dir": n_trades_dir,
            "win_rate": float(win_rate),
            "p_value": float(p_val),
            "gross_sharpe": float(gross_sharpe),
            "net_sharpe": float(net_sharpe),
            "max_drawdown": float(max_dd),
            "total_return": float(net_ret.sum()),
            "mean_return_per_trade": float(net_ret[trading_bars].mean()) if trading_bars.sum() > 0 else 0,
            "hhi": float(hhi),
            "cost_total": float(costs.sum()),
        }
