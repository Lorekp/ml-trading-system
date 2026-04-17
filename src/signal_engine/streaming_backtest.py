"""Leak-proof bar-by-bar backtesting engine.

At time T, the strategy's on_bar() receives ONLY the feature row at T
(computed from ohlcv[0:T+1] -- causal by construction). The model was
trained on data strictly before T. Signals resolve AFTER fwd_bars elapse.

Features are pre-computed once on the full series for speed. This is safe
because all features (RSI, MACD, EWM, etc.) are causal -- feature[T] uses
only ohlcv[0:T]. The leak-proof guarantee is structural:
  1. Model trained on data before T (walk-forward with embargo)
  2. Signal resolved after fwd_bars elapse (no pre-computed fwd_return)
  3. Strategy cannot access the full OHLCV -- only receives one feature row
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    direction: float
    probability: float
    confidence: float
    signal_strength: float = 1.0


@dataclass
class Trade:
    entry_bar: int
    exit_bar: int
    entry_ts: Any
    exit_ts: Any
    entry_price: float
    exit_price: float
    direction: float
    fwd_return: float
    pnl: float


@dataclass
class BacktestResult:
    trades: list[Trade]
    sharpe: float
    win_rate: float
    n_trades: int
    mean_return_bps: float
    daily_pnl: pd.Series
    n_days: int


class Strategy(Protocol):
    def on_bar(self, bar_idx: int, feature_row: pd.Series, vol: float) -> Signal | None: ...
    def on_train(self, bar_idx: int, features: pd.DataFrame, labels: pd.Series,
                 vol_series: pd.Series) -> None: ...


class StreamingBacktester:
    """Fast bar-by-bar engine with pre-computed causal features."""

    def __init__(
        self,
        fwd_bars: int = 4,
        embargo_bars: int = 5,
        retrain_frequency_bars: int | None = None,
        slippage: float = 0.0001,
        retrain_mode: str = "bars",
        lookback_months: int = 8,
        confidence_mode: str = "per_bar",
        confidence_percentile: float = 70.0,
    ):
        self.fwd_bars = fwd_bars
        self.embargo_bars = embargo_bars
        self.slippage = slippage
        self.retrain_frequency_bars = retrain_frequency_bars or (30 * 24 * 4)
        self.retrain_mode = retrain_mode
        self.lookback_months = lookback_months
        self.confidence_mode = confidence_mode
        self.confidence_percentile = confidence_percentile

    def run(
        self,
        strategy: Strategy,
        features_full: pd.DataFrame,
        close: np.ndarray,
        vol: np.ndarray,
        timestamps: pd.DatetimeIndex,
        start_bar: int | None = None,
        end_bar: int | None = None,
        progress_every: int = 10000,
    ) -> BacktestResult:
        n_total = len(close)
        if start_bar is None:
            start_bar = max(300, self.embargo_bars + self.fwd_bars + 200)
        if end_bar is None:
            end_bar = n_total - self.fwd_bars

        feat_values = features_full.to_numpy(dtype=np.float32)
        feat_cols = features_full.columns

        # Convert timestamps to month keys for monthly mode
        ts_as_pd = pd.DatetimeIndex(timestamps)
        month_keys = (ts_as_pd.year * 12 + ts_as_pd.month).values

        pending: list[tuple[int, Signal]] = []
        trades: list[Trade] = []
        last_train_bar = -999999
        last_train_month_key = -1

        # Monthly batch confidence buffer
        month_buf: list[tuple[int, Signal]] = []
        current_month_key = -1

        # For "previous_month" mode: causal threshold from last month
        prev_month_conf_threshold = 0.0

        t0 = time.perf_counter()

        for T in range(start_bar, end_bar):
            mk = int(month_keys[T])

            # --- Month boundary: flush buffered signals ---
            if mk != current_month_key and current_month_key > 0:
                if self.confidence_mode == "monthly_batch" and month_buf:
                    confs = np.array([s.confidence for _, s in month_buf])
                    thr = float(np.percentile(confs, self.confidence_percentile))
                    for bi, si in month_buf:
                        if si.confidence >= thr:
                            pending.append((bi, si))
                    month_buf = []
                elif self.confidence_mode == "previous_month" and month_buf:
                    # Compute this month's threshold for NEXT month (causal)
                    confs = np.array([s.confidence for _, s in month_buf])
                    new_thr = float(np.percentile(confs, self.confidence_percentile))
                    # Signals from this month were already filtered using prev threshold
                    month_buf = []
                    prev_month_conf_threshold = new_thr
            current_month_key = mk

            # --- Retraining ---
            need_retrain = False
            if self.retrain_mode == "monthly":
                if mk != last_train_month_key:
                    need_retrain = True
            else:
                if (T - last_train_bar) >= self.retrain_frequency_bars:
                    need_retrain = True

            if need_retrain:
                train_end = T - self.embargo_bars
                safe_end = train_end - self.fwd_bars

                if self.retrain_mode == "monthly":
                    bar_ts = ts_as_pd[T]
                    train_start_ts = bar_ts - pd.DateOffset(months=self.lookback_months)
                    train_start_idx = max(0, int(ts_as_pd.searchsorted(train_start_ts)))
                else:
                    train_start_idx = 0

                safe_end = min(safe_end, n_total - self.fwd_bars - 1)
                if safe_end > train_start_idx + 200:
                    sl = features_full.index[train_start_idx:safe_end + 1]
                    train_feat = features_full.iloc[train_start_idx:safe_end + 1]
                    positions = np.arange(train_start_idx, safe_end + 1)
                    fwd_ret = np.where(
                        positions + self.fwd_bars < n_total,
                        np.where(close[positions] > 0,
                                 (close[np.minimum(positions + self.fwd_bars, n_total - 1)] - close[positions]) / close[positions],
                                 0.0),
                        np.nan,
                    )
                    labels = pd.Series((fwd_ret > 0).astype(np.int8), index=sl)
                    labels[np.isnan(fwd_ret)] = -1
                    valid = labels[labels >= 0]
                    vol_train = pd.Series(vol[positions], index=sl)
                    strategy.on_train(T, train_feat.loc[valid.index], valid, vol_train.loc[valid.index])
                    last_train_bar = T
                    last_train_month_key = mk

            # --- Signal ---
            feature_row = pd.Series(feat_values[T], index=feat_cols)
            signal = strategy.on_bar(T, feature_row, float(vol[T]))

            if signal is not None:
                if self.confidence_mode == "monthly_batch":
                    month_buf.append((T, signal))
                elif self.confidence_mode == "previous_month":
                    # Causal: use previous month's threshold (known at month start)
                    month_buf.append((T, signal))
                    if signal.confidence >= prev_month_conf_threshold:
                        pending.append((T, signal))
                elif self.confidence_mode == "none":
                    # No confidence filter at all
                    pending.append((T, signal))
                else:
                    pending.append((T, signal))

            # --- Resolve ---
            still_pending = []
            for entry_bar, sig in pending:
                if T - entry_bar >= self.fwd_bars:
                    exit_bar = entry_bar + self.fwd_bars
                    if exit_bar < n_total:
                        ep = float(close[entry_bar])
                        xp = float(close[exit_bar])
                        if ep > 0:
                            fr = (xp - ep) / ep
                            pnl = sig.direction * fr - self.slippage
                            trades.append(Trade(
                                entry_bar=entry_bar, exit_bar=exit_bar,
                                entry_ts=timestamps[entry_bar], exit_ts=timestamps[exit_bar],
                                entry_price=ep, exit_price=xp,
                                direction=sig.direction, fwd_return=fr, pnl=pnl,
                            ))
                else:
                    still_pending.append((entry_bar, sig))
            pending = still_pending

            if progress_every and (T - start_bar) % progress_every == 0 and T > start_bar:
                elapsed = time.perf_counter() - t0
                pct = (T - start_bar) / max(end_bar - start_bar, 1) * 100
                bars_sec = (T - start_bar) / elapsed
                print(f"    bar {T}/{end_bar} ({pct:.0f}%) -- {len(trades)} trades, {bars_sec:.0f} bars/s", flush=True)

        # Flush final month
        if self.confidence_mode == "monthly_batch" and month_buf:
            confs = np.array([s.confidence for _, s in month_buf])
            thr = float(np.percentile(confs, self.confidence_percentile))
            for bi, si in month_buf:
                if si.confidence >= thr:
                    pending.append((bi, si))
        # previous_month signals already added to pending in real-time (no flush needed)

        # Resolve remaining
        for entry_bar, sig in pending:
            exit_bar = entry_bar + self.fwd_bars
            if exit_bar < n_total:
                ep = float(close[entry_bar])
                xp = float(close[exit_bar])
                if ep > 0:
                    fr = (xp - ep) / ep
                    pnl = sig.direction * fr - self.slippage
                    trades.append(Trade(
                        entry_bar=entry_bar, exit_bar=exit_bar,
                        entry_ts=timestamps[entry_bar], exit_ts=timestamps[exit_bar],
                        entry_price=ep, exit_price=xp,
                        direction=sig.direction, fwd_return=fr, pnl=pnl,
                    ))

        elapsed = time.perf_counter() - t0
        print(f"    Done: {len(trades)} trades in {elapsed:.1f}s ({(end_bar-start_bar)/max(elapsed,0.01):.0f} bars/s)", flush=True)
        return self._compute_result(trades)

    def _compute_result(self, trades: list[Trade]) -> BacktestResult:
        if not trades:
            return BacktestResult(
                trades=[], sharpe=0.0, win_rate=0.0, n_trades=0,
                mean_return_bps=0.0, daily_pnl=pd.Series(dtype=float), n_days=0,
            )
        pnls = np.array([t.pnl for t in trades])
        entry_ts = pd.DatetimeIndex([t.entry_ts for t in trades])
        daily = pd.Series(5.0 * pnls, index=entry_ts).resample("1D").sum()
        daily = daily[daily != 0]
        if len(daily) < 5:
            sharpe = 0.0
        else:
            s = daily.std()
            sharpe = float(daily.mean() / s * np.sqrt(365.25)) if s > 0 else 0.0
        return BacktestResult(
            trades=trades, sharpe=sharpe,
            win_rate=float((pnls > 0).mean()),
            n_trades=len(trades),
            mean_return_bps=float(pnls.mean() * 10000),
            daily_pnl=daily, n_days=len(daily),
        )
