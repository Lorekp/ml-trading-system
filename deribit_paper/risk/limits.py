"""Pre-trade risk checks.

Every signal passes through :meth:`RiskLimits.check_signal` before
execution.  Rejected signals are logged and turned into no-ops (the
engine does NOT abort the bar loop — it continues to the next bar).

Checks:

- ``max_open_positions`` — hard cap on simultaneous open straddles
- ``max_margin_util_hard`` — total margin used / equity must stay under this
- ``max_single_trade_btc`` — single-trade size cap
- ``dte_band`` — no 0DTE, no super-long-dated (default 1-14 days)
- ``min_index_price`` — sanity check (refuse to trade if spot looks bogus)

The :class:`RiskLimits` object is stateless except for the config
thresholds; it queries the :class:`PositionTracker` and
:class:`EquityTracker` for current state at each check.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from deribit_paper.position.equity import EquityTracker
from deribit_paper.position.tracker import PositionTracker
from deribit_paper.signal.predictor import Signal

logger = logging.getLogger(__name__)


@dataclass
class RiskCheck:
    """Result of :meth:`RiskLimits.check_signal`."""
    passed: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.passed


class RiskLimits:
    """Pre-trade risk filter.

    Injected into the engine at startup with config thresholds, then
    queried on every signal before execution.
    """

    def __init__(
        self,
        *,
        position_tracker: PositionTracker,
        equity_tracker: EquityTracker,
        max_open_positions: int = 5,
        max_margin_util_hard: float = 0.20,
        max_single_trade_btc: float = 2.0,
        min_single_trade_btc: float = 0.1,
        dte_min_days: float = 1.0,
        dte_max_days: float = 14.0,
        min_index_price: float = 1_000.0,
        im_fraction: float = 0.20,
    ):
        self._positions = position_tracker
        self._equity = equity_tracker
        self.max_open_positions = max_open_positions
        self.max_margin_util_hard = max_margin_util_hard
        self.max_single_trade_btc = max_single_trade_btc
        self.min_single_trade_btc = min_single_trade_btc
        self.dte_min_days = dte_min_days
        self.dte_max_days = dte_max_days
        self.min_index_price = min_index_price
        self.im_fraction = im_fraction

    def check_signal(
        self,
        signal: Signal,
        *,
        proposed_n_contracts_btc: float,
        now: pd.Timestamp | None = None,
    ) -> RiskCheck:
        """Validate a signal + proposed trade size against all limits."""
        if not signal.is_actionable:
            return RiskCheck(False, "signal not actionable")

        if signal.straddle is None:
            return RiskCheck(False, "signal has no straddle")

        # 1. Position count cap
        n_open = self._positions.n_open
        if n_open >= self.max_open_positions:
            return RiskCheck(
                False,
                f"max_open_positions: {n_open} >= {self.max_open_positions}",
            )

        # 2. Trade size bounds
        if proposed_n_contracts_btc < self.min_single_trade_btc:
            return RiskCheck(
                False,
                f"trade_size {proposed_n_contracts_btc:.3f} < min {self.min_single_trade_btc:.3f}",
            )
        if proposed_n_contracts_btc > self.max_single_trade_btc:
            return RiskCheck(
                False,
                f"trade_size {proposed_n_contracts_btc:.3f} > max {self.max_single_trade_btc:.3f}",
            )

        # 3. DTE band
        now = now or pd.Timestamp.now(tz="UTC")
        dte_days = (pd.Timestamp(signal.straddle.expiry_dt) - now).total_seconds() / 86_400.0
        if dte_days < self.dte_min_days:
            return RiskCheck(
                False,
                f"dte {dte_days:.2f} < min {self.dte_min_days:.2f}",
            )
        if dte_days > self.dte_max_days:
            return RiskCheck(
                False,
                f"dte {dte_days:.2f} > max {self.dte_max_days:.2f}",
            )

        # 4. Index price sanity
        spot = signal.straddle.index_price
        if spot < self.min_index_price:
            return RiskCheck(
                False,
                f"index_price {spot:.2f} < min {self.min_index_price:.2f}",
            )

        # 5. Margin utilisation hard cap
        # Approximate margin used: im_fraction × (current open notional + this trade's notional)
        current_notional_btc = self._positions.total_notional_btc()
        total_notional_btc = current_notional_btc + proposed_n_contracts_btc
        # Convert notional to USD via the signal's spot
        total_notional_usd = total_notional_btc * spot
        margin_used_usd = self.im_fraction * total_notional_usd
        equity_usd = self._equity.current_equity
        if equity_usd <= 0:
            return RiskCheck(False, f"equity {equity_usd:.2f} is non-positive")
        margin_util = margin_used_usd / equity_usd

        # Minimum lot exemption: at small equity the sizing formula
        # can't produce ≥0.1 BTC, so the floor kicks in.  The margin
        # util will be very high but there's no smaller trade on Deribit.
        # Allow exactly ONE minimum-size trade as the first position.
        is_minimum_lot = abs(proposed_n_contracts_btc - self.min_single_trade_btc) < 0.01
        if is_minimum_lot and n_open == 0:
            pass  # skip margin util check — can't trade smaller
        elif margin_util > self.max_margin_util_hard:
            return RiskCheck(
                False,
                f"margin_util {margin_util:.3f} > hard cap {self.max_margin_util_hard:.3f}",
            )

        return RiskCheck(True, "all checks passed")
