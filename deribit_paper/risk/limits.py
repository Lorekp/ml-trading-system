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
- ``max_trades_per_hour`` (H15) — burst-rate cap on signal firing.
  A bug in the signal pipeline could fire dozens of trades in minutes;
  this cap is the last line of defence.
- ``max_gross_notional_usd`` (H15) — hard cap on total open USD notional.
  Independent of position-count + per-trade caps; defends against the
  edge case where N open trades each within per-trade limit still
  exceed the operator's intended total exposure.

The :class:`RiskLimits` object is stateless except for the config
thresholds and a rolling deque of recent trade timestamps (for
the per-hour cap); it queries the :class:`PositionTracker` and
:class:`EquityTracker` for current state at each check.
"""

from __future__ import annotations

import logging
from collections import deque
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
        # H15: new caps
        max_trades_per_hour: int = 0,    # 0 = disabled
        max_gross_notional_usd: float = 0.0,   # 0 = disabled
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
        # H15: per-hour signal burst cap. A bug in the predictor or
        # signal-stack could fire dozens of trades within minutes; this
        # cap is the last line of defence between a runaway signal and
        # the venue. Disabled by default (0) so existing tests stay
        # green; engine populates from cfg.
        self.max_trades_per_hour = int(max_trades_per_hour)
        self._recent_trade_timestamps: deque[pd.Timestamp] = deque(maxlen=512)
        # H15: hard cap on total open USD notional. Independent of the
        # position-count cap (max_open_positions) — N small positions
        # could still aggregate above the operator's intended exposure.
        self.max_gross_notional_usd = float(max_gross_notional_usd)

    def check_signal(
        self,
        signal: Signal,
        *,
        proposed_n_contracts_btc: float,
        now: pd.Timestamp | None = None,
        bar_index_price: float | None = None,
    ) -> RiskCheck:
        """Validate a signal + proposed trade size against all limits.

        ``bar_index_price`` (R-9, parity-r3 cycle 3 — 2026-05-08): the
        current bar's spot. If provided, used for the margin envelope's
        BTC→USD conversion; otherwise falls back to
        ``signal.straddle.index_price`` (the snapshot taken at signal
        emit). Pre-fix the gate always used the snapshot, which can lag
        the bar by tens of seconds during high-volatility ticks and
        under-/over-state the gate by the spot-drift × notional. The
        backtester's sizing layer reads the bar's spot directly
        (``vol_arb/sizing.py:31-55``); this kwarg restores parity.
        """
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
        # D10 (parity 2026-05-07): ``max_single_trade_btc <= 0`` disables
        # the per-trade size cap (mirrors ``BacktestConfig.max_position_btc=0.0``
        # convention). Default 2.0 still applies under non-V2 presets;
        # the V2 preset sets this to 0.0 because dynamic_btc_cap is the
        # V2 backtester's only ceiling.
        if (
            self.max_single_trade_btc > 0.0
            and proposed_n_contracts_btc > self.max_single_trade_btc
        ):
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

        # 4. Index price sanity (uses signal snapshot — staleness here
        # would let a degenerate spot sneak through; the snapshot's
        # ATM strike pairing was anchored to it, so it is the correct
        # check for "is the trade idea well-formed").
        spot_signal = signal.straddle.index_price
        if spot_signal < self.min_index_price:
            return RiskCheck(
                False,
                f"index_price {spot_signal:.2f} < min {self.min_index_price:.2f}",
            )

        # 5. Margin utilisation hard cap
        # R-9 (parity-r3 cycle 3 — 2026-05-08): use the live bar spot
        # for BTC→USD when available; offline ``vol_arb/sizing.py``
        # uses the bar's spot, not a stale snapshot. Snapshot fallback
        # preserves test-stub callers and the no-bar legacy contract.
        spot = bar_index_price if (
            bar_index_price is not None
            and bar_index_price >= self.min_index_price
        ) else spot_signal
        # Approximate margin used: im_fraction × (current open notional + this trade's notional)
        current_notional_btc = self._positions.total_notional_btc()
        total_notional_btc = current_notional_btc + proposed_n_contracts_btc
        total_notional_usd = total_notional_btc * spot
        margin_used_usd = self.im_fraction * total_notional_usd
        equity_usd = self._equity.current_equity
        if equity_usd <= 0:
            return RiskCheck(False, f"equity {equity_usd:.2f} is non-positive")
        margin_util = margin_used_usd / equity_usd

        # R-4 (parity-r3 cycle 3 — 2026-05-08): minimum-lot exemption.
        # Pre-fix this gate cleared only when ``n_open == 0``, which
        # silently blocked DD-recovery: an engine that crashed with one
        # min-lot trade open and restarted into a depleted-equity state
        # could not enter a second min-lot trade because the gate now
        # saw ``n_open == 1`` and required margin_util ≤ hard_cap. The
        # forward-looking semantic is **"the absolute floor lot is
        # always exempt regardless of book size"** — Deribit does not
        # expose a smaller tradeable size, so the operator's only way
        # to scale OUT of a degraded equity state is to keep entering
        # min-lot trades. The gate's protective intent (stop runaway
        # margin growth) is preserved because the exemption only
        # applies when the proposed size IS the absolute floor; any
        # larger size still passes through the margin gate normally.
        is_floor_lot = (
            abs(proposed_n_contracts_btc - self.min_single_trade_btc) < 1e-9
        )
        if is_floor_lot:
            pass  # skip margin util check — can't trade smaller
        elif margin_util > self.max_margin_util_hard:
            return RiskCheck(
                False,
                f"margin_util {margin_util:.3f} > hard cap {self.max_margin_util_hard:.3f}",
            )

        # H15: gross notional cap (total open USD across all positions
        # plus the proposed trade). Only checked when configured > 0.
        if self.max_gross_notional_usd > 0:
            if total_notional_usd > self.max_gross_notional_usd:
                return RiskCheck(
                    False,
                    f"gross_notional ${total_notional_usd:,.0f} > "
                    f"hard cap ${self.max_gross_notional_usd:,.0f}",
                )

        # H15: per-hour rate cap. Counts trade-fire events recorded via
        # ``record_trade_fired`` within the last 60 minutes. Only
        # checked when configured > 0.
        if self.max_trades_per_hour > 0:
            cutoff = now - pd.Timedelta(hours=1)
            recent = sum(
                1 for ts in self._recent_trade_timestamps if ts >= cutoff
            )
            if recent >= self.max_trades_per_hour:
                return RiskCheck(
                    False,
                    f"max_trades_per_hour: {recent} fired in past 1h "
                    f">= cap {self.max_trades_per_hour} — burst-rate "
                    f"defence engaged",
                )

        return RiskCheck(True, "all checks passed")

    def record_trade_fired(self, at: pd.Timestamp | None = None) -> None:
        """H15: record a trade-fire event for the per-hour cap.

        Engine calls this after a signal passes the risk gate AND has
        actually been scheduled for execution. Rejected signals do NOT
        count against the cap (otherwise a buggy predictor that always
        fires would silently hit the cap and stay there).
        """
        if self.max_trades_per_hour <= 0:
            return
        ts = at or pd.Timestamp.now(tz="UTC")
        self._recent_trade_timestamps.append(ts)
