"""Position sizing and exposure management.

Extracted from simulate_with_breaker():320-397 in circuit_breaker_v2_experiment.py.
Same sizing logic: probability edge, regime multiplier, vol overlay,
deleveraging, per-asset/total exposure caps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from live.config import BASELINE_WR, PortfolioConfig

logger = logging.getLogger(__name__)


def _edge_from_probability(prob: float, sig: float) -> float:
    """Same as circuit_breaker_v2:85-86."""
    if np.isfinite(prob):
        return float(np.clip(abs(2.0 * prob - 1.0), 0.0, 1.0))
    return float(np.clip(abs(sig), 0.0, 1.0))


@dataclass
class Position:
    """An open position tracked by the portfolio manager."""
    id: int = 0
    asset: str = ""
    symbol: str = ""
    timeframe: str = ""
    side: str = ""          # "long" or "short"
    direction: float = 1.0
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    size_usd: float = 0.0
    hold_bars: int = 4
    bars_held: int = 0
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    status: str = "open"    # "open", "closed"
    fill_method: str = ""   # "maker", "taker", "blended"
    order_id: str = ""


class PortfolioManager:
    """Position sizing and exposure management.

    Same logic as simulate_with_breaker() lines 320-397.
    """

    def __init__(self, config: PortfolioConfig, starting_capital: float) -> None:
        self._config = config
        self.equity = starting_capital
        self.peak_equity = starting_capital
        self._positions: list[Position] = []

    @property
    def open_positions(self) -> list[Position]:
        return [p for p in self._positions if p.status == "open"]

    @property
    def halted(self) -> bool:
        dd = self.drawdown
        return dd >= self._config.max_dd

    @property
    def drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity

    def compute_size(
        self,
        asset: str,
        probability: float,
        signal_strength: float,
        regime_mult: float,
        vol_mult: float,
        breaker_mult: float,
        strategy: str = "default",
    ) -> float:
        """Compute position size in USD. Same as sim lines 332-388.

        Returns:
            Size in USD, or 0.0 if halted/capped.
        """
        c = self._config

        # Drawdown check
        dd = self.drawdown
        if dd >= c.max_dd:
            return 0.0

        # Deleveraging (line 331)
        if dd > c.delever_start:
            size_mult = max(
                1.0 - (dd - c.delever_start) / (c.max_dd - c.delever_start),
                0.0,
            )
        else:
            size_mult = 1.0

        # Probability edge sizing (lines 333-337)
        sig_abs = _edge_from_probability(probability, signal_strength)
        sig_abs *= float(np.clip(regime_mult, c.regime_exposure_floor, 1.0))

        raw_size = self.equity * c.base_sizing_pct * sig_abs * size_mult * vol_mult
        raw_size = min(raw_size, self.equity * c.base_sizing_pct * c.max_signal_mult * size_mult)

        # F1 breaker multiplier (line 378)
        raw_size *= breaker_mult

        # Exposure limits (lines 382-387)
        asset_exposure = sum(p.size_usd for p in self.open_positions if p.asset == asset)
        max_asset = self.equity * c.max_per_asset
        if asset_exposure + raw_size > max_asset:
            raw_size = max(max_asset - asset_exposure, 0.0)

        total_exposure = sum(p.size_usd for p in self.open_positions)
        max_total = self.equity * c.max_exposure
        if total_exposure + raw_size > max_total:
            raw_size = max(max_total - total_exposure, 0.0)

        if raw_size < 1.0:
            return 0.0

        return raw_size

    def open_position(self, position: Position) -> None:
        """Record a new open position."""
        self._positions.append(position)
        logger.info("Opened %s %s %.0f USD @ %.4f (%s, h=%d)",
                     position.side, position.asset, position.size_usd,
                     position.entry_price, position.timeframe, position.hold_bars)

    def close_position(self, position: Position, exit_price: float,
                       exit_time: datetime) -> float:
        """Close a position and update equity. Returns PnL."""
        if position.direction > 0:
            pnl = position.size_usd * (exit_price - position.entry_price) / position.entry_price
        else:
            pnl = position.size_usd * (position.entry_price - exit_price) / position.entry_price

        position.exit_price = exit_price
        position.exit_time = exit_time
        position.pnl = pnl
        position.status = "closed"

        self.equity += pnl
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        logger.info("Closed %s %s PnL=$%.2f (equity=$%.0f, dd=%.1f%%)",
                     position.side, position.asset, pnl,
                     self.equity, self.drawdown * 100)
        return pnl

    def compute_vol_mult(self, asset: str, vol_series: dict) -> float:
        """Vol overlay — same as apply_vol() from circuit_breaker_v2:174-187.

        Scales sizing by 0.025 / current_vol, clipped to [0.3, 1.5].
        """
        vs = vol_series.get(asset)
        if vs is None or len(vs) == 0:
            return 1.0

        v = vs.iloc[-1]
        if not np.isfinite(v) or v <= 0:
            return 1.0

        return float(np.clip(0.025 / v, 0.3, 1.5))

    def restore_positions(self, positions: list[Position]) -> None:
        """Restore open positions from DB on restart."""
        self._positions = positions
        logger.info("Restored %d positions (%d open)",
                     len(positions), len(self.open_positions))
