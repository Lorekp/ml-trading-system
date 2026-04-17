"""Cross-strategy signal aggregation and conflict resolution.

Handles:
- Opposite signals for same asset from different strategies → drop both
- Per-strategy position caps
- Per-strategy exposure budgets
- Portfolio-level total exposure cap
- Deterministic ordering by (asset, timeframe) — matches backtest iteration.

Sizing note: the aggregator does NOT estimate exposure on its own. It delegates
to `strategy.preview_size(signal, portfolio)` so the cap math uses the EXACT
notional the strategy would produce downstream. This eliminates the old
`equity × base_sizing_pct × edge` approximation which ignored the strategy's
drawdown-deleverage, vol-overlay, breaker, and correlation-throttle multipliers.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from live.core.interfaces import Signal, PortfolioView

if TYPE_CHECKING:
    from live.core.interfaces import Strategy

logger = logging.getLogger(__name__)


class SignalAggregator:
    """Aggregate and filter signals from multiple strategies."""

    def __init__(
        self,
        max_positions_per_strategy: int = 20,
        max_exposure_pct: float = 0.90,
        max_per_asset_pct: float = 0.06,
        max_signals_per_bar: int | None = None,
    ) -> None:
        self._max_positions = max_positions_per_strategy
        self._max_exposure_pct = max_exposure_pct
        self._max_per_asset_pct = max_per_asset_pct
        self._max_signals_per_bar = max_signals_per_bar

    def aggregate(
        self,
        signals: list[tuple[Strategy, Signal]],
        portfolio: PortfolioView,
    ) -> list[tuple[Strategy, Signal]]:
        """Filter and order signals. Returns list of (strategy, signal) to execute.

        Steps:
            1. Conflict resolution: opposite directions for same asset → drop both
            2. Deterministic sort by (asset, timeframe) — matches backtest
               (experiments/drawdown_improvements.py:148) iteration order
            3. Apply per-strategy, per-asset, and portfolio exposure caps, using
               `strategy.preview_size(sig, portfolio)` for the exact notional
            4. Enforce max_signals_per_bar by keeping the first N after caps
        """
        if not signals:
            return []

        # 1. Conflict resolution: detect opposite directions for same asset
        asset_directions: dict[str, list[tuple[Strategy, Signal]]] = defaultdict(list)
        for strategy, sig in signals:
            asset_directions[sig.asset].append((strategy, sig))

        resolved = []
        for asset, sigs in asset_directions.items():
            directions = set(s.direction for _, s in sigs)
            if len(directions) > 1:
                # Conflicting signals — drop all for this asset
                strat_names = [s.strategy_id for _, s in sigs]
                logger.info("Conflict on %s: strategies %s disagree, dropping all",
                            asset, strat_names)
                continue
            resolved.extend(sigs)

        # 2. Deterministic sort — matches backtest's post-assemble_signals order.
        # Backtest sorts by timestamp then processes rows in concat order
        # (asset, then tf within asset). Within a single live bar all signals
        # share the same bar_time, so we order by (asset, timeframe) directly.
        resolved.sort(key=lambda x: (x[1].asset, x[1].timeframe))

        # 3. Apply caps, using exact strategy sizing
        filtered = []
        strategy_positions: dict[str, int] = defaultdict(int)
        asset_new_exposure: dict[str, float] = {}

        for strategy, sig in resolved:
            # Per-strategy position cap
            existing_positions = sum(
                1 for p in portfolio.open_positions
                if p.strategy_id == sig.strategy_id
            )
            if existing_positions + strategy_positions[sig.strategy_id] >= self._max_positions:
                continue

            # Per-asset exposure (existing + new from this bar)
            existing_asset_exp = portfolio.exposure_by_asset.get(sig.asset, 0)
            new_asset_exp = asset_new_exposure.get(sig.asset, 0)
            if existing_asset_exp + new_asset_exp > portfolio.equity * self._max_per_asset_pct:
                continue

            # Portfolio total exposure
            total_existing = portfolio.total_exposure
            total_new = sum(asset_new_exposure.values())
            if total_existing + total_new > portfolio.equity * self._max_exposure_pct:
                continue

            # Ask the strategy for the ACTUAL notional it would size this to.
            est_size = float(strategy.preview_size(sig, portfolio))
            if est_size <= 0.0:
                # Strategy would reject or zero-size this trade; skip it.
                continue

            filtered.append((strategy, sig))
            strategy_positions[sig.strategy_id] += 1
            asset_new_exposure[sig.asset] = asset_new_exposure.get(sig.asset, 0) + est_size

        # 4. Max signals per bar: keep first N after the deterministic sort.
        if self._max_signals_per_bar is not None and len(filtered) > self._max_signals_per_bar:
            logger.info("MaxSig cap: %d → %d signals", len(filtered), self._max_signals_per_bar)
            filtered = filtered[:self._max_signals_per_bar]

        if len(filtered) < len(resolved):
            logger.info("Aggregator: %d/%d signals passed filters",
                        len(filtered), len(resolved))

        return filtered
