"""Core interfaces (protocols) for the modular trading engine.

All components code against these protocols, never against concrete
implementations. This enables swapping strategies, data sources,
execution algorithms, and exchanges without touching the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd


# ── Data types ──


@dataclass
class Signal:
    """A trading signal produced by a strategy."""
    asset: str
    symbol: str              # e.g. "BTCUSDT"
    timeframe: str           # e.g. "15m", "4h"
    direction: float         # +1.0 (long) or -1.0 (short)
    probability: float       # [0, 1]
    confidence: float        # abs(prob - 0.5)
    signal_strength: float   # discretized AFML signal
    hold_bars: int           # bars to hold on this TF
    strategy_id: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class TradeResult:
    """Result of an execution attempt."""
    filled: bool = False
    fill_price: float = 0.0
    fill_qty: float = 0.0
    method: str = ""           # "maker", "taker", "blended"
    order_id: str = ""
    exchange_order_id: str = ""
    decision_mid: float = 0.0  # TCA benchmark


@dataclass
class Position:
    """An open or closed position."""
    id: int = 0
    asset: str = ""
    symbol: str = ""
    timeframe: str = ""
    side: str = ""             # "long" or "short"
    direction: float = 1.0
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    size_usd: float = 0.0
    hold_bars: int = 4
    bars_held: int = 0
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    status: str = "open"       # "open", "closed"
    fill_method: str = ""
    order_id: str = ""
    strategy_id: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class PendingOrder:
    """An order placed on the exchange, awaiting fill."""
    id: int = 0
    symbol: str = ""
    exchange_order_id: str = ""
    side: str = ""
    qty: float = 0.0
    price: Optional[float] = None
    order_type: str = ""       # "limit_postonly", "market"
    strategy_id: str = ""
    placed_at: Optional[datetime] = None
    status: str = "pending"    # "pending", "filled", "cancelled", "orphaned"


@dataclass
class PortfolioView:
    """Read-only snapshot of portfolio state, passed to strategies."""
    equity: float
    peak_equity: float
    drawdown: float
    open_positions: list[Position]
    n_positions_by_asset: dict[str, int]
    exposure_by_asset: dict[str, float]
    total_exposure: float


@dataclass
class ReconcileReport:
    """Result of startup reconciliation."""
    orphaned_positions: list[dict]   # on exchange, not in DB
    ghost_positions: list[dict]      # in DB, not on exchange
    stale_orders_cancelled: int
    adopted_positions: int
    warnings: list[str]


# ── Protocols ──


@runtime_checkable
class Strategy(Protocol):
    """Any trading strategy must implement this protocol."""

    @property
    def name(self) -> str: ...

    @property
    def timeframes(self) -> list[str]: ...

    @property
    def tradeable_universe(self) -> list[str]:
        """Assets this strategy can actually trade right now (have models, data, etc.)."""
        ...

    def on_bar(
        self,
        bar_time: datetime,
        market_data: dict[tuple[str, str], pd.DataFrame],  # (tf, asset) -> OHLCV
        portfolio: PortfolioView,
        active_tfs: set[str],
    ) -> list[Signal]:
        """Generate signals for this bar. Called when any of self.timeframes fires."""
        ...

    def check_exit(
        self,
        position: Position,
        market_data: dict[tuple[str, str], pd.DataFrame],
        bar_time: datetime,
    ) -> Optional[str]:
        """Return exit reason if position should close early, else None."""
        ...

    def on_trade(self, asset: str, pnl: float, notional: float) -> None:
        """Called after a position closes. Update internal state."""
        ...

    def get_sizing_mult(self, asset: str) -> float:
        """Return sizing multiplier for this asset (e.g. F1 breaker)."""
        ...

    def compute_size(
        self,
        signal: Signal,
        equity: float,
        drawdown: float,
        vol_mult: float,
    ) -> float:
        """Compute position size in USD."""
        ...

    def preview_size(
        self,
        signal: Signal,
        portfolio: PortfolioView,
    ) -> float:
        """Return the exact sizing this strategy would produce for `signal`,
        given the current portfolio snapshot.

        Used by `SignalAggregator` for cap enforcement. This must produce the
        SAME value that `compute_size` would eventually produce for the same
        (signal, equity, drawdown, vol_mult) — no approximation. The strategy
        is expected to source its own `vol_mult` (from its vol overlay cache
        or equivalent) so the aggregator does not have to know about it.
        """
        ...

    def get_state(self) -> dict:
        """Serialize strategy state for persistence."""
        ...

    def restore_state(self, state: dict) -> None:
        """Restore strategy state from persistence."""
        ...

    async def prepare_bar(
        self, bar_time: datetime, data_provider: DataProvider,
        exchange: ExchangeAdapter,
    ) -> None:
        """Called before each bar cycle. Use for rebalancing, universe updates, etc.

        Optional — default no-op. Strategies that need exchange access for
        asset selection (e.g. S8 regime-based rebalance) override this.
        """
        ...


@runtime_checkable
class DataProvider(Protocol):
    """Any source of market data.

    Implementations: BinanceFuturesData, BybitExchange (also provides data).
    The engine uses this for OHLCV bars, tickers, and asset discovery.
    """

    async def fetch_ohlcv(
        self, symbol: str, interval: str, limit: int = 200
    ) -> pd.DataFrame:
        """Fetch OHLCV bars. Returns DataFrame with OHLCV columns + datetime index."""
        ...

    async def fetch_ticker(self, symbol: str) -> dict[str, float]:
        """Fetch current ticker.

        Returns at minimum {last_price, bid, ask, volume_24h}. Providers that
        can also supply quote-currency (USDT) 24h volume should include
        `quote_volume_24h` — used by CalibratedPaperExecution for slippage
        size gates. Consumers must tolerate its absence.
        """
        ...

    async def fetch_all_tickers(self) -> list[dict]:
        """Fetch tickers for all tradeable symbols.

        Returns list of {symbol, asset, last_price, volume_24h, turnover_24h}.
        Used by asset selectors for universe discovery + ranking.
        """
        ...

    async def fetch_klines(self, symbol: str, interval: str, limit: int = 200) -> list[dict]:
        """Fetch klines as list of dicts (for asset selector BTC vol computation).

        Returns list of {timestamp, open, high, low, close, volume}.
        """
        ...

    async def close(self) -> None: ...


@runtime_checkable
class ExchangeAdapter(Protocol):
    """Exchange operations: orders, positions, account.

    Implementations: BybitExchange, and future exchange adapters.
    The engine uses this for order placement, position tracking, and risk controls.

    To add a new exchange: implement all methods below. Methods that your
    exchange doesn't support should return sensible defaults (e.g. empty dict)
    and log a warning — the engine handles missing capabilities gracefully.
    """

    async def place_limit_order(
        self, symbol: str, side: str, quantity: float, price: float,
        post_only: bool = True, reduce_only: bool = False,
    ) -> dict: ...

    async def place_market_order(
        self, symbol: str, side: str, quantity: float,
        reduce_only: bool = False,
    ) -> dict: ...

    async def cancel_order(self, symbol: str, order_id: str) -> bool: ...

    async def get_order_status(self, symbol: str, order_id: str) -> dict: ...

    async def get_open_positions(self) -> dict[str, dict]: ...

    async def get_open_orders(self) -> list[dict]: ...

    async def get_account_balance(self) -> float: ...

    async def set_leverage(self, symbol: str, leverage: int) -> bool: ...

    async def get_instrument_info(self, symbol: str) -> dict: ...

    async def place_trading_stop(
        self, symbol: str, side: str, stop_loss: float,
    ) -> bool:
        """Attach a stop-loss to an existing open position.

        Args:
            symbol: Trading pair (e.g. "BTCUSDT")
            side: Position side ("Buy" for long SL, "Sell" for short SL)
            stop_loss: Trigger price for the stop-loss

        Returns:
            True if SL was placed successfully.
        """
        ...

    async def fetch_ticker(self, symbol: str) -> dict[str, float]:
        """Fetch current ticker from the execution exchange.

        Used for price divergence checks when data_provider != exchange.
        Returns {last_price, bid, ask, volume_24h}.
        """
        ...

    async def validate_connection(self) -> bool:
        """Pre-flight check: verify API key, permissions, clock sync.

        Returns True if connection is healthy and ready to trade.
        Called by engine on startup before entering the main loop.
        """
        ...

    async def close(self) -> None: ...


@runtime_checkable
class ExecutionAlgo(Protocol):
    """Order execution algorithm (e.g. MkrTkr60, simple market)."""

    async def execute_entry(
        self, symbol: str, side: str, qty: float,
        strategy_id: str = "",
    ) -> Optional[TradeResult]: ...

    async def execute_exit(
        self, symbol: str, side: str, qty: float,
    ) -> Optional[TradeResult]: ...


@runtime_checkable
class ModelBackend(Protocol):
    """Backend for loading and running ML model inference.

    Separates model I/O and prediction from signal processing logic.
    The default implementation (SklearnBackend) handles joblib + predict_proba.
    Implement this protocol for ONNX, PyTorch, API-based models, etc.
    """

    def load_models(self, model_dir: Path) -> int:
        """Load models from directory. Returns count loaded."""
        ...

    def predict(self, asset: str, tf: str, features: np.ndarray) -> float | None:
        """Predict probability for a single feature row.

        Args:
            asset: Asset name (e.g. "BTC")
            tf: Timeframe label (e.g. "15m")
            features: Raw feature values, shape (1, n_features). Backend
                      handles any preprocessing (scaling, normalization).

        Returns:
            Probability [0, 1] for the positive class, or None if prediction
            fails or no model exists for this asset/tf.
        """
        ...

    def has_model(self, asset: str, tf: str) -> bool:
        """Check if a model exists for this asset/tf pair."""
        ...

    def available_assets(self, tf: str) -> list[str]:
        """Return list of assets with loaded models for a given timeframe."""
        ...

    def get_meta(self, asset: str, tf: str) -> dict | None:
        """Return baked training-time metadata for this asset/tf pair, or None
        if the backend does not supply metadata. Expected keys (all optional):
          - conf_threshold: baked 70th pct of |train_proba-0.5|
          - vol_threshold: baked 50th pct of training-window EWM vol
          - trained_at: ISO timestamp
        Live signal generation prefers baked values over defaults so live and
        backtest apply matching filter thresholds.
        """
        ...
