"""S8+Sym1.5 strategy — wraps existing signal_generator + breaker + selector.

Implements the Strategy protocol. Zero new logic — just wires existing
components behind the standard interface.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from live.core.interfaces import Signal, Position, PortfolioView
from live.signal_generator import SignalGenerator
from live.breaker import F1SymmetricBreaker
from live.asset_selector import S8AssetSelector
from live.model_manager import ModelManager
from live.config import BASELINE_WR, TF_MINUTES

logger = logging.getLogger(__name__)


def _edge_from_probability(prob: float, sig: float) -> float:
    """Same as circuit_breaker_v2:85-86."""
    if np.isfinite(prob):
        return float(np.clip(abs(2.0 * prob - 1.0), 0.0, 1.0))
    return float(np.clip(abs(sig), 0.0, 1.0))


class S8Sym15Strategy:
    """S8 regime-adaptive selection + Sym1.5 F1 breaker.

    Implements the Strategy protocol by wrapping:
    - SignalGenerator (per-asset per-TF walk-forward models)
    - F1SymmetricBreaker (pnl_ratio_symmetric)
    - S8AssetSelector (core9 + regime-adaptive expansion)
    """

    def __init__(self, strategy_id: str = "s8_sym15", config: dict = None):
        config = config or {}
        self._name = strategy_id
        self._timeframes = config.get("timeframes", ["15m", "1h", "3h", "4h", "24h"])

        # Hold map from Full5 config
        hold_list = config.get("holds", [2, 4, 16, 6, 12])
        self._hold_map = dict(zip(self._timeframes, hold_list))

        # Signal generator (configurable features, thresholds, regime filter)
        model_dir = config.get("model_dir", "data/live/models")
        sg_cfg = config.get("signal_generator", {})
        self._signal_gen = SignalGenerator(
            model_dir=model_dir,
            conf_threshold=sg_cfg.get("conf_threshold", 0.08),
            regime_ema_span=sg_cfg.get("regime_ema_span", 20),
            feature_config_overrides=sg_cfg.get("feature_config", {}),
            vol_filter_enabled=sg_cfg.get("vol_filter_enabled", True),
            require_metadata=sg_cfg.get("require_metadata", False),
        )
        self._n_models = self._signal_gen.load_models()

        # F1 symmetric breaker
        breaker_cfg = config.get("breaker", {})
        self._breaker = F1SymmetricBreaker(
            window=breaker_cfg.get("window", 10),
            threshold=breaker_cfg.get("threshold", -0.01),
            floor=breaker_cfg.get("floor", 0.10),
            up_threshold=breaker_cfg.get("up_threshold", 0.01),
            cap=breaker_cfg.get("cap", 1.5),
        )

        # S8 asset selector (uses prep cache for exact backtest parity)
        s8_cfg = config.get("s8", {})
        self._selector = S8AssetSelector(
            core=set(s8_cfg.get("core", ["BTC","ETH","XRP","ADA","LINK","LTC","BNB","BCH","ETC"])),
            max_total=s8_cfg.get("max_total", 30),
            btc_vol_window=s8_cfg.get("btc_vol_window", 20),
            prep_cache_path=s8_cfg.get("prep_cache_path", ""),
        )

        # Portfolio sizing params
        port = config.get("portfolio", {})
        self._base_sizing_pct = port.get("base_sizing_pct", 0.045)
        self._max_signal_mult = port.get("max_signal_mult", 2.0)
        self._max_dd = port.get("max_dd", 0.10)
        self._delever_start = port.get("delever_start", 0.05)
        self._max_per_asset = port.get("max_per_asset", 0.06)
        self._max_exposure = port.get("max_exposure", 0.90)
        self._regime_floor = port.get("regime_exposure_floor", 0.50)

        # Quote currency for symbol formatting
        self._quote_currency = config.get("quote_currency", "USDT")

        # Vol overlay params — inverse-vol sizing: clip(target/vol, floor, cap).
        # floor default lowered 0.3 -> 0.10 (sizing-fix 2026-05-18): 0.3 was
        # too high for the live universe's ~27x vol dispersion.
        vol_cfg = config.get("vol_overlay", {})
        self._vol_target = vol_cfg.get("target_vol", 0.025)
        self._vol_floor = vol_cfg.get("floor", 0.10)
        self._vol_cap = vol_cfg.get("cap", 2.0)

        # Correlation throttle (CT90: cut sizing when >90% signals agree)
        ct_cfg = config.get("correlation_throttle", {})
        self._ct_threshold = ct_cfg.get("threshold", None)
        self._ct_direction = ct_cfg.get("direction", "both")
        self._ct_scale = ct_cfg.get("scale", 0.5)
        self._ct_window = ct_cfg.get("window", 20)
        self._recent_directions: deque[float] = deque(maxlen=self._ct_window)

        # Vol series cache
        self._vol_series: dict[str, pd.Series] = {}
        self._first_bar = True  # force rebalance on first bar after startup

        # Sizing guards — backtest parity with drawdown_improvements.py:462-480.
        # participation_rate=0 disables the gates entirely (default, backward-compat).
        guards_cfg = config.get("sizing_guards", {})
        self._participation_rate = float(guards_cfg.get("participation_rate", 0.0))
        self._guard_pct = float(guards_cfg.get("guard_pct", 0.50))
        cal_path = guards_cfg.get("calibration_path")
        if self._participation_rate > 0 and cal_path:
            from pathlib import Path as _Path
            from signal_engine.slippage import load_per_symbol_calibration
            self._slip_cal = load_per_symbol_calibration(_Path(cal_path))
        else:
            self._slip_cal = {}
        self._vip_tier = (
            str(guards_cfg.get("vip_tier_name", "VIP0")),
            float(guards_cfg.get("vip_maker_bps", 2.0)),
            float(guards_cfg.get("vip_taker_bps", 5.5)),
        )
        # asset -> daily USD volume, populated by _refresh_vol_cache in prepare_bar
        # and read synchronously by compute_size. Silent no-op when empty.
        self._vol_cache_usd: dict[str, float] = {}

        # Background model training manager
        self._model_mgr = ModelManager(model_dir=model_dir)

        logger.info("S8Sym15 initialized: %d models, %d timeframes, ct=%s",
                     self._n_models, len(self._timeframes),
                     f"{self._ct_threshold}" if self._ct_threshold else "off")

    @property
    def name(self) -> str:
        return self._name

    @property
    def timeframes(self) -> list[str]:
        return self._timeframes

    @property
    def tradeable_universe(self) -> list[str]:
        """Assets we can actually trade: selected by S8 AND have models."""
        selected = set(self._selector.universe)
        model_ready = set()
        for tf in self._timeframes:
            model_ready.update(self._signal_gen.available_assets(tf))
        return sorted(selected & model_ready)

    async def prepare_bar(self, bar_time, data_provider, exchange) -> None:
        """Trigger S8 rebalance via exchange (for volume data + BTC vol regime).

        Called by engine before each bar cycle. Forces rebalance on first bar
        after startup (handles mid-month cold start). Otherwise defers to
        S8AssetSelector's monthly schedule.
        """
        # Check if background training finished -> reload models
        prev_models = self._n_models
        if self._model_mgr.tick():
            self._n_models = self._signal_gen.load_models()
            new_count = self._n_models - prev_models
            logger.info("prepare_bar: models reloaded: %d total (%+d new)", self._n_models, new_count)
        elif self._model_mgr.is_training:
            logger.debug("prepare_bar: background download+train in progress (pid=%s)",
                         getattr(self._model_mgr, '_proc', {}) and
                         getattr(self._model_mgr._proc, 'pid', '?'))

        force = self._first_bar
        if force:
            self._first_bar = False
            logger.info("prepare_bar: first bar after startup, forcing rebalance")

        # Build available pool: all assets with models at any TF
        available = set()
        for tf in self._timeframes:
            available.update(self._signal_gen.available_assets(tf))
        logger.debug("prepare_bar: model pool: %d assets across %d TFs", len(available), len(self._timeframes))

        rebalanced = await self._selector.maybe_rebalance(
            bar_time, data_provider, sorted(available), force=force)

        if rebalanced:
            selected = set(self._selector.universe)
            tradeable = selected & available
            gap = sorted(selected - available)

            logger.info("prepare_bar: rebalance done -> %d selected, %d tradeable%s",
                        len(selected), len(tradeable),
                        f", {len(gap)} need download+train: {gap}" if gap else "")

            if not self._model_mgr.is_training:
                if bar_time.day == 1:
                    # Monthly retrain: download fresh data + retrain ALL models.
                    # Matches backtest walk-forward behavior (monthly model refresh).
                    all_assets = sorted(selected | available)
                    logger.info("prepare_bar: monthly retrain — queuing %d assets", len(all_assets))
                    self._model_mgr.queue_training(all_assets, retrain=True)
                elif gap:
                    # Normal gap training (forced rebalance or mid-month)
                    logger.info("prepare_bar: queuing %d gap assets: %s", len(gap), gap)
                    self._model_mgr.queue_training(gap)
        else:
            logger.debug("prepare_bar: no rebalance this bar")

        # Refresh daily-volume cache for the current universe — used by the
        # sizing gates in compute_size. Cheap (one ticker call per asset,
        # throttled by caller's ExchangeAdapter rate limiter).
        if self._participation_rate > 0:
            await self._refresh_vol_cache(data_provider)

    async def _refresh_vol_cache(self, data_provider) -> None:
        """Fetch quote_volume_24h for each asset in the current universe.

        Populates `self._vol_cache_usd`. Silently tolerates fetch errors so a
        single bad symbol does not halt sizing for the rest of the book.
        """
        assets = self.tradeable_universe
        for asset in assets:
            symbol = f"{asset}{self._quote_currency}"
            try:
                ticker = await data_provider.fetch_ticker(symbol)
            except Exception as e:
                logger.debug("vol_cache: fetch_ticker(%s) failed: %s", symbol, e)
                continue
            qv = float(ticker.get("quote_volume_24h", 0.0) or 0.0)
            if qv <= 0.0:
                base_vol = float(ticker.get("volume_24h", 0.0) or 0.0)
                mid = float(ticker.get("last_price", 0.0) or 0.0)
                qv = base_vol * mid if (base_vol > 0 and mid > 0) else 0.0
            if qv > 0:
                self._vol_cache_usd[asset] = qv

    def on_bar(
        self,
        bar_time: datetime,
        market_data: dict[tuple[str, str], pd.DataFrame],
        portfolio: PortfolioView,
        active_tfs: set[str],
    ) -> list[Signal]:
        """Generate signals for active TFs × universe."""
        signals = []

        # Get universe from selector
        universe = self._selector.universe

        # Get 4h data for regime filter
        ohlcv_4h = {}
        for asset in universe:
            key_4h = ("4h", asset)
            if key_4h in market_data:
                ohlcv_4h[asset] = market_data[key_4h]

        evaluated = 0
        no_data = 0
        no_model = 0

        for tf in active_tfs:
            if tf not in self._timeframes:
                continue
            hold = self._hold_map.get(tf, 4)

            for asset in universe:
                key = (tf, asset)
                ohlcv = market_data.get(key)
                if ohlcv is None or len(ohlcv) == 0:
                    no_data += 1
                    continue

                if not self._signal_gen.has_model(asset, tf):
                    no_model += 1
                    continue

                evaluated += 1
                sig = self._signal_gen.generate(
                    asset, tf, ohlcv, ohlcv_4h.get(asset), hold)

                if sig:
                    signals.append(Signal(
                        asset=sig.asset,
                        symbol=f"{sig.asset}{self._quote_currency}",
                        timeframe=sig.timeframe,
                        direction=sig.direction,
                        probability=sig.probability,
                        confidence=sig.confidence,
                        signal_strength=sig.signal_strength,
                        hold_bars=sig.hold_bars,
                        strategy_id=self._name,
                    ))
                    # Track direction for correlation throttle
                    self._recent_directions.append(float(sig.direction))
                    # Accumulate confidence for forward asset selection
                    self._selector.add_live_confidence(sig.asset, sig.confidence)

        logger.info("on_bar: universe=%d tfs=%s evaluated=%d no_data=%d no_model=%d signals=%d",
                     len(universe), sorted(active_tfs), evaluated, no_data, no_model, len(signals))

        return signals

    def check_exit(
        self,
        position: Position,
        market_data: dict[tuple[str, str], pd.DataFrame],
        bar_time: datetime,
    ) -> Optional[str]:
        """S8+Sym1.5 uses pure hold-bar expiry, no early exit."""
        return None

    def on_trade(self, asset: str, pnl: float, notional: float) -> None:
        """Update F1 breaker after a position closes."""
        self._breaker.record_trade(asset, pnl, notional)

    def get_sizing_mult(self, asset: str) -> float:
        """F1 symmetric breaker multiplier."""
        return self._breaker.get_multiplier(asset)

    def compute_size(
        self,
        signal: Signal,
        equity: float,
        drawdown: float,
        vol_mult: float,
    ) -> float:
        """Same sizing math as simulate_with_breaker:332-388."""
        if drawdown >= self._max_dd:
            return 0.0

        if drawdown > self._delever_start:
            size_mult = max(
                1.0 - (drawdown - self._delever_start) /
                (self._max_dd - self._delever_start), 0.0)
        else:
            size_mult = 1.0

        sig_abs = _edge_from_probability(signal.probability, signal.signal_strength)
        # Backtest applies a per-signal regime_mult (drawdown_improvements.py:335)
        # but yolo_compound currently sets every regime_mult = 1.0, so nothing is
        # wired here. If backtest ever populates non-unity regime_mults, thread
        # an `estimate_regime_mult(asset)` hook through and multiply here.

        raw_size = equity * self._base_sizing_pct * sig_abs * size_mult * vol_mult
        raw_size = min(raw_size,
                       equity * self._base_sizing_pct * self._max_signal_mult * size_mult)

        # Apply F1 breaker
        raw_size *= self.get_sizing_mult(signal.asset)

        # Correlation throttle: cut sizing when signals are too correlated
        raw_size *= self._correlation_throttle_mult()

        # Sizing guards — match backtest drawdown_improvements.py:462-480.
        # Only active when participation_rate > 0 AND vol cache is populated.
        # Silent no-op otherwise (backward compat when config omits sizing_guards).
        if self._participation_rate > 0:
            daily_vol = self._vol_cache_usd.get(signal.asset, 0.0)
            if daily_vol > 0:
                from signal_engine.slippage import max_profitable_size
                max_notional = daily_vol * self._participation_rate
                raw_size = min(raw_size, max_notional)
                edge = abs(2.0 * signal.probability - 1.0)
                s_max = max_profitable_size(
                    signal.asset, edge, daily_vol,
                    per_symbol_cal=self._slip_cal,
                    vip_tier=self._vip_tier,
                    guard_pct=self._guard_pct,
                )
                if s_max > 0:
                    raw_size = min(raw_size, s_max)

        if raw_size < 1.0:
            return 0.0

        return raw_size

    def preview_size(self, signal, portfolio) -> float:
        """Exact sizing preview for the aggregator.

        Mirrors `compute_size` exactly — sources vol_mult from the strategy's
        own overlay cache so the aggregator does not have to know about it.
        Returns the same USD notional that `compute_size` would produce when
        the engine later sizes the trade for real.
        """
        return self.compute_size(
            signal,
            equity=portfolio.equity,
            drawdown=portfolio.drawdown,
            vol_mult=self.compute_vol_mult(signal.asset),
        )

    def _correlation_throttle_mult(self) -> float:
        """CT90: if >threshold% of recent signals agree on direction, scale down.

        Mirrors drawdown_improvements.py:216-230 exactly.
        Returns 1.0 (no throttle) or self._ct_scale (throttled).
        """
        if self._ct_threshold is None:
            return 1.0
        if len(self._recent_directions) < self._ct_window:
            return 1.0

        dirs = np.array(self._recent_directions)
        pct_short = float((dirs < 0).mean())
        pct_long = float((dirs > 0).mean())

        if self._ct_direction == "both":
            throttle = max(pct_long, pct_short) >= self._ct_threshold
        elif self._ct_direction == "short":
            throttle = pct_short >= self._ct_threshold
        elif self._ct_direction == "long":
            throttle = pct_long >= self._ct_threshold
        else:
            throttle = False

        if throttle:
            logger.debug("CorrThrottle: %.0f%% long, %.0f%% short → scale %.1f",
                         pct_long * 100, pct_short * 100, self._ct_scale)
            return self._ct_scale
        return 1.0

    def get_state(self) -> dict:
        """Serialize strategy state for persistence."""
        return {
            "breaker": self._breaker.get_state(),
            "universe": self._selector.universe,
            "recent_directions": list(self._recent_directions),
        }

    def restore_state(self, state: dict) -> None:
        """Restore strategy state from persistence."""
        if "breaker" in state:
            self._breaker.restore_state(state["breaker"])
        if "universe" in state:
            self._selector.set_universe(state["universe"])
        if "recent_directions" in state:
            self._recent_directions = deque(
                state["recent_directions"], maxlen=self._ct_window)

    def set_vol_series(self, vol_series: dict[str, pd.Series]) -> None:
        """Set vol series for vol overlay computation."""
        self._vol_series = vol_series

    def compute_vol_mult(self, asset: str) -> float:
        """Per-asset inverse-vol sizing overlay.

        ``vol_mult = clip(target_vol / asset_daily_vol, floor, cap)`` — same
        risk-parity formula as ``apply_vol()`` in circuit_breaker_v2_experiment.py.

        FAILS SAFE: when the asset's volatility is unknown (series
        missing/empty, or a non-finite last value) this returns the clip
        FLOOR — the smallest size — never 1.0. Returning 1.0 was the live
        bug: during warmup / on any failed vol fetch the highest-vol assets
        (DENT, PIPPIN) were sized at full notional and produced 97% of the
        live paper loss.
        """
        vs = self._vol_series.get(asset)
        if vs is None or len(vs) == 0:
            return self._vol_floor
        v = vs.iloc[-1]
        if not np.isfinite(v) or v <= 0:
            return self._vol_floor
        return float(np.clip(self._vol_target / v, self._vol_floor, self._vol_cap))
