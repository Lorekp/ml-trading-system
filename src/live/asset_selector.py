"""S8 regime-adaptive asset selection for live trading.

Exact port of yolo_compound.py:104-136 (_regime_sel + _s8_sel).
Uses the backtest prep cache (v2_prep_cache.pkl) for confidence × volume
ranking, BTC volatility regime, and candidate pool — ensuring the live
system selects the exact same universe as the backtest.

Core9 always included. Expansion pool contracts when BTC vol is high,
expands when vol is low. Rebalanced monthly.

Fallback: if prep cache is missing, falls back to API-based discovery
(24h turnover ranking) with a loud warning.
"""

from __future__ import annotations

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from typing import Protocol, runtime_checkable


@runtime_checkable
class MarketDataClient(Protocol):
    """Any client that provides fetch_all_tickers() and fetch_klines()."""
    async def fetch_all_tickers(self) -> list[dict]: ...
    async def fetch_klines(self, symbol: str, interval: str, limit: int = 200) -> list[dict]: ...

from live.config import OBVIOUS_CORE

logger = logging.getLogger(__name__)

ELIGIBLE_FILE = Path("data/eligible_symbols.json")


class S8AssetSelector:
    """S8 regime-adaptive asset selection.

    Primary mode: load backtest prep cache and run the exact same
    _regime_sel + _s8_sel logic as yolo_compound.py:104-136.

    Fallback mode: API-based discovery (if cache missing/stale).
    """

    def __init__(
        self,
        core: set[str] = None,
        max_total: int = 30,
        btc_vol_window: int = 20,
        prep_cache_path: str = "",
    ) -> None:
        self._core = core or OBVIOUS_CORE
        self._max_total = max_total
        self._btc_vol_window = btc_vol_window
        self._current_universe: list[str] = sorted(self._core)
        self._last_rebalance: Optional[datetime] = None

        # Backtest data (loaded from prep cache)
        self._hold_sigs: dict = {}
        self._asset_volumes: dict = {}
        self._vol_series: dict = {}
        self._cache_loaded = False

        # Live confidence accumulation (extends beyond cache)
        self._live_confidences: dict[str, list[float]] = {}

        if prep_cache_path:
            self._load_prep_cache(Path(prep_cache_path))

        # Fallback data (for API-based mode)
        self._listing_dates: dict[str, datetime] = self._load_listing_dates()

    def _load_prep_cache(self, path: Path) -> None:
        """Load backtest prep cache for exact-match asset selection."""
        if not path.exists():
            logger.warning("S8: prep cache not found at %s — will use API fallback", path)
            return
        try:
            with open(path, "rb") as f:
                cached = pickle.load(f)
            self._hold_sigs = cached.get("hold_sigs", {})
            self._asset_volumes = cached.get("asset_volumes", {})
            self._vol_series = cached.get("vol_series", {})
            self._cache_loaded = True

            # Report cache date range
            btc_vol = self._vol_series.get("BTC")
            cache_end = btc_vol.index.max() if btc_vol is not None else "unknown"
            logger.info("S8: loaded prep cache from %s — %d assets with signals, "
                        "%d with volume, cache ends %s",
                        path, len(self._hold_sigs), len(self._asset_volumes), cache_end)
        except Exception as e:
            logger.error("S8: failed to load prep cache: %s — will use API fallback", e)

    @classmethod
    def from_config(cls, config) -> S8AssetSelector:
        return cls(
            core=set(config.core),
            max_total=config.max_total,
            btc_vol_window=config.btc_vol_window,
            prep_cache_path=getattr(config, "prep_cache_path", ""),
        )

    @property
    def universe(self) -> list[str]:
        return list(self._current_universe)

    def add_live_confidence(self, asset: str, confidence: float) -> None:
        """Accumulate live confidence for forward asset selection."""
        if asset not in self._live_confidences:
            self._live_confidences[asset] = []
        self._live_confidences[asset].append(confidence)

    async def maybe_rebalance(self, bar_time: datetime,
                              client: MarketDataClient,
                              available_assets: list[str],
                              force: bool = False) -> bool:
        """Rebalance if it's the 1st of the month at midnight, or if forced.

        Args:
            force: Skip date checks — used on first bar after startup.

        Returns True if rebalance happened.
        """
        if not force:
            if bar_time.day != 1 or bar_time.hour != 0:
                return False

            if (self._last_rebalance and
                    self._last_rebalance.month == bar_time.month and
                    self._last_rebalance.year == bar_time.year):
                return False

        if self._cache_loaded:
            self._rebalance_from_cache(bar_time, available_assets)
        else:
            await self._rebalance_from_api(client, available_assets)

        self._last_rebalance = bar_time
        return True

    # ──────────────────────────────────────────────────────────────
    #  PRIMARY: exact backtest logic from prep cache
    # ──────────────────────────────────────────────────────────────

    def _rebalance_from_cache(self, bar_time: datetime,
                              available_assets: list[str]) -> None:
        """Exact port of yolo_compound.py:104-136 (_regime_sel + _s8_sel).

        Uses hold_sigs, asset_volumes, vol_series from prep cache.
        Produces the exact same universe as the backtest for any given date.
        """
        ms = pd.Timestamp(bar_time)
        if ms.tzinfo is not None:
            ms = ms.tz_localize(None)

        model_set = set(available_assets)

        # ── Check cache staleness ──
        btc_vol = self._vol_series.get("BTC")
        if btc_vol is not None:
            cache_end = btc_vol.index.max()
            days_stale = (ms - cache_end).days
            if days_stale > 45:
                logger.warning("S8: prep cache is %d days stale (ends %s, bar=%s). "
                               "Run update_prep_cache.py to refresh.",
                               days_stale, cache_end.date(), ms.date())

        # ── 1. BTC vol regime (exact backtest: yolo_compound.py:105-112) ──
        n_assets = self._max_total  # default 30
        if btc_vol is not None:
            avail = btc_vol.loc[:ms - pd.Timedelta(days=1)].dropna()
            if len(avail) > 60:
                current = avail.iloc[-1]
                p80 = avail.quantile(0.80)
                p20 = avail.quantile(0.20)
                if current > p80:
                    n_assets = 15
                    logger.info("S8 BTC vol HIGH (%.4f > p80 %.4f) -> N=15",
                                current, p80)
                elif current < p20:
                    n_assets = 40
                    logger.info("S8 BTC vol LOW (%.4f < p20 %.4f) -> N=40",
                                current, p20)
                else:
                    logger.info("S8 BTC vol NORMAL (%.4f, p20=%.4f, p80=%.4f) -> N=%d",
                                current, p20, p80, n_assets)
            else:
                logger.info("S8 BTC vol: insufficient history (%d valid), using default N=%d",
                            len(avail), n_assets)

        # ── 2. Score by confidence × volume (exact: yolo_compound.py:113-131) ──
        cutoff = ms - pd.DateOffset(months=3)
        scored = []
        for name in self._hold_sigs:
            confs = []
            for tf in self._hold_sigs[name]:
                s = self._hold_sigs[name][tf]
                ts = pd.to_datetime(s["timestamp"])
                if ts.dt.tz is not None:
                    ts = ts.dt.tz_localize(None)
                w = s[((ts >= cutoff) & (ts < ms)).values]
                if "confidence" in w.columns and len(w) > 5:
                    confs.extend(w["confidence"].values)
            # Also merge live confidences if available
            live_confs = self._live_confidences.get(name, [])
            if live_confs:
                confs.extend(live_confs)
            if not confs:
                continue
            vol = self._asset_volumes.get(name)
            if vol is None:
                continue
            avail_v = vol.loc[:ms - pd.Timedelta(days=1)]
            if len(avail_v) == 0:
                continue
            scored.append((name, float(np.mean(confs)) * float(avail_v.iloc[-1])))
        scored.sort(key=lambda x: -x[1])

        # ── 3. Core + expansion (exact: yolo_compound.py:133-136) ──
        regime_sel = [a for a, _ in scored[:n_assets]]
        active_core = sorted(a for a in self._core if a in self._hold_sigs)
        universe = active_core + [a for a in regime_sel if a not in self._core]

        # ── 4. Report ──
        self._current_universe = universe
        n_core = len(active_core)
        n_expansion = len(universe) - n_core
        n_with_models = len(set(universe) & model_set)
        gap = sorted(set(universe) - model_set)

        # Log top expansion for transparency
        top5 = [(a, s) for a, s in scored[:5]]
        top5_str = ", ".join(f"{a}({s/1e6:.0f}M)" for a, s in top5)
        logger.info("S8 rebalance (cache): %d assets (core=%d + expansion=%d), "
                     "target=%d, models=%d/%d, top5=[%s]%s",
                     len(universe), n_core, n_expansion, n_assets,
                     n_with_models, len(universe), top5_str,
                     f", gap={gap}" if gap else "")

    # ──────────────────────────────────────────────────────────────
    #  FALLBACK: API-based discovery (when no prep cache)
    # ──────────────────────────────────────────────────────────────

    async def _rebalance_from_api(self, client: MarketDataClient,
                                  available_assets: list[str]) -> None:
        """Fallback: discover assets from API and rank by 24h turnover.

        WARNING: This does NOT match the backtest ranking (confidence × volume).
        Only used when prep cache is unavailable.
        """
        logger.warning("S8 rebalance: using API fallback (no prep cache). "
                       "Universe will NOT match backtest exactly.")

        model_set = set(available_assets)

        # ── 1. Discover full market ──
        all_tickers = await client.fetch_all_tickers()
        if not all_tickers:
            logger.warning("S8 rebalance: fetch_all_tickers returned empty, keeping current universe")
            return

        active = [t for t in all_tickers if t["turnover_24h"] > 0 and t["last_price"] > 0]
        logger.info("S8 rebalance (API): %d USDT perps discovered (%d active)",
                     len(all_tickers), len(active))

        # Filter by listing age
        min_age_days = 180
        aged_assets = self._filter_by_listing_age(
            {t["asset"] for t in active}, min_age_days)
        young = {t["asset"] for t in active} - aged_assets
        if young:
            logger.info("S8 rebalance: excluded %d assets with < %d days history: %s",
                         len(young), min_age_days, sorted(young)[:10])
        active = [t for t in active if t["asset"] in aged_assets]
        active_assets = {t["asset"] for t in active}

        # ── 2. BTC vol regime (from API, 200 bars only) ──
        n_assets = self._max_total
        try:
            btc_bars = await client.fetch_klines("BTCUSDT", "1d", 200)
            if len(btc_bars) >= 30:
                closes = [b["close"] for b in btc_bars]
                returns = pd.Series(closes).pct_change().dropna()
                vol_20d = returns.rolling(self._btc_vol_window).std()
                current_vol = vol_20d.iloc[-1]

                if len(vol_20d.dropna()) > 60:
                    p80 = vol_20d.dropna().quantile(0.80)
                    p20 = vol_20d.dropna().quantile(0.20)
                    if current_vol > p80:
                        n_assets = 15
                    elif current_vol < p20:
                        n_assets = 40
        except Exception as e:
            logger.warning("S8 BTC vol computation failed: %s — using default N=%d", e, n_assets)

        # ── 3. Core always included ──
        core_on_exchange = sorted(self._core & active_assets)
        universe = list(core_on_exchange)

        # ── 4. Rank by turnover, fill expansion ──
        turnover_map = {t["asset"]: t["turnover_24h"] for t in active}
        candidates = [(a, turnover_map.get(a, 0)) for a in active_assets if a not in self._core]
        candidates.sort(key=lambda x: -x[1])
        expansion = [a for a, _ in candidates[:n_assets]]
        universe.extend(a for a in expansion if a not in self._core)

        self._current_universe = universe
        n_core = len(core_on_exchange)
        n_expansion = len(universe) - n_core
        n_with_models = len(set(universe) & model_set)
        logger.info("S8 rebalance (API fallback): %d assets (core=%d + expansion=%d), models=%d/%d",
                     len(universe), n_core, n_expansion, n_with_models, len(universe))

    # ──────────────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _load_listing_dates() -> dict[str, datetime]:
        """Load listing dates from eligible_symbols.json."""
        if not ELIGIBLE_FILE.exists():
            return {}
        try:
            import json
            with open(ELIGIBLE_FILE) as f:
                symbols = json.load(f)
            dates = {}
            for s in symbols:
                name = s.get("name", "")
                ld = s.get("listing_date", "")
                if name and ld:
                    dates[name] = datetime.strptime(ld, "%Y-%m-%d")
            return dates
        except Exception as e:
            logger.warning("Failed to load listing dates: %s", e)
            return {}

    def _filter_by_listing_age(self, assets: set[str], min_days: int) -> set[str]:
        """Return only assets with at least min_days of history."""
        now = datetime.utcnow()
        result = set()
        for asset in assets:
            ld = self._listing_dates.get(asset)
            if ld is None:
                result.add(asset)
            elif (now - ld).days >= min_days:
                result.add(asset)
        return result

    def set_universe(self, assets: list[str]) -> None:
        """Manually set universe (e.g. on startup from checkpoint)."""
        self._current_universe = list(assets)
