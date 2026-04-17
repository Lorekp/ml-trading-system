"""Tests for asset and timeframe agnosticity.

Verifies that changing (currency, bar_minutes) propagates correctly
to all derived values — computed properties, file paths, WebSocket
URLs, and downstream config bar-counts.
"""

from __future__ import annotations

import pytest


# ── PaperConfig ────────────────────────────────────────────────────


class TestPaperConfigDefaults:

    def test_btc_15m_defaults_unchanged(self):
        """Default BTC/15m config produces the same values as before."""
        from deribit_paper.config import PaperConfig
        c = PaperConfig()
        assert c.currency == "BTC"
        assert c.bar_minutes == 15
        assert c.bars_per_day == 96
        assert c.bar_seconds == 900
        assert c.asset_pair == "BTCUSDT"
        assert c.binance_ws_url == "wss://fstream.binance.com/ws/btcusdt@aggTrade"
        assert c.features_parquet == "vol_model/output/BTCUSDT_features_15m.parquet"
        assert c.svi_features_parquet == "vol_arb/output/svi_features_15m.parquet"

    def test_eth_5m_paths(self):
        """ETH/5m config resolves all paths correctly."""
        from deribit_paper.config import PaperConfig
        c = PaperConfig(currency="ETH", bar_minutes=5)
        assert c.bars_per_day == 288
        assert c.bar_seconds == 300
        assert c.asset_pair == "ETHUSDT"
        assert "ethusdt" in c.binance_ws_url
        assert "ETHUSDT_features_5m" in c.features_parquet
        assert "svi_features_5m" in c.svi_features_parquet

    def test_sol_1h(self):
        """SOL/1h config resolves correctly."""
        from deribit_paper.config import PaperConfig
        c = PaperConfig(currency="SOL", bar_minutes=60)
        assert c.bars_per_day == 24
        assert c.bar_seconds == 3600
        assert "solusdt" in c.binance_ws_url
        assert "SOLUSDT_features_60m" in c.features_parquet

    def test_explicit_override_preserved(self):
        """Explicit path override is not overwritten by __post_init__."""
        from deribit_paper.config import PaperConfig
        c = PaperConfig(
            currency="ETH",
            features_parquet="custom/path.parquet",
        )
        assert c.features_parquet == "custom/path.parquet"


# ── RNDConfig ──────────────────────────────────────────────────────


class TestRNDConfigBarCounts:

    def test_default_15m_matches_original(self):
        """Default bar counts at 15m match the original hardcoded values."""
        from deribit_paper.risk.rnd_monitor import RNDConfig
        c = RNDConfig()
        assert c.baseline_warmup_bars == 96       # 1 day @ 15min
        assert c.baseline_window_bars == 2880     # 30 days
        assert c.baseline_long_window_bars == 8640  # 90 days

    def test_5m_scales_correctly(self):
        """5-min bars produce 3x the bar counts."""
        from deribit_paper.risk.rnd_monitor import RNDConfig
        c = RNDConfig(bar_minutes=5)
        assert c.baseline_warmup_bars == 288      # 1 day @ 5min
        assert c.baseline_window_bars == 8640     # 30 days
        assert c.baseline_long_window_bars == 25920  # 90 days

    def test_explicit_override_preserved(self):
        """Explicit bar count is preserved even with different bar_minutes."""
        from deribit_paper.risk.rnd_monitor import RNDConfig
        c = RNDConfig(bar_minutes=5, baseline_warmup_bars=100)
        assert c.baseline_warmup_bars == 100


# ── DRLHedgingConfig ──────────────────────────────────────────────


class TestDRLHedgingConfigDtYears:

    def test_default_15m_matches_original(self):
        """Default dt_years at 15m matches the original hardcoded formula."""
        from vol_arb.drl_hedging.config import DRLHedgingConfig
        c = DRLHedgingConfig()
        expected = (15.0 / 60.0 / 24.0 / 365.25) / 10
        assert c.dt_years == pytest.approx(expected)

    def test_5m_dt_years(self):
        """5-min bars produce a different dt_years."""
        from vol_arb.drl_hedging.config import DRLHedgingConfig
        c = DRLHedgingConfig(bar_minutes=5)
        expected = (5.0 / 60.0 / 24.0 / 365.25) / 10
        assert c.dt_years == pytest.approx(expected)
        assert c.dt_years < DRLHedgingConfig().dt_years  # shorter bars -> smaller dt


# ── BacktestConfig ────────────────────────────────────────────────


class TestBacktestConfigForTimeframe:

    def test_for_timeframe_15m_defaults(self):
        """for_timeframe(15) matches the hardcoded defaults."""
        from vol_arb.backtester import BacktestConfig
        c = BacktestConfig.for_timeframe(bar_minutes=15)
        assert c.holding_minutes == 15
        assert c.vrp_window == 672
        assert c.vrp_min_periods == 96

    def test_for_timeframe_5m(self):
        """for_timeframe(5) scales bar counts correctly."""
        from vol_arb.backtester import BacktestConfig
        c = BacktestConfig.for_timeframe(bar_minutes=5)
        assert c.holding_minutes == 5
        assert c.vrp_window == 288 * 7  # 2016
        assert c.vrp_min_periods == 288

    def test_for_timeframe_with_overrides(self):
        """Explicit overrides take precedence."""
        from vol_arb.backtester import BacktestConfig
        c = BacktestConfig.for_timeframe(
            bar_minutes=5, initial_equity=200_000,
        )
        assert c.initial_equity == 200_000
        assert c.holding_minutes == 5


# ── LiveDRLHedger ─────────────────────────────────────────────────


class TestLiveDRLHedgerBarMinutes:

    def test_bar_minutes_stored(self):
        """bar_minutes parameter is stored on the instance."""
        from unittest.mock import MagicMock
        from deribit_paper.signal.drl_hedger_live import LiveDRLHedger
        from deribit_paper.signal.model_loader import ModelBundle
        import numpy as np

        drl = MagicMock()
        drl.predict.return_value = (np.array([[0.0]]), None)
        bundle = MagicMock(spec=ModelBundle)
        bundle.drl_policy = drl
        bundle.drl_vec_normalize_mean = None

        hedger = LiveDRLHedger(bundle=bundle, bar_minutes=5)
        assert hedger._bar_minutes == 5
