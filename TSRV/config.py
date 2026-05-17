"""Configuration for TSRV volatility signature analysis."""

from pydantic import BaseModel, Field


# ── Cycle-39 (2026-05-17): canonical sub-bar dollar threshold SSOT ──
#
# The production BTCUSDT 15-min TSRV series
# (``TSRV/output/BTCUSDT_tsrv_15m.parquet``) is built with this
# sub-bar dollar threshold (via ``run_tsrv_incremental --threshold``).
# The live engine's ``compute_bar_tsrv`` MUST use the IDENTICAL value
# or its TSRV decomposition diverges from the offline pipeline.
#
# Parity audit 2026-05-17: live ``compute_bar_tsrv`` defaulted to
# 50_000 while the offline parquet was built at 250_000 — a 5x
# mismatch that biased live ``jump_fraction`` +0.167 and ``vol_ann``
# by ~0.09 over a 377-bar window. Both paths share
# ``compute_tsrv_window``; only this threshold differed. This module
# constant is the single source of truth: ``deribit_paper`` imports
# it for the live engine, ``run_tsrv_incremental`` falls back to it.
BTCUSDT_TSRV_15M_SUBBAR_THRESHOLD_USD: float = 250_000.0


class TSRVConfig(BaseModel):
    # Data
    tick_dir: str = "data/bookTicker_parquet"
    trade_dir: str = "data/binance-aggTrades-parquet"
    asset: str = "BTCUSDT"
    date_range: tuple[str, str] | None = None  # None = all available

    # Time-based sampling intervals (seconds)
    time_intervals: list[float] = Field(
        default=[1, 2, 5, 8, 10, 15, 30, 60, 120, 300, 600])

    # Dollar bar thresholds (real trade volume, USD)
    dollar_thresholds: list[float] = Field(
        default=[100_000, 500_000, 1_000_000, 2_500_000, 5_000_000,
                 10_000_000, 25_000_000, 50_000_000, 100_000_000])

    # Tick bar thresholds (ticks per bar)
    tick_thresholds: list[int] = Field(
        default=[100, 250, 500, 1000, 2500, 5000, 10000, 25000])

    # RV computation
    rv_method: str = "log_return"  # "log_return" or "simple_return"

    # TSRV
    tsrv_K: int | None = None  # None = optimal (n^{2/3})

    # TSRV time series
    slow_timeframe_s: int = 900  # 15m = 900, 1h = 3600, 4h = 14400
    sub_bar_thresholds: list[float] = Field(
        default=[10_000, 25_000, 50_000, 100_000, 250_000,
                 500_000, 1_000_000, 2_500_000, 5_000_000])
    optimal_sub_bar_threshold: float | None = None

    # Knee detection (KneeLocator)
    knee_sensitivity: float = 1.0
    knee_curve: str = "convex"
    knee_direction: str = "decreasing"
    knee_interp: str = "polynomial"
    knee_poly_degree: int = 7

    # Output
    output_dir: str = "TSRV/output"
    plot: bool = True
