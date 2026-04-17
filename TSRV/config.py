"""Configuration for TSRV volatility signature analysis."""

from pydantic import BaseModel, Field


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
