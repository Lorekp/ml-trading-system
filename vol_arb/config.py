"""Vol-arb pipeline configuration."""

from __future__ import annotations

from pydantic import BaseModel


class VolArbConfig(BaseModel):
    # Data paths
    option_trade_dir: str = "data/deribit/trades/BTC_option"
    dvol_path: str = "data/deribit/dvol/BTC_dvol_60m.parquet"
    tsrv_path: str = "TSRV/output/BTCUSDT_tsrv_15m.parquet"
    features_path: str = ""  # optional: vol_model/output/BTCUSDT_features_15m.parquet
    output_dir: str = "vol_arb/output"

    # Asset
    currency: str = "BTC"

    # ATM filter
    moneyness_lo: float = 0.95
    moneyness_hi: float = 1.05
    dte_min_days: int = 1
    dte_max_days: int = 14

    # Strangle filter (wider for 25-delta OTM strikes)
    strangle_moneyness_lo: float = 0.80
    strangle_moneyness_hi: float = 1.20

    # Greeks
    risk_free_rate: float = 0.0

    # Forward-fill limit for sparse option data (number of 15m bars, None = no fill)
    ffill_limit_bars: int | None = None

    # SOR: forward look window for taker print-through fill detection
    sor_tolerance_seconds: int = 5

    # Bar frequency — must match the TSRV parquet bar size
    # (e.g. "15min" for 15m TSRV, "60min" for 1h TSRV, "4h" for 240m TSRV)
    bucket_freq: str = "15min"

    # SVI surface calibration.
    # Cycle-30 (2026-05-13): default flipped from ``"1h"`` to ``"15min"``
    # to match what ``vol_arb/run_pipeline.py:121`` actually uses
    # (``bar_freq_str = "{args.bar_minutes}min"`` = ``"15min"`` at h=15).
    # Pre-cycle-30 the field default was structurally drifted from the
    # runner default; cycle-28b's parity guard then passed when both
    # config defaults matched even though the actual offline parquet
    # was at the runner-derived "15min" cadence (silent mismatch).
    svi_enabled: bool = False
    svi_bucket_freq: str = "15min"
    svi_min_strikes: int = 5
    svi_fallback_min_strikes: int = 3
    svi_use_pcp_forward: bool = True
    svi_cache: bool = True

    # Slippage surface buckets (per-currency — ETH option trades are 10x larger
    # in native units; IV distribution shifted higher). BTC defaults preserved.
    slippage_size_edges: list[float] = [0, 1, 3, 10, float("inf")]
    slippage_vol_edges: list[float] = [0, 45, 65, float("inf")]
