"""Download configuration — Pydantic model with all parameters."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DownloadConfig(BaseModel):
    # Assets
    currencies: list[str] = ["BTC", "ETH"]
    kinds: list[str] = ["option"]

    # Time range (ISO dates)
    start: str = "2023-05-01"
    end: str = "2024-04-01"

    # Output
    output_dir: str = "data/deribit"

    # Auth (authenticated = 20 req/s, unauthenticated = ~10 req/s)
    client_id: str = ""
    client_secret: str = ""

    # Rate limiting (Deribit credit system)
    credits_per_request: int = 500
    credit_pool: int = 50_000
    credit_refill_per_sec: int = 10_000

    # Retries
    max_retries: int = 10
    backoff_base: float = 1.5
    max_backoff: float = 60.0

    # Parallelism (4 workers for authenticated 20 req/s ceiling)
    max_workers: int = 4

    # Endpoints to download (ordered by priority/speed)
    endpoints: list[str] = Field(
        default=["dvol", "historical_vol", "candles", "mark_price", "trades"])

    # Trades: daily full tick history for 1-min IV resolution
    trades_sample_days: int = 1   # every day (1=daily, 7=weekly)
    trades_max_pages: int = 0     # 0=unlimited (full day coverage)

    # Resolution for DVOL / candles
    # DVOL: "3600" (1h), "43200" (12h), "1D" — seconds or "1D"
    # Candles: "1", "3", "5", "10", "15", "30", "60", "120", "180", "360", "720", "1D"
    dvol_resolution: str = "3600"  # 1h (finest available historically)
    candle_resolution: str = "5"   # 5-min

    # Candle instrument (for get_tradingview_chart_data)
    candle_instruments: list[str] = Field(
        default=["BTC-PERPETUAL", "ETH-PERPETUAL"])

    @property
    def authenticated(self) -> bool:
        return bool(self.client_id and self.client_secret)
