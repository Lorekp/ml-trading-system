"""Binance Futures universe screener.

Fetches historical daily volume for all USDT perpetuals from Binance,
computes 30-day median dollar volume with spike filtering,
and returns the top-N stable crypto assets.

Filters out:
- Non-crypto (gold, silver, stocks, forex)
- Recently listed (<90 days)
- Spike volumes (30d median > 5x the 90d median)
- Illiquid (below minimum volume threshold)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

BINANCE_FAPI = "https://fapi.binance.com"

# Non-crypto symbols to exclude (commodities, stocks, forex, stablecoins)
NON_CRYPTO = {
    "XAUUSDT", "XAGUSDT", "PAXGUSDT",  # gold/silver
    "TSLAUSDT", "AAPLUSDT", "AMZNUSDT", "GOOGLUSDT", "MSFRUSDT",  # stocks
    "EURUSDT", "GBPUSDT", "JPYUSDT",  # forex
    "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "FDUSDUSDT",  # stablecoins
}


def get_all_usdt_perps() -> list[str]:
    """Get all active USDT perpetual contract symbols."""
    r = requests.get(f"{BINANCE_FAPI}/fapi/v1/exchangeInfo", timeout=10)
    r.raise_for_status()
    data = r.json()
    symbols = [
        s["symbol"] for s in data["symbols"]
        if s["contractType"] == "PERPETUAL"
        and s["quoteAsset"] == "USDT"
        and s["status"] == "TRADING"
        and s["symbol"] not in NON_CRYPTO
    ]
    return symbols


def get_daily_klines(symbol: str, days: int = 120) -> pd.DataFrame | None:
    """Fetch daily klines for a symbol.

    Returns DataFrame with columns: open_time, close, volume, quote_volume.
    """
    end_ms = int(datetime.utcnow().timestamp() * 1000)
    start_ms = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)

    try:
        r = requests.get(
            f"{BINANCE_FAPI}/fapi/v1/klines",
            params={
                "symbol": symbol,
                "interval": "1d",
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": min(days + 5, 1500),
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None

    if not data:
        return None

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_vol",
        "taker_buy_quote_vol", "ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["quote_volume"] = df["quote_volume"].astype(float)
    df["close"] = df["close"].astype(float)
    df = df.set_index("open_time")
    return df[["close", "quote_volume"]]


def screen_universe(
    top_n: int = 20,
    min_median_volume: float = 20_000_000,  # $20M minimum 30d median
    min_history_days: int = 90,
    spike_ratio: float = 5.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Screen all Binance USDT perps and return top-N by 30d median volume.

    Args:
        top_n: Number of assets to return.
        min_median_volume: Minimum 30-day median daily dollar volume.
        min_history_days: Minimum trading days required.
        spike_ratio: Max allowed ratio of 30d/90d median (filters spikes).
        verbose: Print progress.

    Returns:
        DataFrame with columns: symbol, base, median_30d, median_90d,
        spike_ratio, days_listed, status.
    """
    symbols = get_all_usdt_perps()
    if verbose:
        print(f"Screening {len(symbols)} USDT perpetuals...")

    results = []
    batch_size = 10  # rate limit friendly
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        for symbol in batch:
            df = get_daily_klines(symbol, days=120)
            if df is None or len(df) < 10:
                continue

            base = symbol.replace("USDT", "")
            qv = df["quote_volume"]
            days_listed = len(qv)

            # 30-day median
            if len(qv) >= 30:
                median_30d = qv.iloc[-30:].median()
            else:
                median_30d = qv.median()

            # 90-day median
            if len(qv) >= 90:
                median_90d = qv.iloc[-90:].median()
            else:
                median_90d = qv.median()

            # Spike check
            ratio = median_30d / median_90d if median_90d > 0 else 999

            status = "OK"
            if days_listed < min_history_days:
                status = "TOO_NEW"
            elif median_30d < min_median_volume:
                status = "LOW_VOL"
            elif ratio > spike_ratio:
                status = "SPIKE"

            results.append({
                "symbol": symbol,
                "base": base,
                "median_30d": median_30d,
                "median_90d": median_90d,
                "spike_ratio": ratio,
                "days_listed": days_listed,
                "status": status,
            })

        if verbose and i % 50 == 0 and i > 0:
            print(f"  {i}/{len(symbols)}...", flush=True)
        time.sleep(0.1)  # rate limit

    result_df = pd.DataFrame(results)
    if verbose:
        print(f"  Fetched {len(result_df)} symbols")

    return result_df


def get_top_assets(
    top_n: int = 20,
    min_median_volume: float = 20_000_000,
    verbose: bool = True,
) -> tuple[list[str], pd.DataFrame]:
    """Get the current top-N crypto assets by 30d median volume.

    Returns:
        (list of base asset names like ["BTC", "ETH", ...], full DataFrame)
    """
    df = screen_universe(top_n=top_n, min_median_volume=min_median_volume, verbose=verbose)
    ok = df[df["status"] == "OK"].sort_values("median_30d", ascending=False)
    top = ok.head(top_n)
    return list(top["base"]), df


def report_download_needs(
    top_assets: list[str],
    available_assets: dict[str, str],
) -> list[str]:
    """Report which assets need to be downloaded.

    Args:
        top_assets: List of base asset names from get_top_assets.
        available_assets: Dict of base -> symbol that we already have data for.

    Returns:
        List of symbols to download.
    """
    need = []
    for base in top_assets:
        symbol = base + "USDT"
        if base not in available_assets:
            need.append(symbol)
    return need
