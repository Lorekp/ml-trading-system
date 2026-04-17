"""Load tick data: bookTicker (quotes) or aggTrades (trades).

One day/month at a time — never loads everything into memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd


def available_dates(tick_dir: str | Path, symbol: str) -> list[str]:
    """List available dates/months sorted chronologically."""
    d = Path(tick_dir) / symbol
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.parquet"))


def load_day(tick_dir: str | Path, symbol: str, date: str) -> pd.DataFrame:
    """Load one day of bookTicker data.

    Returns DataFrame with DatetimeIndex (UTC), columns:
        mid, bid, ask, bid_qty, ask_qty
    """
    path = Path(tick_dir) / symbol / f"{date}.parquet"
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df["transaction_time"], unit="ms", utc=True)
    df["mid"] = (df["best_bid_price"].astype(float)
                 + df["best_ask_price"].astype(float)) / 2
    return df.rename(columns={
        "best_bid_price": "bid", "best_ask_price": "ask",
        "best_bid_qty": "bid_qty", "best_ask_qty": "ask_qty",
    })[["mid", "bid", "ask", "bid_qty", "ask_qty"]]


def iter_days(
    tick_dir: str | Path,
    symbol: str,
    start: str | None = None,
    end: str | None = None,
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield (date_str, df) for each day in range. Never loads all at once."""
    dates = available_dates(tick_dir, symbol)
    for d in dates:
        if start and d < start:
            continue
        if end and d > end:
            continue
        yield d, load_day(tick_dir, symbol, d)


def load_trades(trade_dir: str | Path, symbol: str, month: str) -> pd.DataFrame:
    """Load one month of aggTrades parquet.

    Returns DataFrame with DatetimeIndex (UTC), columns:
        price, quantity, dollar_volume, is_buyer_maker
    """
    path = Path(trade_dir) / symbol / f"{month}.parquet"
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df["transact_time"], unit="ms", utc=True)
    df = df.sort_index()
    return df[["price", "quantity", "dollar_volume", "is_buyer_maker"]]


def available_months(trade_dir: str | Path, symbol: str) -> list[str]:
    """List available months for aggTrades data."""
    return available_dates(trade_dir, symbol)


def iter_trade_days(
    trade_dir: str | Path,
    symbol: str,
    months: list[str] | None = None,
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield (date_str, day_df) from aggTrades, one day at a time.

    Loads month parquets and splits by day for memory efficiency.
    """
    all_months = months or available_months(trade_dir, symbol)
    for month in all_months:
        try:
            month_df = load_trades(trade_dir, symbol, month)
        except Exception:
            continue
        # Split by date
        for date_str, day_df in month_df.groupby(month_df.index.date):
            if len(day_df) > 0:
                yield str(date_str), day_df
