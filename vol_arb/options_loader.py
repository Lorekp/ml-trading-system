"""Load, parse, and filter Deribit option trade data.

Handles instrument name parsing, expiry date extraction, time-to-expiry
calculation, and ATM/DTE filtering — all fully vectorized (no loops, no .apply).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Core columns we need from each parquet file (schema varies across dates)
CORE_COLUMNS = [
    "timestamp", "instrument_name", "iv", "index_price",
    "price", "mark_price", "direction", "amount", "trade_seq",
]


def parse_instrument_names(names: pd.Series) -> pd.DataFrame:
    """Vectorized extraction of strike, expiry string, and option type.

    Input: Series of 'BTC-2JUN23-27000-C' style strings.
    Returns: DataFrame with columns [currency, expiry_str, strike, option_type].
    """
    parts = names.str.split("-", expand=True)
    return pd.DataFrame({
        "currency": parts[0],
        "expiry_str": parts[1],
        "strike": parts[2].astype(np.float64),
        "option_type": parts[3],
    })


def parse_expiry_dates(expiry_strs: pd.Series) -> pd.Series:
    """Convert Deribit expiry strings to datetime at 08:00 UTC.

    Handles variable-width day: '2JUN23' and '30JUN23' both work with %d%b%y.
    Deribit options settle at 08:00:00 UTC on the expiry date.
    """
    dt = pd.to_datetime(expiry_strs, format="%d%b%y", utc=True)
    return dt + pd.Timedelta(hours=8)


def load_option_trades(
    trade_dir: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load all option trade parquet files, parse instruments, compute T and moneyness.

    Parameters
    ----------
    trade_dir : path to directory of YYYY-MM-DD.parquet files
    start_date, end_date : optional ISO date strings to filter files

    Returns
    -------
    DataFrame with original core columns plus:
        strike, expiry_dt, option_type, T_years, moneyness, dte_days
    Rows with T <= 0 or iv <= 0 are dropped.
    """
    trade_dir = Path(trade_dir)
    files = sorted(trade_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {trade_dir}")

    # Filter to date range if specified
    if start_date or end_date:
        filtered = []
        for f in files:
            stem = f.stem  # e.g. "2023-06-01"
            if start_date and stem < start_date:
                continue
            if end_date and stem > end_date:
                continue
            filtered.append(f)
        files = filtered
        logger.info("Filtered to %d files in [%s, %s]", len(files), start_date, end_date)

    if not files:
        raise FileNotFoundError(f"No files in date range [{start_date}, {end_date}]")

    # Load all files, selecting only core columns that exist
    chunks = []
    for f in files:
        df = pd.read_parquet(f)
        cols = [c for c in CORE_COLUMNS if c in df.columns]
        chunks.append(df[cols])

    df = pd.concat(chunks, ignore_index=True)
    logger.info("Loaded %d trades from %d files", len(df), len(files))

    # Parse instrument names (vectorized)
    parsed = parse_instrument_names(df["instrument_name"])
    df["strike"] = parsed["strike"]
    df["option_type"] = parsed["option_type"]

    # Parse expiry dates
    df["expiry_dt"] = parse_expiry_dates(parsed["expiry_str"])

    # Time to expiry in years
    td = (df["expiry_dt"] - df["timestamp"]).dt.total_seconds()
    df["T_years"] = td / (365.25 * 86400)
    df["dte_days"] = td / 86400

    # Moneyness = K / S
    df["moneyness"] = df["strike"] / df["index_price"]

    # Drop invalid rows
    n_before = len(df)
    df = df[(df["T_years"] > 0) & (df["iv"] > 0)].copy()
    dropped = n_before - len(df)
    if dropped:
        logger.info("Dropped %d rows (expired or zero IV)", dropped)

    return df


def filter_atm(
    df: pd.DataFrame,
    moneyness_lo: float = 0.95,
    moneyness_hi: float = 1.05,
    dte_min: int = 1,
    dte_max: int = 14,
) -> pd.DataFrame:
    """Filter to near-ATM, short-dated options. Pure boolean mask, no loops."""
    mask = (
        (df["moneyness"] >= moneyness_lo)
        & (df["moneyness"] <= moneyness_hi)
        & (df["dte_days"] >= dte_min)
        & (df["dte_days"] <= dte_max)
    )
    result = df[mask].copy()
    logger.info(
        "ATM filter: %d / %d trades (%.1f%%)",
        len(result), len(df), 100 * len(result) / max(len(df), 1),
    )
    return result
