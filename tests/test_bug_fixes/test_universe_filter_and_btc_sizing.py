"""Regression tests for the historical S5 C2/C4 candidate overlay.

These tests pin its mechanics, look-ahead safety, and frozen synthetic
comparisons. They do not establish independent PnL realism or validate S5.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "s5_drawdown_reduction"))

from universe_filter_and_btc_sizing import (  # noqa: E402
    c2_universe_filter, c4_btc_corr_sizing, apply_stack,
)


# ── Fixtures ──

@pytest.fixture(scope="module")
def trade_log():
    """Load the S5 honest rebacktest trade log (cached)."""
    path = ROOT / "vol_arb/output/S5_HONEST_TRADE_LOG.parquet"
    if not path.exists():
        pytest.skip(f"Trade log not found at {path}")
    df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    return df


def _safe_inputs(df):
    return (
        df["pnl"].values.astype(np.float64),
        df["size"].values.astype(np.float64),
        df["skipped"].values.astype(bool),
        df["timestamp"].values,
        pd.Series(dtype=float),  # placeholder dpnl_adj (not used by C2/C4)
        df["asset"],
    )


# ── Look-ahead safety ──

def test_c2_no_look_ahead(trade_log):
    """The C2 filter must use only T-1 data when deciding on trade at T.

    A way to test this: construct a synthetic trade log where the T-1
    rolling Sharpe is positive but T's PnL is very negative. C2 should NOT
    skip the T trade (because T's loss wasn't visible at decision time).
    """
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    # Build an asset with positive T-1 rolling Sharpe, then a single bad T
    asset_pnl = pd.Series(0.001, index=dates)  # +0.1% per day
    asset_pnl.iloc[-1] = -0.5  # today: huge loss
    df = pd.DataFrame({
        "timestamp": dates,
        "asset": "TEST",
        "pnl": asset_pnl.values,
        "size": 1000.0,
        "skipped": False,
    })
    pnl, sizes, skipped, ts, dpnl, assets = _safe_inputs(df)
    # T-1 Sharpe > 0; today's loss should NOT trigger skip
    _, _, new_skipped, _ = c2_universe_filter(pnl, sizes, skipped, ts, dpnl, assets)
    # The last trade should NOT be skipped (decision was based on T-1 data only)
    assert not new_skipped[-1], (
        "C2 used today's PnL (look-ahead bias!). "
        "The T-1 rolling Sharpe is positive but today's loss shouldn't skip T.")


def test_c4_no_look_ahead(trade_log):
    """C4 correlation sizing must use T-1 correlation only."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    np.random.seed(42)
    n = len(dates)
    # BTC and TESTASSET are uncorrelated for 59 days, then perfectly correlated on day 60
    btc_pnl = np.random.normal(0, 0.01, n)
    test_pnl = np.random.normal(0, 0.01, n)
    test_pnl[-1] = btc_pnl[-1] * 100  # perfect co-move on day 60 only
    df = pd.DataFrame({
        "timestamp": list(dates) + list(dates),
        "asset": ["BTC"] * n + ["TEST"] * n,
        "pnl": np.concatenate([btc_pnl, test_pnl]),
        "size": 1000.0,
        "skipped": False,
    })
    df = df.sort_values(["timestamp", "asset"]).reset_index(drop=True)
    pnl, sizes, skipped, ts, dpnl, assets = _safe_inputs(df)
    new_pnl, new_sizes, new_skipped, _ = c4_btc_corr_sizing(
        pnl, sizes, skipped, ts, dpnl, assets)
    # The TEST asset on day 60 should NOT be reduced (T-1 correlation was 0)
    test_idx = df.index[df["asset"] == "TEST"].tolist()
    last_test = test_idx[-1]
    assert abs(new_sizes[last_test] - 1000.0) < 1e-6, (
        f"C4 used today's correlation (look-ahead bias!). "
        f"Expected size 1000.0 (T-1 uncorrelated), got {new_sizes[last_test]}")


# ── Functional invariants ──

def test_c2_skips_only_negative_sharpe(trade_log):
    """C2 must skip trades where the T-1 rolling Sharpe is negative."""
    df = trade_log.copy()
    df["asset"] = "FAKE_LOW_SHARPE"
    # Build a small dataset: 30 days of negative PnL, then 1 trade
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    daily_pnl = np.concatenate([np.full(30, -10.0), np.full(10, 10.0)])
    # Trade on day 30 (still in losing streak)
    pnl_arr = np.zeros(len(dates))
    sizes_arr = np.full(len(dates), 1000.0)
    skipped_arr = np.zeros(len(dates), dtype=bool)
    assets_arr = pd.Series(["FAKE_LOW_SHARPE"] * len(dates))
    # Add daily pnl from a side channel: we need to put it in `pnl` per trade
    # For simplicity: put a single trade on day 31
    df2 = pd.DataFrame({
        "timestamp": [dates[31]],
        "asset": ["FAKE_LOW_SHARPE"],
        "pnl": [50.0],
        "size": [1000.0],
        "skipped": [False],
    })
    # Add 30 days of background pnl via a fake asset
    bg = pd.DataFrame({
        "timestamp": dates[:30],
        "asset": ["BG_ASSET"] * 30,
        "pnl": [-10.0] * 30,
        "size": [1000.0] * 30,
        "skipped": [False] * 30,
    })
    full = pd.concat([bg, df2], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    pnl, sizes, skipped, ts, dpnl, assets = _safe_inputs(full)
    _, _, new_skipped, _ = c2_universe_filter(pnl, sizes, skipped, ts, dpnl, assets)
    # The trades on the asset with negative 30d Sharpe (BG_ASSET) should be
    # skipped AFTER the rolling window has warmed up. min_periods=10 means
    # the first 9 BG_ASSET trades don't have enough history; the rest
    # (trade 10+) must be skipped.
    bg_idx = full.index[full["asset"] == "BG_ASSET"].tolist()
    skipped_count = sum(1 for i in bg_idx if new_skipped[i])
    expected_skip = len(bg_idx) - 10  # warmup allows first 10 (min_periods=10)
    assert skipped_count == expected_skip, (
        f"C2 should skip {expected_skip}/{len(bg_idx)} BG_ASSET trades "
        f"(after 9-trade warmup). Skipped {skipped_count}/{len(bg_idx)}")


def test_apply_stack_preserves_skipped(trade_log):
    """apply_stack must never un-skip a previously skipped trade."""
    df = trade_log.copy()
    pnl, sizes, skipped, ts, dpnl, assets = _safe_inputs(df)
    _, _, new_skipped, _ = apply_stack(pnl, sizes, skipped, ts, dpnl, assets)
    # Every originally-skipped trade must remain skipped
    assert (new_skipped[skipped] | ~skipped[skipped]).all() or (skipped & new_skipped).sum() == skipped.sum(), (
        "apply_stack un-skipped a trade that was originally skipped")


# ── Headline metrics pinning ──

def test_stack_headline_metrics(trade_log):
    """Pin the ACKed metrics: Sharpe 3.65, MaxDD 18.5%, 7/7 positive years."""
    df = trade_log.copy()
    pnl, sizes, skipped, ts, dpnl, assets = _safe_inputs(df)
    new_pnl, new_sizes, new_skipped, new_dpnl = apply_stack(pnl, sizes, skipped, ts, dpnl, assets)
    # Sharpe (compound-equity-curve path matching the spec)
    active_pnl = new_pnl[~new_skipped]
    daily = pd.Series(active_pnl).groupby(
        pd.DatetimeIndex(pd.to_datetime(ts[~new_skipped])).normalize()
    ).sum().reindex(
        pd.date_range(pd.to_datetime(ts).min().normalize(),
                      pd.to_datetime(ts).max().normalize(), freq="D"),
        fill_value=0.0)
    dr = daily.values / 250000
    sr = float(dr.mean() / dr.std() * np.sqrt(365.25))
    assert sr >= 3.5, f"Sharpe regression: {sr:.3f} < 3.5 (expected ~3.65)"
    # MaxDD
    eq = np.cumprod(1 + dr)
    cummax = np.maximum.accumulate(eq)
    dd = (cummax - eq) / cummax
    max_dd = float(dd.max() * 100)
    assert max_dd < 25.0, f"MaxDD regression: {max_dd:.1f}% > 25% (expected ~18.5%)"
    # Per-year
    py = daily.resample("YE").sum()
    n_pos = int((py > 0).sum())
    assert n_pos == 7, f"Per-year regression: {n_pos}/7 positive (expected 7/7)"
