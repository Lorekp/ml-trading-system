#!/usr/bin/env python
"""Measure price divergence between Binance and MEXC Futures.

Same methodology as cross_exchange_cost.py but for MEXC.
Simultaneous BBO snapshots to measure mid-to-mid difference and spread.

Usage:
    PYTHONPATH="src;." python experiments/cross_exchange_cost_mexc.py
"""

import sys, os, asyncio, time, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timezone

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src"); sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import aiohttp

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT",
    "ADAUSDT", "ETCUSDT", "LTCUSDT", "BNBUSDT", "LINKUSDT",
    "SUIUSDT", "TAOUSDT", "AVAXUSDT",
]

# MEXC uses slightly different symbol format for some — test first
ROUNDS = 15
INTERVAL_S = 20


async def fetch_binance_bbo(session, symbol):
    url = f"https://fapi.binance.com/fapi/v1/ticker/bookTicker?symbol={symbol}"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            d = await resp.json()
            if isinstance(d, dict):
                return {"bid": float(d.get("bidPrice", 0)), "ask": float(d.get("askPrice", 0))}
    except Exception:
        pass
    return None


async def fetch_mexc_bbo(session, symbol):
    """MEXC Futures V1 API — ticker endpoint."""
    # MEXC futures uses _USDT suffix for contract symbols
    mexc_sym = symbol.replace("USDT", "_USDT")
    url = f"https://contract.mexc.com/api/v1/contract/ticker?symbol={mexc_sym}"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            d = await resp.json()
            if d.get("success") and d.get("data"):
                t = d["data"]
                return {
                    "bid": float(t.get("bid1", 0)),
                    "ask": float(t.get("ask1", 0)),
                    "last": float(t.get("lastPrice", 0)),
                    "volume_24h": float(t.get("volume24", 0)),
                }
    except Exception:
        pass
    # Try alternate endpoint
    url2 = f"https://futures.mexc.com/api/v1/contract/ticker?symbol={mexc_sym}"
    try:
        async with session.get(url2, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            d = await resp.json()
            if d.get("success") and d.get("data"):
                t = d["data"]
                return {
                    "bid": float(t.get("bid1", 0)),
                    "ask": float(t.get("ask1", 0)),
                    "last": float(t.get("lastPrice", 0)),
                }
    except Exception:
        pass
    return None


async def fetch_bybit_bbo(session, symbol):
    url = f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol}"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            d = await resp.json()
            tickers = d.get("result", {}).get("list", [])
            if tickers:
                t = tickers[0]
                return {"bid": float(t.get("bid1Price", 0)), "ask": float(t.get("ask1Price", 0))}
    except Exception:
        pass
    return None


async def main():
    print("=" * 120)
    print("  THREE-EXCHANGE BBO COMPARISON: Binance vs MEXC vs Bybit")
    print(f"  {ROUNDS} rounds x {len(SYMBOLS)} symbols, {INTERVAL_S}s apart")
    print("=" * 120)

    rows = []

    async with aiohttp.ClientSession() as session:
        # First: test which MEXC symbols work
        print("  Testing MEXC symbol availability...")
        available = []
        for sym in SYMBOLS:
            mx = await fetch_mexc_bbo(session, sym)
            if mx and mx.get("bid", 0) > 0:
                available.append(sym)
                print(f"    {sym}: OK (bid={mx['bid']})")
            else:
                print(f"    {sym}: NOT AVAILABLE on MEXC futures")
            await asyncio.sleep(0.2)
        print(f"  {len(available)}/{len(SYMBOLS)} symbols available on MEXC\n")

        symbols = available

        for rnd in range(ROUNDS):
            ts = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")

            tasks = []
            for sym in symbols:
                tasks.append(asyncio.gather(
                    fetch_binance_bbo(session, sym),
                    fetch_mexc_bbo(session, sym),
                    fetch_bybit_bbo(session, sym),
                ))
            results = await asyncio.gather(*tasks)

            for sym, (bn, mx, by) in zip(symbols, results):
                if not bn or not mx or not by:
                    continue
                if bn["bid"] <= 0 or mx.get("bid", 0) <= 0 or by["bid"] <= 0:
                    continue

                bn_mid = (bn["bid"] + bn["ask"]) / 2
                mx_mid = (mx["bid"] + mx["ask"]) / 2
                by_mid = (by["bid"] + by["ask"]) / 2

                bn_spread = (bn["ask"] - bn["bid"]) / bn_mid * 10000
                mx_spread = (mx["ask"] - mx["bid"]) / mx_mid * 10000
                by_spread = (by["ask"] - by["bid"]) / by_mid * 10000

                bn_mx_diff = (mx_mid - bn_mid) / bn_mid * 10000
                bn_by_diff = (by_mid - bn_mid) / bn_mid * 10000

                rows.append({
                    "round": rnd, "symbol": sym,
                    "bn_spread": bn_spread, "mx_spread": mx_spread, "by_spread": by_spread,
                    "bn_mx_diff": bn_mx_diff, "bn_by_diff": bn_by_diff,
                })

            print(f"  Round {rnd+1}/{ROUNDS} ({ts})")
            if rnd < ROUNDS - 1:
                await asyncio.sleep(INTERVAL_S)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("  No data collected!")
        return

    # Per-symbol summary
    print(f"\n{'='*120}")
    print(f"  PER-SYMBOL ({len(df)} observations)")
    print(f"{'='*120}")
    print(f"  {'Symbol':>14s} | {'BN spread':>10s} {'MX spread':>10s} {'BY spread':>10s} | "
          f"{'BN-MX mid':>10s} {'|BN-MX|':>8s} {'BN-BY mid':>10s} {'|BN-BY|':>8s}")
    print(f"  {'':>14s} | {'(bps)':>10s} {'(bps)':>10s} {'(bps)':>10s} | "
          f"{'(bps)':>10s} {'(bps)':>8s} {'(bps)':>10s} {'(bps)':>8s}")
    print(f"  {'-'*120}")

    for sym in symbols:
        sd = df[df["symbol"] == sym]
        if len(sd) == 0:
            continue
        print(f"  {sym:>14s} | "
              f"{sd['bn_spread'].mean():>10.2f} {sd['mx_spread'].mean():>10.2f} {sd['by_spread'].mean():>10.2f} | "
              f"{sd['bn_mx_diff'].mean():>+10.2f} {sd['bn_mx_diff'].abs().mean():>8.2f} "
              f"{sd['bn_by_diff'].mean():>+10.2f} {sd['bn_by_diff'].abs().mean():>8.2f}")

    # Aggregate
    print(f"\n{'='*120}")
    print(f"  AGGREGATE SUMMARY")
    print(f"{'='*120}")
    print(f"  Spreads (mean bps):  Binance={df['bn_spread'].mean():.2f}  MEXC={df['mx_spread'].mean():.2f}  Bybit={df['by_spread'].mean():.2f}")
    print(f"  Spreads (median):    Binance={df['bn_spread'].median():.2f}  MEXC={df['mx_spread'].median():.2f}  Bybit={df['by_spread'].median():.2f}")
    print(f"")
    print(f"  Mid-to-mid vs Binance:")
    print(f"    MEXC:  mean={df['bn_mx_diff'].mean():+.2f} bps  |mean|={df['bn_mx_diff'].abs().mean():.2f} bps  std={df['bn_mx_diff'].std():.2f} bps")
    print(f"    Bybit: mean={df['bn_by_diff'].mean():+.2f} bps  |mean|={df['bn_by_diff'].abs().mean():.2f} bps  std={df['bn_by_diff'].std():.2f} bps")
    print(f"")

    # Which exchange is closer to Binance?
    mx_avg = df['bn_mx_diff'].abs().mean()
    by_avg = df['bn_by_diff'].abs().mean()
    winner = "MEXC" if mx_avg < by_avg else "Bybit"
    print(f"  Closer to Binance: {winner} ({min(mx_avg, by_avg):.2f} bps vs {max(mx_avg, by_avg):.2f} bps)")
    print(f"{'='*120}")


if __name__ == "__main__":
    asyncio.run(main())
