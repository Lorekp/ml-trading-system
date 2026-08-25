#!/usr/bin/env python
"""Measure the real cost of using Binance data but trading on Bybit.

Fetches simultaneous snapshots from both exchanges and computes:
1. Price difference (mid-to-mid) — the execution gap
2. Spread difference — which exchange has tighter spreads
3. Volume ratio — liquidity comparison
4. Simulated slippage: if you generate signal on Binance close, what do you fill at on Bybit?

Runs multiple rounds over ~10 minutes to capture variance.

Usage:
    PYTHONPATH="src;." python experiments/cross_exchange_cost.py
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
    "SUIUSDT", "1000PEPEUSDT", "TAOUSDT", "AVAXUSDT",
]

ROUNDS = 20        # snapshot rounds
INTERVAL_S = 30    # seconds between rounds


async def fetch_binance_bbo(session, symbol):
    """Binance futures best bid/ask."""
    url = f"https://fapi.binance.com/fapi/v1/ticker/bookTicker?symbol={symbol}"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            d = await resp.json()
            return {
                "bid": float(d.get("bidPrice", 0)),
                "ask": float(d.get("askPrice", 0)),
            }
    except Exception:
        return None


async def fetch_bybit_bbo(session, symbol):
    """Bybit futures best bid/ask."""
    url = f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol}"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            d = await resp.json()
            tickers = d.get("result", {}).get("list", [])
            if tickers:
                t = tickers[0]
                return {
                    "bid": float(t.get("bid1Price", 0)),
                    "ask": float(t.get("ask1Price", 0)),
                }
    except Exception:
        pass
    return None


async def fetch_binance_kline_close(session, symbol):
    """Binance last completed 15m candle close (what the signal model sees)."""
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=15m&limit=2"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            data = await resp.json()
            if isinstance(data, list) and len(data) >= 2:
                # Second-to-last is the last COMPLETED candle
                return float(data[-2][4])
    except Exception:
        pass
    return None


async def snapshot_one(session, symbol):
    """Take simultaneous snapshot from both exchanges."""
    bn_bbo, by_bbo, bn_close = await asyncio.gather(
        fetch_binance_bbo(session, symbol),
        fetch_bybit_bbo(session, symbol),
        fetch_binance_kline_close(session, symbol),
    )
    return bn_bbo, by_bbo, bn_close


async def main():
    print("=" * 110)
    print("  CROSS-EXCHANGE COST: Binance Signal -> Bybit Execution")
    print(f"  {ROUNDS} rounds x {len(SYMBOLS)} symbols, {INTERVAL_S}s apart")
    print("=" * 110)

    all_rows = []

    async with aiohttp.ClientSession() as session:
        for rnd in range(ROUNDS):
            t0 = time.monotonic()
            ts = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")

            tasks = [snapshot_one(session, sym) for sym in SYMBOLS]
            results = await asyncio.gather(*tasks)

            for sym, (bn_bbo, by_bbo, bn_close) in zip(SYMBOLS, results):
                if bn_bbo is None or by_bbo is None:
                    continue
                if bn_bbo["bid"] <= 0 or by_bbo["bid"] <= 0:
                    continue

                bn_mid = (bn_bbo["bid"] + bn_bbo["ask"]) / 2
                by_mid = (by_bbo["bid"] + by_bbo["ask"]) / 2
                bn_spread = (bn_bbo["ask"] - bn_bbo["bid"]) / bn_mid * 10000
                by_spread = (by_bbo["ask"] - by_bbo["bid"]) / by_mid * 10000

                mid_diff_bps = (by_mid - bn_mid) / bn_mid * 10000

                # Simulated cross-exchange slippage:
                # Signal generated on Binance close -> fill on Bybit
                # Long: fill at Bybit ask (you buy), signal at Binance close
                # Short: fill at Bybit bid (you sell), signal at Binance close
                if bn_close and bn_close > 0:
                    long_slip = (by_bbo["ask"] - bn_close) / bn_close * 10000
                    short_slip = (bn_close - by_bbo["bid"]) / bn_close * 10000
                else:
                    long_slip = short_slip = np.nan

                all_rows.append({
                    "round": rnd, "symbol": sym, "time": ts,
                    "bn_mid": bn_mid, "by_mid": by_mid,
                    "bn_spread_bps": bn_spread, "by_spread_bps": by_spread,
                    "mid_diff_bps": mid_diff_bps,
                    "long_slip_bps": long_slip, "short_slip_bps": short_slip,
                })

            elapsed = time.monotonic() - t0
            if rnd < ROUNDS - 1:
                wait = max(INTERVAL_S - elapsed, 1)
                print(f"  Round {rnd+1}/{ROUNDS} ({ts}) ... {len([r for r in results if r[0]])} symbols OK, "
                      f"next in {wait:.0f}s")
                await asyncio.sleep(wait)
            else:
                print(f"  Round {rnd+1}/{ROUNDS} ({ts}) ... done")

    df = pd.DataFrame(all_rows)

    # Per-symbol summary
    print(f"\n{'='*110}")
    print(f"  PER-SYMBOL SUMMARY ({len(df)} observations)")
    print(f"{'='*110}")
    print(f"\n  {'Symbol':>14s} | {'BN spread':>10s} {'BY spread':>10s} | {'Mid diff':>10s} {'|Mid diff|':>10s} | "
          f"{'Long slip':>10s} {'Short slip':>10s} {'|Avg slip|':>10s}")
    print(f"  {'':>14s} | {'(bps)':>10s} {'(bps)':>10s} | {'(bps)':>10s} {'(bps)':>10s} | "
          f"{'(bps)':>10s} {'(bps)':>10s} {'(bps)':>10s}")
    print(f"  {'-'*110}")

    for sym in SYMBOLS:
        sd = df[df["symbol"] == sym]
        if len(sd) == 0:
            continue
        print(f"  {sym:>14s} | "
              f"{sd['bn_spread_bps'].mean():>10.2f} {sd['by_spread_bps'].mean():>10.2f} | "
              f"{sd['mid_diff_bps'].mean():>+10.2f} {sd['mid_diff_bps'].abs().mean():>10.2f} | "
              f"{sd['long_slip_bps'].mean():>+10.2f} {sd['short_slip_bps'].mean():>+10.2f} "
              f"{(sd['long_slip_bps'].abs().mean() + sd['short_slip_bps'].abs().mean())/2:>10.2f}")

    # Aggregate
    print(f"\n{'='*110}")
    print(f"  AGGREGATE")
    print(f"{'='*110}")
    print(f"  Binance spread:     mean={df['bn_spread_bps'].mean():.2f} bps  median={df['bn_spread_bps'].median():.2f} bps")
    print(f"  Bybit spread:       mean={df['by_spread_bps'].mean():.2f} bps  median={df['by_spread_bps'].median():.2f} bps")
    print(f"  Spread gap (BY-BN): mean={df['by_spread_bps'].mean() - df['bn_spread_bps'].mean():.2f} bps")
    print(f"")
    print(f"  Mid-to-mid diff:    mean={df['mid_diff_bps'].mean():+.2f} bps  |mean|={df['mid_diff_bps'].abs().mean():.2f} bps  std={df['mid_diff_bps'].std():.2f} bps")
    print(f"")
    long_s = df['long_slip_bps'].dropna()
    short_s = df['short_slip_bps'].dropna()
    avg_slip = (long_s.abs().mean() + short_s.abs().mean()) / 2
    print(f"  Cross-exchange slippage (Binance signal -> Bybit fill):")
    print(f"    Long entry:   mean={long_s.mean():+.2f} bps  std={long_s.std():.2f} bps")
    print(f"    Short entry:  mean={short_s.mean():+.2f} bps  std={short_s.std():.2f} bps")
    print(f"    Avg |slip|:   {avg_slip:.2f} bps")
    print(f"    p95 |slip|:   {max(long_s.abs().quantile(0.95), short_s.abs().quantile(0.95)):.2f} bps")

    # Cost comparison: same-exchange vs cross-exchange
    print(f"\n{'='*110}")
    print(f"  COST COMPARISON (per trade, one-way)")
    print(f"{'='*110}")
    print(f"  Scenario A: Binance signal + Binance execution")
    print(f"    Half-spread:    {df['bn_spread_bps'].mean()/2:.2f} bps")
    print(f"    Taker fee:      5.00 bps (VIP0) / 4.50 bps (BNB discount)")
    print(f"    Total:          {df['bn_spread_bps'].mean()/2 + 5.0:.2f} bps / {df['bn_spread_bps'].mean()/2 + 4.5:.2f} bps")
    print(f"")
    print(f"  Scenario B: Binance signal + Bybit execution")
    print(f"    Cross-exch slip: {avg_slip:.2f} bps")
    print(f"    Half-spread:     {df['by_spread_bps'].mean()/2:.2f} bps (on Bybit)")
    print(f"    Taker fee:       5.50 bps (VIP0)")
    print(f"    Total:           {avg_slip + df['by_spread_bps'].mean()/2 + 5.5:.2f} bps")
    print(f"")
    bn_total = df['bn_spread_bps'].mean()/2 + 4.5  # with BNB discount
    by_total = avg_slip + df['by_spread_bps'].mean()/2 + 5.5
    print(f"  Extra cost of cross-exchange: {by_total - bn_total:.2f} bps/trade")
    print(f"  On 38K trades/yr at $200K avg: ~${38000 * 200000 * (by_total - bn_total) / 10000 / 100:,.0f}/yr")
    print(f"{'='*110}")


if __name__ == "__main__":
    asyncio.run(main())
