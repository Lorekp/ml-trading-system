#!/usr/bin/env python
"""Collect Hyperliquid BBO data via WebSocket (forward collection).

Subscribes to BBO updates for all eligible HL symbols and saves
daily parquets matching the Binance bookTicker format.

Run as a daemon on homeserver to accumulate data from today forward.

Usage:
    PYTHONPATH=src python scripts/collect_hl_bbo_ws.py [--coins BTC,ETH,SOL]
"""
import os, sys, time, json, signal, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("data/hl-bbo")
ELIGIBLE_FILE = Path("data/eligible_symbols.json")
WS_URL = "wss://api.hyperliquid.xyz/ws"
FLUSH_INTERVAL = 300  # Save buffer every 5 minutes
running = True


def signal_handler(sig, frame):
    global running
    print("\n  Shutting down gracefully...")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def get_coins():
    """Get list of HL coins to subscribe to."""
    if "--coins" in sys.argv:
        idx = sys.argv.index("--coins")
        return sys.argv[idx + 1].split(",")

    if ELIGIBLE_FILE.exists():
        with open(ELIGIBLE_FILE) as f:
            eligible = json.load(f)
        return [e["hl_coin"] for e in eligible if e["hl_coin"]]

    # Default: top 20
    return ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "AVAX",
            "LINK", "DOT", "SUI", "APT", "NEAR", "UNI", "AAVE",
            "LTC", "BCH", "FIL", "ARB", "OP"]


def coin_to_symbol(coin):
    """Convert HL coin name to Binance-style symbol for file paths."""
    if coin.startswith("k"):
        return f"1000{coin[1:]}USDT"
    return f"{coin}USDT"


def save_buffer(buffer, date_str):
    """Save accumulated BBO records to daily parquets."""
    saved = 0
    for coin, records in buffer.items():
        if not records:
            continue

        sym = coin_to_symbol(coin)
        out_dir = OUTPUT_DIR / sym
        out_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(records, columns=[
            "best_bid_price", "best_bid_qty",
            "best_ask_price", "best_ask_qty",
            "transaction_time"
        ])
        df["best_bid_price"] = df["best_bid_price"].astype(np.float32)
        df["best_bid_qty"] = df["best_bid_qty"].astype(np.float32)
        df["best_ask_price"] = df["best_ask_price"].astype(np.float32)
        df["best_ask_qty"] = df["best_ask_qty"].astype(np.float32)
        df["transaction_time"] = df["transaction_time"].astype(np.int64)

        path = out_dir / f"{date_str}.parquet"

        # Append to existing if same day
        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df], ignore_index=True)
            df = df.sort_values("transaction_time").reset_index(drop=True)

        df.to_parquet(path, index=False)
        saved += len(records)

    return saved


def run_collector(coins):
    """Main collection loop with WebSocket."""
    import websockets
    import asyncio

    buffer = defaultdict(list)
    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    last_flush = time.time()
    n_total = 0

    async def collect():
        global running
        nonlocal buffer, current_date, last_flush, n_total

        reconnect_delay = 1.0
        while running:
            try:
                async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=30) as ws:
                    print(f"  Connected to {WS_URL}")
                    reconnect_delay = 1.0

                    # Subscribe to BBO for all coins
                    for coin in coins:
                        sub = {
                            "method": "subscribe",
                            "subscription": {"type": "l2Book", "coin": coin}
                        }
                        await ws.send(json.dumps(sub))

                    print(f"  Subscribed to {len(coins)} coins")

                    async for message in ws:
                        if not running:
                            break

                        try:
                            data = json.loads(message)
                            if data.get("channel") != "l2Book":
                                continue

                            inner = data.get("data", {})
                            coin = inner.get("coin")
                            ts_ms = inner.get("time")
                            levels = inner.get("levels", [[], []])

                            if not coin or not ts_ms or not levels[0] or not levels[1]:
                                continue

                            bid = levels[0][0]
                            ask = levels[1][0]

                            record = (
                                float(bid["px"]),
                                float(bid["sz"]),
                                float(ask["px"]),
                                float(ask["sz"]),
                                int(ts_ms),
                            )
                            buffer[coin].append(record)
                            n_total += 1

                            # Check for date rollover
                            now_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                            if now_date != current_date:
                                saved = save_buffer(buffer, current_date)
                                print(f"  Day rollover: saved {saved} records for {current_date}")
                                buffer = defaultdict(list)
                                current_date = now_date
                                last_flush = time.time()

                            # Periodic flush
                            if time.time() - last_flush > FLUSH_INTERVAL:
                                saved = save_buffer(buffer, current_date)
                                n_coins = sum(1 for v in buffer.values() if v)
                                print(f"  Flushed {saved} records ({n_coins} coins, "
                                      f"{n_total:,} total)", flush=True)
                                buffer = defaultdict(list)
                                last_flush = time.time()

                        except (KeyError, ValueError, json.JSONDecodeError):
                            continue

            except Exception as e:
                if running:
                    print(f"  WS disconnected: {e}, reconnecting in {reconnect_delay:.0f}s")
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, 30.0)

        # Final flush
        if buffer:
            saved = save_buffer(buffer, current_date)
            print(f"  Final flush: {saved} records")

    asyncio.run(collect())


def main():
    print("=" * 70)
    print("  HYPERLIQUID BBO COLLECTOR (WebSocket)")
    print("=" * 70)

    coins = get_coins()
    print(f"  Coins: {len(coins)}")
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"  Flush interval: {FLUSH_INTERVAL}s")
    print()

    try:
        import websockets
    except ImportError:
        print("  ERROR: pip install websockets")
        sys.exit(1)

    run_collector(coins)
    print("  Collector stopped.")


if __name__ == "__main__":
    main()
