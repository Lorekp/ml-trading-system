"""Live spot-vs-futures VPIN comparison.

Connect to BOTH Binance Spot and Binance Futures aggTrade WS streams
simultaneously, run them through identical _LiveBucketAccumulator
configs (1000 BTC bucket, 50 rolling), and emit periodic VPIN values.

Spot-vs-Futures BTC volume profile:
  - Futures BTCUSDT: ~50-200 BTC/min (during active hours)
  - Spot BTCUSDT: ~5-30 BTC/min (lower)

To compare meaningfully without waiting hours for a new bucket to
close, we use a SMALL bucket (50 BTC) so we get fresh imbalances
within minutes. The relative pattern is what matters, not the
absolute value at a tiny bucket size.
"""
import asyncio
import json
import time
import sys

sys.path.insert(0, "C:\Claude\ml-trading-system")
from deribit_paper.features.vpin_live import _LiveBucketAccumulator

import websockets


async def stream_to_acc(url, label, acc, stop_evt):
    msg_count = 0
    try:
        async with websockets.connect(url, ping_interval=20, ping_timeout=15) as ws:
            print(f"[{label}] connected", flush=True)
            while not stop_evt.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                except asyncio.TimeoutError:
                    continue
                try:
                    m = json.loads(raw)
                    q = float(m["q"])
                    seller_aggressor = bool(m["m"])
                except (KeyError, ValueError, TypeError, json.JSONDecodeError):
                    continue
                acc.process_trade(q, seller_aggressor)
                msg_count += 1
    except Exception as exc:
        print(f"[{label}] error: {exc}", flush=True)
    print(f"[{label}] disconnected; processed {msg_count} messages", flush=True)


async def reporter(spot_acc, fut_acc, stop_evt, duration):
    t0 = time.time()
    while not stop_evt.is_set() and (time.time() - t0) < duration:
        await asyncio.sleep(15.0)
        elapsed = time.time() - t0
        spot_v = spot_acc._latest_vpin
        fut_v = fut_acc._latest_vpin
        spot_b = spot_acc._buckets_closed
        fut_b = fut_acc._buckets_closed
        print(
            f"[t={elapsed:5.0f}s] spot: VPIN={spot_v if spot_v is not None else 'None':>6} "
            f"buckets={spot_b:3d} partial_vol={spot_acc._partial_vol:6.2f} | "
            f"fut: VPIN={fut_v if fut_v is not None else 'None':>6} "
            f"buckets={fut_b:3d} partial_vol={fut_acc._partial_vol:6.2f}",
            flush=True,
        )
    stop_evt.set()


async def main():
    BUCKET_BTC = 50.0   # small for fast feedback
    spot_acc = _LiveBucketAccumulator(bucket_volume_btc=BUCKET_BTC, rolling_window=20)
    fut_acc = _LiveBucketAccumulator(bucket_volume_btc=BUCKET_BTC, rolling_window=20)
    stop_evt = asyncio.Event()
    DURATION = 180.0  # 3 minutes

    spot_url = "wss://stream.binance.com:9443/ws/btcusdt@aggTrade"
    fut_url = "wss://fstream.binance.com/ws/btcusdt@aggTrade"

    await asyncio.gather(
        stream_to_acc(spot_url, "spot", spot_acc, stop_evt),
        stream_to_acc(fut_url, "fut", fut_acc, stop_evt),
        reporter(spot_acc, fut_acc, stop_evt, DURATION),
    )

    print("=== Final ===", flush=True)
    print(f"Spot: VPIN={spot_acc._latest_vpin}, buckets={spot_acc._buckets_closed}, imbalances={list(spot_acc._recent_imbalances)}", flush=True)
    print(f"Fut:  VPIN={fut_acc._latest_vpin}, buckets={fut_acc._buckets_closed}, imbalances={list(fut_acc._recent_imbalances)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
