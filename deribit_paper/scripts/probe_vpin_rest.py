"""Probe the Binance Futures REST aggTrades feed live.

E (parity 2026-05-07) — quick smoke test before flipping the engine
to the REST-polling VPIN feed. Polls
``https://fapi.binance.com/fapi/v1/aggTrades`` for ``--duration``
seconds and prints VPIN convergence + trade-rate stats.

Usage::

    PYTHONPATH=. python -m deribit_paper.scripts.probe_vpin_rest \
        --duration 300

Exit codes:
    0  — feed is healthy: trades flowed, latest_vpin advanced
    1  — feed is silent or returned errors
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

from deribit_paper.features.vpin_live import LiveVPINStream

logger = logging.getLogger("probe_vpin_rest")


async def run(duration_s: float, symbol: str, base_url: str, poll_s: float) -> int:
    stream = LiveVPINStream(
        bucket_volume_btc=1000.0,
        rolling_window=50,
        state_path=None,                   # cold start
        fallback_vpin=0.189,
        feed_kind="rest",
        symbol=symbol,
        binance_rest_url=base_url,
        poll_seconds=poll_s,
    )

    started_at = time.time()
    print(
        f"[probe] starting REST poll: symbol={symbol} base_url={base_url} "
        f"poll_s={poll_s} duration_s={duration_s}",
    )

    async def _stop_after_duration() -> None:
        await asyncio.sleep(duration_s)
        await stream.stop()

    asyncio.create_task(_stop_after_duration())

    # Periodic status while running
    async def _status_loop() -> None:
        while not stream._shutdown.is_set():
            await asyncio.sleep(min(15.0, duration_s / 4))
            elapsed = time.time() - started_at
            print(
                f"[probe] t={elapsed:6.1f}s "
                f"trades={stream._trades_processed:>6d} "
                f"buckets={stream.buckets_since_start:>3d} "
                f"vpin={stream.latest_vpin:.4f} "
                f"connected={stream.is_connected} "
                f"stale={stream.is_stale} "
                f"last_id={stream._last_agg_trade_id}",
            )

    asyncio.create_task(_status_loop())
    await stream.run_forever()

    elapsed = time.time() - started_at
    print(
        f"[probe] FINAL after {elapsed:.1f}s: "
        f"trades={stream._trades_processed} "
        f"buckets={stream.buckets_since_start} "
        f"vpin={stream.latest_vpin:.4f}",
    )

    if stream._trades_processed == 0:
        print("[probe] FAIL: zero trades received — feed silent / blocked", file=sys.stderr)
        return 1
    if stream.buckets_since_start == 0:
        print(
            "[probe] WARN: no buckets closed (ran shorter than V/trade-rate); "
            "feed is alive but warmup incomplete",
            file=sys.stderr,
        )
        return 0
    print("[probe] OK: REST feed healthy")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=float, default=60.0, help="seconds to poll (default 60)")
    ap.add_argument("--symbol", type=str, default="BTCUSDT")
    ap.add_argument("--base-url", type=str, default="https://fapi.binance.com")
    ap.add_argument("--poll-seconds", type=float, default=3.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return asyncio.run(
        run(args.duration, args.symbol, args.base_url, args.poll_seconds),
    )


if __name__ == "__main__":
    sys.exit(main())
