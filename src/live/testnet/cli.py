"""Testnet execution driver — CLI + wiring.

Wires ``BinanceFuturesTestnet`` (DataProvider + ExchangeAdapter) and
``TestnetExecutor`` (ExecutionAlgo) into the production ``TradingEngine``
without touching ``src/live/core/engine.py`` or ``src/live/strategies/s8_sym15.py``.

This is a parity harness in code form: every code path that the engine
would exercise against ``BinanceFuturesData`` + ``MkrTkr60`` is exercised
against ``BinanceFuturesTestnet`` + ``TestnetExecutor`` instead. The only
delta is the wire endpoint + the post-only wait semantics.

Modes
-----
* ``paper``        — falls back to ``CalibratedPaperExecution`` (no testnet
  creds needed).  Useful for smoke runs that exercise the harness plumbing
  end-to-end without hitting testnet.
* ``live-testnet`` — real HMAC-signed orders to
  ``https://testnet.binancefuture.com``.  Requires
  ``BINANCE_TESTNET_API_KEY`` / ``BINANCE_TESTNET_API_SECRET``.
Usage
-----
::

    PYTHONPATH=src .venv/Scripts/python.exe -m live.testnet.cli \\
        --config configs/paper_binance.yaml \\
        --mode live-testnet \\
        --duration 7d \\
        --fills-out vol_arb/output/S5_TESTNET_FILLS.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

TESTNET_BASE_URL = "https://testnet.binancefuture.com"


def _setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("live.testnet")


def _parse_duration(s: str) -> timedelta:
    """Parse ``7d``, ``12h``, ``30m`` style durations."""
    s = s.strip().lower()
    if s.endswith("d"):
        return timedelta(days=int(s[:-1]))
    if s.endswith("h"):
        return timedelta(hours=int(s[:-1]))
    if s.endswith("m"):
        return timedelta(minutes=int(s[:-1]))
    return timedelta(hours=int(s))


def _build_testnet_components(
    config, mode: str, fills_out: Optional[Path],
):
    """Construct (data_provider, exchange, execution) for the chosen mode."""
    from live.testnet.binance_testnet import BinanceFuturesTestnet
    from live.testnet.testnet_executor import TestnetExecutor

    api_key, api_secret = _credentials_for_mode(mode)

    exchange = BinanceFuturesTestnet(
        api_key=api_key, api_secret=api_secret, base_url=TESTNET_BASE_URL,
        rate_limit_per_sec=getattr(config.exchange, "rate_limit_per_sec", 5),
        timeout_s=getattr(config.exchange, "timeout_s", 10.0),
    )

    # Execution algo.  ``paper`` mode = paper fallback in TestnetExecutor
    # itself (so we still produce FillRecord + reconciliation data) — this
    # is the parity baseline.  ``live-testnet`` = real testnet orders.
    state = None  # StateDB injected by the caller
    mk = config.mkrtk60
    execution = TestnetExecutor(
        exchange, state,
        maker_wait_s=getattr(mk, "maker_wait_s", 0),
        poll_interval_s=getattr(mk, "poll_interval_s", 5.0),
        max_retries=getattr(mk, "max_retries", 2),
        vip_maker_bps=config.execution.vip_maker_bps,
        vip_taker_bps=config.execution.vip_taker_bps,
        fills_path=fills_out,
    )

    # For ``paper`` mode, the executor falls back internally; we still
    # surface that to the caller.
    return exchange, exchange, execution


def _credentials_for_mode(mode: str) -> tuple[str, str]:
    """Return credentials without allowing a production endpoint.

    Paper mode deliberately ignores credentials that happen to be present in
    the environment. Live-testnet mode reads only Binance testnet credentials.
    """
    if mode == "paper":
        return "", ""
    if mode == "live-testnet":
        return (
            os.environ.get("BINANCE_TESTNET_API_KEY", ""),
            os.environ.get("BINANCE_TESTNET_API_SECRET", ""),
        )
    raise ValueError(f"unsupported testnet mode: {mode!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Binance testnet driver")
    parser.add_argument("--config", default="configs/paper_binance.yaml",
                        help="Path to the paper/testnet engine config YAML.")
    parser.add_argument("--mode", default="paper",
                        choices=["paper", "live-testnet"],
                        help="Execution mode (paper simulation or Binance testnet).")
    parser.add_argument("--duration", default="7d",
                        help="How long to run the engine (e.g. 7d, 12h, 30m).")
    parser.add_argument("--capital", type=float, default=None,
                        help="Override starting capital (e.g. 250000).")
    parser.add_argument("--fills-out",
                        default="vol_arb/output/S5_TESTNET_FILLS.jsonl",
                        help="Path for the per-fill JSONL log.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logger = _setup_logging(args.log_level)
    logger.info("Testnet driver: mode=%s config=%s duration=%s",
                args.mode, args.config, args.duration)

    # Load the shared paper/testnet configuration.
    from live.config import load_live_config
    config = load_live_config(Path(args.config))
    if args.capital:
        config.capital = args.capital

    fills_out = Path(args.fills_out)
    fills_out.parent.mkdir(parents=True, exist_ok=True)

    # Build the testnet components.
    exchange, data_provider, execution = _build_testnet_components(
        config, args.mode, fills_out)

    # State DB + notifier + strategies.
    from live.core.state import StateDB
    from live.notifier import TelegramNotifier
    from live.strategies.base import load_strategies
    state = StateDB(config.db_path)
    notifier = TelegramNotifier.from_config(config.telegram)
    strategies = load_strategies(config.strategies)
    if not strategies:
        logger.error("No strategies loaded — check config.strategies section.")
        return 3

    # Engine.
    from live.core.engine import TradingEngine
    engine = TradingEngine(
        strategies=strategies,
        data_provider=data_provider,
        exchange=exchange,
        execution=execution,
        state=state,
        notifier=notifier,
        starting_capital=config.capital,
        bar_interval_m=config.scheduler.bar_interval_m,
        poll_offset_s=config.scheduler.poll_offset_s,
        leverage=config.leverage,
        max_signals_per_bar=config.max_signals_per_bar,
        max_risk_per_trade=config.max_risk_per_trade,
        sl_cap=config.sl_cap,
        portfolio_config=config.portfolio,
        monitor_config=config.monitor,
        reconciler_config=config.reconciler,
        data_validator_config=config.data_validator,
        vol_overlay_config=config.vol_overlay,
        quote_currency=config.exchange.quote_currency,
        tf_lookbacks={tf.tf: tf.lookback for tf in config.full5},
        paper_mode=(args.mode == "paper"),
    )

    duration = _parse_duration(args.duration)
    end_at = datetime.now(tz=timezone.utc) + duration
    logger.info("Driver will stop at %s (after %s)", end_at.isoformat(), duration)

    async def run() -> int:
        await state.initialize()
        await engine.initialize()
        # NEW (B-029/B-031): wrap engine task in try/finally that ALWAYS
        # exports fills and shuts down — even on KeyboardInterrupt / mid-bar
        # crash. The previous code only exported in the happy path; Ctrl+C
        # or engine crash lost the in-memory fills.
        rc = 0
        try:
            # Run the engine until the duration elapses, then ask it to stop.
            stop = asyncio.get_event_loop().time() + duration.total_seconds()
            engine_task = asyncio.create_task(engine.run())
            try:
                while asyncio.get_event_loop().time() < stop:
                    await asyncio.sleep(1.0)
                logger.info("Duration elapsed — requesting engine stop")
                engine_task.cancel()
                try:
                    await engine_task
                except asyncio.CancelledError:
                    pass
                return 0
            except KeyboardInterrupt:
                logger.warning("KeyboardInterrupt — exporting fills and shutting down")
                engine_task.cancel()
                try:
                    await engine_task
                except (asyncio.CancelledError, Exception):
                    pass
                rc = 130
        finally:
            # Dump fills for the reconciler (atomic write via .tmp + os.replace
            # is inside TestnetExecutor.export_fills — B-031).
            if hasattr(execution, "export_fills"):
                try:
                    execution.export_fills(fills_out)
                    logger.info("Wrote fills → %s", fills_out)
                except Exception as e:
                    logger.error("export_fills failed: %s", e)
            try:
                await engine.shutdown()
            except Exception as e:
                logger.error("engine.shutdown failed: %s", e)
            try:
                await exchange.close()
            except Exception as e:
                logger.error("exchange.close failed: %s", e)
        return rc

    try:
        rc = asyncio.run(run())
    except KeyboardInterrupt:
        rc = 130
    logger.info("Driver exiting rc=%d", rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
