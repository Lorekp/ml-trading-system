"""CLI entry point for the Deribit paper/testnet engine.

The command can authenticate or smoke-test against testnet, run the
engine against testnet, or use public mainnet quotes with ``--paper``.
Every CLI mode is a dry run; this module has no real-order route.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

from deribit_paper.client import DeribitAuth, DeribitRESTClient
from deribit_paper.config import PaperConfig
from deribit_paper.risk.dry_run import get_dry_run, set_dry_run

# Exit codes for the wrapper script
EXIT_CLEAN = 0       # clean stop — wrapper should NOT restart
EXIT_SHUTDOWN = 42   # operator-requested permanent shutdown
# Any other non-zero exit → crash → wrapper should restart


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="deribit_paper",
        description="Deribit paper-trading engine for vol-arb strategy",
    )
    parser.add_argument(
        "--test-auth",
        action="store_true",
        help="Authenticate against Deribit testnet and print account info",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a ~60s connection + market-data smoke test, then exit",
    )
    parser.add_argument(
        "--run-paper",
        action="store_true",
        help="Start the full paper-trading engine (dry-run by default)",
    )
    parser.add_argument(
        "--duration-minutes",
        type=int,
        default=0,
        help="Stop the engine after this many minutes (0 = run forever)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO)",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help=(
            "Paper-trade against Deribit MAINNET quotes in forced dry-run "
            "mode; no real orders are submitted. No DERIBIT_PROD_* "
            "credentials required (public WS only via 100ms trade aggregates "
            "+ public ticker / index price endpoints). Fills simulated via "
            "the existing 3-tier IOC ladder against real mainnet quotes; "
            "trades log to vol_arb/output/live_trades_harness.parquet plus "
            "the SQLite store."
        ),
    )
    parser.add_argument(
        "--allow-stale-aligner",
        action="store_true",
        help=(
            "Override startup refusal when the aligner "
            "β* sidecar is ≥ 2 months stale. The lookup will fall back "
            "to the most-recent-prior month — predictions remain biased "
            "toward older regime, but the engine starts. Use only when "
            "a refreshed sidecar is unavailable (e.g. testnet smoke run)."
        ),
    )
    parser.add_argument(
        "--clear-kill-state",
        action="store_true",
        help=(
            "Clear a previously-tripped kill-switch state at startup. "
            "Without this flag, an engine that crashed/restarted while "
            "halted will RESTORE the halted state from SQLite and refuse "
            "to trade. Pass --clear-kill-state when the operator has "
            "verified the original trip cause is resolved (H1)."
        ),
    )
    parser.add_argument(
        "--allow-stale-kelly",
        action="store_true",
        help="Bypass kelly_mult sidecar staleness gate (H9).",
    )
    parser.add_argument(
        "--allow-stale-slip",
        action="store_true",
        help="Bypass bar_execution_quality sidecar staleness gate (H9).",
    )
    parser.add_argument(
        "--allow-stale-features",
        action="store_true",
        help="Bypass historical features parquet staleness gate (H9).",
    )
    parser.add_argument(
        "--allow-stale-svi-features",
        action="store_true",
        help="Bypass SVI features parquet staleness gate (H9).",
    )
    return parser.parse_args()


async def _run_test_auth(cfg: PaperConfig) -> int:
    """Authenticate against Deribit and print account info.

    This is the Phase 1 smoke test — it proves credentials work, the
    HTTP client is wired correctly, and the /private endpoints are
    accessible.  No orders are placed.
    """
    logger = logging.getLogger("test_auth")
    if not cfg.client_id or not cfg.client_secret:
        logger.error(
            "DERIBIT_TESTNET_CLIENT_ID / DERIBIT_TESTNET_CLIENT_SECRET not set in .env — "
            "create a Deribit testnet account at https://test.deribit.com and "
            "generate API credentials under Settings → API",
        )
        return 2

    logger.info("Connecting to %s", cfg.rest_base_url)
    auth = DeribitAuth(cfg.client_id, cfg.client_secret, cfg.rest_base_url)
    async with DeribitRESTClient(auth, cfg.rest_base_url) as rest:
        try:
            summary = await rest.get_account_summary(currency=cfg.currency)
        except Exception as exc:  # noqa: BLE001 - reported directly to user
            logger.error("Account summary request failed: %s", exc)
            await auth.close()
            return 3
        logger.info(
            "Auth OK. currency=%s equity=%.6f available=%.6f",
            summary.get("currency"),
            summary.get("equity", 0.0),
            summary.get("available_funds", 0.0),
        )
        logger.info(
            "margin_balance=%.6f maintenance_margin=%.6f portfolio_margining=%s",
            summary.get("margin_balance", 0.0),
            summary.get("maintenance_margin", 0.0),
            summary.get("portfolio_margining_enabled", False),
        )
    await auth.close()
    return 0


def main() -> None:
    args = _parse_args()
    _setup_logging(args.log_level)

    cfg = PaperConfig.from_env()
    # CLI safety invariant: environment/config values cannot disable dry-run.
    cfg.dry_run = True
    set_dry_run(True)
    if args.allow_stale_aligner:
        cfg.allow_stale_aligner = True
        logging.getLogger("cli").warning(
            "--allow-stale-aligner: bypassing β* staleness check "
            "(predictions may be biased toward older regime)",
        )
    if args.clear_kill_state:
        cfg.clear_kill_state = True
        logging.getLogger("cli").warning(
            "--clear-kill-state: any previously-tripped kill-switch will "
            "be cleared on engine startup",
        )
    if args.allow_stale_kelly:
        cfg.allow_stale_kelly = True
        logging.getLogger("cli").warning(
            "--allow-stale-kelly: bypassing Kelly K1 sidecar staleness "
            "check (sizing may be calibrated to a stale regime)",
        )
    if args.allow_stale_slip:
        cfg.allow_stale_slip = True
        logging.getLogger("cli").warning(
            "--allow-stale-slip: bypassing bar_execution_quality "
            "sidecar staleness check",
        )
    if args.allow_stale_features:
        cfg.allow_stale_features = True
        logging.getLogger("cli").warning(
            "--allow-stale-features: bypassing historical features "
            "parquet staleness check",
        )
    if args.allow_stale_svi_features:
        cfg.allow_stale_svi_features = True
        logging.getLogger("cli").warning(
            "--allow-stale-svi-features: bypassing SVI features "
            "parquet staleness check",
        )
    if args.paper:
        # --paper selects public mainnet quotes without authentication.
        # Dry-run has already been forced above and cannot be disabled by CLI.
        cfg.switch_to_prod(paper_mode=True)
        logging.getLogger("cli").info(
            "Paper mode: mainnet quotes, simulated fills via IOC ladder, "
            "no credentials required",
        )

    logging.getLogger("cli").info(
        "Starting deribit_paper venue=%s paper_mode=%s dry_run=%s",
        cfg.venue, cfg.paper_mode, get_dry_run(),
    )

    if args.test_auth:
        sys.exit(asyncio.run(_run_test_auth(cfg)))

    if args.smoke:
        sys.exit(asyncio.run(_run_smoke(cfg)))

    if args.run_paper:
        sys.exit(asyncio.run(_run_engine(cfg, args.duration_minutes)))

    logging.getLogger("cli").warning(
        "No action specified (--test-auth, --smoke, or --run-paper).",
    )


async def _run_smoke(cfg: PaperConfig) -> int:
    """60-second connection smoke test.

    - Authenticates with Deribit
    - Constructs the full engine via ``initialize()``
    - Subscribes to market data channels
    - Watches for 60 seconds, printing stats every 10s
    - Exits cleanly
    """
    from deribit_paper.engine import TradingEngine
    logger = logging.getLogger("smoke")
    logger.info("=" * 60)
    logger.info("  PAPER ENGINE SMOKE TEST (60s)")
    logger.info("=" * 60)

    engine = TradingEngine(cfg)
    try:
        await engine.initialize()
    except Exception as exc:
        logger.exception("Engine initialise failed: %s", exc)
        return 2

    # Subscribe to market data
    if engine.subscriber is not None:
        try:
            await engine.subscriber.start()
        except Exception as exc:
            logger.exception("Market data subscription failed: %s", exc)
            await engine.shutdown()
            return 3

    # Spawn WS dispatch in background (mirrors what run() does)
    ws_task = asyncio.create_task(
        engine.ws.dispatch_forever(), name="ws_dispatch",
    )

    try:
        for i in range(6):
            await asyncio.sleep(10.0)
            stats = engine.subscriber.stats if engine.subscriber else {}
            perp_len = len(engine.perp_buffer) if engine.perp_buffer else 0
            opt_len = len(engine.option_buffer) if engine.option_buffer else 0
            idx_latest = engine.index_buffer.latest() if engine.index_buffer else None
            logger.info(
                "[%ds] perp_buf=%d opt_buf=%d idx=%s stats=%s",
                (i + 1) * 10,
                perp_len, opt_len,
                f"${idx_latest:.2f}" if idx_latest else "None",
                stats,
            )

        logger.info("=" * 60)
        logger.info("  SMOKE TEST PASSED — connections healthy")
        logger.info("=" * 60)
        return 0
    finally:
        ws_task.cancel()
        try:
            await asyncio.wait_for(ws_task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        await engine.shutdown()


async def _run_engine(cfg: PaperConfig, duration_minutes: int) -> int:
    """Start the full paper-trading engine.

    Runs ``TradingEngine.run()`` until stopped by:
    - ``--duration-minutes`` expiry (if > 0)
    - Ctrl+C / SIGTERM / SIGHUP
    - Kill switch trip + engine.stop()

    Exit codes:
    - 0 = clean stop (wrapper should NOT restart)
    - 42 = operator-requested permanent shutdown
    - 1-41, 43+ = crash (wrapper should restart)
    """
    from deribit_paper.engine import TradingEngine
    logger = logging.getLogger("paper")
    logger.info("=" * 60)
    logger.info("  STARTING DERIBIT PAPER ENGINE")
    logger.info("  venue=%s  paper_mode=%s  dry_run=%s  duration=%s",
                cfg.venue, cfg.paper_mode, get_dry_run(),
                f"{duration_minutes}min" if duration_minutes > 0 else "forever")
    logger.info("=" * 60)

    engine = TradingEngine(cfg)
    try:
        await engine.initialize()
    except Exception as exc:
        logger.exception("Engine initialise failed: %s", exc)
        return 2

    # ── Signal handlers ──
    # SIGTERM/SIGHUP → clean stop (wrapper won't restart).
    # SIGUSR1 → drain mode (finish open trades then stop).
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGTERM, engine.stop)
    except (NotImplementedError, OSError):
        pass  # Windows doesn't support add_signal_handler for all signals
    try:
        loop.add_signal_handler(signal.SIGHUP, engine.stop)
    except (NotImplementedError, OSError, AttributeError):
        pass  # SIGHUP not available on Windows
    try:
        loop.add_signal_handler(signal.SIGUSR1, engine.drain)
    except (NotImplementedError, OSError, AttributeError):
        pass  # SIGUSR1 not available on Windows
    try:
        loop.add_signal_handler(signal.SIGUSR2, engine.pause)
    except (NotImplementedError, OSError, AttributeError):
        pass  # SIGUSR2 not available on Windows

    # Schedule the stop signal if a duration is set
    async def _stop_after_duration():
        await asyncio.sleep(duration_minutes * 60.0)
        logger.info("Duration %dmin reached — stopping engine", duration_minutes)
        engine.stop()

    stop_task = None
    if duration_minutes > 0:
        stop_task = asyncio.create_task(_stop_after_duration(), name="duration_timer")

    try:
        await engine.run()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — stopping engine")
        engine.stop()
    except Exception as exc:
        logger.exception("engine.run() raised: %s", exc)
        return 4
    finally:
        if stop_task is not None and not stop_task.done():
            stop_task.cancel()
        await engine.shutdown()

    logger.info("Engine stopped cleanly. Stats: %s", engine.stats)
    return EXIT_CLEAN


if __name__ == "__main__":
    main()
