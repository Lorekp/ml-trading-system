"""Entry point for the modular trading engine.

Wires together: strategies (from config) + exchange + data provider +
execution algo + state DB + notifier → engine.

Usage:
    PYTHONPATH=src python -m live --config configs/bybit_demo.yaml
"""

import argparse
import asyncio
import logging
import sys

from live.config import load_live_config
from live.core.engine import TradingEngine
from live.core.state import StateDB
from live.execution.mkrtk60 import MkrTkr60
from live.execution.market import MarketExecution
from live.execution.paper import PaperExecution
from live.execution.calibrated import CalibratedPaperExecution
from live.notifier import TelegramNotifier
from live.strategies.base import load_strategies

# Exchange registry — add new exchanges here
_EXCHANGE_FACTORIES = {}

def _lazy_load_exchanges():
    """Import exchange adapters on first use to avoid circular imports."""
    if _EXCHANGE_FACTORIES:
        return
    from live.exchange.bybit import BybitExchange
    from live.exchange.binance import BinanceFuturesData
    _EXCHANGE_FACTORIES["bybit"] = BybitExchange.from_config
    _EXCHANGE_FACTORIES["binance"] = lambda cfg: BinanceFuturesData(
        rate_limit_per_sec=getattr(cfg, "rate_limit_per_sec", 5),
        timeout_s=getattr(cfg, "timeout_s", 10.0),
    )


def _create_adapter(name, config_obj):
    """Create a data provider or exchange adapter by name."""
    _lazy_load_exchanges()
    factory = _EXCHANGE_FACTORIES.get(name)
    if factory is None:
        raise ValueError(
            f"Unknown adapter '{name}'. Available: {list(_EXCHANGE_FACTORIES.keys())}")
    return factory(config_obj)


def build_engine(config):
    """Wire all components from config.

    Supports two modes:
    - Unified: exchange serves as both data provider and order executor
      (when config.data.name is empty or matches config.exchange.name)
    - Split: separate data provider and exchange (e.g. Binance data + Bybit orders)
      (when config.data.name differs from config.exchange.name)
    """
    logger = logging.getLogger("live")

    # Exchange adapter (orders, positions, account)
    exchange = _create_adapter(config.exchange.name, config.exchange)

    # Data provider — separate or unified
    data_name = config.data.name
    if data_name and data_name != config.exchange.name:
        data_provider = _create_adapter(data_name, config.data)
        logger.info("Cross-exchange mode: data=%s, execution=%s", data_name, config.exchange.name)
    else:
        data_provider = exchange
        logger.info("Unified mode: %s for data + execution", config.exchange.name)

    # State DB
    state = StateDB(config.db_path)

    # Execution algorithm
    paper_mode = getattr(config, 'paper', False) or not getattr(config.exchange, 'api_key', '')
    if paper_mode:
        exec_cfg = config.execution
        if exec_cfg.type == "calibrated":
            execution = CalibratedPaperExecution(
                exchange, state,
                vip_tier=(exec_cfg.vip_tier_name,
                          exec_cfg.vip_maker_bps,
                          exec_cfg.vip_taker_bps),
                calibration_path=exec_cfg.calibration_path,
                daily_vol_ttl_s=exec_cfg.daily_vol_ttl_s,
            )
            logger.info("Paper execution mode: CALIBRATED (tier=%s)", exec_cfg.vip_tier_name)
        else:
            execution = PaperExecution(exchange, state)
            logger.info("Paper execution mode: flat_2bps (legacy)")
    elif config.mkrtk60.maker_wait_s > 0:
        execution = MkrTkr60.from_config(exchange, state, config.mkrtk60)
    else:
        execution = MarketExecution(exchange, state)

    # Notifier
    notifier = TelegramNotifier.from_config(config.telegram)

    # Strategies (loaded dynamically from config)
    strategies = load_strategies(config.strategies)
    if not strategies:
        logger.error("No strategies loaded! Check config.")
        sys.exit(1)

    # Build engine
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
        paper_mode=paper_mode,
    )

    return engine, state


def main():
    parser = argparse.ArgumentParser(description="Modular trading engine")
    parser.add_argument("--config", default="configs/bybit_demo.yaml",
                        help="Path to config YAML")
    parser.add_argument("--capital", type=float, default=None,
                        help="Override account capital (e.g. after HyroTrader scaling)")
    args = parser.parse_args()

    config = load_live_config(args.config)
    if args.capital:
        config.capital = args.capital

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("live")

    engine, state = build_engine(config)

    loop = asyncio.new_event_loop()

    async def run():
        await state.initialize()
        await engine.initialize()
        try:
            await engine.run()
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            await engine.shutdown()

    try:
        loop.run_until_complete(run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        loop.run_until_complete(engine.shutdown())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
