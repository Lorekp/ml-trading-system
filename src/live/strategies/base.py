"""Strategy base and loader — dynamically loads strategies from config."""

from __future__ import annotations

import importlib
import logging
from typing import Any

from live.core.interfaces import Strategy

logger = logging.getLogger(__name__)


def load_strategy(strategy_id: str, config: dict) -> Strategy:
    """Load a strategy class from its dotted path in config.

    Config must have:
        class: "live.strategies.s8_sym15.S8Sym15Strategy"
        ... (strategy-specific params)

    Returns:
        Instantiated Strategy.
    """
    class_path = config.get("class", "")
    if not class_path:
        raise ValueError(f"Strategy '{strategy_id}' has no 'class' field")

    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    strategy = cls(strategy_id=strategy_id, config=config)

    if not isinstance(strategy, Strategy):
        logger.warning("Strategy '%s' does not fully implement Strategy protocol", strategy_id)

    logger.info("Loaded strategy: %s (%s)", strategy_id, class_path)
    return strategy


def load_strategies(strategies_config: dict[str, dict]) -> list[Strategy]:
    """Load all enabled strategies from config.

    Args:
        strategies_config: {strategy_id: {class: ..., enabled: true, ...}}

    Returns:
        List of instantiated Strategy objects.
    """
    strategies = []
    for sid, cfg in strategies_config.items():
        if not cfg.get("enabled", True):
            logger.info("Strategy '%s' disabled, skipping", sid)
            continue
        try:
            strategy = load_strategy(sid, cfg)
            strategies.append(strategy)
        except Exception as e:
            logger.error("Failed to load strategy '%s': %s", sid, e)
    return strategies
