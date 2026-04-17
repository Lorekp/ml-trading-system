"""Execution algorithm base — re-exports the protocol from core.interfaces."""

from live.core.interfaces import ExecutionAlgo, TradeResult

__all__ = ["ExecutionAlgo", "TradeResult"]
