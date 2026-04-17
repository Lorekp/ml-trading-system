"""Execution layer: order placement, straddle lifecycle, perp hedging.

Five submodules, each focused and minimal:

- :mod:`deribit_paper.execution.amounts` — Deribit amount convention
  helpers (BTC for options, USD notional for perps, rounding to
  minimum increments).
- :mod:`deribit_paper.execution.idempotency` — client_order_id
  generation + recent-ID dedup to survive reconnects.
- :mod:`deribit_paper.execution.order_manager` — thin async wrapper
  over :class:`DeribitRESTClient` adding retries, idempotent sends,
  and fill-polling via ``get_order_state``.
- :mod:`deribit_paper.execution.perp_hedger` — intra-bar DRL hedge
  loop.  Every 90s for 10 ticks per bar, queries
  :class:`LiveDRLHedger.step` and submits a perp order for the delta.
- :mod:`deribit_paper.execution.straddle_flow` — the top-level
  coordinator: size position → open both legs → spawn perp hedger →
  wait holding period → close both legs → flatten hedge.
"""

from deribit_paper.execution.amounts import (
    option_amount_btc,
    perp_amount_usd,
    round_to_increment,
)
from deribit_paper.execution.idempotency import IdempotencyGuard, new_client_order_id
from deribit_paper.execution.order_manager import OrderManager, OrderResult
from deribit_paper.execution.perp_hedger import PerpHedger
from deribit_paper.execution.straddle_flow import StraddleFlow, TradeOutcome

__all__ = [
    "option_amount_btc",
    "perp_amount_usd",
    "round_to_increment",
    "IdempotencyGuard",
    "new_client_order_id",
    "OrderManager",
    "OrderResult",
    "PerpHedger",
    "StraddleFlow",
    "TradeOutcome",
]
