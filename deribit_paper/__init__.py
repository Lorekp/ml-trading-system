"""Deribit paper-trading integration for the vol-arb strategy.

Standalone module that runs the complete vol-arb loop against Deribit
testnet (``test.deribit.com``):

- Market data via Deribit WebSocket (option + perp trades, index price)
- Feature pipeline reusing the batch code (TSRV, SVI, TransVol, LightGBM)
- DRL delta hedging via the production ``sac_v3_300k_final`` policy
- Order execution on Deribit options + perps (same venue, same collateral)
- Position tracking, risk limits, kill switches, Telegram alerts

Isolated by design from ``src/live/`` (the Bybit-targeted framework) so
paper-trading experiments don't contaminate the production Bybit path.

Entry point:
    python -m deribit_paper --dry-run --duration 24h

See ``C:\\Users\\laure\\.claude\\plans\\flickering-wandering-locket.md`` for
the full plan, or Stage 17 in ``TSRV/PIPELINE_DOCUMENTATION.md`` once
paper validation is complete.
"""

__version__ = "0.1.0"
