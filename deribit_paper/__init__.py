"""Deribit paper-trading integration for the vol-arb strategy.

Standalone module that runs the vol-arb loop against Deribit testnet or
public mainnet quotes in paper mode:

- Market data via Deribit WebSocket (option + perp trades, index price)
- Feature pipeline reusing the batch code (TSRV, SVI, TransVol, LightGBM)
- DRL delta-hedging simulation through the configured policy
- Simulated option/perpetual execution; the CLI never submits real orders
- Position tracking, risk limits, kill switches, Telegram alerts

Isolated by design from other exchange integrations.

Entry point:
    python -m deribit_paper --run-paper --duration-minutes 60
    python -m deribit_paper --paper --run-paper --duration-minutes 60

The application interface is intentionally limited to paper/testnet use.
"""

__version__ = "0.1.0"
