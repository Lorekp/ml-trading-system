"""Binance Futures Testnet execution harness.

Testnet URL: https://testnet.binancefuture.com (HMAC-signed).
Real fills and fees against testnet funds, with no real-money endpoint. It
reuses the directional engine's configuration shape so execution differences
can be studied without a separate strategy implementation.

The harness reuses ``TradingEngine`` and ``S8Sym15Strategy``
unchanged — only the exchange adapter and execution algo are swapped for the
testnet-flavoured versions here. This is infrastructure parity, not evidence
that the historical strategy is valid.

Modules
-------
- ``binance_testnet``: USDT-M Futures Testnet data + Exchange adapter
  (HMAC-signed). Implements the existing DataProvider + ExchangeAdapter protocols
  so the engine treats it as drop-in for ``BinanceFuturesData``.
- ``testnet_executor``: Async executor that posts real orders against the testnet
  endpoint (MkrTkr60-style maker-wait fallback to taker).  Records every fill
  for reconciliation.
- ``reconciler``: Per-run reconciliation report — testnet net PnL vs the
  backtest's assumptions (MkrTkr60 + VIP-tier costs), slippage delta in bps,
  fill-rate delta, fee delta.  Output: ``vol_arb/output/S5_TESTNET_RECONCILIATION.md``.
- ``cli``: ``--mode paper | live-testnet`` driver.

Conventions
-----------
- All secret reads via env (``BINANCE_TESTNET_API_KEY`` / ``_SECRET``); no
  secrets in code.
- Real fills only: testnet orders go to testnet.binancefuture.com.
- Reconciliation compares the SAME signal stream against two execution paths
  (paper-calibrated vs testnet-real) so we measure execution divergence in
  isolation, not signal drift.
"""

__all__ = [
    "binance_testnet",
    "testnet_executor",
    "reconciler",
    "cli",
]
