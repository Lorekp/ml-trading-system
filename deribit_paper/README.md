# `deribit_paper/` — Deribit Demo-Trading Module

Live paper-trading engine for the BTC vol-arb strategy (short straddles + DRL
delta-hedged perp hedge) on **Deribit testnet**. This module is the production
validation surface for every Sharpe number that currently lives in the
backtester: until a signal has survived a live paper run, it has not been proven
outside of historical data.

> **Isolation**: this module deliberately does **not** touch
> `ml-trading-system/src/live/` (the Bybit production framework). Lessons
> learned here will be ported into the Bybit path once the paper run graduates.

## Purpose

1. **Ground-truth the strategy** against a live market on testnet — real order
   books, real market makers, real timing — without any capital at risk.
2. **Catch implementation drift** between the batch backtester and the live
   predictor. Historical Sharpe is meaningless if the live code computes
   different features or signals.
3. **Validate execution assumptions**: maker fill rates, slippage, intra-bar
   hedge timing, the DRL policy's response to real-time data.
4. **Prove operational readiness**: safety flags, kill switches, monitoring,
   reconnect, state persistence. Everything must work before we move to Bybit.

## Quick Start

```bash
# 1. Authenticate against Deribit testnet (round-trip, no orders)
python -m deribit_paper --test-auth

# 2. 60-second connection smoke test (ws connect, market data flow, stats)
python -m deribit_paper --smoke

# 3. Dry-run paper engine for N minutes (no real orders)
python -m deribit_paper --run-paper --duration-minutes 45

# 4. Dry-run forever (Ctrl+C to stop)
python -m deribit_paper --run-paper

# 5. LIVE testnet orders (requires interactive passphrase confirmation)
python -m deribit_paper --run-paper --live --duration-minutes 60
```

All commands default to `venue=testnet` and `dry_run=True`. Live orders require
**both** `--live` *and* typing the passphrase `GO LIVE` at the prompt — there is
no env-var or config-file bypass.

## Required `.env` Variables

```bash
DERIBIT_TESTNET_CLIENT_ID=...
DERIBIT_TESTNET_CLIENT_SECRET=...
DERIBIT_TESTNET_BASE_URL=https://test.deribit.com/api/v2    # optional; default
DERIBIT_TESTNET_WS_URL=wss://test.deribit.com/ws/api/v2     # optional; default
TELEGRAM_BOT_TOKEN=...                                       # for alerts
TELEGRAM_CHAT_ID=...
PAPER_DRY_RUN=true                                           # safety default
```

Generate Deribit testnet credentials at https://test.deribit.com → Settings →
API. The module does **not** reuse testnet credentials in production — a
`switch_to_prod()` call requires separate `DERIBIT_PROD_CLIENT_ID` / `_SECRET`.

## Architecture

```
deribit_paper/
├── cli.py                     CLI entry (--test-auth / --smoke / --run-paper / --live)
├── config.py                  PaperConfig dataclass (from_env + switch_to_prod)
├── engine.py                  TradingEngine — wires all services, runs the main loop
│
├── client/                    Thin async Deribit wrapper
│   ├── auth.py                OAuth2 client_credentials, auto-refresh
│   ├── rest.py                httpx REST: account, instruments, orders, positions
│   ├── ws.py                  websockets with auto-reconnect + lazy WS auth
│   └── errors.py              DeribitRESTError / DeribitWebSocketError
│
├── market_data/               Live feeds → bar-aligned snapshots
│   ├── subscriber.py          WS subscriptions: perp.raw, option.raw, price index
│   ├── buffer.py              24h perp + 4h option + 1h index rolling buffers
│   ├── aggregator.py          15-min bar builder (fires on wall-clock boundaries)
│   └── bar_clock.py           async scheduler anchored to UTC :00/:15/:30/:45
│
├── features/                  Feature pipeline (reuses batch code on rolling windows)
│   ├── tsrv_live.py           calls TSRV/tsrv_series.compute_tsrv() on perp buffer
│   ├── svi_live.py            calls vol_arb.svi.calibrate_surface() every hour
│   └── straddle.py            ATM strike picker against Deribit's live instruments
│
├── signal/                    Model inference
│   ├── model_loader.py        loads latest walk-forward LightGBM + TransVol + DRL
│   ├── feature_assembler.py   builds the 51-dim row at bar close from live state
│   ├── predictor.py           rv_pred → edge → direction (long/short/flat)
│   └── drl_hedger_live.py     scalar per-tick wrapper around DRLHedger.simulate()
│
├── execution/                 Order placement
│   ├── order_manager.py       place/cancel/track with retry + idempotency
│   ├── straddle_flow.py       open + close straddle legs (maker → taker fallback)
│   ├── perp_hedger.py         intra-bar hedge execution from DRL scalar output
│   ├── idempotency.py         client_order_id dedup (60s window)
│   └── amounts.py             Deribit amount rounding (options = BTC, perps = USD)
│
├── position/                  State & P&L
│   ├── tracker.py             open positions, unrealized PnL, greeks
│   ├── equity.py              running equity curve + drawdown
│   └── store.py               SQLite persistence (reconciles on restart)
│
├── risk/                      Safety
│   ├── dry_run.py             process-global DRY_RUN flag + confirmation token
│   ├── limits.py              max_positions, max_margin, max_trade_size, DTE band
│   └── kill_switch.py         halts on daily DD, consecutive errors, network outage
│
├── monitoring/                Observability
│   ├── telegram.py            alerts + heartbeat + daily summary
│   ├── logger.py              JSONL rotating log → deribit_paper/logs/
│   └── metrics_sink.py        per-bar row → deribit_paper/state/live_metrics.parquet
│
├── scripts/
│   └── validate_dry_run.py    Phase 7: signal parity vs backtester on historical data
│
└── tests/                     pytest coverage: 100+ unit tests, all fully mocked
```

### Reused Strategy Code (imports, not copies)

The module owns only **venue-level** code. Strategy logic is imported from the
existing research modules:

| Source | What we reuse |
| --- | --- |
| `vol_arb/svi.py` | `calibrate_surface()`, `evaluate_surface_batch()` |
| `vol_arb/drl_hedging/env.py` | `DRLHedger` wrapped by `LiveDRLHedger` |
| `vol_arb/drl_hedging/reward.py` | `bsm_price_from_iv`, `straddle_delta_from_iv`, `straddle_gamma_from_iv` |
| `vol_arb/drl_hedging/config.py` | `DRLHedgingConfig` constants |
| `TSRV/tsrv_series.py` | `compute_tsrv()` on rolling window |
| `vol_model/features.py` | base feature computation |
| `vol_arb/metrics.py` | `full_metrics_both()` for post-run analysis |

This is non-negotiable: there is **one** implementation of the signal, and the
paper engine must call it, not re-invent it.

## Safety Model

Every execution path is gated by **layered** safety mechanisms. Any one failing
halts the engine before a bad order leaves the building.

1. **`DRY_RUN` process-global flag** (`risk/dry_run.py`). Defaults to `True`.
   Flipping it requires a secret confirmation token returned by the CLI's live
   passphrase prompt — `set_dry_run(False, confirmation_token=...)` refuses
   without it. Every order entry path checks `get_dry_run()` before hitting
   Deribit.
2. **CLI passphrase confirmation** (`cli.py::_confirm_live`). `--live` prompts
   for `GO LIVE` typed interactively via `getpass`. Mismatch → exit(1), stays
   in dry-run. No env-var or config-file bypass.
3. **Testnet by default** (`config.py`). `venue="testnet"`,
   `rest_base_url=test.deribit.com`. Production venue URL flip requires
   `switch_to_prod()` which will **refuse** to reuse testnet credentials.
4. **Risk limits** (`risk/limits.py`). Checked before every signal:
   - `max_open_positions = 5`
   - `max_margin_util_hard = 0.20`
   - `max_single_trade_btc = 2.0`
   - DTE band `[1, 14]` days (no 0DTE, no LEAPS)
   - Signal rejected → logged as `SUPPRESSED` with reason, not traded.
5. **Kill switch** (`risk/kill_switch.py`). Halts new signals when:
   - Daily drawdown from peak > 2%
   - 3 consecutive execution errors
   - WS silent for > 120 seconds
   - State persists in SQLite; restart does **not** auto-resume.
6. **Idempotency** (`execution/idempotency.py`). Client order IDs are dedup'd
   over a 60s window; reconnects can't double-send.
7. **SQLite state store** (`position/store.py`). Every mutation committed.
   On restart, `reconcile_with_deribit()` reads account state before resuming
   position tracking.
8. **Telegram alerts** (`monitoring/telegram.py`). Every meaningful event
   (trade open/close, kill switch, errors, hourly heartbeat, daily summary) is
   pushed to Telegram. We never run silently.

## Key Design Decisions

### Single-process asyncio
One Python process, one event loop, multiple async tasks. 15-min bar cadence
doesn't justify inter-process coordination. Keeps state reasoning simple.

### Deribit WS auth is lazy, not eager
Deribit's `.raw` trade channels require authentication even though they're
nominally "public". The WS client authenticates on first `subscribe()` if a
`DeribitAuth` is present, not at connect time. Re-connects reset the flag so
auth is retried on the fresh session.

### WS reader runs as a background task
`_start_reader_task()` spawns `_read_forever()` immediately after the socket
connects, so RPC responses (including the auth response) are consumed even
before `dispatch_forever()` is called. Without this, the first RPC would hang
because nothing was draining the frame queue.

### Bar alignment uses Deribit's clock, not local
The bar clock anchors on Deribit-server trade timestamps for alignment, not
`datetime.now()`. Local clock drift can't push bars off the 15-min grid.

### SVI recalibration is backgrounded
`calibrate_surface()` takes 10-15s per bucket on the full strike set. Run on
`asyncio.create_task()` every hour, with atomic swap of the cached params at
completion. The bar-close handler never blocks on it.

### Deribit units are NOT consistent
- **Options**: amount denominated in **BTC**, 0.1 BTC increments
- **Perps**: amount denominated in **USD notional**, $10 increments

`execution/amounts.py` centralizes the rounding + unit conversion. Every order
path calls it; no ad-hoc `round()` elsewhere.

### DRL wrapper is scalar + per-trade
`DRLHedger.simulate()` is vectorized for batch rollouts; live trading needs a
single-trade, single-tick API. `signal/drl_hedger_live.py::LiveDRLHedger`
exposes `begin_trade / step / close_trade`, maintains per-trade state
(inventory, SVI slice frozen at entry, cumulative path), and applies
`VecNormalize` with the exact same sign-flipped "virtual short" convention
used in training. Verified to match `simulate()` to 1e-6 in
`test_drl_hedger_live.py`.

## Phase Status

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | Client + auth + config + `DRY_RUN` flag | **Done** |
| 2 | Market data + bar aggregation | **Done** |
| 3 | Feature pipeline (TSRV + SVI live + assembler) | **Done** |
| 4 | Signal + DRL inference (model_loader + predictor + LiveDRLHedger) | **Done** |
| 5 | Execution (order manager + straddle flow + perp hedger) | **Done** |
| 6 | Position + risk + monitoring | **Done** |
| 7 | Dry-run validation vs backtester | **Done — 100% signal parity (673/673 bars)** |
| 8 | Testnet paper run (1-2 weeks) | **In progress** |

### Phase 7 results
`scripts/validate_dry_run.py` compared the live `Predictor` against the batch
backtester's signal path on `vol_arb/output/BTC_vol_arb.parquet`, window
2026-03-01 → 2026-03-08. 673 bars compared, **673 matches, 0 mismatches,
100.0% parity**. Target was ≥95%.

### Phase 8 status
First live paper run launched 2026-04-10 against `test.deribit.com`:
- Auth round-trip → OK
- 1,050 active BTC options discovered
- All three WS channels subscribed (`trades.BTC-PERPETUAL.raw`,
  `trades.option.BTC.raw`, `deribit_price_index.btc_usd`)
- First bar (20:30 UTC): 222 perp trades, 60 option trades, SVI calibrated
  in 2.8ms, LightGBM forward pass ran clean, signal returned FLAT with
  `iv_used unavailable (no SVI slice or no straddle)` — legitimate
  no-action path, not an error
- Engine still running at time of writing; monitoring for executed trades
  over the full paper window

## CLI Commands

```bash
# Authenticate and print account summary
python -m deribit_paper --test-auth

# 60-second smoke test (connect, subscribe, dump stats every 10s)
python -m deribit_paper --smoke

# Dry-run paper engine for 45 minutes
python -m deribit_paper --run-paper --duration-minutes 45

# Dry-run forever
python -m deribit_paper --run-paper

# Live testnet orders with duration cap (requires passphrase)
python -m deribit_paper --run-paper --live --duration-minutes 120

# Flags available on all commands:
#   --venue {testnet,prod}       default: testnet (prod requires --live + passphrase)
#   --log-level INFO|DEBUG|...   default: INFO
```

## Running the Tests

```bash
PYTHONPATH=. pytest deribit_paper/tests/ -v
```

100+ unit tests covering:
- OAuth2 + token refresh + backoff (`test_client.py`)
- Bar aggregation on synthetic trade streams (`test_aggregator.py`)
- Bar clock boundary alignment (`test_bar_clock.py`)
- Rolling trade buffers (`test_buffer.py`)
- Live SVI matches batch `calibrate_surface()` (`test_svi_live.py`)
- Straddle strike picker (`test_straddle.py`)
- Model loader + feature dimensions (`test_model_loader.py`)
- `LiveDRLHedger` scalar ≡ vectorized `DRLHedger.simulate()` (`test_drl_hedger_live.py`)
- Order manager retry + rejection + idempotency (`test_order_manager.py`)
- Straddle flow maker→taker fallback (`test_straddle_flow.py`)
- Perp hedger order path (`test_perp_hedger.py`)
- Deribit amount rounding (`test_amounts.py`)
- Position tracker (`test_position_tracker.py`)
- Equity curve (`test_equity.py`)
- SQLite store reconcile (`test_store.py`)
- Risk limits + kill switch (`test_risk_limits.py`, `test_kill_switch.py`)
- DRY_RUN flag semantics (`test_dry_run.py`)
- Telegram notifier (`test_telegram.py`)
- Signal parity validation (`test_validate_dry_run.py`)
- Engine wiring + `process_bar` dispatch (`test_engine.py`)

All tests fully mocked at the boundary — no real network calls in unit suite.

## State + Logs on Disk

```
deribit_paper/state/
├── paper.db                   SQLite: positions, orders, fills, equity snapshots, kill_switch
├── svi.parquet                cached SVI params (atomic-swap target of hourly recal)
└── live_metrics.parquet       per-bar metrics row (same schema as backtester trade_log)

deribit_paper/logs/
└── deribit_paper.jsonl        rotating structured JSON log (INFO/WARN/ERROR + events)
```

## Acceptance Criteria for Graduating Out of Paper

Per the Phase 8 exit gate in the plan:

1. **Live Sharpe ≥ 5** (daily, non-compound) over 1-2 week window
   (backtest baseline: ~14)
2. **No kill-switch triggers** during the window
3. **Signal parity ≥ 95% bar-for-bar** against synthetic backtester run on
   the same recorded data
4. **DRL hedger produces finite, reasonable actions** (per-tick trade sizes
   within ±0.5 BTC)
5. **Zero uncaught exceptions** — all errors surface to Telegram + are
   recoverable

All 5 must pass before any migration into the Bybit `src/live/` framework. If
live Sharpe is < 2, the interpretation is **"the backtest is over-optimistic"**
and we stop before real capital — paper trading exists precisely to catch
this failure mode.

## Known Gotchas / Lessons Learned

1. **Deribit `.raw` channels require WS auth** even though nominally public.
   Subscription fails with `raw_subscriptions_not_available_for_unauthorized`
   if the WS session is unauthenticated. Fix: `_auth_ws_if_needed()` called
   from `subscribe()` whenever a `DeribitAuth` instance is present.
2. **WS auth uses `client_credentials`, not `bearer`**. My initial attempt
   tried `grant_type="bearer"` with an access_token; Deribit rejected it.
   The correct form is `client_credentials` with `client_id` +
   `client_secret` — the same flow as REST auth.
3. **RPC responses hang without a reader**. The first `public/auth` RPC
   would time out because `dispatch_forever()` hadn't been called yet, so
   no coroutine was consuming frames off the socket. Fix: spawn
   `_read_forever()` as a background task immediately after connect.
4. **Testnet liquidity is thin**. ATM straddle selection can legitimately
   return no candidate at bar close — the front-expiry strike nearest spot
   may not have both a call and a put with recent trades. This is not an
   error; the predictor returns a FLAT signal with reason
   `iv_used unavailable`. Real production Deribit will have better
   liquidity but wider regime exposure.
5. **SVI calibration is slow (~10-15s)**. Must never block the bar-close
   handler. Run in a background asyncio task with atomic cache swap on
   completion.
6. **Deribit option expiry rollover at 08:00 UTC**. The strike picker
   handles this by closing any positions within 1h of expiry — no rolls.
   First-pass decision; can be reconsidered once the paper run shows
   stable behavior across expiry boundaries.
