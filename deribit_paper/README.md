# Deribit paper-engine infrastructure

[Project status](../docs/STATUS.md) ·
[Validation ledger](../docs/VALIDATION.md)

`deribit_paper/` is an asynchronous Deribit market-data, simulated-execution,
position, risk, and monitoring engine. It can assemble the repository's TSRV,
SVI, TransVol, and LightGBM feature path against testnet or public mainnet
quotes.

It is infrastructure, not evidence of a profitable strategy. No configuration
in this module is currently validated for live capital.

## V2 is historical and disabled

`PaperConfig.historical_v2_preset()` remains available to reproduce old replays
and parity checks. It mirrors settings selected against the invalid V2
false-PnL backtest, including aligned predictions, Kelly sizing, and PnL-tuned
risk/filter choices. It is deliberately absent from the CLI and must not drive
paper decisions. See the V2 postmortem through the validation ledger.

Every CLI mode forces dry-run execution. The real-order `--live` route and its
passphrase mechanism have been removed; `--paper` means simulated fills against
public mainnet quotes.

## Setup

Run from the repository root with Python 3.11 or newer:

```bash
python -m pip install -e ".[full,live,dev]"
```

Optional testnet authentication is loaded from `.env` or the environment:

```text
DERIBIT_TESTNET_CLIENT_ID=...
DERIBIT_TESTNET_CLIENT_SECRET=...
PAPER_DRY_RUN=true
TELEGRAM_BOT_TOKEN=...       # optional
TELEGRAM_CHAT_ID=...         # optional
```

Safe infrastructure checks:

```bash
# Public mainnet connection and market-data smoke test, with simulated fills.
python -m deribit_paper --paper --smoke

# Testnet authentication check; does not place orders.
python -m deribit_paper --test-auth

# Bounded testnet dry run. The default configuration is an engineering
# baseline, not validated alpha.
python -m deribit_paper --run-paper --duration-minutes 45

# Bounded mainnet-quote paper run; still forced dry-run.
python -m deribit_paper --paper --run-paper --duration-minutes 45
```

## Components

- `client/` and `market_data/` — REST/WS clients, subscriptions, buffers, and
  exchange-clock bar assembly.
- `features/` and `signal/` — live feature assembly and model inference.
- `execution/` and `position/` — idempotent order flow, hedge accounting, and
  persistent positions.
- `risk/` — dry-run guard, limits, kill switch, DVOL gate, and RND monitor.
- `monitoring/` — logs, metrics, harness output, and optional Telegram alerts.
- `engine.py` and `cli.py` — orchestration and command-line entry point.

Runtime state is written beneath `deribit_paper/state/` and logs beneath
`deribit_paper/logs/`; both should be treated as local operational data.

Run the module tests with:

```bash
python -m pytest -q deribit_paper/tests
```

Passing infrastructure tests does not restore the invalidated V2 edge. A future
strategy must first meet the promotion requirements in `docs/VALIDATION.md`.
