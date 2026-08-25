# ML Trading System

Research monorepo for crypto volatility forecasting, options backtesting, and
multi-asset directional signals.

> **Current status:** no strategy in this repository is validated for live
> capital. The historical Deribit V2 result was invalidated by a PnL-marking
> bug, S3 failed its frozen validation gate, and S5 still needs a clean
> full-universe rerun. The data, forecasting, replay, and validation
> infrastructure remain useful research assets.

## Start here

- [Current status](docs/STATUS.md) — what is active, invalidated, and next.
- [Architecture](docs/ARCHITECTURE.md) — the two research stacks and their
  boundaries.
- [Validation policy](docs/VALIDATION.md) — evidence standards and result
  ledger.
- [Post-mortems](docs/postmortems/) — scoped accounts of the major failures.

Generated reports under `*/output/` and non-source artifacts produced by
`experiments/` are supporting evidence, not current documentation.

## Research areas

| Path | Responsibility |
| --- | --- |
| `TSRV/` | Trade-to-realized-volatility pipeline: dollar bars, TSRV, BPV, MedRV, jumps, and quarticity. |
| `vol_model/` | Volatility labels, features, LightGBM/TransVol forecasting, and walk-forward evaluation. |
| `vol_arb/` | Deribit option alignment, SVI/DVOL features, honest PnL, fees, and options backtesting. |
| `deribit_paper/` | Paper/testnet replay infrastructure. Historical V2 settings are isolated from its CLI. |
| `src/signal_engine/` | Parallel multi-asset directional research stack, including S5. |
| `src/live/` | Directional execution infrastructure. Legacy real connectors exist, but current use is limited to demo/testnet research. |
| `src/afml/` | Shared purged-CV, bootstrap, Deflated Sharpe, and labeling utilities. |
| `downloader/` | Binance and Deribit ingestion. |
| `experiments/` | Tracked research source; caches, data, logs, and generated reports stay local. |
| `scripts/` | Research, audit, regeneration, and infrastructure drivers; names do not imply validated results. |
| `tests/` | Unit, regression, causality, and data-dependent integration tests. |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the data flow and module
contracts.

## Setup

Python 3.11 or newer is required.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[dev,full]"
```

Install the `live` extra only for exchange/testnet work:

```bash
python -m pip install -e ".[dev,full,live]"
```

PyTorch is intentionally not pinned because the correct wheel depends on the
host's CPU/CUDA setup. Install it separately before using TransVol or DRL
modules.

The checkout does not include market data or model weights. Copy
`.env.example` to `.env`, add credentials locally, and populate the ignored
`data/` tree before running data-dependent pipelines.

## Tests

```bash
python -m pytest -q
```

The default configuration excludes tests marked `integration` or `slow`.
Run those groups explicitly when their ignored artifacts or long numerical
calibrations are appropriate:

```bash
python -m pytest -q -m integration
python -m pytest -q -m slow
```

For a fast validity-oriented check:

```bash
python -m pytest tests/test_label_semantics.py tests/test_honest_pnl.py tests/test_validation_gates.py -q
```

Integration tests require large ignored artifacts and are not fresh-clone
tests. A passing unit suite is necessary but does not validate a trading
result; use the gates in [docs/VALIDATION.md](docs/VALIDATION.md).

## Typical pipeline entry points

```bash
python -m downloader --help
python -m TSRV.run_tsrv_series --help
python -m vol_model.run_walkforward_pruned_svi_har_pd --help
python -m vol_arb.run_pipeline --help
python -m vol_arb.run_backtest --help
python scripts/run_baseline_experiments.py --help
```

The root workflow packages run from a repository checkout. The editable
package installs the reusable modules under `src/`.

## Repository policy

- Source, tests, configuration templates, and canonical Markdown under
  `docs/` are versioned.
- Raw data, caches, model weights, plots, logs, and generated reports are not.
- A result is not a claim until its code, data window, reproduction command,
  costs, fills, and validation gates are recorded.
- Credentials belong only in environment variables or ignored local files.
  Never put a key in YAML, source code, or a command committed to Git.

This is proprietary research software. It is not financial advice and is not
licensed for redistribution.
