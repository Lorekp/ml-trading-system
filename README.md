# ML Trading System

A Python research monorepo for crypto realized-volatility estimation, volatility forecasting, options simulation, and multi-asset directional signals. It connects market-data preparation, model evaluation, execution simulation, and checks intended to catch misleading research results.

**Research status: no strategy is currently validated for live capital.** The historical Deribit V2 profit result was invalidated because changes in an SVI surface fit were treated as executable option-price gains. Honest marking produced negative results. S3 failed its frozen validation gate; its later rescue results were non-causal. S5 directional research still requires a clean full-universe rerun and independent execution/cost evidence. Forecast metrics also need regeneration with pinned inputs and corrected label semantics.

The maintained [status](docs/STATUS.md), [validation ledger](docs/VALIDATION.md), and [post-mortems](docs/postmortems/) explain these conclusions. Old experiment names, trade logs, and commit messages are not current performance claims.

## What is implemented

- Tick ingestion and dollar-bar processing, with two-scale realized volatility (TSRV), jump-robust estimators, semivariance, and quarticity.
- Volatility labels and point-in-time features, LightGBM forecasting, walk-forward evaluation, and optional neural encoders.
- Options alignment, SVI features, fees, sizing, and diagnostic marking to recorded same-contract option trades.
- Directional feature, portfolio, slippage, and execution research.
- Purged cross-validation, leakage regressions, and explicit PnL-realism, sizing, and selection-bias gates.
- Replay, paper, and testnet infrastructure. Legacy exchange connectors exist, but their presence does not establish a deployable strategy.

## Architecture

```text
Market data -> downloader -> TSRV -> vol_model -> vol_arb
                                                   -> deribit_paper
Market data -> src/signal_engine -> src/live
Shared      -> src/afml, validation, configs, tests
```

| Directory | Role |
| --- | --- |
| `downloader/`, `TSRV/` | Data ingestion, sampling, realized-volatility labels |
| `vol_model/` | Features, forecasting, label alignment, walk-forward drivers |
| `vol_arb/` | Options research, honest marking, costs, risk, backtesting |
| `deribit_paper/` | Replay/testnet machinery; historical V2 configuration is forensic material |
| `src/signal_engine/`, `src/live/` | Directional research and execution infrastructure |
| `src/afml/`, `validation/` | Financial ML utilities and evidence gates |
| `scripts/`, `experiments/` | Research drivers, audits, and experiment source |
| `tests/`, `deribit_paper/tests/` | Unit, regression, and artifact-dependent checks |

See [architecture and contracts](docs/ARCHITECTURE.md) for module boundaries. Packages under `src/` are installed as top-level packages; root workflow packages currently run from the repository checkout.

## Setup

Use Python **3.11 or newer**, working from the repository root:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[dev,full]"
```

The manifest uses minimum dependency versions rather than a locked environment. Install a CPU/CUDA-appropriate PyTorch build separately for neural or DRL modules. Exchange/testnet dependencies are also available through `python -m pip install -e ".[dev,full,live]"`. Configure credentials locally from `.env.example` only when the chosen workflow requires them.

## Data and a first research workflow

Market data, trained weights, caches, and generated results are excluded from version control. A fresh checkout can inspect code and run synthetic checks, but cannot reproduce the historical research by itself.

The TSRV default aggregate-trade input is `data/binance-aggTrades-parquet/`; input settings live in `TSRV/config.py` and loader details in `TSRV/data_loader.py`. Use the downloader and conversion utilities to prepare the expected data rather than substituting arbitrary OHLCV files. Review [TSRV documentation](TSRV/README.md) and [forecasting documentation](vol_model/README.md) before starting a full run.

With the required local trade data prepared, this builds a 15-minute label series, engineers features, and trains a local research model:

```bash
python -m TSRV.run_tsrv_series --asset BTCUSDT --tf 900 --no-plot
python -m vol_model.run_engineer --asset BTCUSDT --tf 900
python -m vol_model.run_train --asset BTCUSDT --tf 900 --no-shap
```

`--tf` is measured in seconds. Labels are written under `TSRV/output/`; model features and results live under `vol_model/output/`. This is a research starting point, not a certified walk-forward result.

The more extensive walk-forward and options workflows need matching SVI, HAR-PD, prediction, and option-data artifacts. Inspect their arguments and subsystem documentation first:

```bash
python -m downloader --help
python -m vol_model.run_walkforward_pruned_svi_har_pd --help
python -m vol_arb.run_pipeline --help
python -m vol_arb.run_backtest --help
```

Treat timestamps, horizons, coverage, and input hashes as part of the data contract. TSRV files have different historical stamping conventions. Walk-forward callers must load labels with `clip=False` and fit clipping bounds on each training fold: `load_labels` still defaults to full-series clipping for older callers. Do not reuse cached predictions after changing these semantics.

## Tests and evidence

Start with the focused synthetic/regression checks:

```bash
python -m pytest tests/test_label_semantics.py tests/test_honest_pnl.py tests/test_validation_gates.py -q
```

Run the default suite or opt into additional groups:

```bash
python -m pytest -q
python -m pytest -q -m integration
python -m pytest -q -m slow
```

The default pytest configuration excludes `integration` and `slow`. Artifact-dependent tests need their declared local files, and some checks skip when these are absent. A passing unit suite cannot establish profitability or validate historical input artifacts.

The strict options-PnL gate admits only finite tier-1 observations with real trade prices on all four option legs. It requires minimum sample size and coverage, sign agreement, bounded aggregate error, and per-trade correlation. Mark-price fallbacks are diagnostic only; backward trade-tape matching is still not proof that an order could have filled. See [validation policy](docs/VALIDATION.md) before promoting any result.

## Limitations and attribution

Open issues include invalid SVI sidecar coverage/stamping, unpinned historical inputs, unresolved forecast comparisons, incomplete S5 realism evidence, and suspected defects in canonical execution paths. The [correctness backlog](docs/KNOWN_ISSUES.md) records their scope. Large research workflows require external data and may be computationally expensive.

This repository integrates established numerical and ML libraries, including NumPy, pandas, SciPy, scikit-learn, LightGBM, and optional neural tooling. AFML implementations cite Marcos López de Prado's *Advances in Financial Machine Learning* (2018); these methods are not claimed as original algorithms. Commit history records AI-assisted contributions.

The project is proprietary research software. No redistribution license is supplied. Preserve source-level attribution and respect the separate licenses of dependencies and data providers.
