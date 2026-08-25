# Volatility-arbitrage research package

[Project status](../docs/STATUS.md) ·
[Validation ledger](../docs/VALIDATION.md)

`vol_arb/` merges realized-volatility labels and forecasts with Deribit options,
DVOL, SVI surfaces, Greeks, fees, liquidity, sizing, and hedge simulation. It is
the repository's options research and forensic backtesting package.

## Current status

The historical 15-minute BTC V2 result is **invalidated**. Its positive PnL and
Sharpe came from treating changes in a noisy per-bar SVI fit as executable vega
PnL. Honest DVOL and real-fill checks are negative. The CLI flag
`--historical-btc-v1` and `--historical-btc-v2` exist only for forensic
reproduction of invalidated research runs. They are not deployment presets.

SVI calibration, TSRV, forecast features, fee models, and execution tooling may
still be useful independently. They do not inherit a validated edge from the
old V2 result.

## Structure

- `pipeline.py`, `options_loader.py`, and `run_pipeline.py` — options/DVOL/TSRV
  merge and alignment checks.
- `svi.py` — surface calibration and derived features.
- `backtester.py`, `run_backtest.py`, and `metrics.py` — research simulation and
  reporting.
- `honest_pnl.py` and `validation/gates.py` — real-premium marking and result
  gates.
- `deribit_fees.py`, `impact.py`, `sizing.py`, and `capacity.py` — costs and
  exposure constraints.
- `drl_hedging/` — experimental hedge-policy environment and evaluation.
- [Known issues](../docs/KNOWN_ISSUES.md) records defect hypotheses that still
  require direct active-path fixes.

## Setup and exploratory use

Run from the repository root with Python 3.11 or newer:

```bash
python -m pip install -e ".[full,dev]"

# Build the prerequisite 15-minute realized-volatility series.
python TSRV/run_tsrv_series.py --asset BTCUSDT --tf 900 --no-plot

# Build a merged BTC research dataset. Required Deribit option and DVOL inputs
# are resolved by vol_arb/paths.py and vol_arb/config.py.
python vol_arb/run_pipeline.py --currency BTC --bar-minutes 15 --svi

# Diagnostic simulation only; no result is promoted without the gates below.
python vol_arb/run_backtest.py --currency BTC --realistic --extended
```

Outputs under `vol_arb/output/**` are generated evidence and are ignored by Git.
They are not promoted results and may contain superseded or contradictory
experiments.

## Result requirements

Before quoting any PnL or Sharpe:

1. Use a clean pinned commit and record the exact input paths and data window.
2. Confirm `BacktestConfig.honest_vega_iv=True`; never use SVI fit changes as a
   realizable option mark.
3. Mark a covered sample to real option premiums.
4. Pass `vol_arb.honest_pnl.strict_pnl_realism_gate`, which filters to declared
   tier-1 pairs, enforces coverage and sample size, then applies sign,
   aggregate-error, and correlation checks. `gate_no_fake_profit` is only a
   sign-based smoke gate.
5. Apply the precommitted selection-bias, sizing, and statistical gates.

Focused tests:

```bash
python -m pytest -q tests/test_validation_gates.py
# Data-dependent and skipped when the local option tape is absent:
python -m pytest -q -m integration tests/integration/test_false_profit_smoke.py
```

The incident record and current safeguard limitations are documented in
`docs/postmortems/2026-06-v2-fake-pnl.md` via the validation ledger.
