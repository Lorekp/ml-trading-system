# TSRV — realized-volatility labels

[Project status](../docs/STATUS.md) ·
[Validation ledger](../docs/VALIDATION.md)

`TSRV/` builds noise-corrected realized-variance series from tick-level crypto
trades. It implements the two-scale realized-volatility estimator with
K-subgrid averaging and includes jump-robust BPV, MedRV, semivariance, and
directional jump decompositions.

This package produces data and labels. It does not establish a trading edge,
and no PnL claim should be inferred from a TSRV output.

## Main modules

- `config.py` — input locations, thresholds, and output directory.
- `data_loader.py` and `sampling.py` — tick loading and time/tick/dollar bars.
- `realized_vol.py` — realized variance, noise correction, and annualization.
- `jump_robust.py` — BPV, MedRV, semivariance, and jump decomposition.
- `tsrv_series.py` — windowed TSRV calculation and incremental persistence.
- `signature_plot.py` — sampling/threshold diagnostics.
- `diagnostic.py` — label-quality checks.

## Setup and basic use

Run commands from the repository root with Python 3.11 or newer:

```bash
python -m pip install -e ".[full,dev]"

# Build a 15-minute BTC label series.
python TSRV/run_tsrv_series.py --asset BTCUSDT --tf 900 --no-plot

# Inspect sampling behavior on a bounded set of dates.
python TSRV/run_btc.py --n-dates 30 --no-plot

# Run the label diagnostics.
python TSRV/diagnostic.py
```

Defaults read Binance aggregate trades from
`data/binance-aggTrades-parquet/` and write
`TSRV/output/{ASSET}_tsrv_{TF_MIN}m.parquet`. Override inputs and estimator
settings in `TSRV/config.py`; do not silently substitute a differently stamped
or differently thresholded label file.

For an existing series, `run_tsrv_incremental.py` extends the parquet and writes
atomically:

```bash
python TSRV/run_tsrv_incremental.py --asset BTCUSDT --tf 900
```

## Output contract

The output includes TSRV and raw RV, continuous/jump components, signed
semivariance, jump fractions, observation counts, and annualized volatility.
Downstream consumers must treat timestamp convention and prediction horizon as
part of the schema. The 15-minute BTC file and other-horizon files have had
different stamping histories; use the stamping-aware label path in
`vol_model/labels.py` and the checks described in the validation ledger.

Forecasting code lives in [`vol_model/`](../vol_model/README.md). Options merge
and simulation code lives in [`vol_arb/`](../vol_arb/README.md).
