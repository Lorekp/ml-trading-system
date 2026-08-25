# Volatility forecast model

[Project status](../docs/STATUS.md) ·
[Validation ledger](../docs/VALIDATION.md)

`vol_model/` predicts a future realized-volatility label, usually
`log(vol_ann)` from `TSRV/`, using point-in-time market and microstructure
features. The package contains feature engineering, LightGBM/QLIKE training,
walk-forward evaluation, uncertainty models, and optional TransVol embeddings.

The forecast pipeline remains a useful research asset, but current headline
R²/QLIKE numbers are **uncertain** until regenerated from pinned label semantics,
folds, and prediction artifacts. Generated reports under `output/` are evidence,
not canonical status.

## Structure

- `config.py` — asset, timeframe, and data-path configuration.
- `labels.py` — label loading, stamping normalization, and train-only clipping.
- `features/` — dollar-bar, liquidity, derivatives, SVI, HAR-PD, and related
  point-in-time features.
- `pipeline.py` — feature/label assembly.
- `model.py` and `evaluate.py` — LightGBM, QLIKE, purged CV, calibration, and
  baseline comparisons.
- `transvol/` and `patchtst/` — optional neural encoders.
- `run_walkforward*.py` — expanding-window research drivers.

## Setup and basic use

Run from the repository root with Python 3.11 or newer:

```bash
python -m pip install -e ".[full,dev]"

# Prerequisite: create the matching realized-volatility labels.
python TSRV/run_tsrv_series.py --asset BTCUSDT --tf 900 --no-plot

# Build dollar-bar features and train a local research model.
python vol_model/run_engineer.py --asset BTCUSDT --tf 900
python vol_model/run_train.py --asset BTCUSDT --tf 900 --no-shap
```

For a bounded hyperparameter search:

```bash
python vol_model/run_optuna.py --asset BTCUSDT --tf 900 --n-trials 20
```

Walk-forward drivers require their named SVI, HAR-PD, and optional TransVol
artifacts. Inspect `--help` before running them; do not reuse an old predictions
parquet under changed label or feature semantics.

## Validation requirements

- Features at a decision timestamp may use only information available by that
  timestamp.
- Label stamping and horizon must be verified against raw windows.
- Walk-forward evaluation should load unclipped labels and fit any clipping or
  normalization bounds on the training fold only.
- Comparisons must use identical folds and persist both arms. A metric quoted
  only from an in-memory run is not reproducible evidence.
- Forecast quality and strategy PnL are separate claims. A good QLIKE result
  does not prove that options costs can be cleared.

Focused integrity tests:

```bash
python -m pytest -q tests/test_label_semantics.py tests/test_leakage_probes.py
```

Outputs normally live under `vol_model/output/`. Record the code commit, input
paths, data window, command, fold definition, and metric table before promoting
any result into `docs/STATUS.md`.
