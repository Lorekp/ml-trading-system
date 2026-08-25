# Kronos RTX 3090 experiment — archived

[Project status](../docs/STATUS.md) ·
[Validation ledger](../docs/VALIDATION.md)

**Status: archived.** This directory is a leftover handoff bundle for an
experiment that proposed fine-tuning the Kronos time-series model on 15-minute
crypto OHLCV data. It is not part of the active forecast pipeline and carries no
validated performance or deployment claim.

## Why it is not runnable as a package

The directory contains only:

- `config_3090.yaml` — historical training parameters;
- `setup_and_run.sh` and `setup_and_run.bat` — convenience launchers from the
  original handoff.

Those launchers reference files that are not included here, including
`kronos/finetune_csv/train_sequential.py`, `crypto_15m.csv`, and a tokenizer
checkpoint. They also install unpinned dependencies at runtime. Do not run them
as project setup scripts.

The supported volatility research code is [`vol_model/`](../vol_model/README.md).

## If the experiment is revived

Treat revival as a new experiment: vendor or pin the upstream implementation
and license, define immutable train/validation/test splits, inventory the data,
pin dependencies and model revisions, add leakage tests, and compare it against
the current walk-forward baselines. Only then should results be added to
`docs/STATUS.md`.
