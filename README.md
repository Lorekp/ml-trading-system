# ml-trading-system

Private research monorepo for a crypto-derivatives volatility-arbitrage system.
The production target is a BTC vol-arb signal traded as ATM straddles on Deribit,
hedged via a DRL policy on Hyperliquid. Research infrastructure in this repo is
asset-agnostic; ETH is the secondary asset.

> **Private code. Do not distribute.** No live secrets, API keys, private
> keys, exchange account IDs, or raw vendor data are tracked in this repo.
> Those live in `.env` / `data/live/*.pem` / `data/binance-aggTrades-parquet/`,
> all of which are `.gitignore`d. See [Security](#security) below.

---

## Repo layout

| Path | Purpose |
| --- | --- |
| `TSRV/` | Two-scales realized-volatility pipeline (Zhang et al. 2005). Fast/slow RV, BPV, MedRV, jump decomposition, realized quarticity. One row per 15-min window per asset. |
| `vol_model/` | ML training pipeline: feature engineering, LightGBM + TransVol hybrid, walk-forward, spot-checks, signal-research threads. |
| `vol_arb/` | Options-merge pipeline: Deribit option trades + DVOL + SVI calibration → per-15m vol-arb dataset + backtester. |
| `deribit_paper/` | Live paper-trading engine (dry-run on Deribit testnet). |
| `downloader/` | Historical data ingestion (Binance Vision aggTrades, Deribit option trades, HL L2 snapshots). |
| `src/afml/` | Generic quant utilities (PurgedKFold, Deflated Sharpe, bootstrap tests, fractional differencing). |
| `scripts/` | One-off research scripts + forensic audits. |
| `tests/` | `pytest` suite — leakage probes, causality guards, regression locks on Stage 18 semantics. |
| `configs/` | Deployment / asset-specific config YAML. |
| `docs/` | Cross-cutting documentation. |

Large, regenerable, or sensitive directories (`data/`, `models/`, `logs/`,
`results/`, `v*-results/`, `experiments/`, `backups/`, subsystem `*/output/`
caches, `vendor/`, `library/hftbacktest/`) are gitignored — see
[`.gitignore`](.gitignore).

## Quick start

```bash
# 1. Clone + create venv
git clone <url> ml-trading-system && cd ml-trading-system
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev,full]"

# 2. Configure credentials (NEVER commit .env)
cp .env.example .env
$EDITOR .env  # fill in exchange / data-vendor / messaging credentials

# 3. Run the test suite
pytest tests/ -q

# 4. Rebuild data artifacts (optional — large downloads / long compute)
python -m downloader.run_binance_aggtrades --asset BTCUSDT
python -m TSRV.run_tsrv_series --asset BTCUSDT --threshold 500000
python -m vol_arb.run_pipeline --currency BTC --svi
```

## Key dependencies

- **Python 3.11+**, NumPy 2.x, Pandas 2.x, PyArrow, scikit-learn, SciPy, Pydantic.
- **ML**: LightGBM, PyTorch (TransVol / PatchTST encoders), Optuna, XGBoost.
- **DRL**: stable-baselines3 / SB3-contrib (SAC, TQC).
- **Finance / stats**: statsmodels, KneeLocator, lz4.
- See [`pyproject.toml`](pyproject.toml) and [`requirements.txt`](requirements.txt)
  for the pinned list.

## Data expectations

The repo ships without raw data. Downstream scripts assume the following
layout under `data/` (all gitignored):

```
data/
├── binance-aggTrades-parquet/{SYMBOL}/{YYYY-MM}.parquet   # tick trades
├── bookTicker_parquet/{SYMBOL}/{YYYY-MM-DD}.parquet       # L1 quotes
├── hl-bbo-raw/                                            # Hyperliquid L2
├── deribit-option-trades/{CCY}/{YYYY-MM}.parquet          # option fills
├── dvol/{CCY}_dvol_hourly.parquet                         # forward-vol index
├── funding-rates/                                          # perp funding
├── oi/                                                     # open interest
└── live/                                                   # runtime keys, SQLite state
```

Download scripts are in `downloader/`; calibration thresholds (dollar-bar,
sub-bar) are cached in per-asset TSRV config.

## Testing

```bash
pytest tests/ -q                           # full suite
pytest tests/test_label_semantics.py -v    # Stage 18 label regression
pytest tests/test_leakage_probes.py -q     # future-bar / alignment guards
pytest tests/test_hqr.py -v                # conformal sizing wrapper
```

Every new research thread ships with a causality test. Leakage probes run
as a hard gate inside the spot-check gauntlet (`run_spot_check_signal_research.py`)
— a candidate that fails any probe is rejected before training.

## Security

- **`.env`** (exchange API keys, bot tokens, private-key hashes) is
  `.gitignore`d. Always start from `.env.example`.
- **`data/live/*.pem`** (Bybit RSA private keys) is gitignored via the
  catch-all `*.pem` rule; `data/` is gitignored at the directory level too.
- **`backups/`** may contain historical `.env` snapshots; gitignored at
  the directory level.
- Hardcoded credentials in source are not allowed. `scripts/create_demo_keys.py`
  (the one place that previously had them) reads from `BYBIT_MAIN_API_KEY` /
  `BYBIT_MAIN_API_SECRET` env vars.
- **Before pushing to any remote**, run
  `git ls-files | grep -Ei '\.env$|\.pem$|credentials|secrets'` and
  confirm zero matches.
- If a key is ever committed (even to a local branch), **rotate it
  immediately** on the exchange / provider side. Removing it from the
  working tree is not enough — git history is permanent.

## Status

Production target: **BTC vol-arb v2** (NC PnL $726K, NC Trade Sharpe 12.56
on 5-year walk-forward). See `vol_model/output/SIGNAL_RESEARCH_NOTES.md`
for the live research log and `TSRV/PIPELINE_DOCUMENTATION.md` for the
end-to-end pipeline description.

## License

Proprietary. All rights reserved.
