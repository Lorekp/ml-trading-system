# Vol Model — LightGBM Volatility Prediction

Predicts next-window log-annualized realized volatility using dollar-bar-native microstructure features.
Target: `y = ln(vol_ann)` from TSRV (noise-corrected realized vol).

## Architecture

```
vol_model/
  config.py                   # VolModelConfig (Pydantic)
  labels.py                   # y = ln(vol_ann), one-step-ahead alignment
  pipeline.py                 # Load features + labels, build dataset
  model.py                    # LightGBM + QLIKE custom loss, purged K-fold, SHAP
  evaluate.py                 # HAR baseline (dynamic horizons), Diebold-Mariano, Mincer-Zarnowitz
  features/
    engineer.py               # Dollar-bar-native FeatureEngineer (primary)
    memory.py                 # Legacy: Hurst, FFD, Z-score (15m grid)
    toxicity.py               # Legacy: bar velocity, VPIN, imbalance
    liquidity.py              # Legacy: true OFI, spread, book imbalance
    derivatives.py            # Legacy: funding, OI
    windowing.py              # Legacy: lags, EMAs on 15m grid
  run_engineer.py             # Build dollar-bar features (primary entry point)
  run_train.py                # Train + evaluate + SHAP
  run_optuna.py               # Optuna hyperparameter search (TPE + MedianPruner)
  run_features.py             # Legacy per-group feature computation
  run_features_parallel.py    # Legacy parallel liquidity
  test_har_features.py        # HAR feature ablation test
  output/                     # Cached bar series, features, models, metrics
```

## Dollar-Bar-Native Design

**Every feature is computed on the dollar-bar grid, not chronological time.**

The FeatureEngineer (`features/engineer.py`) implements a three-stage pipeline:

```
Stage 1: Build bar series     7.2M dollar bars from aggTrades (cached to parquet)
Stage 2: Compute raw features 18 features on the bar grid
Stage 3: Window + aggregate   bar-grid lags/EMAs -> last bar per window -> 32K rows
```

### Why dollar bars?

A 15-minute window in a quiet market has very different properties than in a crash.
Dollar bars normalize by activity: each bar = $500K of traded volume.
Features computed per bar are comparable across regimes.

### Stage 1: Bar Series Construction

Sequential by month, one month loaded at a time (~1.5GB peak). Cached to `{asset}_bar_series.parquet` (rebuilt only on first run).

Each bar records:
- `close`, `open`, `high`, `low` (prices)
- `duration_s` (seconds to form — fast = high activity)
- `n_trades` (trade count — high = fragmented flow)
- `dollar_volume`, `buy_volume`, `sell_volume`
- `imbalance` = (buy - sell) / total
- `log_return` = log(close / prev_close)
- `rv_bar` = log_return^2
- `vov` = intra-bar return std (volatility of volatility — smooth vs violent bar formation)

Duplicate-millisecond timestamps resolved with nanosecond offsets (multiple bars per ms are real in fast BTC markets).

### Stage 2: Raw Features (18 columns on 7.2M bars)

#### Memory (5 features)
| Feature | Computation | What it measures |
|---------|------------|------------------|
| `hurst_100` | R/S Hurst over last 100 bar returns (subsampled every 50 bars, interpolated) | Vol persistence vs mean-reversion |
| `ffd_rv` | Fractionally differenced bar-level log-prices (d=0.4, numpy convolution) | Stationary vol memory |
| `rv_zscore_1d` | Z-score of rolling RV, window = avg_bars_per_day | Current vol vs last day |
| `rv_zscore_1w` | window = avg_bars_per_day * 7 (capped 50K) | vs last week |
| `rv_zscore_1m` | window = avg_bars_per_day * 30 (capped 200K) | vs last month |

**Dynamic window sizing**: `avg_bars_per_day` is computed from the trailing 30 calendar days of the dataset. "1 day" = N bars (not 96 fixed 15m windows). Adapts to market activity.

#### TSRV (3 features) — asynchronous chronological-to-volumetric bridge
| Feature | Computation | What it measures |
|---------|------------|------------------|
| `tsrv_ffd` | FFD(d=0.4) of forward-filled TSRV label, **shifted by 1 window** | Vol shocks (persistence stripped) |
| `tsrv_mean_1d` | Rolling mean of shifted TSRV over avg_bars_per_day bars | Vol level at 1-day horizon |
| `tsrv_mean_1w` | Rolling mean over avg_bars_per_day * 7 bars (capped 50K) | Vol level at 1-week horizon |

**Point-in-time guarantee**: TSRV labels are shifted by one window before forward-filling to bar timestamps. Bars in [12:00, 12:15) see TSRV from [11:45, 12:00), not [12:00, 12:15). This prevents bar-level look-ahead — the TSRV for the current window isn't known until the window closes.

**Why FFD?** Raw TSRV is highly persistent (autocorrelated). Feeding it directly makes the tree collapse to a glorified linear regression (depth=3). FFD strips persistence, forcing the tree to use microstructure features (VPIN, Hurst, OFI) for the residual signal.

#### Toxicity (5 features)
| Feature | Computation | What it measures |
|---------|------------|------------------|
| `duration_s` | Seconds to form this bar | Activity speed |
| `n_trades` | Trade count in this bar | Flow fragmentation |
| `imbalance` | (buy_vol - sell_vol) / total_vol | Directional pressure |
| `vpin` | Rolling mean |imbalance| over 50 bars | Informed trading probability |
| `vov` | Std of intra-bar tick-by-tick log returns | Volatility of volatility (smooth vs violent bar) |

#### OI (3 features)
| Feature | Source | What it measures |
|---------|--------|------------------|
| `oi_change_5` | HL OI (forward-filled to bar close), pct_change(5 bars) | Position buildup |
| `oi_zscore` | Rolling Z (avg_bars_per_day bars) | OI level vs recent |
| `oi_acceleration` | oi_change.diff(5 bars) | Second derivative |

OI is forward-filled from 1-min HL data to bar close timestamps (no lookahead).

#### Funding (2 features) — sole chronological exception
| Feature | Source | What it measures |
|---------|--------|------------------|
| `funding_basis` | Binance 8h rate, forward-filled to bar close | Perp premium/discount (point-in-time) |
| `time_to_funding` | Seconds until next 00:00/08:00/16:00 UTC settlement | Urgency of funding payment |

No rolling windows on funding. These are exchange-clock-driven events sampled at bar close — the only features that correctly depend on chronological time.

### Stage 3: Windowing + Aggregation

**Windowing on the bar grid** (shift/EMA count bars, not time windows):
- `shift(1)` = previous dollar bar
- `shift(3)` = 3 bars back
- `ewm(span=10)` = 10-bar EMA
- `ewm(span=50)` = 50-bar EMA

18 raw features x 5 variants = 90 features + `n_bars_in_window` = **91 total**.

**Memory-safe**: processes one column at a time. Each column's 7.2M bar-level Series is windowed then grouped to 32K rows immediately. Peak memory: ~100MB per column.

**Aggregation**: for each window, take the LAST dollar bar that closed within it. Also report `n_bars_in_window` (bar count per window — a powerful activity feature, ranked #2 by SHAP).

## Model

**LightGBM** regression with custom **QLIKE** loss.

QLIKE in log-vol space:
```
QLIKE = exp(2(y_true - y_pred)) - 2(y_true - y_pred) - 1
grad  = 2(1 - exp(2(y_true - y_pred)))
hess  = 4 * exp(2(y_true - y_pred))
```

**Cross-validation**: Purged K-Fold (5 folds, 1% embargo) from AFML Ch 7.

**Hyperparameter optimization**: Optuna with TPE sampler, MedianPruner (fold-level pruning), 2-layer efficiency (LGB early stopping + Optuna pruning).

## Evaluation

| Metric | What it tests |
|--------|---------------|
| **QLIKE** | Primary loss (proper volatility metric) |
| **HAR baseline** | Gold standard for RV forecasting (Corsi 2009), dynamic horizons from freq_s |
| **Diebold-Mariano** | Statistical significance of improvement over baselines |
| **Mincer-Zarnowitz** | Calibration: y_true = a + b*y_pred (ideal: a=0, b=1) |
| **RankIC** | Spearman correlation (ranking accuracy) |

## Results (BTC 15m, May 2023 - Mar 2024)

Dollar-bar-native model, 35 pruned features (from 91), 30,278 samples.
Point-in-time TSRV (shifted by 1 window, no bar-level look-ahead).
Optuna-optimized: 150 trials, TPE sampler.

| Metric | Value |
|--------|-------|
| QLIKE | 0.487 |
| vs HAR (Corsi 2009) | -30.4% *** (DM p=0.0002) |
| vs AR(1) | -30.3% *** |
| RankIC | 0.842 |
| Directional accuracy | 49.4% |
| MZ a (ideal 0) | -0.10 |
| MZ b (ideal 1) | 1.12 |
| MZ R^2 | 0.69 |

Optimized hyperparameters:
```
num_leaves=68, max_depth=10, learning_rate=0.0023,
min_data_in_leaf=113, lambda_l1=0.009, lambda_l2=0.62,
colsample_bytree=0.46, subsample=0.8
```

Top SHAP features (pruned model):
1. `duration_s_ema50` (0.162) — smoothed bar formation time (activity proxy)
2. `n_bars_in_window` (0.145) — bars per window (activity level)
3. `duration_s_ema10` (0.054) — short-term activity
4. `n_trades_ema50` (0.046) — smoothed trades per bar (fragmentation)
5. `vov_ema50` (0.027) — smoothed intra-bar vol (microstructure roughness)
6. `tsrv_ffd_lag1` (0.026) — lagged vol shock (FFD'd TSRV)
7. `duration_s` (0.024) — raw bar duration
8. `duration_s_lag3` (0.020) — 3-bar-ago duration
9. `vov_ema10` (0.019) — short-term VoV
10. `tsrv_ffd` (0.019) — current vol shock

**Best model saved**: `vol_model/output/best_15m/` (5 fold models, OOF predictions, SHAP, metrics, feature list).

## Multi-Timeframe Results (BTC, May 2023 - Mar 2024, default hyperparams)

| Metric | 5m (90K samples) | 15m (30K) | 1d (315) |
|--------|-----------------|-----------|----------|
| QLIKE | **0.467** | 0.515 | 0.500 |
| vs HAR | -31.3%*** | -26.7%*** | -43.0% (n.s.) |
| RankIC | **0.856** | 0.827 | 0.454 |
| MZ R^2 | 0.668 | 0.660 | 0.204 |
| MZ calibrated | **YES** | NO | NO |

5m benefits from 3x more training data. 1d is data-starved (315 samples) — HAR improvement not significant.

## Multi-Timeframe Support

The pipeline supports arbitrary timeframes via `--tf` flag (seconds):

```bash
PYTHONPATH=. python vol_model/run_engineer.py --asset BTCUSDT --tf 900   # 15m (default)
PYTHONPATH=. python vol_model/run_engineer.py --asset BTCUSDT --tf 300   # 5m
PYTHONPATH=. python vol_model/run_engineer.py --asset BTCUSDT --tf 86400 # 1d
```

**Dynamic components:**
- TSRV path: auto-resolved from `{asset}_tsrv_{tf}m.parquet` (or manual override)
- Feature files: `{asset}_features_{tf}.parquet` (timeframe-separated, no overwriting)
- HAR horizons: computed from freq_s `[1 bar, 1h, 1d, 1w]` (deduplicated)
- Annualization: `periods_per_year = 365.25 * 24 * 3600 / freq_s`
- Dollar-bar features: timeframe-independent (bar grid is always the same)

**Prerequisite**: TSRV labels must be pre-computed at the target timeframe:
```bash
PYTHONPATH=. python TSRV/run_tsrv.py --symbol BTCUSDT --slow-tf 300  # 5m labels
```

## Data Integrity

### Point-in-time guarantees
- **TSRV features**: Shifted by 1 window before forward-fill. Bars in window [T, T+freq) see TSRV from [T-freq, T), not [T, T+freq).
- **OI features**: Forward-filled from 1-min HL data to bar close timestamps.
- **Funding features**: Forward-filled from 8h rate to bar close timestamps.
- **One-step-ahead**: `X = features[:-1], y = labels[1:]`. Features at window T predict label at window T+1.
- **No feature-label overlap**: Features from [T, T+freq) predict label for [T+freq, T+2*freq).

### Embargo
Purged K-Fold with 1% embargo = 302 windows = 75.5 hours. 300x the label span (1 bar = 15 min).

## Data Sources

| Source | Location | Resolution | Coverage |
|--------|----------|------------|----------|
| aggTrades | `data/binance-aggTrades-parquet/` | Trade-level | May 2023 - Mar 2024 |
| bookTicker | `data/bookTicker_parquet/` | Quote-level | May 16 2023 - Mar 2024 |
| Funding | `data/funding-rates/` | 8-hourly | 2020 - present |
| HL OI | `data/hl-asset-ctxs/cache_15m/` | 1-minute | May 20 2023 - Mar 2024 |
| TSRV labels | `TSRV/output/{ASSET}_tsrv_{TF}.parquet` | Configurable | May 2023 - Mar 2024 |

## Usage

```bash
# 1. Build dollar-bar features (cached after first run)
PYTHONPATH=. python vol_model/run_engineer.py
PYTHONPATH=. python vol_model/run_engineer.py --asset ETHUSDT

# 2. Train model
PYTHONPATH=. python vol_model/run_train.py
PYTHONPATH=. python vol_model/run_train.py --no-shap

# 3. Hyperparameter optimization
PYTHONPATH=. python vol_model/run_optuna.py --n-trials 150 --retrain

# 4. Custom timeframe
PYTHONPATH=. python vol_model/run_engineer.py --asset BTCUSDT --tf 300
PYTHONPATH=. python vol_model/run_train.py --tf 300
```

## Key Design Decisions

1. **Dollar bars as primary index**: All features computed on 7.2M dollar bars. Time windows are only the final aggregation layer for label alignment.
2. **Dynamic Z-score windows**: `avg_bars_per_day` computed from trailing 30 days, translating "1 day/week/month" to bar counts instead of fixed time windows.
3. **TSRV as asynchronous bridge**: Pre-computed TSRV labels forward-filled to bar timestamps with 1-window shift for strict point-in-time. FFD strips persistence to prevent tree collapse.
4. **VoV (Volatility of Volatility)**: Intra-bar return std captures how smoothly each dollar bar formed — microstructure context the tree uses alongside macro vol signals.
5. **Funding as point-in-time sample**: No rolling windows on funding. Just `funding_basis` (current rate) and `time_to_funding` (seconds to settlement) sampled at bar close.
6. **Column-by-column windowing**: Processes each of 18 features independently on 7.2M bars, immediately aggregating to ~32K windows. Peak memory ~100MB instead of 3.7GB.
7. **Bar series caching**: 7.2M-bar parquet cached after first build. Subsequent runs start from features.
8. **QLIKE loss**: Proper volatility loss that penalizes under-prediction (more dangerous) more than over-prediction.
9. **SHAP-based pruning**: 91 windowed features pruned to 35 based on SHAP importance. Removes noise, improves generalization.
10. **Optuna TPE + MedianPruner**: Two-layer efficiency — LGB early stopping kills bad boosting rounds, Optuna fold-level pruning kills bad hyperparams after 2 folds.
11. **Multi-timeframe**: HAR horizons, TSRV paths, and annualization computed dynamically from `freq_s`.

## Dependencies

- lightgbm >= 4.0
- optuna >= 4.0
- shap >= 0.50
- pandas, numpy, scipy
- src/afml/ (PurgedKFold, fracdiff, evaluation)
- TSRV/ (data_loader, sampling)
