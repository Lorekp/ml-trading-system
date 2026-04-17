# TSRV - Two-Scale Realized Volatility

Noise-corrected realized volatility estimation from tick-level crypto trade data, using the Zhang, Mykland & Ait-Sahalia (2005) two-scale estimator with K-subgrid averaging and jump-robust decomposition (BPV, MedRV).

## Overview

Standard realized volatility (sum of squared returns) is biased at high frequencies due to microstructure noise (bid-ask bounce, discrete price ticks). TSRV corrects this by combining two scales:

- **Fast scale (all ticks)**: captures both true volatility AND noise
- **Slow scale (dollar sub-bars)**: captures mostly true volatility, less noise

The correction subtracts the noise estimate, with K-subgrid averaging for variance reduction:

```
K = ceil(n_sub_bars^{2/3})          # optimal subgrid count (Zhang 2005)

For k = 0..K-1:
    RV_slow^{(k)} = sum of squared returns on sub-bar indices [k::K]

RV_slow_avg = mean(RV_slow^{(0)}, ..., RV_slow^{(K-1)})

TSRV_raw = RV_slow_avg - (n_bar_avg / n_ticks) * RV_fast

TSRV = TSRV_raw / (1 - n_bar_avg / n_ticks)    # finite-sample bias correction
```

Jump decomposition via Bipower Variation (Barndorff-Nielsen & Shephard 2004) and Median Realized Variance (Andersen, Dobrev, Schaumburg 2012) separates continuous volatility from discontinuous jumps. Directional jump fractions capture the asymmetric risk profile of crypto markets (down-jumps = liquidation cascades, up-jumps = sell-wall absorption).

## Quick Start

```bash
cd ml-trading-system

# 1. Signature plot: find optimal sampling frequency for BTC
PYTHONPATH="." python TSRV/run_btc.py --n-dates 30

# 2. TSRV time series: compute 15m noise-corrected RV for BTC
PYTHONPATH="." python TSRV/run_tsrv_series.py --asset BTCUSDT --tf 900

# 3. Same for ETH at 1h bars
PYTHONPATH="." python TSRV/run_tsrv_series.py --asset ETHUSDT --tf 3600

# 4. Skip calibration if you already know the threshold
PYTHONPATH="." python TSRV/run_tsrv_series.py --asset BTCUSDT --threshold 500000

# 5. Run label quality diagnostic
PYTHONPATH="." python TSRV/diagnostic.py
```

## Architecture

```
TSRV/
 config.py            All parameters (Pydantic). One place to configure everything.
 data_loader.py       Load bookTicker (quotes) or aggTrades (trades), one day at a time.
 sampling.py          Resample ticks into time bars, tick bars, or dollar bars.
 realized_vol.py      RV estimators: naive, noise variance, TSRV, annualization.
 jump_robust.py       BPV, MedRV, signed semivariance, directional jump decomposition.
 signature_plot.py    Compute signature curves, find knee (KneeLocator), plot.
 tsrv_series.py       Compute TSRV time series with K-subgrid averaging + jump decomposition.
 diagnostic.py        Label quality diagnostic (negative freq, variance reduction, jump contamination).
 run_btc.py           Entry point: signature plots (time + tick + dollar bars).
 run_tsrv_series.py   Entry point: calibrate threshold + compute full TSRV series.
```

## Module Reference

### `jump_robust.py` - Jump-Robust Estimators

| Function | Formula | Reference |
|----------|---------|-----------|
| `bipower_variation(log_prices)` | `(pi/2) * (n/(n-1)) * sum(\|r_{i-1}\| * \|r_i\|)` | BNS 2004 |
| `median_rv(log_prices)` | `c * (M/(M-2)) * sum(median(\|r_{i-1}\|, \|r_i\|, \|r_{i+1}\|)^2)` | ADS 2012 |
| `realized_semivariance(log_prices)` | `(RS_down, RS_up)` where RS = sum(r^2 * 1(sign))` | BNS-KS 2010 |
| `directional_jump_fractions(rv, bpv, rs_down, rs_up)` | `max(RS_side - BPV/2, 0) / rv` per side | Derived |
| `subgrid_rv_slow(log_prices, indices, K)` | K-offset subgrid averaging | Zhang 2005 |
| `decompose_jump_continuous(rv, bpv)` | `(min(rv,bpv), max(rv-bpv,0))` | ABD 2007 |

### `tsrv_series.py` - TSRV Time Series

| Function | Description |
|----------|-------------|
| `compute_tsrv_window(prices, dollar_vol, threshold)` | Single-window TSRV with K-subgrid averaging + BPV/MedRV decomposition + directional jumps. |
| `calibrate_sub_bar_threshold(trade_dir, symbol, thresholds, slow_tf_s, config)` | Find optimal dollar sub-bar threshold via signature plot within windows. |
| `compute_tsrv_series(trade_dir, symbol, threshold, slow_tf_s, config)` | Full time series with all columns. |

**Output columns per window:**

| Column | Description |
|--------|-------------|
| `tsrv` | K-subgrid-averaged, bias-corrected TSRV (clipped >= 0) |
| `rv_fast` | All-tick realized variance |
| `rv_slow` | Single-grid dollar sub-bar RV |
| `bpv` | Bipower Variation on sub-bar returns (jump-robust continuous vol) |
| `medrv` | Median RV on sub-bar returns (more robust to clustered jumps) |
| `continuous_var` | min(rv_slow, bpv) -- continuous component |
| `jump_var` | max(rv_slow - bpv, 0) -- jump component |
| `jump_fraction` | jump_var / rv_slow |
| `rs_down` | Downside realized semivariance |
| `rs_up` | Upside realized semivariance |
| `down_jump_fraction` | max(rs_down - bpv/2, 0) / rv_slow |
| `up_jump_fraction` | max(rs_up - bpv/2, 0) / rv_slow |
| `n_ticks`, `n_sub_bars`, `n_dash` | Counts |
| `rv_ann`, `vol_ann` | Annualized from TSRV |

### `diagnostic.py` - Label Quality Diagnostic

Quantifies estimator quality before making changes:
- Negative TSRV frequency (pre-clip)
- Subgrid variance reduction (simplified vs proper TSRV)
- Noise-to-signal ratio per window
- Jump contamination (BPV/MedRV)
- Finite-sample bias magnitude
- Label distribution (normality, autocorrelation, long memory)

### `realized_vol.py` - RV Estimators

| Function | Formula | Notes |
|----------|---------|-------|
| `realized_variance(prices)` | `sum(R_i^2)` where `R_i = ln(P_i/P_{i-1})` | Raw RV, no noise correction |
| `noise_variance(prices)` | `RV_all / (2n)` | Hansen & Lunde noise estimator |
| `tsrv(prices, K)` | `RV_slow - (n_bar_slow/n_bar_all) * RV_all` | K subgrids, default `K = ceil(n^{2/3})` |
| `annualize_rv(rv, period_days)` | `sqrt(rv * 365.25 / period_days)` | Convert to annualized vol |

## Diagnostic Results (BTC 15m, 2000 sampled windows)

| Test | Result | Implication |
|------|--------|-------------|
| Negative TSRV | 0.00% | Non-negativity is not a problem |
| Subgrid variance reduction | 4.4% | Modest improvement from K-subgrid averaging |
| Noise ratio > 0.5 | 92% of windows | TSRV noise correction is critical |
| Jump fraction (BPV) | mean=9.1%, median=5.3% | ~10% of variance from jumps |
| RV/BPV > 1.5 (large jumps) | 3.1% of windows | Moderate jump contamination |
| Bias correction | 0.044% | Negligible (n_ticks >> n_sub_bars) |
| Label ACF(20) | 0.53 | Long memory (slow decay) |

## Results (BTC 15m, $500K threshold)

| Metric | Value |
|--------|-------|
| Windows | 32,160 |
| Date range | 2023-05-01 to 2024-03-31 |
| TSRV positive | 100% |
| Mean annualized vol | 33.5% |
| Median annualized vol | 25.4% |
| Mean sub-bars/window | 224 |
| Mean ticks/window | 12,188 |
| Mean jump fraction | 9.1% |
| Windows with jump_fraction > 15% | 21.5% |

## Data Sources

### bookTicker (quotes)
- Path: `data/bookTicker_parquet/{SYMBOL}/{YYYY-MM-DD}.parquet`
- Columns: `best_bid_price`, `best_ask_price`, `best_bid_qty`, `best_ask_qty`, `transaction_time` (ms)
- 162 assets, May 2023 - Mar 2024, 4.5M-33M ticks/day
- Used for: time-based and tick-bar signature plots (mid-price has minimal microstructure noise)

### aggTrades (trades)
- Path: `data/binance-aggTrades-parquet/{SYMBOL}/{YYYY-MM}.parquet`
- Columns: `agg_trade_id`, `price`, `quantity`, `first_trade_id`, `last_trade_id`, `transact_time` (ms), `is_buyer_maker`, `dollar_volume`
- 162 assets, May 2023 - Mar 2024, 7B total trades
- Used for: dollar bar signatures and TSRV time series

### Deribit Option Trades
- Path: `data/deribit/trades/{CURRENCY}_option/{YYYY-MM-DD}.parquet`
- Columns: `timestamp`, `instrument_name`, `iv`, `index_price`, `price`, `mark_price`, `direction`, `amount`, `trade_seq`, `trade_id`, `tick_direction`
- BTC: 1,832 daily files, 20.2M trades, 743 MB
- ETH: 1,832 daily files, 14.7M trades, 535 MB
- Apr 2021 - Apr 2026, full tick-level coverage (all trades per day)
- Used for: ATM IV, Black-Scholes Greeks, vol-arb signal

### Deribit DVOL (Implied Volatility Index)
- Path: `data/deribit/dvol/{CURRENCY}_dvol_60m.parquet`
- Columns: `timestamp`, `open`, `high`, `low`, `close` (vol %)
- BTC + ETH, hourly resolution, Apr 2021 - Apr 2026, 43,945 rows each
- Used for: composite ATM 30-day IV, vol-arb spread baseline

### Deribit Candles
- Path: `data/deribit/candles/{INSTRUMENT}_5m.parquet`
- Columns: `timestamp`, `open`, `high`, `low`, `close`, `volume`
- BTC-PERPETUAL + ETH-PERPETUAL, 5-min resolution, Apr 2021 - Apr 2026
- Used for: underlying price history

## Key Academic References

| Paper | Year | What it provides |
|-------|------|-----------------|
| Zhang, Mykland, Ait-Sahalia | 2005 | TSRV: K-subgrid averaging, optimal K, bias correction |
| Barndorff-Nielsen, Shephard | 2004 | BPV: jump-robust continuous variance estimation |
| Barndorff-Nielsen, Shephard | 2006 | Jump test statistic (ratio test) |
| Andersen, Dobrev, Schaumburg | 2012 | MedRV: more robust to clustered jumps than BPV |
| Barndorff-Nielsen, Kinnebrock, Shephard | 2010 | Signed realized semivariance |
| Andersen, Bollerslev, Diebold | 2007 | Max/min jump-continuous decomposition |

## Vol-Arb Pipeline

The `vol_arb/` package merges TSRV labels with Deribit options data for volatility arbitrage backtesting:

```
vol_arb/
 config.py            VolArbConfig (Pydantic). ATM filters, Greek params.
 greeks.py            Vectorized Black-Scholes Greeks engine (r=0). Pure numpy/scipy.
 options_loader.py    Load + parse Deribit trades, instrument name extraction, ATM/DTE filter.
 pipeline.py          merge_asof orchestration: options + DVOL + TSRV features.
 backtester.py        VolArbBacktester: delta-neutral straddle simulator, Greeks PnL attribution.
 run_pipeline.py      Entry point: build merged dataset.
 run_backtest.py      Entry point: run execution simulator.
```

### Merged Dataset (`vol_arb/output/BTC_vol_arb.parquet`)

32,160 rows (15-min bars), 33 columns. 100% IV coverage, 100% DVOL coverage.

| Column Group | Columns |
|-------------|---------|
| TSRV | `tsrv`, `rv_fast`, `rv_slow`, `bpv`, `medrv`, `continuous_var`, `jump_var`, `jump_fraction`, `rs_down`, `rs_up`, `down_jump_fraction`, `up_jump_fraction`, `n_ticks`, `n_sub_bars`, `n_dash`, `rv_ann`, `vol_ann` |
| Options | `iv`, `strike`, `option_type`, `index_price`, `moneyness`, `dte_days`, `T_years`, `delta`, `gamma`, `vega`, `theta` |
| DVOL | `dvol_close` |
| Derived | `rv_pct`, `rv_iv_spread`, `rv_dvol_spread`, `iv_premium` |

### Quick Start

```bash
# 1. Build merged dataset
PYTHONPATH="." python vol_arb/run_pipeline.py

# 2. Run backtest (default params)
PYTHONPATH="." python vol_arb/run_backtest.py

# 3. Run with custom threshold
PYTHONPATH="." python vol_arb/run_backtest.py --threshold 20.0
```

## Downloader

The `downloader/` package fetches historical data from the Deribit API:

```
downloader/
 config.py            DownloadConfig (Pydantic). Assets, dates, rate limits, parallelism.
 client.py            DeribitClient: auth, retries, backoff, error 10028 detection.
 rate_limiter.py      Token-bucket rate limiter mirroring Deribit's credit system.
 runner.py            Parallel orchestrator (ThreadPoolExecutor) with ETA.
 tasks/
   trades.py          Option trades (paginated by timestamp advancement).
   dvol.py            DVOL index (chunked by day).
   candles.py         TradingView chart data (chunked by 3-day windows).
   historical_vol.py  Daily realized vol (single call).
   mark_price.py      Mark price history (endpoint non-functional).
```

```bash
# Download all BTC+ETH option trades, daily, full tick coverage
PYTHONPATH="." python -m downloader --endpoints trades --start 2021-04-01 --end 2026-04-07

# Download DVOL hourly
PYTHONPATH="." python -m downloader --endpoints dvol --start 2021-04-01 --end 2026-04-07
```

**Rate limits:** 50,000 credit pool, 10,000 credits/sec refill, 500 credits/request. Authenticated = 20 req/s sustained. 4 workers, timestamp-based pagination, zero rate limit waits at full throughput.

## Dependencies

- `pandas`, `numpy` -- data manipulation
- `scipy` -- statistical tests (diagnostic), Black-Scholes Greeks
- `kneed` -- knee/elbow detection in signature plots
- `matplotlib` -- plotting
- `pydantic` -- configuration
- `requests` -- Deribit API client
