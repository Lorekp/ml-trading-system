# Multi-Timeframe Threshold Sweep Results

## Overview

Walk-forward backtest results across all timeframes with trained expanding-window LightGBM
models. Each TF was swept at 7 entry thresholds (10–40 vol pts). The non-compound (NC)
Sharpe ratio is the primary figure of merit — it measures signal quality at fixed notional
($200K reference) and is not inflated by compounding. See `scripts/sweep_all_tfs.py` for
the unified sweep script. Individual TF scripts: `scripts/sweep_5m_thresholds.py`,
`scripts/sweep_30m_thresholds.py`.

**Data range**: 2022-02 to 2026-03 (earliest common period across all TF models)
**Equity**: $200K reference notional (non-compound)
**Execution model**: `realistic_taker` + `composite_v2` sizing, IOC cap 2×, delay 1s
**Thresholds**: short=N, long=N+5 (asymmetric — longs need 5vp extra edge to offset slippage)

---

## 30m — Completed 2026-04-13

Predictions from `vol_model/output/BTCUSDT_expanding_30m_predictions.parquet` (51 months OOS).
Vol-arb data: `vol_arb/output/BTC_vol_arb_svi_30m.parquet`.

| threshold (S/L) | trades  | NC Sharpe | NC PnL     | NC MaxDD | Win Rate | Prof Factor |
|-----------------|---------|-----------|------------|----------|----------|-------------|
| 10/15           | 13,009  | 5.36      | $491,750   | -5.4%    | —        | 1.56        |
| 15/20           | 18,912  | 6.70      | $1,340,618 | -3.2%    | —        | 1.93        |
| **20/25**       | **13,652** | **7.33** | **$1,329,342** | **-4.0%** | **50.4%** | **2.28** |
| 25/30           | 8,487   | 7.25      | $1,089,642 | -3.1%    | —        | 2.67        |
| 30/35           | 5,138   | 7.21      | $842,869   | -2.1%    | —        | 3.05        |
| 35/40           | 3,198   | 6.65      | $595,425   | -1.7%    | —        | 3.35        |
| 40/45           | 2,046   | 6.94      | $467,817   | -1.6%    | —        | 3.79        |

**Best: threshold=20/25, NC Sharpe 7.33** (compound Sharpe 21.5 — 12.5× lift at $200K start)

Tearsheet at 20/25:
- NC PnL: $1,329,342 (664% on fixed notional over 4 years)
- NC MaxDD: -4.0%
- Profit factor: 2.20 (NC), 2.28 (compound)
- Win rate: 50.4%

**Notes**:
- Profit factor rises monotonically with threshold (higher-conviction-only trades are cleaner)
- NC Sharpe peaks at 20/25 — the threshold beyond which trade count falls faster than PnL improves
- 15/20 has higher absolute NC PnL ($1.34M) but slightly lower Sharpe due to more low-edge trades

---

## 5m — Pending

Script: `scripts/sweep_5m_thresholds.py`
Predictions: `vol_model/output/BTCUSDT_expanding_5m_predictions.parquet`
Vol-arb: `vol_arb/output/BTC_vol_arb_svi_5m.parquet` (needs build)

*To be updated after `scripts/sweep_all_tfs.py` completes.*

---

## 60m — Pending

Script: `scripts/sweep_all_tfs.py`
Predictions: `vol_model/output/BTCUSDT_expanding_60m_predictions.parquet` (needs extraction)
Vol-arb: `vol_arb/output/BTC_vol_arb_svi_60m.parquet` (needs build)

*To be updated after `scripts/sweep_all_tfs.py` completes.*

---

## 240m — Pending

Script: `scripts/sweep_all_tfs.py`
Predictions: `vol_model/output/BTCUSDT_expanding_240m_predictions.parquet` (needs extraction)
Vol-arb: `vol_arb/output/BTC_vol_arb_svi_240m.parquet` (needs build)

*To be updated after `scripts/sweep_all_tfs.py` completes.*

---

## 480m — Pending

Script: `scripts/sweep_all_tfs.py`
Predictions: `vol_model/output/BTCUSDT_expanding_480m_predictions.parquet` (needs extraction)
Vol-arb: `vol_arb/output/BTC_vol_arb_svi_480m.parquet` (needs build)

*To be updated after `scripts/sweep_all_tfs.py` completes.*

---

## 720m — Pending

Script: `scripts/sweep_all_tfs.py`
Predictions: `vol_model/output/BTCUSDT_expanding_720m_predictions.parquet` (needs extraction)
Vol-arb: `vol_arb/output/BTC_vol_arb_svi_720m.parquet` (needs build)

*To be updated after `scripts/sweep_all_tfs.py` completes.*

---

## Cross-TF Summary (to be completed)

| TF   | Best thresh | NC Sharpe | NC PnL | NC MaxDD |
|------|-------------|-----------|--------|----------|
| 5m   | TBD         | TBD       | TBD    | TBD      |
| 30m  | 20/25       | 7.33      | $1.33M | -4.0%    |
| 60m  | TBD         | TBD       | TBD    | TBD      |
| 240m | TBD         | TBD       | TBD    | TBD      |
| 480m | TBD         | TBD       | TBD    | TBD      |
| 720m | TBD         | TBD       | TBD    | TBD      |

---

## Interpretation Guide

- **NC Sharpe** is the authoritative metric: daily PnL at fixed $200K notional, annualised.
  Compound Sharpe is inflated ~12× at this equity level and is not a reliable signal-quality proxy.
- **Profit factor** rising with threshold means the edge distribution is right-skewed: a few
  very high-edge trades dominate PnL. Lower thresholds include more marginal trades.
- **Win rate ~50%** is expected for a delta-neutral straddle strategy: trades are roughly 50/50
  directionally but with asymmetric payoffs (winners > losers → profit factor > 1).
- **Optimal threshold** balances trade count (Sharpe denominator) vs signal quality (Sharpe numerator).
  Empirically peaks in the 20–30vp range for the 30m model.
