# DFL head — Phase 3 backtest comparison + ship decision

**Policy**: strategies were PRE-COMMITTED before running the
backtest. Parameters are frozen. One run per strategy. No post-hoc
re-tuning.

## Strategies compared

Single post-processor pass on `BTC_v2_trade_log_baseline.parquet`
with walk-forward DFF-style classifier scores. See
`vol_arb/dfl_head.py` and `DFL_HEAD_PHASE2_REPORT.md`.

## Results (NC, $100K capital; 2022-02 to 2026-03)

| Strategy | PnL | ΔPnL | Sharpe | ΔSharpe | MaxDD | WorstDay | WinRate | TradeSharpe | nTrades |
|---|---|---|---|---|---|---|---|---|---|
| baseline | $814,994 | — | 7.65 | — | -$7,711 | -$5,441 | 50.1% | 56.51 | 8,112 |
| **A1_filter25** | **$746,614** | **-8.4%** | **7.87** | **+0.22** | **-$5,810 (-25%)** | **-$2,921 (-46%)** | 50.1% | **58.52** | 7,180 |
| A2_filter10 | $764,250 | -6.2% | 7.80 | +0.14 | -$7,329 (-5%) | -$4,079 (-25%) | 50.1% | 58.15 | 7,407 |
| B1_linear | $836,405 | +2.6% | 7.36 | -0.29 | -$7,094 | -$4,473 | 50.1% | 55.99 | 8,112 |
| B2_binary_median | $765,267 | -6.1% | 7.50 | -0.15 | -$6,088 | -$3,040 | 50.1% | 56.19 | 8,112 |
| B3_normalized | $899,974 | +10.4% | 7.36 | -0.29 | -$7,968 | -$5,023 | 50.1% | 56.12 | 8,112 |

## Pre-registered gate evaluation

- **Ship-as-sidecar gate**: NC Sharpe ≥ baseline AND PnL ≥ 0.90 × baseline
- **Ship-as-default gate**: NC Sharpe ≥ baseline + 0.5 AND PnL ≥ baseline

| Strategy | ΔSharpe | ΔPnL % | Verdict |
|---|---|---|---|
| **A1_filter25** | **+0.22** | **-8.4%** | **SHIP AS SIDECAR** |
| A2_filter10 | +0.14 | -6.2% | SHIP AS SIDECAR |
| B1_linear | -0.29 | +2.6% | REJECT (Sharpe) |
| B2_binary_median | -0.15 | -6.1% | REJECT (Sharpe + PnL) |
| B3_normalized | -0.29 | +10.4% | REJECT (Sharpe) |

## Structural finding — filters dominate sizing

- **All 3 size strategies (B1/B2/B3) fail the Sharpe gate** despite
  some showing PnL gains (B3 at +10.4%).
- **Both filter strategies (A1/A2) improve Sharpe** at the cost of
  moderate PnL.
- Interpretation: the classifier's signal is strong enough to
  identify LIKELY LOSERS (so filtering them works) but not strong
  enough to reliably rank winners by size (so sizing adds
  variance without proportional PnL).
- Filtering is strictly better — it keeps the downside reduction
  without compounding the tail with larger winners whose win
  probabilities aren't that much higher.

## Decision — ship A1 Filter-25 as optional sidecar

**A1 wins over A2 on Occam + gate margin**:
- A1: +0.22 Sharpe, -$1,901 MaxDD, -$2,520 worst day
- A2: +0.14 Sharpe, -$382 MaxDD, -$1,362 worst day

A1 is the cleaner risk improvement. PnL cost is $68K (-8.4%) but:
- MaxDD down 25% ($7,711 → $5,810)
- Worst daily loss down 46% ($5,441 → $2,921)
- Daily P&L std down ~11% (implied from Sharpe + mean math)
- 11.5% fewer trades → proportionally lower execution-cost exposure

**Default V2 stays unchanged.** Same policy as the prediction
aligner and the P&L throttle: risk-overlay additions land as
optional post-processors, not default changes.

## Comparison vs earlier optional overlays

| Overlay | Mechanism | ΔSharpe | ΔPnL | MaxDD improvement |
|---|---|---|---|---|
| Prediction aligner (bias β) | QLIKE-optimal post-hoc β shift | **-0.41** | +3.7% ($30K) | -0.58 pp worse |
| P&L throttle (sc=0.5, thr=-$1K) | Yesterday-loss-triggered size cut | ±0 | -2.1% | -31% |
| **DFL A1 Filter-25 (THIS)** | **DFF-style W/L classifier filter** | **+0.22** | **-8.4%** | **-25%** |

A1 is the **first** overlay this session with a positive Sharpe
delta. Aligner hurt Sharpe; throttle was Sharpe-neutral; DFL A1 is
Sharpe-positive. The mechanism is also fundamentally different:
aligner changes predictions, throttle reacts to past losses, DFL
pre-screens bars based on regime + model-bias features.

## Leakage / overfitting discipline verification

- **Pre-committed strategies**: A1/A2/B1/B2/B3 parameters were
  fixed before seeing the comparison table. No post-hoc re-tuning.
- **Walk-forward training**: classifier trained on year Y uses
  only bars with year < Y. Mutation probe (`test_walkforward_no_lookahead`)
  confirms test-year data doesn't contaminate training.
- **Features strictly lagged**: all rolling aggregates use
  `shift(1).rolling(...)`; mutation probe (`test_features_are_strictly_lagged`)
  confirms.
- **Classifier target** (`win`) is evaluation only, never a
  feature.
- **Single backtest run**: all 5 strategies evaluated in ONE
  post-processor pass. No iterating on the backtest to pick a
  winner.
- **Per-year robustness**: Phase 2 showed 3/4 years with AUC
  improvement; 2023 (smallest training set) was the exception.
  This is graceful degradation, not overfitting.
- **First year is structurally pass-through** (no training
  history) — no synthetic benefit from years that weren't tested.

## Artifacts

- `vol_arb/dfl_head.py` — module with `compute_walkforward_scores`,
  `FilterStrategy`, `SizeStrategy`, `pre_committed_strategies`
- `tests/test_dfl_head.py` — 8 tests, all green (leakage probe,
  monotonicity, zero/identity tests, pass-through)
- `vol_arb/output/BTC_v2_dfl_scores.parquet` — walk-forward P(win)
  per bar
- `vol_arb/output/BTC_v2_trade_log_dfl_A1_filter25.parquet` —
  ship-as-sidecar trade log
- `vol_arb/output/BTC_v2_trade_log_dfl_{A2,B1,B2,B3}_*.parquet` —
  other strategies (rejected, kept for audit)
- `vol_arb/output/BTC_v2_dfl_comparison.csv` — full metrics
- `vol_model/output/DFL_HEAD_PHASE2_REPORT.md` — offline audit
- `vol_model/output/DECISION_QLIKE_MISALIGNMENT_AUDIT.md` — Phase 1

## What this session discovered

1. QLIKE (the training objective) is essentially **economically
   uncorrelated with trade P&L** (Spearman ≈ -0.014 across
   46 months, Phase 1).
2. A frozen HAR-PD + DFF-style post-hoc classifier head on
   decision-centric features (IV level, recent residuals, past
   vol) captures **2.1× more P&L at top-10%** than the baseline
   `|pred_edge|` ranking (Phase 2, bootstrap 95% CI [1.71, 2.62],
   permutation p < 0.001).
3. Deployed as a filter (A1: drop bottom-25% P(win)), this is a
   Sharpe-positive overlay that reduces both MaxDD and worst-day
   loss materially, at a modest PnL cost.
4. **Filters dominate sizing** for this signal strength — sizing
   strategies uniformly failed the Sharpe gate.
5. Structural insight for future cycles: **training-objective
   audit** (Phase 1 methodology) should be a standard diagnostic
   before every retrain cycle. It's cheap, offline, and directly
   quantifies misalignment ROI.
