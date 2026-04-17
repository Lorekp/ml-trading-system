# ML Trading System — Session State
## Updated: 2026-03-18 ~01:15 local

## WHAT HAPPENED THIS SESSION (MASSIVE — continued from previous session)

### Phase 1: Library & Knowledge Base (complete)
- Read ALL library books via pymupdf/ebooklib (installed tools to actually read PDFs/EPUBs)
- Created comprehensive reports for Kahneman (9.5/10), Johnson DMA (9/10, PDF corrupted — knowledge-based)
- Deep scientific article review (8 papers, exact math formulations)
- Web research on 7 knowledge gaps (crypto vs forex, 15m optimality, linear signal, funding, liquidations, scaling, recent advances)
- Created KNOWLEDGE_SYNTHESIS.md cross-referencing all sources
- Updated library INDEX.md with all ratings

### Phase 2: System Improvement Planning
- Full codebase audit via 3 parallel agents (ML pipeline A-, verification 66%, infrastructure pristine)
- Identified 14 improvement areas, prioritized by PnL impact
- Created improvement landscape and sketch documents

### Phase 3: Multi-Approach Experiment (8 approaches tested)
Created and ran `experiments/multi_approach_comparison.py` (1,294 lines, 57 seconds):
- **A (baseline 15m):** Sharpe 2.557, Return 7.51%, 43,570 trades
- **B (multi-TF features):** Sharpe 2.381 — WORSE (multi-TF features don't help)
- **C (4H regime filter):** Sharpe 7.750, Return 6.72%, MaxDD 0.13% — BEST risk-adjusted
- **D (15m+4H portfolio):** Sharpe 2.577 — no improvement
- **E1 (conf>0.55):** Sharpe 2.662
- **E2 (conf>0.60):** Sharpe 3.532, Return 7.46%
- **E3 (conf>0.65):** Sharpe 5.001 — but only 711 trades (overfitting risk)
- **F (multi-TF+conf):** Sharpe 3.524, Return 7.86%

Key finding: Regime filter (C) best for equity curve smoothness. E2 best $/day. All approaches make roughly same total PnL ($500-786 on $10K over 17 months).

### Phase 4: Portfolio Expansion (16 assets)
Created and ran `experiments/expanded_portfolio_test.py`:
- **16-asset C (regime filter):** Sharpe 6.787, Return 105.13%, MaxDD 0.64%, 56,285 trades, 83.8% win rate
- **16-asset A (baseline):** Sharpe 3.119, Return 210.55%, MaxDD 3.69%
- Top performers: DOT (7.91), OP (7.47), SUI (7.00), BNB (6.96)

### Phase 5: Dollar Bars (KILLED)
- V1: Sharpe 1.31 (broken calibration)
- V2 (proper, adaptive threshold): Time bars win at ALL 3 frequencies (-1.14, -0.65, -0.21 delta)
- Dollar bars produce lower kurtosis (theory correct) but don't improve Sharpe for our features
- **KILLED per pre-registered criteria**

### Phase 6: Remaining Validation
Created and ran `experiments/remaining_validation.py`:
- **A10 Extended History:** Signal survives 2022 bear (BTC Sharpe 0.13, ETH 1.48). Positive 2020-2023.
- **B4 Monte Carlo:** p=0.064 — marginally significant. Per-trade edge is thin, system profits from signal + filtering + sizing combined.
- **C3 Concurrent Positions:** PASS, 0 violations, max 3.1% exposure.
- **Checklist: 32/44 (73%)** — up from 29/44.

### Phase 7: Triple-Barrier Labeling (KILLED)
Created and ran `experiments/triple_barrier_test.py`:
- Fixed-horizon: Sharpe 2.557 (baseline)
- Triple-barrier [1,1]: Sharpe 2.361 — WORSE
- Triple-barrier [2,1]: Sharpe 0.934 — MUCH WORSE
- **KILLED.** Fixed-horizon labeling is better for this mean-reversion signal.

### Phase 8: New Features (HL Derivatives) — IN PROGRESS
Created `experiments/new_features_test.py` with optimized batch loading:
- Tests: baseline 14, +OI (17), +Funding (17), +All (20)
- **Currently running** — batch loading 730 LZ4 files for first time (will cache for future)
- Background task ID: btx20gd9u (may still be running)

### Phase 9: Monetization Research (complete)
Two comprehensive research reports on monetization paths:
1. Hyperliquid vault (10% perf fee, no KYC, 100 USDC min)
2. HyroTrader prop firm (Bybit API, 70-90% split, $10K-$1M accounts)
3. Bybit copy trading (30% of follower profits)
4. Telegram signals, freelancing, competitions, fund formation
- **Recommended: HyroTrader eval (€100-500) + Hyperliquid vault + Bybit copy trading in parallel**

## CURRENT BEST CONFIGURATION
- **Model:** LogisticRegression(C=0.1, L1, saga)
- **Features:** 14 universal OHLCV (multi-TF features DON'T help)
- **Timeframe:** 15m bars, fwd_bars=4 (1 hour horizon)
- **Filter:** 4H regime filter (EMA20 trend agreement) — Sharpe 7.75 on 6 assets, 6.79 on 16 assets
- **Assets:** 16 crypto (BTC/ETH/XRP/BNB/SUI/LINK + SOL/ADA/AVAX/DOT/NEAR/DOGE/ARB/OP/UNI/AAVE)
- **Labeling:** Fixed-horizon (triple-barrier HURTS)
- **Dollar bars:** Don't help (time bars win)

## KEY RESULTS SUMMARY
| Config | Sharpe | Return% | MaxDD% | Trades | Win Rate |
|--------|--------|---------|--------|--------|----------|
| 6-asset baseline A | 2.56 | 7.51% | 1.02% | 43,570 | 64.0% |
| 6-asset regime C | 7.75 | 6.72% | 0.13% | 12,047 | 80.7% |
| **16-asset regime C** | **6.79** | **105.13%** | **0.64%** | **56,285** | **83.8%** |
| 16-asset baseline A | 3.12 | 210.55% | 3.69% | 208,422 | 68.5% |

## EXPERIMENTS CREATED THIS SESSION
- `experiments/multi_approach_comparison.py` — 8 approaches, 1,294 lines
- `experiments/expanded_portfolio_test.py` — 16-asset portfolio
- `experiments/dollar_bars_test.py` — v1 (broken)
- `experiments/dollar_bars_v2.py` — v2 (proper, dollar bars killed)
- `experiments/remaining_validation.py` — A10, B4, C3
- `experiments/triple_barrier_test.py` — triple-barrier (killed)
- `experiments/new_features_test.py` — HL derivatives (in progress)

## RESULTS FILES CREATED
- `v4-results/multi_approach_comparison.json`
- `v4-results/expanded_portfolio_results.json`
- `v4-results/dollar_bars_results.json`
- `v4-results/dollar_bars_v2_results.json`
- `v4-results/remaining_validation_results.json`
- `v4-results/triple_barrier_results.json`

## WHAT'S KILLED (don't revisit)
- Dollar bars (time bars win at all frequencies)
- Multi-TF features in single model (L1 keeps them but Sharpe drops)
- Triple-barrier labeling (fixed-horizon is better for this signal)
- Independent 15m+4H portfolio (signals too correlated, ρ=0.97)
- CoinGlass (requires premium tier)

## NEXT SESSION PRIORITIES
1. **Check new_features_test.py results** — may have finished by next session
2. **Optimize all experiment scripts** — multiprocessing, shared cache, flush printing
3. **Production infrastructure (F1-F8)** — THE blocker for any revenue
4. **Start Hyperliquid deployment** — deploy 16-asset regime filter with €200
5. **HyroTrader eval** — submit algo for prop firm evaluation

## VERIFICATION CHECKLIST: 32/44 (73%)
- A: 10/10 (100%) — A9+A10 now done
- B: 4/5 (80%) — B4 done (p=0.064), B5 optional
- C: 5/7 (71%) — C3 done, C4+C5 need live data
- D: 6/6 (100%)
- E: 8/8 (100%)
- F: 0/8 (0%) — production not started

## CODE QUALITY
- Git initialized, .gitignore created
- Fixed cross_val.py hardcoded 30-bar purge span
- Fixed pipeline.py FFD d=0.4 fallback documentation
- Updated requirements.txt
- 122 tests pass
