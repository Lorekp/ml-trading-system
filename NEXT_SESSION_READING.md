# Next Session — Required Reading (in order)

## 1. System State
- `C:\Claude\ml-trading-system\SESSION_STATE.md` — current state, what's done, what's next

## 2. System Report (the full picture)
- `C:\Claude\ml-trading-system\backups\20260317_011916\SYSTEM_REPORT.md` — 3,000 word technical report covering architecture, validation, results

## 3. Session Report (what happened this session)
- `C:\Claude\ml-trading-system\backups\20260317_011916\SESSION_REPORT.md` — everything accomplished, metrics, files created

## 4. Key Results
- `C:\Claude\ml-trading-system\v4-results\tier1_results.json` — Tier 1 (features + sizing): Sharpe 0.41 -> 1.45
- `C:\Claude\ml-trading-system\v4-results\tier2_results.json` — Tier 2 (+HMM regime): Sharpe 1.45 -> 1.72
- `C:\Claude\ml-trading-system\v4-results\cpcv_portfolio_results.json` — CPCV: 93% positive, mean 1.68

## 5. Plans (what to do next)
- `C:\Claude\plans\ml-improvements\04-plan.md` — Tier 1/2/3 improvement plan (Tier 1+2 done, Tier 3 pending)
- `C:\Claude\plans\direction-v3\04-plan.md` — v3 plan including production bot architecture
- `C:\Claude\plans\direction-v3\05-production-architecture.md` — MEXC bot architecture spec

## 6. Code (key experiment scripts)
- `C:\Claude\ml-trading-system\experiments\direction_v4_tier1_fast.py` — fast Tier 1 (the optimized version)
- `C:\Claude\ml-trading-system\experiments\direction_v4_tier2.py` — Tier 2 with precompute-once
- `C:\Claude\ml-trading-system\experiments\direction_v4_cpcv_portfolio.py` — portfolio-level CPCV

## 7. Credentials
- `C:\Claude\ml-trading-system\.env` — MEXC API key + secret

## 8. Gap Analysis (what's left to improve)
- Tier 3 items from `C:\Claude\plans\ml-improvements\04-plan.md`:
  - T3.1: L2 data pipeline (MEXC WebSocket collector)
  - T3.2: Kronos embeddings (blocked on friend's 3090)
  - T3.3: Liquidation cascade detector
  - T3.6: Production MEXC bot (THE priority)
- Data to buy: CoinGlass Hobbyist $29/mo (liquidation, taker ratio, cross-exchange OI)

## 10. Library Reports
- `C:\Claude\library\report-afml.md` — AFML extraction
- `C:\Claude\library\report-ml-asset-managers.md` — ML for Asset Managers
- `C:\Claude\library\report-ml-for-finance-full.md` — ML for Finance (NEW, comprehensive)
- `C:\Claude\library\report-quantitative-trading.md` — Chan
- `C:\Claude\library\report-harris-trading-exchanges.md` — Market microstructure
go through the entire C:\Claude\library and C:\Claude\library\scientific articles and build a solid knowledge foundation. fill any gaps by doing your own research.
also read through C:\Claude\ml-trading-system\backups\20260317_011916, latest build.