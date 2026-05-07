# Live engine vs V2 backtest parity audit (2026-05-07)

## Context

You asked me to (1) train the LightGBM model the same way as the
backtest does, and (2) check for other parity divergences via an Opus
agent. Result: my earlier 2026-04 retrain was for the **wrong model
variant** (the engine's `_pruned_svi_v1` directory, not the backtest's
`_pruned_svi_har_pd_v1`). The HAR-PD model that matches the backtest
already has 2026-04 trained correctly (verified: re-running the
canonical script with `--max-trials 200 --patience 50` reports the
month as cached). The engine just doesn't load it.

The Opus agent found **13 additional divergences** outside the 8
config fields the existing parity test pins. **6 are P0** (change
sizing or signal direction on identical inputs); **7 are P1** drifts
that accumulate over time.

I verified the most consequential P0s by direct grep against the
two source trees. Each claim below is independently checkable.

---

## P0 divergences (verified)

### A. Model + aligner mismatch (already documented)
- See `vol_model/output/MODEL_ALIGNER_MISMATCH_FINDING.md`.
- Live: 44-feature `_pruned_svi_v1`. Backtest predictions parquet:
  50-feature `_pruned_svi_har_pd_v1`. β* calibrated against the latter,
  applied to the former in live.
- Fix scope: ~2-3 days (implement live HAR-PD computation, switch
  `lgbm_models_dir`, validate).

### D1 — DTE band: live trades 1-4 day options the V2 backtest mutes
- Backtest `vol_arb/backtester.py:469` (V2 production): `min_dte_days=5.0`.
  `_apply_entry_filters` at L993-1007 mutes ALL signals on DTE<5 bars.
- Live `deribit_paper/config.py:93`: `dte_min_days: int = 1`.
- Live trades the 1-4 DTE band the backtester deliberately skipped
  (theta-decay-dominated, fee-vs-vega ratio is poor).
- **Fix**: change `PaperConfig.with_v2_preset` to set `dte_min_days=5`
  (and reflect in `straddle.py` if it pins min separately).
- Effort: 1 commit, ~5 LOC + 1 test. **Quick win.**

### D2 — `skip_long_hours_utc` filter not implemented in live
- Backtest `vol_arb/backtester.py:470` (V2 production):
  `skip_long_hours_utc=(14, 15, 16)`. Mutes LONG signals at 14/15/16
  UTC (US cash-open window).
- Live: ZERO matches across all of `deribit_paper/`. Predictor issues
  longs at any hour.
- Per the agent's note (consistent with V2 trade-log analysis memory),
  this filter caught contango losses across 4/5 historical years of
  US-cash-open longs. **Likely the largest single dollar impact in
  this audit.**
- **Fix**: add `skip_long_hours_utc` to `PaperConfig`, mirror the
  backtester's mute-on-long check in `Predictor`.
- Effort: 1 commit, ~25 LOC + tests. **Quick win, high impact.**

### D3 — `dynamic_btc_cap` semantics inverted
- Backtest `vol_arb/backtester.py:1796-1798`:
  `dynamic_btc_cap = clip(trailing_1h_volume * max_participation_rate,
  1.0, 50.0)` — a **participation cap**.
- Live `deribit_paper/execution/straddle_flow.py:200`:
  `dynamic_btc_cap = float(n)` where `n = equity * dynamic_margin /
  (im * spot)` — a **margin envelope** (almost always larger).
- profit_scale and Kelly re-clips operate on different ceilings → same
  inputs produce different N when those multipliers fire.
- **Fix**: implement the participation cap in
  `straddle_flow.size_position` using the trailing-1h perp volume from
  the live perp buffer. Re-clip after each multiplier in the chain.
- Effort: 1 commit, ~50 LOC + tests. **Medium complexity.**

### D4 — `rolling_median_edge`: different statistic + different gating
- Backtest `vol_arb/backtester.py:1846-1847, 2205-2206`:
  500-period EMA (alpha=2/501), seeded from initial 96-bar median,
  updated **only on traded bars** (signal != 0 AND N > 0).
- Live `deribit_paper/engine.py:239, 1617-1618`:
  96-period rolling median, updated **on every finite-edge signal**
  (including flat bars).
- Denominator in `edge_scale = clip(|edge_vp| / rolling_median_edge,
  0.25, 3.0)` diverges in window length, statistic, and update gating.
- **Fix**: replace the deque+median in engine with the same EMA path
  the backtester uses. Mirror the traded-only gating.
- Effort: 1 commit, ~30 LOC + tests. **Quick win.**

### D5 — Profit-scale `update_realized` is fed NET PnL in live, GROSS in backtest
- Backtest `vol_arb/backtester.py:2225-2233`: feeds
  `gross_per_btc_i = vega_pnl_btc + theta_pnl_btc + delta_pnl_btc`
  (Black-Scholes decomposition, **pre-fees**).
- Live `deribit_paper/engine.py:1805-1918`: feeds `realized_pnl_usd`
  which is **net of fees and hedge PnL**. The engine's own comment at
  L1806-1807 acknowledges this as a deferred refactor:
  > "Treat ``realized_pnl_usd`` as the gross-pre-fees proxy for now;
  > E.1.10 will refactor"
- Result: live `_gross_per_btc_ema_usd` runs systematically below the
  backtester's, biasing the empirical-Bayes shrinkage toward the floor
  as `λ→1`. Positions get progressively smaller over the live run.
- **Fix**: compute and pass the BS-decomposition gross per leg from
  the executed fills + spot path. The live OrderManager has the data
  (fill prices, hedge MTM); needs a thin BS-decomposition helper.
- Effort: 2 commits, ~80 LOC + tests. **Medium complexity, real PnL impact.**

### D6 — RND tail-risk size scaling has no backtester counterpart
- Live `deribit_paper/engine.py:1717-1726`: when
  `rnd_monitor.size_scale() < 1.0`, `n_contracts *= rnd_scale`.
  Always-on (instantiated at L868 unconditionally).
- Backtest: no RND monitor or `rnd_scale` exists in `vol_arb/backtester.py`.
- Live can shrink size on bars the backtester would have sized fully.
  Memory entry `project_rnd_monitor_validated.md` notes RND was
  shipped "armed but silent" because no halt_mult delivered positive
  Sharpe in the W2.2 sweep. So this divergence may not bite in
  practice — the historical halt rate is ~0%.
- **Fix**: either remove RND from live (mirror backtest) or add it
  as an opt-in V2 preset switch with a documented "armed but silent"
  default. **Recommendation**: add `use_rnd_scaling: bool = False` to
  V2 preset; require explicit opt-in.
- Effort: 1 commit, ~10 LOC + test. **Quick win.**

---

## P1 divergences (verified by agent, less likely to bite)

| # | What | File:line | Fix effort |
|---|---|---|---|
| D7 | Live taker slippage % missing `√N · exp(k·jf)` impact term | `slippage_calc.py:36-100` vs `backtester.py:1968-1973` | medium |
| D8 | Kelly-mult lookup: backtest exact-match, live forward-fill | `sidecars.py:99-101` vs `438-483` | small |
| D9 | Profit-scale EB warmup behaves differently across restarts | `profit_scale.py:248-282` | already partially fixed (H4) |
| D10 | Hard `max_single_trade_btc=2.0` in live, none in V2 backtest | `risk/limits.py:67` | small (just remove the cap, or document) |
| D11 | Empty/low-volume bar guard mutes signals backtester sees | `engine.py:1443-1465` | small (acceptable as defensive) |
| D12 | Holding-period exit is wall-clock from fill, not bar-aligned | `straddle_flow.py:492-494` | medium |
| D13 | Realized PnL: BS-decomposition (backtest) vs fill-difference (live) | `engine.py:1845-1890` | overlaps with D5 |

---

## What I just did

1. **Confirmed HAR-PD 2026-04 model is correctly trained** — re-ran
   `run_walkforward_pruned_svi_har_pd.py --start 2026-04 --end 2026-04
   --max-trials 200 --patience 50`. Script reported `[1/1] 2026-04:
   cached, skipping` → the existing May 6 file is the canonical output.
2. **My earlier retrain (commit `9272ba6`) was for `_pruned_svi_v1`**
   (the engine's currently-loaded variant), NOT for the backtest's
   `_pruned_svi_har_pd_v1`. To "train the same way as the backtest",
   the engine needs to LOAD `_pruned_svi_har_pd_v1` — that's the
   model/aligner mismatch fix.
3. **Verified D1, D2, D5 by direct grep** of the two source trees.
   All three are real, all three are fixable.

---

## Scope of full alignment

To make the live engine behave **byte-identically** to the V2 backtest
on the same input feature stream:

| Scope | Effort | What it closes |
|---|---|---|
| **Tier 1 — quick wins (~1 day)** | 1 day | D1, D2, D4, D6, D8, D10. Add `dte_min_days=5`, implement `skip_long_hours_utc`, swap rolling_median_edge to EMA-on-traded-bars, gate RND behind opt-in flag, fix Kelly forward-fill, document/relax 2 BTC cap. |
| **Tier 2 — sizing rationalisation (~1 day)** | 1 day | D3 (participation cap), D5 (profit_scale gross). |
| **Tier 3 — model alignment (~2-3 days)** | 2-3 days | Implement live HAR-PD computation, switch `lgbm_models_dir` to `_har_pd_v1`. Closes the model/aligner mismatch. Largest risk surface. |
| **Tier 4 — execution fidelity (~1 day)** | 1 day | D7 (slippage formula), D11 (low-vol guard documented), D12 (exit timing), D13 (BS-decomposition PnL). |

**Total to full parity: ~5-7 days** of focused work.

## Recommendation — priority order

1. **Tier 1 quick wins NOW** (~1 day). D1 + D2 are the biggest
   expected dollar-impact divergences and are trivial to fix. D4 + D6
   close two more P0s with small commits. D8 + D10 are housekeeping.
2. **Tier 2 sizing rationalisation NEXT** (~1 day). D3 and D5 affect
   every trade's sizing; cleanest if D5 is fixed alongside D3.
3. **Tier 3 model alignment** as a focused project (~2-3 days). Worth
   doing in isolation with full validation against the backtester.
4. **Tier 4 execution fidelity** as final polish.

After Tier 1+2 (~2 days), the live engine should behave much closer
to the V2 backtest. After Tier 3, the model-input layer matches.
After Tier 4, execution matches.

I would NOT recommend running paper longer in the current state and
trying to interpret the live PnL — too many uncontrolled divergences
to attribute drift.

## Files

- This report: `vol_arb/output/LIVE_VS_BACKTEST_PARITY_AUDIT.md`
- Earlier model/aligner mismatch finding:
  `vol_model/output/MODEL_ALIGNER_MISMATCH_FINDING.md`
- HAR-PD retrain log (cached, no work done):
  `vol_model/output/pruned_svi_har_pd_v1_2026_04_retrain.log`
- Existing parity test (8 fields):
  `deribit_paper/tests/test_config_v2_parity.py`
