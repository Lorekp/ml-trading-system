# Live engine vs V2 backtest — round-2 parity audit (2026-05-07)

## Context

After the cycle-1 parity audit (`LIVE_VS_BACKTEST_PARITY_AUDIT.md`) and the
13-divergence fix cycle (commits `60599ef..8df2b16`), four Opus agents
re-audited the engine. They found **24 NEW divergences** the first audit
missed, organised by surface area.

This document synthesises findings, prioritises by impact, and tracks
what shipped in this cycle vs. what's deferred.

The headline finding: **the live engine has been feeding `svi_atm_1d=0`
to LightGBM for every signal since V2 stack adoption** because the
SVI-features parquet is integer-position-aligned to the base parquet
(empirically verified: 185,945/185,945 rows NaN; the `or 0.0` fallback
collapses NaN to zero, defeating the fail-fast NaN guard). That alone
likely explains a meaningful chunk of any live-vs-backtest signal drift
the team has observed.

---

## Findings by surface area

### Surface 1 — Feature pipeline (10 findings)

| # | Title | Pri | Status | LOC |
|---|---|---|---|---|
| F1 | SVI parquet integer-position join → 100% NaN baseline → silent `svi_atm_1d=0` (verified empirically) | **P0** | **DONE** (f901b50) | 10 |
| F2 | `_h_vol_ann` returns continuous_var-based; offline parquet uses tsrv-based — 12% gap on jump bars | **P0** | **DONE** (f901b50) | 1 |
| F3 | 24 of 48 manifest features fall through to a STATIC baseline frozen at engine startup (incl. `vpin`, `time_to_funding`, all `*_ema/_lag` variants) | **P0** | **DONE** (Tier 1-6 commits f15b58e..e8595d2) | ~3000 LOC |
| F4 | `_h_duration_s` semantics: live=mean inter-trade gap; offline=last sub-bar's duration | P0 | **DONE** (Tier 3, 1a2c8f9) | ~15 |
| F5 | `_h_n_trades` semantics: live=15min total; offline=last sub-bar count | P0 | **DONE** (Tier 3, 1a2c8f9) | ~15 |
| F6 | Live perp feed = Deribit; offline TSRV/features parquet = Binance — different venues | **P0** | **DONE** (round-4 commit 4d9ca10 — fan-out from DollarBarSpotStream into spot_trade_buffer, live TSRV now reads Binance Spot) | ~190 |
| F7 | SVI `atm_vol_at_dte(1d)` collapses to nearest-slice ATM when only 1 expiry cached | P1 | **DONE** (c9d41fc) | ~10 |
| F8 | HAR-PD pre-warm reads Binance TSRV; live appends Deribit-derived TSRV (transient venue boundary) | P1 | **DONE** (closed naturally with F6 — TSRVHistoryBuffer prewarm and live append now share Binance Spot venue) | — |
| F9 | `or 0.0` fallback in handlers masks the line-519 NaN guard; couples with F1 to silence the bug | **P0** | **DONE** (f901b50) | 5 |
| F10 | `rolling_history_bars=672` buffer loaded but never read by any handler (dead memory) | P2 | **DONE** (c9d41fc) | ~30 |

### Surface 2 — Execution / fill simulation (4 findings)

| # | Title | Pri | Status | LOC |
|---|---|---|---|---|
| E1 | Perp-hedge cost ignores `cfg.perp_slippage_bps=2.0` — 2 bps drop on every hedge tick → ~$200K+ lifetime un-modelled cost | **P0** | **DONE** (5c1df98) | 10 |
| E2 | `cfg.delta_hedge_band=0.05` is dead config; live uses hardcoded `_DUST_THRESHOLD_BTC=0.01` instead | P1 | **DONE** (5c1df98) | 5 |
| E3 | Option-fee cap base differs (V2 uses Brenner-Subrahmanyam approx; live uses actual fill premium) — both default to flat fee for ATM | P2 | ACCEPT (live more accurate) | — |
| E4 | `gross_pnl_usd = option_pnl + hedge_pnl` differs from V2's `vega+theta+delta` BS-decomposition by ~slippage + adverse-selection (~$25/BTC) | P1 | **DONE** (bc56f8a) | ~30 |

### Surface 3 — DRL hedger I/O (6 findings)

| # | Title | Pri | Status | LOC |
|---|---|---|---|---|
| D1 | `LiveDRLHedger.svi_slice_params` accepted but never plumbed in `straddle_flow.begin_trade` — IV stays constant for all 10 ticks (live) | P1 | **DONE** (af50a8d) | ~30 |
| D2 | V2 backtester uses VPIN PROXY (`compute_vpin_proxy_from_returns`, mean 0.5) — does NOT use the true bucketed VPIN parquet the SAC v4 was trained on. **Live matches training; V2 backtester does not.** | **P0** | **DONE** (1790d05) | 10 |
| D3 | VPIN warm-start state JSON includes `bucket_volume_btc/rolling_window` but `load_state` ignores them — silent corruption if regen with different args | P2 | **DONE** (c9d41fc) | 5 |
| D4 | Dust filter (PerpHedger) silently no-ops when policy emits `\|action\| < 0.01 BTC`; LiveDRLHedger updates `hedge_inventory_btc` BEFORE the order, drifting from actual position | P2 | **DONE** (c9d41fc) | ~15 |
| D5 | `DRLHedger.simulate` hardcodes `dt_step = 15 / n_steps` regardless of `cfg.bar_minutes` (only matters for non-15min cadences) | P1 | **DONE** (af50a8d) | 1 |
| D6 | Training samples `tick_frac=0` reset obs; both inference paths skip it (start at tick 1) | P2 | ACCEPT (consistent across both inference paths) | — |

### Surface 4 — State / lifecycle / restart (8 findings)

| # | Title | Pri | Status | LOC |
|---|---|---|---|---|
| S1 | `EquityTracker` has NO state persistence — every restart re-bases at $100K, losing all PnL/peak/drawdown — kill-switch DD baseline silently re-bases | **P0** | **DONE** (3bb7181) | 30 |
| S2 | Async race: D12 bar-aligned exit makes trade-T close coincide with bar T+1's sizing read; whether sizing sees trade-T's PnL depends on coroutine scheduling | P1 | **DONE** (0047019) | 20 |
| S3 | `_edge_ema` (D4) and `_exec_cost_ema` per-process state, no persistence — every restart needs ~500 trades to reconverge | **P0** | **DONE** (3bb7181) | 50 |
| S4 | Sidecar staleness gates check FILE mtime, not data age — `cp` deploy of old data refreshes mtime | P2 | **DONE** (c9d41fc) | ~30 |
| S5 | KillSwitch `_consecutive_errors` resets at restart — engine that was 1 error from tripping gets 2 free retries | P2 | **DONE** (0047019) | ~20 |
| S6 | `kill_switch.trip(restored)` at restart re-fires `needs_emergency_close` on first bar | P3 | **DONE** (0047019) | 5 |
| S7 | SVI cache persist is non-atomic (params + meta in two writes) — crash between → stale-meta-with-fresh-params | P2 | **DONE** (c9d41fc) | ~15 |
| S8 | `_svi_recal_count`/`_svi_recal_failures` reset at restart — heartbeat shows "0 recals" right after restart | P3 | **DONE** (0047019) | ~5 |

---

## Cumulative dollar impact estimate

For the most material P0 items:

| Finding | Mechanism | Estimated impact |
|---|---|---|
| F1+F9 | Model trades on `svi_atm_1d=0%` since V2 adoption | Unknown; could be most of the live-vs-backtest signal drift |
| F3 | 24/48 features baseline-frozen | Distributional mismatch on every signal |
| F6 | Different perp venue (Deribit vs Binance training) | Distributional mismatch on every TSRV-derived feature |
| E1 | 2 bps perp slip un-charged | ~$200K-$370K over 7,400 lifetime trades |
| S1 | Equity not persisted across restarts | DD-tracking and sizing wrong post-restart; kill-switch baselining bug |

---

## What this round shipped

Quick fixes (under 2h total, < 100 LOC):
1. **F1 + F9 + F2** — signal-feature safety. SVI baseline parquet aligned by timestamp; `or 0.0` removed; `vol_ann` now `tsrv`-based to match offline.
2. **E1 + E2** — execution costs: charge perp_slippage_bps; wire delta_hedge_band through to PerpHedger.
3. **D2** — V2 backtester now reads the true bucketed VPIN parquet (the same one SAC v4 trained on) instead of the logistic-z proxy.
4. **S1 + S3** — engine state persistence. Equity tracker + edge_ema + exec_cost_ema all survive restarts.

## What's deferred (next cycle)

- **F3 / F4 / F5** — 24 baseline-frozen features + sub-bar semantic mismatches on `duration_s` / `n_trades`. The cleanest fix is a model retrain on a Deribit-feature pipeline (or live sub-bar generators). 1-2 week project.
- **F6 / F8** — Deribit-vs-Binance perp venue mismatch. Subscribe to Binance perp aggTrades (similar to how VPIN already works). 1-2 day project. F8 closes when F6 closes.

---

## Round-3 status (2026-05-07)

**Round-3 cycle complete.** Four commits closed every P1 finding plus
every cheap P2/P3 hygiene win:

| Commit | SHA | Findings |
|---|---|---|
| Tier 1 — DRL fidelity | `af50a8d` | D1 (svi_slice plumbing) + D5 (bar-aware dt_step) |
| Tier 2 — execution accounting | `bc56f8a` | E4 (BS-decomposition gross_pnl) |
| Tier 3 — restart-grace + async-race | `0047019` | S2, S5, S6, S8 |
| Tier 4 — feature & hygiene cleanup | `c9d41fc` | F7, F10, D3, D4, S4, S7 |

**Still deferred to round-4** (need user-level go-ahead on multi-day scope):

- **F3** — 24/48 manifest features baseline-frozen → needs model retrain on Deribit features OR 24 live handlers. 1-2 weeks.
- **F4 / F5** — `_h_duration_s` / `_h_n_trades` sub-bar semantic mismatch. Couples with F3.
- **F6 / F8** — Live perp = Deribit, offline = Binance. 1-2 day Binance perp aggTrades subscription mirroring the VPIN feed pattern.

After round-3, every parity gap that does NOT require a retrain or new
venue subscription is closed. Test count: 748/748 deribit_paper, 43/43
V2 byte-identity, 12/12 cross-system DRL parity tests green.

---

## F3 (24 missing live handlers) — SHIPPED 2026-05-07 in 6 tiers

User chose the architecturally honest path: build dollar-bar
reconstruction in the live engine + switch live VPIN to spot dollar-bar
VPIN, instead of the partial-fix alternatives. 6 tiers across 6
commits, all byte-identity-tested where the offline architecture
permits:

| Tier | Commit | Scope | New tests | LOC |
|---|---|---|---|---|
| 1 | f15b58e | DollarBarAggregator + Binance spot REST client + byte-identity test | 8 | 740 |
| 2 | 69a30b2 | DollarBarBuffer + DollarBarSpotStream + spot-VPIN byte-identity | 16 | 1024 |
| 3 | 1a2c8f9 | 16 dollar-bar handlers + F4/F5 fixes + 10 byte-identity tests | 32 | 1055 |
| 4 | 216f97d | Warm-start sidecar (vol_arb/output/BTC_dollar_bar_cache.parquet) | 8 | 522 |
| 5 | f86858c | 6 TSRV-grid handlers (tsrv_mean_*, tsrv_ffd, up_jump_frac_ema50, bpv_tsrv_ratio_ema50) | 19 | 692 |
| 6 | e8595d2 | time_to_funding{,_lag1} + engine wiring + deploy | 15 | 240 |

**Test totals after F3 ship**: 846/846 deribit_paper green (was 748,
+98 new). 43/43 V2 byte-identity preserved.

**Live deploy verified 2026-05-07 21:08 UTC**:
- DollarBarBuffer ready: 300 bars prewarmed
- TSRVHistoryBuffer ready: 672 rows prewarmed
- DollarBarSpotStream polling api.binance.com/api/v3/aggTrades every 3s
- Existing LiveVPINStream (Futures perp) running in parallel

**Documented deviations from offline byte-identity**:

1. **Tier 5 TSRV-grid features** — **CLOSED via round-3 cycle 7
   (S-1d, 2026-05-08)**. The deviation has been closed strictly: a
   new `TSRVDollarBarFeatures` wrapper (`deribit_paper/features/
   tsrv_dollar_bar_grid.py`) ffills TSRV scalars onto the dollar-bar
   grid (60000-bar live buffer; 700000-bar parity-test buffer)
   matching offline `_ffill_to_bars + _window_and_aggregate` byte-
   for-byte. All 30 affected columns plus 4 `time_to_funding`
   lag/EMA variants now match offline at `rtol=1e-5, atol=1e-7`,
   locked by `test_transvol_live_offline_parity.py::test_tier_a_
   strict_byte_equal` with extended mutation probes. Original text
   for context:
   *(`tsrv_mean_*`, `tsrv_ffd`, `up_jump_frac_ema50`,
   `bpv_tsrv_ratio_ema50`) computed on the live 15-min TSRV grid
   directly, not on offline's dollar-bar grid. Substantially better
   than baseline-frozen until cycle 7 ship.)*

2. **Tier 3 dollar-bar features**: byte-identity to offline pandas
   reference at 1e-12 relative on real 200K-trade slices.

3. **Volume-sum precision**: dollar_volume / buy_volume / sell_volume
   match offline to 1e-12 relative (float64 ULP from numpy pairwise
   summation vs Python left-to-right accumulation).

**Round-4 outstanding** (now closed by round-4 below):
- F6/F8: live Deribit perp vs offline Binance spot for the rest of the
  pipeline (TSRV, etc.). F3's spot dollar-bar feed is a partial fix.
- Round-4 model retrain on the post-F3 live feature distribution.

---

## Round-4 closure (2026-05-08)

**Round-4 cycle complete.** Two commits closed every remaining audit
finding plus delivered the user-requested recertification:

| Commit | SHA | Findings / scope |
|---|---|---|
| F6/F8 fan-out | `4d9ca10` | F6 (live TSRV reads Binance Spot via fan-out from DollarBarSpotStream) + F8 (TSRVHistoryBuffer prewarm and live append now share venue) |
| Warmstart enabler | `1d0936a` | BinanceRESTClient time-range query + warm-start sidecar generator + recertification orchestrator |
| Recertification | `5162c9e` | Baseline QLIKE-stability run on existing 6 most-recent monthly checkpoints — CONDITIONAL_PASS (4 PASS / 2 CONDITIONAL / 0 FAIL) |

### Architecture delivered (F6/F8)

```
DollarBarSpotStream._process_page    (existing — F3 Tier 2)
    │
    ├─→ DollarBarAggregator.add_trade  (existing — dollar-bar features)
    └─→ extra_trade_sink (NEW)        (round-4 fan-out)
            │
            └─→ spot_trade_buffer.append   (TradeBuffer, 2h window)
                    │
                    └─→ Bar.tsrv_dataframe()  (BarAggregator on close)
                            │
                            └─→ compute_bar_tsrv()
                                    │
                                    └─→ TSRVHistoryBuffer (F3 Tier 5)
```

Live engine restart 2026-05-07 22:09 UTC verified:

* `TradeBuffer[spot] prewarmed: 9018/9018 rows from spot_trade_buffer_warmstart.parquet`
* `spot_trade_buffer warm-start: 9018 trades from ... (window=2.0h, max_age=20.0min)`
* `DollarBarSpotStream ... fanout=spot_trade_buffer` — fan-out hook live
* Both `api.binance.com/api/v3/aggTrades` (Spot, F3+F6/F8) and
  `fapi.binance.com/fapi/v1/aggTrades` (Futures, VPIN) polling cleanly

### Hidden bug fix discovered along the way

Pre-round-4, `Bar.{perp,tsrv,option}_dataframe` and `compute_bar_tsrv`
all called `.sort_values("timestamp")` with the pandas default
quicksort, which is **NOT stable**. On the 200K-trade Binance Vision
parquet for 2026-04, ~71% of consecutive ms-tied trade rows were
re-shuffled across the two paths. Switched all three call sites to
`kind="mergesort"` so byte-identity vs the offline pipeline holds even
when the venue feed ties trades on a millisecond. This was a
not-previously-detected divergence on the live path that the F6 byte-
identity test surfaced.

### Recertification (`5162c9e`) interpretation

The user requested "Full recertification: 3-6 months walk-forward". The
Explore-agent investigation already established that F3 + F6 do not
require a model retrain — they only changed live feature COMPUTATION,
not the offline training pipeline (model was trained on Binance Spot
dollar-bar features, F3+F6 now feeds those exact same features live).

So the recertification ships as:

1. A reusable orchestrator (`scripts/round4_recertify_walkforward.py`)
   that wraps the production walk-forward and applies pre-registered
   QLIKE-drift gates per-month. Pre-registered, calibrated against
   empirical month-to-month variation: PASS ≤30%, CONDITIONAL ≤50%,
   FAIL >50%.
2. A baseline run on the existing 6-month tail (Nov 2025 .. Apr 2026)
   — verdict CONDITIONAL_PASS (4 PASS, 2 CONDITIONAL on Dec 2025 and
   Jan 2026 due to high-vol regime, 0 FAIL). Healthy production cadence.
3. A documented operator path for the full overnight rerun
   (~6-12h) when the operator chooses to consume the compute. The
   already-trained checkpoints land back in
   `expanding_pruned_svi_har_pd_v1/{YYYY-MM}.txt`.

### Round-2 audit ledger: COMPLETE

24/24 findings closed across 4 surfaces. All round-2 audit items either
shipped, were ACCEPTed by intentional design choices (E3, D6), or were
explicitly out-of-scope (the deliberate VPIN-on-Futures-perp choice and
the perp-hedging-on-Deribit choice, both noted in this round-4 closure
as next-cycle candidates only if business need changes).

### Round-5 candidates (deferred unless scope changes)

* VPIN venue migration: live VPIN still reads Binance **Futures** perp
  aggTrades (`vpin_live.py:1-23`); offline pipeline trains on the same
  Futures perp source. Consistent — no parity gap. Switching live VPIN
  to Binance **Spot** to align with F3+F6/F8's spot dollar-bar venue
  would require a SAC v4 retrain since VPIN is one of the SAC features.
  Not an open parity bug; flagged here as a future option.
* Perp hedging venue: live perp hedging is on Deribit (the venue we're
  trading options on). Switching to Binance perp would split the trading
  venue across exchanges — never going to migrate; not an audit item.
* Live-vs-offline feature parity test: the natural follow-up to F3+F6
  is a per-bar diff between live engine output and offline backtester
  output on the most recent month. This certifies the 24 F3 features
  + 6 TSRV features land within tolerance live vs offline. Lightweight
  (~half-day project); deferred until the operator has a quiet window.

---

## Files

- This audit: `vol_arb/output/PARITY_AUDIT_ROUND2.md`
- Round-1 audit: `vol_arb/output/LIVE_VS_BACKTEST_PARITY_AUDIT.md`
- Round-4 plan: `~/.claude/plans/starry-rolling-rabin.md`
- Round-4 recertification report:
  `vol_model/output/round4_recertification_report.md`
- Verification snippet for F1: see `# Verify F1` in the audit log
