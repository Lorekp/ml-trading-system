# Live engine vs V2 backtest — round-3 parity audit (2026-05-08)

## Context

After the round-2 audit (`PARITY_AUDIT_ROUND2.md`) closed all 24 known
divergences via round-3, round-3.5 (F3 6-tier ship), and round-4 (F6/F8
fan-out + recertification baseline), this round-3 audit is a **fresh
sweep** to catch anything the prior rounds missed and to document
items that were only partly addressed.

**Audit cycle nomenclature reminder**:
- Round-1 *audit*: `LIVE_VS_BACKTEST_PARITY_AUDIT.md` (13 divergences, all closed)
- Round-2 *audit*: `PARITY_AUDIT_ROUND2.md` (24 divergences, ALL closed via round-3 + round-4 *cycles*)
- Round-3 *audit*: this document (in progress; 6 Opus agents auditing 6 surfaces in parallel)

---

## Carryover from round-2 (partly-addressed or deliberately-deferred)

Six items fit the "partly addressed" or "explicitly deferred" criterion
from the prior round. None of these are NEW divergences — they are known
gaps that the closure tickets explicitly documented as ship-with-deviation
or follow-up-required. Listing here so the round-3 cycle can pick them up
or formally accept them.

| # | Title | Class | Source | LOC est |
|---|---|---|---|---|
| C1 | F3 Tier 5 TSRV-grid features compute on 15-min grid (live) vs dollar-bar grid (offline) | **PARTIAL** | F3 commit f86858c, documented in `PARITY_AUDIT_ROUND2.md` § "Documented deviations from offline byte-identity" | ~200 |
| C2 | "Full recertification: 3-6 months walk-forward" — orchestrator + baseline shipped (`5162c9e`); the actual overnight retrain run is **not yet executed** | **PARTIAL** | round-4 plan: user explicitly chose "Full recertification: 3-6 months walk-forward" via AskUserQuestion, but only the orchestration scaffolding shipped — the 8-12h compute job hasn't been triggered | n/a (op time) |
| C3 | HAR-PD silent append failure pre-`fd3e04b` — historical signal-drift impact **unmeasured** | **UNMEASURED** | round-4 fix `fd3e04b` (added `BarTSRV.rv_ann` property); discovered during 22:15 UTC bar verification; "buffer never advanced from prewarm seed in production" | ~30 (audit script) |
| C4 | Live-vs-offline feature **parity test** on most-recent month — deferred | **DEFERRED** | round-4 closure § "Round-5 candidates" — flagged as ~half-day project, never started | ~150 |
| C5 | VPIN venue split within the LIVE feed — dollar bars on Binance Spot (F3) but VPIN still on Binance Futures perp | **ACCEPTED** | round-4 closure § "Round-5 candidates" — explicit design choice; would need SAC v4 retrain to migrate | (none) |
| C6 | Perp hedging venue split — live hedges Deribit perp, offline simulates Binance perp | **ACCEPTED** | round-4 closure § "Round-5 candidates" — never going to migrate (we trade options on Deribit, hedge belongs there) | (none) |

### Detail per carryover

#### C1 — Tier 5 TSRV-grid features: 15-min vs dollar-bar grid

Six features computed on a 15-min wall-clock TSRV grid in live, but on
the dollar-bar grid offline:

- `tsrv_mean_1d_lag3`
- `tsrv_mean_1w`
- `tsrv_ffd`
- `up_jump_frac_ema50`
- `bpv_tsrv_ratio_ema50`
- (one more — see `_LIVE_HANDLERS` Tier 5 block in `feature_assembler.py`)

**Why partial**: Full byte-identity would need a ~1750-bar dollar buffer
+ per-handler ffill of TSRV scalars onto dollar-bar timestamps.
**Mathematical impact**: Equivalent under constant-rate trading; deviates
on bursty-volume days (jump bars, news events).
**Mitigation**: Shipped because it is *substantially* better than the
baseline-frozen pre-F3 state (reads real distribution, just on a
slightly different grid).
**Closure path**: ~200 LOC + a 1750-bar `DollarBarBuffer` extension; only
worth doing if the recertification reveals a real signal gap.

#### C2 — Overnight walk-forward retrain not executed

The user explicitly authorised "Full recertification: 3-6 months
walk-forward" in round-4. What shipped:
- `scripts/round4_recertify_walkforward.py` (orchestrator wrapping the
  production walk-forward + per-month QLIKE-drift gates)
- Baseline run on existing checkpoints — verdict CONDITIONAL_PASS
  (4 PASS / 2 CONDITIONAL on Dec 2025 + Jan 2026 / 0 FAIL)
- Documented operator path for the 8-12h compute job

**What's missing**: the operator hasn't triggered the actual overnight
retrain. Re-running the existing months would either skip-then-do-nothing
(orchestrator's `_existing_checkpoints` check) or overwrite — neither
gives new information without code changes. A genuine recertification
needs either (a) a `--force` flag added to the orchestrator + an
overnight rerun, or (b) acceptance that the baseline run + the analytical
"F3/F6 don't affect training distribution" argument is sufficient.

**Recommendation**: round-3 audit should explicitly accept option (b)
and close C2 — OR the operator triggers the overnight job.

#### C3 — Pre-`fd3e04b` HAR-PD silent failure: historical impact unmeasured

The round-4 deploy verification surfaced that every live bar pre-fix had
been logging `'BarTSRV' object has no attribute 'rv_ann'` with the
warning swallowed by a try/except. The HAR-PD circular buffer never
advanced from its prewarm seed.

**What we know**:
- The fix (`BarTSRV.rv_ann` derived property) is correct going forward.
- The pre-fix buffer held the 96-bar prewarm rows from offline parquet
  permanently (never overwritten).
- HAR-PD features (`har_pd_R1_a005`, etc.) were therefore computed on
  *stale* rs_up/rs_down/rv_ann inputs throughout the live engine's
  uptime since F3 Tier 5 shipped.

**What we don't know**:
- How much signal divergence this caused on the live signal output vs
  what the model expected.
- Whether the live engine's signal stream over the affected window
  (~F3 Tier 5 ship date through 2026-05-08) materially differs from
  what a fully-aligned engine would have produced.

**Closure path**: a small audit script that replays a recent bar through
both pre-fix (stale prewarm) and post-fix (live-advancing) HAR-PD paths
and quantifies the per-bar `har_pd_R*_*` feature delta. ~30 LOC, ~1h work.

#### C4 — Live-vs-offline feature parity test still deferred

Per round-4 closure: per-bar diff between live engine output and offline
backtester output on the most recent month. Tests every feature in the
manifest under realistic conditions.

**Status**: deferred to "next quiet window". Still relevant — and arguably
the strongest possible certification of F3/F6/F8.
**Closure path**: ~150 LOC harness (replay engine.process_bar against
offline feature parquet rows for a recent month, assert per-feature diff).

#### C5 — VPIN venue split (Binance Futures) vs dollar bars (Binance Spot)

`vpin_live.py:1-23` documents this as deliberate: VPIN was trained on
Binance Futures perp aggTrades. Both sides (live + offline) read
Futures perp, so there is **no parity gap** — VPIN and dollar bars just
happen to live on different Binance feeds.

**Class**: accepted by design. Documenting only so the round-3 audit
ledger has it explicit.

#### C6 — Perp hedging venue (Deribit live, Binance offline)

The live engine hedges on Deribit perp because we trade options on
Deribit and hedge belongs on the same venue. The offline backtester
simulates fills against Binance perp data.

This **is** a venue divergence on the hedge side. But it's
philosophical, not actionable: migrating live hedging to Binance perp
would split trading across exchanges with no PnL upside.

**Class**: accepted by design. Documenting only.

---

## Round-3 fresh audit findings

Six Opus agents audited six surfaces in parallel. All returned. **53
fresh findings, 11 P0s** across the six surfaces — meaningfully more
than round-2 (24) at comparable depth. Each agent was briefed on the
24+ already-closed findings so they don't re-discover them; everything
below is genuinely new.

| Surface | Findings | P0 | P1 | P2/P3 |
|---|---:|---:|---:|---:|
| Feature pipeline (post-F3/F6/F8 refresh) | 9 | 3 | 1 | 5 |
| Execution & fills (post-E1/E2/E4 refresh) | 8 | 1 | 3 | 4 |
| DRL hedger I/O (post-D1/D5 refresh) | 7 | 1 | 3 | 3 |
| Signal generation (NEW surface) | 10 | 2 | 4 | 4 |
| Risk & margin accounting (NEW surface) | 10 | 2 | 3 | 5 |
| Order management & cancel-replace (NEW surface) | 9 | 2 | 4 | 3 |
| **Total** | **53** | **11** | **18** | **24** |

---

## Surface 1 — Feature pipeline (9 findings)

| # | Title | Pri | LOC |
|---|---|---|---|
| F-1 | TSRV history `append_bar` runs BEFORE `predictor.predict` → 1-bar look-ahead leak on 6 TSRV-grid features | **P0** | ~5 |
| F-2 | `parkinson_1d_ema50` EMAs raw bars (live) vs rolling-mean column (offline) | **P0** | ~10 |
| F-3 | `parkinson_1d` rolling window hardcoded 250 bars (live) vs dynamic `bpd` per asset (offline) | P1 | ~3 |
| F-4 | `LiveSVICalibrator.atm_vol_at_dte` uses unshifted `bucket_ts.max()` → ~1h IV leak vs offline | **P0** | ~10 |
| F-5 | `n_bars_in_window` permanently baseline-frozen (handler returns last historical row's value) | P2 | ~15 |
| F-6 | TSRV append vs HAR-PD append ordering inconsistency (HAR-PD correct, TSRV wrong — covered by F-1) | P3 | (with F-1) |
| F-7 | `_h_duration_s` / `_h_n_trades` warmup fallback uses wall-clock window stats, not last sub-bar | P2 | ~5 |
| F-8 | Handler-exception silent baseline fallback (F9 partly mitigates, but exception path not covered) | P2 | ~10 |
| F-9 | `time_to_funding` clean (no finding — included for completeness) | — | 0 |

### F-1 (P0) — TSRV history leak: 1-bar look-ahead on 6 features

- **Live path**: `deribit_paper/engine.py:1736-1737` calls
  `tsrv_history_buffer.append_bar(bar.timestamp, bar_tsrv)` BEFORE
  `predictor.predict → assembler.build_row` at `engine.py:1914`. So at
  handler time `buf[-1] == current bar`. `tsrv_features.py:98-100` does
  `ln.rolling(96).mean().iloc[-1]` and includes the current bar.
- **Offline path**: `vol_model/features/engineer.py:251`
  `ln_vol_lagged = ln_vol.shift(1)`; rolling/EMA at line 267 then
  operate on the shifted series → strict point-in-time, current-bar
  TSRV is excluded.
- **Mechanism**: At each bar T, live `tsrv_mean_1d` includes vol_ann(T)
  in the average; offline uses vol_ann(T-1)..vol_ann(T-96). Same for
  `tsrv_mean_1d_lag3`, `tsrv_mean_1d_ema50`, `tsrv_mean_1w`,
  `tsrv_ffd`, `up_jump_frac_ema50`, `bpv_tsrv_ratio_ema50`. Direction:
  if vol is rising, live `tsrv_mean_*` reads systematically higher than
  offline. Model trained on shift(1) data is fed shift(0) data → systematic bias.
- **Origin**: I introduced this in F3 Tier 6 (`e8595d2`); the test
  suite missed it because the buffer-append path was tested in
  isolation, not the bar-loop ordering vs `predictor.predict`.
- **LOC to close**: ~5 — move `append_bar` call AFTER `predictor.predict`
  (or read `iloc[-2]` consistently in TSRVFeatures).

### F-2 (P0) — `parkinson_1d_ema50` source-series mismatch

- **Live path**: `deribit_paper/features/dollar_bar_features.py:213-228`
  `parkinson_1d_ema(50)` runs `pd.Series(parkinson_bar).ewm(span=50,
  adjust=True, min_periods=1).mean().iloc[-1]` on the **raw per-bar
  parkinson series**.
- **Offline path**: `vol_model/features/engineer.py:362-363` builds
  `feats["parkinson_1d"] = parkinson_bar.rolling(w_park,
  min_periods=...).mean()` (a rolling-mean column). Then
  `_window_and_aggregate:460` runs
  `series.ewm(span=50, min_periods=1).mean()` on that **rolling-mean
  column**.
- **Mechanism**: Live ≈ EMA of high-frequency noisy parkinson_bar
  values; offline ≈ EMA of a smoothed rolling mean. Materially higher
  live variance, faster reactivity. Distribution shift on a feature
  LightGBM was trained on.
- **LOC to close**: ~10 — cache an additional `parkinson_1d_rolling`
  series in `DollarBarFeatures` (using `bpd ≈ 250` proxy) and EMA
  over it.

### F-3 (P1) — `parkinson_1d` window hardcode

- **Live path**: `deribit_paper/signal/feature_assembler.py:435`
  `_make_dollar_bar_handler("parkinson_1d", "parkinson_1d", 250)`.
- **Offline path**: `vol_model/features/engineer.py:361-363`
  `w_park = min(max(bpd, 10), 50_000)` where
  `bpd = avg_bars_per_day(bars.index, trailing_days=30)` —
  recomputed from data per training run.
- **LOC to close**: ~3 — pass live `bpd` through engine config →
  assembler → handler.

### F-4 (P0) — SVI atm_vol_at_dte uses unshifted bucket_ts (~1h IV leak)

- **Live path**: `deribit_paper/features/svi_live.py:253`
  `latest_ts = self._params_df["timestamp"].max()`. The
  `params_df["timestamp"]` is the **bucket START**.
- **Offline path**: `vol_arb/rebuild_svi_features_with_spot.py:117`
  `spot_df["ts_shifted"] = spot_df["bucket_ts"] +
  Timedelta(args.bucket_freq)` — bucket_ts shifted forward by
  bucket_freq (1h) before reindex onto the 15m grid.
- **Mechanism**: Live `svi_atm_1d` is up to ~1 hour fresher than
  offline. Live IV reacts to options flow the model was trained NOT
  to see. Vol-shift bias on signal generation.
- **LOC to close**: ~10 — filter `params_df` by
  `params_df["timestamp"] + bucket_freq <= ctx.bar.timestamp` instead
  of `max()` blindly.

### F-5..F-9 (P2/P3) — see surface table summary above

---

## Surface 2 — Execution & fill simulation (8 findings)

| # | Title | Pri | LOC |
|---|---|---|---|
| E-5 | Live engine NEVER invokes `place_maker_then_taker` — backtester models maker fills via SOR | **P0** | ~30 |
| E-6 | IOC `max_slippage_pct` (1.0% live ceiling) ≠ backtester's `ioc_cap_multiplier × bar_slip` | P1 | ~5 |
| E-7 | `maker_adverse_selection_iv` charged offline on maker fills only — paired with E-5, never charged live | P1 | (with E-5) |
| E-8 | Live cancellation-exit (`asyncio.CancelledError`) uses naked `market` orders, bypassing IOC cap and tiered ladder | P1 | ~10 |
| E-9 | Bar-aligned exit (D12) anchored to `pd.Timestamp.now(tz="UTC")` not bar timestamp → drifts under broker latency | P2 | ~5 |
| E-10 | `IdempotencyGuard` ttl_seconds=60 informational-only finding | P3 | 0 |
| E-11 | Strangle routing absent in live (relevant only when `use_instrument_routing=True`) | P2 | ~150 |
| E-12 | Live perp hedger MTM uses real spot path; backtester uses synthetic Brownian (`hedge_sim_seed=42`) | P2 | (structural) |

### E-5 (P0) — Live never invokes maker-then-taker

- **Live path**: `straddle_flow.py:421-438` and `:606-625` — both entry
  and exit always call `place_ioc_taker(...)` (taker-only, optional
  `tiered=True` ladder).
- **Offline path**: `vol_arb/backtester.py:2215-2234` —
  `maker_filled_arr[i]` set when SOR detects a taker print through the
  would-be-maker, in which case `straddle_slip = 0`.
- **Mechanism**: `OrderManager.place_maker_then_taker`
  (`order_manager.py:819-972`) and `cfg.maker_wait_seconds`
  (`engine.py:861`) are wired but never called. Backtester's V2
  default `use_sor=True` flips ~30-60% of bars to maker (zero option
  slippage). Live pays full taker on every leg.
- **$ impact / trade**: ~$5-15/trade live shortfall × 7,400 V2 trades ≈
  **$40-100K of phantom backtest PnL** not reproducible live.
- **LOC to close**: ~30.

### E-6..E-12 — see surface table summary above

---

## Surface 3 — DRL hedger I/O (7 findings)

| # | Title | Pri | LOC |
|---|---|---|---|
| D-7 | `DRLHedger.simulate` (offline V2 backtest path) freezes BSM Greeks at entry σ + T | **P0** | ~10 |
| D-8 | Live `vol_ann_entry` uses current bar; env + simulate use lagged bar | P1 | 1+1 |
| D-9 | Live `iv_entry` source = `signal.iv_used_pct/100`; training env uses `slice.iv(K)` | P1 | ~5 |
| D-10 | VPIN sampled at `begin_trade`; no NaN guard on stale stream | P2 | 1 |
| D-11 | `hedge_inventory_btc` not persisted across engine restart → 2× hedge on next tick after restart | P1 | ~30 |
| D-12 | Action sign convention at `delta_required == 0` (match — no issue) | P3 | 0 |
| D-13 | `dt_years` derivation paths 3-way (env reads `cfg.dt_years`; live + simulate derive) | P3 | ~5 |

### D-7 (P0) — Frozen Greeks in offline backtest path

- **Live path**: `deribit_paper/signal/drl_hedger_live.py:315-322` —
  `_straddle_delta_from_iv(S=current_spot, K, T=tau_t, sigma=iv_t)`
  per tick.
- **Training-env path**: `vol_arb/drl_hedging/env.py:418` —
  `straddle_delta_from_iv(S=P_t, K, T=tau_t, sigma=iv_t)` per tick.
- **Offline backtest path (the divergent one)**:
  `vol_arb/drl_hedging/env.py:786-790, 819-820, 826-827` — precomputes
  `safe_T = T_years` and `safe_sigma = sigma` at entry, then per-tick
  uses `d1 = (log(S_new/K) + half_sigma2_T) / sigma_sqrt_T` for both
  delta AND gamma; only `tau_t` enters the gamma denominator. No
  `iv_t` recomputation, no SVI slice evaluated per tick.
- **Mechanism**: SAC v4 was trained where Greeks-in-obs use the live
  `(iv_t, tau_t)` pair (env). Live inference does the same. The V2
  production backtest's `DRLHedger.simulate` evaluates `delta_obs`,
  `gamma_obs` with **frozen entry IV/T**. Across a 15-min bar the
  variance term `half_sigma²·T` is ~9× too large vs `half_sigma²·tau_t`
  at the last tick. **Every certified V2 SAC v4 backtest number** —
  including the headline trade-Sharpe 51 / NC PnL $1.029M — was scored
  on a corrupted obs distribution.
- **LOC to close**: ~10 — replace lines 819, 826-827 with per-tick d1
  using `tau_t`.

### D-8..D-13 — see surface table summary above

---

## Surface 4 — Signal generation logic (10 findings)

| # | Title | Pri | LOC |
|---|---|---|---|
| S-1 | Live passes `transvol_embedding=None` → 16/44 LightGBM inputs from STALE baseline row | **P0** | ~50 |
| S-2 | Kelly K1 lookup: backtester exact-match (`sidecars.py:118`); live walk-forward asof (`sidecars.py:496`) | P1 | ~20 |
| S-3 | Live `iv_used` has no fallback chain to `iv` / `dvol_close` like backtester does — silent emit/skip asymmetry | P1 | ~10 |
| S-4 | Live `ProfitScaleEstimator.estimate` multiplies `vega_per_vol_pt_usd × 2.0`; backtester uses single-leg `df["vega"]` without 2× | **P0** | ~5 |
| S-5 | Live `max_open_positions` cap absent in backtester (non-binding under V2 today, lethal at h=5m) | P2 | 0 (document) |
| S-6 | Aligner β* most-recent-prior fallback semantics differ between rolling live + end-to-end offline | P2 | ~15 |
| S-7 | Default-preset short threshold = 25 vs V2 = 22; CLI guards but defence-in-depth missing | P2 | ~5 |
| S-8 | `base_spread_iv` not flipped 1.5→2.0 in `with_v2_preset` (canonical V2 trade log generated `with_realistic_costs()`) | P1 | ~3 |
| S-9 | Term-slope degenerate-slice handling differs | P2 | ~10 |
| S-10 | profit_scale warm-resume vs cold backtest | P3 | 0 (accepted) |

### S-1 (P0) — Live TransVol embeddings missing

- **Live path**: `deribit_paper/engine.py:1916` passes
  `transvol_embedding=None` to the predictor; `FeatureAssembler`
  fills 16/44 LightGBM inputs (`tv_0..tv_15`) from a **stale baseline
  row**.
- **Offline path**: predictions parquet was generated with per-bar
  fresh TransVol embeddings.
- **Mechanism**: Aligner β* (scalar bias) cannot correct feature-row
  drift across 16 inputs. Every live prediction systematically
  diverges from the offline-equivalent. The bias is far larger than
  what the round-2 F1 SVI-NaN bug caused (which was 1 feature; this is 16).
- **LOC to close**: ~50 — plumb live TransVol embeddings into predictor.

### S-4 (P0) — profit_scale 2× factor

- **Live path**: `profit_scale.py:194` —
  `est_gross = vega_per_vol_pt_usd * 2.0 * iv_diff_vol_points` (the
  2× to convert single-leg vega to straddle vega).
- **Offline path**: `vol_arb/backtester.py:2099-2107, 2131-2134` —
  uses `df["vega"]` (single-leg, from `greeks.py:80`) **without**
  multiplying by 2.
- **Mechanism**: Live `est_gross` is ~2× backtester `est_gross` for
  the same trade. profit_scale clips less aggressively in live →
  systematically larger live notional than V2 canon says.
- **LOC to close**: ~5 — drop the 2× in `profit_scale.estimate`,
  OR change `df["vega"]` to `2 * df["vega"]` in backtester (whichever
  matches the model's training-time semantics).

### S-2..S-10 — see surface table summary above

---

## Surface 5 — Risk & margin accounting (10 findings)

| # | Title | Pri | LOC |
|---|---|---|---|
| R-1 | Kill-switch "daily DD" actually trips on LIFETIME peak, not daily window | **P0** | ~15 |
| R-2 | `set_equity()` (reconciliation path) silently advances `peak_equity` upward → re-bases kill-switch DD reference | **P0** | ~10 |
| R-3 | Live margin-util gate uses option-leg notional sum; ignores perp delta-hedge leg | P1 | ~15 |
| R-4 | Min-lot exemption suppresses margin gate on first position only — silent corner-case in DD-recovery | P1 | ~5 |
| R-5 | `_consecutive_empty_bars` not persisted across restart — sibling of S5/S8, missed | P1 | ~12 |
| R-6 | H15 `_recent_trade_timestamps` deque not persisted; per-hour cap silently re-bases | P2 | ~10 |
| R-7 | V2 preset disables `max_single_trade_btc` but not `min_single_trade_btc` floor in the gate | P2 | ~5 |
| R-8 | Latency-breach trip path increments shared `_consecutive_errors` counter | P2 | ~10 |
| R-9 | Margin envelope gate uses signal's `straddle.index_price`, not bar spot | P2 | ~3 |
| R-10 | Live equity drift vs backtester accumulates open-PnL as zero (semantic agreement, mid-bar telemetry diverges) | P3 | 0 |

### R-1 (P0) — Kill-switch "daily DD" tripping on lifetime peak

- **Live path**: `deribit_paper/risk/kill_switch.py:134`
  `self._equity.dropped_below_threshold(self.max_daily_dd_pct)`
  calling `equity.py:186-188` `return self.drawdown_pct <=
  -abs(...)` — `drawdown_pct` is `(current/peak − 1)*100` over
  the **lifetime peak** (`equity.py:99-103`).
- **Offline path**: backtester has no kill switch.
- **Mechanism**: Config `kill_switch_daily_dd_pct=2.0` and the
  docstring at `kill_switch.py:6-8` and `RUNBOOK_PAPER_V2.md:208` all
  say "daily DD". Implementation actually uses peak-relative DD.
  `EquityTracker.daily_return_pct()` exists at `equity.py:171-184` but
  is never called. Operator's mental model says one thing, code does
  another. After equity is in a 5% drawdown, ANY 0.01% further dip
  trips. Mid-week 1.5%-loss days behave differently than the
  operator expects.
- **LOC to close**: ~15 — swap to
  `daily_return_pct(now=bar.timestamp) <= -threshold` in
  `kill_switch.py:134`, OR rename config + docstrings to `peak_dd_pct`.

### R-2 (P0) — `set_equity` silently advances peak

- **Live path**: `deribit_paper/position/equity.py:152-153` —
  `set_equity` does `if self._current > self._peak: self._peak =
  self._current`. Called from startup reconcile (`engine.py:880-887`).
- **Offline path**: `vol_arb/backtester.py:2278` —
  `peak_eq = max(peak_eq, eq)` only advances on realized trade PnL,
  never on a "set" operation.
- **Mechanism**: A live restart re-seeds equity from Deribit's
  reported equity. After a winning week the exchange-reported equity >
  prior peak, and `set_equity` advances peak silently. From there, the
  kill-switch DD threshold uses the new peak as its baseline. With S1
  restoring `peak_equity` from disk, this is doubly weird: the
  restored peak loses on `max(restored_peak, exchange_equity)` to the
  lower of the two if reconciliation runs first.
- **LOC to close**: ~10 — restrict `set_equity` peak advance to
  `reason="reconcile-deposit"` paths, never on plain restart reconcile.

### R-3..R-10 — see surface table summary above

---

## Surface 6 — Order management & cancel-replace (9 findings)

| # | Title | Pri | LOC |
|---|---|---|---|
| O-1 | 60s cancel-replace gap unmodeled (live: maker bar i + taker bar i+δ; backtester: same bar) | P1 | ~20 |
| O-2 | Tiered taker fills are blended live (`_merge_tier_results`) but uniform offline (`ioc_cap_multiplier=2.0` treats capped as 100% at cap) | P1 | ~15 |
| O-3 | `IdempotencyGuard` TTL=60s exactly equals `maker_wait_seconds=60.0`; in-memory only (no SQLite mirror) | P2 | ~15 |
| O-4 | Tick hardcoded at 0.0005 BTC (`order_manager.py:402,494`); Deribit option tick is 0.0001 for mark < 0.05 BTC | P1 | ~3 |
| O-5 | Naked single-leg position can survive engine crash (no recovery in `engine_state` for open orders) | **P0** | ~80 |
| O-6 | Bar-aligned exit (D12) wakes at boundary then runs 2× ticker fetch + 2× IOC; backtester assumes instant fill | P1 | ~10 |
| O-7 | Perp $10-lot rounding (`amounts.py:60-83`) accumulates one-sided drift across hedge ticks | P2 | ~5 |
| O-8 | Tier-3 silently skips residual <10% of size; `n_contracts_btc` recorded as REQUESTED size, not `filled_amount` | P2 | ~10 |
| O-9 | WS reconnect (`engine.py:403`) only sends Telegram message; no `get_open_orders_by_currency` re-fetch | **P0** | ~20 |

### O-5 (P0) — Naked single-leg position survives engine crash

- **Live path**: `straddle_flow.py:421-461` opens legs via
  `asyncio.gather`, retries unilateral close 3× with 2s backoff
  (`_close_if_filled` L659-721). On permanent failure (e.g., instrument
  paused), CRITICAL log fires but state is **NOT persisted** —
  `_build_engine_state_dict` (engine.py:2549) covers equity/EMAs only,
  not open orders.
- **Offline path**: backtester assumes both legs fill atomically.
- **Mechanism**: Restart has no recovery path for a half-open straddle.
  The engine boots clean and starts trading new positions with one leg
  of an old straddle still riding.
- **LOC to close**: ~80 — extend `engine_state` to cover open-order
  state; recovery routine on startup that closes any leftover legs
  before resuming normal flow.

### O-9 (P0) — WS reconnect doesn't re-fetch open orders

- **Live path**: `engine.py:403` — `_on_ws_reconnect` only sends
  Telegram message. REST poll path (`get_order_state`) is resilient
  per-order.
- **Mechanism**: A coincident WS hiccup + 3 REST timeouts can trip the
  kill-switch on a transient network blip. No
  `get_open_orders_by_currency` re-fetch on reconnect to reconcile
  what the engine *thinks* is open vs what the venue *says* is open.
- **LOC to close**: ~20 — call
  `rest.get_open_orders_by_currency(currency)` on reconnect; reconcile
  against in-memory state; cancel orphans.

### O-1..O-8 — see surface table summary above

---

## P0 ledger — 11 silent-wrong-PnL or operator-semantic findings

Sorted by criticality (highest first):

| # | Title | Surface | LOC | Estimated impact |
|---|---|---|---|---|
| D-7 | `DRLHedger.simulate` freezes BSM Greeks at entry σ + T | DRL | ~10 | **Every** certified V2 SAC v4 number is on a corrupted obs distribution |
| S-1 | Live `transvol_embedding=None` → 16/44 LightGBM inputs from stale baseline | Signal | ~50 | **Every** live prediction diverges from offline-equivalent |
| F-1 | TSRV history `append_bar` before `predictor.predict` → 1-bar leak on 6 features | Feature | ~5 | Model trained on shift(1) data fed shift(0) data |
| F-4 | SVI `atm_vol_at_dte` uses unshifted bucket_ts | Feature | ~10 | Live IV ~1h fresher than offline → IV-edge calc bias |
| F-2 | `parkinson_1d_ema50` EMAs raw bars vs rolling-mean | Feature | ~10 | Distributional shift on a SHAP-relevant feature |
| S-4 | `profit_scale` uses 2× single-leg vega vs `df["vega"]` | Signal | ~5 | Live notional ~2× systematically larger than V2 canon |
| R-1 | Kill-switch "daily DD" actually trips on lifetime peak | Risk | ~15 | Operator-semantic mismatch; spec says one thing, code does another |
| R-2 | `set_equity` reconciliation silently advances peak_equity | Risk | ~10 | Defeats kill-switch protection on the path most likely to need it |
| E-5 | Live never invokes `place_maker_then_taker` | Execution | ~30 | $40-100K phantom over 7,400 V2 trades |
| O-9 | WS reconnect doesn't re-fetch open orders | Order mgmt | ~20 | Coincident WS hiccup + REST timeouts → kill-switch trip on transient blip |
| O-5 | Naked single-leg position can survive engine crash | Order mgmt | ~80 | Catastrophic on instrument-paused event |

---

## Cumulative dollar impact estimate

| Finding | Mechanism | Estimated impact |
|---|---|---|
| F-1 + F-4 | TSRV + SVI temporal misalignment | Unknown; likely material on every signal |
| F-2 | parkinson_1d_ema50 source-series shift | Distributional drift on SHAP-active feature |
| E-5 + E-7 | Live never makers; backtester models 30-60% maker fills | $40-100K phantom over 7,400 V2 trades |
| D-7 | Frozen Greeks in offline simulate path | Every certified V2 number scored on corrupted obs |
| S-1 | Stale TransVol embeddings live | Every live prediction systematically diverges |
| S-4 | profit_scale 2× factor | Live notional 2× systematically larger than V2 says |
| R-1 + R-2 | Kill-switch semantic + reconciliation | Protection weaker than spec on highest-risk paths |
| O-5 + O-9 | Open-order recovery + WS reconcile | Catastrophic-tail event, unknown frequency |

---

## Recommended close order

Ordered by cost-to-fix vs impact ratio. P0 fixes ≤30 LOC come first:

1. **F-1** (~5 LOC) — move `tsrv_history_buffer.append` after `predictor.predict`
2. **F-4** (~10 LOC) — filter `params_df` by shifted bucket_ts in `atm_vol_at_dte`
3. **R-2** (~10 LOC) — restrict `set_equity` peak advance to deposit-reason path
4. **R-1** (~15 LOC) — wire `daily_return_pct` to the kill-switch (or rename config + docs to `peak_dd_pct`)
5. **S-4** (~5 LOC) — drop the 2× in `profit_scale.estimate` (or align backtester)
6. **F-2** (~10 LOC) — fix `parkinson_1d_ema50` source series
7. **D-7** (~10 LOC) — per-tick d1 recompute in `DRLHedger.simulate`
8. **O-9** (~20 LOC) — re-fetch open orders on WS reconnect
9. **E-5** (~30 LOC) — wire `place_maker_then_taker` into `straddle_flow.execute`
10. **S-1** (~50 LOC) — plumb live TransVol embeddings into predictor
11. **O-5** (~80 LOC) — extend `engine_state` to cover open-order recovery

P1 backlog (18 findings) and P2/P3 (24 findings) follow in surface
order; pick whichever fit the next cycle's scope.

---

## Round-3 audit ledger

| ID | Pri | Status |
|---|---|---|
| C1..C6 | varies | carryover from round-2 (see top of doc) |
| **F-1** | P0 | **DONE** — round-3 cycle commit (TSRV append moved after `predictor.predict`; strict-past contract enforced; 3 regression tests in `test_tsrv_history_strict_past.py`) |
| **F-2** | P0 | **DONE** — round-3 cycle commit (`parkinson_1d_ema` now operates on rolling-mean series; 6 byte-identity tests in `test_parkinson_1d_ema_source_series.py`) |
| **F-3** | P1 | **DONE** — round-3 cycle 2 commit (dynamic `bpd` via `DollarBarFeatures.bars_per_day`; closes F-2 sibling; 8 tests in `test_dollar_bar_features_dynamic_bpd.py`; existing handler tests updated) |
| **F-4** | P0 | **DONE** — round-3 cycle commit (`atm_vol_at_dte` filters `params_df` by shifted `bucket_ts`; 7 tests in `test_svi_atm_dte_bucket_shift.py`; predictor + assembler call sites updated) |
| F-5..F-9 | P2/P3 | OPEN |
| **E-5** | P0 | **DONE** — round-3 cycle 2 commit `803be95` (`place_maker_then_taker` wired into `straddle_flow.execute` entry path; exit stays on `place_ioc_taker`; 5 tests in `test_straddle_flow_maker_then_taker.py`; existing tiered-ioc tests updated) |
| E-6, E-7 | P1 | OPEN — backtester model-philosophy decisions; deferred |
| **E-8** | P1 | **DONE** — round-3 cycle 2 commit `803be95` (4 cancel/recovery paths swap naked `order_type="market"` for `place_ioc_taker(tiered=True)`; new `cfg.cancel_exit_max_slippage_pct=1.0`; tests in `test_straddle_flow_maker_then_taker.py`) |
| E-9..E-12 | P2/P3 | OPEN |
| **D-7** | P0 | **DONE** — round-3 cycle 2 commit `1d4ca5a` (per-tick Greeks in `DRLHedger.simulate`; V2 NC PnL $1,024,449.18 → $1,024,054.51 (-$394.67, -0.039%) with Kelly K1 sidecar regen; n_trades unchanged at 7,543; 4 unit tests + V2 byte-identity relocked; rebaseline report at `vol_arb/output/parity_r3_cycle2_d7_rebaseline.md`) |
| **D-8** | P1 | **DONE** — round-3 cycle 2 commit `803be95` (engine captures `vol_ann_lagged` from `tsrv_history_buffer.latest_vol_ann_lagged()` BEFORE F-1 append, plumbs through to `LiveDRLHedger.begin_trade`; matches env.py:172's `df["vol_ann"].shift(1)`; tests in `test_drl_hedger_iv_and_vol_ann_sources.py`) |
| **D-9** | P1 | **DONE** — round-3 cycle 2 commit `803be95` (`begin_trade` evaluates `slice.iv(K)` to recover `iv_entry` whenever a slice is provided; matches env.py:412-415 reset semantic) |
| **D-11** | P1 | **DONE** — round-3 cycle 2 commit `803be95` (new `drl_trade_state` SQLite table FK-linked to `open_positions`; persister/remover callbacks emit on every begin_trade/step/close_trade; `_resume_trade` rehydrates via `resume_trade_from_dict`; 13 tests in `test_drl_trade_state_persistence.py`) |
| D-10, D-12, D-13 | P2/P3 | OPEN |
| S-1 | P0 | OPEN — multi-day feature build (no h15m TransVol checkpoint, no live inference path); split into S-1a (training) + S-1b (live wiring); deferred to dedicated TransVol-live cycle |
| S-2 | P1 | OPEN |
| **S-3** | P1 | **DONE** — round-3 cycle 2 commit (predictor `_lookup_iv` falls through to TSRV proxy when SVI uncalibrated/empty; new `Signal.iv_source` field tags "svi" / "tsrv_proxy"; 7 tests in `test_predictor_iv_used_fallback.py`; warmup-emit behavior change documented) |
| **S-4** | P0 | **REJECT** — agent mis-read backtester semantics. `df["vega"]` is the COMPOSITE (call+put = straddle) vega, set in `_composite_greeks` at `vol_arb/pipeline.py:241-242`. `bsm_vega_per_vol_pt` (live) returns single-leg vega; live correctly multiplies by 2 to convert to straddle. Both sides end at straddle vega; no divergence. |
| S-6 | P1 | OPEN |
| **S-8** | P1 | **DONE** — round-3 cycle 2 commit (`PaperConfig.with_v2_preset` flips `base_spread_iv 1.5 → 2.0` to match `BacktestConfig.with_realistic_costs`; 3 tests in `test_v2_preset_base_spread_iv.py` including cross-engine sanity check) |
| S-5, S-7, S-9, S-10 | P2/P3 | OPEN |
| **R-1** | P0 | **DONE** — round-3 cycle commit (kill-switch wired to `daily_return_pct` instead of `drawdown_pct`; existing `test_kill_switch.py` updated to assert corrected semantic; new `test_kill_switch_daily_dd_and_set_equity_peak.py` covers contract) |
| **R-2** | P0 | **DONE** — round-3 cycle commit (`set_equity` peak advance only on `reason="deposit"`; default `reason="reconcile"` preserves peak; unknown reasons raise; 4 tests in `test_kill_switch_daily_dd_and_set_equity_peak.py`) |
| R-3, R-4, R-5 | P1 | OPEN — risk-engine cycle (worth a dedicated audit) |
| R-6..R-10 | P2/P3 | OPEN |
| **O-5** | P0 | **DONE** — round-3 cycle 2 commit `803be95` (the existing `_reconcile_positions_on_startup` already handled full-straddle recovery; the genuine gap was DRL hedger state — closed via D-11; orphan-close path also upgraded from naked market to tiered IOC via E-8) |
| **O-9** | P0 | **DONE** — round-3 cycle 2 commit `803be95` (`_on_ws_reconnect` now schedules `_reconcile_open_orders_on_reconnect`; age-gated cleanup of stale orphans; new `cfg.ws_reconcile_orphan_age_seconds=300`; 8 tests in `test_engine_ws_reconnect_reconcile.py`) |
| O-1, O-2, O-6 | P1 | OPEN — backtester semantics changes; deferred to execution-modeling cycle |
| **O-4** | P1 | **DONE** — round-3 cycle 2 commit `803be95` (Deribit option tick now mark-price-aware via `_option_tick_for_mark`; 0.0001 BTC for mark<0.05, else 0.0005; 8 tests in `test_order_manager_option_tick.py`) |
| O-3, O-7, O-8 | P2/P3 | OPEN |

### Round-3 cycle 1 closure summary (2026-05-08)

| Status | Count |
|---|---:|
| DONE — closed this cycle | 5 P0s (F-1, F-2, F-4, R-1, R-2) |
| REJECT — agent mis-read | 1 P0 (S-4) |
| OPEN — deferred to next cycle | 5 P0s (D-7, S-1, E-5, O-5, O-9) + 18 P1s + 24 P2/P3s |

**~65 LOC source + ~700 LOC tests** (24 new regression tests across 5
new files). 886/886 deribit_paper tests pass; 43/43 V2 byte-identity
preserved.

### Round-3 cycle 2 closure summary (2026-05-08)

| Status | Count | Items |
|---|---:|---|
| DONE — closed this cycle | 4 P0s | E-5, O-5, O-9, D-7 |
| DONE — closed this cycle | 7 P1s | E-8, O-4, D-8, D-9, D-11, S-3, S-8, F-3 |
| OPEN — deferred (multi-day project) | 1 P0 | S-1 (TransVol live; needs h15m checkpoint training + sequence buffer + encoder loader) |
| OPEN — deferred (modeling debate) | 9 P1s | E-6, E-7 (cap-multiplier vs %-cap); O-1, O-2, O-6 (cancel-replace gap, tier blending, bar-aligned exit timing); R-3, R-4, R-5 (risk-engine cycle); S-2, S-6 (sidecar lookup convention) |
| OPEN — deferred (P2/P3 hygiene) | 24 | F-5..F-9, E-9..E-12, D-10, D-12, D-13, S-5, S-7, S-9, S-10, R-6..R-10, O-3, O-7, O-8 |

**Cycle 2 totals**: ~250 LOC source + ~1,500 LOC tests across 8 new
test files + 5 modified. 4 sequenced commits:
- `803be95` — operational P0s + DRL hygiene P1s (E-5, E-8, O-4, O-5+D-11, O-9, D-8, D-9)
- `1d4ca5a` — DRL D-7 + V2 rebaseline ($1,024,449.18 → $1,024,054.51, -0.039%)
- *cycle 2C* — signal/feature parity P1s (S-3, S-8, F-3)
- *cycle 2D* — this audit ledger update

V2 byte-identity preserved at the new D-7-corrected baseline of
`$1,024,054.5149 / 7,543 trades`. Kelly K1 sidecar regenerated
(mean multiplier 1.0025, essentially unchanged). All deribit_paper
tests green.

**Cumulative round-3 progress (cycle 1 + cycle 2)**:
- 9 P0s closed (F-1, F-2, F-4, R-1, R-2, E-5, O-5, O-9, D-7), 1 P0
  rejected (S-4 mis-read), 1 P0 deferred (S-1 multi-day project).
- 8 P1s closed (E-8, O-4, D-8, D-9, D-11, S-3, S-8, F-3),
  9 P1s deferred to dedicated cycles (modeling-philosophy debates +
  risk-engine + sidecar conventions).
- All P2/P3 hygiene items remain OPEN.

**Files**:
- Cycle 2 plan: `~/.claude/plans/starry-rolling-rabin.md`
- D-7 rebaseline report: `vol_arb/output/parity_r3_cycle2_d7_rebaseline.md`
- Test files (cycle 2): `test_straddle_flow_maker_then_taker.py`,
  `test_engine_ws_reconnect_reconcile.py`, `test_order_manager_option_tick.py`,
  `test_drl_trade_state_persistence.py`, `test_drl_hedger_iv_and_vol_ann_sources.py`,
  `tests/test_drl_simulate_per_tick_greeks.py`, `test_predictor_iv_used_fallback.py`,
  `test_v2_preset_base_spread_iv.py`, `test_dollar_bar_features_dynamic_bpd.py`

---

## Cross-references

- Round-2 audit: `vol_arb/output/PARITY_AUDIT_ROUND2.md` (24/24
  findings closed via round-3 + round-4 cycles, audit ledger marked
  COMPLETE)
- Round-1 audit: `vol_arb/output/LIVE_VS_BACKTEST_PARITY_AUDIT.md`
  (13/13 closed)
- Per-agent detailed audit reports cached at:
  - `~/.claude/plans/starry-rolling-rabin-agent-a757e830223ee4ef8.md`
    (feature pipeline)
  - `~/.claude/plans/starry-rolling-rabin-agent-a5a90424bf507f2d4.md`
    (execution & fills)
  - `~/.claude/plans/starry-rolling-rabin-agent-a5d66953e9ce3880b.md`
    (DRL hedger I/O)
  - `~/.claude/plans/starry-rolling-rabin-agent-a9ea2fdcd6466ff3d.md`
    (signal generation)
  - `~/.claude/plans/starry-rolling-rabin-agent-ab13ab5a1a6baea66.md`
    (risk & margin)
  - `~/.claude/plans/starry-rolling-rabin-agent-a9f9d25daf417e7a2.md`
    (order management)

---

## Files

- This audit: `vol_arb/output/PARITY_AUDIT_ROUND3.md`
- Round-2 audit: `vol_arb/output/PARITY_AUDIT_ROUND2.md`
- Round-1 audit: `vol_arb/output/LIVE_VS_BACKTEST_PARITY_AUDIT.md`
- Recertification report: `vol_model/output/round4_recertification_report.md`
- Round-3 plan: `~/.claude/plans/starry-rolling-rabin.md`
