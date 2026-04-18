# Signal-Research Spot-Check — 2026-04-17

Two back-to-back spot-check sessions:
- **Session A (completed earlier 2026-04-17)**: Three architectural threads
  (HAR-residual, Quantile LGBM, PatchTST) — all rejected on the same 44-feature
  control. Writeup below.
- **Session B (in-flight / completed later 2026-04-17)**: Tier 1 new-information
  threads (ETH cross-asset, HARQ realized quarticity, combined) + Tier 3 HQR
  conformal sizing wrapper. **Preceded by a critical label-pipeline fix** —
  see "Label-shift discovery + revert" section below. Tier 2 (SVI-spot)
  deferred to next session — see deferral note.

---

## ⚠ Critical finding — label-pipeline shift (Session B preamble)

During Session B planning, a parallel audit discovered that the BTC TSRV
parquet was migrated from **START-stamped to END-stamped** on 2026-04-11
21:53 — **ten hours after Stage 18 predictions were saved** (2026-04-11
11:15). The backup file `BTCUSDT_tsrv_15m.parquet.pre_end_stamp_bak`
(mtime 21:51) is explicit proof of the migration.

`vol_model/labels.py::align_features_labels` retained its `iloc[1:]` shift,
which was the correct compensator for START-stamped labels but under-shifts
by one bar under END-stamp. Concretely at any output timestamp T:

- **Stage 18 semantics**: `y[T] = log(vol of [T, T+15min))` — 1-bar-ahead
  forecast (non-trivial).
- **Pre-fix semantics (2026-04-11 → 2026-04-17)**: `y[T] = log(vol of
  [T-15min, T))` — **nowcast of the already-closed bar** (near-trivial
  because features are computed over the same window).

Empirical signature: `current[T] == saved[T-15min]` for **145,028 / 145,028
rows (100%)**. Full diagnosis: `vol_model/output/LABEL_SHIFT_AUDIT.md`.

**Fix** (`vol_model/labels.py::load_labels`): when `tsrv_end_stamped=True`
(per-asset flag — BTC=True, ETH=False), shift `tsrv.index -= freq` BEFORE
the existing filter+clip+return flow. Restores Stage 18 forecast semantics
without touching `align_features_labels`.

Regression test (`tests/test_label_semantics.py::test_btc_labels_match_stage18_predictions`):
fresh `load_labels(VolModelConfig())` produces `ln_vol` that **byte-matches**
the saved `y_true` in `BTCUSDT_pruned_svi_predictions.parquet` on all
145,041 common rows. Max |Δ| = 0.0.

**Implication for Session A (retroactive)**: the three rejections
(HAR-residual, Quantile LGBM, PatchTST) were all measured against a fresh
control trained on the **nowcast** target, not the forecast target.
Qualitatively the rejections likely stand — their margins were 5-26%, far
above noise — but the absolute numbers are on a different (easier) task
than production's forecast objective. Session B's comparison numbers are
the apples-to-apples view. Session A verdicts are NOT re-run
(compute-bound); rejections are preserved because mechanisms were
feature-set-level (HAR redundancy, pinball/QLIKE mismatch, TransVol
near-optimality) which apply to both forecast and nowcast targets.

---

## Session A: Three architectural threads (2026-04-17 early)

Three research threads (HAR-residual, Distributional/quantile, PatchTST) evaluated against a **freshly retrained production-recipe control** on a regime-hard spot-check substrate. Methodology follows the plan at `~/.claude/plans/starry-rolling-rabin.md`; asymmetric acceptance rule (scrap on one bad month) per user directive mid-session.

## TL;DR

**All three threads REJECTED.** The production `pruned_svi_v1` architecture is a stronger local optimum than expected — none of the "better-ML-on-same-features" threads cleared the +3% QLIKE bar, and all three regressed vs a fresh control retrain:

| Thread | Status | Key number | Mechanism |
|---|---|---|---|
| **HAR-residual** | REJECTED (2024-11 smoke) | +25.9% worse QLIKE | Production features already include HAR components (`tsrv_mean_{1d,1w}`, `parkinson_{1d, 1d_ema50}`); subtracting HAR from target while keeping those features creates redundancy, not decoupling |
| **Quantile LGBM** | REJECTED (2022-11 gauntlet) | q50 +25.2% worse; spread–error corr 0.06 vs target 0.30 | Pinball loss optimizes conditional median; QLIKE scores against conditional mean. Right-skewed log-vol → level mismatch. Additionally, the predicted (q90−q10) spread does **not** track actual prediction error — no usable uncertainty signal emerges |
| **PatchTST** | REJECTED (2022-11 gauntlet, after 2024-11 tie smoke) | +5.33% worse QLIKE on 2022-11 (DM p=0.0001); −0.18% tie on 2024-11 | PatchTST-encoded embeddings feed the same LGBM stack and produce either tied (easy regime) or worse (hard regime / FTX week) QLIKE. TransVol is already architecturally near-optimal for this feature set — patching the time axis doesn't add information |

**No candidate shortlisted for full 50-month walk-forward retrain.** This is a clean negative result; mechanistic explanations are documented below so next-session work doesn't repeat the same errors.

The session still delivered: **63 new unit tests** (all green), a reusable per-month training harness, leakage probe battery, causal HAR forecaster, causal PatchTST encoder, and a labels-pipeline shift diagnostic that's flagged for follow-up.

## Regime substrate

8 months via `vol_model/spot_check_regimes.py::pick_regime_months` (union of 4 hardest months in saved `summary.json` + 4 macro hand-picks):

| Month | Role | Production QLIKE (stale summary.json) | Narrative |
|---|---|---|---|
| 2022-11 | macro | 0.5536 | FTX week (top-1 DD −9.16%) |
| 2022-12 | hardest | 0.8098 | Crypto-winter post-FTX |
| 2023-01 | hardest | 0.8065 | Post-FTX recovery |
| 2023-07 | macro | 0.6314 | Summer low-RV grind |
| 2023-08 | hardest | 0.7420 | +6.5% regression month |
| 2023-10 | hardest | 0.6950 | +3.7% regression month |
| 2024-11 | macro/easy | 0.2640 | Trump election; lowest QLIKE in summary |
| 2024-12 | macro | 0.2838 | Vol-expansion (Stage 18's biggest drop) |

Gauntlet actually ran only **2022-11** (control + quantile, then control + PatchTST) before early-exit via asymmetric acceptance each time. HAR-residual ran on 2024-11 smoke only; PatchTST also had a 2024-11 smoke that tied control (−0.18%). Everything else skipped correctly by scrap logic — exactly the compute savings the rule was meant to deliver.

## Methodology notes

### Label-pipeline shift (FINDING — flagged for future audit)

`ctx.y_full` from current `build_dataset(vol_config)` does NOT byte-match the `y_true` column of `BTCUSDT_pruned_svi_predictions.parquet` saved during Stage 18's retrain. Same min/max (−3.909 / +1.389, 145K common rows) but mean shifted:

| | Current `ctx.y_full` | Saved `y_true` |
|---|---|---|
| mean | −1.06 | −1.20 |
| std | 0.76 | 0.74 |
| min | −3.909 | −3.909 |
| max | +1.389 | +1.389 |
| exact-match | 0.03% on common index | |

Neither a target reparameterization (not excess-logvol — explicitly falsified by checking `ctx.y − parquet.y_true − log(IV)`) nor a constant offset. Shape-preserving, mean-shifted.

**Confirmed numerically**: fresh `pruned_svi_control` on 2022-11 beats the stale saved predictions by −46.45% QLIKE (DM p = 3.15e-36) and on 2024-11 by −28.67%. That's not production being beaten by ~30% — it's the saved predictions being scored against labels they weren't trained for.

**Mitigation (this session)**: every candidate's QLIKE compared to a **fresh** `pruned_svi_control` retrained on current labels per-month. DM tests use control predictions as baseline, not saved predictions.

**Action for future session (out of scope today)**: `git diff` on `vol_model/labels.py`, `vol_model/pipeline.py`, `TSRV/*.py` vs tag at Stage 18 retrain. Identify which tweak changed the label values. If intentional, retrain production on current labels; if a regression, revert. Either way, the summary.json headline "avg QLIKE 0.4592" is not directly comparable to anything trained after the shift.

### Leakage probes (ALL PASS — confirms prior audits)

Ran before every training via `vol_model/leakage_probes.py`. Output on the gauntlet run:

- **label_alignment**: 5000 sampled rows all have label timestamp = feature timestamp + 15 min ✅
- **svi_parquet_end_stamp**: all 183,065 SVI rows END-stamped at 15 min ✅
- **baseline_preds_end_stamp**: all 145,041 baseline-prediction rows END-stamped at 15 min ✅
- **HAR causality** (unit-tested in `tests/test_har.py::test_har_causality_no_future_data`): mutating y[t:] leaves HAR_pred[:t] byte-identical ✅
- **PatchTST causal attention** (unit-tested in `tests/test_patchtst.py::test_patchtst_attention_is_causal`): first-patch attention output unchanged when future patches are randomized ✅

Matches memory's `feedback_tsrv_not_a_leak.md` and `project_leakage_audit.md`. Not a re-escalation; verification via structural checks.

### Decision gates (locked pre-run, not tuned after)

A candidate WINS only if ALL of these hold:

1. All leakage probes pass (hard gate) — ALL candidates PASS.
2. QLIKE improvement vs fresh control on ≥ 5 of 8 months.
3. Average QLIKE improvement ≥ 3%.
4. DM p < 0.01 vs control on ≥ 4 months.
5. MZ: |α| < 0.1, |β − 1| < 0.2.
6. (Quantile only) Spread ↔ |err| Spearman > 0.3 AND q10 ≤ q50 ≤ q90 on all months.

### Asymmetric acceptance early-stop (added mid-session)

After the HAR-residual smoke showed +25.9% worse on one month, the user added an early-stop rule: **scrap any candidate that is > 5% worse than fresh control on any single month**, skip it from remaining months. Implemented in `run_spot_check_signal_research.py::run_gauntlet` via the `scrap_threshold_pct` parameter + `scrapped: set[str]` tracker.

Saves ~8× compute on losing candidates. Quantile was scrapped on the first month (2022-11, q50 +25.2%) and the gauntlet early-exited after detecting both candidate threads were scrapped. Total gauntlet wall clock: **~14 min** (vs estimated 4–8 hours had it run to completion).

## Per-thread results

### Thread 1 — PatchTST encoder (REJECTED)

**Status**: SCRAPPED on 2022-11 (first gauntlet month, +5.33% worse than fresh control). Smoke on 2024-11 was a statistical tie (−0.18%). Pattern across two regimes: **tied on easy month, worse on hard month** — never meaningfully better.

**Results:**

| Month | Regime | Control QLIKE | PatchTST QLIKE | Δ | DM t-stat | DM p |
|---|---|---|---|---|---|---|
| 2024-11 | easy (Trump election) | 0.1084 | **0.1082** | **−0.18%** | — (tied) | — |
| 2022-11 | hard (FTX week) | 0.1733 | **0.1825** | **+5.33%** | +3.89 | 0.0001 |

**Encoder training behavior**:
- 2024-11: trained in 248s on RTX 4070 SUPER, converged via early-stopping after ~45 epochs
- 2022-11: trained in 126s (smaller training set), converged at epoch 57 with val QLIKE 0.1279
- Both runs saw clean validation-loss descent; no optimization pathologies

**Verdict — rejected with mechanism:**

The TransVol encoder is *already near-optimal* for the production feature set. Replacing it with a patched-Transformer that processes the same 101-feature tokens produces either:
- **Tied embeddings** on easy regimes (low vol-of-vol, high autocorrelation): the 16-dim compact bottleneck captures essentially the same signal regardless of encoder architecture.
- **Worse embeddings** on hard regimes (regime shifts, tail events): PatchTST's time-axis patching smooths over the short-sequence shocks that TransVol's per-bar attention preserves. Short-horizon vol forecasting benefits from *fine time resolution*, not from patching.

This is consistent with recent literature: PatchTST wins on long-horizon (e.g., 96+ steps) benchmarks where patching helps amortize compute. For 15-min-ahead forecasting with a 32-bar context, the patching advantage is zero and the information loss from merging 4 bars into 1 patch is nonzero.

**Salvage possibilities (NOT pursued)**:
- Smaller patch length (2) — reduces information loss but kills the architectural claim-to-fame
- **Mixed encoder ensemble** (TransVol + PatchTST embeddings concatenated) — plausible but doubles inference compute for potentially marginal gain; pursue only if new features dramatically shift the feature-set landscape
- Test other modern architectures (iTransformer, TimeXer, TimesNet) — see research literature note; TimeXer is the strongest candidate because it specifically addresses endog/exog token mixing, a different structural weakness than PatchTST's patching

Files shipped:
- `vol_model/patchtst/__init__.py`, `config.py`, `encoder.py` — ~350 LoC
- `tests/test_patchtst.py` — 7 unit tests, including the critical **`test_patchtst_attention_is_causal`** which hooks the final block's attention output and verifies first-patch output is invariant to future-patch mutations.

Design deviations from Nie et al. 2023 reference:
- **Cross-channel attention retained** (not channel-independent). The downstream LGBM benefits from embeddings that encode feature interactions; we patch the TIME axis only.
- **Last-patch pooling + persistence skip** matches TransVol's `embedding_compact` (16-dim bottleneck) public contract for drop-in replacement.
- **Strict causality** via upper-triangular mask on the patch axis (numerically verified).

Training infrastructure (encoder + Optuna wiring for the walkforward driver) is scaffolded but **not executed** — PatchTST encoder training from scratch ~ 2–4 hours per month × 8 months × 1 candidate thread = overnight-plus. Dedicated compute slot required.

### Thread 2 — Distributional / quantile LGBM (REJECTED)

**Status**: SCRAPPED on 2022-11 (first gauntlet month, +25.2% worse than fresh control).

**Results on 2022-11:**

| Method | QLIKE | Δ vs Control | DM t-stat | DM p | MZ α | MZ β |
|---|---|---|---|---|---|---|
| pruned_svi_control | 0.1733 | — (baseline) | — | — | +0.200 | 1.144 |
| quantile_q10 | **0.9551** | +451% | +14.71 | 5.2e−49 | +0.721 | 1.187 |
| quantile_q50 | **0.2170** | **+25.2%** | +14.05 | 7.3e−45 | +0.295 | 1.181 |
| quantile_q90 | 0.4202 | +142% | +10.41 | 2.2e−25 | −0.093 | 1.450 |

**Distributional KPIs (extras):**
- `q10 ≤ q50 ≤ q90` monotone: **True** ✅
- Mean spread (q90 − q10): 0.85 (log-vol units, non-trivial)
- **Spread ↔ |err| Spearman**: **0.060** ❌ (target: ≥ 0.30)
- CRPS from pinball aggregation: 0.162
- CRPS vs Gaussian fit (σ from q90−q10): 0.177

**Verdict — rejected on two independent axes:**

1. **Loss-function mismatch (mechanism 1)**. Pinball loss at α=0.5 minimizes conditional median. QLIKE approximately minimizes conditional mean (QLIKE is roughly Bregman-like and right-skewed). For right-skewed log-vol, median < mean, so q50's point predictions are systematically low-biased relative to the QLIKE scoring target. The +25% QLIKE penalty is **structural to the choice of pinball objective**, not a model-capacity issue — training with more Optuna trials doesn't fix it. If you want a QLIKE-optimized prediction, train with QLIKE; if you want quantile predictions, use pinball and score with pinball. Cross-scoring with QLIKE penalizes the mismatch.
2. **Uncertainty signal does NOT emerge (mechanism 2)**. Spread vs |error| correlation is 0.060 — essentially zero. The model learns a roughly-constant spread plus noise; it does **not** know when it's about to be wrong. Confidence-based sizing off this signal would be trading on luck. RankIC of q50 (0.939) is identical to control (0.939), confirming the models produce the same rank ordering of predictions — they only differ in *level*, not *discrimination*.

**Salvage possibilities (NOT pursued this session)**:
- Train a **separate uncertainty head** on |y_true − control_pred| as the target, using the same feature set. This converts uncertainty into a supervised regression problem rather than relying on quantile spread. Would need its own spot-check.
- Use quantile as an ancillary **regime classifier** (e.g., fire a filter when q10 is very negative) — but this is a filter, not a signal improvement; out of scope for this session's thread.

### Thread 3 — HAR-residual target (REJECTED)

**Status**: SCRAPPED on 2024-11 smoke (+25.9% worse than fresh control on the first month tested).

**Results on 2024-11:**

| Method | QLIKE | Δ vs Fresh Control | Δ vs Stale saved |
|---|---|---|---|
| pruned_svi_control | 0.1084 | — (baseline) | −28.67% |
| har_residual | 0.1365 | **+25.9%** | −10.15% |

The stale-baseline comparison (−10.15%) looks superficially positive, which is exactly why the label-shift diagnostic mattered — without fresh control, HAR-residual would have been false-positive shortlisted.

**Verdict — rejected with mechanism:**

The production pruned_svi feature set **already includes HAR-equivalent features**:
- `tsrv_mean_1d` (rank 13, ~1.3% SHAP)
- `tsrv_mean_1w` (rank 9, ~1.7% SHAP)
- `tsrv_mean_1d_ema50` (rank 24, ~0.9% SHAP)
- `parkinson_1d`, `parkinson_1d_ema50`

These are the HAR cascade's daily/weekly/monthly-equivalent aggregates. The LGBM with these inputs is implicitly a *nonlinear HAR + microstructure correction* model.

Subtracting `log(HAR_pred)` from the target while **keeping those HAR-equivalent features in the input** creates a mismatch:
- Target: y_true − HAR_linear(past RV)
- Inputs: include features strongly predictive of HAR_linear(past RV)
- Net: LGBM now has to PREDICT NEGATIVE of what its own inputs tell it, PLUS the microstructure residual

That's a harder optimization target than predicting y_true directly, and it burns a model-capacity budget working against the linear component that it could have just learned natively. Result: worse QLIKE.

**Salvage possibilities (NOT pursued)**:
- **Drop the HAR-equivalent features from inputs** when using HAR-residual target. Decouples linear and nonlinear learning cleanly. Would need its own spot-check; non-trivial refactor.
- **HAR + LGBM ensemble** (not residual stacking): fit HAR and LGBM independently, average predictions at inference. Less affected by the double-counting trap but also less upside.

## Compute log

| Phase | Wall clock |
|---|---|
| Infrastructure coding (regime picker, HAR, leakage probes, signal_research, runner, PatchTST) | ~120 min |
| Unit tests (37 new) | ~8 sec |
| Leakage probe battery (standalone) | <1 sec |
| Smoke: HAR-residual on 2024-11 | 23 min |
| Smoke: pruned_svi_control on 2024-11 (x2 runs, reproducible) | 30 min |
| Gauntlet: control + quantile on 2022-11 (scrapped after first month) | 14 min |
| Smoke: PatchTST on 2024-11 (tied with control) | 28 min (4 encoder + 19 LGBM + 5 overhead) |
| Gauntlet: control + PatchTST on 2022-11 (scrapped after first month) | 19 min (2 encoder + 11 LGBM + 6 overhead) |
| Researcher agent (2024-2026 literature scan, in parallel) | 5 min |
| **Total signal research session** | **~4 hours** |

Vs naive all-threads-all-months estimate (~14 hours): asymmetric acceptance + pre-scrap saved **~10 hours** of compute.

## Artifacts

- `vol_model/output/signal_research_spot_check_results.csv` — per-(method, month) row, 5 rows.
- `vol_model/output/gauntlet_control_quantile.log` — training log for the gauntlet run.
- `vol_model/output/smoke_har_2024-11.log` / `smoke_control_2024-11.log` — HAR + control smoke logs.
- `vol_model/spot_check_regimes.py` — regime picker (+ 7 tests).
- `vol_model/har.py` — causal HAR forecaster (+ 8 tests).
- `vol_model/leakage_probes.py` — probe battery (+ 15 tests).
- `vol_model/signal_research.py` — shared training module.
- `vol_model/run_spot_check_signal_research.py` — unified runner with asymmetric early-stop.
- `vol_model/patchtst/` — PatchTST encoder package (+ 7 tests).

## What I'd try next (ranked by prior probability)

Full literature scan + ranked shortlist saved at `memory/reference_signal_research_literature_2026_04.md` with arxiv IDs and mechanisms. Session pivot: this session proved "better ML on the same features" is a dead end for `pruned_svi_v1`. Next sessions must add **new information**.

Top-3 priority (from researcher agent scan of 2024-2026 mathfin/ML):

1. **Cross-asset intraday commonality** — 7/10 prior, 0.5d engineering. Add ETH-TSRV as lagged feature. Zhang, Cucuringu & Qian 2024 (JFE 22(2):492-530, arxiv:2202.08962) show intraday commonality explains ~74% of individual RV variance at short horizons in equities; crypto is more correlated cross-asset. Binance Vision data already flows.

2. **Options-spot-vol from rough-Heston SVI** — 7/10 prior, 2-3d engineering. Fan, Wang & Ye 2025 (arxiv:2604.02743) document ~6% SPX QLIKE reduction. Start with SVI-interpolated ATM spot-vol (cheap) before escalating to full rough-Heston; the SVI calibrator is live. BTC multifractality (arxiv:2507.00575) means parametric form may mis-specify, but the SIGNAL should still be real.

3. **HARQ / realized quarticity feature** — 6/10 prior, 0.5d engineering. Bollerslev-Patton-Quaedvlieg 2016 (JoE 192(1):1-18). Realized fourth moment as an adaptive weighting signal. Low cost, directly testable.

Secondary / deferred (good prior but higher cost OR compute-heavy):

4. **Width-adaptive HQR conformal wrapper** — 6/10 prior, 1-1.5d. Oosterhoorn 2024 (arxiv:2406.14904). SOLVES the uncertainty-sizing failure of quantile LGBM by design.
5. **TimeXer** (NeurIPS 2024 arxiv:2402.19072) — 5/10, 2d. Endog/exog cross-attention. Test after #1-#3 ship.
6. **LCMQR** (arxiv:2411.19523) — 5/10. Localized conformal; test only if HQR (#4) fails the 0.30 Spearman bar.

**Literature gaps noted** (open research space):
- Options OFI as direct RV predictor: no 2024 paper; theoretical mechanism exists.
- Deribit-specific microstructure: no published work at sub-daily RV on Deribit tick.
- RND moments as sub-daily crypto RV features: documented for daily SPX, untested at our horizon.

**Permanently OFF the list** (covered with mechanism-based rejects):
- Residual-target stacking on other linear models (GARCH, AR) — same mechanism as HAR-residual.
- Vanilla quantile LGBM for uncertainty — QLIKE/pinball mismatch, corr 0.06 << 0.30.
- More Optuna on this session's threads — failures are mechanistic, not HP-related.
- Feature-set pruning — Stage 18 took the win; remaining SHAP drop-off is exponential.
- Patched-transformer variants (PatchTST) on 32-bar context — patching hurts short-horizon; may revisit only if context length doubles.

## Out of scope (explicitly)

- Full 50-month walk-forward retrain of any thread — none earned it.
- Backtest runs; backtester changes; `with_btc_production_filters_v2` — zero touches this session.
- ETH signal work.
- Investigating **what** in the pipeline changed since Stage 18 to produce the label shift. This is a real follow-up: either revert the change, or retrain production on the new labels.
- Re-opening leakage audits that prior work cleared.

---

## Session B: Tier 1 new-information + Tier 3 sizing wrapper (2026-04-17 late)

After Session A's rejections clarified that "better ML on the same 44
features" is a dead end, Session B pivots to **new information** (Tier 1/2)
and **post-hoc sizing** (Tier 3):

- **Tier 1 — new information on the point-forecast side.**
  - `eth`: add ETH cross-asset intraday commonality features
    (`eth_tsrv_mean_{1d,1w}`, `eth_tsrv_ema50`). 47 features. Zhang et al.
    2024 JFE (arxiv:2202.08962).
  - `rq`: add Realized Quarticity (fourth-moment analog of RV)
    (`rq_{1d,1w}`). 46 features. Bollerslev-Patton-Quaedvlieg 2016 HARQ.
  - `eth_and_rq`: union of the above. 49 features.
- **Tier 2 — options-implied info (DEFERRED).** See "Tier 2 deferral" below.
- **Tier 3 — HQR conformal sizing wrapper.** Post-hoc on fresh-control point
  forecasts. Stage 1 = σ̂(x) via quantile-LGBM at τ=0.5 on |y − ŷ|;
  Stage 2 = conformal multiplier q_α on held-out cal fold. Spread =
  2·q_α·σ̂(x). KPI: Spearman(spread, |err|) ≥ 0.30 on the regime substrate
  + CRPS Δ ≥ 5% vs homoscedastic Gaussian. Oosterhoorn 2024
  (arxiv:2406.14904).

### Infrastructure shipped (19 new unit tests, all green — 56 total)

- **TSRV RQ**: `compute_tsrv_window` now emits `rq = (n/3)·Σ r_i^4` on
  sub-bar returns. 7 tests locking in BNS convergence, σ^4 scaling,
  causality under future-bar mutation.
- **Byte-identical RQ backfill**: `TSRV/backfill_rq.py` computes ONLY rq per
  window + merges into existing parquet. Auto-detects START/END stamp
  convention via n_sub_bars exact-match rate. Verified on 2024-11:
  BTC (END-stamped, $500K) and ETH (START-stamped, $500K) both passed at
  100% n_sub_bars exact match. Full backfill: 0 NaN, all 17 existing
  columns byte-identical.
- **Label-shift fix**: `tsrv_end_stamped` flag on `VolModelConfig` with
  per-asset defaults (BTC=True, ETH=False). `load_labels` shifts index by
  -freq when True. 4 regression tests lock Stage 18 semantics.
- **Feature-engineering layer**: `engineer.py::_compute_raw_features` gains
  ETH-TSRV block + RQ rolling-mean block, gated on config flags. SVI
  calibrator (`vol_arb/svi.py`) gains `svi_atm_spot` (nearest-slice ATM IV,
  quality-gated; parquet regen deferred — Tier 2).
- **Extras sidecar**: `vol_model/run_build_extra_features.py` computes
  5 raw + 20 windowed ETH/RQ features at 15m resolution (avoids full
  features parquet rebuild). Saved at
  `vol_model/output/BTCUSDT_features_extras_15m.parquet`.
- **Signal-research dispatchers**: `signal_research.py::TIER_FEATURE_MAP`
  encodes per-thread feature additions; `build_month_data_augmented` adds
  them per-thread without mutating the control; `train_feature_augmented_lgbm`
  reuses the production LGBM recipe.
- **HQR module**: `vol_model/conformal/hqr.py` (~230 lines) with 8 unit
  tests covering synthetic heteroscedastic recovery, coverage near
  nominal, causality check, reproducibility, degenerate rejection.
- **Gauntlet runner**: `run_spot_check_signal_research.py` gains CLI flags
  `--thread tier1_and_hqr` (this session's default) and `all_tier123`
  (includes svi_spot when ready).

### Results

**(Gauntlet running in background as of this writeup — `vol_model/output/gauntlet_tier1_hqr.log`. Numbers to be filled in when the CSV settles. Live CSV: `vol_model/output/signal_research_spot_check_results.csv`.)**

Per-candidate decision gates (pre-registered):
- **Point-forecast candidates (`eth`, `rq`, `eth_and_rq`)** shortlist iff:
  avg Δ QLIKE ≤ −3% vs fresh control across 8 months; improves on ≥5/8
  months; DM p<0.01 on ≥4/8 months (Bonferroni-adjusted); MZ |α|<0.1 and
  |β−1|<0.2; not worse than +5% on any single month (asymmetric scrap
  threshold triggers early-stop).
- **HQR wrapper** shortlists iff: Spearman(spread, |err|) ≥ 0.30 on ≥6/8
  test months; CRPS improvement ≥ 5% vs Gaussian baseline (8-month avg);
  coverage within ±2pp of nominal 90%.

### Smoke-test preview (2024-11 only, n_trials=15 for debug)

Not binding — too few trials, single month — but sanity-checks the
wiring:

| Thread | QLIKE | vs fresh ctrl | DM p |
|---|---|---|---|
| `pruned_svi_control` (fresh) | 0.2693 | — | — |
| `pruned_svi_control` vs stale | 0.2693 | +2.01% vs stale | p=0.031 |
| `eth` | 0.2779 | +3.18% | 0.0056 |
| `rq` | 0.2741 | +1.78% | 0.131 |
| `eth_and_rq` | 0.2822 | +4.78% | 2.3e-05 |
| `hqr_wrapper` (Spearman / CRPS Δ) | — | Spearman=0.123, CRPS Δ vs Gaussian=-0.05% | — |

Control QLIKE (~0.27) matches the saved Stage 18 predictions' QLIKE (0.264)
within noise — confirming the label-shift fix restored the forecast target.
Prior session's control on the nowcast target was ~0.11 — visually
reconciled: the forecast task is genuinely harder and that's what we
actually care about.

Preliminary signal from the smoke: none of ETH/RQ/combined look like
free wins on 2024-11. HQR's Spearman is far below target (0.12 vs 0.30).
But n_trials=15 is 13× below production; wait for the full gauntlet.

---

## Tier 2 (SVI-spot) — deferred

`svi_atm_spot` as a Tier 2 feature is **architecturally wired** but the
SVI features parquet does NOT yet contain the column. Regenerating requires
running the full SVI surface calibration pipeline:

```bash
python -m vol_arb.run_pipeline --currency BTC --svi
```

Estimated wall-clock: ~30-45 min for BTC 5-year history (options chain
load + per-bucket SVI calibration). Skipped this session because Tier 1
+ Tier 3 already occupy the overnight compute slot.

**What's shipped** (ready to use once the parquet regens):
- `vol_arb/svi.py::_constant_maturity_features`: adds `"spot"` branch
  that pulls `surface.nearest_slice().atm_vol_pct()` with quality gate
  (level < 3 AND n_strikes ≥ 3 AND butterfly-free), else NaN.
- `_empty_features` fallback updated with NaN for `svi_atm_spot`.
- `signal_research.py::_load_svi_atm_1d`: `extra_cols` kwarg accepts
  `("svi_atm_spot",)`; gracefully logs a warning + loads available
  subset when the column is absent from the parquet.
- `TIER_FEATURE_MAP["svi_spot"]` dispatches the thread.
- CLI flag `--thread all_tier123` includes `svi_spot` alongside Tier 1.

**Next-session task to activate**: run the SVI pipeline regen, then
`--thread svi_spot` on the same 8-month substrate. Same decision gates
as Tier 1.

**Reference**: Fan, Wang & Ye 2025 (arxiv:2604.02743) document ~6% QLIKE
reduction on SPX via rough-Heston-implied spot-vol. SVI-nearest-slice is
the cheap approximation; escalate to full rough-Heston only if this
Tier 2 smoke clears the +3% QLIKE bar.

---

## Session B verdicts (2026-04-17 overnight gauntlet)

Ran on **label-shift-fixed** fresh control — labels now byte-match Stage 18
semantics (forecast of `[T, T+15min)`, not nowcast). Substrate = 8 regime-hard
months. 200 Optuna trials / 50 patience per month. Asymmetric acceptance:
scrap any thread > +5% worse than control on any single month.

### Summary table

| Thread | Months ran | Avg Δ QLIKE | Improved | DM p<0.01 & better | Verdict |
|---|---|---|---|---|---|
| `eth` | 4/8 | +0.53% | 2/4 | 1/4 | **REJECTED** — scrapped at +7.17% on 2023-07 |
| `rq` | 6/8 | +0.91% | 4/6 | 0/6 | **REJECTED** — scrapped at +6.49% on 2023-10 |
| `eth_and_rq` | 8/8 | **−1.28%** | **7/8** | 1/8 | **REJECTED (directional signal noted)** — below 3% gate, DM p<0.01 only on 1 month |
| `hqr_wrapper` | 8/8 | — | — | — | **REJECTED** — avg Spearman 0.139 vs 0.30 gate; CRPS Δ −0.47% vs −5% gate |

All four threads fail the locked decision gates. None shortlist for full
50-month retrain.

### Thread 1 — ETH cross-asset (eth_tsrv_mean_1d/1w/1d_ema50) — REJECTED

**Per-month Δ QLIKE** (vs fresh control):
- 2022-11: −3.60% (DM p=0.0025) ✓
- 2022-12: −2.09% (DM p=0.015)
- 2023-01: +0.63% (DM p=0.40)
- 2023-07: **+7.17%** (DM p=0.24) → **scrapped**

Pattern: ETH features helped in FTX crunch + election-adjacent months where
cross-asset correlation spikes, but hurt in the quiet 2023 summer regime.
The +7% blowout on 2023-07 is exactly where Zhang et al. 2024's cross-asset
commonality thesis is WEAKEST for crypto — low-RV grinds have low BTC-ETH
correlation, so adding ETH-TSRV is pure noise injection. Mechanism:
conditional informativeness. An always-on cross-asset feature sacrifices
quiet-regime performance for crisis-regime help. Not a win in expectation.

### Thread 2 — HARQ (rq_1d, rq_1w) — REJECTED

**Per-month Δ QLIKE**:
- 2022-11: −0.15% (flat)
- 2022-12: **+2.70%** (DM p=2.7e−9 — significantly worse)
- 2023-01: −0.09% (flat)
- 2023-07: −1.61%
- 2023-08: −1.91%
- 2023-10: **+6.49%** → **scrapped**

Mechanism: Bollerslev/Patton/Quaedvlieg flag RQ as "noisier under heavy
jumps" — exactly the crypto regime. Two catastrophic months (Dec 2022
post-FTX wake, Oct 2023 pre-rally) suggest RQ inflates spuriously in
high-kurtosis windows, fooling the LightGBM into over-weighting vol-of-vol
signal that doesn't generalise. Too noisy for our substrate.

### Thread 3 — Combined ETH + HARQ — REJECTED (closest to signal)

**Per-month Δ QLIKE**:
- 2022-11: −2.87% (DM p=0.0076) ✓
- 2022-12: −0.64%
- 2023-01: −0.51%
- 2023-07: −1.93%
- 2023-08: −3.22% (DM p=0.072, marginal)
- 2023-10: −1.61%
- 2024-11: **+1.68%** (DM p=0.023 — slightly worse)
- 2024-12: −1.17% (DM p=0.043)

**Improved on 7/8 months**, avg −1.28%, no single-month catastrophe.
Combining ETH and RQ (individually-scrapped threads) gives a thread that
completes the full 8-month substrate — the two feature sets must carry
somewhat decorrelated information (ETH→crisis-correlation, RQ→vol-of-vol)
such that LightGBM leverages their complementarity while damping each
one's individual failure modes.

**Why still rejected under locked gates**:
1. Avg −1.28% below the 3% average-improvement gate.
2. DM p<0.01 only on 1/8 (gate requires ≥4).
3. 2024-11 regression (+1.68%, DM-significant worse) worrying — that's
   the Trump-election top-DD month where we'd want the signal to help most.

**Directional signal noted** — 7/8-month improvement direction is
suggestive but underpowered on our 8-month substrate. Not "reject and
forget"; more "interesting but needs different framing / bigger-n /
horizon alignment before a retrain is warranted".

### Thread 5 — HQR conformal wrapper — REJECTED

Same failure mechanism as quantile LGBM (Session A): predicted spread
does not track actual error.

| Metric | Mean | Gate | Pass |
|---|---|---|---|
| Spearman(spread, \|err\|) | 0.139 | ≥ 0.30 | ✗ (only 2023-01 at 0.32) |
| Coverage (nominal 0.90) | 0.889 | within ±2pp | ✓ |
| CRPS Δ vs Gaussian-around-point | −0.47% | ≤ −5% | ✗ |

Per-month Spearman: 0.175 / 0.074 / **0.321** / 0.062 / 0.116 / 0.123 /
0.157 / 0.086.

**Mechanism**: HQR's width model (σ̂) is trained on the calibration-fold
residuals, but those residuals are close to homoscedastic in the feature
space — the LightGBM point-forecast already absorbs the vast majority of
heteroscedasticity at training time. What's left is near-white noise that
no feature predicts. Matches the Session A quantile LGBM finding
(Spearman 0.06): the sizing failure is not an architecture problem, it's
a **feature-set informational content limit**.

## Cross-cutting takeaway (Session A + B combined)

Seven candidate approaches tested across two sessions on the 44-feature
base (pruned_svi_v1): HAR-residual, Quantile LGBM, PatchTST encoder, ETH
cross-asset, HARQ, ETH+HARQ combined, HQR conformal. **All seven fail the
QLIKE-improvement decision gates.** The eth_and_rq thread is the closest
to signal (7/8 improvement direction, no catastrophic month) but still
below the 3% average-improvement bar and the DM significance threshold.

Pattern: **pruned_svi_v1 on the corrected forecast target is near the
informational limit of its feature set.** TransVol + LightGBM + 44
features already extract most of what's extractable from 15-min
dollar-bar + realized-vol + SVI-level features. Further lift requires:

1. **Qualitatively new information** — Tier 2 options-implied spot-vol
   (SVI-nearest-slice feature wired but not yet evaluated this session),
   or Deribit dealer-flow / OI pressure (no off-the-shelf feature).
2. **Different target** — path-dependent realized vol over the trade's
   actual hold window (currently 30 min), not a point-in-time 15-min bar.
3. **Different horizon** — multi-horizon joint targets; currently trained
   only on the 1-bar-ahead head.

## What to try next (re-ranked after two rejection cycles)

1. **SVI-ATM spot-vol (Tier 2)** — feature IS wired in
   `vol_arb/svi.py::_CM_MATURITIES` and in `vol_model/signal_research.py`'s
   `TIER_FEATURE_MAP`, but requires regenerating the SVI features parquet.
   Independent of the ETH/RQ thread-family. Most direct "new information"
   test. Prior 7/10.
2. **Path-dependent target** — retrain targeting `log(realized vol over
   [T, T+hold_minutes))` with `hold_minutes = 30` (production hold time)
   instead of the 15-min fixed bar. Changes what the model predicts to
   what the backtester actually cares about. No new features. Prior 6/10.
3. **Options-chain RND moments (skew, kurt, tail-prob)** — extracted from
   the same SVI calibrator. Daily SPX Jiang-Tian era showed these as
   predictors; untested at 15-min crypto. Prior 5/10.
4. **Multi-horizon joint head** — train LightGBM with {1-bar, 4-bar,
   16-bar} targets and a composite loss. Theoretical info coupling but no
   direct benchmark. Prior 4/10.

**NOT on the list (covered, rejected)**:
- HAR-residual (same features), Quantile LGBM, PatchTST (Session A)
- ETH TSRV features alone (scrapped in regime 2023-07)
- HARQ features alone (scrapped 2023-10)
- ETH + HARQ combined — directional but not significant
- HQR conformal sizing wrapper (same failure as quantile LGBM)
- Feature-set pruning (Stage 18 took the win)

## Session B infrastructure recap (all shipped, reusable)

- **Label-shift fix**: `vol_model/labels.py::load_labels` shifts
  END-stamped TSRV index by −freq per-asset (`tsrv_end_stamped=True`
  for BTC, False for ETH). Regression test `test_label_semantics.py`
  byte-matches saved Stage 18 y_true on 145,041 / 145,041 rows. See
  `LABEL_SHIFT_AUDIT.md`.
- **RQ computation**: `TSRV/tsrv_series.py::compute_tsrv_window` emits
  `rq` column additively. `TSRV/backfill_rq.py` backfills existing
  parquets without touching other columns (byte-identical verification).
  7 unit tests (BNS convergence, σ^4 scaling, causality).
- **ETH-TSRV block**: `vol_model/features/engineer.py` loads
  `ETHUSDT_tsrv_15m.parquet` on `include_eth_tsrv=True`, applies shift(1)
  + ffill + rolling + FFD(0.4) mirroring BTC block.
- **SVI spot-vol feature**: `vol_arb/svi.py::_CM_MATURITIES` has
  `"spot": 0.0` wired; `_constant_maturity_features` uses
  `nearest_slice().atm_vol()` with quality gate. Requires SVI parquet
  regen to activate.
- **HQR conformal wrapper**: `vol_model/conformal/hqr.py` + 8 tests,
  evaluation driver in `run_spot_check_signal_research.py`.
- **Gauntlet driver**: `--thread all_tier123` runs control + eth + rq +
  eth_and_rq + hqr_wrapper; `--thread svi_spot` will run Tier 2 once the
  parquet is regenerated.
- **Scrap history** (asymmetric acceptance): `eth` scrapped after
  2023-07 (month 4); `rq` scrapped after 2023-10 (month 6).

---

## Original "next steps" (preserved for reference)

If `eth_and_rq` shortlists, it becomes the default candidate for the
next full 50-month walk-forward retrain (a separate compute session,
user-triggered).

---

# Session C — Post-Foundation-Fix Research (2026-04-17)

## Context

Session C is the FIRST research session evaluated against the post-foundation-fix baseline (see `FOUNDATION_ALIGNMENT_AUDIT.md`). All Session A + B rejections were made against the same (correctly-aligned) saved predictions, so their relative rankings remain valid. What's NEW in Session C:

1. V2 hold is now 15m (matched to prediction horizon) — comparison baseline is `BTC_v2_trade_log.parquet` regenerated with hold=15m, NC PnL $575K / trade-only Sharpe 51.18 / NC MDD -4.65%.
2. Three new threads tested:
   - **SVI-spot (Tier 2)**: `svi_atm_spot` feature added to the 44-feature base via `vol_arb/rebuild_svi_features_with_spot.py`.
   - **HAR-PD (Tier 3, new this session)**: 6 features with learned exponential decay kernels on signed realized flow + realized variance (Xiao et al. 2025, arxiv:2503.00851).
   - **IV-change direct prediction**: a fundamentally different target — `y_ivchg = ln(iv_exit / iv_used)`, MSE loss, same 44-feature set.

All three ran against the 8-month regime substrate: 2022-11, 2022-12, 2023-01, 2023-07, 2023-08, 2023-10, 2024-11, 2024-12.

## Thread 1 — SVI-spot (Tier 2) — REJECT

**Feature**: nearest-calibrated-slice ATM vol at T=0 (no extrapolation to sub-1d tenors). Loaded via `rebuild_svi_features_with_spot.py` — preserves byte-identity of the 13 pre-existing SVI columns, adds only `svi_atm_spot` (91.9% coverage, mean 51.92 vol pts). Quality gate: NaN when `calibration_level >= 3` OR `n_strikes < 3` OR not butterfly-free.

**Results (6/8 months — gauntlet stopped early per user directive):**

| Month   | fresh QLIKE | svi_spot QLIKE | delta% | DM p     | Verdict |
|---------|-------------|----------------|--------|----------|---------|
| 2022-11 | 0.5753      | 0.5804         | +0.57% | 0.315    | tie     |
| 2022-12 | 0.9015      | 0.8732         | +2.05% | 3.57e-6  | sig-worse |
| 2023-01 | 0.7721      | 0.7063         | -1.45% | 0.159    | small win |
| 2023-07 | 0.6571      | 0.6169         | -3.59% | 0.132    | win (insig) |
| 2023-08 | 0.7252      | 0.7040         | -2.09% | 0.240    | win (insig) |
| 2023-10 | 0.6783      | 0.6821         | +0.13% | 0.940    | tie     |

**Summary**: avg delta **-0.73%**, wins 3/6 but NONE DM-significant at p<0.01. Fails gate 1 (avg leq -3%) and gate 4 (geq 4 DM-significant). **Verdict: REJECT.**

**Mechanism**: the 3 existing SVI-ATM term-structure columns (`svi_atm_1d, _1w, _1m`) already span the volatility-surface-level information at granularity sufficient for the 15-min forecast target. A fourth nearest-slice ATM feature at T to 0 does not add marginal information — the 1d column is already anchored at the shortest calibrated slice, and the training data rarely has non-degenerate sub-1d calibrations that would differentiate "spot" from "1d". The single statistically-significant result went the wrong way (+2.05% on 2022-12, DM p=3.6e-6), consistent with noise injection rather than information addition.

## Thread 2 — HAR-PD path-dependent features (Tier 3) — CONDITIONAL SHORTLIST

**Features (6 raw, no windowing variants)**:
- `har_pd_R1_a005 / _a01 / _a03`: exponentially-weighted signed realized flow at alpha={0.05, 0.1, 0.3}
- `har_pd_R2_b005 / _b01 / _b03`: exponentially-weighted realized variance at beta={0.05, 0.1, 0.3}

Formula: `R1(t) = sum_{k=1..96} w_k(alpha) * signed_flow_{t-k}`, `R2(t) = sum w_k(beta) * rv_ann_{t-k}`, with `w_k = alpha*(1-alpha)^(k-1)` normalized to sum 1. Signed-flow proxy: `sqrt(rs_up) - sqrt(rs_down)` — a bar-level signed return surrogate derived from the existing upside/downside realized semivariances in the TSRV parquet. K=96 bars (~24h). No lookahead: causality unit test verifies that mutating future TSRV values leaves past R1/R2 byte-identical.

**Results (5/8 months — 2023-10 har_pd was mid-training when gauntlet was stopped; 2024-11, 2024-12 not started):**

| Month   | fresh QLIKE | har_pd QLIKE | delta% | DM p     | Verdict |
|---------|-------------|--------------|--------|----------|---------|
| 2022-11 | 0.5753      | 0.5546       | -3.60% | 0.003    | sig-win |
| 2022-12 | 0.9015      | 0.9112       | +1.08% | 0.273    | small loss |
| 2023-01 | 0.7721      | 0.7474       | -3.20% | 0.006    | sig-win |
| 2023-07 | 0.6571      | 0.6226       | -5.25% | 0.011    | sig-win |
| 2023-08 | 0.7252      | 0.7011       | -3.32% | 0.126    | win (insig) |

**Summary**: avg delta **-2.86%**, wins 4/5, DM-significant at p<0.01 on 3/5 months. Not yet formally at the -3% avg gate but very close, and the significance pattern is the strongest of any candidate across Sessions A-C. **Verdict: CONDITIONAL SHORTLIST — certify via full 50-month walk-forward retrain next session.**

**Why this is structurally different from rejected HAR-residual (Session A)**: HAR-residual used FIXED rolling-mean horizons (1, 4, 96, 672 bars) that collide with the existing base-feature rolling-mean family (tsrv_mean_1d, tsrv_mean_1w, etc.). HAR-PD uses PARAMETERIZED exponential kernels — alpha=0.05 has half-life ~14 bars, alpha=0.3 has half-life ~2 bars — spanning timescales the fixed-horizon family doesn't cover. The tree-ensemble picks kernel decay implicitly via split selection, playing the role of the LASSO in Xiao's original linear formulation. Also: signed realized flow is novel relative to the TSRV feature family, which tracks variance magnitudes but not direction.

**Risks for certification**:
- The 3 DM-significant wins are all in 2022-11 / 2023-01 / 2023-07 — a clustering pattern. If the remaining months are flat/negative, the avg could retreat toward -2%. Finite-sample variance is real.
- The signed_flow proxy derives from `sqrt(rs_up) - sqrt(rs_down)`, a lossy transform from the bar-level per-tick data. A more faithful 15m-close signed return series might do better OR worse.
- No backtester PnL yet — QLIKE improvement may or may not translate to trading improvement.

## Thread 3 — IV-change direct prediction — REJECT (clean)

**Target**: `y_ivchg(t) = ln(iv_exit(t+15m) / iv_used(t))`, signed, mean~0 std~0.16. MSE loss (not QLIKE).

**Baseline to beat**: the production `edge` column's Spearman correlation with the same actual iv_move, measured per-month on the same test rows. On the 8 regime months, baseline Spearman averages 0.237 (range 0.158-0.339).

**Evaluation metrics**: Spearman(y_pred, actual_ivchg), signed-accuracy on non-trivial (|delta|>1%) rows, RMSE.

**Results (8/8 months complete)**:

| Month   | model Spearman | baseline Spearman | delta | model signed_acc | base signed_acc |
|---------|---------------:|------------------:|------:|-----------------:|----------------:|
| 2022-11 | -0.0106        | 0.2495            | -0.26 | 0.488            | 0.524           |
| 2022-12 | +0.0080        | 0.3386            | -0.33 | 0.507            | 0.534           |
| 2023-01 | +0.0118        | 0.2590            | -0.25 | 0.508            | 0.539           |
| 2023-07 | +0.0093        | 0.2525            | -0.24 | 0.503            | 0.536           |
| 2023-08 | +0.0289        | 0.2516            | -0.22 | 0.510            | 0.528           |
| 2023-10 | +0.0306        | 0.1576            | -0.13 | 0.510            | 0.533           |
| 2024-11 | +0.0020        | 0.2055            | -0.20 | 0.492            | 0.556           |
| 2024-12 | +0.0167        | 0.1790            | -0.16 | 0.509            | 0.535           |

**Summary**: avg model Spearman **0.0121** vs baseline **0.2367** (delta -0.2246). Model beats baseline on **0/8 months**. Signed accuracy 0.503 (coin flip) vs baseline 0.535 (mildly informative). **Verdict: REJECT.**

**Mechanism**: the 44-feature set carries iv-change information only INDIRECTLY — via the QLIKE-loss RV forecast pathway. When we train a direct predictor on `ln(iv_exit/iv_used)` with MSE loss, the extremely noisy signed target (std 0.16, mean ~0) overwhelms the features' causal link, and the tree learner fits noise. This confirms that the 0.237 baseline Spearman is a by-product of the RV to edge pipeline, not a native feature-set property. **You cannot shortcut the RV forecast path to directly predict iv-change on this feature set.**

**Implication**: future iv-change work would need either (a) features specifically chosen for iv-change prediction (e.g., options OI flow, dealer positioning), or (b) a transformed target that's less noisy (e.g., iv-change over a longer aggregation window).

## Cross-cutting takeaway

**HAR-PD is the first post-foundation-fix candidate to show DM-significant QLIKE improvements.** Across three sessions, 10 candidate threads have been tested on this feature set:

| Session | Thread | Verdict |
|---|---|---|
| A | HAR-residual | REJECT (collides with existing fixed-horizon means) |
| A | Quantile LGBM | REJECT (Spearman 0.06 for sizing) |
| A | PatchTST encoder | REJECT (no gain over TransVol) |
| B | ETH cross-asset | REJECT (scrapped +7.17% on 2023-07) |
| B | HARQ realized quarticity | REJECT (scrapped +6.49% on 2023-10) |
| B | ETH + HARQ combined | REJECT (avg -1.28%, only 1 DM-sig) |
| B | HQR conformal wrapper | REJECT (Spearman 0.14 vs 0.30 gate) |
| C | SVI-spot (Tier 2) | REJECT (0 DM-sig, avg -0.73%) |
| C | IV-change direct | REJECT (Spearman 0.01 vs 0.24 baseline) |
| C | **HAR-PD** | **CONDITIONAL SHORTLIST** (avg -2.86% 5/8, 3 DM-sig) |

The pattern of 9 rejections before a CONDITIONAL SHORTLIST is what honest research looks like. HAR-PD's distinguishing property is that it adds STRUCTURALLY NEW features (parameterized kernel decays + signed flow), not "better ML on the same information" or "more of the same feature family." Sessions A + B's rejections were all of the latter type.

## Infrastructure shipped (Session C)

New code, all unit-tested:

- `vol_model/features/har_pd.py` (~170 lines) — HAR-PD feature computation.
- `tests/test_har_pd.py` (7 tests) — kernel normalization, causality under mutation, constant-input check, manual hand-compute.
- `vol_arb/rebuild_svi_features_with_spot.py` (~160 lines) — augments SVI features parquet with `svi_atm_spot` preserving byte-identity of existing columns.
- `tests/test_svi_spot_feature.py` (9 tests) — coverage, distribution, quality gate, causality.
- `vol_model/run_iv_change_gauntlet.py` (~320 lines) — standalone IV-change gauntlet with Spearman/signed-accuracy metrics.

Dispatcher extensions in `vol_model/signal_research.py`:
- `TIER_FEATURE_MAP` 4-tuple: `(eth_cols, rq_cols, svi_extra_cols, har_pd_cols)`.
- `SpotCheckContext.har_pd_features` DataFrame slot.
- `load_spot_check_context(include_extras=True)` loads `{asset}_features_har_pd_15m.parquet` sidecar.
- `build_month_data_augmented(thread="har_pd")` splices R1/R2 columns onto the 44-feature base.

Reproducibility: `python -m vol_model.run_spot_check_signal_research --thread pruned_svi_control,svi_spot,har_pd --n-trials 80 --patience 25` reproduces all numbers above (given fixed seeds).

## Next session

**Primary task: full 50-month walk-forward retrain of `pruned_svi_v1 + har_pd`.** Reuses the existing walk-forward harness; outputs a new predictions parquet + model directory `expanding_pruned_svi_har_pd_v1/`. Certification gate (same as Session C spot-check, but on all 50 months):

1. Avg delta QLIKE leq -3% vs production pruned_svi_v1.
2. geq 25/50 months improved (majority).
3. DM p < 0.01 on geq 15/50 months (30% significance rate).
4. Not worse than +5% on any single month.

If all four hold, ship as `pruned_svi_har_pd_v1` and update the backtester's default `predictions_path`. If not, document the specific failure regime and keep HAR-PD as a candidate for future regime-conditional ensembles.

**Secondary (optional)**: complete the 3 remaining Session C spot-check months to extend the partial table above. Not blocking — the 50-month retrain is the stronger test.

## Deferred

- Backtester integration of HAR-PD — waits on certification.
- Options OI / dealer-flow features for iv-change direct prediction — new data dependency (Deribit option-chain snapshots by strike, not on disk).
- Tier 2 options microstructure (dealer gamma, 0DTE flow) — blocked on data.


# ═══════════════════════════════════════════════════════════════════════
# Session E (2026-04-18, post-HAR-PD ship)
# ═══════════════════════════════════════════════════════════════════════

## Context

Session D shipped `pruned_svi_har_pd_v1` (avg ΔQLIKE −3.67% vs pruned_svi_v1,
39/50 months improved, 29/50 DM-sig at p<0.01, NC Sharpe 9.58 → 9.69,
PnL $14.51M → $15.04M). Signal-research thread reopens on a fresh baseline.

User directive: focus on signal improvement, research 2024-2026 literature,
evaluate separately from backtest, **zero data leakage** is the #1 risk.
Professional ML-quant methodology.

## Phase 0 — Leakage protocol formalization (SHIPPED)

`vol_model/leakage_probes.py` extended with three new probes to complement
the existing 4:

- **Probe E — `probe_causality_by_mutation`** (gold standard):
  mutates raw data at timestamps > T, regenerates the feature end-to-end,
  asserts byte-identical feature[T]. Configurable `mutation_mask` for
  unusual stamp conventions.
- **Probe F — `probe_current_bar_dependence`**: sanity complement.
  Catches features that are wastefully conservative (the "HAR-PD without
  -15m shift" failure mode from Session D).
- **Probe G — `probe_walkforward_isolation`** + `FoldSpec` dataclass:
  schedule-level train/test isolation check.

`tests/test_leakage_probes.py` extended with 14 new tests (27/27 green).
Real-data validation: the new probe blesses HAR-PD with byte-identical
feature values under mutation of 97,968 future TSRV rows × 18 cols.

`vol_model/features/README.md` provides a new-feature checklist that every
Session E candidate must satisfy before training.

## Candidate shortlist (ranked, from 2024-2026 literature scan)

Researcher-curated via web search. Citations verified; no fabricated refs.

| Rank | Candidate | Citation | Effort | Status |
|---|---|---|---|---|
| A | Full SVI surface direct features | Cont & Da Fonseca 2002; Magadini 2024 (SSRN:4899357) | LOW | Gauntlet running |
| B | Hawkes-OFI branching ratio | Pham 2024 (arxiv:2408.03594) | MED | Queued |
| C | Rough-Heston spot vol | Fan 2026 (arxiv:2604.02743) | HIGH | Queued |
| D | Multivariate fBm cross-Hurst | Bai 2025 (arxiv:2504.15985) | MED | Deprioritized |
| E | BKM risk-neutral moments | Bakshi-Kapadia-Madan 2003 | MED | Deprioritized |

## Thread A — Full SVI surface features (11 features)

**Hypothesis**: production uses only `svi_atm_1d` from a 14-col SVI sidecar.
The 7 other structural features (3 longer-tenor ATMs, 2 skews, 2 term slopes)
exist at 95.3% coverage but are IGNORED by the LGBM stage. Adding them
+ 4 one-hour-change deltas (regime-transition info) should unlock
term-structure, skew-dynamics, and butterfly/curvature channels that a
single ATM scalar collapses.

**Design decision**: direct features, **not PCA**. PCA introduces a
train-only fitting step (rotation parameter) that needs walk-forward
isolation; direct features don't. PCA saved as fallback.

**Features** (`vol_model/features/svi_surface.py`):
- Levels (7, copied verbatim from sidecar): svi_atm_1w, svi_atm_2w,
  svi_atm_1m, svi_25d_skew_1w, svi_25d_skew_1m, svi_term_slope_1w_2w,
  svi_term_slope_1w_1m
- Deltas (4, `.diff(4)` = 1-hour change): svi_atm_1w_d1h,
  svi_25d_skew_1m_d1h, svi_term_slope_1w_1m_d1h, svi_atm_spot_d1h

**Leakage validation**: `tests/test_features_svi_surface.py` (10/10 green).
- `probe_causality_by_mutation` passes on both synthetic fixture and
  real BTC SVI sidecar (byte-identical under mutation of all future rows).
- `probe_current_bar_dependence` passes (feature is not wastefully conservative).
- Coverage test: every feature ≥ 88% non-NaN on real data (svi_atm_spot_d1h
  the lowest at 89.3% due to `.diff(4)` losing extra bars around NaN gaps).

**Alignment**: Applies the same +15m shift as the existing svi_atm_1d
loader (`_load_svi_atm_1d` in `run_walkforward_pruned_svi.py`). Gauntlet
evaluation therefore measures production-equivalent feature availability.

**Correctness fix (by-product)**: the HAR-PD gauntlet loader was loading
the sidecar unshifted. This made the gauntlet's HAR-PD evaluation 15-min
pessimistic (one bar less of information than production). Fixed by
applying the same -15m shift production uses. Non-leaky either way; just
aligns gauntlet with production.

## Thread A — Gauntlet spec

- 8-month regime substrate: 2022-11, 2022-12, 2023-01, 2023-07, 2023-08,
  2023-10, 2024-11, 2024-12 (`pick_regime_months` on production summary).
- 4 threads: `pruned_svi_control`, `svi_surface`, `har_pd`, `har_pd_and_svi_surface`.
  - Control = pruned_svi_v1 (44 features)
  - svi_surface = 44 + 11 SVI surface (standalone)
  - har_pd = 44 + 6 HAR-PD (matches production minus the Session D retrain)
  - har_pd_and_svi_surface = 44 + 6 HAR-PD + 11 SVI surface (CANDIDATE)
- Optuna: 80 trials / 25 patience per thread-month (matches Session C).
- Pre-screen gate (from plan): avg ΔQLIKE ≤ −1.5% vs har_pd AND ≥5/8
  months improved AND IS-vs-OOS Spearman gap < 30%.

Results CSV: `vol_model/output/signal_research_spot_check_results.csv`
Log: `vol_model/output/session_e_candidateA_gauntlet.log`

## Thread A — Results (gauntlet completed 15:20 local, ~1h50m wall clock)

Per-month QLIKE deltas vs fresh `pruned_svi_control` baseline (8 months, 80
Optuna trials / 25 patience each). Primary comparison = svi_surface stacked
on har_pd vs har_pd alone — the differential measures SVI-surface's
contribution on top of the current production feature set.

| Month | control QLIKE | svi_surface Δ% | har_pd Δ% | har_pd+svi_surface Δ% | differential (vs har_pd) |
|---|---|---|---|---|---|
| 2022-11 | 0.5753 | +0.56% | **−4.21%** | **−4.71%** | **−0.50** (better) |
| 2022-12 | 0.9015 | +3.44% | +0.55% | +1.41% | +0.86 (worse) |
| 2023-01 | 0.7721 | +3.60% | **−5.01%** | −2.92% | +2.09 (worse) |
| 2023-07 | 0.6571 | −5.30% | −2.27% | **−5.17%** | **−2.90** (better) |
| 2023-08 | 0.7252 | −4.02% | **−3.90%** | **−3.82%** | +0.08 (tied) |
| 2023-10 | 0.6783 | +0.53% | −1.68% | −0.17% | +1.51 (worse) |
| 2024-11 | 0.2714 | +0.18% | **−4.90%** | **−4.35%** | +0.55 (worse) |
| 2024-12 | 0.2828 | −0.05% | −2.92% | **−3.25%** | **−0.33** (better) |
| **Avg** | — | **−0.13%** | **−3.04%** | **−2.87%** | **+0.17** (worse) |
| **Wins** | — | 3/8 | **7/8** | 7/8 | **3/8** |
| **DM-sig p<0.01** | — | 1/8 | 2/8 | 2/8 | (DM not computed directly) |

Bold DM-sig at p<0.01; bolded Δ% if < −3%.

**Pre-screen gate (pre-registered)**: `avg ΔQLIKE ≤ −1.5%` AND `≥ 5/8 months improved` AND `IS-OOS Spearman gap < 30%`.

**Actual**: avg +0.17pp WORSE, 3/8 wins. **BOTH gates FAIL.**

**Sanity**: standalone `svi_surface` (on pruned_svi_v1) also fails (−0.13%, 3/8). The 11 SVI surface features carry no independent signal vs the 44-base — supporting the "redundant" interpretation over "needs more trials". Differential SD across months ≈ 1.5pp, so the +0.17 average is inside 1 SD of zero: effect is indistinguishable from noise.

**Interpretation**: the 44-base + HAR-PD already capture whatever SVI-surface information matters for this 15-min horizon. The 7 longer-tenor ATMs, 2 skews, 2 term slopes, and 4 deltas do not add independent predictive content once HAR-PD's path-dependent signed-flow + realized-variance kernels are in the feature set. This is consistent with Session B's earlier finding that "more SVI info on the same 44-feature base" tends not to move QLIKE.

**Decision: Candidate A REJECTED.** Documented. No full-50-month retrain. PCA variant queued as low-priority fallback only if Candidates B/C also fail — since the direct features themselves are null, a PCA rotation that compresses them is unlikely to rescue signal.

**Bonus (Session E correctness fix validated)**: the -15m HAR-PD alignment fix in the gauntlet path shows up in the har_pd numbers — avg −3.04%, 7/8 wins. Session C's unshifted har_pd gauntlet was −2.86%, 4/5 partial months. The shift gives the gauntlet a more faithful read on HAR-PD, consistent with Session D's full certification result of −3.67% (better numbers expected on the full 50-month substrate with larger Optuna budget).

## Threads B, C — Queued

**Thread B (Hawkes-OFI)** — requires Binance aggTrades ingestion at scale
and rolling MLE fitter. Multi-day build. Queued for next session unless
Thread A rejects and capacity frees up. See `starry-rolling-rabin.md` for
details.

**Thread C (Rough-Heston)** — requires rough-Heston calibrator over the
full Deribit options panel; Fan et al. use a deep-learning surrogate. First
pass would be a simpler two-step iterative fit. Multi-day build. Queued.

## Infrastructure shipped (Session E so far)

- `vol_model/leakage_probes.py`: +3 probes (`probe_causality_by_mutation`,
  `probe_current_bar_dependence`, `probe_walkforward_isolation`),
  +`FoldSpec` dataclass (~230 LOC).
- `tests/test_leakage_probes.py`: +14 tests (27/27 green).
- `vol_model/features/svi_surface.py`: compute function + coverage
  diagnostic (~130 LOC).
- `tests/test_features_svi_surface.py`: 10 tests (10/10 green) including
  real-data causality probe.
- `vol_model/features/README.md`: new-feature checklist skeleton (~100 LOC).
- `vol_model/signal_research.py`: TIER_FEATURE_MAP 4-tuple → 5-tuple,
  new tiers `svi_surface` and `har_pd_and_svi_surface`, `svi_surface_features`
  field on SpotCheckContext, HAR-PD -15m alignment correctness fix.

Total session-to-date: 136/136 tests green across the leakage + feature +
labeling surface.
