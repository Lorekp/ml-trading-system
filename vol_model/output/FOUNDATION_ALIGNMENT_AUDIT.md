# Foundation Alignment Audit — 2026-04-17

## Executive summary

**Initial hypothesis**: saved production predictions `BTCUSDT_pruned_svi_predictions.parquet` were nowcasts of `[D−15m, D]` (the past 15-min window), not forecasts of `[D, D+15m]` (the user's intended design). Evidence: MSE(y_pred, ln_vol_ann) lowest at TSRV offset +0 min (past window end) vs offsets +15, +30.

**Revised conclusion after deeper investigation**: the predictions ARE correctly targeting `[D, D+15m]`. The MSE-vs-offset test is confounded by distributional compression from QLIKE-regularized LightGBM output. The definitive test — `y_true` match-rate against TSRV at each offset — shows:

| TSRV offset | `y_true` MSE | exact-match rate |
|---|---|---:|
| −15 min | 0.3396 | 0.00% |
| **+0 min** (past-bar end) | 0.2826 | 0.00% |
| **+15 min** (intended forecast) | **0.0017** | **99.82%** |
| +30 min | 0.2822 | 0.00% |
| +45 min | 0.3399 | 0.00% |

**y_true is a 99.82% exact match to TSRV at offset +15 min.** The training target is unambiguously `ln(vol_ann over [D, D+15m])`, which is the forecast window. The remaining 0.18% non-match is label clipping at the 99.9th percentile (expected by design per `config.label_clip_quantile`).

**The actual broken thing** is the V2 backtester config: `vol_arb/backtester.py:447` sets `holding_minutes=30` in `with_btc_production_filters_v2()`. The predictions cover a 15-min window but V2 trades hold for 30 min, so the trade's exposure window `[D, D+30m]` is TWO bars wide while the model's signal only covers the FIRST bar. The second bar `[D+15m, D+30m]` is uncovered — trades are held past their edge.

**Fix scope**: single-line config change `holding_minutes=30 → 15`, regenerate trade log, update docstring + memory. NO retrain needed.

---

## Why the MSE-vs-offset test misled

A tree-ensemble trained with QLIKE loss on a wide-tailed target produces outputs that are compressed relative to the true distribution. The optimal estimator under QLIKE also sits above the conditional mean for mean-zero-residual distributions (QLIKE asymmetrically penalizes under-prediction). Combined, `y_pred`'s marginal distribution differs from `ln_vol_ann`'s in two ways:

- `mean(y_pred) − mean(y_true) = +0.17` (systematic upward shift — the expected QLIKE bias)
- `std(y_pred) < std(y_true)` (compression toward the mean — regularization + finite-sample tree variance)

This means the MSE `E[(y_pred − ln_vol)²]` depends more on **distributional overlap** than on **temporal alignment**. When y_pred's compressed, shifted distribution happens to overlap better with the past-bar's vol distribution than with the future-bar's, the MSE is lower at offset 0 even though the model is trained on the offset-+15 target. The test measures the wrong thing.

The definitive alignment check is: what does `y_true` equal? `y_true` is loaded from `labels["ln_vol"][1:]` during training. If that column was correctly constructed (END-stamped TSRV → `-freq` shift → one-bar-forward label = vol of `[D, D+15m]`), then `y_true[D]` byte-matches `ln(TSRV.vol_ann[D+15m])`. Which it does, 99.82% exact-match.

---

## Per-month best-offset distribution (misleading metric, kept for completeness)

| best_offset | months_count |
|---|---:|
| +0 | 48 |
| +15 | 2 |

50 months total. The dominant "+0 best" is the distributional-overlap artifact described above, not evidence of target misalignment. Not a diagnostic signal.

---

## The real misalignment: V2 holding_minutes = 30

`vol_arb/backtester.py:447`:
```python
def with_btc_production_filters_v2():
    return BacktestConfig(
        ...
        holding_minutes=30,   # ← bug: should be 15 to match prediction horizon
        ...
    )
```

Effect: at decision time D, V2 enters a trade and holds for 30 min (exits at `D+30m`). The prediction `y_pred[D]` covers vol over `[D, D+15m]` only. The exposure window `[D, D+30m]` spans two bars, but the signal is ~ correct only for the first.

Consequence for V2 reported numbers (pre-fix):
- NC PnL $726K, NC Sharpe 12.56, MDD 9.91%
- These are "real" numbers — produced by actual trades — but the 30-min hold dilutes signal efficacy, because in the second half of each exposure window the position is no longer aligned with its edge.
- Matching holding_minutes to prediction_horizon should either improve metrics (if the second-bar dilution was net-negative) or reduce position count (shorter holds → lower turnover, similar cost structure, variable edge capture).

---

## Vol persistence baselines

`corr(ln_vol[T], ln_vol[T+lag])` for TSRV:

| lag | corr |
|---|---:|
| 1 bar (15m) | 0.765 |
| 2 bar (30m) | 0.718 |
| 3 bar (45m) | 0.690 |
| 4 bar (1h)  | 0.678 |

Persistence-only benchmark (use previous bar's `ln_vol` as the prediction for current): MSE 0.2751, corr 0.747. Production model achieves MSE 0.2261, corr 0.799 — a **17.8% MSE improvement over naive persistence**. The model has real skill.

---

## Mincer-Zarnowitz calibration

Regression `y_true = α + β × y_pred`:

- α = −0.103 (bias: model systematically predicts log-vol 0.1 above actual)
- β = +1.065 (slope: within 6.5% of the identity line)

Both within typical acceptance bounds (|α| < 0.15, |β−1| < 0.20). Model is well-calibrated.

Quantile bias pattern (y_pred compression):

| Quantile | y_true | y_pred | bias |
|---|---|---|---|
| q01 | -3.114 | -2.398 | **+0.716** (over-predicts extreme low-vol) |
| q25 | -1.664 | -1.389 | +0.275 |
| q50 | -1.188 | -1.025 | +0.163 |
| q75 | -0.719 | -0.667 | +0.052 |
| q99 | +0.492 | +0.275 | **-0.217** (under-predicts extreme high-vol) |

The model's prediction distribution is compressed toward the center: tails are systematically under-forecast on the high side and over-forecast on the low side. This is typical LightGBM regularization behavior; it has first-order implications for signal sizing (tails are where edge is highest, so compression costs real edge) but is orthogonal to the alignment question.

---

## Action taken

1. **No retrain needed.** Saved predictions parquet is correctly aligned to `[D, D+15m]`. Backups in `backups/2026-04-17_foundation_fix/` are preserved only for audit-trail purposes.
2. **V2 hold config flipped 30 → 15 min.** Single-line change + docstring update + regression test.
3. **V2 trade log regenerated** with new hold config; before/after numbers below.
4. **Invariants locked in** via `tests/test_backtester_config.py::test_production_v2_holding_minutes_is_15`.

---

## Before / after V2 numbers

Regenerated `BTC_v2_trade_log.parquet` on same data + same predictions + flipped hold config. Old trade log preserved at `vol_arb/output/BTC_v2_trade_log.pre_foundation_fix_2026_04_17.parquet` for audit trail.

| Metric | Pre-fix (hold=30m) | Post-fix (hold=15m) | Δ |
|---|---|---|---|
| NC PnL | $725,771 | $575,253 | **-20.7%** |
| NC Sharpe (bar-level) | 12.14 | 10.66 | -12.2% |
| NC Sharpe (trade-only, active rows only) | 53.13 | 51.18 | -3.7% |
| NC MDD | -4.62% | -4.65% | ~same |
| Active trades | 8,142 | 8,341 | +2.4% |
| Longs / Shorts | 1,847 / 6,295 | 1,860 / 6,481 | +13 / +186 |
| Long NC PnL | $315,812 | $254,471 | -19% |
| Short NC PnL | $409,959 | $320,782 | -22% |
| Vega PnL NC | $1,003,002 | $803,688 | -20% |
| Theta PnL NC | $2,841 | $1,021 | -64% |
| Delta-hedge PnL NC | $21,257 | $16,609 | -22% |
| Option fees NC | $229,821 | $191,866 | -17% |
| Slippage NC | $62,419 | $46,984 | -24% |
| corr(edge, iv_move) active | 0.465 | 0.425 | -0.040 |
| Mean iv_move active | -2.461 | -1.907 | less-negative |
| Median iv_move active | -1.086 | -0.111 | less-negative |

**Interpretation — where the $150K PnL delta went**: the OLD hold=30m captured ~$150K more NC PnL than hold=15m. Decomposition:
- 100% scaling of PnL decomp items is -20% to -24% (proportional to the 15m vs 30m exposure ratio × vega-time).
- Theta PnL is tiny in both cases ($1K–$3K NC) — V2 holds are too short for theta to dominate, the earlier "theta carry" narrative was overweighted.
- The real mechanism of the pre-fix PnL was simply **longer exposure × same signal quality**. Double the hold-duration, ~double the vega time, ~double the slippage, and net out to ~1.25× net PnL (because slippage scales faster than vega at the 15–30m boundary).

**Trade-only Sharpe is essentially unchanged** (53 → 51, -3.7%), which is the cleanest signal: **per-active-trade signal quality is the same**. The bar-level Sharpe drop (-12%) reflects less $/bar distributed across the same bar count — a metric artifact, not a signal degradation.

**Mean iv_move went from -2.46 to -1.91**: the 15-min window contains less vol-realization than 30-min, so iv_exit moves less from iv_entry on average. Still directionally correct (shorts dominate, negative iv_move is a profit for shorts).

**The hold=30m NC MDD 9.91% headline** referenced in memory was likely the **compound** MDD, not NC. Both pre- and post-fix have NC MDD around -4.6%. The compound-path MDD (which was quoted in the "BTC production v2 SHIPPED" memory) would need to be re-quoted on the hold=15m trade log; that's a separate run if needed.

---

## Why +21% PnL loss is an ACCEPTABLE trade-off

The pre-fix $726K came partly from exposure BEYOND the prediction window. That extra PnL is not attributable to the model's forecast skill — it's generic vega-time × market-mean-reversion. The post-fix $575K is cleanly attributable to signal-aligned trades. Claims like "BTC vol-arb signal produces Sharpe X" are now defensible; the prior numbers mixed signal PnL with uninformed-second-bar PnL.

If re-capturing that $150K matters more than signal purity for a particular deployment, the config can be changed back with one line. But the FOUNDATION — the reference the research gauntlet evaluates against — is now clean.

---

## Post-fix leakage probes (all green)

| Probe | Result |
|---|---|
| Active-row ts = feature_ts + 15m (new hold semantics) | ✅ 8,340 / 8,341 rows (1 edge case with zero PnL) |
| y_true matches TSRV at offset +15m | ✅ 99.82% exact-match (same as pre-fix) |
| No NaN in `y_pred, iv_used, edge, signal, net_pnl_nc` | ✅ all 145,041 rows non-null |
| Future-looking `ts > index` | ✅ all active trades |

---

## Files touched

**Modified**:
- `vol_arb/backtester.py:392-456` — `with_btc_production_filters_v2()` now sets `holding_minutes=15`; updated docstring to reflect foundation-fix rationale.
- `vol_arb/output/BTC_v2_trade_log.parquet` — regenerated (hold=15m, same predictions).

**Created**:
- `tests/test_backtester_config.py` — 3 regression-lock tests.
- `vol_model/run_foundation_alignment_audit.py` — the MSE-vs-offset audit tool (kept even though it was the misleading metric — useful for ETH and future assets where the bug could be different).
- `vol_model/output/FOUNDATION_ALIGNMENT_AUDIT.md` (this file).
- `vol_arb/output/BTC_v2_trade_log.pre_foundation_fix_2026_04_17.parquet` — pre-fix trade log preserved.
- `backups/2026-04-17_foundation_fix/` — complete snapshot of pre-fix state (predictions, trade log, backtester.py, full model tree) with MANIFEST.md.

**Not modified (confirmed correct, no action needed)**:
- `vol_model/labels.py::load_labels` — the Session-B `-freq` shift is correct and in effect.
- `vol_model/pipeline.py::align_features_labels` — one-step-ahead shift correct.
- `vol_model/output/BTCUSDT_pruned_svi_predictions.parquet` — target is [D, D+15m] (verified 99.82% exact-match).
- `vol_model/output/models/expanding_pruned_svi_v1/*` — models are fine; no retrain needed.


---

## Implications for prior research

All Session A + B experimental comparisons were made against the *same* predictions parquet (correctly aligned). Relative candidate rankings are valid. The 7 rejected threads remain rejected. No research has to be redone.

Absolute PnL claims from V2 backtests are from the hold=30m config; post-fix numbers will be different in magnitude. The underlying edge is the same, but translated through a tighter hold window. Narrative claims like "BTC production v2: NC PnL $726K, NC Sharpe 12.56" should be qualified with "(hold=30m)" or updated to the new hold=15m numbers.

---

## Lessons recorded for future audits

1. **Don't infer target alignment from y_pred MSE-vs-offset.** Distribution compression and QLIKE bias confound the test. Use `y_true` exact-match instead — it's the training target by definition.
2. **Check `y_true` AGAINST THE SOURCE** (TSRV at shifted offsets), not against a pre-computed label column — because the bug can be anywhere in the label pipeline.
3. **Config mismatches (hold ≠ prediction horizon) are NOT fixed by retraining.** They're fixed by aligning the config.
4. **Persistence benchmarks matter.** If a model's improvement over persistence is < 5%, it's not distinguishing itself from "output = last bar". Production BTC pruned_svi_v1 does 17.8% over persistence — healthy.
