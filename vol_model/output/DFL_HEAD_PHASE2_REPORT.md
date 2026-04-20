# Decision-Focused Learning head — Phase 2 feasibility report

## Executive summary

A **DFF-style** (Wen et al. AAAI 2025) post-hoc classifier head on top
of the frozen HAR-PD backbone captures **2.10× more P&L at top-10%**
than the existing `|pred_edge|` ranking, evaluated walk-forward OOS.
The improvement is statistically decisive (bootstrap 95% CI [1.71,
2.62], permutation p < 0.001) but exhibits moderate month-to-month
variance. Evaluation is **fully offline**; no backtester runs.

## Methodology

### DFF (Decision-Focused Fine-tuning)

Per Wen et al. (AAAI 2025, arxiv 2501.01874): **freeze the backbone
predictor, fine-tune a small output-correction head with decision
loss**. GBDT-compatible; doesn't require backprop through the
predictor. Chosen over SPO+ custom objective because:
1. SPO+ has Hessian degeneracy in tree boosting (briefing, §B)
2. DFF is the only DFL method with explicit GBDT (XGBoost) results
3. Keeping HAR-PD unchanged preserves all certified production
   behavior; head is strictly additive

### Architecture

- **Backbone**: production `pruned_svi_har_pd_v1` walk-forward
  predictions (frozen, no retrain).
- **Head input features** (10):
  - `y_pred`, `iv_used`, `iv_log = log(IV/100)`
  - `pred_edge_log = y_pred - iv_log`, `|pred_edge_log|`
  - `past_vol_24h`, `past_vol_6h`, `past_vol_std` (from `y_true`,
    strictly lagged via `shift(1).rolling(...)`)
  - `recent_resid_24h`, `recent_abs_resid_24h` (from residuals,
    strictly lagged)
- **Head target**: `win = (net_pnl_nc > 0)` binary classification
  on trade-fired bars only.
- **Head model**: `LGBMClassifier(num_leaves=15, max_depth=5,
  learning_rate=0.05, n_estimators=300, min_child_samples=50,
  reg_lambda=1.0, seed=42)`. Early stopping on eval AUC.

### Walk-forward evaluation

4 fiscal-year folds with **expanding training**:
- Test 2023: train on 2022 → 1,522 test trades
- Test 2024: train on 2022–2023 → 2,025 test trades
- Test 2025: train on 2022–2024 → 1,070 test trades
- Test 2026: train on 2022–2025 → 380 test trades
- **Concatenated OOS: 4,997 trades, $658,516 P&L**

### Leakage controls

- **Strict temporal splits**: test-year trades never leak back into
  training via features or target.
- **Features use only past data**: `shift(1).rolling(...)` pattern
  guarantees no same-bar leakage.
- **Backbone predictions are already walk-forward OOS**
  (`y_pred_prod` was the OOF prediction from the original HAR-PD
  walk-forward, so features include only information available at
  decision time).
- **`y_true` appears in features** only in strictly-lagged rolling
  aggregates — not as itself.
- **`net_pnl_nc` is the eval target only**, never a feature.

## Results

### Headline metrics (walk-forward OOS, seed=42)

| Metric | Value |
|---|---|
| Mean AUC delta (clf vs `|pred_edge|`) | **+0.037** |
| Years with AUC improvement | **3/4** |
| Top-10% P&L by classifier | **$168,853** |
| Top-10% P&L by `|pred_edge|` | $80,574 |
| **P&L capture ratio (top-10%)** | **2.10×** |
| Top-25% P&L ratio | 1.77× |
| Top-50% P&L ratio | 1.36× |

### Robustness

- **Bootstrap 95% CI on top-10% ratio: [1.71, 2.62]**; `P(ratio > 1.5) = 0.998`
- **Permutation p-value**: < 0.0005 (shuffled scores never match real ranking)
- **Seed-stable**: identical results across 5 LightGBM seeds
- **Per-month**: 20/38 months improved, 18/38 worse. PnL-weighted
  aggregate improvement dominated by high-volatility months where
  the classifier excels (2024-08: 6.98×; 2024-11: 1.63×; 2025-02:
  2.96×). The classifier is **especially valuable in regime-stress
  months** — the same months that drove V2 drawdowns.

### Filter-style deployment preview

Drop bottom-K% by `P(win)`, report retained P&L (OOS, walk-forward):

| Drop K% | Retained P&L | Ratio | N dropped |
|---|---|---|---|
| 10% | $629,789 | 95.6% | 499 |
| 25% | $576,508 | 87.5% | 1,249 |
| 50% | $483,562 | 73.4% | 2,498 |

Dropping the lowest-confidence 25% retains **87.5% of P&L** while
eliminating **25% of trades** — a favorable cost/P&L ratio if
execution fees are proportional to trade count (which they are).

## Feature importance (from main fit)

| Feature | Importance |
|---|---|
| `iv_used` | 49 |
| `recent_resid_24h` | 40 |
| `recent_abs_resid_24h` | 33 |
| `pred_edge_log` | 31 |
| `past_vol_std` | 28 |
| `past_vol_6h` | 27 |
| `abs_pred_edge_log` | 24 |
| `y_pred` | 23 |
| `past_vol_24h` | 13 |
| `iv_log` | 11 |

IV level, recent model bias (residuals), and past volatility
dominate. These are **decision-centric regime features** that the
point HAR-PD model does not explicitly use for the decision. This is
the gap decision-focused learning exploits.

## Interpretation

1. **The misalignment is real** (Phase 1 audit: ρ(QLIKE, |PnL|) =
   -0.014 across 46 months) and **a decision-focused head can
   measurably close it** (2.1× PnL capture at top-10%).
2. The head uses **regime-level features the point model doesn't
   consult**: recent residuals (how wrong the model has been lately),
   past realized vol profile, IV level itself. This matches the
   intuition from the V2 drawdown audit — losses cluster in regimes
   that the point model doesn't flag.
3. **Not a silver bullet**: month-by-month variance is high; the
   classifier is valuable ON AVERAGE and especially in stress months,
   but not every month.
4. **Deployment surface**: easiest as a SIZING/FILTER signal — not
   a replacement for the point model. The point model still decides
   direction; the DFL head decides confidence/size.

## Pre-registered next-step decisions

Before Phase 3 (actual backtester integration), user decides which of
these paths to take:

**Path A (surgical)** — ship the DFL head as a **sidecar filter**
only. Drop bottom-25% P(win) bars at entry. Expected: -12.5% PnL,
-25% trade count, likely net-positive Sharpe from reduced slippage.
No point-model change. ~2 hours to integrate + test.

**Path B (sizing)** — use `P(win)` as a position-size multiplier.
More aggressive — higher upside and higher variance. ~4 hours.

**Path C (stop here)** — document finding, no production change.
Evidence is documented for future cycles. DFL head is a tested
intervention ready to ship when needed.

**Path D (full decision-focused retrain)** — train a full HAR-PD
variant with quantile/hinge objective directly on decision outcomes.
Days of compute. Not recommended as first move — the DFF head already
achieves most of the low-hanging fruit per Wen et al. (paper shows
DFF narrowly beats NN-SPO+).

## References

- Wen et al. (AAAI 2025). _Decision-Focused Fine-Tuning for Smarter
  Predict-then-Optimize with Limited Data_. arXiv 2501.01874.
  **The paper whose methodology we implemented.**
- Elmachtoub & Grigas (2022). _Smart Predict, then Optimize_.
  Management Science 68(1). SPO+ formulation; proves 1D-binary case
  reduces to hinge loss.
- Mandi et al. (JAIR 2024). _Decision-Focused Learning: Foundations,
  State of the Art, Benchmark and Future Opportunities_.
  arXiv 2307.13565. Normalized regret as canonical DFL metric.
- Vanderschueren et al. (2022). _Predict-then-optimize or predict-
  and-optimize? An empirical evaluation_. Information Sciences.
  The paper that documents DFL's underperformance in small-data /
  high-noise regimes — justifies the conservative DFF choice.

## Artifacts produced

- `scripts/phase2_dfl_feasibility.py` — single-split feasibility test
- `scripts/phase2_dfl_economic.py` — walk-forward economic value
- `scripts/phase2_dfl_robustness.py` — bootstrap / permutation /
  seed-sweep / per-month diagnostics
- `vol_model/output/DFL_HEAD_PHASE2_REPORT.md` — this document
- `vol_model/output/DECISION_QLIKE_MISALIGNMENT_AUDIT.md` — Phase 1

## Caveats (intellectual honesty)

1. **2.1× is top-K-ratio**, not total P&L improvement. Full trade
   deployment gain is smaller — filter-style 87.5% retained.
2. **Per-month stability is moderate.** The 2023 test fold had
   AUC worse than baseline (classifier had only 2022 training data).
   Deployment requires ≥12–18 months of history to train head.
3. **Offline evaluation only.** We measure P&L-ranking quality on the
   trade log P&L — not the P&L that would have resulted from actually
   using the classifier to size/filter trades. Second-order effects
   (slippage, execution) require a real backtester run to measure.
4. **Not a point-model improvement.** The HAR-PD predictions
   themselves are unchanged. This is orthogonal to signal-research
   cycles I-1 through I-3 (which tried to improve the point model).
