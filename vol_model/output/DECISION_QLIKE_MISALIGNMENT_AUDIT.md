# Decision-QLIKE misalignment audit — 2026-04-20

## Question

HAR-PD is trained on **QLIKE** (symmetric log-vol forecasting loss).
The downstream strategy makes a **threshold decision**:
`trade = sign(y_pred_vp - IV) if |y_pred_vp - IV| > τ`.

Are these two objectives aligned? If QLIKE training is already
decision-optimal at the current model capacity, a decision-focused
loss adds little. If it's NOT aligned, there's room for gain.

## Methodology

Purely **offline**, no backtester runs, zero retraining. Uses:
- Production walk-forward predictions
  (`BTCUSDT_pruned_svi_har_pd_predictions.parquet`, 145k bars).
- Existing baseline V2 trade log (P&L is for evaluation only, never
  a feature or label).

Per-bar diagnostics, restricted to bars where the strategy actually
fired (8,112 trades):
- `QLIKE_i = exp(2·(y_true - y_pred)) - 2·(y_true - y_pred) - 1`.
- `|pred_edge|_i = |exp(y_pred)·100 - IV_i|`, `|real_edge|_i` similarly.
- `|PnL|_i`, `sign(PnL)_i` from the trade log.

## Findings

### 1. QLIKE is economically uninformative (primary finding)

**Spearman(QLIKE, |trade PnL|) = -0.038** (p=7e-4). Statistically
nonzero, but economically zero — QLIKE rank explains ~0.1% of the
variance in `|PnL|` rank.

Per-month check (46 months):
- Mean: **-0.014**
- Only **1/46 months** has `|ρ| > 0.2`
- 23/46 months have **negative** correlation

This is stable across the full 4-year backtest — not a single-regime
artifact.

### 2. Winner vs loser QLIKE: statistically indistinguishable

| | n | QLIKE mean | QLIKE median | QLIKE p75 |
|---|---|---|---|---|
| Winners | 4,064 | 0.462 | 0.158 | 0.453 |
| Losers | 4,048 | 0.473 | 0.172 | 0.469 |

- Mann-Whitney U: **p = 0.10** (not significant at α=0.05)
- Cliff's delta: **-0.021** (negligible effect size)

QLIKE is not a usable feature for discriminating winning trades from
losing ones. The ROC-AUC of `-QLIKE` as a win-vs-loss classifier is
**0.511** (random = 0.500).

### 3. Predicted edge DOES carry economic signal, but weakly

| Metric | Value |
|---|---|
| Spearman(`|pred_edge|`, `|real_edge|`) | **+0.44** (0.00 < p) |
| Spearman(`|pred_edge|`, `|PnL|`) | +0.08 (mean across months) |
| ROC-AUC(`|pred_edge|` → W/L) | **0.547** |
| Direction accuracy (sign match) | 93.1% |

The prediction HAS signal — it correlates with what actually happens.
But its **economic rank-correlation is weak** (+0.08 mean across months).

### 4. Ranking-gap quantification — 3× PnL-capture upside

Ranking bars by predicted edge magnitude, what fraction of total PnL
is captured by the top-K%?

| Top-K | By `|pred_edge|` | By `|PnL|` (oracle) | Ratio |
|---|---|---|---|
| 10% | $202,929 | $585,316 | 0.35 |
| 25% | $360,045 | $811,894 | 0.44 |
| 50% | $525,919 | $870,866 | 0.60 |
| 75% | $652,259 | $829,956 | 0.79 |

The top-10% pred-edge ranking captures only **35%** of what a perfect-
PnL oracle would capture. Decision-focused training operates in
**exactly this gap**: improving the model's ability to identify
high-|PnL|-magnitude trades is the upside.

### 5. Alignment-gap aggregate metric

Mean `|rank(QLIKE) - rank(|PnL|)|` across 8,112 trades = **2,638**.
Normalized to [0, 1] where 1 = random independent ranks: **0.976**.
Equivalently: QLIKE rank is statistically **indistinguishable from
random** w.r.t. PnL rank.

## Verdict

**Proceed to Phase 2 (decision-focused retraining).** The empirical
case for training misalignment is overwhelming:
- Essentially zero correlation between training loss rank and trade
  value rank
- Winners and losers have indistinguishable QLIKE distributions
- 3× ranking upside if the gap can be closed

## Caveats (intellectual honesty)

1. **Not all of the 3× is achievable.** The oracle benchmark includes
   trades whose magnitude was driven by delta-hedge path variance and
   unforecastable microstructure — not forecastable signal. A realistic
   upside is materially less than the oracle ceiling.
2. **Directional accuracy is already high (93%)**. The sign of
   `y_pred - IV` almost always matches reality (most trades fire with
   large edges). The remaining gap is in **magnitude-ranking**, not
   direction. A decision-focused loss should prioritize better magnitude
   calibration, not direction flipping.
3. **QLIKE ≈ forecasting accuracy**, not decision value, is consistent
   with what the literature on decision-focused learning predicts
   (Elmachtoub & Grigas 2022; Mandi et al. 2020). Our empirical finding
   is the expected phenomenon at a scale large enough to matter.

## Phase 2 plan (next)

Three candidate decision-focused losses, ranked by tree-compatibility
and expected economic gain:

1. **Edge-magnitude-weighted regression**: LGBM with custom objective
   = MSE with sample weights `w_i = max(|y_true - IV|, threshold)`.
   Simplest possible decision-focused loss; upweights economically
   important bars without changing the fundamental geometry.
2. **IV-centered hinge loss**: minimize
   `max(0, margin - (y_pred - IV) · sign(y_true - IV))^2`. Directly
   penalizes directional errors with margin — the true decision-
   focused objective for threshold-trading.
3. **SPO+ / decision-focused** (Elmachtoub & Grigas 2022): convex
   surrogate for decision regret. Most principled but most complex;
   usually applied to neural nets, not LGBM. Would require gradient-
   based adaptation.

Evaluation (all offline):
- Primary: ROC-AUC(new `|pred_edge|` → W/L) on held-out months
  (baseline 0.547; target ≥ 0.58)
- Secondary: per-month Spearman(`|pred_edge|`, `|real_edge|`) —
  should rise above current ~0.44
- Tertiary: QLIKE (expect marginal degradation — acceptable trade-off)
- Trade-ranking PnL capture at top-K% (target: 0.35 → 0.45+)

Leakage controls (strict):
- Walk-forward retraining identical to production HAR-PD
- IV is a feature at training time (it already is, via `svi_atm_1d`);
  no new leakage surface introduced
- Trade-P&L NEVER enters as feature or label — evaluation only
- Train/val temporal splits preserved per-fold

## Files produced

- `vol_model/output/DECISION_QLIKE_MISALIGNMENT_AUDIT.md` — this file.

## References

- Elmachtoub, A.N. & Grigas, P. (2022). "Smart Predict, then Optimize."
  _Management Science_ 68(1). The SPO+ convex surrogate loss.
- Mandi, J. et al. (2020). "Smart Predict-and-Optimize for Hard
  Combinatorial Optimization Problems." _NeurIPS_. Extension to
  non-linear downstream.
- Wilder, B., Dilkina, B. & Tambe, M. (2019). "Melding the Data-Decision
  Pipeline: Decision-Focused Learning for Combinatorial Optimization."
  _AAAI_.
