# Hurst-100 np.interp Leak (2026-05-08, cycle-8 SHAP audit)

## What

`vol_model/features/engineer.py:_hurst_subsampled` (lines 143-190) computes
the rolling Hurst exponent at every `step=50` dollar bars (anchor positions),
then linearly interpolates between anchors with `np.interp` (line 187).

`np.interp(x, xp, fp)` is **two-sided** — for any `x` between two anchor
positions `vp_left < x < vp_right`, the result is a convex combination of
`hurst_vals[vp_left]` and `hurst_vals[vp_right]`. The right-side anchor was
fitted on `returns[vp_right - 100 : vp_right]`, which contains bars at
positions `> x` (i.e., FUTURE relative to bar `x`).

## Confirmed leak horizon

Up to `step = 50` dollar bars per anchor. Typical dollar bar at $500K
threshold is ~3 seconds in active periods, so leak horizon is ~150 seconds
≈ 2.5 minutes — non-trivial relative to the 15-minute label window.

After the 15-min `groupby(window_labels).last()` aggregation in
`engineer.py`, the row at floor-stamp `T=12:00` inherits `hurst_100` from
the last bar in `[12:00, 12:15)`. If that bar's anchor pair extends past
12:15, the feature contains label-window data → LEAK.

## Suggested fix

Replace `np.interp` with backward step-fill at line 187:

```python
result = np.full(n, np.nan)
for k in range(len(vp)):
    end = vp[k + 1] if k + 1 < len(vp) else n
    result[vp[k]:end] = vv[k]   # step function: anchor value held forward
```

This is causal: bar `i` uses only the anchor at `vp[k] <= i`, never the
next anchor at `vp[k+1] > i`.

## Production impact assessment (2026-05-08 cycle-8 SHAP audit)

Per `shap_audit_6_regimes_decision.csv`, all 5 hurst_100* features rank
**below median position 80 across 6 regime-diverse months** with mean SHAP
< 0.002:

| feature           | mean SHAP | best rank | decision |
|-------------------|----------:|----------:|----------|
| `hurst_100`       | 0.00162   | 33        | REVIEW   |
| `hurst_100_ema50` | 0.00120   | 44        | REVIEW   |
| `hurst_100_lag1`  | 0.00046   | 70        | DROP     |
| `hurst_100_lag3`  | 0.00036   | 76        | DROP     |
| `hurst_100_ema10` | 0.00034   | 78        | DROP     |

**None of the 5 hurst features land in the SHAP-audit KEEP set.** The
post-cycle-8 production prune list excludes all hurst_100* features by
construction, so the leak does not affect the trained production model.

## Disposition

- **Cycle 8 retrain**: NOT BLOCKED. Hurst features are excluded from the
  empirically-justified prune list anyway.
- **Follow-up**: file the np.interp → step-fill fix for cycle 9 cleanup.
  Add a `probe_causality_by_mutation` test against `_hurst_subsampled`
  (the test gap was identified by the audit agent — `tests/` does not
  apply Probe E to this function).
