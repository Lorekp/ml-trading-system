# Close-model lookahead incident

**Status:** Historical close-model result **invalidated**; source correction is
present; a mechanism-specific regression test is still desirable.

## Summary

The ML close model reported a post-cost Sharpe of 7.64. At each 15-minute exit
decision, its one-hour regime features could include the still-forming hourly
bar, exposing prices as much as 59 minutes in the future. The result measured
lookahead, not exit-timing skill.

The audit reported Sharpe 0.82 after causal alignment, below the fixed-hold
baseline of 1.51. The 7.64 claim must not be cited.

## Cause and correction

The regime frame was aligned directly to 15-minute decision timestamps. The
causal implementation first shifts to the previous completed hourly bar:

```python
regime_frame = frame.shift(1).reindex(decision_idx, method="ffill")
```

Current source: `src/signal_engine/close_model.py:534-539`.

## Scope

This invalidates the reported ML close-model improvement and any decision based
on its 7.64 Sharpe. It does not by itself invalidate unrelated entry models,
portfolio logic, or fixed-hold baselines.

## Evidence and follow-up

- Raw audit: `experiments/REPORT_leakage_audit.md:5-38,101-110`. This file is
  ignored generated evidence; this postmortem is the canonical conclusion.
- Corrected source: `src/signal_engine/close_model.py:534-539`.
- `tests/test_signal_engine/test_regression.py:48-62` pins broader honest
  full-pipeline metrics after an unfinished-bar correction, but it is not a
  targeted test of this close-model join.

Add a regression that mutates all hourly data after a decision timestamp and
asserts that the features and decision at that timestamp do not change.
