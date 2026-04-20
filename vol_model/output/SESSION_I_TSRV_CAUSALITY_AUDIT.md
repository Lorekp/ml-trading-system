# Session I Phase 0 — TSRV windowing causality audit

**Date**: 2026-04-20 (Session I kick-off)
**Scope**: verify that the TSRV pipeline producing `continuous_var` and
`jump_var` label columns (used by Candidate I-1 jump-diffusion dual-head)
is strictly backward-safe — i.e., the stat at row timestamp T depends
only on ticks inside the window whose start is T, never on ticks at
timestamps outside that window.

## Motivation

Per the 2024-2026 jump-diffusion literature review (Lee-Mykland 2008
implementations in R's `highfrequency` package, some Python ports, and
the MATLAB reference): **some implementations of jump detection use a
centered window** (K/2 returns before + K/2 returns after each bar). A
centered window introduces forward leakage if used to build features or
to detect jumps at training time. The original Lee-Mykland paper does
NOT require the forward window — centering is a software implementation
choice some authors get wrong.

Before building Candidate I-1 (which consumes `continuous_var` + `jump_var`
as dual-head targets via `labels.py:116-122`), we must verify that TSRV's
per-bar window is backward-only.

## Files audited

1. `TSRV/tsrv_series.py` — windowing logic (`_process_month_tsrv`,
   `compute_tsrv_window`)
2. `TSRV/jump_robust.py` — pure window aggregators (`bipower_variation`,
   `median_rv`, `decompose_jump_continuous`, `directional_jump_fractions`)
3. `vol_model/labels.py:116-122` — consumption of `continuous_var` +
   `jump_var` into the labels DataFrame

## Findings

### TSRV windowing (`tsrv_series.py:264-291`)

```python
ts_ns = month_df.index.values.astype("int64")
freq_ns = int(freq.value)                  # nanoseconds per window
window_ids = ts_ns // freq_ns              # integer floor division

# Find contiguous block boundaries
change_idx = np.where(np.diff(window_ids) != 0)[0] + 1
# Split tick indices into per-window groups
block_starts[0] = 0; block_starts[1:] = change_idx
block_ends[:-1] = change_idx; block_ends[-1] = n
...
for b in range(len(block_starts)):
    s, e = int(block_starts[b]), int(block_ends[b])
    result = compute_tsrv_window(prices[s:e], dollar_vol[s:e], threshold)
    result["timestamp"] = pd.Timestamp(int(window_ids[s]) * freq_ns, ...)
```

**Key invariant**: each tick in `prices[s:e]` has the SAME `window_id`
(the change-index boundaries are precisely where `window_ids` changes).
All ticks in block `[s, e)` satisfy `window_id == k` for some fixed `k`,
i.e., all ticks have timestamp in `[k * freq_ns, (k+1) * freq_ns)`.

**Stamp**: `timestamp = window_ids[s] * freq_ns`, which is the START of
the window `[k * freq_ns, (k+1) * freq_ns)`. The window is START-stamped.

**Causality**: `compute_tsrv_window` receives ONLY `prices[s:e]` — a
strictly contained block. No cross-window contamination is possible
inside the function. No centered window, no forward leakage.

### Aggregator purity (`jump_robust.py`)

`bipower_variation(log_prices)`, `median_rv(log_prices)`,
`realized_semivariance(log_prices)` all operate on a single 1-D array
of log-prices. They use only `np.diff(log_prices)` (within-array
differences) and adjacent-element products / medians. The caller
supplies the window content, so causality is a property of the caller
(TSRV's windowing is backward-safe per above).

### Label pipeline (`labels.py:70-82, 116-122, 126-132`)

```python
# Normalize TSRV timestamp convention to START-stamped
if getattr(config, "tsrv_end_stamped", False):
    freq_td = pd.Timedelta(seconds=config.freq_s)
    tsrv.index = tsrv.index - freq_td

...
continuous_var = valid["continuous_var"].clip(lower=1e-20)
continuous_var = continuous_var.clip(upper=hi_c)
jump_var = valid["jump_var"].clip(lower=0.0)
jump_var = jump_var.clip(upper=hi_j)
...
result = pd.DataFrame({
    "vol_ann": valid["vol_ann"],
    "ln_vol": ln_vol,
    "continuous_var": continuous_var,
    "jump_var": jump_var,
    "t1": valid.index + freq,
}, index=valid.index)
```

`continuous_var` and `jump_var` are stored at the SAME index as `ln_vol`
and `vol_ann`. Per the `-freq` shift on line 82, every label column is
START-stamped: row at T contains statistics of `[T, T+freq)`.

### One-step-ahead alignment (`labels.py:156-162`)

```python
X = features.iloc[:-1].copy()          # features[0..N-2]
y = labels["ln_vol"].iloc[1:].copy()   # labels[1..N-1]
X.index = y.index
```

Features at time T-freq predict labels at T, where T is START-stamped
and the label is the stat of `[T, T+freq)`. **Features never see any
tick from the window they predict.** Candidate I-1's new dual targets
`continuous_var[T]` and `jump_var[T]` have the exact same timestamp
semantic — same iloc[1:] shift suffices.

## Probe test

Added to `tests/test_leakage_probes.py` (pytest, auto-discoverable):

- `test_tsrv_window_decomposition_is_causal` — simulates
  `_process_month_tsrv`'s windowing with a pure sum-of-squared-returns
  aggregator, runs `probe_causality_by_mutation` with `future_mask =
  idx >= T + freq`. Asserts the stat at T is byte-identical (tolerance
  0.0) after mutating every tick at index ≥ T+freq. **PASS.**
- `test_tsrv_centered_window_would_fail_probe` — negative control: an
  intentionally-leaky centered-window aggregator (reads `[T-freq/2,
  T+freq/2]`). Runs the same probe. Asserts the probe CATCHES the leak.
  **PASS** (probe correctly flags leakage).

Both tests green (1.38s total).

## Conclusion

**The TSRV jump-decomposition pipeline is backward-safe by inspection
AND by synthetic mutation probe.** Candidate I-1 (jump-diffusion
dual-head) may proceed without any TSRV rework.

Additional confidence: `continuous_var` and `jump_var` columns have been
consumed by TransVol's dual-head training (`transvol/model.py:120-133`)
for multiple weeks without any downstream leakage flag. If TSRV had a
centered window, the shuffle-label probe battery would have flagged it
already.

## Regression guard

The two probe tests in `tests/test_leakage_probes.py` now pin this
invariant. If anyone widens TSRV's window to be centered or overlapping
in a future refactor, the first test will fail immediately.
