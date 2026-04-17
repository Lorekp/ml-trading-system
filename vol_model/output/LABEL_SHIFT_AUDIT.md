# BTC Vol-Forecasting Label-Shift Audit

**Date:** 2026-04-17
**Scope:** Why does `ctx.y_full` from `vol_model.pipeline.build_dataset(VolModelConfig())`
no longer byte-match the `y_true` column saved in
`vol_model/output/BTCUSDT_pruned_svi_predictions.parquet` (Stage 18 artifact,
written 2026-04-11 at 11:15)?

**Artifacts:**
- Script: `vol_model/run_label_shift_audit.py`
- Per-row deltas: `vol_model/output/label_shift_per_row.csv`
- Full stdout: `vol_model/output/label_shift_audit_stdout.txt`

---

## 1. Observed quantitative deltas

From `python -m vol_model.run_label_shift_audit`:

| Metric | Saved `y_true` | Current `y_full` | delta (c - s) |
|---|---|---|---|
| n (common index) | 145,041 | 182,676 / 145,038 on common | — |
| mean | -1.2023 | -1.2022 (on common) / -1.0636 (on its own index) | +0.0000 |
| median | -1.1879 | -1.1879 | +0.0268 |
| std | 0.7366 | 0.7366 | 0.5245 |
| min | -3.9093 | -3.9093 | -3.7095 |
| max | +1.3894 | +1.3894 | +2.5997 |

- **Exact-matches on common index:** 41 / 145,038 (0.028%) — essentially none.
- **Out-of-envelope rows:** 0 / 145,038 — current stays inside `[saved_min, saved_max]`.
- **Per-decile shift:** monotone fan — lowest saved decile mean -2.55 vs current
  -2.20 (`delta = +0.36`); highest saved decile mean +0.06 vs current -0.26
  (`delta = -0.32`). This is the signature of a **local-time shift** of the
  series: regimes that were "high" at timestamp T become "less high" and vice
  versa because the series now reports the *other* bar at each index.

### Lag hypothesis (KEY RESULT)

Checking whether `current[T]` equals `saved` at a nearby timestamp:

| offset | exact matches / n | close (<1e-4) / n |
|---|---|---|
| `current[T]` vs `saved[T - 15min]` | **145,028 / 145,028** | 145,028 / 145,028 |
| `current[T]` vs `saved[T + 15min]` | 41 / 145,025 | 62 / 145,025 |
| `current[T]` vs `saved[T - 30min]` | 41 / 145,015 | 64 / 145,015 |
| `current[T]` vs `saved[T + 30min]` | 32 / 145,015 | 61 / 145,015 |

**The current label pipeline is the saved labels shifted forward in time by
exactly +15 minutes (one 15m bar).** Expressed the other way: at any output
timestamp `T`, `current y_full[T] == saved y_true[T - 15min]` for 100.0% of
rows where both exist.

### Pass 2 vs Pass 1

Default `VolModelConfig()` and explicit `predict_excess_logvol=False` produce
**byte-identical** `y_full` (182,676 / 182,676 exact match, max |delta| = 0).
=> `predict_excess_logvol` is **not** the cause.

---

## 2. Code-level evidence

### 2.1 File mtimes

```
TSRV/output/BTCUSDT_tsrv_15m.parquet                 2026-04-11 21:53
TSRV/output/BTCUSDT_tsrv_15m.parquet.pre_end_stamp_bak  2026-04-11 21:51
vol_model/output/BTCUSDT_pruned_svi_predictions.parquet 2026-04-11 11:15
vol_model/config.py                                  2026-04-16 17:51
vol_model/labels.py                                  2026-04-16 17:52
vol_model/pipeline.py                                2026-04-08 18:53
```

The TSRV parquet was re-stamped on **2026-04-11 21:53**, ten hours *after*
the Stage 18 predictions file was written. The filename
`.pre_end_stamp_bak` is explicit: this was the "start-stamped → end-stamped"
migration of the TSRV bars.

### 2.2 Backup TSRV vs current TSRV

```
Backup index range:  2021-01-01 00:00:00+00 .. 2026-03-31 23:45:00+00
Current index range: 2021-01-01 00:15:00+00 .. 2026-04-01 00:00:00+00

=> Backup index is exactly (current index) - 15min. Same vol_ann values,
   timestamps shifted +15min.
```

Confirmed empirically: `bak.index + 15min == cur.index` for all 183,065 rows.
Row `i`'s `vol_ann`, `continuous_var`, `jump_var`, … are byte-identical between
the two files — only the timestamp convention changed.

### 2.3 `vol_model/labels.py` — `load_labels`

```python
def load_labels(config: VolModelConfig) -> pd.DataFrame:
    tsrv = pd.read_parquet(config.get_tsrv_path())
    valid = tsrv[tsrv["vol_ann"] > 0].copy()
    ln_vol = np.log(valid["vol_ann"])
    if getattr(config, "predict_excess_logvol", False):
        ...
        ln_vol = ln_vol - log_iv
        ...
    # Clip extremes to prevent QLIKE loss explosion
    lo = ln_vol.quantile(1 - config.label_clip_quantile)
    hi = ln_vol.quantile(config.label_clip_quantile)
    ln_vol = ln_vol.clip(lo, hi)
    ...
    result = pd.DataFrame({
        "vol_ann": valid["vol_ann"],
        "ln_vol": ln_vol,
        "continuous_var": continuous_var,
        "jump_var": jump_var,
        "t1": valid.index + freq,
    }, index=valid.index)
    return result
```

No explicit shift of the TSRV index. The DataFrame is returned with
**`valid.index` as-is** — whatever convention the parquet file uses.

### 2.4 `vol_model/labels.py` — `align_features_labels`

```python
def align_features_labels(features, labels):
    common = features.index.intersection(labels.index)
    features = features.loc[common]
    labels = labels.loc[common]

    # One-step-ahead: features[:-1] predict labels[1:]
    X = features.iloc[:-1].copy()
    y = labels["ln_vol"].iloc[1:].copy()
    t1 = labels["t1"].iloc[1:].copy()

    # Re-align indices so X[i] and y[i] share the same index
    X.index = y.index
    return X, y, t1
```

`iloc[1:]` shifts labels by one position — the "one-bar lag safety belt"
documented in `feedback_tsrv_not_a_leak.md` (2026-04-11). This was designed
for the **start-stamped** TSRV convention: at output index `T`,
`labels.index[i+1] = T_start`, `labels.ln_vol[T_start] = log(vol of [T, T+15))`
— a forward prediction.

After the end-stamp fix, `labels.index[i+1] = T_end`,
`labels.ln_vol[T_end] = log(vol of [T-15, T))` — the label at output index T is
now the vol of **the bar ending at T** (backward-looking) instead of the bar
starting at T (forward-looking). **The `iloc[1:]` shift was not adjusted.**

### 2.5 `vol_model/config.py` — relevant flags

```python
label_clip_quantile: float = 0.999          # unchanged
predict_excess_logvol: bool = False         # default unchanged, and pass 2 verified it's not the cause
```

No flag in `config.py` changes the TSRV column used (`vol_ann` is hard-coded
in `labels.py`). No flag touches timestamp conventions.

### 2.6 `vol_model/pipeline.py::build_dataset`

Trivial wrapper — concatenates features + labels, aligns, drops NaNs. No
label transformation. `pipeline.py` has not been touched since 2026-04-08.

### 2.7 Direct byte-level verification

For output timestamp `T = 2022-02-01 00:00` (first row in saved predictions):

- `saved y_true[T] = -0.0484`
- `current y_full[T] = -0.8568`
- `log(current TSRV.vol_ann[T]) = -0.8568` (vol of trades `[T-15, T)`,
  end-stamped)
- `log(current TSRV.vol_ann[T+15min]) = -0.0484` (vol of trades `[T, T+15)`)
- `log(backup TSRV.vol_ann[T]) = -0.0484` (vol of trades `[T, T+15)`,
  start-stamped)

Saved y_true uses the forward-looking vol (start-stamped interpretation).
Current y_full uses the backward-looking vol (end-stamped interpretation).
Both are computed via the same `iloc[1:]` shift — the shift itself was never
changed, but the *input TSRV timestamps* were.

---

## 3. Candidate-mechanism analysis

### (a) `label_clip_quantile` changed value — REJECTED
- Config value is 0.999, identical to what Stage 18 was trained under.
- Clipping can't produce a `+15min` lag pattern. It would compress outliers,
  not shift the whole series.
- Only 41 exact-zero deltas in Pass 1 (those are rows where both saved and
  current clipped to the same bound or happened to coincide), which
  explains the tiny tail of 41 matches everywhere including `T+15min`.

### (b) `predict_excess_logvol` toggled True — REJECTED
- Default config pass and explicit `predict_excess_logvol=False` pass are
  byte-identical (182,676 / 182,676, max |delta| = 0).
- If excess-logvol were the cause, the shift would be an additive
  IV-dependent term (roughly -log(0.6) ≈ +0.5 level shift), not a
  time-lag pattern.

### (c) TSRV column source changed (e.g., vol_ann → continuous_var) — REJECTED
- `load_labels` still reads `valid["vol_ann"]`.
- Current y on current TSRV matches `log(current_tsrv.vol_ann)` at 182,311 /
  182,676 exact rows — same column, same file.

### (d) Log transform changed — REJECTED
- `ln_vol = np.log(valid["vol_ann"])` is unchanged and still matches
  directly.

### (e) Clipping logic expanded/tightened — REJECTED
- saved_y: min=-3.9093, max=+1.3894
- current_y on common index: min=-3.9093, max=+1.3894
- Identical clip bounds. Distribution quantiles are unchanged (mean, median,
  std on common index all match to 4 decimal places).

### (f) TSRV timestamp convention changed (start-stamped → end-stamped) — **CONFIRMED**
- Backup file `BTCUSDT_tsrv_15m.parquet.pre_end_stamp_bak` proves the
  migration: index shifted +15min, values byte-identical.
- `current[T] == saved[T - 15min]` for 145,028 / 145,028 rows — the exact
  signature of a one-bar upstream-timestamp shift that isn't absorbed
  downstream.
- `align_features_labels` retains `iloc[1:]`, which was the correct
  compensator for start-stamped labels but under-shifts for end-stamped
  labels by exactly one bar.

---

## 4. Primary diagnosis

**The TSRV parquet was migrated from start-stamped to end-stamped on
2026-04-11 21:53 (ten hours after Stage 18 predictions were written).
`vol_model/labels.py::align_features_labels` still performs a single
`iloc[1:]` shift. Under the old convention this produced a 1-bar-ahead
forecast target; under the new convention it silently turns the label
into a backward-looking vol (the 15-minute bar ending AT the feature
observation time, not starting FROM it).**

Concretely, at output timestamp `T`:
- **Stage 18 training labels (pre-migration):** `y[T] = log(vol of [T, T+15))`,
  i.e. the future 15-minute bar. This is what the backtester and the DRL
  hedger consume (`edge = exp(y_pred) - iv_used`).
- **Current labels (post-migration):** `y[T] = log(vol of [T-15, T))`, i.e.
  the 15-minute bar that just closed. This is a *nowcast* of the already-
  observed bar, not a forecast.

This is not a leak (features at output position `i` have original index
`T - 15min`; the label bar `[T-15, T)` ends exactly at that feature
observation time, no future info bleeds in). But it is a **target
definition change** — the model is now being asked to "predict" the bar
it can already measure from the trades in its own feature window, rather
than the next 15-minute bar.

**Confidence: HIGH.** All evidence converges — file mtimes, `.pre_end_stamp_bak`
backup, byte-identical `vol_ann` across the two TSRV files, 100%
`current[T] == saved[T - 15min]` match, monotone per-decile fan with
zero envelope violation, and the two TSRV files differing *only* by
+15min index shift.

---

## 5. Recommendation: **REVERT-BEFORE-COMPARISON**

Any new model trained on the current `build_dataset` output is predicting a
**different target** than Stage 18 was trained on. Comparing the new model's
QLIKE / RankIC / Sharpe / edge-PnL against Stage 18 baselines is
apples-to-oranges — a few percentage points of "improvement" could just be
the easier task of explaining the just-closed bar (features are trivially
correlated with the vol that produced them).

Before re-running any signal-research gauntlet (HAR-residual, quantile LGBM,
PatchTST, any cross-asset work), the label definition must be restored to
its Stage 18 semantics:

```python
# In vol_model/labels.py::load_labels, add a one-bar shift for end-stamped TSRV:
# End-stamped: labels.index[i] represents vol of [T_i-freq, T_i).
# To get the FORWARD 15min bar at label key T (Stage-18 semantics), shift the
# TSRV index backward by freq so valid.index[i] = T means "vol of [T, T+freq)".
valid.index = valid.index - pd.Timedelta(seconds=config.freq_s)
```

Alternative (equivalent): change `align_features_labels` to `iloc[2:]` /
`iloc[:-2]` so the shift compensates for BOTH the end-stamp convention and
the one-step-ahead semantics. The `labels.index - freq` approach is cleaner
because it preserves the "label.index = prediction timestamp" semantic.

Rationale for REVERT, not ACCEPT:
1. **Incumbent wins on ties.** Stage 18 is the production baseline that all
   live engine, backtests, DRL hedger, and downstream experiments were
   calibrated against. Silently changing the target breaks reproducibility.
2. **Comparison validity.** Signal-research memo
   `project_signal_research_2026_04_16.md` already rejected three
   architectures against a "fresh pruned_svi control" that was presumably
   retrained on this post-migration target. Those comparisons are now
   questionable and should be re-validated once labels are back to Stage 18
   semantics.
3. **Forecast vs nowcast is a real task change.** Features computed over
   `[T-15, T)` to predict vol `[T-15, T)` is nearly trivial (the RV is a
   function of the very trades the features summarize). The model could
   appear to "improve" by dozens of percent in RMSE while having zero
   edge-PnL, because the new target is almost directly observable.
4. **No measured benefit of the new target yet.** No downstream experiment
   has shown the end-stamped-label pipeline produces *better* forecasts of
   *the original forward target*. Until that's demonstrated, keep Stage 18.

---

## 6. Action items for the next session

1. **Patch `vol_model/labels.py::load_labels`**: add `valid.index =
   valid.index - freq` after loading the end-stamped TSRV so `labels.ln_vol[T]
   = log(vol of [T, T+freq))` as before. Add a regression test that
   `load_labels(VolModelConfig())` produces a series where `ln_vol[T]`
   equals `log(backup.vol_ann[T])` on the overlap with the backup (or
   equivalently `log(current.vol_ann[T+freq])`).
2. **Re-run `extract_pruned_svi_predictions`** on the fixed labels to
   confirm `y_true` byte-matches `vol_model/output/BTCUSDT_pruned_svi_predictions.parquet`
   (up to 41 tie rows near the clip bound).
3. **Invalidate the "fresh pruned_svi control"** used in the 2026-04-16
   signal-research memo. Retrain that control under the corrected labels
   and re-score the three rejected architectures. Expected result:
   HAR-residual, quantile LGBM, PatchTST all stay rejected (their
   rejections were large-margin, ~5-26%), but the deltas may shift by a
   few percentage points.
4. **Add an end-to-end probe** in `leakage_probes.py` that explicitly
   checks `load_labels(cfg).ln_vol[T]` corresponds to `log(vol of
   [T, T+freq))`, not `log(vol of [T-freq, T))`, by back-solving from
   raw aggTrades on a handful of sample timestamps. This complements the
   existing `probe_parquet_end_stamped` (which enforces the PARQUET
   convention but not the label SEMANTIC).
5. **Update `feedback_tsrv_not_a_leak.md`** with a new paragraph: "When
   the TSRV parquet convention was flipped on 2026-04-11, the `iloc[1:]`
   in `align_features_labels` became a 0-bar-ahead nowcast instead of a
   1-bar-ahead forecast. Verified 2026-04-17. The one-bar safety belt is
   convention-dependent — whoever touches TSRV timestamp convention also
   owns `labels.py`." File is otherwise accurate.
6. **Check ETH vol-arb labels** — same code path, likely same issue once
   ETH TSRV is also end-stamped (or still open-stamped; verify). Relevant
   to the ETH negative-PnL investigation in `project_eth_vol_arb_status.md`.
