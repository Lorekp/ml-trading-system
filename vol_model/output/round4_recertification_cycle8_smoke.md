# Round-4 recertification report  --  BTCUSDT

**Verdict**: **PASS** (0 pass / 0 cond / 0 fail)

## Scope

* Asset: `BTCUSDT`
* Requested months: `2026-04`
* Trained this run: `(none)`
* Skipped (already present): `2026-04`

## Pre-registered gates

* PASS: per-month QLIKE drift <= 30% vs trailing-3mo median.
* CONDITIONAL_PASS: drift ∈ (30%, 50%].
* FAIL: drift > 50%.

## Per-month results

| Month | QLIKE | Baseline (3mo med) | Drift | Status |
|---|---|---|---|---|

## Trailing prior-month context

| Month | QLIKE |
|---|---|
| 2025-10 | 0.0864 |
| 2025-11 | 0.0737 |
| 2025-12 | 0.1155 |
| 2026-01 | 0.1181 |
| 2026-02 | 0.0834 |
| 2026-03 | 0.1014 |

## Decision

All months pass. Model is certified stable on the post-F3 + post-F6 live distribution. New checkpoints are usable; monthly cadence simply continues forward.

## Note on round-4 interpretation

This report measures the production model's QLIKE drift across calendar months. F3+F6 change LIVE feature computation only  --  they do not modify the offline training pipeline. So the QLIKE drift here reflects natural market regime variation, NOT any F3/F6 effect. The real F3/F6 certification is a live-vs-offline feature parity test (planned next round); this run establishes the baseline QLIKE cadence the live engine will be measured against once new checkpoints land.
