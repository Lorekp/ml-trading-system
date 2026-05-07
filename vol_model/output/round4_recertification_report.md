# Round-4 recertification report — BTCUSDT

**Verdict**: **CONDITIONAL_PASS** (4 pass / 2 cond / 0 fail)

## Scope

* Asset: `BTCUSDT`
* Requested months: `2025-11, 2025-12, 2026-01, 2026-02, 2026-03, 2026-04`
* Trained this run: `(none)`
* Skipped (already present): `2025-11, 2025-12, 2026-01, 2026-02, 2026-03, 2026-04`

## Pre-registered gates

* PASS: per-month QLIKE drift ≤ 30% vs trailing-3mo median.
* CONDITIONAL_PASS: drift ∈ (30%, 50%].
* FAIL: drift > 50%.

## Per-month results

| Month | QLIKE | Baseline (3mo med) | Drift | Status |
|---|---|---|---|---|
| 2025-11 | 0.0737 | 0.0898 | -17.95% | PASS |
| 2025-12 | 0.1155 | 0.0864 | +33.66% | CONDITIONAL |
| 2026-01 | 0.1181 | 0.0864 | +36.70% | CONDITIONAL |
| 2026-02 | 0.0834 | 0.1155 | -27.76% | PASS |
| 2026-03 | 0.1014 | 0.1155 | -12.20% | PASS |
| 2026-04 | 0.1050 | 0.1014 | +3.60% | PASS |

## Trailing prior-month context

| Month | QLIKE |
|---|---|
| 2025-05 | 0.0796 |
| 2025-06 | 0.0925 |
| 2025-07 | 0.0885 |
| 2025-08 | 0.0898 |
| 2025-09 | 0.1004 |
| 2025-10 | 0.0864 |

## Decision

Some months are marginal (drift in (30%, 50%]). Trend likely still stable; ship with operator note. Inspect SHAP stability for the marginal months before flipping any default.

## Note on round-4 interpretation

This report measures the production model's QLIKE drift across calendar months. F3+F6 change LIVE feature computation only — they do not modify the offline training pipeline. So the QLIKE drift here reflects natural market regime variation, NOT any F3/F6 effect. The real F3/F6 certification is a live-vs-offline feature parity test (planned next round); this run establishes the baseline QLIKE cadence the live engine will be measured against once new checkpoints land.
