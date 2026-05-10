# Round-4 recertification report  --  BTCUSDT

**Verdict**: **FAIL** (43 pass / 5 cond / 1 fail)

## Scope

* Asset: `BTCUSDT`
* Requested months: `2022-02, 2022-03, 2022-04, 2022-05, 2022-06, 2022-07, 2022-08, 2022-09, 2022-10, 2022-11, 2022-12, 2023-01, 2023-02, 2023-03, 2023-04, 2023-05, 2023-06, 2023-07, 2023-08, 2023-09, 2023-10, 2023-11, 2023-12, 2024-01, 2024-02, 2024-03, 2024-04, 2024-05, 2024-06, 2024-07, 2024-08, 2024-09, 2024-10, 2024-11, 2024-12, 2025-01, 2025-02, 2025-03, 2025-04, 2025-05, 2025-06, 2025-07, 2025-08, 2025-09, 2025-10, 2025-11, 2025-12, 2026-01, 2026-02, 2026-03, 2026-04`
* Trained this run: `2022-02, 2022-03, 2022-04, 2022-05, 2022-06, 2022-07, 2022-08, 2022-09, 2022-10, 2022-11, 2022-12, 2023-01, 2023-02, 2023-03, 2023-04, 2023-05, 2023-06, 2023-07, 2023-08, 2023-09, 2023-10, 2023-11, 2023-12, 2024-01, 2024-02, 2024-03, 2024-04, 2024-05, 2024-06, 2024-07, 2024-08, 2024-09, 2024-10, 2024-11, 2024-12, 2025-01, 2025-02, 2025-03, 2025-04, 2025-05, 2025-06, 2025-07, 2025-08, 2025-09, 2025-10, 2025-11, 2025-12, 2026-01, 2026-02, 2026-03, 2026-04`
* Skipped (already present): `(none)`

## Pre-registered gates

* PASS: per-month QLIKE drift <= 30% vs trailing-3mo median.
* CONDITIONAL_PASS: drift ∈ (30%, 50%].
* FAIL: drift > 50%.

## Per-month results

| Month | QLIKE | Baseline (3mo med) | Drift | Status |
|---|---|---|---|---|
| 2022-02 | 0.0982 |  --  |  --  | NO_BASELINE |
| 2022-03 | 0.1090 | 0.0982 | +11.01% | PASS |
| 2022-04 | 0.1180 | 0.1036 | +13.87% | PASS |
| 2022-05 | 0.0980 | 0.1090 | -10.13% | PASS |
| 2022-06 | 0.0995 | 0.1090 | -8.80% | PASS |
| 2022-07 | 0.1066 | 0.0995 | +7.18% | PASS |
| 2022-08 | 0.1007 | 0.0995 | +1.22% | PASS |
| 2022-09 | 0.1012 | 0.1007 | +0.48% | PASS |
| 2022-10 | 0.1506 | 0.1012 | +48.84% | CONDITIONAL |
| 2022-11 | 0.1303 | 0.1012 | +28.84% | PASS |
| 2022-12 | 0.2250 | 0.1303 | +72.67% | FAIL |
| 2023-01 | 0.1741 | 0.1506 | +15.66% | PASS |
| 2023-02 | 0.1148 | 0.1741 | -34.08% | PASS |
| 2023-03 | 0.1091 | 0.1741 | -37.36% | PASS |
| 2023-04 | 0.1286 | 0.1148 | +12.02% | PASS |
| 2023-05 | 0.1309 | 0.1148 | +14.03% | PASS |
| 2023-06 | 0.1325 | 0.1286 | +3.01% | PASS |
| 2023-07 | 0.1500 | 0.1309 | +14.60% | PASS |
| 2023-08 | 0.1639 | 0.1325 | +23.76% | PASS |
| 2023-09 | 0.1421 | 0.1500 | -5.25% | PASS |
| 2023-10 | 0.1413 | 0.1500 | -5.77% | PASS |
| 2023-11 | 0.0982 | 0.1421 | -30.88% | PASS |
| 2023-12 | 0.0950 | 0.1413 | -32.79% | PASS |
| 2024-01 | 0.0890 | 0.0982 | -9.42% | PASS |
| 2024-02 | 0.0795 | 0.0950 | -16.33% | PASS |
| 2024-03 | 0.0662 | 0.0890 | -25.60% | PASS |
| 2024-04 | 0.0752 | 0.0795 | -5.38% | PASS |
| 2024-05 | 0.0897 | 0.0752 | +19.31% | PASS |
| 2024-06 | 0.1052 | 0.0752 | +39.91% | CONDITIONAL |
| 2024-07 | 0.0869 | 0.0897 | -3.20% | PASS |
| 2024-08 | 0.0874 | 0.0897 | -2.58% | PASS |
| 2024-09 | 0.0892 | 0.0874 | +2.04% | PASS |
| 2024-10 | 0.0808 | 0.0874 | -7.60% | PASS |
| 2024-11 | 0.0646 | 0.0874 | -26.13% | PASS |
| 2024-12 | 0.0662 | 0.0808 | -18.08% | PASS |
| 2025-01 | 0.0743 | 0.0662 | +12.28% | PASS |
| 2025-02 | 0.0703 | 0.0662 | +6.19% | PASS |
| 2025-03 | 0.0646 | 0.0703 | -8.05% | PASS |
| 2025-04 | 0.0707 | 0.0703 | +0.60% | PASS |
| 2025-05 | 0.0796 | 0.0703 | +13.34% | PASS |
| 2025-06 | 0.0925 | 0.0707 | +30.78% | CONDITIONAL |
| 2025-07 | 0.0885 | 0.0796 | +11.08% | PASS |
| 2025-08 | 0.0898 | 0.0885 | +1.54% | PASS |
| 2025-09 | 0.1004 | 0.0898 | +11.77% | PASS |
| 2025-10 | 0.0864 | 0.0898 | -3.84% | PASS |
| 2025-11 | 0.0737 | 0.0898 | -17.95% | PASS |
| 2025-12 | 0.1155 | 0.0864 | +33.66% | CONDITIONAL |
| 2026-01 | 0.1181 | 0.0864 | +36.70% | CONDITIONAL |
| 2026-02 | 0.0834 | 0.1155 | -27.76% | PASS |
| 2026-03 | 0.1014 | 0.1155 | -12.20% | PASS |

## Decision

**Hard fail**: at least one month exceeded the 50% drift gate. DO NOT deploy. File as round-5 finding; investigate whether this is a regime shift, a data-pipeline drift, or a model capacity issue. Consider rolling back to the previous month's checkpoint until the cause is identified.

## Note on round-4 interpretation

This report measures the production model's QLIKE drift across calendar months. F3+F6 change LIVE feature computation only  --  they do not modify the offline training pipeline. So the QLIKE drift here reflects natural market regime variation, NOT any F3/F6 effect. The real F3/F6 certification is a live-vs-offline feature parity test (planned next round); this run establishes the baseline QLIKE cadence the live engine will be measured against once new checkpoints land.
